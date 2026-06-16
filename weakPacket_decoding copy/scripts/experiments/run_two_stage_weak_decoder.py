#!/usr/bin/env python3
"""Run the two-stage weak-packet decoder on header-first packet rows.

The symbol CSV supplies weak-sync/header-first timing and optional GT bins for
evaluation.  GT fields are only used after decoding to compute metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.candidate_pruning import (  # noqa: E402
    PhaseGatedMetricConfig,
    PhaseTrend,
    estimate_payload_residual_offset,
    fit_early_payload_phase_trend,
    phase_gated_scores,
)
from weak_decoder.chirp import bin_to_grlora_symbol, build_downchirp  # noqa: E402
from weak_decoder.phase_guided_demod import (  # noqa: E402
    PhaseLine,
    extract_header_anchors,
    fit_phase_line,
)
from weak_decoder.two_stage_weak_decoder import (  # noqa: E402
    TwoStageWeakConfig,
    decode_two_stage_weak_payload,
    summarize_with_ground_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the two-stage codec-block weak-packet decoder."
    )
    parser.add_argument("-i", "--input-iq", type=Path, required=True, help="complex64 IQ file")
    parser.add_argument(
        "-s",
        "--symbol-csv",
        type=Path,
        required=True,
        help="header-first symbols CSV with header/payload rows",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="per-packet CSV output")
    parser.add_argument("--summary-json", type=Path, default=None, help="optional summary JSON")
    parser.add_argument("--packet", type=int, default=None, help="optional packet_index filter")
    parser.add_argument("--no-gt", action="store_true", help="do not treat CSV raw_fft_bin as GT")

    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument(
        "--fft-evidence-mode",
        choices=("center", "multi-offset", "phase-gated"),
        default="center",
        help=(
            "center uses one decimated FFT; multi-offset fuses all oversampling "
            "phases; phase-gated uses multi-offset screening plus phase consistency"
        ),
    )
    parser.add_argument("--preamble-len", type=float, default=8.0)

    parser.add_argument("--top-k-metrics", type=int, default=32)
    parser.add_argument("--amplitude-floor-db", type=float, default=24.0)
    parser.add_argument(
        "--phase-weight",
        type=float,
        default=0.10,
        help="legacy header-line phase penalty for center/multi-offset modes; ignored by phase-gated raw scores",
    )
    parser.add_argument(
        "--phase-gated-evidence",
        choices=("center", "multi-offset"),
        default="multi-offset",
        help="energy evidence used before phase-gated scoring",
    )
    parser.add_argument("--phase-gated-weight", type=float, default=0.20)
    parser.add_argument("--phase-gated-width-pi", type=float, default=0.50)
    parser.add_argument(
        "--phase-gated-trend-source",
        choices=("header", "header-offset", "early-payload"),
        default="early-payload",
        help="packet-local phase predictor for phase-gated mode",
    )
    parser.add_argument("--phase-gated-early-symbols", type=int, default=8)
    parser.add_argument("--phase-gated-early-min-margin-db", type=float, default=0.0)
    parser.add_argument("--phase-gated-early-trim-frac", type=float, default=0.25)
    parser.add_argument("--phase-gated-energy-preselect-count", type=int, default=128)
    parser.add_argument("--phase-gated-energy-preselect-factor", type=int, default=4)
    parser.add_argument("--phase-gated-noise-floor-percentile", type=float, default=50.0)
    parser.add_argument("--phase-gated-amp-floor-db", type=float, default=-18.0)
    parser.add_argument("--phase-gated-amp-temp-db", type=float, default=4.0)
    parser.add_argument("--phase-gated-rel-db-floor", type=float, default=-30.0)
    parser.add_argument("--bit-metric", choices=("max", "logsumexp"), default="max")
    parser.add_argument("--nibble-candidates", type=int, default=4)
    parser.add_argument("--row-beam-width", type=int, default=256)
    parser.add_argument("--block-candidate-limit", type=int, default=64)
    parser.add_argument("--global-beam-width", type=int, default=64)
    parser.add_argument("--final-candidate-limit", type=int, default=128)
    parser.add_argument("--block-trim-fraction", type=float, default=0.20)
    parser.add_argument("--projection-trim-fraction", type=float, default=0.10)
    parser.add_argument("--projection-score-weight", type=float, default=1.0)
    parser.add_argument(
        "--trajectory-score-weight",
        type=float,
        default=2.0,
        help=(
            "weight for packet-local re-encoded payload trajectory scoring via "
            "_score_payload_symbol_prior_candidate; set 0 to disable"
        ),
    )
    parser.add_argument("--crc-observed-bonus", type=float, default=4.0)
    parser.add_argument("--crc-candidate-min-evidence-margin", type=float, default=0.0)
    parser.add_argument("--crc-candidate-max-beam-rank", type=int, default=2048)
    parser.add_argument(
        "--disable-argmax-fallback",
        action="store_true",
        help="force beam output even when no CRC-valid payload candidate is found",
    )
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    if value == "":
        return int(default)
    return int(float(value))


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = str(row.get(key, "")).strip()
    if value == "":
        return float(default)
    return float(value)


def _first_int(row: dict[str, str], keys: tuple[str, ...], default: int = 0) -> int:
    for key in keys:
        if str(row.get(key, "")).strip() != "":
            return _int(row, key, default)
    return int(default)


def _first_float(row: dict[str, str], keys: tuple[str, ...], default: float = float("nan")) -> float:
    for key in keys:
        if str(row.get(key, "")).strip() != "":
            return _float(row, key, default)
    return float(default)


def load_packets(symbol_csv: Path, packet_filter: int | None) -> dict[int, dict[str, Any]]:
    with symbol_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    packets: dict[int, dict[str, Any]] = {}
    for row in rows:
        packet_index = _int(row, "packet_index", -1)
        if packet_index < 0:
            continue
        if packet_filter is not None and packet_index != int(packet_filter):
            continue

        packet = packets.setdefault(
            packet_index,
            {
                "packet_index": packet_index,
                "frame_index": _int(row, "frame_index", packet_index),
                "event_index": _int(row, "event_index", packet_index),
                "header_symbols": [],
                "payload_symbols": [],
                "header_start_sample": None,
                "sf": _int(row, "sf", 10),
                "bw": _float(row, "bw", 125000.0),
                "os_factor": _int(row, "os_factor", 4),
                "cfo_int": _int(row, "cfo_int", 0),
                "cfo_frac": _float(row, "cfo_frac", 0.0),
                "preamble_len": _first_float(row, ("preamble_len",), float("nan")),
                "payload_len": _int(row, "payload_len", 0),
                "cr": _first_int(row, ("payload_cr", "cr"), 1),
                "has_crc": bool(_first_int(row, ("payload_has_crc", "has_crc"), 1)),
                "ldro": bool(_first_int(row, ("payload_ldro", "ldro"), 0)),
                "header_valid": bool(_int(row, "header_valid", 0)),
            },
        )

        stage = str(row.get("stage", "")).strip().lower()
        stage_symbol_index = _int(row, "stage_symbol_index", -1)
        if stage == "header":
            if stage_symbol_index == 0:
                packet["header_start_sample"] = _int(row, "start_sample", 0)
            if 0 <= stage_symbol_index < 8:
                packet["header_symbols"].append((stage_symbol_index, _int(row, "symbol_value", 0)))
        elif stage == "payload":
            packet["payload_symbols"].append(
                {
                    "payload_symbol_index": stage_symbol_index,
                    "start_sample": _int(row, "start_sample", 0),
                    "gt_bin": _int(row, "raw_fft_bin", -1),
                }
            )

    valid: dict[int, dict[str, Any]] = {}
    for packet_index, packet in packets.items():
        if not packet["header_valid"]:
            continue
        if packet["header_start_sample"] is None:
            continue
        packet["header_symbols"] = [
            value for _, value in sorted(packet["header_symbols"], key=lambda item: item[0])
        ]
        packet["payload_symbols"].sort(key=lambda item: int(item["payload_symbol_index"]))
        if len(packet["header_symbols"]) == 8 and packet["payload_symbols"]:
            if not math.isfinite(float(packet["preamble_len"])):
                packet["preamble_len"] = 8.0
            valid[packet_index] = packet
    return valid


def _symbol_sample_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    return int(start_sample) + int(os_value // 2) + os_value * np.arange(n_bins, dtype=np.int64)


def extract_fft(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    downchirp: np.ndarray,
    cfo_total: float,
    header_start_sample: int,
    cfo_correction_mode: str,
) -> np.ndarray:
    n_bins = 1 << int(sf)
    indexes = _symbol_sample_indexes(start_sample, sf, os_factor)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(f"symbol at {start_sample} exceeds IQ range")
    symbol = np.asarray(samples[indexes], dtype=np.complex64)
    if str(cfo_correction_mode) == "continuous":
        rel_chip_start = float(start_sample - header_start_sample) / float(os_factor)
        cfo_phase = float(2.0 * math.pi * float(cfo_total) * rel_chip_start / n_bins)
        symbol = (symbol * np.exp(-1j * cfo_phase)).astype(np.complex64)
    return np.fft.fft((symbol * downchirp).astype(np.complex64)).astype(np.complex64)


def extract_multi_offset_fft_evidence(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    downchirp: np.ndarray,
    cfo_total: float,
    header_start_sample: int,
    cfo_correction_mode: str,
) -> np.ndarray:
    """Fuse all oversampling phases into one FFT-evidence spectrum."""

    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    fused_power = np.zeros(n_bins, dtype=np.float64)
    center_spectrum: np.ndarray | None = None
    for offset in range(os_value):
        indexes = int(start_sample) + int(offset) + os_value * np.arange(n_bins, dtype=np.int64)
        if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
            raise ValueError(f"symbol at {start_sample} offset {offset} exceeds IQ range")
        symbol = np.asarray(samples[indexes], dtype=np.complex64)
        if str(cfo_correction_mode) == "continuous":
            rel_chip_start = float(start_sample - header_start_sample) / float(os_value)
            cfo_phase = float(2.0 * math.pi * float(cfo_total) * rel_chip_start / n_bins)
            symbol = (symbol * np.exp(-1j * cfo_phase)).astype(np.complex64)
        spectrum = np.fft.fft((symbol * downchirp).astype(np.complex64)).astype(np.complex64)
        power = np.abs(spectrum).astype(np.float64) ** 2
        fused_power += power / (float(np.max(power)) + 1e-30)
        if offset == os_value // 2:
            center_spectrum = spectrum

    if center_spectrum is None:
        center_spectrum = np.ones(n_bins, dtype=np.complex64)
    phase = np.exp(1j * np.angle(center_spectrum))
    return (np.sqrt(fused_power).astype(np.float64) * phase).astype(np.complex64)


def build_config(args: argparse.Namespace) -> TwoStageWeakConfig:
    return TwoStageWeakConfig(
        top_k_metrics=int(args.top_k_metrics),
        amplitude_floor_db=float(args.amplitude_floor_db),
        phase_weight=float(args.phase_weight),
        bit_metric=str(args.bit_metric),
        nibble_candidates_per_codeword=int(args.nibble_candidates),
        row_beam_width=int(args.row_beam_width),
        block_candidate_limit=int(args.block_candidate_limit),
        global_beam_width=int(args.global_beam_width),
        final_candidate_limit=int(args.final_candidate_limit),
        block_trim_fraction=float(args.block_trim_fraction),
        projection_trim_fraction=float(args.projection_trim_fraction),
        projection_score_weight=float(args.projection_score_weight),
        trajectory_score_weight=float(args.trajectory_score_weight),
        crc_observed_bonus=float(args.crc_observed_bonus),
        crc_candidate_min_evidence_margin=float(args.crc_candidate_min_evidence_margin),
        crc_candidate_max_beam_rank=int(args.crc_candidate_max_beam_rank),
        crc_mode=str(args.crc_mode),
        argmax_fallback_on_crc_failure=not bool(args.disable_argmax_fallback),
    )


def _argmax_ser_against_gt(
    spectra: list[np.ndarray],
    gt_bins: list[int],
    sf: int,
    ldro: bool,
) -> tuple[float, float]:
    raw_errors = 0
    symbol_errors = 0
    compared = 0
    for spectrum, gt_bin in zip(spectra, gt_bins):
        gt = int(gt_bin)
        if gt < 0:
            continue
        power = np.abs(spectrum).astype(np.float64) ** 2
        argmax_bin = int(np.argmax(power))
        gt_symbol = bin_to_grlora_symbol(gt, sf=sf, is_header=False, ldro=ldro)
        argmax_symbol = bin_to_grlora_symbol(argmax_bin, sf=sf, is_header=False, ldro=ldro)
        raw_errors += int(argmax_bin != gt)
        symbol_errors += int(argmax_symbol != gt_symbol)
        compared += 1
    if compared <= 0:
        return 0.0, 0.0
    return float(raw_errors / compared), float(symbol_errors / compared)


def evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    config: TwoStageWeakConfig,
) -> dict[str, Any]:
    sf = int(packet["sf"])
    cfo_total = float(packet["cfo_int"]) + float(packet["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=packet["cfo_int"], cfo_frac=packet["cfo_frac"])
    header_start = int(packet["header_start_sample"])

    header_abs, header_phases = extract_header_anchors(
        samples=samples,
        header_start_sample=header_start,
        sf=sf,
        os_factor=int(packet["os_factor"]),
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        cfo_correction_mode=str(args.cfo_correction_mode),
        preamble_len=float(packet.get("preamble_len", args.preamble_len)),
        header_symbol_values=tuple(packet["header_symbols"]),
    )
    phase_line = PhaseLine()
    if header_abs.size >= 2:
        phase_line = fit_phase_line(header_abs, header_phases)

    center_spectra: list[np.ndarray] = []
    multi_spectra: list[np.ndarray] = []
    gt_bins: list[int] = []
    payload_indices: list[int] = []
    abs_indices: list[float] = []
    for symbol in packet["payload_symbols"]:
        try:
            center_spectrum = extract_fft(
                samples=samples,
                start_sample=int(symbol["start_sample"]),
                sf=sf,
                os_factor=int(packet["os_factor"]),
                downchirp=downchirp,
                cfo_total=cfo_total,
                header_start_sample=header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
            multi_spectrum = extract_multi_offset_fft_evidence(
                samples=samples,
                start_sample=int(symbol["start_sample"]),
                sf=sf,
                os_factor=int(packet["os_factor"]),
                downchirp=downchirp,
                cfo_total=cfo_total,
                header_start_sample=header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
            payload_index = int(symbol.get("payload_symbol_index", len(payload_indices)))
            center_spectra.append(center_spectrum)
            multi_spectra.append(multi_spectrum)
            gt_bins.append(int(symbol.get("gt_bin", -1)))
            payload_indices.append(payload_index)
            abs_indices.append(float(packet.get("preamble_len", args.preamble_len)) + 12.25 + float(payload_index))
        except ValueError:
            continue

    phase_trend: PhaseTrend | None = None
    raw_score_overrides: list[np.ndarray] | None = None
    if str(args.fft_evidence_mode) == "center":
        spectra = center_spectra
        decode_phase_line = phase_line
    elif str(args.fft_evidence_mode) == "multi-offset":
        spectra = multi_spectra
        decode_phase_line = phase_line
    else:
        spectra = center_spectra
        evidence_spectra = (
            multi_spectra
            if str(args.phase_gated_evidence) == "multi-offset"
            else center_spectra
        )
        evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in evidence_spectra]
        if str(args.phase_gated_trend_source) == "early-payload":
            phase_trend = fit_early_payload_phase_trend(
                center_spectra=center_spectra,
                evidence_powers=evidence_powers,
                abs_indices=abs_indices,
                fallback_line=phase_line,
                max_symbols=int(args.phase_gated_early_symbols),
                min_margin_db=float(args.phase_gated_early_min_margin_db),
                trim_frac=float(args.phase_gated_early_trim_frac),
            )
        else:
            offset = 0.0
            offset_std = float("inf")
            offset_count = 0
            source = "header"
            if str(args.phase_gated_trend_source) == "header-offset":
                offset, offset_std, offset_count = estimate_payload_residual_offset(
                    center_spectra=center_spectra,
                    evidence_powers=evidence_powers,
                    abs_indices=abs_indices,
                    phase_line=phase_line,
                    max_symbols=int(args.phase_gated_early_symbols),
                    min_margin_db=float(args.phase_gated_early_min_margin_db),
                )
                source = "header-offset"
            phase_trend = PhaseTrend(
                line=phase_line,
                residual_offset_rad=offset,
                residual_std_rad=offset_std,
                early_anchor_count=offset_count,
                source=source,
            )
        phase_cfg = PhaseGatedMetricConfig(
            phase_bonus_weight=float(args.phase_gated_weight),
            phase_gate_width_rad=float(args.phase_gated_width_pi) * math.pi,
            energy_preselect_count=int(args.phase_gated_energy_preselect_count),
            energy_preselect_factor=int(args.phase_gated_energy_preselect_factor),
            noise_floor_percentile=float(args.phase_gated_noise_floor_percentile),
            rel_db_floor=float(args.phase_gated_rel_db_floor),
            amp_gate_floor_db=float(args.phase_gated_amp_floor_db),
            amp_gate_temp_db=float(args.phase_gated_amp_temp_db),
        )
        raw_score_overrides = [
            phase_gated_scores(
                center_spectrum=center_spectrum,
                evidence_power=evidence_power,
                predicted_phase_rad=phase_trend.predict(abs_index),
                phase_trend_quality=phase_trend.quality,
                max_top_l=int(args.top_k_metrics),
                config=phase_cfg,
            )
            for center_spectrum, evidence_power, abs_index in zip(
                center_spectra,
                evidence_powers,
                abs_indices,
            )
        ]
        decode_phase_line = phase_trend.line

    result = decode_two_stage_weak_payload(
        payload_spectra=spectra,
        header_symbol_values=tuple(packet["header_symbols"]),
        phase_line=decode_phase_line,
        payload_symbol_start_abs_index=float(packet.get("preamble_len", args.preamble_len)) + 12.25,
        sf=sf,
        cr=int(packet["cr"]),
        payload_len=int(packet["payload_len"]),
        has_crc=bool(packet["has_crc"]),
        ldro=bool(packet["ldro"]),
        config=config,
        raw_score_overrides=raw_score_overrides,
        trajectory_spectra=center_spectra,
    )

    gt_summary = {}
    if not args.no_gt and any(value >= 0 for value in gt_bins):
        center_raw_ser, center_symbol_ser = _argmax_ser_against_gt(
            center_spectra,
            gt_bins,
            sf=sf,
            ldro=bool(packet["ldro"]),
        )
        multi_raw_ser, multi_symbol_ser = _argmax_ser_against_gt(
            multi_spectra,
            gt_bins,
            sf=sf,
            ldro=bool(packet["ldro"]),
        )
        gt_summary = summarize_with_ground_truth(
            result,
            gt_raw_bins=gt_bins,
            sf=sf,
            ldro=bool(packet["ldro"]),
            top_k=int(args.top_k_metrics),
        )
        gt_summary["center_argmax_raw_ser"] = float(center_raw_ser)
        gt_summary["center_argmax_symbol_ser"] = float(center_symbol_ser)
        gt_summary["multi_offset_argmax_raw_ser"] = float(multi_raw_ser)
        gt_summary["multi_offset_argmax_symbol_ser"] = float(multi_symbol_ser)

    selected = result.selected
    row: dict[str, Any] = {
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "success": int(result.success),
        "error": result.error,
        "payload_len": int(packet["payload_len"]),
        "cr": int(packet["cr"]),
        "has_crc": int(bool(packet["has_crc"])),
        "ldro": int(bool(packet["ldro"])),
        "fft_evidence_mode": str(args.fft_evidence_mode),
        "payload_hex": selected.payload_bytes.hex() if selected else "",
        "observed_crc_valid": int(selected.observed_crc_valid) if selected else 0,
        "crc_computed": f"0x{selected.crc_computed:04x}" if selected else "",
        "crc_received": f"0x{selected.crc_received:04x}" if selected else "",
        "candidate_payload_count": int(result.metrics.get("candidate_payload_count", 0)),
        "observed_crc_valid_count": int(result.metrics.get("observed_crc_valid_count", 0)),
        "avg_block_candidates": f"{float(result.metrics.get('avg_block_candidates', 0.0)):.3f}",
        "selected_beam_rank": int(selected.beam_rank) if selected else -1,
        "selected_total_score": f"{selected.total_score:.9f}" if selected else "",
        "selected_block_score": f"{selected.block_score:.9f}" if selected else "",
        "selected_projection_score": f"{selected.projection_score:.9f}" if selected else "",
        "selected_trajectory_score": f"{selected.trajectory_score:.9f}" if selected else "",
        "selected_source": selected.selection_source if selected else "",
        "selected_source_is_argmax_fallback": int(
            result.metrics.get("selected_source_is_argmax_fallback", 0)
        ),
        "phase_line_slope_pi": f"{phase_line.slope_pi:.9f}",
        "phase_line_r2": f"{phase_line.fit_r2:.9f}" if math.isfinite(phase_line.fit_r2) else "",
        "phase_gated_evidence": str(args.phase_gated_evidence) if str(args.fft_evidence_mode) == "phase-gated" else "",
        "phase_gated_weight": f"{float(args.phase_gated_weight):.9f}" if str(args.fft_evidence_mode) == "phase-gated" else "",
        "phase_gated_width_pi": f"{float(args.phase_gated_width_pi):.9f}" if str(args.fft_evidence_mode) == "phase-gated" else "",
        "phase_gated_trend_source": phase_trend.source if phase_trend is not None else "",
        "phase_gated_trend_quality": f"{phase_trend.quality:.9f}" if phase_trend is not None else "",
        "phase_gated_trend_slope_pi": f"{phase_trend.line.slope_pi:.9f}" if phase_trend is not None else "",
        "phase_gated_trend_r2": (
            f"{phase_trend.line.fit_r2:.9f}"
            if phase_trend is not None and math.isfinite(phase_trend.line.fit_r2)
            else ""
        ),
        "phase_gated_residual_offset_pi": (
            f"{phase_trend.residual_offset_rad / math.pi:.9f}"
            if phase_trend is not None
            else ""
        ),
        "phase_gated_residual_std_pi": (
            f"{phase_trend.residual_std_rad / math.pi:.9f}"
            if phase_trend is not None and math.isfinite(phase_trend.residual_std_rad)
            else ""
        ),
        "phase_gated_early_anchor_count": int(phase_trend.early_anchor_count) if phase_trend is not None else "",
    }
    for key, value in result.timings_ms.items():
        row[key] = f"{float(value):.3f}"
    for key, value in gt_summary.items():
        row[key] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def avg(key: str) -> float:
        values = []
        for row in rows:
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else 0.0

    return {
        "packet_count": len(rows),
        "success_rate": avg("success"),
        "observed_crc_valid_rate": avg("observed_crc_valid"),
        "argmax_raw_ser": avg("argmax_raw_ser"),
        "two_stage_raw_ser": avg("two_stage_raw_ser"),
        "argmax_symbol_ser": avg("argmax_symbol_ser"),
        "two_stage_symbol_ser": avg("two_stage_symbol_ser"),
        "topk_recall": avg("topk_recall"),
        "mean_gt_rank": avg("mean_gt_rank"),
        "center_argmax_raw_ser": avg("center_argmax_raw_ser"),
        "center_argmax_symbol_ser": avg("center_argmax_symbol_ser"),
        "multi_offset_argmax_raw_ser": avg("multi_offset_argmax_raw_ser"),
        "multi_offset_argmax_symbol_ser": avg("multi_offset_argmax_symbol_ser"),
        "argmax_fallback_rate": avg("selected_source_is_argmax_fallback"),
        "mean_total_ms": avg("total_ms"),
        "mean_likelihood_ms": avg("likelihood_ms"),
        "mean_block_search_ms": avg("block_search_ms"),
        "mean_global_beam_ms": avg("global_beam_ms"),
        "mean_final_projection_ms": avg("final_projection_ms"),
        "mean_selected_trajectory_score": avg("selected_trajectory_score"),
        "parameters": {
            "top_k_metrics": args.top_k_metrics,
            "amplitude_floor_db": args.amplitude_floor_db,
            "phase_weight": args.phase_weight,
            "bit_metric": args.bit_metric,
            "nibble_candidates": args.nibble_candidates,
            "row_beam_width": args.row_beam_width,
            "block_candidate_limit": args.block_candidate_limit,
            "global_beam_width": args.global_beam_width,
            "final_candidate_limit": args.final_candidate_limit,
            "block_trim_fraction": args.block_trim_fraction,
            "projection_trim_fraction": args.projection_trim_fraction,
            "trajectory_score_weight": args.trajectory_score_weight,
            "crc_mode": args.crc_mode,
            "fft_evidence_mode": args.fft_evidence_mode,
            "phase_gated_evidence": args.phase_gated_evidence,
            "phase_gated_weight": args.phase_gated_weight,
            "phase_gated_width_pi": args.phase_gated_width_pi,
            "phase_gated_trend_source": args.phase_gated_trend_source,
            "phase_gated_early_symbols": args.phase_gated_early_symbols,
            "phase_gated_early_min_margin_db": args.phase_gated_early_min_margin_db,
            "phase_gated_early_trim_frac": args.phase_gated_early_trim_frac,
            "phase_gated_energy_preselect_count": args.phase_gated_energy_preselect_count,
            "phase_gated_energy_preselect_factor": args.phase_gated_energy_preselect_factor,
            "crc_candidate_min_evidence_margin": args.crc_candidate_min_evidence_margin,
            "crc_candidate_max_beam_rank": args.crc_candidate_max_beam_rank,
            "argmax_fallback_on_crc_failure": not bool(args.disable_argmax_fallback),
            "uses_payload_template": False,
            "uses_counter_prior": False,
            "uses_cross_packet_joint_prior": False,
        },
    }


def main() -> int:
    args = parse_args()
    samples = np.fromfile(args.input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {args.input_iq}")

    packets = load_packets(args.symbol_csv, args.packet)
    if not packets:
        raise SystemExit("no header-valid packets found in symbol CSV")

    config = build_config(args)
    rows: list[dict[str, Any]] = []
    for packet_index in sorted(packets):
        print(f"packet {packet_index}: decoding", flush=True)
        row = evaluate_packet(samples, packets[packet_index], args, config)
        rows.append(row)
        if "argmax_symbol_ser" in row:
            print(
                "  argmax_symbol_ser={:.3f} two_stage_symbol_ser={:.3f} "
                "crc_obs={} time={} ms".format(
                    float(row.get("argmax_symbol_ser", 0.0)),
                    float(row.get("two_stage_symbol_ser", 0.0)),
                    row.get("observed_crc_valid", 0),
                    row.get("total_ms", ""),
                ),
                flush=True,
            )
        else:
            print(
                f"  success={row.get('success', 0)} crc_obs={row.get('observed_crc_valid', 0)} "
                f"time={row.get('total_ms', '')} ms",
                flush=True,
            )

    write_csv(args.output, rows)
    summary = build_summary(rows, args)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"wrote={args.output}")
    if args.summary_json is not None:
        print(f"wrote_summary={args.summary_json}")
    print(
        "summary: packets={packet_count} argmax_symbol_ser={argmax_symbol_ser:.3f} "
        "two_stage_symbol_ser={two_stage_symbol_ser:.3f} crc_obs={observed_crc_valid_rate:.3f} "
        "topk={topk_recall:.3f}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

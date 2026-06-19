#!/usr/bin/env python3
"""Run symbol-level phase-aware two-stage FFT-bin selection.

This runner evaluates the corrected two-stage design:

* lock high-confidence symbols from multi-offset FFT evidence,
* keep Top-L bins only for low-confidence symbols,
* fit a packet-local phase line from locked symbols,
* select low-confidence bins with a symbol-level phase-smooth beam,
* decode the resulting hard symbol sequence only after peak selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_two_stage_weak_decoder import (  # noqa: E402
    extract_fft,
    extract_multi_offset_fft_evidence,
    load_packets,
)
from weak_decoder.chirp import bin_to_grlora_symbol, build_downchirp  # noqa: E402
from weak_decoder.payload_codec import decode_explicit_frame_symbols  # noqa: E402
from weak_decoder.phase_guided_demod import (  # noqa: E402
    PhaseLine,
    extract_header_anchors,
    fit_phase_line,
)
from weak_decoder.symbol_phase_two_stage import (  # noqa: E402
    SymbolPhaseConfig,
    select_symbol_bins_two_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate symbol-level phase-aware two-stage peak selection."
    )
    parser.add_argument("-i", "--input-iq", type=Path, required=True)
    parser.add_argument("-s", "--symbol-csv", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--no-gt", action="store_true")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)

    parser.add_argument("--top-l-low-confidence", type=int, default=24)
    parser.add_argument("--lock-margin-db", type=float, default=1.5)
    parser.add_argument("--lock-peak-to-median-db", type=float, default=5.0)
    parser.add_argument("--lock-phase-score", type=float, default=0.35)
    parser.add_argument("--min-locked-for-line", type=int, default=4)
    parser.add_argument("--line-trim-frac", type=float, default=0.25)
    parser.add_argument("--phase-model", choices=("linear", "quadratic"), default="linear")
    parser.add_argument("--selection-mode", choices=("override", "smooth", "coherence", "window", "window_guarded"), default="coherence")
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--trajectory-rmse-scale-pi", type=float, default=0.30)
    parser.add_argument("--trajectory-phase-weight", type=float, default=0.20)
    parser.add_argument("--trajectory-line-weight", type=float, default=0.00)
    parser.add_argument("--trajectory-amp-weight", type=float, default=0.80)
    parser.add_argument("--trajectory-profile-weight", type=float, default=0.00)
    parser.add_argument("--phase-override-min-gain", type=float, default=0.15)
    parser.add_argument("--phase-override-max-drop-db", type=float, default=0.60)
    parser.add_argument("--phase-override-score-margin", type=float, default=0.06)
    parser.add_argument("--phase-override-min-line-anchors", type=int, default=8)
    parser.add_argument("--phase-override-max-line-rmse-pi", type=float, default=0.25)
    parser.add_argument("--coherence-weight", type=float, default=0.0)
    parser.add_argument("--coherence-candidate-top-l", type=int, default=0)
    parser.add_argument("--lock-min-coherence", type=float, default=0.0)
    parser.add_argument("--smooth-phase-weight", type=float, default=0.05)
    parser.add_argument("--smooth-amp-weight", type=float, default=0.50)
    parser.add_argument("--smooth-coherence-weight", type=float, default=0.90)
    parser.add_argument("--smooth-slope-penalty", type=float, default=0.05)
    parser.add_argument("--smooth-curvature-penalty", type=float, default=0.10)
    parser.add_argument("--smooth-max-energy-drop-db", type=float, default=20.0)
    parser.add_argument("--smooth-min-line-anchors", type=int, default=4)
    parser.add_argument("--smooth-min-locked-ratio", type=float, default=0.0)
    parser.add_argument("--smooth-max-line-rmse-pi", type=float, default=float("inf"))
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--window-degree", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--window-phase-weight", type=float, default=0.05)
    parser.add_argument("--window-amp-weight", type=float, default=0.50)
    parser.add_argument("--window-coherence-weight", type=float, default=0.90)
    parser.add_argument("--window-slope-weight", type=float, default=0.00)
    parser.add_argument("--window-curvature-weight", type=float, default=0.00)
    parser.add_argument("--window-phase-scale-pi", type=float, default=0.25)
    parser.add_argument("--window-slope-scale-pi", type=float, default=0.45)
    parser.add_argument("--window-curvature-scale-pi", type=float, default=0.25)
    parser.add_argument("--window-recent-decay", type=float, default=0.75)
    parser.add_argument("--window-anchor-span", type=float, default=8.0)
    parser.add_argument("--window-anchor-min", type=int, default=2)
    parser.add_argument("--window-anchor-max-rmse-pi", type=float, default=0.40)
    parser.add_argument("--window-min-locked-ratio", type=float, default=0.10)
    parser.add_argument("--window-guard-min-phase-gain", type=float, default=0.10)
    parser.add_argument("--window-guard-max-energy-drop-db", type=float, default=0.75)
    parser.add_argument("--window-guard-max-coherence-drop", type=float, default=0.08)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SymbolPhaseConfig:
    return SymbolPhaseConfig(
        top_l_low_confidence=int(args.top_l_low_confidence),
        lock_margin_db=float(args.lock_margin_db),
        lock_peak_to_median_db=float(args.lock_peak_to_median_db),
        lock_phase_score=float(args.lock_phase_score),
        min_locked_for_line=int(args.min_locked_for_line),
        line_trim_frac=float(args.line_trim_frac),
        phase_model=str(args.phase_model),
        selection_mode=str(args.selection_mode),
        beam_width=int(args.beam_width),
        trajectory_rmse_scale_pi=float(args.trajectory_rmse_scale_pi),
        phase_weight=float(args.trajectory_phase_weight),
        line_weight=float(args.trajectory_line_weight),
        amp_weight=float(args.trajectory_amp_weight),
        profile_weight=float(args.trajectory_profile_weight),
        phase_override_min_gain=float(args.phase_override_min_gain),
        phase_override_max_drop_db=float(args.phase_override_max_drop_db),
        phase_override_score_margin=float(args.phase_override_score_margin),
        phase_override_min_line_anchors=int(args.phase_override_min_line_anchors),
        phase_override_max_line_rmse_pi=float(args.phase_override_max_line_rmse_pi),
        coherence_weight=float(args.coherence_weight),
        coherence_candidate_top_l=int(args.coherence_candidate_top_l),
        lock_min_coherence=float(args.lock_min_coherence),
        smooth_phase_weight=float(args.smooth_phase_weight),
        smooth_amp_weight=float(args.smooth_amp_weight),
        smooth_coherence_weight=float(args.smooth_coherence_weight),
        smooth_slope_penalty=float(args.smooth_slope_penalty),
        smooth_curvature_penalty=float(args.smooth_curvature_penalty),
        smooth_max_energy_drop_db=float(args.smooth_max_energy_drop_db),
        smooth_min_line_anchors=int(args.smooth_min_line_anchors),
        smooth_min_locked_ratio=float(args.smooth_min_locked_ratio),
        smooth_max_line_rmse_pi=float(args.smooth_max_line_rmse_pi),
        window_size=int(args.window_size),
        window_degree=int(args.window_degree),
        window_phase_weight=float(args.window_phase_weight),
        window_amp_weight=float(args.window_amp_weight),
        window_coherence_weight=float(args.window_coherence_weight),
        window_slope_weight=float(args.window_slope_weight),
        window_curvature_weight=float(args.window_curvature_weight),
        window_phase_scale_pi=float(args.window_phase_scale_pi),
        window_slope_scale_pi=float(args.window_slope_scale_pi),
        window_curvature_scale_pi=float(args.window_curvature_scale_pi),
        window_recent_decay=float(args.window_recent_decay),
        window_anchor_span=float(args.window_anchor_span),
        window_anchor_min=int(args.window_anchor_min),
        window_anchor_max_rmse_pi=float(args.window_anchor_max_rmse_pi),
        window_min_locked_ratio=float(args.window_min_locked_ratio),
        window_guard_min_phase_gain=float(args.window_guard_min_phase_gain),
        window_guard_max_energy_drop_db=float(args.window_guard_max_energy_drop_db),
        window_guard_max_coherence_drop=float(args.window_guard_max_coherence_drop),
    )


def _packet_phase_line(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> PhaseLine:
    header_abs, header_phases = extract_header_anchors(
        samples=samples,
        header_start_sample=int(packet["header_start_sample"]),
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        cfo_correction_mode=str(args.cfo_correction_mode),
        preamble_len=float(packet.get("preamble_len", args.preamble_len)),
        header_symbol_values=tuple(packet["header_symbols"]),
    )
    if header_abs.size >= 2:
        return fit_phase_line(header_abs, header_phases)
    return PhaseLine()


def extract_multi_offset_fft_evidence_with_coherence(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    downchirp: np.ndarray,
    cfo_total: float,
    header_start_sample: int,
    cfo_correction_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fused multi-offset evidence plus offset-phase coherence per bin."""

    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    fused_power = np.zeros(n_bins, dtype=np.float64)
    sum_complex = np.zeros(n_bins, dtype=np.complex128)
    sum_magnitude = np.zeros(n_bins, dtype=np.float64)
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
        sum_complex += spectrum.astype(np.complex128)
        sum_magnitude += np.abs(spectrum).astype(np.float64)
        if offset == os_value // 2:
            center_spectrum = spectrum

    if center_spectrum is None:
        center_spectrum = np.ones(n_bins, dtype=np.complex64)
    phase = np.exp(1j * np.angle(center_spectrum))
    fused = (np.sqrt(fused_power).astype(np.float64) * phase).astype(np.complex64)
    coherence = np.clip(np.abs(sum_complex) / (sum_magnitude + 1e-30), 0.0, 1.0)
    return fused, coherence.astype(np.float64)


def _extract_payload_spectra(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[int]]:
    sf = int(packet["sf"])
    cfo_total = float(packet["cfo_int"]) + float(packet["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=packet["cfo_int"], cfo_frac=packet["cfo_frac"])
    header_start = int(packet["header_start_sample"])

    center_spectra: list[np.ndarray] = []
    multi_spectra: list[np.ndarray] = []
    abs_indices: list[float] = []
    gt_bins: list[int] = []
    for symbol in packet["payload_symbols"]:
        try:
            center = extract_fft(
                samples=samples,
                start_sample=int(symbol["start_sample"]),
                sf=sf,
                os_factor=int(packet["os_factor"]),
                downchirp=downchirp,
                cfo_total=cfo_total,
                header_start_sample=header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
            multi = extract_multi_offset_fft_evidence(
                samples=samples,
                start_sample=int(symbol["start_sample"]),
                sf=sf,
                os_factor=int(packet["os_factor"]),
                downchirp=downchirp,
                cfo_total=cfo_total,
                header_start_sample=header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
        except ValueError:
            continue
        payload_idx = int(symbol.get("payload_symbol_index", len(center_spectra)))
        center_spectra.append(center)
        multi_spectra.append(multi)
        abs_indices.append(float(packet.get("preamble_len", args.preamble_len)) + 12.25 + float(payload_idx))
        gt_bins.append(int(symbol.get("gt_bin", -1)))
    return center_spectra, multi_spectra, abs_indices, gt_bins


def _extract_payload_spectra_with_coherence(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[int], list[np.ndarray]]:
    sf = int(packet["sf"])
    cfo_total = float(packet["cfo_int"]) + float(packet["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=packet["cfo_int"], cfo_frac=packet["cfo_frac"])
    header_start = int(packet["header_start_sample"])

    center_spectra: list[np.ndarray] = []
    multi_spectra: list[np.ndarray] = []
    abs_indices: list[float] = []
    gt_bins: list[int] = []
    coherences: list[np.ndarray] = []
    for symbol in packet["payload_symbols"]:
        try:
            center = extract_fft(
                samples=samples,
                start_sample=int(symbol["start_sample"]),
                sf=sf,
                os_factor=int(packet["os_factor"]),
                downchirp=downchirp,
                cfo_total=cfo_total,
                header_start_sample=header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
            multi, coherence = extract_multi_offset_fft_evidence_with_coherence(
                samples=samples,
                start_sample=int(symbol["start_sample"]),
                sf=sf,
                os_factor=int(packet["os_factor"]),
                downchirp=downchirp,
                cfo_total=cfo_total,
                header_start_sample=header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
        except ValueError:
            continue
        payload_idx = int(symbol.get("payload_symbol_index", len(center_spectra)))
        center_spectra.append(center)
        multi_spectra.append(multi)
        abs_indices.append(float(packet.get("preamble_len", args.preamble_len)) + 12.25 + float(payload_idx))
        gt_bins.append(int(symbol.get("gt_bin", -1)))
        coherences.append(coherence)
    return center_spectra, multi_spectra, abs_indices, gt_bins, coherences


def _argmax_bins(spectra: Sequence[np.ndarray]) -> tuple[int, ...]:
    return tuple(int(np.argmax(np.abs(spec).astype(np.float64) ** 2)) for spec in spectra)


def _ser(raw_bins: Sequence[int], gt_bins: Sequence[int], sf: int, ldro: bool) -> tuple[float, float, int]:
    n = min(len(raw_bins), len(gt_bins))
    raw_errors = 0
    symbol_errors = 0
    compared = 0
    for idx in range(n):
        gt = int(gt_bins[idx])
        if gt < 0:
            continue
        pred = int(raw_bins[idx])
        raw_errors += int(pred != gt)
        pred_symbol = bin_to_grlora_symbol(pred, sf=sf, is_header=False, ldro=ldro)
        gt_symbol = bin_to_grlora_symbol(gt, sf=sf, is_header=False, ldro=ldro)
        symbol_errors += int(pred_symbol != gt_symbol)
        compared += 1
    if compared <= 0:
        return 0.0, 0.0, 0
    return float(raw_errors / compared), float(symbol_errors / compared), int(compared)


def _candidate_recall(result, gt_bins: Sequence[int]) -> tuple[int, int]:
    hits = 0
    total = 0
    for idx, ev in enumerate(result.evidences):
        if idx >= len(gt_bins) or result.locked_mask[idx]:
            continue
        gt = int(gt_bins[idx])
        if gt < 0:
            continue
        total += 1
        hits += int(gt in set(int(v) for v in ev.top_bins))
    return int(hits), int(total)


def _false_locks(result, gt_bins: Sequence[int]) -> tuple[int, int]:
    false_locks = 0
    locked_with_gt = 0
    for idx, locked in enumerate(result.locked_mask):
        if idx >= len(gt_bins) or not locked:
            continue
        gt = int(gt_bins[idx])
        if gt < 0:
            continue
        locked_with_gt += 1
        false_locks += int(int(result.selected_raw_bins[idx]) != gt)
    return int(false_locks), int(locked_with_gt)


def _decode_selected(packet: dict[str, Any], selected_raw_bins: Sequence[int], args: argparse.Namespace) -> dict[str, Any]:
    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    selected_symbols = tuple(
        bin_to_grlora_symbol(raw_bin, sf=sf, is_header=False, ldro=ldro)
        for raw_bin in selected_raw_bins
    )
    try:
        decoded = decode_explicit_frame_symbols(
            header_symbol_values=tuple(packet["header_symbols"]),
            payload_symbol_values=selected_symbols,
            sf=sf,
            bw=float(packet["bw"]),
            ldro_mode=int(args.ldro_mode),
            crc_mode=str(args.crc_mode),
        )
        return {
            "payload_hex": decoded.payload.payload_bytes.hex(),
            "crc_valid": int(decoded.payload.crc_valid),
            "crc_computed": f"0x{decoded.payload.crc_computed:04x}",
            "crc_received": f"0x{decoded.payload.crc_received:04x}",
            "decoded_payload_len": int(len(decoded.payload.payload_bytes)),
            "decode_error": "",
        }
    except Exception as exc:
        return {
            "payload_hex": "",
            "crc_valid": 0,
            "crc_computed": "",
            "crc_received": "",
            "decoded_payload_len": 0,
            "decode_error": str(exc),
        }


def evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    config: SymbolPhaseConfig,
) -> dict[str, Any]:
    center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
        samples, packet, args
    )
    evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
    header_line = _packet_phase_line(samples, packet, args)
    result = select_symbol_bins_two_stage(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=config,
        fallback_line=header_line,
        offset_coherences=coherences,
    )

    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    center_bins = _argmax_bins(center_spectra)
    multi_bins = _argmax_bins(multi_spectra)
    selected_bins = tuple(result.selected_raw_bins)
    center_raw_ser, center_symbol_ser, compared = _ser(center_bins, gt_bins, sf=sf, ldro=ldro)
    multi_raw_ser, multi_symbol_ser, _ = _ser(multi_bins, gt_bins, sf=sf, ldro=ldro)
    selected_raw_ser, selected_symbol_ser, _ = _ser(selected_bins, gt_bins, sf=sf, ldro=ldro)
    candidate_hits, candidate_total = _candidate_recall(result, gt_bins)
    false_locks, locked_with_gt = _false_locks(result, gt_bins)
    decoded = _decode_selected(packet, selected_bins, args)
    selected_coherences: list[float] = []
    for idx, raw_bin in enumerate(selected_bins):
        if idx < len(coherences):
            b = int(raw_bin)
            if 0 <= b < coherences[idx].size:
                selected_coherences.append(float(coherences[idx][b]))

    row: dict[str, Any] = {
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "payload_len": int(packet["payload_len"]),
        "cr": int(packet["cr"]),
        "has_crc": int(bool(packet["has_crc"])),
        "ldro": int(ldro),
        "symbol_count": int(len(selected_bins)),
        "gt_compared_symbols": int(compared),
        "center_argmax_raw_ser": float(center_raw_ser),
        "center_argmax_symbol_ser": float(center_symbol_ser),
        "multi_offset_argmax_raw_ser": float(multi_raw_ser),
        "multi_offset_argmax_symbol_ser": float(multi_symbol_ser),
        "selected_raw_ser": float(selected_raw_ser),
        "selected_symbol_ser": float(selected_symbol_ser),
        "ser_gain_vs_center": float(center_symbol_ser - selected_symbol_ser),
        "ser_gain_vs_multi_offset": float(multi_symbol_ser - selected_symbol_ser),
        "locked_symbol_count": int(result.locked_count),
        "uncertain_symbol_count": int(result.uncertain_count),
        "locked_ratio": float(result.locked_count / max(1, len(selected_bins))),
        "false_lock_count": int(false_locks),
        "locked_with_gt_count": int(locked_with_gt),
        "false_lock_rate": float(false_locks / max(1, locked_with_gt)),
        "uncertain_candidate_hits": int(candidate_hits),
        "uncertain_candidate_total": int(candidate_total),
        "uncertain_candidate_recall": float(candidate_hits / max(1, candidate_total)),
        "trajectory_score": float(result.trajectory_score),
        "line_score": float(result.line_score),
        "mean_phase_score": float(result.mean_phase_score),
        "mean_amp_score": float(result.mean_amp_score),
        "mean_selected_offset_coherence": float(np.mean(selected_coherences)) if selected_coherences else 0.0,
        "phase_line_slope_pi": float(result.phase_line.slope_pi),
        "phase_line_r2": float(result.phase_line.fit_r2) if math.isfinite(result.phase_line.fit_r2) else "",
        "phase_line_rmse_pi": float(result.phase_line.fit_rmse_pi) if math.isfinite(result.phase_line.fit_rmse_pi) else "",
        "phase_line_anchor_count": int(result.phase_line.anchor_count),
        "beam_final_size": int(result.beam_final_size),
        "error": result.error,
    }
    row.update(decoded)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "packet_count": int(len(rows)),
        "center_argmax_symbol_ser": _avg(rows, "center_argmax_symbol_ser"),
        "multi_offset_argmax_symbol_ser": _avg(rows, "multi_offset_argmax_symbol_ser"),
        "selected_symbol_ser": _avg(rows, "selected_symbol_ser"),
        "ser_gain_vs_center": _avg(rows, "ser_gain_vs_center"),
        "ser_gain_vs_multi_offset": _avg(rows, "ser_gain_vs_multi_offset"),
        "crc_valid_rate": _avg(rows, "crc_valid"),
        "mean_locked_ratio": _avg(rows, "locked_ratio"),
        "mean_false_lock_rate": _avg(rows, "false_lock_rate"),
        "mean_uncertain_candidate_recall": _avg(rows, "uncertain_candidate_recall"),
        "mean_selected_offset_coherence": _avg(rows, "mean_selected_offset_coherence"),
        "mean_phase_line_r2": _avg(rows, "phase_line_r2"),
        "mean_phase_line_rmse_pi": _avg(rows, "phase_line_rmse_pi"),
        "parameters": {
            "top_l_low_confidence": args.top_l_low_confidence,
            "lock_margin_db": args.lock_margin_db,
            "lock_peak_to_median_db": args.lock_peak_to_median_db,
            "lock_phase_score": args.lock_phase_score,
            "min_locked_for_line": args.min_locked_for_line,
            "line_trim_frac": args.line_trim_frac,
            "phase_model": args.phase_model,
            "selection_mode": args.selection_mode,
            "beam_width": args.beam_width,
            "trajectory_rmse_scale_pi": args.trajectory_rmse_scale_pi,
            "trajectory_phase_weight": args.trajectory_phase_weight,
            "trajectory_line_weight": args.trajectory_line_weight,
            "trajectory_amp_weight": args.trajectory_amp_weight,
            "trajectory_profile_weight": args.trajectory_profile_weight,
            "phase_override_min_gain": args.phase_override_min_gain,
            "phase_override_max_drop_db": args.phase_override_max_drop_db,
            "phase_override_score_margin": args.phase_override_score_margin,
            "phase_override_min_line_anchors": args.phase_override_min_line_anchors,
            "phase_override_max_line_rmse_pi": args.phase_override_max_line_rmse_pi,
            "coherence_weight": args.coherence_weight,
            "coherence_candidate_top_l": args.coherence_candidate_top_l,
            "lock_min_coherence": args.lock_min_coherence,
            "smooth_phase_weight": args.smooth_phase_weight,
            "smooth_amp_weight": args.smooth_amp_weight,
            "smooth_coherence_weight": args.smooth_coherence_weight,
            "smooth_slope_penalty": args.smooth_slope_penalty,
            "smooth_curvature_penalty": args.smooth_curvature_penalty,
            "smooth_max_energy_drop_db": args.smooth_max_energy_drop_db,
            "smooth_min_line_anchors": args.smooth_min_line_anchors,
            "smooth_min_locked_ratio": args.smooth_min_locked_ratio,
            "smooth_max_line_rmse_pi": args.smooth_max_line_rmse_pi,
            "window_size": args.window_size,
            "window_degree": args.window_degree,
            "window_phase_weight": args.window_phase_weight,
            "window_amp_weight": args.window_amp_weight,
            "window_coherence_weight": args.window_coherence_weight,
            "window_slope_weight": args.window_slope_weight,
            "window_curvature_weight": args.window_curvature_weight,
            "window_phase_scale_pi": args.window_phase_scale_pi,
            "window_slope_scale_pi": args.window_slope_scale_pi,
            "window_curvature_scale_pi": args.window_curvature_scale_pi,
            "window_recent_decay": args.window_recent_decay,
            "window_anchor_span": args.window_anchor_span,
            "window_anchor_min": args.window_anchor_min,
            "window_anchor_max_rmse_pi": args.window_anchor_max_rmse_pi,
            "window_min_locked_ratio": args.window_min_locked_ratio,
            "window_guard_min_phase_gain": args.window_guard_min_phase_gain,
            "window_guard_max_energy_drop_db": args.window_guard_max_energy_drop_db,
            "window_guard_max_coherence_drop": args.window_guard_max_coherence_drop,
            "uses_packet_structure_prior": False,
            "uses_counter_prior": False,
            "uses_cross_packet_joint_prior": False,
            "uses_payload_byte_enumeration": False,
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
        print(f"packet {packet_index}: symbol-level phase selection", flush=True)
        row = evaluate_packet(samples, packets[packet_index], args, config)
        rows.append(row)
        print(
            "  center_ser={:.3f} multi_ser={:.3f} selected_ser={:.3f} "
            "crc={} locked={}/{} cand_recall={:.3f}".format(
                float(row.get("center_argmax_symbol_ser", 0.0)),
                float(row.get("multi_offset_argmax_symbol_ser", 0.0)),
                float(row.get("selected_symbol_ser", 0.0)),
                row.get("crc_valid", 0),
                int(row.get("locked_symbol_count", 0)),
                int(row.get("symbol_count", 0)),
                float(row.get("uncertain_candidate_recall", 0.0)),
            ),
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
        "summary: packets={packet_count} center_ser={center_argmax_symbol_ser:.3f} "
        "multi_ser={multi_offset_argmax_symbol_ser:.3f} selected_ser={selected_symbol_ser:.3f} "
        "crc={crc_valid_rate:.3f}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

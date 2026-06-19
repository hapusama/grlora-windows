#!/usr/bin/env python3
"""Threshold sweep for phase-line dominated payload candidate selection.

This script keeps the v3/current selector intact and adds a fourth method:
`phase_line`, implemented under `weak_decoder.phase_line`.  The candidate set
still comes from multi-offset FFT evidence, but the second-stage decision is a
phase-smooth path search with coherence only as a weak tie-break/guard.
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
EXPERIMENT_DIR = SCRIPT_DIR.parent
WEAK_ROOT = SCRIPT_DIR.parents[2]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_symbol_phase_threshold_sweep import (  # noqa: E402
    DEFAULT_DATASETS,
    _avg,
    _dataset_paths,
    _snr_values,
    _threshold_tables,
    _write_csv,
)
from run_symbol_phase_two_stage import (  # noqa: E402
    _argmax_bins,
    _candidate_recall,
    _decode_selected,
    _extract_payload_spectra_with_coherence,
    _false_locks,
    _packet_phase_line,
    _ser,
    build_config,
)
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.candidate_pruning import wrap_phase  # noqa: E402
from weak_decoder.phase_line import PhaseLineSelectorConfig, select_phase_smooth_path  # noqa: E402
from weak_decoder.phase_guided_demod import PhaseLine, fit_phase_line  # noqa: E402
from weak_decoder.symbol_phase_two_stage import select_symbol_bins_two_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v3 coherence selector against phase-line path selector.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snr-start", type=float, default=-20.0)
    parser.add_argument("--snr-stop", type=float, default=-24.0)
    parser.add_argument("--snr-step", type=float, default=-1.0)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "phase_line_selector")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--independent-noise", action="store_true")
    parser.add_argument(
        "--signal-reference-power",
        type=float,
        default=None,
        help="Optional clean-signal reference power for AWGN. If omitted, metadata is used when present; otherwise mean clean IQ power is used.",
    )

    # v3 selector knobs are forwarded through the existing builder.
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

    # New phase-line selector knobs.
    parser.add_argument("--phase-line-top-l", type=int, default=24)
    parser.add_argument("--phase-line-beam-width", type=int, default=96)
    parser.add_argument("--phase-line-window-size", type=int, default=5)
    parser.add_argument("--phase-line-window-degree", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--phase-line-recent-decay", type=float, default=0.78)
    parser.add_argument("--phase-line-low-conf-weight", type=float, default=0.82)
    parser.add_argument("--phase-line-high-conf-weight", type=float, default=0.38)
    parser.add_argument("--phase-line-energy-low-conf-weight", type=float, default=0.15)
    parser.add_argument("--phase-line-energy-high-conf-weight", type=float, default=0.57)
    parser.add_argument("--phase-line-coherence-weight", type=float, default=0.03)
    parser.add_argument("--phase-line-scale-pi", type=float, default=0.28)
    parser.add_argument("--phase-line-max-drop-low-conf-db", type=float, default=18.0)
    parser.add_argument("--phase-line-max-drop-high-conf-db", type=float, default=3.0)
    parser.add_argument("--phase-line-header-reference-weight", type=float, default=0.0)
    return parser.parse_args()


def build_phase_line_config(args: argparse.Namespace) -> PhaseLineSelectorConfig:
    return PhaseLineSelectorConfig(
        top_l=int(args.phase_line_top_l),
        beam_width=int(args.phase_line_beam_width),
        window_size=int(args.phase_line_window_size),
        window_degree=int(args.phase_line_window_degree),
        recent_decay=float(args.phase_line_recent_decay),
        phase_weight_low_conf=float(args.phase_line_low_conf_weight),
        phase_weight_high_conf=float(args.phase_line_high_conf_weight),
        energy_weight_low_conf=float(args.phase_line_energy_low_conf_weight),
        energy_weight_high_conf=float(args.phase_line_energy_high_conf_weight),
        coherence_weight=float(args.phase_line_coherence_weight),
        phase_scale_pi=float(args.phase_line_scale_pi),
        max_energy_drop_db_low_conf=float(args.phase_line_max_drop_low_conf_db),
        max_energy_drop_db_high_conf=float(args.phase_line_max_drop_high_conf_db),
        header_reference_weight=float(args.phase_line_header_reference_weight),
        line_trim_frac=float(args.line_trim_frac),
    )


def _load_metadata(paths: dict[str, Path], dataset: str) -> dict[str, Any]:
    if paths["metadata"].exists():
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))
    low_snr_root = WEAK_ROOT / "data" / "low_snr_gt_bin"
    matches = sorted(low_snr_root.glob(f"{dataset}*/*_metadata.json"))
    for path in matches:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "signal_reference_power" in metadata:
            print(f"{dataset}: using metadata fallback {path}", flush=True)
            return metadata
    return {}


def _load_clean_payload_phase_map(symbol_csv: Path) -> dict[int, list[float]]:
    """Load clean payload GT peak phases from header-first symbol CSV."""

    phases_by_packet: dict[int, list[tuple[int, float]]] = {}
    with symbol_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("stage", "")).strip().lower() != "payload":
                continue
            try:
                packet_index = int(float(str(row.get("packet_index", "")).strip()))
                symbol_index = int(float(str(row.get("stage_symbol_index", "")).strip()))
                phase = float(str(row.get("peak_phase", "")).strip())
            except (TypeError, ValueError):
                continue
            phases_by_packet.setdefault(packet_index, []).append((symbol_index, phase))
    out: dict[int, list[float]] = {}
    for packet_index, items in phases_by_packet.items():
        out[int(packet_index)] = [float(phase) for _idx, phase in sorted(items, key=lambda item: item[0])]
    return out


def _prefix_decode(prefix: str, packet: dict[str, Any], raw_bins: Sequence[int], args: argparse.Namespace) -> dict[str, Any]:
    decoded = _decode_selected(packet, raw_bins, args)
    return {
        f"{prefix}_crc_valid": int(decoded.get("crc_valid", 0)),
        f"{prefix}_payload_hex": decoded.get("payload_hex", ""),
        f"{prefix}_decode_error": decoded.get("decode_error", ""),
    }


def _fit_gt_phase_line(
    center_spectra: Sequence[np.ndarray],
    gt_bins: Sequence[int],
    abs_indices: Sequence[float],
    clean_gt_phases: Sequence[float] | None = None,
) -> PhaseLine:
    xs: list[float] = []
    phases: list[float] = []
    for spectrum, gt_bin, abs_index in zip(center_spectra, gt_bins, abs_indices):
        b = int(gt_bin)
        if b < 0 or b >= np.asarray(spectrum).size:
            continue
        phase_index = len(phases)
        xs.append(float(abs_index))
        if clean_gt_phases is not None and phase_index < len(clean_gt_phases):
            phases.append(float(clean_gt_phases[phase_index]))
        else:
            phases.append(float(np.angle(np.asarray(spectrum, dtype=np.complex64)[b])))
    if len(xs) < 2:
        return PhaseLine()
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    x_arr = np.asarray(xs, dtype=np.float64)[order]
    phase_arr = np.unwrap(np.asarray(phases, dtype=np.float64)[order])
    return fit_phase_line(x_arr, phase_arr, trim_frac=0.20)


def _phase_oracle_bins(phase_result, gt_line: PhaseLine) -> tuple[int, ...]:
    selected: list[int] = []
    for ev in phase_result.evidences:
        if int(gt_line.anchor_count) < 2:
            selected.append(int(ev.top1_bin))
            continue
        best_bin = int(ev.top1_bin)
        best_score = -float("inf")
        top_power = float(ev.evidence_power[int(ev.top1_bin)]) if 0 <= int(ev.top1_bin) < ev.evidence_power.size else 0.0
        for raw_bin in ev.top_bins:
            b = int(raw_bin)
            if b < 0 or b >= ev.center_spectrum.size:
                continue
            energy_drop_db = 10.0 * math.log10(float(ev.evidence_power[b] + 1e-30) / float(top_power + 1e-30))
            if energy_drop_db < -12.0:
                continue
            energy_score = float(ev.evidence_power[b] / (top_power + 1e-30))
            residual = float(wrap_phase(float(np.angle(ev.center_spectrum[b])) - gt_line.predict(ev.abs_symbol_index)))
            phase_score = 0.5 + 0.5 * float(math.cos(residual))
            score = 0.80 * phase_score + 0.20 * max(0.0, min(1.0, energy_score))
            if score > best_score:
                best_score = score
                best_bin = b
        selected.append(int(best_bin))
    return tuple(selected)


def _local_clean_prediction(
    clean_gt_phases: Sequence[float] | None,
    abs_indices: Sequence[float],
    target_index: int,
    radius: int = 4,
) -> tuple[bool, float]:
    if clean_gt_phases is None:
        return False, 0.0
    idx = int(target_index)
    xs: list[float] = []
    phases: list[float] = []
    for j in range(max(0, idx - int(radius)), min(len(abs_indices), idx + int(radius) + 1)):
        if j == idx or j >= len(clean_gt_phases):
            continue
        xs.append(float(abs_indices[j]))
        phases.append(float(clean_gt_phases[j]))
    if len(xs) < 2:
        return False, 0.0
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    x_arr = np.asarray(xs, dtype=np.float64)[order]
    y_arr = np.unwrap(np.asarray(phases, dtype=np.float64)[order])
    try:
        coef = np.polyfit(x_arr - float(abs_indices[idx]), y_arr, deg=min(1, len(x_arr) - 1))
    except np.linalg.LinAlgError:
        return False, 0.0
    return True, float(np.polyval(coef, 0.0))


def _local_phase_oracle_bins(
    phase_result,
    abs_indices: Sequence[float],
    clean_gt_phases: Sequence[float] | None,
    fallback_line: PhaseLine,
) -> tuple[int, ...]:
    selected: list[int] = []
    for idx, ev in enumerate(phase_result.evidences):
        has_local, predicted = _local_clean_prediction(clean_gt_phases, abs_indices, idx)
        if not has_local:
            if int(fallback_line.anchor_count) < 2:
                selected.append(int(ev.top1_bin))
                continue
            predicted = float(fallback_line.predict(ev.abs_symbol_index))
        best_bin = int(ev.top1_bin)
        best_score = -float("inf")
        top_power = float(ev.evidence_power[int(ev.top1_bin)]) if 0 <= int(ev.top1_bin) < ev.evidence_power.size else 0.0
        for raw_bin in ev.top_bins:
            b = int(raw_bin)
            if b < 0 or b >= ev.center_spectrum.size:
                continue
            energy_drop_db = 10.0 * math.log10(float(ev.evidence_power[b] + 1e-30) / float(top_power + 1e-30))
            if energy_drop_db < -12.0:
                continue
            energy_score = float(ev.evidence_power[b] / (top_power + 1e-30))
            residual = float(wrap_phase(float(np.angle(ev.center_spectrum[b])) - predicted))
            phase_score = 0.5 + 0.5 * float(math.cos(residual))
            score = 0.80 * phase_score + 0.20 * max(0.0, min(1.0, energy_score))
            if score > best_score:
                best_score = score
                best_bin = b
        selected.append(int(best_bin))
    return tuple(selected)


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    v3_config,
    phase_config: PhaseLineSelectorConfig,
    clean_gt_phases: Sequence[float] | None,
) -> dict[str, Any]:
    center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
        samples, packet, args
    )
    evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
    header_line = _packet_phase_line(samples, packet, args)

    v3_result = select_symbol_bins_two_stage(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=v3_config,
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    phase_result = select_phase_smooth_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=phase_config,
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    gt_line = _fit_gt_phase_line(center_spectra, gt_bins, abs_indices, clean_gt_phases=clean_gt_phases)
    phase_oracle_bins = _local_phase_oracle_bins(phase_result, abs_indices, clean_gt_phases, gt_line)

    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    center_bins = _argmax_bins(center_spectra)
    multi_bins = _argmax_bins(multi_spectra)
    v3_bins = tuple(v3_result.selected_raw_bins)
    phase_bins = tuple(phase_result.selected_raw_bins)

    center_raw_ser, center_symbol_ser, compared = _ser(center_bins, gt_bins, sf=sf, ldro=ldro)
    multi_raw_ser, multi_symbol_ser, _ = _ser(multi_bins, gt_bins, sf=sf, ldro=ldro)
    v3_raw_ser, v3_symbol_ser, _ = _ser(v3_bins, gt_bins, sf=sf, ldro=ldro)
    phase_raw_ser, phase_symbol_ser, _ = _ser(phase_bins, gt_bins, sf=sf, ldro=ldro)
    oracle_raw_ser, oracle_symbol_ser, _ = _ser(phase_oracle_bins, gt_bins, sf=sf, ldro=ldro)
    phase_hits, phase_total = _candidate_recall(phase_result, gt_bins)
    phase_false_locks, phase_locked_with_gt = _false_locks(phase_result, gt_bins)

    row: dict[str, Any] = {
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "payload_len": int(packet["payload_len"]),
        "symbol_count": int(len(phase_bins)),
        "gt_compared_symbols": int(compared),
        "center_raw_ser": float(center_raw_ser),
        "center_symbol_ser": float(center_symbol_ser),
        "center_symbol_accuracy": float(1.0 - center_symbol_ser),
        "multi_raw_ser": float(multi_raw_ser),
        "multi_symbol_ser": float(multi_symbol_ser),
        "multi_symbol_accuracy": float(1.0 - multi_symbol_ser),
        "selected_raw_ser": float(phase_raw_ser),
        "selected_symbol_ser": float(phase_symbol_ser),
        "selected_symbol_accuracy": float(1.0 - phase_symbol_ser),
        "v3_raw_ser": float(v3_raw_ser),
        "v3_symbol_ser": float(v3_symbol_ser),
        "v3_symbol_accuracy": float(1.0 - v3_symbol_ser),
        "phase_line_raw_ser": float(phase_raw_ser),
        "phase_line_symbol_ser": float(phase_symbol_ser),
        "phase_line_symbol_accuracy": float(1.0 - phase_symbol_ser),
        "phase_oracle_raw_ser": float(oracle_raw_ser),
        "phase_oracle_symbol_ser": float(oracle_symbol_ser),
        "phase_oracle_symbol_accuracy": float(1.0 - oracle_symbol_ser),
        "phase_oracle_ser_gain_vs_v3": float(v3_symbol_ser - oracle_symbol_ser),
        "phase_line_ser_gain_vs_v3": float(v3_symbol_ser - phase_symbol_ser),
        "phase_line_locked_symbol_count": int(phase_result.locked_count),
        "phase_line_uncertain_symbol_count": int(phase_result.uncertain_count),
        "phase_line_false_lock_count": int(phase_false_locks),
        "phase_line_locked_with_gt_count": int(phase_locked_with_gt),
        "phase_line_false_lock_rate": float(phase_false_locks / max(1, phase_locked_with_gt)),
        "phase_line_uncertain_candidate_hits": int(phase_hits),
        "phase_line_uncertain_candidate_total": int(phase_total),
        "phase_line_uncertain_candidate_recall": float(phase_hits / max(1, phase_total)),
        "phase_line_phase_rmse_pi": float(phase_result.phase_line.fit_rmse_pi)
        if math.isfinite(float(phase_result.phase_line.fit_rmse_pi))
        else "",
        "phase_line_phase_r2": float(phase_result.phase_line.fit_r2)
        if math.isfinite(float(phase_result.phase_line.fit_r2))
        else "",
        "phase_line_error": phase_result.error,
        "phase_oracle_rmse_pi": float(gt_line.fit_rmse_pi) if math.isfinite(float(gt_line.fit_rmse_pi)) else "",
        "phase_oracle_r2": float(gt_line.fit_r2) if math.isfinite(float(gt_line.fit_r2)) else "",
        "phase_oracle_source": "clean_header_first" if clean_gt_phases is not None else "noisy_gt_bin",
        "v3_phase_rmse_pi": float(v3_result.phase_line.fit_rmse_pi)
        if math.isfinite(float(v3_result.phase_line.fit_rmse_pi))
        else "",
        "v3_phase_r2": float(v3_result.phase_line.fit_r2)
        if math.isfinite(float(v3_result.phase_line.fit_r2))
        else "",
    }
    row.update(_prefix_decode("center", packet, center_bins, args))
    row.update(_prefix_decode("multi", packet, multi_bins, args))
    row.update(_prefix_decode("v3", packet, v3_bins, args))
    row.update(_prefix_decode("phase_line", packet, phase_bins, args))
    row.update(_prefix_decode("phase_oracle", packet, phase_oracle_bins, args))
    row["selected_crc_valid"] = row["phase_line_crc_valid"]
    return row


def _summarize(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "packet_count": int(len(rows)),
        "center_symbol_ser": _avg(rows, "center_symbol_ser"),
        "center_symbol_accuracy": _avg(rows, "center_symbol_accuracy"),
        "center_crc_valid_rate": _avg(rows, "center_crc_valid"),
        "multi_symbol_ser": _avg(rows, "multi_symbol_ser"),
        "multi_symbol_accuracy": _avg(rows, "multi_symbol_accuracy"),
        "multi_crc_valid_rate": _avg(rows, "multi_crc_valid"),
        "v3_symbol_ser": _avg(rows, "v3_symbol_ser"),
        "v3_symbol_accuracy": _avg(rows, "v3_symbol_accuracy"),
        "v3_crc_valid_rate": _avg(rows, "v3_crc_valid"),
        "selected_symbol_ser": _avg(rows, "phase_line_symbol_ser"),
        "selected_symbol_accuracy": _avg(rows, "phase_line_symbol_accuracy"),
        "selected_crc_valid_rate": _avg(rows, "phase_line_crc_valid"),
        "phase_line_symbol_ser": _avg(rows, "phase_line_symbol_ser"),
        "phase_line_symbol_accuracy": _avg(rows, "phase_line_symbol_accuracy"),
        "phase_line_crc_valid_rate": _avg(rows, "phase_line_crc_valid"),
        "phase_oracle_symbol_ser": _avg(rows, "phase_oracle_symbol_ser"),
        "phase_oracle_symbol_accuracy": _avg(rows, "phase_oracle_symbol_accuracy"),
        "phase_oracle_crc_valid_rate": _avg(rows, "phase_oracle_crc_valid"),
        "phase_line_ser_gain_vs_v3": _avg(rows, "phase_line_ser_gain_vs_v3"),
        "phase_oracle_ser_gain_vs_v3": _avg(rows, "phase_oracle_ser_gain_vs_v3"),
        "selected_ser_gain_vs_center": _avg(rows, "center_symbol_ser") - _avg(rows, "phase_line_symbol_ser"),
        "selected_crc_gain_vs_center": _avg(rows, "phase_line_crc_valid") - _avg(rows, "center_crc_valid"),
        "selected_ser_gain_vs_multi": _avg(rows, "multi_symbol_ser") - _avg(rows, "phase_line_symbol_ser"),
        "selected_crc_gain_vs_multi": _avg(rows, "phase_line_crc_valid") - _avg(rows, "multi_crc_valid"),
        "mean_phase_line_false_lock_rate": _avg(rows, "phase_line_false_lock_rate"),
        "mean_phase_line_uncertain_candidate_recall": _avg(rows, "phase_line_uncertain_candidate_recall"),
        "mean_phase_line_rmse_pi": _avg(rows, "phase_line_phase_rmse_pi"),
        "mean_phase_line_r2": _avg(rows, "phase_line_phase_r2"),
    }


def _write_method_thresholds(path: Path, summary_rows: Sequence[dict[str, Any]]) -> None:
    methods = ("center", "multi", "v3", "phase_line")
    metrics = (
        ("SER<=10%", "symbol_ser", 0.10, False),
        ("accuracy>=90%", "symbol_accuracy", 0.90, True),
        ("CRC/PRR>=90%", "crc_valid_rate", 0.90, True),
        ("CRC/PRR>=80%", "crc_valid_rate", 0.80, True),
        ("CRC/PRR>=50%", "crc_valid_rate", 0.50, True),
    )
    rows: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in summary_rows}):
        curve = [row for row in summary_rows if str(row["dataset"]) == dataset]
        for method in methods:
            for metric_name, suffix, target, higher_is_better in metrics:
                key = f"{method}_{suffix}"
                if method == "phase_line":
                    key = f"phase_line_{suffix}"
                if method == "multi":
                    key = f"multi_{suffix}"
                threshold, status = _threshold_from_curve(curve, key, target, higher_is_better)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric_name,
                        "threshold_snr_db": "" if threshold is None else float(threshold),
                        "status": status,
                    }
                )
    _write_csv(path, rows)


def _threshold_from_curve(
    curve: Sequence[dict[str, Any]],
    metric_key: str,
    target: float,
    higher_is_better: bool,
) -> tuple[float | None, str]:
    points: list[tuple[float, float]] = []
    for row in curve:
        try:
            snr = float(row["target_snr_db"])
            value = float(row[metric_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(snr) and math.isfinite(value):
            points.append((snr, value))
    if not points:
        return None, "missing"
    points.sort(key=lambda item: item[0], reverse=True)

    def passes(value: float) -> bool:
        return value >= float(target) if higher_is_better else value <= float(target)

    flags = [passes(value) for _snr, value in points]
    if all(flags):
        return min(snr for snr, _value in points), "all_pass"
    if not any(flags):
        return None, "all_fail"
    for idx in range(len(points) - 1):
        snr_hi, value_hi = points[idx]
        snr_lo, value_lo = points[idx + 1]
        if passes(value_hi) and not passes(value_lo):
            if abs(value_lo - value_hi) <= 1e-12:
                return snr_hi, "flat_crossing"
            ratio = (float(target) - value_hi) / (value_lo - value_hi)
            ratio = max(0.0, min(1.0, float(ratio)))
            return float(snr_hi + ratio * (snr_lo - snr_hi)), "interpolated"
    return None, "non_monotonic"


def main() -> int:
    args = parse_args()
    v3_config = build_config(args)
    phase_config = build_phase_line_config(args)
    snrs = _snr_values(args.snr_start, args.snr_stop, args.snr_step)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        paths = _dataset_paths(str(dataset))
        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        packets = load_packets(paths["symbols"], None)
        clean_phase_map = _load_clean_payload_phase_map(paths["symbols"])
        if samples.size == 0:
            raise ValueError(f"empty IQ file: {paths['iq']}")
        if not packets:
            raise ValueError(f"no packets loaded: {paths['symbols']}")
        metadata = _load_metadata(paths, str(dataset))
        if args.signal_reference_power is not None:
            signal_power = float(args.signal_reference_power)
        elif "signal_reference_power" in metadata:
            signal_power = float(metadata["signal_reference_power"])
        else:
            signal_power = float(np.mean(np.abs(samples).astype(np.float64) ** 2))
            print(
                f"{dataset}: metadata missing; using mean clean IQ power as signal_reference_power={signal_power:.6g}",
                flush=True,
            )
        base_seed = int(metadata.get("seed", 42))

        unit_noise: np.ndarray | None = None
        if not bool(args.independent_noise):
            rng = np.random.default_rng(base_seed)
            noise_i = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
            noise_q = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
            unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)

        for step_index, snr_db in enumerate(snrs):
            noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
            if unit_noise is None:
                rng = np.random.default_rng(base_seed + step_index)
                sigma = math.sqrt(float(noise_power) / 2.0)
                noise_i = rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                noise_q = rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                noisy = (samples + (noise_i + 1j * noise_q)).astype(np.complex64, copy=False)
            else:
                sigma = math.sqrt(float(noise_power) / 2.0)
                noisy = (samples + sigma * unit_noise).astype(np.complex64, copy=False)

            rows: list[dict[str, Any]] = []
            for packet_index in sorted(packets):
                row = _evaluate_packet(
                    noisy,
                    packets[packet_index],
                    args,
                    v3_config,
                    phase_config,
                    clean_phase_map.get(int(packet_index)),
                )
                row["dataset"] = str(dataset)
                row["target_snr_db"] = float(snr_db)
                rows.append(row)
                packet_rows.append(row)
            summary = _summarize(rows, str(dataset), float(snr_db))
            summary_rows.append(summary)
            print(
                f"{dataset} snr={snr_db:>6.1f} "
                f"v3_ser={summary['v3_symbol_ser']:.3f} phase_ser={summary['phase_line_symbol_ser']:.3f} "
                f"v3_crc={summary['v3_crc_valid_rate']:.3f} phase_crc={summary['phase_line_crc_valid_rate']:.3f}",
                flush=True,
            )

    for snr_db in snrs:
        group = [row for row in summary_rows if abs(float(row["target_snr_db"]) - float(snr_db)) < 1e-9]
        aggregate: dict[str, Any] = {
            "dataset": "mean_of_datasets",
            "target_snr_db": float(snr_db),
            "packet_count": sum(int(row["packet_count"]) for row in group),
        }
        for key in summary_rows[0]:
            if key in aggregate or key in ("dataset", "target_snr_db", "packet_count"):
                continue
            aggregate[key] = float(np.mean([float(row[key]) for row in group]))
        summary_rows.append(aggregate)

    _write_csv(out_dir / "per_packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "snr_curve_summary.csv", summary_rows)
    threshold_rows, gain_rows = _threshold_tables(summary_rows)
    _write_csv(out_dir / "threshold_table_phase_as_selected.csv", threshold_rows)
    _write_csv(out_dir / "gain_table_phase_as_selected.csv", gain_rows)
    _write_method_thresholds(out_dir / "method_threshold_table.csv", summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "snr_values": snrs,
                "datasets": list(args.datasets),
                "phase_line_config": phase_config.__dict__,
                "summary_rows": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote={out_dir / 'snr_curve_summary.csv'}")
    print(f"wrote={out_dir / 'method_threshold_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

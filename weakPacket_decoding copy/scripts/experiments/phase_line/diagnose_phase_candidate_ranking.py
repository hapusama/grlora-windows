#!/usr/bin/env python3
"""Diagnose whether phase residual can rank the GT bin inside Top-L candidates."""

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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_phase_line_threshold_sweep import (  # noqa: E402
    _dataset_paths,
    _fit_gt_phase_line,
    _load_clean_payload_phase_map,
    _load_metadata,
)
from run_symbol_phase_two_stage import (  # noqa: E402
    _argmax_bins,
    _extract_payload_spectra_with_coherence,
    _packet_phase_line,
    build_config,
)
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.candidate_pruning import wrap_phase  # noqa: E402
from weak_decoder.phase_line import PhaseLineSelectorConfig  # noqa: E402
from weak_decoder.symbol_phase_two_stage import build_symbol_evidences, select_symbol_bins_two_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Top-L candidates by phase residual for diagnosis.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snr-db", type=float, default=-22.0)
    parser.add_argument("--output", type=Path, default=WEAK_ROOT / "data" / "phase_line_diagnostics" / "candidate_phase_ranking.csv")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--top-l", type=int, default=24)
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--packet", type=int, default=None)

    # Minimal v3 args needed by build_config defaults.
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


def _rank(values: dict[int, float], target: int, reverse: bool = True) -> int:
    order = sorted(values, key=lambda key: values[key], reverse=reverse)
    try:
        return int(order.index(int(target)) + 1)
    except ValueError:
        return 0


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
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


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else 0.0


def _local_clean_prediction(
    clean_phases: Sequence[float] | None,
    abs_indices: Sequence[float],
    target_index: int,
    radius: int = 4,
) -> tuple[bool, float]:
    if clean_phases is None:
        return False, 0.0
    idx = int(target_index)
    xs: list[float] = []
    phases: list[float] = []
    for j in range(max(0, idx - int(radius)), min(len(abs_indices), idx + int(radius) + 1)):
        if j == idx or j >= len(clean_phases):
            continue
        xs.append(float(abs_indices[j]))
        phases.append(float(clean_phases[j]))
    if len(xs) < 2:
        return False, 0.0
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    x_arr = np.asarray(xs, dtype=np.float64)[order]
    y_arr = np.unwrap(np.asarray(phases, dtype=np.float64)[order])
    degree = min(1, len(x_arr) - 1)
    try:
        coef = np.polyfit(x_arr - float(abs_indices[idx]), y_arr, deg=degree)
    except np.linalg.LinAlgError:
        return False, 0.0
    return True, float(np.polyval(coef, 0.0))


def main() -> int:
    args = parse_args()
    paths = _dataset_paths(str(args.dataset))
    metadata = _load_metadata(paths, str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {paths['iq']}")
    packets = load_packets(paths["symbols"], args.packet)
    clean_phase_map = _load_clean_payload_phase_map(paths["symbols"])
    if args.signal_reference_power is not None:
        signal_power = float(args.signal_reference_power)
    elif "signal_reference_power" in metadata:
        signal_power = float(metadata["signal_reference_power"])
    else:
        signal_power = float(np.mean(np.abs(samples).astype(np.float64) ** 2))
    base_seed = int(metadata.get("seed", 42))

    rng = np.random.default_rng(base_seed)
    noise_i = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    noise_q = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)
    noise_power = signal_power * (10.0 ** (-float(args.snr_db) / 10.0))
    noisy = (samples + math.sqrt(noise_power / 2.0) * unit_noise).astype(np.complex64, copy=False)

    v3_config = build_config(args)
    phase_stage1 = PhaseLineSelectorConfig(top_l=int(args.top_l))
    rows: list[dict[str, Any]] = []
    for packet_index in sorted(packets):
        packet = packets[packet_index]
        center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
            noisy, packet, args
        )
        evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
        header_line = _packet_phase_line(noisy, packet, args)
        v3 = select_symbol_bins_two_stage(
            center_spectra=center_spectra,
            evidence_powers=evidence_powers,
            abs_indices=abs_indices,
            config=v3_config,
            fallback_line=header_line,
            offset_coherences=coherences,
        )
        stage1_config = v3_config.__class__(
            top_l_low_confidence=int(phase_stage1.top_l),
            lock_margin_db=float("inf"),
            lock_peak_to_median_db=float("inf"),
            coherence_candidate_top_l=0,
        )
        evidences = build_symbol_evidences(center_spectra, evidence_powers, abs_indices, stage1_config, coherences)
        clean_line = _fit_gt_phase_line(
            center_spectra,
            gt_bins,
            abs_indices,
            clean_gt_phases=clean_phase_map.get(int(packet_index)),
        )
        clean_phases = clean_phase_map.get(int(packet_index))
        multi_bins = _argmax_bins(multi_spectra)
        for idx, ev in enumerate(evidences):
            if idx >= len(gt_bins):
                continue
            gt = int(gt_bins[idx])
            if gt < 0:
                continue
            candidates = [int(v) for v in ev.top_bins[: int(args.top_l)]]
            if gt not in candidates:
                rows.append(
                    {
                        "dataset": args.dataset,
                        "target_snr_db": float(args.snr_db),
                        "packet_index": int(packet_index),
                        "symbol_index": int(idx),
                        "gt_in_top_l": 0,
                    }
                )
                continue
            phase_scores: dict[int, float] = {}
            local_phase_scores: dict[int, float] = {}
            combined_scores: dict[int, float] = {}
            local_combined_scores: dict[int, float] = {}
            energy_scores: dict[int, float] = {}
            has_local_pred, local_pred = _local_clean_prediction(clean_phases, abs_indices, idx)
            top_power = float(ev.evidence_power[int(ev.top1_bin)]) if 0 <= int(ev.top1_bin) < ev.evidence_power.size else 0.0
            for b in candidates:
                energy = float(ev.evidence_power[b] / (top_power + 1e-30)) if 0 <= b < ev.evidence_power.size else 0.0
                residual = float(wrap_phase(float(np.angle(ev.center_spectrum[b])) - clean_line.predict(ev.abs_symbol_index)))
                phase_score = 0.5 + 0.5 * math.cos(residual)
                if has_local_pred:
                    local_residual = float(wrap_phase(float(np.angle(ev.center_spectrum[b])) - local_pred))
                    local_phase_score = 0.5 + 0.5 * math.cos(local_residual)
                else:
                    local_phase_score = 0.0
                energy_scores[b] = max(0.0, min(1.0, energy))
                phase_scores[b] = phase_score
                local_phase_scores[b] = local_phase_score
                combined_scores[b] = 0.80 * phase_score + 0.20 * energy_scores[b]
                local_combined_scores[b] = 0.80 * local_phase_score + 0.20 * energy_scores[b]
            v3_bin = int(v3.selected_raw_bins[idx]) if idx < len(v3.selected_raw_bins) else -1
            multi_bin = int(multi_bins[idx]) if idx < len(multi_bins) else -1
            rows.append(
                {
                    "dataset": args.dataset,
                    "target_snr_db": float(args.snr_db),
                    "packet_index": int(packet_index),
                    "symbol_index": int(idx),
                    "gt_in_top_l": 1,
                    "v3_is_error": int(v3_bin != gt),
                    "multi_is_error": int(multi_bin != gt),
                    "gt_energy_rank": _rank(energy_scores, gt, reverse=True),
                    "gt_phase_rank": _rank(phase_scores, gt, reverse=True),
                    "gt_local_phase_rank": _rank(local_phase_scores, gt, reverse=True) if has_local_pred else "",
                    "gt_combined_rank": _rank(combined_scores, gt, reverse=True),
                    "gt_local_combined_rank": _rank(local_combined_scores, gt, reverse=True) if has_local_pred else "",
                    "gt_energy_score": energy_scores.get(gt, ""),
                    "gt_phase_score": phase_scores.get(gt, ""),
                    "gt_local_phase_score": local_phase_scores.get(gt, ""),
                    "gt_combined_score": combined_scores.get(gt, ""),
                    "gt_local_combined_score": local_combined_scores.get(gt, ""),
                    "top1_bin": int(ev.top1_bin),
                    "gt_bin": gt,
                    "v3_bin": v3_bin,
                    "multi_bin": multi_bin,
                    "clean_line_r2": float(clean_line.fit_r2),
                    "clean_line_rmse_pi": float(clean_line.fit_rmse_pi),
                }
            )

    _write_csv(args.output, rows)
    top_rows = [row for row in rows if int(row.get("gt_in_top_l", 0)) == 1]
    v3_error_rows = [row for row in top_rows if int(row.get("v3_is_error", 0)) == 1]
    summary = {
        "dataset": args.dataset,
        "target_snr_db": float(args.snr_db),
        "rows": len(rows),
        "gt_recall_top_l": float(len(top_rows) / max(1, len(rows))),
        "mean_gt_energy_rank": _mean(top_rows, "gt_energy_rank"),
        "mean_gt_phase_rank": _mean(top_rows, "gt_phase_rank"),
        "mean_gt_local_phase_rank": _mean(top_rows, "gt_local_phase_rank"),
        "mean_gt_combined_rank": _mean(top_rows, "gt_combined_rank"),
        "mean_gt_local_combined_rank": _mean(top_rows, "gt_local_combined_rank"),
        "v3_error_rows": len(v3_error_rows),
        "v3_error_mean_gt_phase_rank": _mean(v3_error_rows, "gt_phase_rank"),
        "v3_error_mean_gt_local_phase_rank": _mean(v3_error_rows, "gt_local_phase_rank"),
        "v3_error_mean_gt_combined_rank": _mean(v3_error_rows, "gt_combined_rank"),
        "v3_error_mean_gt_local_combined_rank": _mean(v3_error_rows, "gt_local_combined_rank"),
        "v3_error_gt_phase_rank1_rate": float(
            sum(1 for row in v3_error_rows if int(row.get("gt_phase_rank", 0)) == 1)
            / max(1, len(v3_error_rows))
        ),
        "v3_error_gt_local_phase_rank1_rate": float(
            sum(1 for row in v3_error_rows if str(row.get("gt_local_phase_rank", "")).strip() != "" and int(row.get("gt_local_phase_rank", 0)) == 1)
            / max(1, len(v3_error_rows))
        ),
        "v3_error_gt_combined_rank1_rate": float(
            sum(1 for row in v3_error_rows if int(row.get("gt_combined_rank", 0)) == 1)
            / max(1, len(v3_error_rows))
        ),
        "v3_error_gt_local_combined_rank1_rate": float(
            sum(1 for row in v3_error_rows if str(row.get("gt_local_combined_rank", "")).strip() != "" and int(row.get("gt_local_combined_rank", 0)) == 1)
            / max(1, len(v3_error_rows))
        ),
    }
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote={args.output}")
    print(f"wrote_summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

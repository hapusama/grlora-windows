#!/usr/bin/env python3
"""Diagnose whether FFT-bin-dependent phase correction helps phase_line.

This script is intentionally evaluation-only.  It uses GT labels only to
measure whether a non-GT correction would be worth adding to the selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


THIS = Path(__file__).resolve()
PHASE_LINE_DIR = THIS.parents[1]
WEAK_DECODER_DIR = THIS.parents[2]
WEAK_ROOT = THIS.parents[3]
EXPERIMENT_DIR = WEAK_ROOT / "scripts" / "experiments"
PHASE_EXPERIMENT_DIR = EXPERIMENT_DIR / "phase_line"
for path in (str(WEAK_ROOT), str(EXPERIMENT_DIR), str(PHASE_EXPERIMENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_phase_line_threshold_sweep import _dataset_paths, _load_metadata  # noqa: E402
from run_symbol_phase_two_stage import _extract_payload_spectra_with_coherence  # noqa: E402
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.candidate_pruning import wrap_phase  # noqa: E402
from weak_decoder.phase_line import PhasePathSelectorConfig  # noqa: E402
from weak_decoder.phase_line.selector import (  # noqa: E402
    _augment_evidences_with_phase_proposals,
    _is_hard_anchor,
    _is_high_confidence,
    _viterbi_candidates,
)
from weak_decoder.symbol_phase_two_stage import SymbolPhaseConfig, build_symbol_evidences  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check bin-dependent phase correction for phase_line.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-24.0, -25.0])
    parser.add_argument("--top-l", type=int, default=24)
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PHASE_LINE_DIR / "_eval" / "bin_phase_correction")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--beta-min", type=float, default=-0.035)
    parser.add_argument("--beta-max", type=float, default=0.035)
    parser.add_argument("--beta-steps", type=int, default=281)
    parser.add_argument(
        "--fixed-betas",
        nargs="+",
        type=float,
        default=[0.0017, -0.0017, 0.0030679615757712823, -0.0030679615757712823],
    )
    return parser.parse_args()


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


def _fit_rmse(
    xs: Sequence[float],
    phases: Sequence[float],
    weights: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    if len(xs) < 2:
        return float("nan"), 0.0, 0.0
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    x = np.asarray(xs, dtype=np.float64)[order]
    y = np.unwrap(np.asarray(phases, dtype=np.float64)[order])
    w = None if weights is None else np.asarray(weights, dtype=np.float64)[order]
    try:
        coef = np.polyfit(x - float(np.mean(x)), y, deg=1, w=w)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan"), 0.0, 0.0
    fitted = np.polyval(coef, x - float(np.mean(x)))
    if w is not None and np.sum(w) > 0.0:
        rmse = math.sqrt(float(np.average((y - fitted) ** 2, weights=w)))
    else:
        rmse = math.sqrt(float(np.mean((y - fitted) ** 2)))
    return float(rmse / math.pi), float(coef[0]), float(coef[1] - coef[0] * float(np.mean(x)))


def _best_beta(
    xs: Sequence[float],
    bins: Sequence[int],
    phases: Sequence[float],
    weights: Sequence[float] | None,
    beta_grid: np.ndarray,
) -> tuple[float, float, float, float]:
    if len(xs) < 3:
        return 0.0, float("nan"), 0.0, 0.0
    raw_rmse, raw_slope, raw_intercept = _fit_rmse(xs, phases, weights)
    best_beta = 0.0
    best_rmse = raw_rmse
    best_slope = raw_slope
    best_intercept = raw_intercept
    b_arr = np.asarray(bins, dtype=np.float64)
    p_arr = np.asarray(phases, dtype=np.float64)
    for beta in beta_grid:
        corrected = np.angle(np.exp(1j * (p_arr - float(beta) * b_arr)))
        rmse, slope, intercept = _fit_rmse(xs, corrected, weights)
        if math.isfinite(rmse) and (not math.isfinite(best_rmse) or rmse < best_rmse):
            best_beta = float(beta)
            best_rmse = float(rmse)
            best_slope = float(slope)
            best_intercept = float(intercept)
    return float(best_beta), float(best_rmse), float(best_slope), float(best_intercept)


def _rank(scores: dict[int, float], target: int) -> int:
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    try:
        return int(ordered.index(int(target)) + 1)
    except ValueError:
        return 0


def _local_prediction(
    evidences: Sequence[Any],
    anchor_mask: Sequence[bool],
    target_index: int,
    beta: float,
    radius: int = 8,
    min_anchors: int = 4,
) -> tuple[bool, float, float]:
    idx = int(target_index)
    target_abs = float(evidences[idx].abs_symbol_index)
    rows: list[tuple[float, float, float]] = []
    for j in range(max(0, idx - radius), min(len(evidences), idx + radius + 1)):
        if j == idx or j >= len(anchor_mask) or not bool(anchor_mask[j]):
            continue
        ev = evidences[j]
        b = int(ev.top1_bin)
        if b < 0 or b >= ev.center_spectrum.size:
            continue
        phase = float(wrap_phase(float(np.angle(ev.center_spectrum[b])) - float(beta) * b))
        distance = abs(float(ev.abs_symbol_index) - target_abs)
        weight = math.exp(-0.5 * (distance / max(1e-6, 0.5 * radius)) ** 2)
        rows.append((float(ev.abs_symbol_index), phase, float(weight)))
    if len(rows) < min_anchors:
        return False, 0.0, float("nan")
    rows.sort(key=lambda item: item[0])
    xs = np.asarray([row[0] for row in rows], dtype=np.float64)
    phases = np.unwrap(np.asarray([row[1] for row in rows], dtype=np.float64))
    weights = np.asarray([row[2] for row in rows], dtype=np.float64)
    try:
        coef = np.polyfit(xs - target_abs, phases, deg=1, w=weights)
    except np.linalg.LinAlgError:
        return False, 0.0, float("nan")
    fitted = np.polyval(coef, xs - target_abs)
    rmse = math.sqrt(float(np.average((phases - fitted) ** 2, weights=weights))) / math.pi
    return True, float(np.polyval(coef, 0.0)), float(rmse)


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


def _rate(rows: Sequence[dict[str, Any]], key: str, value: int = 1) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if int(row.get(key, 0)) == int(value)) / len(rows))


def _beta_label(beta: float) -> str:
    return f"beta_{float(beta):+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")


def main() -> int:
    args = parse_args()
    paths = _dataset_paths(str(args.dataset))
    metadata = _load_metadata(paths, str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    packets = load_packets(paths["symbols"], args.packet)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {paths['iq']}")
    if args.signal_reference_power is not None:
        signal_power = float(args.signal_reference_power)
    elif "signal_reference_power" in metadata:
        signal_power = float(metadata["signal_reference_power"])
    else:
        signal_power = float(np.mean(np.abs(samples).astype(np.float64) ** 2))

    base_seed = int(metadata.get("seed", 42))
    rng = np.random.default_rng(base_seed)
    unit_noise = (
        rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        + 1j * rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    ).astype(np.complex64)
    beta_grid = np.linspace(float(args.beta_min), float(args.beta_max), int(args.beta_steps), dtype=np.float64)

    path_config = PhasePathSelectorConfig(top_l=int(args.top_l))
    stage1_config = SymbolPhaseConfig(
        top_l_low_confidence=int(path_config.top_l),
        lock_margin_db=float("inf"),
        lock_peak_to_median_db=float("inf"),
        coherence_candidate_top_l=0,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []

    for snr_db in args.snrs:
        noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
        noisy = (samples + math.sqrt(noise_power / 2.0) * unit_noise).astype(np.complex64, copy=False)
        snr_detail_rows: list[dict[str, Any]] = []
        snr_packet_rows: list[dict[str, Any]] = []

        for packet_index in sorted(packets):
            packet = packets[packet_index]
            center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
                noisy,
                packet,
                SimpleNamespace(
                    cfo_correction_mode=str(args.cfo_correction_mode),
                    preamble_len=float(args.preamble_len),
                ),
            )
            evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
            evidences = tuple(build_symbol_evidences(center_spectra, evidence_powers, abs_indices, stage1_config, coherences))
            anchor_mask = [bool(_is_high_confidence(ev, path_config)) for ev in evidences]
            evidences = _augment_evidences_with_phase_proposals(evidences, anchor_mask, path_config)
            hard_anchor_mask = tuple(bool(_is_hard_anchor(ev, path_config)) for ev in evidences)

            gt_x: list[float] = []
            gt_b: list[int] = []
            gt_p: list[float] = []
            for ev, gt in zip(evidences, gt_bins):
                b = int(gt)
                if b < 0 or b >= ev.center_spectrum.size:
                    continue
                gt_x.append(float(ev.abs_symbol_index))
                gt_b.append(b)
                gt_p.append(float(np.angle(ev.center_spectrum[b])))
            gt_raw_rmse, _gt_raw_slope, _gt_raw_intercept = _fit_rmse(gt_x, gt_p)
            gt_beta, gt_best_rmse, _gt_slope, _gt_intercept = _best_beta(gt_x, gt_b, gt_p, None, beta_grid)

            anchor_x: list[float] = []
            anchor_b: list[int] = []
            anchor_p: list[float] = []
            anchor_w: list[float] = []
            anchor_false = 0
            anchor_total = 0
            for idx, ev in enumerate(evidences):
                if idx >= len(hard_anchor_mask) or not bool(hard_anchor_mask[idx]):
                    continue
                b = int(ev.top1_bin)
                if b < 0 or b >= ev.center_spectrum.size:
                    continue
                anchor_x.append(float(ev.abs_symbol_index))
                anchor_b.append(b)
                anchor_p.append(float(np.angle(ev.center_spectrum[b])))
                anchor_w.append(float(1.0 + max(0.0, min(2.0, ev.top1_margin_db / 8.0))))
                if idx < len(gt_bins) and int(gt_bins[idx]) >= 0:
                    anchor_total += 1
                    anchor_false += int(b != int(gt_bins[idx]))
            anchor_raw_rmse, _anchor_raw_slope, _anchor_raw_intercept = _fit_rmse(anchor_x, anchor_p, anchor_w)
            anchor_beta, anchor_best_rmse, _anchor_slope, _anchor_intercept = _best_beta(
                anchor_x, anchor_b, anchor_p, anchor_w, beta_grid
            )

            packet_rows.append(
                {
                    "dataset": args.dataset,
                    "target_snr_db": float(snr_db),
                    "packet_index": int(packet_index),
                    "gt_count": int(len(gt_x)),
                    "gt_raw_rmse_pi": float(gt_raw_rmse),
                    "gt_best_beta_rad_per_bin": float(gt_beta),
                    "gt_best_rmse_pi": float(gt_best_rmse),
                    "gt_rmse_gain_pi": float(gt_raw_rmse - gt_best_rmse) if math.isfinite(gt_raw_rmse) and math.isfinite(gt_best_rmse) else "",
                    "hard_anchor_count": int(len(anchor_x)),
                    "hard_anchor_false_count": int(anchor_false),
                    "hard_anchor_false_rate": float(anchor_false / max(1, anchor_total)),
                    "anchor_raw_rmse_pi": float(anchor_raw_rmse),
                    "anchor_best_beta_rad_per_bin": float(anchor_beta),
                    "anchor_best_rmse_pi": float(anchor_best_rmse),
                    "anchor_rmse_gain_pi": float(anchor_raw_rmse - anchor_best_rmse)
                    if math.isfinite(anchor_raw_rmse) and math.isfinite(anchor_best_rmse)
                    else "",
                }
            )
            snr_packet_rows.append(packet_rows[-1])

            for idx, ev in enumerate(evidences):
                if idx >= len(gt_bins):
                    continue
                gt = int(gt_bins[idx])
                if gt < 0:
                    continue
                cand_bins = [int(c.raw_bin) for c in _viterbi_candidates(ev, path_config)]
                if gt not in cand_bins:
                    row = {
                        "dataset": args.dataset,
                        "target_snr_db": float(snr_db),
                        "packet_index": int(packet_index),
                        "symbol_index": int(idx),
                        "gt_in_candidates": 0,
                    }
                    detail_rows.append(row)
                    snr_detail_rows.append(row)
                    continue
                no_has, no_pred, no_rmse = _local_prediction(evidences, hard_anchor_mask, idx, beta=0.0)
                beta_has, beta_pred, beta_rmse = _local_prediction(evidences, hard_anchor_mask, idx, beta=anchor_beta)
                no_scores: dict[int, float] = {}
                beta_scores: dict[int, float] = {}
                fixed_scores: dict[str, dict[int, float]] = {str(_beta_label(v)): {} for v in args.fixed_betas}
                fixed_preds: dict[str, tuple[bool, float, float]] = {
                    str(_beta_label(v)): _local_prediction(evidences, hard_anchor_mask, idx, beta=float(v))
                    for v in args.fixed_betas
                }
                for b in cand_bins:
                    raw_phase = float(np.angle(ev.center_spectrum[b]))
                    if no_has:
                        no_scores[b] = 0.5 + 0.5 * math.cos(float(wrap_phase(raw_phase - no_pred)))
                    if beta_has:
                        corr_phase = float(wrap_phase(raw_phase - anchor_beta * b))
                        beta_scores[b] = 0.5 + 0.5 * math.cos(float(wrap_phase(corr_phase - beta_pred)))
                    for fixed_beta in args.fixed_betas:
                        label = str(_beta_label(float(fixed_beta)))
                        fixed_has, fixed_pred, _fixed_rmse = fixed_preds[label]
                        if fixed_has:
                            corr_phase = float(wrap_phase(raw_phase - float(fixed_beta) * b))
                            fixed_scores[label][b] = 0.5 + 0.5 * math.cos(float(wrap_phase(corr_phase - fixed_pred)))
                no_rank = _rank(no_scores, gt) if no_scores else 0
                beta_rank = _rank(beta_scores, gt) if beta_scores else 0
                row = {
                    "dataset": args.dataset,
                    "target_snr_db": float(snr_db),
                    "packet_index": int(packet_index),
                    "symbol_index": int(idx),
                    "gt_in_candidates": 1,
                    "gt_rank_no_beta": int(no_rank),
                    "gt_rank_anchor_beta": int(beta_rank),
                    "rank_improved": int(beta_rank > 0 and (no_rank == 0 or beta_rank < no_rank)),
                    "rank_worsened": int(no_rank > 0 and (beta_rank == 0 or beta_rank > no_rank)),
                    "no_beta_local_rmse_pi": float(no_rmse),
                    "anchor_beta_local_rmse_pi": float(beta_rmse),
                    "anchor_beta_rad_per_bin": float(anchor_beta),
                    "hard_anchor_count": int(len(anchor_x)),
                    "hard_anchor_false_rate": float(anchor_false / max(1, anchor_total)),
                    "gt_bin": int(gt),
                    "top1_bin": int(ev.top1_bin),
                }
                for fixed_beta in args.fixed_betas:
                    label = str(_beta_label(float(fixed_beta)))
                    fixed_rank = _rank(fixed_scores[label], gt) if fixed_scores[label] else 0
                    row[f"gt_rank_{label}"] = int(fixed_rank)
                    row[f"{label}_local_rmse_pi"] = float(fixed_preds[label][2])
                    row[f"{label}_rank_improved"] = int(fixed_rank > 0 and (no_rank == 0 or fixed_rank < no_rank))
                    row[f"{label}_rank_worsened"] = int(no_rank > 0 and (fixed_rank == 0 or fixed_rank > no_rank))
                detail_rows.append(row)
                snr_detail_rows.append(row)

        candidate_rows = [row for row in snr_detail_rows if int(row.get("gt_in_candidates", 0)) == 1]
        no_ranked = [row for row in candidate_rows if int(row.get("gt_rank_no_beta", 0)) > 0]
        beta_ranked = [row for row in candidate_rows if int(row.get("gt_rank_anchor_beta", 0)) > 0]
        summary = {
            "dataset": args.dataset,
            "target_snr_db": float(snr_db),
            "packet_count": int(len(snr_packet_rows)),
            "symbol_rows": int(len(snr_detail_rows)),
            "candidate_recall": float(len(candidate_rows) / max(1, len(snr_detail_rows))),
            "mean_gt_raw_rmse_pi": _mean(snr_packet_rows, "gt_raw_rmse_pi"),
            "mean_gt_best_rmse_pi": _mean(snr_packet_rows, "gt_best_rmse_pi"),
            "mean_gt_rmse_gain_pi": _mean(snr_packet_rows, "gt_rmse_gain_pi"),
            "mean_abs_gt_best_beta_rad_per_bin": _mean(
                [{"v": abs(float(row["gt_best_beta_rad_per_bin"]))} for row in snr_packet_rows],
                "v",
            ),
            "mean_hard_anchor_count": _mean(snr_packet_rows, "hard_anchor_count"),
            "mean_hard_anchor_false_rate": _mean(snr_packet_rows, "hard_anchor_false_rate"),
            "mean_anchor_raw_rmse_pi": _mean(snr_packet_rows, "anchor_raw_rmse_pi"),
            "mean_anchor_best_rmse_pi": _mean(snr_packet_rows, "anchor_best_rmse_pi"),
            "mean_anchor_rmse_gain_pi": _mean(snr_packet_rows, "anchor_rmse_gain_pi"),
            "mean_abs_anchor_beta_rad_per_bin": _mean(
                [{"v": abs(float(row["anchor_best_beta_rad_per_bin"]))} for row in snr_packet_rows],
                "v",
            ),
            "mean_gt_rank_no_beta": _mean(no_ranked, "gt_rank_no_beta"),
            "mean_gt_rank_anchor_beta": _mean(beta_ranked, "gt_rank_anchor_beta"),
            "gt_rank1_no_beta_rate": _rate(no_ranked, "gt_rank_no_beta", 1),
            "gt_rank1_anchor_beta_rate": _rate(beta_ranked, "gt_rank_anchor_beta", 1),
            "rank_improved_rate": _rate(candidate_rows, "rank_improved", 1),
            "rank_worsened_rate": _rate(candidate_rows, "rank_worsened", 1),
        }
        for fixed_beta in args.fixed_betas:
            label = str(_beta_label(float(fixed_beta)))
            ranked = [row for row in candidate_rows if int(row.get(f"gt_rank_{label}", 0)) > 0]
            summary[f"mean_gt_rank_{label}"] = _mean(ranked, f"gt_rank_{label}")
            summary[f"gt_rank1_{label}_rate"] = _rate(ranked, f"gt_rank_{label}", 1)
            summary[f"{label}_rank_improved_rate"] = _rate(candidate_rows, f"{label}_rank_improved", 1)
            summary[f"{label}_rank_worsened_rate"] = _rate(candidate_rows, f"{label}_rank_worsened", 1)
        summary_rows.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    _write_csv(out_dir / "packet_beta_summary.csv", packet_rows)
    _write_csv(out_dir / "symbol_beta_ranking.csv", detail_rows)
    _write_csv(out_dir / "snr_beta_summary.csv", summary_rows)
    (out_dir / "snr_beta_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={out_dir / 'snr_beta_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

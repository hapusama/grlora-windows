#!/usr/bin/env python3
"""GT-only diagnostics for phase models on ambiguous Top-8 symbols.

This script does not change decoding decisions and does not use GT at inference
time.  It uses GT only to measure whether alternative phase predictors would
rank the correct bin higher when multi-offset Top-1 is wrong but the correct
bin is already present in the Top-8 candidate set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_symbol_phase_two_stage import _extract_payload_spectra, _packet_phase_line  # noqa: E402
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.candidate_pruning import wrap_phase  # noqa: E402
from weak_decoder.symbol_phase_two_stage import (  # noqa: E402
    SymbolPhaseConfig,
    _fit_line_for_bins,
    build_symbol_evidences,
)


DEFAULT_DATASETS = (
    "0_0_0_10_14_8",
    "0_0_0_10_14_16",
    "0_0_0_10_14_32",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose phase model ranking on ambiguous Top-8 symbols.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snrs", type=float, nargs="+", default=[-18.0, -19.0, -20.0, -21.0, -22.0])
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "symbol_phase_model_diagnostics")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--top-l-low-confidence", type=int, default=8)
    parser.add_argument("--lock-margin-db", type=float, default=1.5)
    parser.add_argument("--lock-peak-to-median-db", type=float, default=5.0)
    parser.add_argument("--lock-phase-score", type=float, default=0.35)
    parser.add_argument("--line-trim-frac", type=float, default=0.25)
    parser.add_argument("--local-window-anchors", type=int, default=10)
    return parser.parse_args()


def _dataset_paths(dataset: str) -> dict[str, Path]:
    return {
        "iq": WEAK_ROOT.parent / "data" / "USRP_IQ" / f"{dataset}.bin",
        "symbols": WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / f"{dataset}_header_first_symbols.csv",
        "metadata": WEAK_ROOT / "data" / "low_snr_gt_bin" / dataset / f"{dataset}_low_snr_gt_bin_metadata.json",
    }


def _phase_score(phase: float, pred: float) -> float:
    residual = float(wrap_phase(float(phase) - float(pred)))
    return float(0.5 + 0.5 * math.cos(residual))


def _linear_predictor(xs: Sequence[float], phases: Sequence[float]) -> Callable[[float], float] | None:
    x = np.asarray(xs, dtype=np.float64)
    y = np.unwrap(np.asarray(phases, dtype=np.float64))
    if x.size < 2:
        return None
    coef = np.polyfit(x, y, deg=1)
    return lambda value: float(np.polyval(coef, float(value)))


def _quadratic_predictor(xs: Sequence[float], phases: Sequence[float]) -> Callable[[float], float] | None:
    x = np.asarray(xs, dtype=np.float64)
    y = np.unwrap(np.asarray(phases, dtype=np.float64))
    if x.size < 3:
        return _linear_predictor(xs, phases)
    coef = np.polyfit(x, y, deg=2)
    return lambda value: float(np.polyval(coef, float(value)))


def _local_linear_prediction(xs: Sequence[float], phases: Sequence[float], target_x: float, count: int) -> float | None:
    x = np.asarray(xs, dtype=np.float64)
    y = np.unwrap(np.asarray(phases, dtype=np.float64))
    if x.size < 2:
        return None
    keep = np.argsort(np.abs(x - float(target_x)))[: max(2, min(int(count), x.size))]
    predictor = _linear_predictor(x[keep], y[keep])
    if predictor is None:
        return None
    return predictor(float(target_x))


def _fit_anchor_points(evidences, selected_bins, locked_mask, trim_frac: float) -> tuple[list[float], list[float], Any]:
    indexes = [idx for idx, locked in enumerate(locked_mask) if locked]
    xs: list[float] = []
    phases: list[float] = []
    for idx in indexes:
        ev = evidences[idx]
        b = int(selected_bins[idx])
        if 0 <= b < ev.center_spectrum.size:
            xs.append(float(ev.abs_symbol_index))
            phases.append(float(np.angle(ev.center_spectrum[b])))
    line = _fit_line_for_bins(
        [ev for ev, locked in zip(evidences, locked_mask) if locked],
        [b for b, locked in zip(selected_bins, locked_mask) if locked],
        trim_frac=float(trim_frac),
    )
    return xs, phases, line


def _refined_locked(evidences, selected_bins, config: SymbolPhaseConfig, fallback_line) -> tuple[list[bool], Any]:
    locked = [bool(ev.locked) for ev in evidences]
    line = _fit_line_for_bins(
        [ev for ev, is_locked in zip(evidences, locked) if is_locked],
        [bin_value for bin_value, is_locked in zip(selected_bins, locked) if is_locked],
        trim_frac=float(config.line_trim_frac),
    )
    if int(line.anchor_count) < max(2, int(config.min_locked_for_line)) and fallback_line is not None:
        line = fallback_line
    if int(line.anchor_count) >= max(2, int(config.min_locked_for_line)):
        refined: list[bool] = []
        for ev, is_locked in zip(evidences, locked):
            if not is_locked:
                refined.append(False)
                continue
            score = _phase_score(float(np.angle(ev.center_spectrum[ev.top1_bin])), line.predict(ev.abs_symbol_index))
            refined.append(bool(score >= float(config.lock_phase_score)))
        if sum(refined) >= max(2, int(config.min_locked_for_line)):
            locked = refined
            line = _fit_line_for_bins(
                [ev for ev, is_locked in zip(evidences, locked) if is_locked],
                [bin_value for bin_value, is_locked in zip(selected_bins, locked) if is_locked],
                trim_frac=float(config.line_trim_frac),
            )
    return locked, line


def _rank(values: Sequence[tuple[int, float]], target_bin: int) -> int:
    ordered = sorted(values, key=lambda item: item[1], reverse=True)
    for idx, (bin_value, _score) in enumerate(ordered, start=1):
        if int(bin_value) == int(target_bin):
            return int(idx)
    return 0


def _diagnose_packet(samples: np.ndarray, packet: dict[str, Any], args: argparse.Namespace, config: SymbolPhaseConfig) -> list[dict[str, Any]]:
    center_spectra, multi_spectra, abs_indices, gt_bins = _extract_payload_spectra(samples, packet, args)
    powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
    evidences = list(build_symbol_evidences(center_spectra, powers, abs_indices, config))
    selected = [int(ev.top1_bin) for ev in evidences]
    header_line = _packet_phase_line(samples, packet, args)
    locked, lock_line = _refined_locked(evidences, selected, config, header_line)
    anchor_xs, anchor_phases, _ = _fit_anchor_points(evidences, selected, locked, float(config.line_trim_frac))
    linear_pred = _linear_predictor(anchor_xs, anchor_phases)
    quad_pred = _quadratic_predictor(anchor_xs, anchor_phases)

    # Oracle leave-one-out GT predictors are upper-bound diagnostics only.
    gt_xs: list[float] = []
    gt_phases: list[float] = []
    for ev, gt in zip(evidences, gt_bins):
        b = int(gt)
        if 0 <= b < ev.center_spectrum.size:
            gt_xs.append(float(ev.abs_symbol_index))
            gt_phases.append(float(np.angle(ev.center_spectrum[b])))

    rows: list[dict[str, Any]] = []
    for idx, ev in enumerate(evidences):
        if idx >= len(gt_bins):
            continue
        gt = int(gt_bins[idx])
        top = tuple(int(v) for v in ev.top_bins)
        if gt < 0 or int(ev.top1_bin) == gt or gt not in set(top):
            continue
        candidates = top[: int(config.top_l_low_confidence)]
        candidate_phases = {b: float(np.angle(ev.center_spectrum[b])) for b in candidates}

        scores: dict[str, list[tuple[int, float]]] = {"linear": [], "quadratic": [], "local_linear": [], "oracle_gt_linear_loo": [], "oracle_gt_quadratic_loo": []}
        for b in candidates:
            phase = candidate_phases[int(b)]
            if linear_pred is not None:
                scores["linear"].append((int(b), _phase_score(phase, linear_pred(ev.abs_symbol_index))))
            if quad_pred is not None:
                scores["quadratic"].append((int(b), _phase_score(phase, quad_pred(ev.abs_symbol_index))))
            local_pred = _local_linear_prediction(anchor_xs, anchor_phases, ev.abs_symbol_index, int(args.local_window_anchors))
            if local_pred is not None:
                scores["local_linear"].append((int(b), _phase_score(phase, local_pred)))

            loo_xs = [x for j, x in enumerate(gt_xs) if j != idx]
            loo_phases = [p for j, p in enumerate(gt_phases) if j != idx]
            oracle_linear = _linear_predictor(loo_xs, loo_phases)
            oracle_quad = _quadratic_predictor(loo_xs, loo_phases)
            if oracle_linear is not None:
                scores["oracle_gt_linear_loo"].append((int(b), _phase_score(phase, oracle_linear(ev.abs_symbol_index))))
            if oracle_quad is not None:
                scores["oracle_gt_quadratic_loo"].append((int(b), _phase_score(phase, oracle_quad(ev.abs_symbol_index))))

        top1_power = float(ev.evidence_power[int(ev.top1_bin)])
        gt_power = float(ev.evidence_power[gt])
        row: dict[str, Any] = {
            "packet_index": int(packet["packet_index"]),
            "payload_symbol_index": int(idx),
            "gt_bin": int(gt),
            "multi_top1_bin": int(ev.top1_bin),
            "top1_margin_db": float(ev.top1_margin_db),
            "gt_energy_drop_db": float(10.0 * math.log10((gt_power + 1e-30) / (top1_power + 1e-30))),
            "locked_count": int(sum(locked)),
            "linear_anchor_count": int(len(anchor_xs)),
            "linear_line_rmse_pi": float(lock_line.fit_rmse_pi) if math.isfinite(lock_line.fit_rmse_pi) else "",
            "linear_line_r2": float(lock_line.fit_r2) if math.isfinite(lock_line.fit_r2) else "",
        }
        for model, model_scores in scores.items():
            row[f"{model}_gt_rank"] = _rank(model_scores, gt) if model_scores else 0
            row[f"{model}_top1_rank"] = _rank(model_scores, ev.top1_bin) if model_scores else 0
            score_map = {int(b): float(score) for b, score in model_scores}
            row[f"{model}_gt_score"] = score_map.get(gt, "")
            row[f"{model}_top1_score"] = score_map.get(int(ev.top1_bin), "")
            if gt in score_map and int(ev.top1_bin) in score_map:
                row[f"{model}_gt_minus_top1"] = float(score_map[gt] - score_map[int(ev.top1_bin)])
            else:
                row[f"{model}_gt_minus_top1"] = ""
        rows.append(row)
    return rows


def _summarize(rows: Sequence[dict[str, Any]], dataset: str, snr: float) -> list[dict[str, Any]]:
    models = ("linear", "quadratic", "local_linear", "oracle_gt_linear_loo", "oracle_gt_quadratic_loo")
    out: list[dict[str, Any]] = []
    for model in models:
        ranks = [int(row.get(f"{model}_gt_rank", 0)) for row in rows if int(row.get(f"{model}_gt_rank", 0)) > 0]
        deltas = []
        for row in rows:
            value = row.get(f"{model}_gt_minus_top1", "")
            try:
                deltas.append(float(value))
            except (TypeError, ValueError):
                pass
        out.append(
            {
                "dataset": dataset,
                "target_snr_db": float(snr),
                "model": model,
                "ambiguous_count": int(len(rows)),
                "gt_phase_rank1_rate": float(np.mean([rank == 1 for rank in ranks])) if ranks else 0.0,
                "gt_phase_rank_le2_rate": float(np.mean([rank <= 2 for rank in ranks])) if ranks else 0.0,
                "gt_phase_rank_mean": float(np.mean(ranks)) if ranks else 0.0,
                "gt_minus_top1_mean": float(np.mean(deltas)) if deltas else 0.0,
                "gt_minus_top1_positive_rate": float(np.mean([value > 0 for value in deltas])) if deltas else 0.0,
            }
        )
    return out


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


def main() -> int:
    args = parse_args()
    config = SymbolPhaseConfig(
        top_l_low_confidence=int(args.top_l_low_confidence),
        lock_margin_db=float(args.lock_margin_db),
        lock_peak_to_median_db=float(args.lock_peak_to_median_db),
        lock_phase_score=float(args.lock_phase_score),
        line_trim_frac=float(args.line_trim_frac),
    )
    out_dir = args.output_dir.resolve()
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        paths = _dataset_paths(str(dataset))
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        signal_power = float(metadata["signal_reference_power"])
        base_seed = int(metadata.get("seed", 42))
        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        packets = load_packets(paths["symbols"], None)
        rng = np.random.default_rng(base_seed)
        noise_i = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        noise_q = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)
        for snr in args.snrs:
            noise_power = signal_power * (10.0 ** (-float(snr) / 10.0))
            sigma = math.sqrt(float(noise_power) / 2.0)
            noisy = (samples + sigma * unit_noise).astype(np.complex64, copy=False)
            rows: list[dict[str, Any]] = []
            for packet_index in sorted(packets):
                rows.extend(_diagnose_packet(noisy, packets[packet_index], args, config))
            for row in rows:
                row["dataset"] = str(dataset)
                row["target_snr_db"] = float(snr)
            detail_rows.extend(rows)
            summary_rows.extend(_summarize(rows, str(dataset), float(snr)))
            print(f"{dataset} snr={snr:.1f} ambiguous={len(rows)}", flush=True)
    _write_csv(out_dir / "phase_model_detail.csv", detail_rows)
    _write_csv(out_dir / "phase_model_summary.csv", summary_rows)
    (out_dir / "phase_model_summary.json").write_text(
        json.dumps({"summary": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote={out_dir / 'phase_model_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Diagnose phase-line selectable misses for current phase_line selector."""

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
from weak_decoder.phase_line import PhasePathSelectorConfig, select_phase_viterbi_path  # noqa: E402
from weak_decoder.phase_line.selector import _coherence_score, _energy_score, _viterbi_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose selectable misses.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-24.0, -25.0])
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PHASE_LINE_DIR / "_eval" / "selectable_miss_diagnostic")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--radius", type=int, default=6)
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


def _fit_local_prediction(
    evidences: Sequence[Any],
    selected: Sequence[int],
    idx: int,
    mask: Sequence[bool] | None,
    radius: int,
    min_points: int,
) -> tuple[bool, float, float, int]:
    target_abs = float(evidences[idx].abs_symbol_index)
    points: list[tuple[float, float, float]] = []
    lo = max(0, int(idx) - int(radius))
    hi = min(len(evidences), int(idx) + int(radius) + 1)
    for j in range(lo, hi):
        if j == idx:
            continue
        if mask is not None and (j >= len(mask) or not bool(mask[j])):
            continue
        b = int(selected[j])
        ev = evidences[j]
        if b < 0 or b >= ev.center_spectrum.size:
            continue
        dist = abs(float(ev.abs_symbol_index) - target_abs)
        weight = math.exp(-0.5 * (dist / max(1e-6, 0.5 * float(radius))) ** 2)
        points.append((float(ev.abs_symbol_index), float(np.angle(ev.center_spectrum[b])), float(weight)))
    if len(points) < int(min_points):
        return False, 0.0, float("nan"), len(points)
    points.sort(key=lambda item: item[0])
    xs = np.asarray([item[0] for item in points], dtype=np.float64)
    phases = np.unwrap(np.asarray([item[1] for item in points], dtype=np.float64))
    weights = np.asarray([item[2] for item in points], dtype=np.float64)
    try:
        coef = np.polyfit(xs - target_abs, phases, deg=1, w=weights)
    except np.linalg.LinAlgError:
        return False, 0.0, float("nan"), len(points)
    fitted = np.polyval(coef, xs - target_abs)
    rmse = math.sqrt(float(np.average((phases - fitted) ** 2, weights=weights))) / math.pi
    return True, float(np.polyval(coef, 0.0)), float(rmse), len(points)


def _phase_score(ev: Any, raw_bin: int, predicted: float, scale_pi: float = 0.35) -> float:
    b = int(raw_bin)
    if b < 0 or b >= ev.center_spectrum.size:
        return 0.0
    residual = float(wrap_phase(float(np.angle(ev.center_spectrum[b])) - float(predicted)))
    return float(math.exp(-((residual / max(1e-6, float(scale_pi) * math.pi)) ** 2)))


def _rank_candidate(ev: Any, raw_bin: int, cfg: PhasePathSelectorConfig) -> int:
    for rank, cand in enumerate(_viterbi_candidates(ev, cfg), start=1):
        if int(cand.raw_bin) == int(raw_bin):
            return int(rank)
    return 0


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else 0.0


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
    rng = np.random.default_rng(int(metadata.get("seed", 42)))
    unit_noise = (
        rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        + 1j * rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    ).astype(np.complex64)
    cfg = PhasePathSelectorConfig()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for snr_db in args.snrs:
        noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
        noisy = (samples + math.sqrt(noise_power / 2.0) * unit_noise).astype(np.complex64, copy=False)
        snr_rows: list[dict[str, Any]] = []
        for packet_index in sorted(packets):
            packet = packets[packet_index]
            center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
                noisy,
                packet,
                SimpleNamespace(cfo_correction_mode=str(args.cfo_correction_mode), preamble_len=float(args.preamble_len)),
            )
            evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
            result = select_phase_viterbi_path(
                center_spectra=center_spectra,
                evidence_powers=evidence_powers,
                abs_indices=abs_indices,
                config=cfg,
                offset_coherences=coherences,
            )
            selected = tuple(int(v) for v in result.selected_raw_bins)
            hard_mask = tuple(bool(v) for v in result.locked_mask)
            for idx, (ev, sel, gt) in enumerate(zip(result.evidences, selected, gt_bins)):
                gt = int(gt)
                sel = int(sel)
                if gt < 0 or sel == gt:
                    continue
                gt_rank = _rank_candidate(ev, gt, cfg)
                if gt_rank <= 0:
                    continue
                hard_has, hard_pred, hard_rmse, hard_count = _fit_local_prediction(
                    result.evidences, selected, idx, hard_mask, int(args.radius), min_points=4
                )
                path_has, path_pred, path_rmse, path_count = _fit_local_prediction(
                    result.evidences, selected, idx, None, int(args.radius), min_points=6
                )
                row = {
                    "dataset": args.dataset,
                    "target_snr_db": float(snr_db),
                    "packet_index": int(packet_index),
                    "symbol_index": int(idx),
                    "gt_bin": int(gt),
                    "selected_bin": int(sel),
                    "gt_rank": int(gt_rank),
                    "selected_energy": _energy_score(ev, sel),
                    "gt_energy": _energy_score(ev, gt),
                    "energy_gain_gt_minus_sel": _energy_score(ev, gt) - _energy_score(ev, sel),
                    "selected_coherence": _coherence_score(ev, sel),
                    "gt_coherence": _coherence_score(ev, gt),
                    "coherence_gain_gt_minus_sel": _coherence_score(ev, gt) - _coherence_score(ev, sel),
                    "hard_has_pred": int(hard_has),
                    "hard_count": int(hard_count),
                    "hard_rmse_pi": float(hard_rmse),
                    "path_has_pred": int(path_has),
                    "path_count": int(path_count),
                    "path_rmse_pi": float(path_rmse),
                }
                if hard_has:
                    sel_phase = _phase_score(ev, sel, hard_pred)
                    gt_phase = _phase_score(ev, gt, hard_pred)
                    row["hard_selected_phase"] = sel_phase
                    row["hard_gt_phase"] = gt_phase
                    row["hard_phase_gain_gt_minus_sel"] = gt_phase - sel_phase
                if path_has:
                    sel_phase = _phase_score(ev, sel, path_pred)
                    gt_phase = _phase_score(ev, gt, path_pred)
                    row["path_selected_phase"] = sel_phase
                    row["path_gt_phase"] = gt_phase
                    row["path_phase_gain_gt_minus_sel"] = gt_phase - sel_phase
                rows.append(row)
                snr_rows.append(row)
        summary = {
            "dataset": args.dataset,
            "target_snr_db": float(snr_db),
            "selectable_misses": int(len(snr_rows)),
            "mean_gt_rank": _avg(snr_rows, "gt_rank"),
            "mean_energy_gain_gt_minus_sel": _avg(snr_rows, "energy_gain_gt_minus_sel"),
            "mean_coherence_gain_gt_minus_sel": _avg(snr_rows, "coherence_gain_gt_minus_sel"),
            "hard_pred_rate": float(sum(int(r.get("hard_has_pred", 0)) for r in snr_rows) / max(1, len(snr_rows))),
            "path_pred_rate": float(sum(int(r.get("path_has_pred", 0)) for r in snr_rows) / max(1, len(snr_rows))),
            "mean_hard_phase_gain_gt_minus_sel": _avg(snr_rows, "hard_phase_gain_gt_minus_sel"),
            "mean_path_phase_gain_gt_minus_sel": _avg(snr_rows, "path_phase_gain_gt_minus_sel"),
            "hard_phase_gt_better_rate": float(
                sum(1 for r in snr_rows if float(r.get("hard_phase_gain_gt_minus_sel", -1.0)) > 0.0)
                / max(1, sum(int(r.get("hard_has_pred", 0)) for r in snr_rows))
            ),
            "path_phase_gt_better_rate": float(
                sum(1 for r in snr_rows if float(r.get("path_phase_gain_gt_minus_sel", -1.0)) > 0.0)
                / max(1, sum(int(r.get("path_has_pred", 0)) for r in snr_rows))
            ),
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    out_dir = Path(args.output_dir)
    _write_csv(out_dir / "selectable_miss_rows.csv", rows)
    _write_csv(out_dir / "selectable_miss_summary.csv", summaries)
    (out_dir / "selectable_miss_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={out_dir / 'selectable_miss_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

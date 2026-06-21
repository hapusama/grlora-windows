"""Diagnose whether the selected path can rescue missing GT candidates.

This script is intentionally diagnostic-only.  It asks whether a local phase
prediction fitted from the current selected path would rank missing GT bins
highly inside a wider energy preselection set.
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
WEAK_ROOT = THIS.parents[3]
EXPERIMENT_DIR = WEAK_ROOT / "scripts" / "experiments"
PHASE_EXPERIMENT_DIR = EXPERIMENT_DIR / "phase_line"
for path in (str(WEAK_ROOT), str(EXPERIMENT_DIR), str(PHASE_EXPERIMENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_phase_line_threshold_sweep import _dataset_paths, _load_metadata  # noqa: E402
from run_symbol_phase_two_stage import _extract_payload_spectra_with_coherence, _ser  # noqa: E402
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.candidate_pruning import top_bins, wrap_phase  # noqa: E402
from weak_decoder.phase_line import PhasePathSelectorConfig, select_phase_viterbi_path  # noqa: E402
from weak_decoder.phase_line.selector import _viterbi_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose selected-path phase rescue potential.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-24.0, -25.0])
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--preselect", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--min-anchors", type=int, default=4)
    parser.add_argument("--phase-scale-pi", type=float, default=0.35)
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=PHASE_LINE_DIR / "_eval" / "path_rescue_potential")
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


def _candidate_rank(values: Sequence[int], target: int) -> int:
    for idx, value in enumerate(values, start=1):
        if int(value) == int(target):
            return int(idx)
    return 0


def _local_path_prediction(
    center_spectra: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    selected: Sequence[int],
    target_idx: int,
    radius: int,
    min_anchors: int,
) -> tuple[bool, float, float]:
    idx = int(target_idx)
    lo = max(0, idx - max(1, int(radius)))
    hi = min(len(center_spectra), idx + max(1, int(radius)) + 1)
    rows: list[tuple[float, float, float]] = []
    for j in range(lo, hi):
        if j == idx or j >= len(selected) or j >= len(abs_indices):
            continue
        b = int(selected[j])
        spec = np.asarray(center_spectra[j])
        if b < 0 or b >= spec.size:
            continue
        distance = abs(j - idx)
        weight = math.exp(-0.5 * (float(distance) / max(1e-6, 0.5 * float(radius))) ** 2)
        rows.append((float(abs_indices[j]), float(np.angle(spec[b])), float(weight)))
    if len(rows) < max(2, int(min_anchors)):
        return False, 0.0, float("nan")
    rows.sort(key=lambda item: item[0])
    xs = np.asarray([item[0] for item in rows], dtype=np.float64)
    phases = np.unwrap(np.asarray([item[1] for item in rows], dtype=np.float64))
    weights = np.asarray([item[2] for item in rows], dtype=np.float64)
    target_abs = float(abs_indices[idx])
    try:
        coef = np.polyfit(xs - target_abs, phases, deg=1, w=weights)
    except np.linalg.LinAlgError:
        return False, 0.0, float("nan")
    fitted = np.polyval(coef, xs - target_abs)
    rmse_pi = float(math.sqrt(float(np.average((phases - fitted) ** 2, weights=weights))) / math.pi)
    return True, float(np.polyval(coef, 0.0)), float(rmse_pi)


def _score_bins(
    spectrum: np.ndarray,
    power: np.ndarray,
    coherence: np.ndarray | None,
    bins: Sequence[int],
    predicted: float,
    phase_scale_pi: float,
) -> list[tuple[float, int, float, float, float]]:
    phase_scale = max(1e-6, float(phase_scale_pi) * math.pi)
    max_power = float(np.max(power)) if power.size else 0.0
    out: list[tuple[float, int, float, float, float]] = []
    for raw_bin in bins:
        b = int(raw_bin)
        if b < 0 or b >= spectrum.size or b >= power.size:
            continue
        residual = float(wrap_phase(float(np.angle(spectrum[b])) - float(predicted)))
        phase_score = float(math.exp(-((residual / phase_scale) ** 2)))
        energy_score = float(power[b] / (max_power + 1e-30)) if max_power > 0.0 else 0.0
        coherence_score = 0.0
        if coherence is not None and b < coherence.size:
            coherence_score = float(max(0.0, min(1.0, coherence[b])))
        score = 0.55 * phase_score + 0.25 * energy_score + 0.20 * coherence_score
        out.append((float(score), b, float(phase_score), float(energy_score), float(coherence_score)))
    out.sort(key=lambda item: item[0], reverse=True)
    return out


def main() -> int:
    args = parse_args()
    paths = _dataset_paths(str(args.dataset))
    metadata = _load_metadata(paths, str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    packets = load_packets(paths["symbols"], args.packet)
    signal_power = float(metadata.get("signal_reference_power", np.mean(np.abs(samples).astype(np.float64) ** 2)))
    rng = np.random.default_rng(int(metadata.get("seed", 42)))
    unit_noise = (
        rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        + 1j * rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    ).astype(np.complex64)

    cfg = PhasePathSelectorConfig()
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

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
            _raw, current_ser, _compared = _ser(result.selected_raw_bins, gt_bins, sf=int(packet["sf"]), ldro=bool(packet["ldro"]))
            for idx, ev in enumerate(result.evidences):
                if idx >= len(gt_bins):
                    continue
                gt = int(gt_bins[idx])
                if gt < 0:
                    continue
                actual_bins = [int(c.raw_bin) for c in _viterbi_candidates(ev, cfg)]
                if gt in set(actual_bins):
                    continue
                has_pred, predicted, pred_rmse_pi = _local_path_prediction(
                    center_spectra,
                    abs_indices,
                    result.selected_raw_bins,
                    idx,
                    radius=int(args.radius),
                    min_anchors=int(args.min_anchors),
                )
                row: dict[str, Any] = {
                    "dataset": str(args.dataset),
                    "target_snr_db": float(snr_db),
                    "packet_index": int(packet_index),
                    "current_ser": float(current_ser),
                    "payload_symbol_index": int(idx),
                    "gt_bin": int(gt),
                    "selected_bin": int(result.selected_raw_bins[idx]) if idx < len(result.selected_raw_bins) else -1,
                    "has_path_pred": int(has_pred),
                    "path_pred_rmse_pi": float(pred_rmse_pi),
                    "selected_hit": int(idx < len(result.selected_raw_bins) and int(result.selected_raw_bins[idx]) == gt),
                }
                power = np.asarray(evidence_powers[idx], dtype=np.float64)
                for preselect in args.preselect:
                    bins = [int(v) for v in top_bins(power, min(int(preselect), power.size))]
                    row[f"gt_energy_rank_{preselect}"] = _candidate_rank(bins, gt)
                    if has_pred:
                        scored = _score_bins(
                            np.asarray(center_spectra[idx]),
                            power,
                            np.asarray(coherences[idx]) if idx < len(coherences) else None,
                            bins,
                            predicted,
                            float(args.phase_scale_pi),
                        )
                        ranked = [int(item[1]) for item in scored]
                        row[f"gt_path_rank_{preselect}"] = _candidate_rank(ranked, gt)
                        row[f"sel_path_rank_{preselect}"] = _candidate_rank(
                            ranked,
                            int(result.selected_raw_bins[idx]) if idx < len(result.selected_raw_bins) else -1,
                        )
                    else:
                        row[f"gt_path_rank_{preselect}"] = 0
                        row[f"sel_path_rank_{preselect}"] = 0
                detail_rows.append(row)
                snr_rows.append(row)

        summary: dict[str, Any] = {
            "dataset": str(args.dataset),
            "target_snr_db": float(snr_db),
            "missing_count": int(len(snr_rows)),
            "path_pred_rate": float(sum(int(row["has_path_pred"]) for row in snr_rows) / max(1, len(snr_rows))),
        }
        for preselect in args.preselect:
            key = f"gt_path_rank_{preselect}"
            present = [row for row in snr_rows if int(row.get(key, 0)) > 0]
            le4 = [row for row in present if int(row[key]) <= 4]
            le8 = [row for row in present if int(row[key]) <= 8]
            summary[f"gt_energy_present_{preselect}"] = float(
                sum(1 for row in snr_rows if int(row.get(f"gt_energy_rank_{preselect}", 0)) > 0) / max(1, len(snr_rows))
            )
            summary[f"gt_path_present_{preselect}"] = float(len(present) / max(1, len(snr_rows)))
            summary[f"gt_path_rank_le4_{preselect}"] = float(len(le4) / max(1, len(snr_rows)))
            summary[f"gt_path_rank_le8_{preselect}"] = float(len(le8) / max(1, len(snr_rows)))
            summary[f"mean_gt_path_rank_{preselect}"] = float(np.mean([int(row[key]) for row in present])) if present else 0.0
        summary_rows.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    out_dir = Path(args.output_dir)
    _write_csv(out_dir / "path_rescue_rows.csv", detail_rows)
    _write_csv(out_dir / "path_rescue_summary.csv", summary_rows)
    (out_dir / "path_rescue_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={out_dir / 'path_rescue_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

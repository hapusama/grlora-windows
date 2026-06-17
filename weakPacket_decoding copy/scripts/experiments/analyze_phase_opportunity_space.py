#!/usr/bin/env python3
"""Analyze how much Top-L opportunity remains for phase-aware selection.

This script does not change the decoder.  It reuses the existing header-first
timing/GT rows, synthetic AWGN setup, multi-offset FFT evidence, and the current
symbol-level phase/coherence selector, then measures the candidate-stage ceiling:

* already_correct: multi-offset Top-1 is already the GT bin.
* repairable: multi-offset Top-1 is wrong, but GT is still inside energy Top-L.
* unrecoverable: multi-offset Top-1 is wrong, and energy Top-L misses GT.

CRC, payload byte priors, templates, and cross-packet information are not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from run_symbol_phase_two_stage import (  # noqa: E402
    _extract_payload_spectra_with_coherence,
    _packet_phase_line,
)
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.candidate_pruning import rank_of, top_bins  # noqa: E402
from weak_decoder.symbol_phase_two_stage import (  # noqa: E402
    SymbolPhaseConfig,
    select_symbol_bins_two_stage,
)


DEFAULT_DATASETS = (
    "0_0_0_10_14_8",
    "0_0_0_10_14_16",
    "0_0_0_10_14_32",
)
DEFAULT_TOP_L = (8, 16, 24, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Top-L repairable space for phase/coherence selection."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--top-l-values", type=int, nargs="+", default=list(DEFAULT_TOP_L))
    parser.add_argument("--snr-start", type=float, default=-20.0)
    parser.add_argument("--snr-stop", type=float, default=-25.0)
    parser.add_argument("--snr-step", type=float, default=-1.0)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "phase_opportunity_space")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--independent-noise", action="store_true")

    parser.add_argument("--selection-mode", choices=("coherence", "smooth", "override"), default="coherence")
    parser.add_argument("--lock-margin-db", type=float, default=1.5)
    parser.add_argument("--lock-peak-to-median-db", type=float, default=5.0)
    parser.add_argument("--lock-phase-score", type=float, default=0.35)
    parser.add_argument("--lock-min-coherence", type=float, default=0.0)
    parser.add_argument("--min-locked-for-line", type=int, default=4)
    parser.add_argument("--line-trim-frac", type=float, default=0.25)
    parser.add_argument("--phase-model", choices=("linear", "quadratic"), default="linear")
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--coherence-candidate-top-l", type=int, default=0)

    parser.add_argument("--smooth-phase-weight", type=float, default=0.05)
    parser.add_argument("--smooth-amp-weight", type=float, default=0.50)
    parser.add_argument("--smooth-coherence-weight", type=float, default=0.90)
    parser.add_argument("--smooth-slope-penalty", type=float, default=0.05)
    parser.add_argument("--smooth-curvature-penalty", type=float, default=0.10)
    parser.add_argument("--smooth-max-energy-drop-db", type=float, default=20.0)
    parser.add_argument("--smooth-min-line-anchors", type=int, default=4)
    parser.add_argument("--smooth-min-locked-ratio", type=float, default=0.0)
    parser.add_argument("--smooth-max-line-rmse-pi", type=float, default=float("inf"))

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
    return parser.parse_args()


def _snr_values(start: float, stop: float, step: float) -> list[float]:
    if abs(float(step)) <= 1e-12:
        raise ValueError("snr-step must be nonzero")
    values: list[float] = []
    current = float(start)
    if step < 0:
        while current >= float(stop) - 1e-9:
            values.append(round(current, 6))
            current += float(step)
    else:
        while current <= float(stop) + 1e-9:
            values.append(round(current, 6))
            current += float(step)
    return values


def _dataset_paths(dataset: str) -> dict[str, Path]:
    base = WEAK_ROOT / "data" / "low_snr_gt_bin" / dataset
    return {
        "iq": WEAK_ROOT.parent / "data" / "USRP_IQ" / f"{dataset}.bin",
        "symbols": WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / f"{dataset}_header_first_symbols.csv",
        "metadata": base / f"{dataset}_low_snr_gt_bin_metadata.json",
    }


def _config_for_l(args: argparse.Namespace, top_l: int) -> SymbolPhaseConfig:
    return SymbolPhaseConfig(
        top_l_low_confidence=int(top_l),
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
    )


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(10.0 * math.log10((float(numerator) + 1e-30) / (float(denominator) + 1e-30)))


def _mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else 0.0


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if float(den) > 0 else 0.0


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
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


def _row_for_symbol(
    *,
    dataset: str,
    snr_db: float,
    packet: dict[str, Any],
    symbol_index: int,
    top_l: int,
    evidence_power: np.ndarray,
    coherence: np.ndarray | None,
    multi_top1: int,
    selected_bin: int,
    gt_bin: int,
    locked: bool,
    top1_margin_db: float,
    top1_peak_to_median_db: float,
) -> dict[str, Any]:
    candidates = tuple(int(v) for v in top_bins(evidence_power, int(top_l)))
    in_top_l = int(gt_bin in candidates)
    multi_correct = int(multi_top1 == gt_bin)
    selected_correct = int(selected_bin == gt_bin)
    if multi_correct:
        category = "already_correct"
    elif in_top_l:
        category = "repairable"
    else:
        category = "unrecoverable"

    max_power = float(np.max(evidence_power)) if evidence_power.size else 0.0
    gt_power = float(evidence_power[gt_bin]) if 0 <= gt_bin < evidence_power.size else 0.0
    selected_power = float(evidence_power[selected_bin]) if 0 <= selected_bin < evidence_power.size else 0.0
    gt_rank = rank_of(evidence_power, gt_bin)
    selected_rank = rank_of(evidence_power, selected_bin)
    top1_power = float(evidence_power[multi_top1]) if 0 <= multi_top1 < evidence_power.size else 0.0
    gt_coherence = ""
    selected_coherence = ""
    top1_coherence = ""
    if coherence is not None and coherence.size == evidence_power.size:
        gt_coherence = float(coherence[gt_bin]) if 0 <= gt_bin < coherence.size else ""
        selected_coherence = float(coherence[selected_bin]) if 0 <= selected_bin < coherence.size else ""
        top1_coherence = float(coherence[multi_top1]) if 0 <= multi_top1 < coherence.size else ""

    return {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "top_l": int(top_l),
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "payload_symbol_index": int(symbol_index),
        "gt_bin": int(gt_bin),
        "multi_top1_bin": int(multi_top1),
        "selected_bin": int(selected_bin),
        "multi_correct": int(multi_correct),
        "selected_correct": int(selected_correct),
        "selected_changed_top1": int(selected_bin != multi_top1),
        "gt_in_top_l": int(in_top_l),
        "category": category,
        "is_already_correct": int(category == "already_correct"),
        "is_repairable": int(category == "repairable"),
        "is_unrecoverable": int(category == "unrecoverable"),
        "phase_repaired": int(category == "repairable" and selected_correct),
        "phase_damaged": int(category == "already_correct" and not selected_correct),
        "locked_by_selector": int(bool(locked)),
        "gt_energy_rank": int(gt_rank),
        "selected_energy_rank": int(selected_rank),
        "gt_energy_drop_db": _db_ratio(gt_power, top1_power),
        "selected_energy_drop_db": _db_ratio(selected_power, top1_power),
        "gt_within_energy_drop_gate": int(_db_ratio(gt_power, top1_power) >= -20.0),
        "top1_margin_db": float(top1_margin_db),
        "top1_peak_to_median_db": float(top1_peak_to_median_db),
        "top1_offset_coherence": top1_coherence,
        "gt_offset_coherence": gt_coherence,
        "selected_offset_coherence": selected_coherence,
        "top_l_candidates": " ".join(str(v) for v in candidates),
        "multi_top1_power": float(top1_power),
        "gt_power": float(gt_power),
        "selected_power": float(selected_power),
        "max_power": float(max_power),
    }


def _summarize_group(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float, top_l: int) -> dict[str, Any]:
    total = len(rows)
    already = sum(int(row["is_already_correct"]) for row in rows)
    repairable = sum(int(row["is_repairable"]) for row in rows)
    unrecoverable = sum(int(row["is_unrecoverable"]) for row in rows)
    selected_errors = sum(1 - int(row["selected_correct"]) for row in rows)
    multi_errors = total - already
    selected_repairs = sum(int(row["phase_repaired"]) for row in rows)
    selected_damage = sum(int(row["phase_damaged"]) for row in rows)
    selected_changed = sum(int(row["selected_changed_top1"]) for row in rows)
    repairable_selected_errors = sum(
        1 - int(row["selected_correct"]) for row in rows if row["category"] == "repairable"
    )
    already_selected_errors = sum(
        1 - int(row["selected_correct"]) for row in rows if row["category"] == "already_correct"
    )

    return {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "top_l": int(top_l),
        "symbol_count": int(total),
        "multi_top1_correct_count": int(already),
        "multi_top1_error_count": int(multi_errors),
        "already_correct_count": int(already),
        "repairable_count": int(repairable),
        "unrecoverable_count": int(unrecoverable),
        "selected_error_count": int(selected_errors),
        "selected_repair_count": int(selected_repairs),
        "selected_damage_count": int(selected_damage),
        "selected_changed_top1_count": int(selected_changed),
        "multi_top1_accuracy": _safe_rate(already, total),
        "multi_top1_SER": _safe_rate(multi_errors, total),
        "gt_recall@L": _safe_rate(already + repairable, total),
        "gt_recall_at_L": _safe_rate(already + repairable, total),
        "repairable_error_rate": _safe_rate(repairable, total),
        "unrecoverable_error_rate": _safe_rate(unrecoverable, total),
        "repairable_fraction_of_multi_errors": _safe_rate(repairable, multi_errors),
        "unrecoverable_fraction_of_multi_errors": _safe_rate(unrecoverable, multi_errors),
        "oracle_topL_SER": _safe_rate(unrecoverable, total),
        "selected_SER": _safe_rate(selected_errors, total),
        "selected_accuracy": 1.0 - _safe_rate(selected_errors, total),
        "phase_repair_rate": _safe_rate(selected_repairs, repairable),
        "phase_damage_rate": _safe_rate(selected_damage, already),
        "selected_SER_on_repairable": _safe_rate(repairable_selected_errors, repairable),
        "selected_SER_on_already_correct": _safe_rate(already_selected_errors, already),
        "selected_vs_multi_gain": _safe_rate(multi_errors, total) - _safe_rate(selected_errors, total),
        "selected_vs_multi_gain_on_repairable": _safe_rate(selected_repairs, repairable),
        "net_selected_gain_count": int(selected_repairs - selected_damage),
        "net_selected_gain_rate": _safe_rate(selected_repairs - selected_damage, total),
        "selected_changed_top1_rate": _safe_rate(selected_changed, total),
        "mean_gt_energy_rank": _mean([float(row["gt_energy_rank"]) for row in rows]),
        "mean_gt_energy_drop_db": _mean([float(row["gt_energy_drop_db"]) for row in rows]),
        "mean_selected_energy_rank": _mean([float(row["selected_energy_rank"]) for row in rows]),
        "mean_top1_margin_db": _mean([float(row["top1_margin_db"]) for row in rows]),
        "mean_top1_peak_to_median_db": _mean([float(row["top1_peak_to_median_db"]) for row in rows]),
    }


def _mean_of_dataset_rows(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_key: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        if row["dataset"] == "mean_of_datasets":
            continue
        by_key[(float(row["target_snr_db"]), int(row["top_l"]))].append(row)

    count_fields = {
        "symbol_count",
        "multi_top1_correct_count",
        "multi_top1_error_count",
        "already_correct_count",
        "repairable_count",
        "unrecoverable_count",
        "selected_error_count",
        "selected_repair_count",
        "selected_damage_count",
        "selected_changed_top1_count",
        "net_selected_gain_count",
    }
    for (snr_db, top_l), rows in sorted(by_key.items(), key=lambda item: (item[0][0], item[0][1]), reverse=True):
        rec: dict[str, Any] = {
            "dataset": "mean_of_datasets",
            "target_snr_db": float(snr_db),
            "top_l": int(top_l),
        }
        for key in rows[0]:
            if key in rec or key in ("dataset", "target_snr_db", "top_l"):
                continue
            if key in count_fields:
                rec[key] = int(sum(int(row[key]) for row in rows))
            else:
                rec[key] = float(np.mean([float(row[key]) for row in rows]))
        out.append(rec)
    return out


def main() -> int:
    args = parse_args()
    snrs = _snr_values(args.snr_start, args.snr_stop, args.snr_step)
    top_l_values = sorted({int(v) for v in args.top_l_values})
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    configs = {int(top_l): _config_for_l(args, int(top_l)) for top_l in top_l_values}

    for dataset in args.datasets:
        paths = _dataset_paths(str(dataset))
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        signal_power = float(metadata["signal_reference_power"])
        base_seed = int(metadata.get("seed", 42))
        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        if samples.size == 0:
            raise ValueError(f"empty IQ file: {paths['iq']}")
        packets = load_packets(paths["symbols"], None)
        if not packets:
            raise ValueError(f"no packets loaded: {paths['symbols']}")

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

            group_rows: dict[int, list[dict[str, Any]]] = {top_l: [] for top_l in top_l_values}
            for packet_index in sorted(packets):
                packet = packets[packet_index]
                center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
                    noisy, packet, args
                )
                if not multi_spectra:
                    continue
                evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
                multi_bins = tuple(int(np.argmax(power)) for power in evidence_powers)
                fallback_line = _packet_phase_line(noisy, packet, args)

                for top_l in top_l_values:
                    result = select_symbol_bins_two_stage(
                        center_spectra=center_spectra,
                        evidence_powers=evidence_powers,
                        abs_indices=abs_indices,
                        config=configs[top_l],
                        fallback_line=fallback_line,
                        offset_coherences=coherences,
                    )
                    selected_bins = tuple(int(v) for v in result.selected_raw_bins)
                    limit = min(len(evidence_powers), len(multi_bins), len(selected_bins), len(gt_bins))
                    for idx in range(limit):
                        gt_bin = int(gt_bins[idx])
                        if gt_bin < 0:
                            continue
                        ev = result.evidences[idx] if idx < len(result.evidences) else None
                        row = _row_for_symbol(
                            dataset=str(dataset),
                            snr_db=float(snr_db),
                            packet=packet,
                            symbol_index=idx,
                            top_l=int(top_l),
                            evidence_power=evidence_powers[idx],
                            coherence=coherences[idx] if idx < len(coherences) else None,
                            multi_top1=int(multi_bins[idx]),
                            selected_bin=int(selected_bins[idx]),
                            gt_bin=gt_bin,
                            locked=bool(result.locked_mask[idx]) if idx < len(result.locked_mask) else False,
                            top1_margin_db=float(ev.top1_margin_db) if ev is not None else 0.0,
                            top1_peak_to_median_db=float(ev.top1_peak_to_median_db) if ev is not None else 0.0,
                        )
                        detail_rows.append(row)
                        group_rows[top_l].append(row)

            for top_l in top_l_values:
                summary = _summarize_group(group_rows[top_l], str(dataset), float(snr_db), int(top_l))
                summary_rows.append(summary)
                print(
                    f"{dataset} snr={snr_db:>6.1f} L={top_l:>2} "
                    f"multi_acc={summary['multi_top1_accuracy']:.3f} "
                    f"repairable={summary['repairable_error_rate']:.3f} "
                    f"unrecoverable={summary['unrecoverable_error_rate']:.3f} "
                    f"repair_rate={summary['phase_repair_rate']:.3f} "
                    f"damage={summary['phase_damage_rate']:.3f} "
                    f"selected_ser={summary['selected_SER']:.3f}",
                    flush=True,
                )

    summary_rows.extend(_mean_of_dataset_rows(summary_rows))

    detail_path = out_dir / "phase_opportunity_detail.csv"
    summary_path = out_dir / "phase_opportunity_summary.csv"
    json_path = out_dir / "phase_opportunity_summary.json"
    _write_csv(detail_path, detail_rows)
    _write_csv(summary_path, summary_rows)
    json_path.write_text(
        json.dumps(
            {
                "datasets": list(args.datasets),
                "snr_values": snrs,
                "top_l_values": top_l_values,
                "definition": {
                    "already_correct": "multi_top1 == gt_bin",
                    "repairable": "multi_top1 != gt_bin and gt_bin in energy Top-L",
                    "unrecoverable": "multi_top1 != gt_bin and gt_bin not in energy Top-L",
                    "oracle_topL_SER": "unrecoverable_count / symbol_count",
                    "phase_repair_rate": "P(selected_bin == gt_bin | repairable)",
                    "phase_damage_rate": "P(selected_bin != gt_bin | already_correct)",
                },
                "summary_rows": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote={detail_path}")
    print(f"wrote={summary_path}")
    print(f"wrote={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

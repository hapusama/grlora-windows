#!/usr/bin/env python3
"""Diagnose whether GT bins are present in phase-DP candidate sets."""

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

from run_symbol_phase_threshold_sweep import _avg, _dataset_paths, _snr_values, _write_csv  # noqa: E402
from run_symbol_phase_two_stage import (  # noqa: E402
    _argmax_bins,
    _extract_payload_spectra_with_coherence,
    _packet_phase_line,
    _ser,
)
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.phase_line import (  # noqa: E402
    PhaseLineSelectorConfig,
    PhasePathSelectorConfig,
    select_phase_smooth_path,
    select_phase_viterbi_path,
)
from weak_decoder.phase_line.selector import _viterbi_candidates  # noqa: E402
from weak_decoder.symbol_phase_two_stage import SymbolPhaseConfig, select_symbol_bins_two_stage  # noqa: E402


METHODS = (
    "multi",
    "v3",
    "phase_line",
    "phase_dp_first",
    "phase_dp_second",
    "phase_dp_second_anchor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose GT candidate coverage for phase Viterbi selectors.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snr-start", type=float, default=-22.0)
    parser.add_argument("--snr-stop", type=float, default=-26.0)
    parser.add_argument("--snr-step", type=float, default=-1.0)
    parser.add_argument("--packet", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "phase_dp_candidate_coverage")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol"), default="continuous")
    parser.add_argument("--preamble-len", type=float, default=8.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--top-l", type=int, default=24)
    parser.add_argument("--dp-max-energy-drop-db", type=float, default=24.0)
    parser.add_argument("--dp-energy-weight", type=float, default=0.25)
    parser.add_argument("--dp-coherence-weight", type=float, default=0.40)
    parser.add_argument("--dp-rank-weight", type=float, default=0.05)
    parser.add_argument("--first-lambda", type=float, default=0.18)
    parser.add_argument("--first-scale-pi", type=float, default=0.75)
    parser.add_argument("--second-lambda", type=float, default=0.05)
    parser.add_argument("--second-first-lambda", type=float, default=0.0)
    parser.add_argument("--second-scale-pi", type=float, default=0.35)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--anchor-bonus", type=float, default=0.05)
    parser.add_argument("--anchor-top-k", type=int, default=0)
    parser.add_argument("--high-conf-margin-db", type=float, default=3.0)
    parser.add_argument("--high-conf-peak-db", type=float, default=7.0)
    parser.add_argument("--high-conf-min-coherence", type=float, default=0.0)
    return parser.parse_args()


def _load_metadata(paths: dict[str, Path], dataset: str) -> dict[str, Any]:
    if paths["metadata"].exists():
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))
    low_snr_root = WEAK_ROOT / "data" / "low_snr_gt_bin"
    for path in sorted(low_snr_root.glob(f"{dataset}*/*_metadata.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "signal_reference_power" in metadata:
            print(f"{dataset}: using metadata fallback {path}", flush=True)
            return metadata
    return {}


def _build_v3_config(top_l: int) -> SymbolPhaseConfig:
    return SymbolPhaseConfig(
        top_l_low_confidence=int(top_l),
        lock_margin_db=1.5,
        lock_peak_to_median_db=5.0,
        lock_phase_score=0.35,
        min_locked_for_line=4,
        line_trim_frac=0.25,
        phase_model="linear",
        selection_mode="coherence",
        beam_width=128,
        trajectory_rmse_scale_pi=0.30,
        phase_weight=0.20,
        line_weight=0.00,
        amp_weight=0.80,
        profile_weight=0.00,
        phase_override_min_gain=0.15,
        phase_override_max_drop_db=0.60,
        phase_override_score_margin=0.06,
        phase_override_min_line_anchors=8,
        phase_override_max_line_rmse_pi=0.25,
        coherence_weight=0.0,
        coherence_candidate_top_l=0,
        lock_min_coherence=0.0,
        smooth_phase_weight=0.05,
        smooth_amp_weight=0.50,
        smooth_coherence_weight=0.90,
        smooth_slope_penalty=0.05,
        smooth_curvature_penalty=0.10,
        smooth_max_energy_drop_db=20.0,
        smooth_min_line_anchors=4,
        smooth_min_locked_ratio=0.0,
        smooth_max_line_rmse_pi=float("inf"),
        window_size=5,
        window_degree=1,
        window_phase_weight=0.05,
        window_amp_weight=0.50,
        window_coherence_weight=0.90,
        window_slope_weight=0.00,
        window_curvature_weight=0.00,
        window_phase_scale_pi=0.25,
        window_slope_scale_pi=0.45,
        window_curvature_scale_pi=0.25,
        window_recent_decay=0.75,
        window_anchor_span=8.0,
        window_anchor_min=2,
        window_anchor_max_rmse_pi=0.40,
        window_min_locked_ratio=0.10,
        window_guard_min_phase_gain=0.10,
        window_guard_max_energy_drop_db=0.75,
        window_guard_max_coherence_drop=0.08,
    )


def _phase_line_config(top_l: int) -> PhaseLineSelectorConfig:
    return PhaseLineSelectorConfig(
        top_l=int(top_l),
        coherence_weight=0.03,
        phase_weight_low_conf=0.82,
        phase_weight_high_conf=0.38,
        energy_weight_low_conf=0.15,
        energy_weight_high_conf=0.57,
        phase_scale_pi=0.28,
        max_energy_drop_db_low_conf=18.0,
        max_energy_drop_db_high_conf=3.0,
    )


def _base_path_config(args: argparse.Namespace, phase_order: int) -> PhasePathSelectorConfig:
    return PhasePathSelectorConfig(
        top_l=int(args.top_l),
        phase_order=int(phase_order),
        energy_weight=float(args.dp_energy_weight),
        coherence_weight=float(args.dp_coherence_weight),
        rank_weight=float(args.dp_rank_weight),
        first_order_weight=float(args.first_lambda if phase_order == 1 else args.second_first_lambda),
        second_order_weight=float(args.second_lambda if phase_order >= 2 else 0.0),
        first_order_scale_pi=float(args.first_scale_pi),
        second_order_scale_pi=float(args.second_scale_pi),
        huber_delta=float(args.huber_delta),
        max_energy_drop_db=float(args.dp_max_energy_drop_db),
        high_confidence_margin_db=float(args.high_conf_margin_db),
        high_confidence_peak_to_median_db=float(args.high_conf_peak_db),
        high_confidence_min_coherence=float(args.high_conf_min_coherence),
        top1_soft_bonus=0.0,
        high_confidence_top_k=0,
    )


def _anchor_path_config(args: argparse.Namespace) -> PhasePathSelectorConfig:
    base = _base_path_config(args, phase_order=2)
    return PhasePathSelectorConfig(
        **{
            **base.__dict__,
            "top1_soft_bonus": float(args.anchor_bonus),
            "high_confidence_top_k": int(args.anchor_top_k),
        }
    )


def _rank_in(values: Sequence[int], target: int) -> int:
    for idx, value in enumerate(values):
        if int(value) == int(target):
            return int(idx + 1)
    return 0


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    v3_config: SymbolPhaseConfig,
    first_config: PhasePathSelectorConfig,
    second_config: PhasePathSelectorConfig,
    anchor_config: PhasePathSelectorConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    center_spectra, multi_spectra, abs_indices, gt_bins, coherences = _extract_payload_spectra_with_coherence(
        samples,
        packet,
        args,
    )
    evidence_powers = [np.abs(spec).astype(np.float64) ** 2 for spec in multi_spectra]
    header_line = _packet_phase_line(samples, packet, args)
    v3 = select_symbol_bins_two_stage(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=v3_config,
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    phase_line = select_phase_smooth_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=_phase_line_config(args.top_l),
        fallback_line=header_line,
        offset_coherences=coherences,
    )
    first = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=first_config,
        offset_coherences=coherences,
    )
    second = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=second_config,
        offset_coherences=coherences,
    )
    anchor = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=anchor_config,
        offset_coherences=coherences,
    )
    multi_bins = _argmax_bins(multi_spectra)
    methods: dict[str, Sequence[int]] = {
        "multi": multi_bins,
        "v3": tuple(v3.selected_raw_bins),
        "phase_line": tuple(phase_line.selected_raw_bins),
        "phase_dp_first": tuple(first.selected_raw_bins),
        "phase_dp_second": tuple(second.selected_raw_bins),
        "phase_dp_second_anchor": tuple(anchor.selected_raw_bins),
    }

    rows: list[dict[str, Any]] = []
    compared = 0
    gt_top_l_hits = 0
    gt_actual_hits = 0
    for idx, ev in enumerate(first.evidences):
        if idx >= len(gt_bins):
            continue
        gt = int(gt_bins[idx])
        if gt < 0:
            continue
        compared += 1
        top_l_bins = [int(v) for v in ev.top_bins[: int(args.top_l)]]
        actual_bins = [int(c.raw_bin) for c in _viterbi_candidates(ev, first_config)]
        gt_in_top_l = int(gt in set(top_l_bins))
        gt_in_actual = int(gt in set(actual_bins))
        gt_top_l_hits += gt_in_top_l
        gt_actual_hits += gt_in_actual
        row: dict[str, Any] = {
            "packet_index": int(packet["packet_index"]),
            "payload_symbol_index": int(idx),
            "gt_bin": int(gt),
            "top1_bin": int(ev.top1_bin),
            "gt_in_stage1_top_l": gt_in_top_l,
            "gt_stage1_rank": _rank_in(top_l_bins, gt),
            "gt_in_viterbi_candidates": gt_in_actual,
            "gt_viterbi_rank": _rank_in(actual_bins, gt),
            "top1_margin_db": float(ev.top1_margin_db),
            "top1_peak_to_median_db": float(ev.top1_peak_to_median_db),
            "top1_coherence": float(ev.top1_coherence_score),
        }
        for method, bins in methods.items():
            pred = int(bins[idx]) if idx < len(bins) else -1
            row[f"{method}_bin"] = pred
            row[f"{method}_hit"] = int(pred == gt)
            row[f"{method}_error_type"] = (
                "hit"
                if pred == gt
                else ("selectable_miss" if gt_in_actual else ("top_l_present_but_filtered" if gt_in_top_l else "gt_missing_top_l"))
            )
        rows.append(row)

    packet_summary: dict[str, Any] = {
        "packet_index": int(packet["packet_index"]),
        "payload_len": int(packet["payload_len"]),
        "symbol_count": int(compared),
        "gt_stage1_top_l_recall": float(gt_top_l_hits / max(1, compared)),
        "gt_viterbi_candidate_recall": float(gt_actual_hits / max(1, compared)),
    }
    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    for method, bins in methods.items():
        raw_ser, symbol_ser, _ = _ser(bins, gt_bins, sf=sf, ldro=ldro)
        packet_summary[f"{method}_raw_ser"] = float(raw_ser)
        packet_summary[f"{method}_symbol_ser"] = float(symbol_ser)
        method_rows = [row for row in rows if int(row.get(f"{method}_hit", 0)) == 0]
        packet_summary[f"{method}_errors"] = int(len(method_rows))
        packet_summary[f"{method}_selectable_misses"] = int(
            sum(1 for row in method_rows if str(row.get(f"{method}_error_type")) == "selectable_miss")
        )
        packet_summary[f"{method}_gt_missing_top_l_errors"] = int(
            sum(1 for row in method_rows if str(row.get(f"{method}_error_type")) == "gt_missing_top_l")
        )
    return rows, packet_summary


def _summarize(rows: Sequence[dict[str, Any]], packet_summaries: Sequence[dict[str, Any]], snr_db: float, dataset: str) -> dict[str, Any]:
    total = len(rows)
    out: dict[str, Any] = {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "packet_count": int(len(packet_summaries)),
        "symbol_count": int(total),
        "gt_stage1_top_l_recall": float(sum(int(r["gt_in_stage1_top_l"]) for r in rows) / max(1, total)),
        "gt_viterbi_candidate_recall": float(sum(int(r["gt_in_viterbi_candidates"]) for r in rows) / max(1, total)),
        "gt_rank1_rate": float(sum(1 for r in rows if int(r["gt_stage1_rank"]) == 1) / max(1, total)),
        "gt_rank_le3_rate": float(sum(1 for r in rows if 1 <= int(r["gt_stage1_rank"]) <= 3) / max(1, total)),
        "gt_rank_le5_rate": float(sum(1 for r in rows if 1 <= int(r["gt_stage1_rank"]) <= 5) / max(1, total)),
        "gt_rank_le8_rate": float(sum(1 for r in rows if 1 <= int(r["gt_stage1_rank"]) <= 8) / max(1, total)),
        "mean_gt_rank_when_present": _avg([r for r in rows if int(r["gt_stage1_rank"]) > 0], "gt_stage1_rank"),
    }
    for method in METHODS:
        errors = [r for r in rows if int(r.get(f"{method}_hit", 0)) == 0]
        hits = total - len(errors)
        out[f"{method}_raw_ser"] = float(len(errors) / max(1, total))
        out[f"{method}_hit_rate"] = float(hits / max(1, total))
        out[f"{method}_error_count"] = int(len(errors))
        out[f"{method}_selectable_miss_count"] = int(
            sum(1 for r in errors if str(r.get(f"{method}_error_type")) == "selectable_miss")
        )
        out[f"{method}_gt_missing_top_l_error_count"] = int(
            sum(1 for r in errors if str(r.get(f"{method}_error_type")) == "gt_missing_top_l")
        )
        out[f"{method}_error_selectable_miss_rate"] = float(
            out[f"{method}_selectable_miss_count"] / max(1, out[f"{method}_error_count"])
        )
    return out


def main() -> int:
    args = parse_args()
    paths = _dataset_paths(str(args.dataset))
    metadata = _load_metadata(paths, str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    packets = load_packets(paths["symbols"], args.packet)
    if not packets:
        raise ValueError(f"no packets loaded for {args.dataset}")
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v3_config = _build_v3_config(args.top_l)
    first_config = _base_path_config(args, phase_order=1)
    second_config = _base_path_config(args, phase_order=2)
    anchor_config = _anchor_path_config(args)

    all_rows: list[dict[str, Any]] = []
    packet_summary_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for snr_db in _snr_values(args.snr_start, args.snr_stop, args.snr_step):
        noise_power = signal_power * (10.0 ** (-float(snr_db) / 10.0))
        noisy = (samples + math.sqrt(float(noise_power) / 2.0) * unit_noise).astype(np.complex64, copy=False)
        snr_rows: list[dict[str, Any]] = []
        snr_packet_summaries: list[dict[str, Any]] = []
        for packet_index in sorted(packets):
            rows, packet_summary = _evaluate_packet(
                noisy,
                packets[packet_index],
                args,
                v3_config,
                first_config,
                second_config,
                anchor_config,
            )
            for row in rows:
                row["dataset"] = str(args.dataset)
                row["target_snr_db"] = float(snr_db)
            packet_summary["dataset"] = str(args.dataset)
            packet_summary["target_snr_db"] = float(snr_db)
            all_rows.extend(rows)
            snr_rows.extend(rows)
            packet_summary_rows.append(packet_summary)
            snr_packet_summaries.append(packet_summary)
        summary = _summarize(snr_rows, snr_packet_summaries, float(snr_db), str(args.dataset))
        summary_rows.append(summary)
        print(
            f"{args.dataset} snr={snr_db:>6.1f} "
            f"topL={summary['gt_stage1_top_l_recall']:.3f} "
            f"vitCand={summary['gt_viterbi_candidate_recall']:.3f} "
            f"dp1_ser={summary['phase_dp_first_raw_ser']:.3f} "
            f"dp1_selectable_miss={summary['phase_dp_first_selectable_miss_count']} "
            f"dp1_gt_missing={summary['phase_dp_first_gt_missing_top_l_error_count']}",
            flush=True,
        )

    suffix = f"{args.dataset}_snr_{args.snr_start:g}_to_{args.snr_stop:g}".replace("-", "m").replace(".", "p")
    if args.packet is not None:
        suffix += f"_packet_{int(args.packet)}"
    _write_csv(out_dir / f"{suffix}_per_symbol.csv", all_rows)
    _write_csv(out_dir / f"{suffix}_per_packet.csv", packet_summary_rows)
    _write_csv(out_dir / f"{suffix}_summary.csv", summary_rows)
    (out_dir / f"{suffix}_summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "top_l": int(args.top_l),
                    "dp_max_energy_drop_db": float(args.dp_max_energy_drop_db),
                    "dp_energy_weight": float(args.dp_energy_weight),
                    "dp_coherence_weight": float(args.dp_coherence_weight),
                    "dp_rank_weight": float(args.dp_rank_weight),
                    "first_lambda": float(args.first_lambda),
                    "second_lambda": float(args.second_lambda),
                },
                "summary_rows": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote={out_dir / f'{suffix}_summary.csv'}")
    print(f"wrote={out_dir / f'{suffix}_per_symbol.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Diagnose symbol-level gates between dual and multi-origin FFT-bin paths.

This script is intentionally FFT-bin only. It rebuilds the same dual-evidence
and multi-origin Stage-1 paths used by ``evaluate_multi_origin.py``, records the
symbols where the two paths disagree, and optionally sweeps simple runtime
features with a train/test split. Ground truth is used only for this offline
diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[4]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.phase_line.savaux_stage1 import (  # noqa: E402
    SavauxStage1Config,
    default_savaux_phase_path_config,
    payload_abs_indices,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.dual_evidence import (  # noqa: E402
    DualEvidenceFusionConfig,
    build_dual_savaux_stage1_packet_evidence,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.evaluate_island_dp import (  # noqa: E402
    DEFAULT_DATASETS,
    _compare_paths,
    _dataset_paths,
    _err_count,
    _load_packets,
    _noise_samples,
    _snr_values,
    _sum,
    _write_csv,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.multi_origin_evidence import (  # noqa: E402
    MultiOriginFusionConfig,
    build_multi_origin_stage1_packet_evidence,
)
from weak_decoder.phase_line.variants.v1_one_order_dp import select_phase_viterbi_path  # noqa: E402


FeatureFn = Callable[[dict[str, Any]], float]


def _parse_origins(values: Sequence[int] | None) -> tuple[int, ...] | None:
    if not values:
        return None
    return tuple(int(v) for v in values)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _norm_power(power: np.ndarray, raw_bin: int) -> float:
    values = np.asarray(power, dtype=np.float64)
    if values.size == 0:
        return 0.0
    peak = float(np.max(values))
    if peak <= 0.0:
        return 0.0
    b = int(raw_bin)
    if b < 0 or b >= values.size:
        return 0.0
    return float(values[b] / (peak + 1e-30))


def _rank(power: np.ndarray, raw_bin: int) -> int:
    values = np.asarray(power, dtype=np.float64)
    b = int(raw_bin)
    if values.size == 0 or b < 0 or b >= values.size:
        return int(values.size + 1)
    return int(1 + np.sum(values > values[b]))


def _origin_stats(origin_powers: Sequence[np.ndarray], dual_bin: int, multi_bin: int) -> dict[str, float | int]:
    margins: list[float] = []
    dual_top1 = 0
    multi_top1 = 0
    for power in origin_powers:
        row = np.asarray(power, dtype=np.float64)
        if row.size == 0:
            continue
        top1 = int(np.argmax(row))
        dual_top1 += int(top1 == int(dual_bin))
        multi_top1 += int(top1 == int(multi_bin))
        margins.append(_norm_power(row, int(multi_bin)) - _norm_power(row, int(dual_bin)))
    values = np.asarray(margins, dtype=np.float64)
    if values.size == 0:
        return {
            "origin_dual_top1_count": 0,
            "origin_multi_top1_count": 0,
            "origin_vote_margin": 0,
            "origin_mean_margin": 0.0,
            "origin_median_margin": 0.0,
            "origin_max_margin": 0.0,
            "origin_min_margin": 0.0,
        }
    return {
        "origin_dual_top1_count": int(dual_top1),
        "origin_multi_top1_count": int(multi_top1),
        "origin_vote_margin": int(multi_top1 - dual_top1),
        "origin_mean_margin": float(np.mean(values)),
        "origin_median_margin": float(np.median(values)),
        "origin_max_margin": float(np.max(values)),
        "origin_min_margin": float(np.min(values)),
    }


def _packet_rows_for_one(
    samples: np.ndarray,
    packet: dict[str, Any],
    path_config: Any,
    dual_config: DualEvidenceFusionConfig,
    multi_config: MultiOriginFusionConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = list(packet["payload_symbols"])
    start_samples = [int(item["start_sample"]) for item in payload]
    gt_bins = [int(item["gt_bin"]) for item in payload]
    residual_sto_chips = [float(item.get("sfo_cum_before", 0.0)) for item in payload]
    abs_indices = payload_abs_indices(
        [int(item["payload_symbol_index"]) for item in payload],
        preamble_len=float(packet.get("preamble_len", 8.0)),
    )
    stage1_base = SavauxStage1Config(retain_dechirped_symbols=True)
    dual = build_dual_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        abs_indices=abs_indices,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=int(packet["header_start_sample"]),
        residual_sto_chips=residual_sto_chips,
        stage1_config=stage1_base,
        fusion_config=dual_config,
    )
    multi = build_multi_origin_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        abs_indices=abs_indices,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=int(packet["header_start_sample"]),
        residual_sto_chips=residual_sto_chips,
        stage1_config=stage1_base,
        fusion_config=multi_config,
    )
    v1 = select_phase_viterbi_path(
        center_spectra=dual.old_stage1.center_spectra,
        evidence_powers=dual.old_stage1.evidence_powers,
        abs_indices=dual.old_stage1.abs_indices,
        config=path_config,
        offset_coherences=dual.old_stage1.branch_phase_agreements,
    )
    dual_v1 = select_phase_viterbi_path(
        center_spectra=dual.center_spectra,
        evidence_powers=dual.evidence_powers,
        abs_indices=dual.abs_indices,
        config=path_config,
        offset_coherences=dual.branch_phase_agreements,
    )
    multi_v1 = select_phase_viterbi_path(
        center_spectra=multi.center_spectra,
        evidence_powers=multi.evidence_powers,
        abs_indices=multi.abs_indices,
        config=path_config,
        offset_coherences=multi.branch_phase_agreements,
    )

    v1_err, compared = _err_count(v1.selected_raw_bins, gt_bins)
    dual_err, _ = _err_count(dual_v1.selected_raw_bins, gt_bins)
    multi_err, _ = _err_count(multi_v1.selected_raw_bins, gt_bins)
    multi_rescue, multi_break, multi_changes = _compare_paths(
        multi_v1.selected_raw_bins,
        dual_v1.selected_raw_bins,
        gt_bins,
    )
    packet_row = {
        "packet_index": int(packet["packet_index"]),
        "symbol_count": int(compared),
        "v1_err": int(v1_err),
        "dual_err": int(dual_err),
        "multi_err": int(multi_err),
        "multi_rescue_vs_dual": int(multi_rescue),
        "multi_break_vs_dual": int(multi_break),
        "multi_changes_vs_dual": int(multi_changes),
        "v1_trajectory_score": float(v1.trajectory_score),
        "dual_trajectory_score": float(dual_v1.trajectory_score),
        "multi_trajectory_score": float(multi_v1.trajectory_score),
        "dual_mean_amp_score": float(dual_v1.mean_amp_score),
        "multi_mean_amp_score": float(multi_v1.mean_amp_score),
        "multi_minus_dual_trajectory": float(multi_v1.trajectory_score - dual_v1.trajectory_score),
        "multi_minus_dual_amp": float(multi_v1.mean_amp_score - dual_v1.mean_amp_score),
    }

    changed_rows: list[dict[str, Any]] = []
    count = min(len(gt_bins), len(dual_v1.selected_raw_bins), len(multi_v1.selected_raw_bins))
    for idx in range(count):
        dual_bin = int(dual_v1.selected_raw_bins[idx])
        multi_bin = int(multi_v1.selected_raw_bins[idx])
        if dual_bin == multi_bin:
            continue
        gt_bin = int(gt_bins[idx])
        old_power = np.asarray(dual.old_stage1.evidence_powers[idx], dtype=np.float64)
        corrected_power = np.asarray(dual.corrected_stage1.evidence_powers[idx], dtype=np.float64)
        dual_power = np.asarray(dual.evidence_powers[idx], dtype=np.float64)
        multi_power = np.asarray(multi.evidence_powers[idx], dtype=np.float64)
        origin_powers = [stage1.evidence_powers[idx] for stage1 in multi.origin_stage1]
        row: dict[str, Any] = {
            "packet_index": int(packet["packet_index"]),
            "symbol_index": int(idx),
            "abs_symbol_index": float(abs_indices[idx]),
            "gt_bin": int(gt_bin),
            "v1_bin": int(v1.selected_raw_bins[idx]) if idx < len(v1.selected_raw_bins) else "",
            "dual_bin": int(dual_bin),
            "multi_bin": int(multi_bin),
            "v1_correct": int(idx < len(v1.selected_raw_bins) and int(v1.selected_raw_bins[idx]) == gt_bin),
            "dual_correct": int(dual_bin == gt_bin),
            "multi_correct": int(multi_bin == gt_bin),
            "packet_trajectory_gain": float(packet_row["multi_minus_dual_trajectory"]),
            "packet_amp_gain": float(packet_row["multi_minus_dual_amp"]),
            "packet_changes_vs_dual": int(multi_changes),
        }
        for prefix, power in (
            ("old", old_power),
            ("corrected", corrected_power),
            ("dual_fused", dual_power),
            ("multi_fused", multi_power),
        ):
            row[f"{prefix}_dual_norm"] = _norm_power(power, dual_bin)
            row[f"{prefix}_multi_norm"] = _norm_power(power, multi_bin)
            row[f"{prefix}_margin"] = float(row[f"{prefix}_multi_norm"] - row[f"{prefix}_dual_norm"])
            row[f"{prefix}_dual_rank"] = _rank(power, dual_bin)
            row[f"{prefix}_multi_rank"] = _rank(power, multi_bin)
            row[f"{prefix}_multi_rank_score"] = -float(row[f"{prefix}_multi_rank"])
        row.update(_origin_stats(origin_powers, dual_bin, multi_bin))
        changed_rows.append(row)
    return packet_row, changed_rows


def _candidate_thresholds(values: Sequence[float], max_count: int = 80) -> list[float]:
    finite = sorted({float(v) for v in values if np.isfinite(float(v))})
    if not finite:
        return [0.0]
    if len(finite) <= int(max_count):
        return finite
    qs = np.linspace(0.0, 1.0, int(max_count))
    return sorted({float(np.quantile(np.asarray(finite, dtype=np.float64), q)) for q in qs})


def _row_filter(row: dict[str, Any], seeds: set[int] | None) -> bool:
    if seeds is None:
        return True
    return int(row["seed"]) in seeds


def _gate_error_count(
    packet_rows: Sequence[dict[str, Any]],
    changed_rows: Sequence[dict[str, Any]],
    row_filter: Callable[[dict[str, Any]], bool],
    selector: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    base_dual = int(sum(int(row["dual_err"]) for row in packet_rows if row_filter(row)))
    base_v1 = int(sum(int(row["v1_err"]) for row in packet_rows if row_filter(row)))
    base_multi = int(sum(int(row["multi_err"]) for row in packet_rows if row_filter(row)))
    rescue = 0
    break_count = 0
    changes = 0
    for row in changed_rows:
        if not row_filter(row) or not selector(row):
            continue
        changes += 1
        dual_ok = bool(int(row["dual_correct"]))
        multi_ok = bool(int(row["multi_correct"]))
        rescue += int((not dual_ok) and multi_ok)
        break_count += int(dual_ok and (not multi_ok))
    errors = int(base_dual - rescue + break_count)
    return {
        "v1_err": int(base_v1),
        "dual_err": int(base_dual),
        "multi_err": int(base_multi),
        "gated_err": int(errors),
        "rescue_vs_dual": int(rescue),
        "break_vs_dual": int(break_count),
        "changes_vs_dual": int(changes),
    }


def _feature_value(row: dict[str, Any], feature: str) -> float:
    return _as_float(row.get(feature), default=-float("inf"))


def _sweep_two_feature_gate(
    packet_rows: Sequence[dict[str, Any]],
    changed_rows: Sequence[dict[str, Any]],
    train_seeds: set[int] | None,
    test_seeds: set[int] | None,
    features: Sequence[str],
) -> list[dict[str, Any]]:
    train_filter = lambda row: _row_filter(row, train_seeds)
    test_filter = lambda row: _row_filter(row, test_seeds)
    train_rows = [row for row in changed_rows if train_filter(row)]
    trajectory_thresholds = _candidate_thresholds(
        [_feature_value(row, "packet_trajectory_gain") for row in train_rows],
        max_count=80,
    )
    out: list[dict[str, Any]] = []
    for feature in features:
        feature_thresholds = _candidate_thresholds(
            [_feature_value(row, feature) for row in train_rows],
            max_count=80,
        )
        best: dict[str, Any] | None = None
        for tg_threshold in trajectory_thresholds:
            for feature_threshold in feature_thresholds:
                selector = lambda row, t=tg_threshold, f=feature_threshold, name=feature: (
                    _feature_value(row, "packet_trajectory_gain") >= float(t)
                    and _feature_value(row, name) >= float(f)
                )
                train = _gate_error_count(packet_rows, changed_rows, train_filter, selector)
                candidate = {
                    "feature": feature,
                    "trajectory_threshold": float(tg_threshold),
                    "feature_threshold": float(feature_threshold),
                    "train_gated_err": int(train["gated_err"]),
                    "train_rescue_vs_dual": int(train["rescue_vs_dual"]),
                    "train_break_vs_dual": int(train["break_vs_dual"]),
                    "train_changes_vs_dual": int(train["changes_vs_dual"]),
                }
                if best is None:
                    best = candidate
                    continue
                key = (
                    int(candidate["train_gated_err"]),
                    int(candidate["train_break_vs_dual"]),
                    int(candidate["train_changes_vs_dual"]),
                )
                best_key = (
                    int(best["train_gated_err"]),
                    int(best["train_break_vs_dual"]),
                    int(best["train_changes_vs_dual"]),
                )
                if key < best_key:
                    best = candidate
        if best is None:
            continue
        selector = lambda row, b=best, name=feature: (
            _feature_value(row, "packet_trajectory_gain") >= float(b["trajectory_threshold"])
            and _feature_value(row, name) >= float(b["feature_threshold"])
        )
        train_eval = _gate_error_count(packet_rows, changed_rows, train_filter, selector)
        test_eval = _gate_error_count(packet_rows, changed_rows, test_filter, selector)
        all_eval = _gate_error_count(packet_rows, changed_rows, lambda _row: True, selector)
        best.update(
            {
                "train_v1_err": int(train_eval["v1_err"]),
                "train_dual_err": int(train_eval["dual_err"]),
                "test_v1_err": int(test_eval["v1_err"]),
                "test_dual_err": int(test_eval["dual_err"]),
                "test_gated_err": int(test_eval["gated_err"]),
                "test_rescue_vs_dual": int(test_eval["rescue_vs_dual"]),
                "test_break_vs_dual": int(test_eval["break_vs_dual"]),
                "test_changes_vs_dual": int(test_eval["changes_vs_dual"]),
                "all_v1_err": int(all_eval["v1_err"]),
                "all_dual_err": int(all_eval["dual_err"]),
                "all_gated_err": int(all_eval["gated_err"]),
                "all_rescue_vs_dual": int(all_eval["rescue_vs_dual"]),
                "all_break_vs_dual": int(all_eval["break_vs_dual"]),
                "all_changes_vs_dual": int(all_eval["changes_vs_dual"]),
            }
        )
        out.append(best)
    out.sort(key=lambda row: (int(row["test_gated_err"]), int(row["all_gated_err"]), int(row["train_gated_err"])))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snrs", nargs="*", type=float, default=[-25.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--train-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--test-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-packets", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=THIS_FILE.parent / "_eval" / "dual_multi_symbol_gate_diag")
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--v1-top-l", type=int, default=16)
    parser.add_argument("--dual-mode", choices=["sum_norm", "product_norm", "max_norm"], default="product_norm")
    parser.add_argument("--dual-corrected-weight", type=float, default=0.5)
    parser.add_argument(
        "--multi-mode",
        choices=["sum_norm", "product_norm", "max_norm", "median_norm", "top2_mean_norm", "trimmed_mean_norm"],
        default="sum_norm",
    )
    parser.add_argument(
        "--multi-phase-mode",
        choices=["fixed_origin", "weighted_unit", "weighted_complex", "max_bin", "best_symbol_origin"],
        default="weighted_unit",
    )
    parser.add_argument("--multi-corrected", action="store_true")
    parser.add_argument("--multi-origins", nargs="*", type=int, default=None)
    parser.add_argument("--multi-phase-origin", type=int, default=None)
    parser.add_argument("--stage1-top-k", type=int, default=40)
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path_config = default_savaux_phase_path_config(top_l=int(args.v1_top_l))
    dual_config = DualEvidenceFusionConfig(
        mode=str(args.dual_mode),
        corrected_weight=float(args.dual_corrected_weight),
        stage1_top_k=int(args.stage1_top_k),
        retain_dechirped_symbols=True,
    )
    multi_config = MultiOriginFusionConfig(
        mode=str(args.multi_mode),
        phase_mode=str(args.multi_phase_mode),
        stage1_top_k=int(args.stage1_top_k),
        origins=_parse_origins(args.multi_origins),
        phase_origin=args.multi_phase_origin,
        corrected=bool(args.multi_corrected),
        retain_dechirped_symbols=True,
    )
    out_dir = Path(args.output_dir).resolve()
    packet_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = _dataset_paths(str(dataset))
        clean = np.fromfile(iq_path, dtype=np.complex64)
        packets = _load_packets(symbol_path)
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        for seed in args.seeds:
            for snr_db in _snr_values(args.snrs):
                samples = _noise_samples(clean, snr_db, int(seed), args.signal_reference_power)
                group_packets: list[dict[str, Any]] = []
                group_changed: list[dict[str, Any]] = []
                for packet in packets:
                    packet_row, rows = _packet_rows_for_one(samples, packet, path_config, dual_config, multi_config)
                    common = {"dataset": str(dataset), "snr_db": "" if snr_db is None else float(snr_db), "seed": int(seed)}
                    packet_row.update(common)
                    for row in rows:
                        row.update(common)
                    group_packets.append(packet_row)
                    group_changed.extend(rows)
                    packet_rows.append(packet_row)
                    changed_rows.extend(rows)
                symbols = _sum(group_packets, "symbol_count")
                summary = {
                    "dataset": str(dataset),
                    "snr_db": "" if snr_db is None else float(snr_db),
                    "seed": int(seed),
                    "packet_count": int(len(group_packets)),
                    "symbol_count": int(symbols),
                    "v1_err": _sum(group_packets, "v1_err"),
                    "dual_err": _sum(group_packets, "dual_err"),
                    "multi_err": _sum(group_packets, "multi_err"),
                    "changed_symbols": int(len(group_changed)),
                    "multi_rescue_vs_dual": _sum(group_packets, "multi_rescue_vs_dual"),
                    "multi_break_vs_dual": _sum(group_packets, "multi_break_vs_dual"),
                }
                summary_rows.append(summary)
                print(
                    f"{dataset} snr={snr_db} seed={seed}: "
                    f"v1={summary['v1_err']} dual={summary['dual_err']} multi={summary['multi_err']} "
                    f"changed={summary['changed_symbols']} "
                    f"multi rescue/break vs dual={summary['multi_rescue_vs_dual']}/{summary['multi_break_vs_dual']}",
                    flush=True,
                )
    _write_csv(out_dir / "packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "changed_symbols.csv", changed_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)

    sweep_rows: list[dict[str, Any]] = []
    if bool(args.sweep):
        features = [
            "old_margin",
            "corrected_margin",
            "dual_fused_margin",
            "multi_fused_margin",
            "origin_mean_margin",
            "origin_median_margin",
            "origin_max_margin",
            "origin_vote_margin",
            "old_multi_norm",
            "multi_fused_multi_norm",
            "old_multi_rank_score",
            "multi_fused_multi_rank_score",
        ]
        train_seeds = None if args.train_seeds is None else {int(v) for v in args.train_seeds}
        test_seeds = None if args.test_seeds is None else {int(v) for v in args.test_seeds}
        sweep_rows = _sweep_two_feature_gate(packet_rows, changed_rows, train_seeds, test_seeds, features)
        _write_csv(out_dir / "gate_sweep.csv", sweep_rows)
        print("top gate sweep rows:")
        for row in sweep_rows[:10]:
            print(
                f"{row['feature']} tg>={row['trajectory_threshold']:.9g} "
                f"feat>={row['feature_threshold']:.9g} "
                f"train {row['train_dual_err']}->{row['train_gated_err']} "
                f"test {row['test_dual_err']}->{row['test_gated_err']} "
                f"all {row['dual_err'] if 'dual_err' in row else row['all_dual_err']}->{row['all_gated_err']}",
                flush=True,
            )
    manifest = {
        "datasets": list(args.datasets),
        "snrs": list(args.snrs),
        "seeds": list(args.seeds),
        "train_seeds": args.train_seeds,
        "test_seeds": args.test_seeds,
        "packet_count": len(packet_rows),
        "changed_symbol_count": len(changed_rows),
        "sweep_count": len(sweep_rows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

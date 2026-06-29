#!/usr/bin/env python3
"""Evaluate multi-origin FFT-bin evidence fusion.

The runner compares only FFT-bin selectors:

* center-origin v1 one-order DP;
* dual old/corrected evidence fusion;
* multi-origin evidence fusion across sampling phases.

No codec or CRC feedback is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[4]
GR_LORA_ROOT = WEAK_ROOT.parent
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
    MultiOriginGateConfig,
    arbitrate_dual_vs_multi_origin_path,
    build_multi_origin_stage1_packet_evidence,
)
from weak_decoder.phase_line.variants.v1_one_order_dp import select_phase_viterbi_path  # noqa: E402


def _parse_origins(values: Sequence[int] | None) -> tuple[int, ...] | None:
    if not values:
        return None
    return tuple(int(v) for v in values)


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    path_config,
    dual_config: DualEvidenceFusionConfig,
    multi_config: MultiOriginFusionConfig,
    gate_config: MultiOriginGateConfig,
) -> dict[str, Any]:
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
    gated = arbitrate_dual_vs_multi_origin_path(
        dual_v1,
        multi_v1,
        gate_config,
        old_evidence_powers=dual.old_stage1.evidence_powers,
    )

    v1_err, compared = _err_count(v1.selected_raw_bins, gt_bins)
    dual_err, _ = _err_count(dual_v1.selected_raw_bins, gt_bins)
    multi_err, _ = _err_count(multi_v1.selected_raw_bins, gt_bins)
    gated_err, _ = _err_count(gated.selected.selected_raw_bins, gt_bins)
    dual_rescue, dual_break, dual_changes = _compare_paths(dual_v1.selected_raw_bins, v1.selected_raw_bins, gt_bins)
    multi_rescue, multi_break, multi_changes = _compare_paths(multi_v1.selected_raw_bins, v1.selected_raw_bins, gt_bins)
    multi_vs_dual_rescue, multi_vs_dual_break, multi_vs_dual_changes = _compare_paths(
        multi_v1.selected_raw_bins, dual_v1.selected_raw_bins, gt_bins
    )
    gated_rescue, gated_break, gated_changes = _compare_paths(
        gated.selected.selected_raw_bins, v1.selected_raw_bins, gt_bins
    )
    gated_vs_dual_rescue, gated_vs_dual_break, gated_vs_dual_changes = _compare_paths(
        gated.selected.selected_raw_bins, dual_v1.selected_raw_bins, gt_bins
    )
    return {
        "packet_index": int(packet["packet_index"]),
        "symbol_count": int(compared),
        "v1_err": int(v1_err),
        "dual_err": int(dual_err),
        "multi_err": int(multi_err),
        "gated_err": int(gated_err),
        "v1_ser": float(v1_err / max(1, compared)),
        "dual_ser": float(dual_err / max(1, compared)),
        "multi_ser": float(multi_err / max(1, compared)),
        "gated_ser": float(gated_err / max(1, compared)),
        "dual_gain_vs_v1": float((v1_err - dual_err) / max(1, compared)),
        "multi_gain_vs_v1": float((v1_err - multi_err) / max(1, compared)),
        "multi_gain_vs_dual": float((dual_err - multi_err) / max(1, compared)),
        "gated_gain_vs_v1": float((v1_err - gated_err) / max(1, compared)),
        "gated_gain_vs_dual": float((dual_err - gated_err) / max(1, compared)),
        "dual_rescue_vs_v1": int(dual_rescue),
        "dual_break_vs_v1": int(dual_break),
        "dual_changes_vs_v1": int(dual_changes),
        "multi_rescue_vs_v1": int(multi_rescue),
        "multi_break_vs_v1": int(multi_break),
        "multi_changes_vs_v1": int(multi_changes),
        "multi_rescue_vs_dual": int(multi_vs_dual_rescue),
        "multi_break_vs_dual": int(multi_vs_dual_break),
        "multi_changes_vs_dual": int(multi_vs_dual_changes),
        "gated_rescue_vs_v1": int(gated_rescue),
        "gated_break_vs_v1": int(gated_break),
        "gated_changes_vs_v1": int(gated_changes),
        "gated_rescue_vs_dual": int(gated_vs_dual_rescue),
        "gated_break_vs_dual": int(gated_vs_dual_break),
        "gated_changes_vs_dual": int(gated_vs_dual_changes),
        "gated_used_multi": int(gated.decision.use_multi),
        "gated_reason": str(gated.decision.reason),
        "gated_trajectory_gain": float(gated.decision.trajectory_gain),
        "gated_amp_gain": float(gated.decision.amp_gain),
        "gated_changes_decision_vs_dual": int(gated.decision.changes_vs_dual),
        "v1_trajectory_score": float(v1.trajectory_score),
        "v1_mean_amp_score": float(v1.mean_amp_score),
        "dual_trajectory_score": float(dual_v1.trajectory_score),
        "dual_mean_amp_score": float(dual_v1.mean_amp_score),
        "multi_trajectory_score": float(multi_v1.trajectory_score),
        "multi_mean_amp_score": float(multi_v1.mean_amp_score),
        "multi_minus_dual_trajectory": float(multi_v1.trajectory_score - dual_v1.trajectory_score),
        "multi_minus_dual_amp": float(multi_v1.mean_amp_score - dual_v1.mean_amp_score),
        "multi_phase_origin": int(multi.phase_origin),
        "multi_origin_count": int(len(multi.origin_shifts)),
    }


def _summary(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float | None, seed: int) -> dict[str, Any]:
    symbols = _sum(rows, "symbol_count")
    v1_err = _sum(rows, "v1_err")
    dual_err = _sum(rows, "dual_err")
    multi_err = _sum(rows, "multi_err")
    gated_err = _sum(rows, "gated_err")
    return {
        "dataset": dataset,
        "snr_db": "" if snr_db is None else float(snr_db),
        "seed": int(seed),
        "packet_count": int(len(rows)),
        "symbol_count": int(symbols),
        "v1_ser": float(v1_err / max(1, symbols)),
        "dual_ser": float(dual_err / max(1, symbols)),
        "multi_ser": float(multi_err / max(1, symbols)),
        "gated_ser": float(gated_err / max(1, symbols)),
        "dual_gain_vs_v1": float((v1_err - dual_err) / max(1, symbols)),
        "multi_gain_vs_v1": float((v1_err - multi_err) / max(1, symbols)),
        "multi_gain_vs_dual": float((dual_err - multi_err) / max(1, symbols)),
        "gated_gain_vs_v1": float((v1_err - gated_err) / max(1, symbols)),
        "gated_gain_vs_dual": float((dual_err - gated_err) / max(1, symbols)),
        "dual_rescue_vs_v1": _sum(rows, "dual_rescue_vs_v1"),
        "dual_break_vs_v1": _sum(rows, "dual_break_vs_v1"),
        "multi_rescue_vs_v1": _sum(rows, "multi_rescue_vs_v1"),
        "multi_break_vs_v1": _sum(rows, "multi_break_vs_v1"),
        "multi_rescue_vs_dual": _sum(rows, "multi_rescue_vs_dual"),
        "multi_break_vs_dual": _sum(rows, "multi_break_vs_dual"),
        "gated_rescue_vs_v1": _sum(rows, "gated_rescue_vs_v1"),
        "gated_break_vs_v1": _sum(rows, "gated_break_vs_v1"),
        "gated_rescue_vs_dual": _sum(rows, "gated_rescue_vs_dual"),
        "gated_break_vs_dual": _sum(rows, "gated_break_vs_dual"),
        "gated_used_multi_count": _sum(rows, "gated_used_multi"),
        "mean_v1_trajectory_score": float(np.mean([float(row["v1_trajectory_score"]) for row in rows])) if rows else 0.0,
        "mean_dual_trajectory_score": float(np.mean([float(row["dual_trajectory_score"]) for row in rows])) if rows else 0.0,
        "mean_multi_trajectory_score": float(np.mean([float(row["multi_trajectory_score"]) for row in rows])) if rows else 0.0,
        "mean_v1_amp_score": float(np.mean([float(row["v1_mean_amp_score"]) for row in rows])) if rows else 0.0,
        "mean_dual_amp_score": float(np.mean([float(row["dual_mean_amp_score"]) for row in rows])) if rows else 0.0,
        "mean_multi_amp_score": float(np.mean([float(row["multi_mean_amp_score"]) for row in rows])) if rows else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snrs", nargs="*", type=float, default=[-25.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-packets", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=THIS_FILE.parent / "_eval" / "multi_origin_compare")
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
        default="fixed_origin",
    )
    parser.add_argument("--multi-corrected", action="store_true")
    parser.add_argument("--multi-origins", nargs="*", type=int, default=None)
    parser.add_argument("--multi-phase-origin", type=int, default=None)
    parser.add_argument("--stage1-top-k", type=int, default=40)
    parser.add_argument("--enable-gate", action="store_true")
    parser.add_argument("--gate-min-trajectory-gain", type=float, default=0.003267)
    parser.add_argument("--gate-min-amp-gain", type=float, default=None)
    parser.add_argument("--gate-max-changes-vs-dual", type=int, default=None)
    parser.add_argument("--enable-symbol-gate", action="store_true")
    parser.add_argument("--symbol-min-trajectory-gain", type=float, default=0.005618281741596176)
    parser.add_argument("--symbol-min-old-power-margin", type=float, default=-0.3055904873940537)
    parser.add_argument("--symbol-min-old-multi-norm", type=float, default=None)
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
    gate_config = MultiOriginGateConfig(
        enabled=bool(args.enable_gate),
        min_trajectory_gain=float(args.gate_min_trajectory_gain),
        min_amp_gain=args.gate_min_amp_gain,
        max_changes_vs_dual=args.gate_max_changes_vs_dual,
        symbol_gate_enabled=bool(args.enable_symbol_gate),
        symbol_min_trajectory_gain=float(args.symbol_min_trajectory_gain),
        symbol_min_old_power_margin=float(args.symbol_min_old_power_margin),
        symbol_min_old_multi_norm=args.symbol_min_old_multi_norm,
    )
    out_dir = Path(args.output_dir).resolve()
    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = _dataset_paths(str(dataset))
        if not iq_path.exists():
            raise FileNotFoundError(iq_path)
        if not symbol_path.exists():
            raise FileNotFoundError(symbol_path)
        clean = np.fromfile(iq_path, dtype=np.complex64)
        packets = _load_packets(symbol_path)
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        for seed in args.seeds:
            for snr_db in _snr_values(args.snrs):
                samples = _noise_samples(clean, snr_db, int(seed), args.signal_reference_power)
                rows: list[dict[str, Any]] = []
                for packet in packets:
                    row = _evaluate_packet(samples, packet, path_config, dual_config, multi_config, gate_config)
                    row.update({"dataset": str(dataset), "snr_db": "" if snr_db is None else float(snr_db), "seed": int(seed)})
                    rows.append(row)
                    packet_rows.append(row)
                summary = _summary(rows, str(dataset), snr_db, int(seed))
                summary_rows.append(summary)
                print(
                    f"{dataset} snr={snr_db} seed={seed}: "
                    f"v1={summary['v1_ser']:.4f} dual={summary['dual_ser']:.4f} "
                    f"multi={summary['multi_ser']:.4f} gated={summary['gated_ser']:.4f} "
                    f"multi_gain={summary['multi_gain_vs_v1']:.4f} "
                    f"gated_gain={summary['gated_gain_vs_v1']:.4f} "
                    f"multi_vs_dual={summary['multi_gain_vs_dual']:.4f} "
                    f"gated_used={summary['gated_used_multi_count']} "
                    f"rescue/break={summary['gated_rescue_vs_v1']}/{summary['gated_break_vs_v1']}",
                    flush=True,
                )
    _write_csv(out_dir / "packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

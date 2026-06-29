#!/usr/bin/env python3
"""Diagnose oracle rescue space for the island reconstruction branch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[4]
GR_LORA_ROOT = WEAK_ROOT.parent
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.candidate_pruning import top_bins  # noqa: E402
from weak_decoder.phase_line.savaux_stage1 import (  # noqa: E402
    SavauxStage1Config,
    build_savaux_stage1_packet_evidence,
    default_savaux_phase_path_config,
    payload_abs_indices,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction import IslandReconstructionConfig  # noqa: E402
from weak_decoder.phase_line.variants.island_dp_reconstruction.evaluate_island_dp import (  # noqa: E402
    DEFAULT_DATASETS,
    _dataset_paths,
    _load_packets,
    _noise_samples,
    _snr_values,
    _write_csv,
)
from weak_decoder.phase_line.variants.v1_one_order_dp import select_phase_viterbi_path  # noqa: E402


def _candidate_set(
    hard_bin: int,
    v1_bin: int,
    power: np.ndarray,
    margin_db: float,
    cfg: IslandReconstructionConfig,
) -> set[int]:
    bins = {int(v) for v in top_bins(np.asarray(power, dtype=np.float64), min(int(cfg.top_l), int(power.size)))}
    bins.add(int(v1_bin))
    if not bool(cfg.allow_third_bin_when_baseline_differs) and int(v1_bin) != int(hard_bin):
        bins = {int(v1_bin), int(hard_bin)}.intersection(bins)
    if int(v1_bin) != int(hard_bin) and float(margin_db) < float(cfg.return_to_hard_min_margin_db):
        bins.discard(int(hard_bin))
    return bins


def _row_for_packet(
    samples: np.ndarray,
    dataset: str,
    snr_db: float | None,
    seed: int,
    packet: dict[str, Any],
    path_config,
    island_config: IslandReconstructionConfig,
) -> dict[str, Any]:
    payload = list(packet["payload_symbols"])
    start_samples = [int(item["start_sample"]) for item in payload]
    gt_bins = [int(item["gt_bin"]) for item in payload]
    abs_indices = payload_abs_indices(
        [int(item["payload_symbol_index"]) for item in payload],
        preamble_len=float(packet.get("preamble_len", 8.0)),
    )
    stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        abs_indices=abs_indices,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=int(packet["header_start_sample"]),
        config=SavauxStage1Config(retain_dechirped_symbols=False),
    )
    hard_bins = tuple(int(symbol.top1_bin) for symbol in stage1.symbols)
    v1 = select_phase_viterbi_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=path_config,
        offset_coherences=stage1.branch_phase_agreements,
    )
    miss = 0
    miss_gt_in_top_l = 0
    miss_gt_in_current_candidates = 0
    miss_gt_eq_hard = 0
    miss_gt_eq_third = 0
    miss_gt_absent = 0
    current_constraint_lost_gt = 0
    hard_correct_v1_wrong = 0
    v1_correct_hard_wrong = 0
    count = min(len(gt_bins), len(hard_bins), len(v1.selected_raw_bins), len(stage1.symbols))
    for idx in range(count):
        gt = int(gt_bins[idx])
        hard = int(hard_bins[idx])
        v1_bin = int(v1.selected_raw_bins[idx])
        if v1_bin == gt:
            if hard != gt:
                v1_correct_hard_wrong += 1
            continue
        miss += 1
        if hard == gt:
            hard_correct_v1_wrong += 1
            miss_gt_eq_hard += 1
        elif gt != v1_bin:
            miss_gt_eq_third += 1
        power = np.asarray(stage1.evidence_powers[idx], dtype=np.float64)
        top_l_set = {int(v) for v in top_bins(power, min(int(island_config.top_l), int(power.size)))}
        if gt in top_l_set:
            miss_gt_in_top_l += 1
        else:
            miss_gt_absent += 1
        cand = _candidate_set(
            hard_bin=hard,
            v1_bin=v1_bin,
            power=power,
            margin_db=float(stage1.symbols[idx].top1_margin_db),
            cfg=island_config,
        )
        if gt in cand:
            miss_gt_in_current_candidates += 1
        elif gt in top_l_set:
            current_constraint_lost_gt += 1
    return {
        "dataset": dataset,
        "snr_db": "" if snr_db is None else float(snr_db),
        "seed": int(seed),
        "packet_index": int(packet["packet_index"]),
        "symbol_count": int(count),
        "v1_miss": int(miss),
        "v1_miss_gt_in_top_l": int(miss_gt_in_top_l),
        "v1_miss_gt_in_current_candidates": int(miss_gt_in_current_candidates),
        "v1_miss_gt_eq_hard": int(miss_gt_eq_hard),
        "v1_miss_gt_eq_third": int(miss_gt_eq_third),
        "v1_miss_gt_absent_from_top_l": int(miss_gt_absent),
        "current_constraint_lost_gt": int(current_constraint_lost_gt),
        "hard_correct_v1_wrong": int(hard_correct_v1_wrong),
        "v1_correct_hard_wrong": int(v1_correct_hard_wrong),
    }


def _summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["snr_db"]), int(row["seed"])), []).append(row)
    out: list[dict[str, Any]] = []
    fields = [
        "symbol_count",
        "v1_miss",
        "v1_miss_gt_in_top_l",
        "v1_miss_gt_in_current_candidates",
        "v1_miss_gt_eq_hard",
        "v1_miss_gt_eq_third",
        "v1_miss_gt_absent_from_top_l",
        "current_constraint_lost_gt",
        "hard_correct_v1_wrong",
        "v1_correct_hard_wrong",
    ]
    for (dataset, snr_db, seed), items in sorted(groups.items()):
        row: dict[str, Any] = {"dataset": dataset, "snr_db": snr_db, "seed": seed, "packet_count": len(items)}
        for field in fields:
            row[field] = int(sum(int(item[field]) for item in items))
        row["top_l_oracle_recall_on_v1_miss"] = float(
            row["v1_miss_gt_in_top_l"] / max(1, row["v1_miss"])
        )
        row["current_candidate_recall_on_v1_miss"] = float(
            row["v1_miss_gt_in_current_candidates"] / max(1, row["v1_miss"])
        )
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snrs", nargs="*", type=float, default=[-25.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=THIS_FILE.parent / "_eval" / "rescue_space")
    parser.add_argument("--v1-top-l", type=int, default=16)
    parser.add_argument("--island-top-l", type=int, default=40)
    parser.add_argument("--allow-third-bin-when-baseline-differs", action="store_true")
    parser.add_argument("--return-to-hard-min-margin-db", type=float, default=0.80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path_config = default_savaux_phase_path_config(top_l=int(args.v1_top_l))
    island_config = IslandReconstructionConfig(
        top_l=int(args.island_top_l),
        allow_third_bin_when_baseline_differs=bool(args.allow_third_bin_when_baseline_differs),
        return_to_hard_min_margin_db=float(args.return_to_hard_min_margin_db),
    )
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = _dataset_paths(str(dataset))
        clean = np.fromfile(iq_path, dtype=np.complex64)
        packets = _load_packets(symbol_path)
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        for seed in args.seeds:
            for snr_db in _snr_values(args.snrs):
                samples = _noise_samples(clean, snr_db, int(seed), None)
                for packet in packets:
                    rows.append(_row_for_packet(samples, str(dataset), snr_db, int(seed), packet, path_config, island_config))
    out_dir = Path(args.output_dir).resolve()
    _write_csv(out_dir / "packet_rescue_space.csv", rows)
    _write_csv(out_dir / "summary_rescue_space.csv", _summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

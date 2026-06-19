#!/usr/bin/env python3
"""Diagnose where the ground-truth codec path is pruned.

This is an evaluation-only tool for synthetic noisy sweeps where clean
header-first symbols provide payload raw-bin ground truth.  It does not feed
ground truth into the decoder.  Instead it rebuilds the same Savaux+codec
likelihoods and reports whether the GT path survives each internal stage:

1. per-symbol Savaux Top-K evidence,
2. per-row Hamming nibble lists,
3. per-block candidates,
4. global block beam,
5. final CRC candidate window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = WEAK_ROOT / "scripts" / "experiments"
SAVAUX_RUNNER_DIR = EXPERIMENTS_DIR / "baselines" / "savaux_oversampled"
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(SAVAUX_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(SAVAUX_RUNNER_DIR))

from run_paper_oversampled_baseline import _decode_payload, load_packets as load_savaux_packets  # noqa: E402
from run_savaux_current_threshold_sweep import (  # noqa: E402
    DEFAULT_DATASETS,
    _dataset_paths,
    _payload_reference_power,
)
from run_savaux_codec_sweep import _codec_config, _extract_savaux_spectra  # noqa: E402
from run_symbol_phase_threshold_sweep import _write_csv  # noqa: E402
from weak_decoder.chirp import bin_to_grlora_symbol  # noqa: E402
from weak_decoder.payload_codec import (  # noqa: E402
    explicit_header_tail_nibbles,
    nibbles_to_dewhitened_bytes,
    payload_symbols_to_nibbles,
)
from weak_decoder.phase_guided_demod import PhaseLine  # noqa: E402
from weak_decoder.two_stage_weak_decoder import (  # noqa: E402
    TwoStageWeakConfig,
    _beam_blocks,
    _candidate_evidence_score,
    _rank_diverse_block_states,
    _rank_cost_block_states,
    _raw_bin_change_count,
    _verify_payload_crc,
    decode_two_stage_weak_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace GT path survival inside Savaux+codec decoder.")
    parser.add_argument("--dataset", default="0_0_0_10_14_8")
    parser.add_argument("--snr-db", type=float, default=-23.0)
    parser.add_argument("--packets", nargs="*", type=int, default=None)
    parser.add_argument(
        "--from-metrics",
        type=Path,
        default=None,
        help="Optional per_packet_metrics.csv used to auto-select failed packets.",
    )
    parser.add_argument("--max-auto-packets", type=int, default=8)
    parser.add_argument("--min-topk-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--independent-noise", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "codec_gt_path_diagnostics",
    )

    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol", "none"), default="continuous")
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--paper-origin-shift", type=int, default=None)

    parser.add_argument("--top-k-metrics", type=int, default=64)
    parser.add_argument("--amplitude-floor-db", type=float, default=30.0)
    parser.add_argument("--phase-weight", type=float, default=0.0)
    parser.add_argument("--bit-metric", choices=("max", "logsumexp"), default="max")
    parser.add_argument("--nibble-candidates", type=int, default=16)
    parser.add_argument("--row-beam-width", type=int, default=2048)
    parser.add_argument("--block-candidate-limit", type=int, default=512)
    parser.add_argument("--block-symbol-seed-top-m", type=int, default=0)
    parser.add_argument("--block-symbol-seed-deep-top-l", type=int, default=0)
    parser.add_argument("--block-symbol-seed-max-deep-positions", type=int, default=1)
    parser.add_argument("--block-symbol-seed-quota", type=int, default=0)
    parser.add_argument("--block-symbol-seed-max-combinations", type=int, default=50000)
    parser.add_argument("--global-beam-width", type=int, default=2048)
    parser.add_argument("--global-rank-diverse-top-r", type=int, default=0)
    parser.add_argument("--global-rank-diverse-beam-width", type=int, default=0)
    parser.add_argument("--global-rank-cost-max", type=float, default=0.0)
    parser.add_argument("--global-rank-cost-state-limit", type=int, default=0)
    parser.add_argument("--final-candidate-limit", type=int, default=2048)
    parser.add_argument("--block-trim-fraction", type=float, default=0.20)
    parser.add_argument("--projection-trim-fraction", type=float, default=0.10)
    parser.add_argument("--projection-score-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-score-weight", type=float, default=0.0)
    parser.add_argument("--crc-observed-bonus", type=float, default=6.0)
    parser.add_argument("--crc-candidate-min-evidence-margin", type=float, default=-999.0)
    parser.add_argument("--crc-candidate-max-beam-rank", type=int, default=64)
    parser.add_argument("--crc-candidate-high-rank-max-beam-rank", type=int, default=256)
    parser.add_argument("--crc-candidate-high-rank-min-evidence-margin", type=float, default=4.0)
    parser.add_argument(
        "--crc-selection-policy",
        choices=("best_score", "earliest_rank", "score_unless_low_margin"),
        default="earliest_rank",
    )
    parser.add_argument("--crc-non-earliest-min-evidence-margin", type=float, default=0.50)
    parser.add_argument(
        "--crc-failure-selection-policy",
        choices=("argmax_fallback", "best_score", "small_change_best_score"),
        default="argmax_fallback",
    )
    parser.add_argument("--crc-failure-max-symbol-changes", type=int, default=2)
    parser.add_argument("--crc-failure-min-evidence-margin", type=float, default=-0.50)
    parser.add_argument("--crc-state-search-block-top-r", type=int, default=0)
    parser.add_argument("--crc-state-search-state-limit", type=int, default=20000)
    parser.add_argument("--crc-state-search-keep-per-key", type=int, default=1)
    parser.add_argument("--crc-state-search-max-candidates", type=int, default=256)
    parser.add_argument("--crc-state-search-min-evidence-margin", type=float, default=0.0)
    parser.add_argument("--disable-argmax-fallback", action="store_true")

    # Path-likelihood knobs are present only so _extract_savaux_spectra can use
    # the same namespace as the sweep runner.  They stay disabled by default.
    parser.add_argument("--adaptive-path-top-k", type=int, default=0)
    parser.add_argument("--timing-path-top-k", type=int, default=0)
    parser.add_argument("--structured-path-top-k", type=int, default=0)
    return parser.parse_args()


def _dataset_index(dataset: str) -> int:
    try:
        return list(DEFAULT_DATASETS).index(str(dataset))
    except ValueError:
        return 0


def _noisy_samples(samples: np.ndarray, reference_power: float, args: argparse.Namespace) -> np.ndarray:
    dataset_seed = int(args.seed) + 100000 * _dataset_index(str(args.dataset))
    noise_power = float(reference_power) * (10.0 ** (-float(args.snr_db) / 10.0))
    sigma = math.sqrt(float(noise_power) / 2.0)
    rng = np.random.default_rng(dataset_seed)
    if bool(args.independent_noise):
        rng = np.random.default_rng(dataset_seed + int(round(float(args.snr_db) * 1000.0)))
    noise_i = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    noise_q = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)
    return (samples + sigma * unit_noise).astype(np.complex64, copy=False)


def _auto_packets(args: argparse.Namespace) -> list[int]:
    if args.from_metrics is None:
        return []
    with args.from_metrics.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: list[int] = []
    for row in rows:
        if str(row.get("dataset", "")) != str(args.dataset):
            continue
        try:
            snr = float(row.get("target_snr_db", "nan"))
        except ValueError:
            continue
        if abs(snr - float(args.snr_db)) > 1e-9:
            continue
        try:
            topk = float(row.get("savaux_codec_topk_recall", 0.0))
            paper_crc = int(float(row.get("savaux_paper_crc_valid", 0)))
            codec_crc = int(float(row.get("savaux_codec_crc_valid", 0)))
        except ValueError:
            continue
        if topk >= float(args.min_topk_recall) and paper_crc == 0 and codec_crc == 0:
            selected.append(int(float(row["packet_index"])))
        if len(selected) >= int(args.max_auto_packets):
            break
    return selected


def _raw_rank(raw_scores: np.ndarray, raw_bin: int) -> int:
    raw_bin = int(raw_bin)
    if raw_bin < 0 or raw_bin >= raw_scores.size:
        return -1
    order = np.argsort(raw_scores)[::-1]
    matches = np.where(order == raw_bin)[0]
    return int(matches[0] + 1) if matches.size else -1


def _value_rank(score_by_value: np.ndarray, value: int) -> int:
    value = int(value)
    if value < 0 or value >= score_by_value.size:
        return -1
    finite = np.isfinite(score_by_value)
    if not finite[value]:
        return -1
    order = np.argsort(score_by_value)[::-1]
    matches = np.where(order == value)[0]
    return int(matches[0] + 1) if matches.size else -1


def _find_tuple_rank(values: Sequence[int], candidates: Sequence[Any], attr: str) -> int:
    target = tuple(int(v) for v in values)
    for index, candidate in enumerate(candidates):
        if tuple(int(v) for v in getattr(candidate, attr)) == target:
            return int(index + 1)
    return -1


def _find_tuple_candidate(values: Sequence[int], candidates: Sequence[Any], attr: str) -> Any | None:
    target = tuple(int(v) for v in values)
    for candidate in candidates:
        if tuple(int(v) for v in getattr(candidate, attr)) == target:
            return candidate
    return None


def _diagnose_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    config: TwoStageWeakConfig,
) -> dict[str, Any]:
    sf = int(packet["sf"])
    cr = int(packet["cr"])
    ldro = bool(packet["ldro"])
    spectra, raw_score_overrides, gt_bins_full = _extract_savaux_spectra(samples, packet, args)
    hard_bins = [int(np.argmax(score)) for score in raw_score_overrides]
    hard_decode = _decode_payload(packet, hard_bins, args.crc_mode, args.ldro_mode)
    gt_decode = _decode_payload(packet, gt_bins_full, args.crc_mode, args.ldro_mode)

    result = decode_two_stage_weak_payload(
        payload_spectra=spectra,
        header_symbol_values=tuple(packet["header_symbols"]),
        phase_line=PhaseLine(),
        payload_symbol_start_abs_index=12.25 + 8.0,
        sf=sf,
        cr=cr,
        payload_len=int(packet["payload_len"]),
        has_crc=bool(packet["has_crc"]),
        ldro=ldro,
        config=config,
        raw_score_overrides=raw_score_overrides,
        trajectory_spectra=spectra,
    )

    n = min(len(result.likelihoods), len(gt_bins_full))
    gt_bins = tuple(int(v) for v in gt_bins_full[:n])
    gt_symbols = tuple(
        int(bin_to_grlora_symbol(raw_bin, sf=sf, is_header=False, ldro=ldro))
        for raw_bin in gt_bins
    )
    cw_len = cr + 4
    full_symbol_count = (len(gt_symbols) // cw_len) * cw_len
    gt_symbols = gt_symbols[:full_symbol_count]
    gt_bins = gt_bins[:full_symbol_count]

    ranks = [_raw_rank(raw_score_overrides[idx], gt_bins[idx]) for idx in range(len(gt_bins))]
    topk_hits = sum(int(rank > 0 and rank <= int(config.top_k_metrics)) for rank in ranks)

    gt_payload_nibbles = tuple(
        int(v) for v in payload_symbols_to_nibbles(gt_symbols, sf=sf, cr=cr, ldro=ldro)
    )
    block_size = int(sf) - 2 if ldro else int(sf)
    block_rows: list[dict[str, Any]] = []
    missing_rows: list[str] = []
    true_block_candidates: list[Any | None] = []
    for block_index, block_candidates in enumerate(result.block_candidates):
        sym_start = block_index * cw_len
        nib_start = block_index * block_size
        true_block_symbols = gt_symbols[sym_start:sym_start + cw_len]
        true_block_nibbles = gt_payload_nibbles[nib_start:nib_start + block_size]
        block_symbol_rank = _find_tuple_rank(true_block_symbols, block_candidates, "symbol_values")
        block_nibble_rank = _find_tuple_rank(true_block_nibbles, block_candidates, "nibbles")
        true_block_candidates.append(_find_tuple_candidate(true_block_nibbles, block_candidates, "nibbles"))
        true_value_ranks = tuple(
            _value_rank(result.likelihoods[sym_start + offset].score_by_value, symbol_value)
            for offset, symbol_value in enumerate(true_block_symbols)
            if sym_start + offset < len(result.likelihoods)
        )
        seed_count = sum(int(getattr(candidate, "source", "") == "symbol_seed") for candidate in block_candidates)
        row_count = sum(int(getattr(candidate, "source", "") != "symbol_seed") for candidate in block_candidates)

        row_present = 0
        first_missing = -1
        for row_offset, true_nibble in enumerate(true_block_nibbles):
            list_index = block_index * block_size + row_offset
            if list_index >= len(result.codeword_lists):
                first_missing = row_offset if first_missing < 0 else first_missing
                continue
            codeword_list = result.codeword_lists[list_index]
            present = any(int(candidate.nibble) == int(true_nibble) for candidate in codeword_list.candidates)
            row_present += int(present)
            if not present and first_missing < 0:
                first_missing = row_offset
        if first_missing >= 0:
            missing_rows.append(f"{block_index}:{first_missing}")
        block_rows.append(
            {
                "block": block_index,
                "symbol_rank": block_symbol_rank,
                "nibble_rank": block_nibble_rank,
                "row_present": row_present,
                "row_total": len(true_block_nibbles),
                "first_missing_row": first_missing,
                "candidate_count": len(block_candidates),
                "seed_candidate_count": int(seed_count),
                "row_candidate_count": int(row_count),
                "value_ranks": true_value_ranks,
            }
        )

    block_beam = _beam_blocks(result.block_candidates, config.global_beam_width)
    rank_diverse_states = _rank_diverse_block_states(
        result.block_candidates,
        top_r=int(getattr(config, "global_rank_diverse_top_r", 0)),
        beam_width=int(getattr(config, "global_rank_diverse_beam_width", 0) or getattr(config, "global_beam_width", 64)),
    )
    rank_cost_states = _rank_cost_block_states(
        result.block_candidates,
        top_r=int(getattr(config, "global_rank_diverse_top_r", 0)),
        max_cost=float(getattr(config, "global_rank_cost_max", 0.0)),
        state_limit=int(getattr(config, "global_rank_cost_state_limit", 0)),
    )
    global_symbol_rank = _find_tuple_rank(gt_symbols, block_beam, "symbol_values")
    global_nibble_rank = _find_tuple_rank(gt_payload_nibbles[: len(gt_symbols) // cw_len * block_size], block_beam, "nibbles")
    rank_diverse_symbol_rank = _find_tuple_rank(gt_symbols, rank_diverse_states, "symbol_values")
    rank_diverse_nibble_rank = _find_tuple_rank(
        gt_payload_nibbles[: len(gt_symbols) // cw_len * block_size],
        rank_diverse_states,
        "nibbles",
    )
    rank_cost_symbol_rank = _find_tuple_rank(gt_symbols, rank_cost_states, "symbol_values")
    rank_cost_nibble_rank = _find_tuple_rank(
        gt_payload_nibbles[: len(gt_symbols) // cw_len * block_size],
        rank_cost_states,
        "nibbles",
    )
    final_window_rank = global_symbol_rank if 0 < global_symbol_rank <= int(config.final_candidate_limit) else -1

    selected = result.selected
    selected_payload_hex = selected.payload_bytes.hex() if selected is not None else ""
    gt_payload_hex = str(gt_decode.get("payload_hex", ""))
    candidate_payload_rank = -1
    candidate_symbol_rank = -1
    candidate_crc_valid = 0
    for index, candidate in enumerate(result.candidates):
        if candidate_payload_rank < 0 and candidate.payload_bytes.hex() == gt_payload_hex:
            candidate_payload_rank = int(index + 1)
            candidate_crc_valid = int(candidate.observed_crc_valid)
        if candidate_symbol_rank < 0 and tuple(candidate.payload_symbol_values[: len(gt_symbols)]) == tuple(gt_symbols):
            candidate_symbol_rank = int(index + 1)
    selected_raw = tuple(selected.raw_bins) if selected is not None else ()
    selected_change_count = _raw_bin_change_count(selected_raw, hard_bins) if selected is not None else 0

    exact_blocks_available = all(candidate is not None for candidate in true_block_candidates) and bool(true_block_candidates)
    exact_block_payload_hex = ""
    exact_block_crc_valid = 0
    exact_block_payload_matches_gt = 0
    exact_block_rank_cost = -1.0
    exact_block_score = 0.0
    if exact_blocks_available:
        exact_nibbles: list[int] = []
        exact_score = 0.0
        exact_rank_cost = 0.0
        for block_index, candidate in enumerate(true_block_candidates):
            if candidate is None:
                continue
            exact_nibbles.extend(int(v) for v in candidate.nibbles)
            exact_score += float(candidate.total_score)
            rank = _find_tuple_rank(candidate.nibbles, result.block_candidates[block_index], "nibbles")
            if rank > 0:
                exact_rank_cost += math.log2(float(rank))
        try:
            header_tail = explicit_header_tail_nibbles(packet["header_symbols"], sf=sf)
            exact_payload, exact_crc = nibbles_to_dewhitened_bytes(
                tuple(header_tail) + tuple(exact_nibbles),
                payload_len=int(packet["payload_len"]),
                has_crc=bool(packet["has_crc"]),
            )
            exact_crc_ok, _exact_crc_computed, _exact_crc_received = _verify_payload_crc(
                exact_payload,
                exact_crc,
                has_crc=bool(packet["has_crc"]),
                crc_mode=str(args.crc_mode),
            )
            exact_block_payload_hex = exact_payload.hex()
            exact_block_crc_valid = int(exact_crc_ok)
            exact_block_payload_matches_gt = int(exact_block_payload_hex == gt_payload_hex)
            exact_block_rank_cost = float(exact_rank_cost)
            exact_block_score = float(exact_score)
        except Exception:
            exact_block_payload_hex = ""

    return {
        "dataset": str(args.dataset),
        "target_snr_db": float(args.snr_db),
        "packet_index": int(packet["packet_index"]),
        "payload_len": int(packet["payload_len"]),
        "payload_symbol_count": int(len(gt_symbols)),
        "hard_crc_valid": int(hard_decode["crc_valid"]),
        "gt_crc_valid": int(gt_decode["crc_valid"]),
        "selected_crc_valid": int(selected.observed_crc_valid) if selected is not None else 0,
        "selected_source": selected.selection_source if selected is not None else "",
        "selected_payload_matches_gt": int(selected_payload_hex == gt_payload_hex),
        "selected_change_count": int(selected_change_count),
        "observed_crc_valid_count": int(result.metrics.get("observed_crc_valid_count", 0)),
        "accepted_crc_valid_count": int(result.metrics.get("accepted_crc_valid_count", 0)),
        "rank_diverse_state_count": int(result.metrics.get("rank_diverse_state_count", 0)),
        "rank_cost_state_count": int(result.metrics.get("rank_cost_state_count", 0)),
        "crc_state_state_count": int(result.metrics.get("crc_state_state_count", 0)),
        "final_state_count": int(result.metrics.get("final_state_count", 0)),
        "topk_recall": float(topk_hits / len(gt_bins)) if gt_bins else 0.0,
        "mean_gt_raw_rank": float(np.mean([rank for rank in ranks if rank > 0])) if any(rank > 0 for rank in ranks) else 0.0,
        "max_gt_raw_rank": int(max(ranks)) if ranks else -1,
        "row_gt_present_count": int(sum(int(item["row_present"]) for item in block_rows)),
        "row_gt_total": int(sum(int(item["row_total"]) for item in block_rows)),
        "all_rows_present": int(all(item["row_present"] == item["row_total"] for item in block_rows)),
        "missing_rows": " ".join(missing_rows),
        "block_gt_present_count": int(sum(int(item["symbol_rank"] > 0) for item in block_rows)),
        "block_count": int(len(block_rows)),
        "all_blocks_present": int(all(item["symbol_rank"] > 0 for item in block_rows)),
        "block_symbol_ranks": " ".join(str(item["symbol_rank"]) for item in block_rows),
        "block_nibble_ranks": " ".join(str(item["nibble_rank"]) for item in block_rows),
        "block_gt_value_ranks": ";".join(
            ",".join(str(rank) for rank in item["value_ranks"]) for item in block_rows
        ),
        "missing_block_gt_value_ranks": ";".join(
            f'{item["block"]}:{",".join(str(rank) for rank in item["value_ranks"])}'
            for item in block_rows
            if int(item["symbol_rank"]) < 0
        ),
        "block_seed_candidate_counts": " ".join(str(item["seed_candidate_count"]) for item in block_rows),
        "block_row_candidate_counts": " ".join(str(item["row_candidate_count"]) for item in block_rows),
        "global_symbol_rank": int(global_symbol_rank),
        "global_nibble_rank": int(global_nibble_rank),
        "rank_diverse_symbol_rank": int(rank_diverse_symbol_rank),
        "rank_diverse_nibble_rank": int(rank_diverse_nibble_rank),
        "rank_cost_symbol_rank": int(rank_cost_symbol_rank),
        "rank_cost_nibble_rank": int(rank_cost_nibble_rank),
        "gt_in_final_window": int(final_window_rank > 0),
        "final_window_rank": int(final_window_rank),
        "candidate_payload_rank_after_sort": int(candidate_payload_rank),
        "candidate_symbol_rank_after_sort": int(candidate_symbol_rank),
        "candidate_gt_crc_valid": int(candidate_crc_valid),
        "exact_blocks_available": int(exact_blocks_available),
        "exact_block_payload_matches_gt": int(exact_block_payload_matches_gt),
        "exact_block_crc_valid": int(exact_block_crc_valid),
        "exact_block_rank_cost": float(exact_block_rank_cost),
        "exact_block_score": float(exact_block_score),
        "exact_block_payload_hex": exact_block_payload_hex,
        "selected_evidence_margin": float(result.metrics.get("selected_evidence_margin", 0.0)),
        "gt_payload_hex": gt_payload_hex,
        "selected_payload_hex": selected_payload_hex,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = _dataset_paths(str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    reference_power, _sample_count, _packet_count = _payload_reference_power(samples, paths["symbols"])
    noisy = _noisy_samples(samples, reference_power, args)
    packets = load_savaux_packets(paths["symbols"], packet_filter=None, max_packets=None)
    by_packet = {int(packet["packet_index"]): packet for packet in packets}

    packet_ids = list(args.packets or [])
    if not packet_ids:
        packet_ids = _auto_packets(args)
    if not packet_ids:
        packet_ids = sorted(by_packet)[: int(args.max_auto_packets)]

    config = _codec_config(args)
    rows = [
        _diagnose_packet(noisy, by_packet[int(packet_id)], args, config)
        for packet_id in packet_ids
        if int(packet_id) in by_packet
    ]
    stem = f"{args.dataset}_snr{float(args.snr_db):+.1f}".replace(".", "p").replace("+", "p").replace("-", "m")
    out_csv = args.output_dir / f"{stem}_gt_path_diagnostics.csv"
    _write_csv(out_csv, rows)
    (args.output_dir / f"{stem}_gt_path_diagnostics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"diagnosed_packets={len(rows)}")
    print(f"wrote={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired sweep for Savaux OSR spectra plus the local LoRa codec/CRC beam.

This is intentionally not a paper baseline: the paper-only implementation lives
under baselines/savaux_oversampled and still reports the hard OSR argmax.  This
script asks a separate engineering question: if the Savaux OSR periodogram often
contains the right bin in its Top-K list, can the local LoRa PHY constraints and
CRC recover packets that the hard argmax misses?
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
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

from run_paper_oversampled_baseline import (  # noqa: E402
    _decode_payload,
    _ser,
    load_packets as load_savaux_packets,
)
from run_savaux_current_threshold_sweep import (  # noqa: E402
    DEFAULT_DATASETS,
    _current_default_args,
    _dataset_paths,
    _mean_of_datasets,
    _payload_reference_power,
    _threshold_tables,
)
from run_symbol_phase_threshold_sweep import (  # noqa: E402
    _evaluate_packet_methods as evaluate_current_packet,
    _snr_values,
    _threshold_from_curve,
    _write_csv,
)
from run_symbol_phase_two_stage import build_config  # noqa: E402
from run_two_stage_weak_decoder import load_packets as load_current_packets  # noqa: E402
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.adaptive_path_demod import score_adaptive_path_candidates  # noqa: E402
from weak_decoder.chirp import bin_to_grlora_symbol  # noqa: E402
from weak_decoder.phase_guided_demod import PhaseLine  # noqa: E402
from weak_decoder.structured_path_demod import score_structured_path_candidates  # noqa: E402
from weak_decoder.timing_path_demod import score_timing_path_candidates  # noqa: E402
from weak_decoder.two_stage_weak_decoder import (  # noqa: E402
    TwoStageWeakConfig,
    decode_two_stage_weak_payload,
    summarize_with_ground_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Savaux OSR hard argmax against Savaux+LoRa codec/CRC beam."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snr-start", type=float, default=-22.0)
    parser.add_argument("--snr-stop", type=float, default=-26.0)
    parser.add_argument("--snr-step", type=float, default=-1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "savaux_codec_sweep",
    )
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--independent-noise", action="store_true")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol", "none"), default="continuous")
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--paper-origin-shift", type=int, default=None)

    parser.add_argument("--top-k-metrics", type=int, default=64)
    parser.add_argument("--amplitude-floor-db", type=float, default=30.0)
    parser.add_argument("--phase-weight", type=float, default=0.0)
    parser.add_argument("--bit-metric", choices=("max", "logsumexp"), default="max")
    parser.add_argument("--nibble-candidates", type=int, default=8)
    parser.add_argument("--row-beam-width", type=int, default=512)
    parser.add_argument("--block-candidate-limit", type=int, default=128)
    parser.add_argument(
        "--block-symbol-seed-top-m",
        type=int,
        default=0,
        help="0 disables symbol-tuple block seeds; >1 decodes Top-M symbol tuples into extra block candidates",
    )
    parser.add_argument(
        "--block-symbol-seed-deep-top-l",
        type=int,
        default=0,
        help="optional one-outlier seed depth; each block may expand selected positions from Top-M to Top-L",
    )
    parser.add_argument(
        "--block-symbol-seed-max-deep-positions",
        type=int,
        default=1,
        help="maximum number of symbol positions expanded to Top-L at once",
    )
    parser.add_argument(
        "--block-symbol-seed-quota",
        type=int,
        default=0,
        help="reserve this many per-block candidates for symbol-tuple seeds after deduplication",
    )
    parser.add_argument(
        "--block-symbol-seed-max-combinations",
        type=int,
        default=50000,
        help="cap Top-M tuple combinations per block; effective M is reduced when needed",
    )
    parser.add_argument("--global-beam-width", type=int, default=512)
    parser.add_argument(
        "--global-rank-diverse-top-r",
        type=int,
        default=0,
        help="0 disables rank-diverse final states; >0 appends low-rank block diagonal patterns to CRC evaluation",
    )
    parser.add_argument("--global-rank-diverse-beam-width", type=int, default=0)
    parser.add_argument("--global-rank-cost-max", type=float, default=0.0)
    parser.add_argument("--global-rank-cost-state-limit", type=int, default=0)
    parser.add_argument("--final-candidate-limit", type=int, default=512)
    parser.add_argument("--block-trim-fraction", type=float, default=0.20)
    parser.add_argument("--projection-trim-fraction", type=float, default=0.10)
    parser.add_argument("--projection-score-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-score-weight", type=float, default=0.0)
    parser.add_argument("--crc-observed-bonus", type=float, default=6.0)
    parser.add_argument(
        "--crc-candidate-min-evidence-margin",
        type=float,
        default=-999.0,
        help="negative values accept CRC-valid beam candidates below hard-argmax evidence",
    )
    parser.add_argument(
        "--crc-candidate-max-beam-rank",
        type=int,
        default=64,
        help="accept CRC-valid codec rescues only from early beam ranks; 64 preserved validated rescues and rejected seed-search false positives in OS=4 probes",
    )
    parser.add_argument(
        "--crc-candidate-high-rank-max-beam-rank",
        type=int,
        default=256,
        help="allow later CRC-valid rescues up to this rank only when the high-rank evidence margin gate is also satisfied; negative disables the high-rank cap",
    )
    parser.add_argument(
        "--crc-candidate-high-rank-min-evidence-margin",
        type=float,
        default=4.0,
        help="extra evidence margin required for CRC-valid rescues beyond --crc-candidate-max-beam-rank",
    )
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
    parser.add_argument(
        "--adaptive-path-top-k",
        type=int,
        default=0,
        help="0 disables adaptive path likelihood; >0 applies path bonus to Savaux Top-K bins before codec beam",
    )
    parser.add_argument("--adaptive-switch-penalty", type=float, default=0.40)
    parser.add_argument("--adaptive-step-penalty", type=float, default=0.10)
    parser.add_argument("--adaptive-path-gain-power", type=float, default=0.18)
    parser.add_argument("--adaptive-switch-penalty-power", type=float, default=0.40)
    parser.add_argument(
        "--adaptive-score-mix",
        type=float,
        default=1.0,
        help="1 uses path-composite score on Top-K bins; 0 leaves pure Savaux log power",
    )
    parser.add_argument(
        "--timing-path-top-k",
        type=int,
        default=0,
        help="0 disables timing-path likelihood; >0 applies linear fractional timing rerank to Top-K bins",
    )
    parser.add_argument("--timing-tau-grid", default="-0.5,0,0.5")
    parser.add_argument("--timing-slope-grid", default="-0.5,0,0.5")
    parser.add_argument("--timing-path-gain-power", type=float, default=0.20)
    parser.add_argument("--timing-slope-penalty-power", type=float, default=0.10)
    parser.add_argument("--timing-score-mix", type=float, default=0.25)
    parser.add_argument(
        "--structured-path-top-k",
        type=int,
        default=0,
        help="0 disables structured non-uniform path ensemble; >0 scores Savaux Top-K bins with path evidence",
    )
    parser.add_argument("--structured-path-ratio-power", type=float, default=0.20)
    parser.add_argument(
        "--structured-score-mix",
        type=float,
        default=0.20,
        help="bounded log-bonus weight for structured path evidence on Top-K bins",
    )
    parser.add_argument(
        "--structured-score-clip-db",
        type=float,
        default=1.5,
        help="clip structured path bonus magnitude so noisy paths cannot dominate Savaux power",
    )
    return parser.parse_args()


def _float_grid(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(text).split(",") if item.strip() != "")


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def _codec_config(args: argparse.Namespace) -> TwoStageWeakConfig:
    return TwoStageWeakConfig(
        top_k_metrics=int(args.top_k_metrics),
        amplitude_floor_db=float(args.amplitude_floor_db),
        phase_weight=float(args.phase_weight),
        bit_metric=str(args.bit_metric),
        nibble_candidates_per_codeword=int(args.nibble_candidates),
        row_beam_width=int(args.row_beam_width),
        block_candidate_limit=int(args.block_candidate_limit),
        block_symbol_seed_top_m=int(args.block_symbol_seed_top_m),
        block_symbol_seed_deep_top_l=int(args.block_symbol_seed_deep_top_l),
        block_symbol_seed_max_deep_positions=int(args.block_symbol_seed_max_deep_positions),
        block_symbol_seed_quota=int(args.block_symbol_seed_quota),
        block_symbol_seed_max_combinations=int(args.block_symbol_seed_max_combinations),
        global_beam_width=int(args.global_beam_width),
        global_rank_diverse_top_r=int(args.global_rank_diverse_top_r),
        global_rank_diverse_beam_width=int(args.global_rank_diverse_beam_width),
        global_rank_cost_max=float(args.global_rank_cost_max),
        global_rank_cost_state_limit=int(args.global_rank_cost_state_limit),
        final_candidate_limit=int(args.final_candidate_limit),
        block_trim_fraction=float(args.block_trim_fraction),
        projection_trim_fraction=float(args.projection_trim_fraction),
        projection_score_weight=float(args.projection_score_weight),
        trajectory_score_weight=float(args.trajectory_score_weight),
        crc_observed_bonus=float(args.crc_observed_bonus),
        crc_candidate_min_evidence_margin=float(args.crc_candidate_min_evidence_margin),
        crc_candidate_max_beam_rank=int(args.crc_candidate_max_beam_rank),
        crc_candidate_high_rank_max_beam_rank=int(args.crc_candidate_high_rank_max_beam_rank),
        crc_candidate_high_rank_min_evidence_margin=float(args.crc_candidate_high_rank_min_evidence_margin),
        crc_selection_policy=str(args.crc_selection_policy),
        crc_non_earliest_min_evidence_margin=float(args.crc_non_earliest_min_evidence_margin),
        crc_failure_selection_policy=str(args.crc_failure_selection_policy),
        crc_failure_max_symbol_changes=int(args.crc_failure_max_symbol_changes),
        crc_failure_min_evidence_margin=float(args.crc_failure_min_evidence_margin),
        crc_state_search_block_top_r=int(args.crc_state_search_block_top_r),
        crc_state_search_state_limit=int(args.crc_state_search_state_limit),
        crc_state_search_keep_per_key=int(args.crc_state_search_keep_per_key),
        crc_state_search_max_candidates=int(args.crc_state_search_max_candidates),
        crc_state_search_min_evidence_margin=float(args.crc_state_search_min_evidence_margin),
        crc_mode=str(args.crc_mode),
        argmax_fallback_on_crc_failure=not bool(args.disable_argmax_fallback),
    )


def _path_likelihood_enabled(args: argparse.Namespace) -> bool:
    return (
        int(getattr(args, "adaptive_path_top_k", 0)) > 0
        or int(getattr(args, "timing_path_top_k", 0)) > 0
        or int(getattr(args, "structured_path_top_k", 0)) > 0
    )


def _without_path_likelihood(args: argparse.Namespace) -> argparse.Namespace:
    out = copy.copy(args)
    setattr(out, "adaptive_path_top_k", 0)
    setattr(out, "timing_path_top_k", 0)
    setattr(out, "structured_path_top_k", 0)
    return out


def _decode_from_scores(
    spectra: Sequence[np.ndarray],
    raw_score_overrides: Sequence[np.ndarray],
    packet: dict[str, Any],
    config: TwoStageWeakConfig,
):
    return decode_two_stage_weak_payload(
        payload_spectra=spectra,
        header_symbol_values=tuple(packet["header_symbols"]),
        phase_line=PhaseLine(),
        payload_symbol_start_abs_index=12.25 + 8.0,
        sf=int(packet["sf"]),
        cr=int(packet["cr"]),
        payload_len=int(packet["payload_len"]),
        has_crc=bool(packet["has_crc"]),
        ldro=bool(packet["ldro"]),
        config=config,
        raw_score_overrides=raw_score_overrides,
        trajectory_spectra=spectra,
    )


def _extract_savaux_spectra(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    origin_shift = int(args.paper_origin_shift) if args.paper_origin_shift is not None else os_factor // 2
    paper_header_start = int(packet["header_start_sample"]) + origin_shift

    spectra: list[np.ndarray] = []
    raw_score_overrides: list[np.ndarray] = []
    gt_bins: list[int] = []
    for symbol in packet["payload_symbols"]:
        start_sample = int(symbol["start_sample"]) + origin_shift
        spectrum, _branches, _phase = paper_oversampled_spectrum(
            samples=samples,
            start_sample=start_sample,
            sf=sf,
            os_factor=os_factor,
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=paper_header_start,
            cfo_correction_mode=str(args.cfo_correction_mode),
        )
        power = np.abs(spectrum).astype(np.float64) ** 2
        raw_score = np.log(power + 1e-30)
        if int(getattr(args, "adaptive_path_top_k", 0)) > 0:
            top_bins = np.argsort(power)[::-1][: int(args.adaptive_path_top_k)]
            adaptive_candidates, _paths = score_adaptive_path_candidates(
                samples=samples,
                start_sample=start_sample,
                sf=sf,
                os_factor=os_factor,
                candidate_bins=tuple(int(v) for v in top_bins),
                cfo_int=int(packet["cfo_int"]),
                cfo_frac=float(packet["cfo_frac"]),
                header_start_sample=paper_header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
                switch_penalty=float(args.adaptive_switch_penalty),
                step_penalty=float(args.adaptive_step_penalty),
                path_gain_power=float(args.adaptive_path_gain_power),
                switch_penalty_power=float(args.adaptive_switch_penalty_power),
                savaux_power=power,
            )
            mix = max(0.0, min(1.0, float(args.adaptive_score_mix)))
            for item in adaptive_candidates:
                raw_bin = int(item.raw_fft_bin)
                path_score = math.log(float(item.composite_score) + 1e-30)
                raw_score[raw_bin] = (1.0 - mix) * float(raw_score[raw_bin]) + mix * path_score
        if int(getattr(args, "timing_path_top_k", 0)) > 0:
            top_bins = np.argsort(power)[::-1][: int(args.timing_path_top_k)]
            timing_candidates = score_timing_path_candidates(
                samples=samples,
                start_sample=start_sample,
                sf=sf,
                os_factor=os_factor,
                candidate_bins=tuple(int(v) for v in top_bins),
                cfo_int=int(packet["cfo_int"]),
                cfo_frac=float(packet["cfo_frac"]),
                header_start_sample=paper_header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
                tau_grid=_float_grid(args.timing_tau_grid),
                slope_grid=_float_grid(args.timing_slope_grid),
                path_gain_power=float(args.timing_path_gain_power),
                slope_penalty_power=float(args.timing_slope_penalty_power),
                savaux_power=power,
            )
            mix = max(0.0, min(1.0, float(args.timing_score_mix)))
            if timing_candidates:
                timing_scores = np.asarray(
                    [math.log(float(item.timing_path_power) + 1e-30) for item in timing_candidates],
                    dtype=np.float64,
                )
                timing_scores = timing_scores - float(np.max(timing_scores))
                for item, timing_score in zip(timing_candidates, timing_scores):
                    raw_bin = int(item.raw_fft_bin)
                    raw_score[raw_bin] = float(raw_score[raw_bin]) + mix * float(timing_score)
        if int(getattr(args, "structured_path_top_k", 0)) > 0:
            top_bins = np.argsort(power)[::-1][: int(args.structured_path_top_k)]
            structured_candidates = score_structured_path_candidates(
                samples=samples,
                start_sample=start_sample,
                sf=sf,
                os_factor=os_factor,
                candidate_bins=tuple(int(v) for v in top_bins),
                cfo_int=int(packet["cfo_int"]),
                cfo_frac=float(packet["cfo_frac"]),
                header_start_sample=paper_header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
                path_ratio_power=float(args.structured_path_ratio_power),
                savaux_power=power,
            )
            if structured_candidates:
                mix = max(0.0, min(1.0, float(args.structured_score_mix)))
                clip = math.log(10.0) * max(0.0, float(args.structured_score_clip_db)) / 10.0
                for item in structured_candidates:
                    raw_bin = int(item.raw_fft_bin)
                    path_log = math.log(float(item.composite_score) + 1e-30)
                    bonus = path_log - float(raw_score[raw_bin])
                    bonus = max(-clip, min(clip, bonus))
                    raw_score[raw_bin] = float(raw_score[raw_bin]) + mix * bonus
        spectra.append(np.asarray(spectrum, dtype=np.complex64))
        raw_score_overrides.append(raw_score.astype(np.float64))
        gt_bins.append(int(symbol.get("gt_raw_fft_bin", -1)))
    return spectra, raw_score_overrides, gt_bins


def _decode_symbols(packet: dict[str, Any], raw_bins: Sequence[int], args: argparse.Namespace) -> dict[str, Any]:
    return _decode_payload(packet, raw_bins, args.crc_mode, args.ldro_mode)


def _evaluate_savaux_codec_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    config: TwoStageWeakConfig,
) -> dict[str, Any]:
    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    base_args = _without_path_likelihood(args)
    spectra, raw_score_overrides, gt_bins = _extract_savaux_spectra(samples, packet, base_args)
    hard_bins = [int(np.argmax(np.abs(spectrum).astype(np.float64) ** 2)) for spectrum in spectra]
    hard_symbols = [
        bin_to_grlora_symbol(raw_bin, sf=sf, is_header=False, ldro=ldro)
        for raw_bin in hard_bins
    ]

    hard_raw_ser, hard_symbol_ser, compared = _ser(hard_bins, gt_bins, sf=sf, ldro=ldro)
    hard_decode = _decode_symbols(packet, hard_bins, args)

    if int(hard_decode["crc_valid"]):
        return {
            "savaux_hard_raw_ser": float(hard_raw_ser),
            "savaux_hard_symbol_ser": float(hard_symbol_ser),
            "savaux_codec_raw_ser": float(hard_raw_ser),
            "savaux_codec_symbol_ser": float(hard_symbol_ser),
            "savaux_codec_gain_vs_hard": 0.0,
            "gt_compared_symbols": int(compared),
            "savaux_hard_crc_valid": 1,
            "savaux_codec_crc_valid": 1,
            "success": 1,
            "selected_source": "savaux_hard_crc_preserved",
            "selection_variant": "savaux_hard_crc_preserved",
            "base_codec_crc_valid": 1,
            "selected_beam_rank": -1,
            "candidate_payload_count": 0,
            "observed_crc_valid_count": 0,
            "accepted_crc_valid_count": 0,
            "accepted_crc_min_beam_rank": -1,
            "accepted_crc_max_beam_rank": -1,
            "accepted_crc_best_evidence_margin": 0.0,
            "selected_evidence_margin": 0.0,
            "selected_is_earliest_accepted_crc": 0,
            "selected_source_is_argmax_fallback": 1,
            "topk_recall": 0.0,
            "mean_gt_rank": 0.0,
            "symbol_change_count": 0,
            "symbol_fix_count": 0,
            "symbol_break_count": 0,
            "hard_payload_hex": str(hard_decode["payload_hex"]),
            "selected_payload_hex": str(hard_decode["payload_hex"]),
            "hard_symbol_prefix": " ".join(str(v) for v in hard_symbols[:8]),
        }

    base_result = _decode_from_scores(
        spectra=spectra,
        raw_score_overrides=raw_score_overrides,
        packet=packet,
        config=config,
    )
    result = base_result
    selection_variant = "savaux_codec"
    if _path_likelihood_enabled(args):
        _unused_spectra, path_score_overrides, _unused_gt_bins = _extract_savaux_spectra(
            samples,
            packet,
            args,
        )
        path_result = _decode_from_scores(
            spectra=spectra,
            raw_score_overrides=path_score_overrides,
            packet=packet,
            config=config,
        )
        base_crc_valid = bool(base_result.selected is not None and base_result.selected.observed_crc_valid)
        path_crc_valid = bool(path_result.selected is not None and path_result.selected.observed_crc_valid)
        if (not base_crc_valid) and path_crc_valid:
            result = path_result
            selection_variant = "path_likelihood_rescue"

    gt_summary = summarize_with_ground_truth(
        result,
        gt_raw_bins=gt_bins,
        sf=sf,
        ldro=ldro,
        top_k=int(args.top_k_metrics),
    )
    selected = result.selected
    selected_bins = tuple(int(v) for v in selected.raw_bins) if selected else ()
    selected_decode = {
        "crc_valid": int(selected.observed_crc_valid) if selected else 0,
        "payload_hex": selected.payload_bytes.hex() if selected else "",
    }
    if selected is not None:
        selected_decode["crc_computed"] = f"0x{selected.crc_computed:04x}"
        selected_decode["crc_received"] = f"0x{selected.crc_received:04x}"
    selected_raw_ser, selected_symbol_ser, _ = _ser(selected_bins, gt_bins, sf=sf, ldro=ldro)

    fix_count = 0
    break_count = 0
    change_count = 0
    for hard_bin, selected_bin, gt_bin in zip(hard_bins, selected_bins, gt_bins):
        if int(gt_bin) < 0:
            continue
        if int(hard_bin) != int(selected_bin):
            change_count += 1
        if int(hard_bin) != int(gt_bin) and int(selected_bin) == int(gt_bin):
            fix_count += 1
        if int(hard_bin) == int(gt_bin) and int(selected_bin) != int(gt_bin):
            break_count += 1

    return {
        "savaux_hard_raw_ser": float(hard_raw_ser),
        "savaux_hard_symbol_ser": float(hard_symbol_ser),
        "savaux_codec_raw_ser": float(gt_summary.get("two_stage_raw_ser", selected_raw_ser)),
        "savaux_codec_symbol_ser": float(gt_summary.get("two_stage_symbol_ser", selected_symbol_ser)),
        "savaux_codec_gain_vs_hard": float(hard_symbol_ser - selected_symbol_ser),
        "gt_compared_symbols": int(compared),
        "savaux_hard_crc_valid": int(hard_decode["crc_valid"]),
        "savaux_codec_crc_valid": int(selected_decode["crc_valid"]),
        "success": int(result.success),
        "selected_source": selected.selection_source if selected else "",
        "selection_variant": selection_variant,
        "base_codec_crc_valid": int(base_result.selected is not None and base_result.selected.observed_crc_valid),
        "selected_beam_rank": int(selected.beam_rank) if selected else -1,
        "candidate_payload_count": int(result.metrics.get("candidate_payload_count", 0)),
        "observed_crc_valid_count": int(result.metrics.get("observed_crc_valid_count", 0)),
        "accepted_crc_valid_count": int(result.metrics.get("accepted_crc_valid_count", 0)),
        "accepted_crc_min_beam_rank": int(result.metrics.get("accepted_crc_min_beam_rank", -1)),
        "accepted_crc_max_beam_rank": int(result.metrics.get("accepted_crc_max_beam_rank", -1)),
        "accepted_crc_best_evidence_margin": float(result.metrics.get("accepted_crc_best_evidence_margin", 0.0)),
        "selected_evidence_margin": float(result.metrics.get("selected_evidence_margin", 0.0)),
        "selected_is_earliest_accepted_crc": int(result.metrics.get("selected_is_earliest_accepted_crc", 0)),
        "selected_source_is_argmax_fallback": int(
            result.metrics.get("selected_source_is_argmax_fallback", 0)
        ),
        "topk_recall": float(gt_summary.get("topk_recall", 0.0)),
        "mean_gt_rank": float(gt_summary.get("mean_gt_rank", 0.0)),
        "symbol_change_count": int(change_count),
        "symbol_fix_count": int(fix_count),
        "symbol_break_count": int(break_count),
        "hard_payload_hex": str(hard_decode["payload_hex"]),
        "selected_payload_hex": str(selected_decode["payload_hex"]),
        "hard_symbol_prefix": " ".join(str(v) for v in hard_symbols[:8]),
    }


def _paired_packet_row(
    dataset: str,
    snr_db: float,
    current_row: dict[str, Any],
    codec_row: dict[str, Any],
) -> dict[str, Any]:
    traditional = float(current_row["center_symbol_ser"])
    multi = float(current_row["multi_symbol_ser"])
    current = float(current_row["selected_symbol_ser"])
    savaux = float(codec_row["savaux_hard_symbol_ser"])
    codec = float(codec_row["savaux_codec_symbol_ser"])
    row = {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "packet_index": int(current_row["packet_index"]),
        "frame_index": int(current_row["frame_index"]),
        "event_index": int(current_row["event_index"]),
        "payload_len": int(current_row["payload_len"]),
        "symbol_count": int(current_row["symbol_count"]),
        "traditional_fft_symbol_ser": traditional,
        "multi_offset_argmax_symbol_ser": multi,
        "current_selected_symbol_ser": current,
        "savaux_paper_symbol_ser": savaux,
        "savaux_codec_symbol_ser": codec,
        "current_gain_vs_traditional_fft": traditional - current,
        "savaux_gain_vs_traditional_fft": traditional - savaux,
        "savaux_codec_gain_vs_traditional_fft": traditional - codec,
        "savaux_gain_vs_current": current - savaux,
        "savaux_codec_gain_vs_current": current - codec,
        "savaux_codec_gain_vs_savaux": savaux - codec,
        "traditional_fft_crc_valid": int(current_row["center_crc_valid"]),
        "multi_offset_argmax_crc_valid": int(current_row["multi_crc_valid"]),
        "current_selected_crc_valid": int(current_row["selected_crc_valid"]),
        "savaux_paper_crc_valid": int(codec_row["savaux_hard_crc_valid"]),
        "savaux_codec_crc_valid": int(codec_row["savaux_codec_crc_valid"]),
        "savaux_codec_success": int(codec_row["success"]),
        "savaux_codec_selected_source": str(codec_row["selected_source"]),
        "savaux_codec_selection_variant": str(codec_row["selection_variant"]),
        "savaux_codec_base_crc_valid": int(codec_row["base_codec_crc_valid"]),
        "savaux_codec_selected_beam_rank": int(codec_row["selected_beam_rank"]),
        "savaux_codec_candidate_payload_count": int(codec_row["candidate_payload_count"]),
        "savaux_codec_observed_crc_valid_count": int(codec_row["observed_crc_valid_count"]),
        "savaux_codec_accepted_crc_valid_count": int(codec_row["accepted_crc_valid_count"]),
        "savaux_codec_accepted_crc_min_beam_rank": int(codec_row["accepted_crc_min_beam_rank"]),
        "savaux_codec_accepted_crc_max_beam_rank": int(codec_row["accepted_crc_max_beam_rank"]),
        "savaux_codec_accepted_crc_best_evidence_margin": float(codec_row["accepted_crc_best_evidence_margin"]),
        "savaux_codec_selected_evidence_margin": float(codec_row["selected_evidence_margin"]),
        "savaux_codec_selected_is_earliest_accepted_crc": int(codec_row["selected_is_earliest_accepted_crc"]),
        "savaux_codec_selected_source_is_argmax_fallback": int(
            codec_row["selected_source_is_argmax_fallback"]
        ),
        "savaux_codec_topk_recall": float(codec_row["topk_recall"]),
        "savaux_codec_mean_gt_rank": float(codec_row["mean_gt_rank"]),
        "savaux_codec_symbol_change_count": int(codec_row["symbol_change_count"]),
        "savaux_codec_symbol_fix_count": int(codec_row["symbol_fix_count"]),
        "savaux_codec_symbol_break_count": int(codec_row["symbol_break_count"]),
    }
    values = {
        "traditional_fft": traditional,
        "multi_offset_argmax": multi,
        "current_selected": current,
        "savaux_paper": savaux,
        "savaux_codec": codec,
    }
    row["winner_by_symbol_ser"] = min(values, key=values.get)
    return row


def _summarize(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float) -> dict[str, Any]:
    traditional = _avg(rows, "traditional_fft_symbol_ser")
    multi = _avg(rows, "multi_offset_argmax_symbol_ser")
    current = _avg(rows, "current_selected_symbol_ser")
    savaux = _avg(rows, "savaux_paper_symbol_ser")
    codec = _avg(rows, "savaux_codec_symbol_ser")
    return {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "packet_count": int(len(rows)),
        "traditional_fft_symbol_ser": traditional,
        "traditional_fft_symbol_accuracy": 1.0 - traditional,
        "traditional_fft_crc_valid_rate": _avg(rows, "traditional_fft_crc_valid"),
        "multi_offset_argmax_symbol_ser": multi,
        "multi_offset_argmax_symbol_accuracy": 1.0 - multi,
        "multi_offset_argmax_crc_valid_rate": _avg(rows, "multi_offset_argmax_crc_valid"),
        "current_selected_symbol_ser": current,
        "current_selected_symbol_accuracy": 1.0 - current,
        "current_selected_crc_valid_rate": _avg(rows, "current_selected_crc_valid"),
        "savaux_paper_symbol_ser": savaux,
        "savaux_paper_symbol_accuracy": 1.0 - savaux,
        "savaux_paper_crc_valid_rate": _avg(rows, "savaux_paper_crc_valid"),
        "savaux_codec_symbol_ser": codec,
        "savaux_codec_symbol_accuracy": 1.0 - codec,
        "savaux_codec_crc_valid_rate": _avg(rows, "savaux_codec_crc_valid"),
        "savaux_gain_vs_current": current - savaux,
        "savaux_codec_gain_vs_current": current - codec,
        "savaux_codec_gain_vs_savaux": savaux - codec,
        "savaux_codec_crc_gain_vs_current": _avg(rows, "savaux_codec_crc_valid")
        - _avg(rows, "current_selected_crc_valid"),
        "savaux_codec_crc_gain_vs_savaux": _avg(rows, "savaux_codec_crc_valid")
        - _avg(rows, "savaux_paper_crc_valid"),
        "savaux_codec_success_rate": _avg(rows, "savaux_codec_success"),
        "savaux_codec_argmax_fallback_rate": _avg(
            rows,
            "savaux_codec_selected_source_is_argmax_fallback",
        ),
        "savaux_codec_mean_topk_recall": _avg(rows, "savaux_codec_topk_recall"),
        "savaux_codec_mean_gt_rank": _avg(rows, "savaux_codec_mean_gt_rank"),
        "savaux_codec_mean_symbol_change_count": _avg(rows, "savaux_codec_symbol_change_count"),
        "savaux_codec_mean_symbol_fix_count": _avg(rows, "savaux_codec_symbol_fix_count"),
        "savaux_codec_mean_symbol_break_count": _avg(rows, "savaux_codec_symbol_break_count"),
    }


def _threshold_tables_with_codec(summary_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_thresholds, base_gains = _threshold_tables(summary_rows)
    specs = (
        ("SER<=10%", "symbol_ser", 0.10, False),
        ("accuracy>=90%", "symbol_accuracy", 0.90, True),
        ("CRC/PRR>=80%", "crc_valid_rate", 0.80, True),
        ("CRC/PRR>=50%", "crc_valid_rate", 0.50, True),
    )
    datasets = sorted({str(row["dataset"]) for row in summary_rows})
    extra_thresholds: list[dict[str, Any]] = []
    for dataset in datasets:
        curve = [row for row in summary_rows if str(row["dataset"]) == dataset]
        for metric_name, suffix, target, higher_is_better in specs:
            key = f"savaux_codec_{suffix}"
            threshold, status = _threshold_from_curve(curve, key, target, higher_is_better)
            extra_thresholds.append(
                {
                    "dataset": dataset,
                    "method": "savaux_codec",
                    "metric": metric_name,
                    "threshold_snr_db": "" if threshold is None else float(threshold),
                    "status": status,
                }
            )
    thresholds = base_thresholds + extra_thresholds
    by_key = {(row["dataset"], row["method"], row["metric"]): row for row in thresholds}
    gains = list(base_gains)
    for dataset in datasets:
        for baseline in ("traditional_fft", "current_selected", "savaux_paper"):
            for metric_name, _suffix, _target, _higher in specs:
                try:
                    method_thr = float(by_key[(dataset, "savaux_codec", metric_name)]["threshold_snr_db"])
                    base_thr = float(by_key[(dataset, baseline, metric_name)]["threshold_snr_db"])
                    gain = base_thr - method_thr
                    status = "ok"
                except (TypeError, ValueError, KeyError):
                    method_thr = float("nan")
                    base_thr = float("nan")
                    gain = float("nan")
                    status = "missing_threshold"
                gains.append(
                    {
                        "dataset": dataset,
                        "method": "savaux_codec",
                        "baseline": baseline,
                        "metric": metric_name,
                        "method_threshold_snr_db": "" if not math.isfinite(method_thr) else method_thr,
                        "baseline_threshold_snr_db": "" if not math.isfinite(base_thr) else base_thr,
                        "gain_db": "" if not math.isfinite(gain) else gain,
                        "status": status,
                    }
                )
    return thresholds, gains


def main() -> int:
    args = parse_args()
    snrs = _snr_values(args.snr_start, args.snr_stop, args.snr_step)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_args = _current_default_args(args)
    current_config = build_config(current_args)
    codec_config = _codec_config(args)

    manifest_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset_index, dataset in enumerate(args.datasets):
        paths = _dataset_paths(str(dataset))
        if not paths["iq"].exists():
            raise FileNotFoundError(paths["iq"])
        if not paths["symbols"].exists():
            raise FileNotFoundError(paths["symbols"])

        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        if samples.size == 0:
            raise ValueError(f"empty IQ file: {paths['iq']}")
        reference_power, reference_sample_count, reference_packet_count = _payload_reference_power(
            samples,
            paths["symbols"],
        )
        current_packets = load_current_packets(paths["symbols"], None)
        savaux_packets = load_savaux_packets(paths["symbols"], packet_filter=None, max_packets=None)
        savaux_by_packet = {int(packet["packet_index"]): packet for packet in savaux_packets}
        paired_packet_ids = sorted(set(current_packets).intersection(savaux_by_packet))
        if not paired_packet_ids:
            raise ValueError(f"no paired packets available for {dataset}")

        dataset_seed = int(args.seed) + 100000 * int(dataset_index)
        unit_noise: np.ndarray | None = None
        if not bool(args.independent_noise):
            rng = np.random.default_rng(dataset_seed)
            noise_i = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
            noise_q = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
            unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)

        manifest_rows.append(
            {
                "dataset": str(dataset),
                "input_iq": str(paths["iq"]),
                "gt_symbol_csv": str(paths["symbols"]),
                "iq_complex64_samples": int(samples.size),
                "reference_power": float(reference_power),
                "reference_power_db": float(10.0 * math.log10(reference_power)),
                "reference_sample_count": int(reference_sample_count),
                "reference_packet_count": int(reference_packet_count),
                "paired_packet_count": int(len(paired_packet_ids)),
                "seed": int(dataset_seed),
                "independent_noise": int(bool(args.independent_noise)),
                "top_k_metrics": int(args.top_k_metrics),
                "nibble_candidates": int(args.nibble_candidates),
                "row_beam_width": int(args.row_beam_width),
                "block_candidate_limit": int(args.block_candidate_limit),
                "block_symbol_seed_top_m": int(args.block_symbol_seed_top_m),
                "block_symbol_seed_deep_top_l": int(args.block_symbol_seed_deep_top_l),
                "block_symbol_seed_max_deep_positions": int(args.block_symbol_seed_max_deep_positions),
                "block_symbol_seed_quota": int(args.block_symbol_seed_quota),
                "block_symbol_seed_max_combinations": int(args.block_symbol_seed_max_combinations),
                "global_beam_width": int(args.global_beam_width),
                "global_rank_diverse_top_r": int(args.global_rank_diverse_top_r),
                "global_rank_diverse_beam_width": int(args.global_rank_diverse_beam_width),
                "global_rank_cost_max": float(args.global_rank_cost_max),
                "global_rank_cost_state_limit": int(args.global_rank_cost_state_limit),
                "final_candidate_limit": int(args.final_candidate_limit),
                "crc_candidate_min_evidence_margin": float(args.crc_candidate_min_evidence_margin),
                "crc_candidate_max_beam_rank": int(args.crc_candidate_max_beam_rank),
                "crc_candidate_high_rank_max_beam_rank": int(args.crc_candidate_high_rank_max_beam_rank),
                "crc_candidate_high_rank_min_evidence_margin": float(args.crc_candidate_high_rank_min_evidence_margin),
                "crc_selection_policy": str(args.crc_selection_policy),
                "crc_state_search_block_top_r": int(args.crc_state_search_block_top_r),
                "crc_state_search_state_limit": int(args.crc_state_search_state_limit),
                "crc_state_search_keep_per_key": int(args.crc_state_search_keep_per_key),
                "crc_state_search_max_candidates": int(args.crc_state_search_max_candidates),
                "crc_state_search_min_evidence_margin": float(args.crc_state_search_min_evidence_margin),
                "adaptive_path_top_k": int(args.adaptive_path_top_k),
                "adaptive_switch_penalty": float(args.adaptive_switch_penalty),
                "adaptive_step_penalty": float(args.adaptive_step_penalty),
                "adaptive_path_gain_power": float(args.adaptive_path_gain_power),
                "adaptive_switch_penalty_power": float(args.adaptive_switch_penalty_power),
                "adaptive_score_mix": float(args.adaptive_score_mix),
                "timing_path_top_k": int(args.timing_path_top_k),
                "timing_tau_grid": str(args.timing_tau_grid),
                "timing_slope_grid": str(args.timing_slope_grid),
                "timing_path_gain_power": float(args.timing_path_gain_power),
                "timing_slope_penalty_power": float(args.timing_slope_penalty_power),
                "timing_score_mix": float(args.timing_score_mix),
            }
        )

        for step_index, snr_db in enumerate(snrs):
            noise_power = reference_power * (10.0 ** (-float(snr_db) / 10.0))
            sigma = math.sqrt(float(noise_power) / 2.0)
            if unit_noise is None:
                rng = np.random.default_rng(dataset_seed + step_index)
                noise_i = rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                noise_q = rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                noisy = (samples + (noise_i + 1j * noise_q)).astype(np.complex64, copy=False)
            else:
                noisy = (samples + sigma * unit_noise).astype(np.complex64, copy=False)

            rows: list[dict[str, Any]] = []
            for packet_id in paired_packet_ids:
                current_row = evaluate_current_packet(
                    noisy,
                    current_packets[int(packet_id)],
                    current_args,
                    current_config,
                )
                codec_row = _evaluate_savaux_codec_packet(
                    noisy,
                    savaux_by_packet[int(packet_id)],
                    args,
                    codec_config,
                )
                row = _paired_packet_row(str(dataset), float(snr_db), current_row, codec_row)
                rows.append(row)
                packet_rows.append(row)

            summary = _summarize(rows, str(dataset), float(snr_db))
            summary_rows.append(summary)
            print(
                f"{dataset} snr={snr_db:>6.1f} "
                f"current={summary['current_selected_symbol_ser']:.3f} "
                f"savaux={summary['savaux_paper_symbol_ser']:.3f} "
                f"codec={summary['savaux_codec_symbol_ser']:.3f} "
                f"crc={summary['savaux_codec_crc_valid_rate']:.3f} "
                f"fix={summary['savaux_codec_mean_symbol_fix_count']:.2f} "
                f"break={summary['savaux_codec_mean_symbol_break_count']:.2f}",
                flush=True,
            )

    summary_rows.extend(_mean_of_datasets(summary_rows, snrs))
    threshold_rows, gain_rows = _threshold_tables_with_codec(summary_rows)
    _write_csv(output_dir / "manifest.csv", manifest_rows)
    _write_csv(output_dir / "per_packet_metrics.csv", packet_rows)
    _write_csv(output_dir / "snr_curve_summary.csv", summary_rows)
    _write_csv(output_dir / "threshold_table.csv", threshold_rows)
    _write_csv(output_dir / "gain_table.csv", gain_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "snr_values": snrs,
                "datasets": list(args.datasets),
                "manifest_rows": manifest_rows,
                "summary_rows": summary_rows,
                "threshold_rows": threshold_rows,
                "gain_rows": gain_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote={output_dir / 'snr_curve_summary.csv'}")
    print(f"wrote={output_dir / 'threshold_table.csv'}")
    print(f"wrote={output_dir / 'gain_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

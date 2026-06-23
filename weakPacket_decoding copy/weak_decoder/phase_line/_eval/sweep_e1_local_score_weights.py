#!/usr/bin/env python3
"""Sweep local-score weights for the Savaux Phase-Line selector."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np


THIS = Path(__file__).resolve()
PHASE_LINE_DIR = THIS.parents[1]
WEAK_ROOT = THIS.parents[3]
EXPERIMENT_DIR = WEAK_ROOT / "scripts" / "experiments"
PHASE_EXPERIMENT_DIR = EXPERIMENT_DIR / "phase_line"
for path in (str(WEAK_ROOT), str(EXPERIMENT_DIR), str(PHASE_EXPERIMENT_DIR), str(PHASE_LINE_DIR / "_eval")):
    if path not in sys.path:
        sys.path.insert(0, path)

from diagnose_savaux_stage1_selector import (  # noqa: E402
    _payload_reference_power,
    _payload_starts_and_abs,
)
from run_phase_line_threshold_sweep import _dataset_paths, _load_metadata  # noqa: E402
from run_symbol_phase_two_stage import _ser  # noqa: E402
from run_two_stage_weak_decoder import load_packets  # noqa: E402
from weak_decoder.chirp import bin_to_grlora_symbol  # noqa: E402
from weak_decoder.phase_line import (  # noqa: E402
    SavauxStage1Config,
    build_savaux_stage1_packet_evidence,
    default_savaux_phase_path_config,
    select_phase_viterbi_path,
)
from weak_decoder.phase_line.selector import (  # noqa: E402
    _ViterbiCandidate,
    _anchor_phase_line_rmse_pi,
    _apply_anchor_phase_bias,
    _db_ratio,
    _estimate_anchor_slope,
    _confidence_blend,
    _is_hard_anchor,
    _is_high_confidence,
    _transition_penalty,
    _viterbi_candidates,
)


CATEGORY_ORDER = (
    "A_savaux_ok_phase_ok",
    "B_savaux_wrong_phase_rescue",
    "C_savaux_ok_phase_break",
    "D_stage1_top40_miss",
    "E0_gt_in_top40_but_not_actual_candidates",
    "E1_gt_in_candidates_local_score_loss",
    "E2_gt_in_candidates_transition_loss",
    "E3_lock_or_high_conf_removed_gt",
    "Z_other",
)


@dataclass(frozen=True)
class CandidateComponents:
    raw_bin: int
    phase: float
    energy: float
    coherence: float
    rank: float
    phase_local: float
    energy_gap_db: float


@dataclass(frozen=True)
class PacketCase:
    dataset: str
    target_snr_db: float
    noise_seed: int
    packet_index: int
    sf: int
    ldro: bool
    gt_bins: tuple[int, ...]
    savaux_bins: tuple[int, ...]
    stage1_top_sets: tuple[frozenset[int], ...]
    candidates: tuple[tuple[CandidateComponents, ...], ...]
    hard_anchor_mask: tuple[bool, ...]
    high_confidence_mask: tuple[bool, ...]
    confidence_blend: tuple[float, ...]
    anchor_slope: float | None
    transition_matrices: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-26.0, -25.0, -24.0, -23.0, -22.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--top-l", type=int, default=16)
    parser.add_argument("--stage1-top-k", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "baseline_comparison" / "phase_line_e1_local_score_sweep")
    parser.add_argument("--energy-weights", nargs="+", type=float, default=[0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--coherence-weights", nargs="+", type=float, default=[0.15, 0.20, 0.25])
    parser.add_argument("--phase-local-weights", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--rank-weights", nargs="+", type=float, default=[0.0, 0.03, 0.05])
    parser.add_argument("--adaptive-grid", action="store_true")
    parser.add_argument("--low-energy-weights", nargs="+", type=float, default=[0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--low-coherence-weights", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--low-phase-local-weights", nargs="+", type=float, default=[0.20, 0.25, 0.30, 0.35])
    parser.add_argument("--low-rank-weights", nargs="+", type=float, default=[0.0])
    parser.add_argument("--penalty-grid", action="store_true")
    parser.add_argument("--penalty-weights", nargs="+", type=float, default=[0.0, 0.03, 0.06, 0.09, 0.12])
    parser.add_argument("--penalty-min-energies", nargs="+", type=float, default=[0.80, 0.85, 0.90])
    parser.add_argument("--penalty-coherence-refs", nargs="+", type=float, default=[0.88, 0.92, 0.95])
    parser.add_argument("--penalty-phase-ref", type=float, default=0.55)
    parser.add_argument("--penalty-gap-ref-db", type=float, default=1.50)
    parser.add_argument("--limit-variants", type=int, default=0)
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


def _effective_selector_state(evidences: Sequence[Any], base_config: Any) -> tuple[Any, tuple[bool, ...], tuple[bool, ...], float | None]:
    cfg = base_config
    high_confidence_mask = tuple(bool(_is_high_confidence(ev, cfg)) for ev in evidences)
    pre_hard_anchor_mask = tuple(bool(_is_hard_anchor(ev, cfg)) for ev in evidences)
    pre_hard_anchor_count = int(sum(pre_hard_anchor_mask))
    hard_anchor_rmse_pi = _anchor_phase_line_rmse_pi(evidences, pre_hard_anchor_mask)

    adaptive_top_l_high = int(getattr(cfg, "adaptive_top_l_high", 0))
    adaptive_anchor_threshold = int(getattr(cfg, "adaptive_top_l_anchor_threshold", 0))
    if (
        adaptive_top_l_high > int(cfg.top_l)
        and adaptive_anchor_threshold > 0
        and pre_hard_anchor_count >= adaptive_anchor_threshold
    ):
        cfg = replace(cfg, top_l=adaptive_top_l_high)
    else:
        adaptive_top_l_mid = int(getattr(cfg, "adaptive_top_l_mid", 0))
        adaptive_mid_threshold = int(getattr(cfg, "adaptive_top_l_mid_anchor_threshold", 0))
        adaptive_mid_rmse = float(getattr(cfg, "adaptive_top_l_mid_max_anchor_rmse_pi", float("inf")))
        if (
            adaptive_top_l_mid > int(cfg.top_l)
            and adaptive_mid_threshold > 0
            and pre_hard_anchor_count >= adaptive_mid_threshold
            and hard_anchor_rmse_pi <= adaptive_mid_rmse
        ):
            cfg = replace(cfg, top_l=adaptive_top_l_mid)

    adaptive_phase_threshold = int(getattr(cfg, "adaptive_phase_relax_anchor_threshold", 0))
    if (
        adaptive_phase_threshold > 0
        and pre_hard_anchor_count >= adaptive_phase_threshold
        and (int(cfg.phase_order) < 2 or float(cfg.second_order_weight) <= 0.0)
    ):
        cfg = replace(
            cfg,
            first_order_weight=float(getattr(cfg, "adaptive_phase_relax_first_order_weight", cfg.first_order_weight)),
            anchor_slope_weight=float(getattr(cfg, "adaptive_phase_relax_anchor_slope_weight", cfg.anchor_slope_weight)),
            anchor_slope_max_rmse_pi=float(
                getattr(cfg, "adaptive_phase_relax_anchor_slope_max_rmse_pi", cfg.anchor_slope_max_rmse_pi)
            ),
        )

    if bool(getattr(cfg, "adaptive_hard_anchor_softening_enabled", False)):
        soft_min_count = int(getattr(cfg, "adaptive_hard_anchor_softening_min_count", 0))
        soft_max_rmse = float(getattr(cfg, "adaptive_hard_anchor_softening_max_rmse_pi", float("inf")))
        if pre_hard_anchor_count >= soft_min_count and hard_anchor_rmse_pi <= soft_max_rmse:
            cfg = replace(
                cfg,
                hard_anchor_soft_top_k=max(
                    int(getattr(cfg, "hard_anchor_soft_top_k", 1)),
                    int(getattr(cfg, "adaptive_hard_anchor_softening_top_k", 1)),
                ),
                hard_anchor_soft_max_margin_db=float(getattr(cfg, "adaptive_hard_anchor_softening_max_margin_db", 0.0)),
                hard_anchor_soft_max_peak_to_median_db=float(
                    getattr(cfg, "adaptive_hard_anchor_softening_max_peak_to_median_db", 0.0)
                ),
            )

    hard_anchor_mask = tuple(bool(_is_hard_anchor(ev, cfg)) for ev in evidences)
    anchor_slope = _estimate_anchor_slope(evidences, hard_anchor_mask, cfg)
    return cfg, high_confidence_mask, hard_anchor_mask, anchor_slope


def _phase_local_components(evidences: Sequence[Any], cfg: Any, high_mask: Sequence[bool], hard_mask: Sequence[bool]) -> tuple[tuple[CandidateComponents, ...], ...]:
    base_rows = tuple(_viterbi_candidates(ev, cfg) for ev in evidences)
    phase_ref_mask = tuple(bool(high_mask[idx] or hard_mask[idx]) for idx in range(len(evidences)))
    phase_cfg = replace(
        cfg,
        anchor_phase_bias_weight=1.0,
        phase_local_weight=0.0,
        adaptive_local_score_enabled=False,
        energy_weight=0.0,
        coherence_weight=0.0,
        rank_weight=0.0,
        top1_soft_bonus=0.0,
    )
    phase_rows = _apply_anchor_phase_bias(base_rows, evidences, hard_mask, phase_ref_mask, phase_cfg)
    out: list[tuple[CandidateComponents, ...]] = []
    for ev, base_row, phase_row in zip(evidences, base_rows, phase_rows):
        phase_by_bin = {int(c.raw_bin): float(c.local_score) for c in phase_row}
        row: list[CandidateComponents] = []
        for cand in base_row:
            phase_local = float(phase_by_bin.get(int(cand.raw_bin), float(cand.local_score)) - float(cand.local_score))
            competitor = 0.0
            b = int(cand.raw_bin)
            for raw_other in ev.top_bins:
                other = int(raw_other)
                if other == b or other < 0 or other >= ev.evidence_power.size:
                    continue
                competitor = max(competitor, float(ev.evidence_power[other]))
            candidate_power = float(ev.evidence_power[b]) if 0 <= b < ev.evidence_power.size else 0.0
            energy_gap_db = (
                _db_ratio(candidate_power, competitor)
                if candidate_power > 0.0 and competitor > 0.0
                else 0.0
            )
            row.append(
                CandidateComponents(
                    raw_bin=int(cand.raw_bin),
                    phase=float(cand.phase),
                    energy=float(cand.energy_score),
                    coherence=float(cand.coherence_score),
                    rank=float(cand.rank_score),
                    phase_local=float(max(0.0, min(1.0, phase_local))),
                    energy_gap_db=float(energy_gap_db),
                )
            )
        out.append(tuple(row))
    return tuple(out)


def _component_to_candidate(component: CandidateComponents, local_score: float = 0.0) -> _ViterbiCandidate:
    return _ViterbiCandidate(
        raw_bin=int(component.raw_bin),
        phase=float(component.phase),
        local_score=float(local_score),
        energy_score=float(component.energy),
        coherence_score=float(component.coherence),
        rank_score=float(component.rank),
    )


def _transition_matrices(
    component_rows: Sequence[Sequence[CandidateComponents]],
    config: Any,
    anchor_slope: float | None,
) -> tuple[np.ndarray, ...]:
    out: list[np.ndarray] = []
    apply_ref = bool(anchor_slope is not None)
    for t in range(1, len(component_rows)):
        prev_row = tuple(_component_to_candidate(component) for component in component_rows[t - 1])
        curr_row = tuple(_component_to_candidate(component) for component in component_rows[t])
        matrix = np.zeros((len(prev_row), len(curr_row)), dtype=np.float64)
        for i, prev in enumerate(prev_row):
            for j, curr in enumerate(curr_row):
                matrix[i, j] = float(
                    _transition_penalty(
                        None,
                        prev,
                        curr,
                        config,
                        reference_slope=anchor_slope,
                        apply_reference_slope=apply_ref,
                    )
                )
        out.append(matrix)
    return tuple(out)


def _select_path(
    packet_case: PacketCase,
    config: Any,
    energy_weight: float,
    coherence_weight: float,
    phase_local_weight: float,
    rank_weight: float,
    low_energy_weight: float | None = None,
    low_coherence_weight: float | None = None,
    low_phase_local_weight: float | None = None,
    low_rank_weight: float | None = None,
    penalty_weight: float = 0.0,
    penalty_min_energy: float = 0.80,
    penalty_coherence_ref: float = 0.92,
    penalty_phase_ref: float = 0.55,
    penalty_gap_ref_db: float = 1.50,
) -> tuple[tuple[int, ...], tuple[tuple[_ViterbiCandidate, ...], ...]]:
    local_rows: list[np.ndarray] = []
    candidate_rows: list[tuple[_ViterbiCandidate, ...]] = []
    adaptive = low_energy_weight is not None
    for row_idx, component_row in enumerate(packet_case.candidates):
        if not component_row:
            local_rows.append(np.zeros(0, dtype=np.float64))
            candidate_rows.append(())
            continue
        if adaptive:
            blend = (
                float(packet_case.confidence_blend[row_idx])
                if row_idx < len(packet_case.confidence_blend)
                else 1.0
            )
            ew = (1.0 - blend) * float(low_energy_weight) + blend * float(energy_weight)
            cw = (1.0 - blend) * float(low_coherence_weight) + blend * float(coherence_weight)
            pw = (1.0 - blend) * float(low_phase_local_weight) + blend * float(phase_local_weight)
            rw = (1.0 - blend) * float(low_rank_weight) + blend * float(rank_weight)
        else:
            ew = float(energy_weight)
            cw = float(coherence_weight)
            pw = float(phase_local_weight)
            rw = float(rank_weight)
        energy = np.asarray([float(v.energy) for v in component_row], dtype=np.float64)
        coherence = np.asarray([float(v.coherence) for v in component_row], dtype=np.float64)
        phase_local = np.asarray([float(v.phase_local) for v in component_row], dtype=np.float64)
        rank = np.asarray([float(v.rank) for v in component_row], dtype=np.float64)
        energy_gap_db = np.asarray([float(v.energy_gap_db) for v in component_row], dtype=np.float64)
        local = (
            ew * energy
            + cw * coherence
            + pw * phase_local
            + rw * rank
        )
        if float(penalty_weight) > 0.0:
            energy_excess = np.clip(
                (energy - float(penalty_min_energy)) / max(1e-6, 1.0 - float(penalty_min_energy)),
                0.0,
                1.0,
            )
            coherence_deficit = np.clip(
                (float(penalty_coherence_ref) - coherence) / max(1e-6, float(penalty_coherence_ref)),
                0.0,
                1.0,
            )
            phase_deficit = np.clip(
                (float(penalty_phase_ref) - phase_local) / max(1e-6, float(penalty_phase_ref)),
                0.0,
                1.0,
            )
            gap_deficit = np.clip(
                (float(penalty_gap_ref_db) - energy_gap_db) / max(1e-6, float(penalty_gap_ref_db)),
                0.0,
                1.0,
            )
            suspicion = np.maximum(
                coherence_deficit,
                0.5 * coherence_deficit + 0.3 * phase_deficit + 0.2 * gap_deficit,
            )
            local = local - float(penalty_weight) * energy_excess * suspicion
        local_rows.append(local)
        candidate_rows.append(
            tuple(
                _component_to_candidate(component, local_score=float(score))
                for component, score in zip(component_row, local)
            )
        )
    candidates = tuple(candidate_rows)
    if not candidates:
        return (), candidates
    if len(candidates) == 1:
        best = max(candidates[0], key=lambda cand: cand.local_score)
        return (int(best.raw_bin),), candidates

    if not all(row.size > 0 for row in local_rows):
        return tuple(int(v) for v in packet_case.savaux_bins), candidates
    dp = local_rows[0].copy()
    backptrs: list[np.ndarray] = []
    for t in range(1, len(local_rows)):
        penalties = packet_case.transition_matrices[t - 1]
        scores = dp[:, np.newaxis] + local_rows[t][np.newaxis, :] - penalties
        prev = np.argmax(scores, axis=0)
        dp = scores[prev, np.arange(scores.shape[1])]
        backptrs.append(prev.astype(np.int32, copy=False))
    last = int(np.argmax(dp))
    path = [0 for _ in range(len(local_rows))]
    path[-1] = last
    for t in range(len(local_rows) - 1, 0, -1):
        path[t - 1] = int(backptrs[t - 1][path[t]])
    return tuple(int(candidates[t][path[t]].raw_bin) for t in range(len(path))), candidates


def _classify_symbol(
    sav_ok: bool,
    phase_ok: bool,
    gt_in_stage1: bool,
    gt_cand: _ViterbiCandidate | None,
    selected_cand: _ViterbiCandidate | None,
    hard_anchor: bool,
    high_confidence: bool,
    config: Any,
) -> str:
    if sav_ok and phase_ok:
        return "A_savaux_ok_phase_ok"
    if (not sav_ok) and phase_ok and gt_in_stage1:
        return "B_savaux_wrong_phase_rescue"
    if sav_ok and (not phase_ok):
        return "C_savaux_ok_phase_break"
    if not gt_in_stage1:
        return "D_stage1_top40_miss"
    if gt_cand is None:
        if bool(hard_anchor) or (bool(high_confidence) and int(getattr(config, "high_confidence_top_k", 0)) > 0):
            return "E3_lock_or_high_conf_removed_gt"
        return "E0_gt_in_top40_but_not_actual_candidates"
    if gt_cand is not None and selected_cand is not None:
        if float(gt_cand.local_score) < float(selected_cand.local_score) - 1e-12:
            return "E1_gt_in_candidates_local_score_loss"
        return "E2_gt_in_candidates_transition_loss"
    return "Z_other"


def _evaluate_variant(
    cases: Sequence[PacketCase],
    config: Any,
    energy_weight: float,
    coherence_weight: float,
    phase_local_weight: float,
    rank_weight: float,
    low_energy_weight: float | None = None,
    low_coherence_weight: float | None = None,
    low_phase_local_weight: float | None = None,
    low_rank_weight: float | None = None,
    penalty_weight: float = 0.0,
    penalty_min_energy: float = 0.80,
    penalty_coherence_ref: float = 0.92,
    penalty_phase_ref: float = 0.55,
    penalty_gap_ref_db: float = 1.50,
) -> dict[str, Any]:
    category_counts = Counter()
    by_snr_ser: dict[float, list[float]] = defaultdict(list)
    savaux_by_snr_ser: dict[float, list[float]] = defaultdict(list)
    total_symbols = 0
    stage1_hits = 0
    actual_hits = 0
    selected_errors = 0
    savaux_errors = 0
    for case in cases:
        selected_bins, candidate_rows = _select_path(
            case,
            config,
            energy_weight=energy_weight,
            coherence_weight=coherence_weight,
            phase_local_weight=phase_local_weight,
            rank_weight=rank_weight,
            low_energy_weight=low_energy_weight,
            low_coherence_weight=low_coherence_weight,
            low_phase_local_weight=low_phase_local_weight,
            low_rank_weight=low_rank_weight,
            penalty_weight=penalty_weight,
            penalty_min_energy=penalty_min_energy,
            penalty_coherence_ref=penalty_coherence_ref,
            penalty_phase_ref=penalty_phase_ref,
            penalty_gap_ref_db=penalty_gap_ref_db,
        )
        _raw_ser, symbol_ser, _compared = _ser(selected_bins, case.gt_bins, sf=case.sf, ldro=case.ldro)
        _sav_raw_ser, sav_symbol_ser, _ = _ser(case.savaux_bins, case.gt_bins, sf=case.sf, ldro=case.ldro)
        by_snr_ser[float(case.target_snr_db)].append(float(symbol_ser))
        savaux_by_snr_ser[float(case.target_snr_db)].append(float(sav_symbol_ser))
        for idx, gt_raw in enumerate(case.gt_bins):
            if idx >= len(selected_bins) or gt_raw < 0:
                continue
            selected_raw = int(selected_bins[idx])
            savaux_raw = int(case.savaux_bins[idx])
            gt_symbol = int(bin_to_grlora_symbol(int(gt_raw), sf=case.sf, is_header=False, ldro=case.ldro))
            selected_symbol = int(bin_to_grlora_symbol(selected_raw, sf=case.sf, is_header=False, ldro=case.ldro))
            savaux_symbol = int(bin_to_grlora_symbol(savaux_raw, sf=case.sf, is_header=False, ldro=case.ldro))
            cand_map = {int(c.raw_bin): c for c in candidate_rows[idx]} if idx < len(candidate_rows) else {}
            gt_cand = cand_map.get(int(gt_raw))
            selected_cand = cand_map.get(selected_raw)
            gt_in_stage1 = bool(int(gt_raw) in case.stage1_top_sets[idx])
            category = _classify_symbol(
                sav_ok=bool(savaux_symbol == gt_symbol),
                phase_ok=bool(selected_symbol == gt_symbol),
                gt_in_stage1=gt_in_stage1,
                gt_cand=gt_cand,
                selected_cand=selected_cand,
                hard_anchor=bool(case.hard_anchor_mask[idx]),
                high_confidence=bool(case.high_confidence_mask[idx]),
                config=config,
            )
            category_counts[category] += 1
            total_symbols += 1
            stage1_hits += int(gt_in_stage1)
            actual_hits += int(gt_cand is not None)
            selected_errors += int(selected_symbol != gt_symbol)
            savaux_errors += int(savaux_symbol != gt_symbol)
    row: dict[str, Any] = {
        "energy_weight": float(energy_weight),
        "coherence_weight": float(coherence_weight),
        "phase_local_weight": float(phase_local_weight),
        "rank_weight": float(rank_weight),
        "adaptive": int(low_energy_weight is not None),
        "low_energy_weight": "" if low_energy_weight is None else float(low_energy_weight),
        "low_coherence_weight": "" if low_coherence_weight is None else float(low_coherence_weight),
        "low_phase_local_weight": "" if low_phase_local_weight is None else float(low_phase_local_weight),
        "low_rank_weight": "" if low_rank_weight is None else float(low_rank_weight),
        "penalty_weight": float(penalty_weight),
        "penalty_min_energy": float(penalty_min_energy),
        "penalty_coherence_ref": float(penalty_coherence_ref),
        "penalty_phase_ref": float(penalty_phase_ref),
        "penalty_gap_ref_db": float(penalty_gap_ref_db),
        "symbol_count": int(total_symbols),
        "phase_line_ser": float(selected_errors / max(1, total_symbols)),
        "savaux_ser": float(savaux_errors / max(1, total_symbols)),
        "stage1_top40_recall": float(stage1_hits / max(1, total_symbols)),
        "actual_viterbi_candidate_recall": float(actual_hits / max(1, total_symbols)),
    }
    for snr in sorted(by_snr_ser):
        row[f"ser_{snr:g}"] = float(np.mean(by_snr_ser[snr]))
        row[f"savaux_ser_{snr:g}"] = float(np.mean(savaux_by_snr_ser[snr]))
    for category in CATEGORY_ORDER:
        row[category] = int(category_counts[category])
        row[f"{category}_rate"] = float(category_counts[category] / max(1, total_symbols))
    return row


def _load_cases(args: argparse.Namespace, base_config: Any) -> tuple[list[PacketCase], dict[str, Any]]:
    paths = _dataset_paths(str(args.dataset))
    metadata = _load_metadata(paths, str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    packets = load_packets(paths["symbols"], None)
    if "signal_reference_power" in metadata:
        signal_power = float(metadata["signal_reference_power"])
    else:
        signal_power, _ref_count = _payload_reference_power(samples, packets)
    stage1_config = SavauxStage1Config(cfo_correction_mode="continuous", top_k=int(args.stage1_top_k), branch_agreement_power=0.0)

    cases: list[PacketCase] = []
    for seed in [int(v) for v in args.seeds]:
        rng = np.random.default_rng(seed)
        unit_noise = (
            rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
            + 1j * rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        ).astype(np.complex64)
        for snr_db in [float(v) for v in args.snrs]:
            sigma = math.sqrt(signal_power * (10.0 ** (-snr_db / 10.0)) / 2.0)
            noisy = (samples + sigma * unit_noise).astype(np.complex64, copy=False)
            for packet_index in sorted(packets):
                packet = packets[packet_index]
                starts, abs_indices, gt_bins = _payload_starts_and_abs(packet, 8.0)
                stage1 = build_savaux_stage1_packet_evidence(
                    samples=noisy,
                    start_samples=starts,
                    sf=int(packet["sf"]),
                    os_factor=int(packet["os_factor"]),
                    abs_indices=abs_indices,
                    cfo_int=int(packet["cfo_int"]),
                    cfo_frac=float(packet["cfo_frac"]),
                    header_start_sample=int(packet["header_start_sample"]),
                    config=stage1_config,
                )
                result = select_phase_viterbi_path(
                    stage1.center_spectra,
                    stage1.evidence_powers,
                    stage1.abs_indices,
                    config=base_config,
                    offset_coherences=stage1.branch_phase_agreements,
                )
                evidences = tuple(result.evidences)
                cfg, high_mask, hard_mask, anchor_slope = _effective_selector_state(evidences, base_config)
                components = _phase_local_components(evidences, cfg, high_mask, hard_mask)
                transitions = _transition_matrices(components, cfg, anchor_slope)
                cases.append(
                    PacketCase(
                        dataset=str(args.dataset),
                        target_snr_db=float(snr_db),
                        noise_seed=int(seed),
                        packet_index=int(packet["packet_index"]),
                        sf=int(packet["sf"]),
                        ldro=bool(packet["ldro"]),
                        gt_bins=tuple(int(v) for v in gt_bins),
                        savaux_bins=tuple(int(ev.top1_bin) for ev in evidences),
                        stage1_top_sets=tuple(frozenset(int(v) for v in ev.top_bins) for ev in evidences),
                        candidates=components,
                        hard_anchor_mask=tuple(bool(v) for v in hard_mask),
                        high_confidence_mask=tuple(bool(v) for v in high_mask),
                        confidence_blend=tuple(float(_confidence_blend(ev, cfg)) for ev in evidences),
                        anchor_slope=anchor_slope,
                        transition_matrices=transitions,
                    )
                )
    manifest = {
        "dataset": str(args.dataset),
        "snrs": [float(v) for v in args.snrs],
        "seeds": [int(v) for v in args.seeds],
        "case_count": int(len(cases)),
        "signal_reference_power": float(signal_power),
        "input_iq": str(paths["iq"]),
        "symbol_csv": str(paths["symbols"]),
    }
    return cases, manifest


def main() -> int:
    args = parse_args()
    base_config = default_savaux_phase_path_config(top_l=int(args.top_l))
    cases, manifest = _load_cases(args, base_config)
    high_variants = list(
        itertools.product(
            [float(v) for v in args.energy_weights],
            [float(v) for v in args.coherence_weights],
            [float(v) for v in args.phase_local_weights],
            [float(v) for v in args.rank_weights],
        )
    )
    if bool(args.adaptive_grid):
        low_variants = list(
            itertools.product(
                [float(v) for v in args.low_energy_weights],
                [float(v) for v in args.low_coherence_weights],
                [float(v) for v in args.low_phase_local_weights],
                [float(v) for v in args.low_rank_weights],
            )
        )
        variants = [
            (*high_variant, *low_variant)
            for high_variant in high_variants
            for low_variant in low_variants
        ]
    else:
        variants = [(*high_variant, None, None, None, None) for high_variant in high_variants]
    if bool(args.penalty_grid):
        penalty_variants = list(
            itertools.product(
                [float(v) for v in args.penalty_weights],
                [float(v) for v in args.penalty_min_energies],
                [float(v) for v in args.penalty_coherence_refs],
                [float(args.penalty_phase_ref)],
                [float(args.penalty_gap_ref_db)],
            )
        )
    else:
        penalty_variants = [(0.0, float(args.penalty_min_energies[0]), float(args.penalty_coherence_refs[0]), float(args.penalty_phase_ref), float(args.penalty_gap_ref_db))]
    variants = [
        (*variant, *penalty_variant)
        for variant in variants
        for penalty_variant in penalty_variants
    ]
    if int(args.limit_variants) > 0:
        variants = variants[: int(args.limit_variants)]
    rows: list[dict[str, Any]] = []
    for idx, (
        energy_weight,
        coherence_weight,
        phase_local_weight,
        rank_weight,
        low_energy_weight,
        low_coherence_weight,
        low_phase_local_weight,
        low_rank_weight,
        penalty_weight,
        penalty_min_energy,
        penalty_coherence_ref,
        penalty_phase_ref,
        penalty_gap_ref_db,
    ) in enumerate(variants, start=1):
        row = _evaluate_variant(
            cases,
            base_config,
            energy_weight=energy_weight,
            coherence_weight=coherence_weight,
            phase_local_weight=phase_local_weight,
            rank_weight=rank_weight,
            low_energy_weight=low_energy_weight,
            low_coherence_weight=low_coherence_weight,
            low_phase_local_weight=low_phase_local_weight,
            low_rank_weight=low_rank_weight,
            penalty_weight=penalty_weight,
            penalty_min_energy=penalty_min_energy,
            penalty_coherence_ref=penalty_coherence_ref,
            penalty_phase_ref=penalty_phase_ref,
            penalty_gap_ref_db=penalty_gap_ref_db,
        )
        row["variant"] = f"e{energy_weight:.2f}_c{coherence_weight:.2f}_p{phase_local_weight:.2f}_r{rank_weight:.2f}"
        if low_energy_weight is not None:
            row["variant"] += (
                f"__lo_e{float(low_energy_weight):.2f}"
                f"_c{float(low_coherence_weight):.2f}"
                f"_p{float(low_phase_local_weight):.2f}"
                f"_r{float(low_rank_weight):.2f}"
            )
        if float(penalty_weight) > 0.0:
            row["variant"] += (
                f"__np_w{float(penalty_weight):.2f}"
                f"_emin{float(penalty_min_energy):.2f}"
                f"_cref{float(penalty_coherence_ref):.2f}"
            )
        rows.append(row)
        if idx == 1 or idx % 12 == 0 or idx == len(variants):
            print(
                f"{idx}/{len(variants)} {row['variant']} "
                f"ser={row['phase_line_ser']:.6f} "
                f"E1={row['E1_gt_in_candidates_local_score_loss_rate']:.4f} "
                f"C={row['C_savaux_ok_phase_break_rate']:.4f}",
                flush=True,
            )
    rows.sort(key=lambda row: (float(row["phase_line_ser"]), float(row["C_savaux_ok_phase_break_rate"])))
    out_dir = Path(args.output_dir)
    _write_csv(out_dir / "local_score_sweep_results.csv", rows)
    top_rows = rows[: min(20, len(rows))]
    _write_csv(out_dir / "local_score_sweep_top20.csv", top_rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "variant_count": int(len(variants)),
                "adaptive_grid": bool(args.adaptive_grid),
                "base_selector_config": {name: getattr(base_config, name) for name in base_config.__dataclass_fields__},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote={out_dir / 'local_score_sweep_results.csv'}")
    if rows:
        best = rows[0]
        print(
            "best "
            f"{best['variant']} ser={best['phase_line_ser']:.6f} "
            f"B={best['B_savaux_wrong_phase_rescue_rate']:.4f} "
            f"C={best['C_savaux_ok_phase_break_rate']:.4f} "
            f"E1={best['E1_gt_in_candidates_local_score_loss_rate']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Packet-level Stage-2 variant decisions and arbitration helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ....symbol_phase_two_stage import SymbolPhaseConfig, SymbolPhaseResult, build_symbol_evidences
from ...configs import PhasePathSelectorConfig
from ..stage2_common.types import Stage2VariantDecision


def path_change_stats(a: Sequence[int], b: Sequence[int]) -> tuple[int, float]:
    """Return count and ratio of raw-bin differences between two paths."""

    compared = min(len(a), len(b))
    change_count = int(
        sum(int(a[idx]) != int(b[idx]) for idx in range(compared))
        + abs(len(a) - len(b))
    )
    return change_count, float(change_count / max(1, compared))


def stage1_evidence_stats(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PhasePathSelectorConfig,
    offset_coherences: Sequence[np.ndarray] | None,
) -> tuple[float, float, float]:
    """Summarize how many strict Stage-1 anchors exist in the packet."""

    stage1_config = SymbolPhaseConfig(
        top_l_low_confidence=max(1, int(config.top_l)),
        lock_margin_db=float("inf"),
        lock_peak_to_median_db=float("inf"),
        coherence_candidate_top_l=0,
    )
    evidences = tuple(
        build_symbol_evidences(center_spectra, evidence_powers, abs_indices, stage1_config, offset_coherences)
    )
    if not evidences:
        return 0.0, 0.0, 0.0

    anchors = 0
    margins: list[float] = []
    peaks: list[float] = []
    for ev in evidences:
        margins.append(float(ev.top1_margin_db))
        peaks.append(float(ev.top1_peak_to_median_db))
        coherence_ok = (
            ev.offset_coherence is None
            or float(config.island_anchor_min_coherence) <= 0.0
            or float(ev.top1_coherence_score) >= float(config.island_anchor_min_coherence)
        )
        anchors += int(
            float(ev.top1_margin_db) >= float(config.island_anchor_margin_db)
            and float(ev.top1_peak_to_median_db) >= float(config.island_anchor_peak_to_median_db)
            and coherence_ok
        )
    return (
        float(anchors / max(1, len(evidences))),
        float(np.mean(margins)) if margins else 0.0,
        float(np.mean(peaks)) if peaks else 0.0,
    )


def decide_stage2_variant(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PhasePathSelectorConfig,
    offset_coherences: Sequence[np.ndarray] | None = None,
) -> Stage2VariantDecision:
    """Decide whether a packet should use stable or aggressive island Viterbi."""

    anchor_ratio, mean_margin, mean_peak = stage1_evidence_stats(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=config,
        offset_coherences=offset_coherences,
    )
    if not bool(config.adaptive_aggressive_enabled):
        return Stage2VariantDecision("island_stable", anchor_ratio, mean_margin, mean_peak, "disabled")
    if anchor_ratio <= float(config.adaptive_aggressive_anchor_ratio_threshold):
        return Stage2VariantDecision("island_aggressive", anchor_ratio, mean_margin, mean_peak, "low_anchor_ratio")
    if mean_margin <= float(config.adaptive_aggressive_mean_margin_db):
        return Stage2VariantDecision("island_aggressive", anchor_ratio, mean_margin, mean_peak, "low_mean_margin")
    if mean_peak <= float(config.adaptive_aggressive_mean_peak_to_median_db):
        return Stage2VariantDecision("island_aggressive", anchor_ratio, mean_margin, mean_peak, "low_mean_peak")
    return Stage2VariantDecision("island_stable", anchor_ratio, mean_margin, mean_peak, "stable_confidence")


def arbitrate_aggressive_path(
    stable_result: SymbolPhaseResult,
    aggressive_result: SymbolPhaseResult,
    decision: Stage2VariantDecision,
    config: PhasePathSelectorConfig,
) -> tuple[SymbolPhaseResult, Stage2VariantDecision]:
    """Accept or reject an aggressive path using production-available signals."""

    change_count, change_ratio = path_change_stats(
        aggressive_result.selected_raw_bins,
        stable_result.selected_raw_bins,
    )
    accepted_decision = replace(decision, change_count=change_count, change_ratio=change_ratio)
    if not bool(getattr(config, "adaptive_aggressive_arbitration_enabled", True)):
        return aggressive_result, accepted_decision

    max_change_count = int(getattr(config, "adaptive_aggressive_max_change_count", 0))
    max_change_ratio = float(getattr(config, "adaptive_aggressive_max_change_ratio", 0.0))
    over_count = max_change_count > 0 and change_count > max_change_count
    over_ratio = max_change_ratio > 0.0 and change_ratio > max_change_ratio
    if over_count or over_ratio:
        return stable_result, Stage2VariantDecision(
            "island_stable",
            decision.anchor_ratio,
            decision.mean_margin_db,
            decision.mean_peak_to_median_db,
            "aggressive_rejected_change_limit",
            change_count,
            change_ratio,
        )
    if change_count == 0:
        return stable_result, Stage2VariantDecision(
            "island_stable",
            decision.anchor_ratio,
            decision.mean_margin_db,
            decision.mean_peak_to_median_db,
            "aggressive_no_change",
            change_count,
            change_ratio,
        )
    return aggressive_result, accepted_decision


__all__ = [
    "arbitrate_aggressive_path",
    "decide_stage2_variant",
    "path_change_stats",
    "stage1_evidence_stats",
]

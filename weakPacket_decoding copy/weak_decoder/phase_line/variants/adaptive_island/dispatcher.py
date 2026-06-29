"""Named Stage-2 selector variants for weak-packet phase-line decoding."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ....symbol_phase_two_stage import SymbolPhaseResult
from ...configs import PhasePathSelectorConfig
from ..anchor_bounded_island.selector import select_anchor_bounded_island_viterbi_path
from ..bidirectional_rerank.selector import select_anchor_bounded_bidirectional_rerank_path
from ..stage2_common.types import Stage2VariantDecision, Stage2VariantName
from ..v1_risk_arbiter.selector import select_v1_risk_arbiter_path
from .decision import arbitrate_aggressive_path, decide_stage2_variant
from .profiles import aggressive_island_config, rewrite_island_config


def select_aggressive_island_viterbi_path(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PhasePathSelectorConfig | None = None,
    offset_coherences: Sequence[np.ndarray] | None = None,
) -> SymbolPhaseResult:
    """Run island Viterbi with the explicit aggressive low-SNR profile."""

    cfg = aggressive_island_config(config or PhasePathSelectorConfig())
    return select_anchor_bounded_island_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=cfg,
        offset_coherences=offset_coherences,
    )


def select_rewrite_island_viterbi_path(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PhasePathSelectorConfig | None = None,
    offset_coherences: Sequence[np.ndarray] | None = None,
) -> SymbolPhaseResult:
    """Run a permissive island Viterbi profile for low-confidence rewrites."""

    cfg = rewrite_island_config(config or PhasePathSelectorConfig())
    return select_anchor_bounded_island_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=cfg,
        offset_coherences=offset_coherences,
    )


def select_adaptive_aggressive_island_viterbi_path(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PhasePathSelectorConfig | None = None,
    offset_coherences: Sequence[np.ndarray] | None = None,
) -> tuple[SymbolPhaseResult, Stage2VariantDecision]:
    """Run stable island Viterbi, or an arbitrated aggressive low-confidence path."""

    cfg = config or PhasePathSelectorConfig()
    decision = decide_stage2_variant(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=cfg,
        offset_coherences=offset_coherences,
    )
    stable_result = select_anchor_bounded_island_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=cfg,
        offset_coherences=offset_coherences,
    )
    if decision.variant != "island_aggressive":
        return stable_result, decision

    aggressive_result = select_anchor_bounded_island_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=aggressive_island_config(cfg),
        offset_coherences=offset_coherences,
    )
    return arbitrate_aggressive_path(stable_result, aggressive_result, decision, cfg)


def select_stage2_variant_path(
    variant: Stage2VariantName,
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PhasePathSelectorConfig | None = None,
    offset_coherences: Sequence[np.ndarray] | None = None,
) -> tuple[SymbolPhaseResult, Stage2VariantDecision]:
    """Dispatch a named Stage-2 variant and return result plus mode metadata."""

    cfg = config or PhasePathSelectorConfig()
    if variant == "bidirectional":
        result = select_anchor_bounded_bidirectional_rerank_path(
            center_spectra=center_spectra,
            evidence_powers=evidence_powers,
            abs_indices=abs_indices,
            config=cfg,
            offset_coherences=offset_coherences,
        )
        return result, Stage2VariantDecision("bidirectional", math.nan, math.nan, math.nan, "explicit")
    if variant == "island_aggressive":
        result = select_aggressive_island_viterbi_path(
            center_spectra=center_spectra,
            evidence_powers=evidence_powers,
            abs_indices=abs_indices,
            config=cfg,
            offset_coherences=offset_coherences,
        )
        return result, Stage2VariantDecision("island_aggressive", math.nan, math.nan, math.nan, "explicit")
    if variant == "island_rewrite":
        result = select_rewrite_island_viterbi_path(
            center_spectra=center_spectra,
            evidence_powers=evidence_powers,
            abs_indices=abs_indices,
            config=cfg,
            offset_coherences=offset_coherences,
        )
        return result, Stage2VariantDecision("island_rewrite", math.nan, math.nan, math.nan, "explicit")
    if variant == "island_adaptive":
        return select_adaptive_aggressive_island_viterbi_path(
            center_spectra=center_spectra,
            evidence_powers=evidence_powers,
            abs_indices=abs_indices,
            config=cfg,
            offset_coherences=offset_coherences,
        )
    if variant == "v1_risk_arbiter":
        return select_v1_risk_arbiter_path(
            center_spectra=center_spectra,
            evidence_powers=evidence_powers,
            abs_indices=abs_indices,
            config=cfg,
            offset_coherences=offset_coherences,
        )
    result = select_anchor_bounded_island_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=cfg,
        offset_coherences=offset_coherences,
    )
    return result, Stage2VariantDecision("island_stable", math.nan, math.nan, math.nan, "explicit")


__all__ = [
    "Stage2VariantDecision",
    "Stage2VariantName",
    "aggressive_island_config",
    "decide_stage2_variant",
    "rewrite_island_config",
    "select_adaptive_aggressive_island_viterbi_path",
    "select_aggressive_island_viterbi_path",
    "select_rewrite_island_viterbi_path",
    "select_v1_risk_arbiter_path",
    "select_stage2_variant_path",
]

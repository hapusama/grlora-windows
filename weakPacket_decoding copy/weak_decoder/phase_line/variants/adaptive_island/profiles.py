"""Config profiles for named Stage-2 selector variants."""

from __future__ import annotations

from dataclasses import replace

from ...configs import PhasePathSelectorConfig


def aggressive_island_config(config: PhasePathSelectorConfig) -> PhasePathSelectorConfig:
    """Return the low-confidence packet profile.

    This profile locks only extreme anchors, treats top1 as a weak local prior,
    widens the candidate set, and permits unanchored island Viterbi.
    """

    return replace(
        config,
        island_anchor_margin_db=float(config.aggressive_anchor_margin_db),
        island_anchor_peak_to_median_db=float(config.aggressive_anchor_peak_to_median_db),
        island_anchor_min_coherence=float(config.aggressive_anchor_min_coherence),
        island_top_l=max(int(config.top_l), int(config.aggressive_top_l)),
        island_energy_weight=float(config.aggressive_energy_weight),
        island_coherence_weight=float(config.aggressive_coherence_weight),
        island_profile_weight=float(config.aggressive_profile_weight),
        island_phase_scale_pi=float(config.aggressive_phase_scale_pi),
        island_transition_weight=float(config.aggressive_transition_weight),
        island_boundary_weight=float(config.aggressive_boundary_weight),
        island_top1_bonus=float(config.aggressive_top1_bonus),
        island_max_energy_drop_db=float(config.aggressive_max_energy_drop_db),
        island_allow_unanchored_viterbi=True,
    )


def rewrite_island_config(config: PhasePathSelectorConfig) -> PhasePathSelectorConfig:
    """Return a more permissive profile for explicit low-confidence rewrites."""

    return replace(
        aggressive_island_config(config),
        island_top_l=max(int(config.top_l), int(config.rewrite_top_l)),
        island_energy_weight=float(config.rewrite_energy_weight),
        island_coherence_weight=float(config.rewrite_coherence_weight),
        island_profile_weight=float(config.rewrite_profile_weight),
        island_phase_scale_pi=float(config.rewrite_phase_scale_pi),
        island_transition_weight=float(config.rewrite_transition_weight),
        island_boundary_weight=float(config.rewrite_boundary_weight),
        island_top1_bonus=float(config.rewrite_top1_bonus),
        island_max_energy_drop_db=float(config.rewrite_max_energy_drop_db),
    )


__all__ = ["aggressive_island_config", "rewrite_island_config"]

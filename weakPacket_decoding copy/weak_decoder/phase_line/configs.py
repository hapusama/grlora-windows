"""Configuration for phase-line dominated weak-symbol path selection."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhaseLineSelectorConfig:
    """Parameters for the phase-smooth candidate path selector.

    The first stage uses multi-offset FFT evidence only to build compact Top-L
    candidate sets.  The second stage scores candidate paths mostly by local
    phase continuity, with energy/coherence used as reliability guards.
    """

    top_l: int = 24
    beam_width: int = 96
    window_size: int = 5
    window_degree: int = 1
    recent_decay: float = 0.78
    top1_lock_margin_db: float = 3.5
    top1_lock_peak_to_median_db: float = 8.0
    top1_lock_min_coherence: float = 0.78
    reliable_margin_db: float = 4.0
    reliable_peak_to_median_db: float = 9.0
    reliability_temperature: float = 2.0
    phase_weight_low_conf: float = 0.82
    phase_weight_high_conf: float = 0.38
    energy_weight_low_conf: float = 0.15
    energy_weight_high_conf: float = 0.57
    coherence_weight: float = 0.03
    phase_scale_pi: float = 0.28
    slope_scale_pi: float = 0.45
    curvature_scale_pi: float = 0.28
    slope_penalty_weight: float = 0.04
    curvature_penalty_weight: float = 0.08
    max_energy_drop_db_low_conf: float = 18.0
    max_energy_drop_db_high_conf: float = 3.0
    min_phase_gain_to_switch_low_conf: float = 0.08
    min_phase_gain_to_switch_high_conf: float = 0.18
    anchor_span_symbols: float = 10.0
    anchor_min_count: int = 2
    anchor_max_rmse_pi: float = 0.40
    fallback_to_energy_without_phase: bool = True
    header_reference_weight: float = 0.0
    min_phase_history: int = 2
    line_trim_frac: float = 0.20
    phase_proposal_enabled: bool = True
    phase_proposal_use_hard_anchors: bool = False
    phase_proposal_gated_hard_anchors: bool = True
    phase_proposal_gated_hard_anchor_min_count: int = 5
    phase_proposal_gated_hard_anchor_max_count: int = 8
    phase_proposal_gated_hard_anchor_max_rmse_pi: float = 0.12
    phase_proposal_rescue_count: int = 4
    phase_proposal_energy_preselect_count: int = 128
    phase_proposal_energy_preselect_factor: int = 6
    phase_proposal_max_energy_drop_db: float = 24.0
    phase_proposal_min_phase_score: float = 0.25
    phase_proposal_phase_scale_pi: float = 0.32
    phase_proposal_phase_weight: float = 0.78
    phase_proposal_energy_weight: float = 0.18
    phase_proposal_coherence_weight: float = 0.04
    phase_proposal_anchor_span_symbols: float = 12.0
    phase_proposal_anchor_min_count: int = 2
    phase_proposal_anchor_max_rmse_pi: float = 0.45
    phase_proposal_coherence_rescue_count: int = 12
    phase_proposal_coherence_preselect_factor: int = 4
    phase_proposal_coherence_min_gain: float = 0.10
    phase_proposal_keep_energy_backup: bool = True
    extra_coherence_candidates: int = 0
    extra_coherence_preselect_factor: int = 4
    extra_coherence_min_gain: float = 0.05
    phase_proposal_consensus_enabled: bool = False
    phase_proposal_consensus_top_l: int = 12
    phase_proposal_consensus_slope_steps: int = 25
    phase_proposal_consensus_intercept_steps: int = 48
    phase_proposal_consensus_phase_weight: float = 0.70
    phase_proposal_consensus_min_score: float = 0.20

    def phase_scale_rad(self) -> float:
        return max(1e-6, float(self.phase_scale_pi) * math.pi)

    def slope_scale_rad(self) -> float:
        return max(1e-6, float(self.slope_scale_pi) * math.pi)

    def curvature_scale_rad(self) -> float:
        return max(1e-6, float(self.curvature_scale_pi) * math.pi)


@dataclass(frozen=True)
class PhasePathSelectorConfig:
    """Parameters for payload-only phase-smooth Viterbi selection.

    This selector keeps multi-offset FFT evidence as the proposal generator,
    but the second stage scores complete payload paths by local phase
    smoothness.  No packet-local line is fitted.
    """

    top_l: int = 24
    adaptive_top_l_high: int = 40
    adaptive_top_l_anchor_threshold: int = 10
    adaptive_top_l_mid: int = 28
    adaptive_top_l_mid_anchor_threshold: int = 8
    adaptive_top_l_mid_max_anchor_rmse_pi: float = 0.30
    adaptive_phase_relax_anchor_threshold: int = 10
    adaptive_phase_relax_first_order_weight: float = 0.10
    adaptive_phase_relax_anchor_slope_weight: float = 0.08
    adaptive_phase_relax_anchor_slope_max_rmse_pi: float = 0.35
    phase_order: int = 1
    energy_weight: float = 0.35
    coherence_weight: float = 0.40
    rank_weight: float = 0.00
    phase_local_weight: float = 0.00
    adaptive_local_score_enabled: bool = False
    adaptive_local_score_margin_low_db: float = 0.40
    adaptive_local_score_margin_high_db: float = 2.50
    adaptive_local_score_peak_low_db: float = 7.0
    adaptive_local_score_peak_high_db: float = 12.0
    adaptive_low_conf_energy_weight: float = 0.50
    adaptive_low_conf_coherence_weight: float = 0.20
    adaptive_low_conf_phase_local_weight: float = 0.25
    adaptive_low_conf_rank_weight: float = 0.00
    noise_peak_penalty_enabled: bool = False
    noise_peak_penalty_weight: float = 0.00
    noise_peak_penalty_min_energy: float = 0.80
    noise_peak_penalty_coherence_ref: float = 0.92
    noise_peak_penalty_phase_ref: float = 0.55
    noise_peak_penalty_gap_ref_db: float = 1.50
    reliable_top1_protection_enabled: bool = False
    reliable_top1_protection_bonus: float = 0.00
    reliable_top1_protection_margin_db: float = 0.25
    reliable_top1_protection_peak_to_median_db: float = 10.0
    reliable_top1_protection_min_coherence: float = 0.88
    first_order_weight: float = 0.18
    second_order_weight: float = 0.00
    first_order_scale_pi: float = 0.75
    second_order_scale_pi: float = 0.35
    huber_delta: float = 1.0
    max_energy_drop_db: float = 24.0
    high_confidence_margin_db: float = 3.0
    high_confidence_peak_to_median_db: float = 7.0
    high_confidence_min_coherence: float = 0.0
    top1_soft_bonus: float = 0.00
    high_confidence_top_k: int = 0
    header_slope_weight: float = 0.00
    header_slope_scale_pi: float = 0.50
    header_slope_span: int = 0
    line_trim_frac: float = 0.20
    hard_anchor_top_k: int = 1
    hard_anchor_margin_db: float = 0.0
    hard_anchor_peak_to_median_db: float = 6.0
    hard_anchor_min_coherence: float = 0.85
    hard_anchor_soft_top_k: int = 1
    hard_anchor_soft_max_margin_db: float = 0.0
    hard_anchor_soft_max_peak_to_median_db: float = 0.0
    adaptive_hard_anchor_softening_enabled: bool = True
    adaptive_hard_anchor_softening_min_count: int = 10
    adaptive_hard_anchor_softening_max_rmse_pi: float = 0.50
    adaptive_hard_anchor_softening_top_k: int = 3
    adaptive_hard_anchor_softening_max_margin_db: float = 0.50
    adaptive_hard_anchor_softening_max_peak_to_median_db: float = 99.0
    path_arbiter_enabled: bool = True
    path_arbiter_first_abs_pi_threshold: float = 0.35
    path_arbiter_second_abs_pi_threshold: float = 0.42
    path_arbiter_top_l: int = 24
    phase_proposal_enabled: bool = True
    phase_proposal_use_hard_anchors: bool = False
    phase_proposal_gated_hard_anchors: bool = True
    phase_proposal_gated_hard_anchor_min_count: int = 5
    phase_proposal_gated_hard_anchor_max_count: int = 8
    phase_proposal_gated_hard_anchor_max_rmse_pi: float = 0.12
    phase_proposal_rescue_count: int = 4
    phase_proposal_energy_preselect_count: int = 128
    phase_proposal_energy_preselect_factor: int = 6
    phase_proposal_max_energy_drop_db: float = 24.0
    phase_proposal_min_phase_score: float = 0.25
    phase_proposal_phase_scale_pi: float = 0.32
    phase_proposal_phase_weight: float = 0.78
    phase_proposal_energy_weight: float = 0.18
    phase_proposal_coherence_weight: float = 0.04
    phase_proposal_anchor_span_symbols: float = 12.0
    phase_proposal_anchor_min_count: int = 2
    phase_proposal_anchor_max_rmse_pi: float = 0.45
    phase_proposal_coherence_rescue_count: int = 12
    phase_proposal_coherence_preselect_factor: int = 4
    phase_proposal_coherence_min_gain: float = 0.10
    phase_proposal_keep_energy_backup: bool = True
    extra_coherence_candidates: int = 0
    extra_coherence_preselect_factor: int = 4
    extra_coherence_min_gain: float = 0.05
    phase_proposal_consensus_enabled: bool = False
    phase_proposal_consensus_top_l: int = 12
    phase_proposal_consensus_slope_steps: int = 25
    phase_proposal_consensus_intercept_steps: int = 48
    phase_proposal_consensus_phase_weight: float = 0.70
    phase_proposal_consensus_min_score: float = 0.20
    sliding_window_refine_enabled: bool = True
    sliding_window_radius: int = 6
    sliding_window_min_anchors: int = 4
    sliding_window_max_rmse_pi: float = 0.25
    sliding_window_phase_scale_pi: float = 0.35
    sliding_window_phase_weight: float = 0.15
    sliding_window_energy_weight: float = 0.35
    sliding_window_coherence_weight: float = 0.50
    sliding_window_min_gain: float = 0.00
    sliding_window_max_energy_drop_db: float = 24.0
    sliding_window_max_switch_loss_db: float = 999.0
    sliding_window_min_coherence_gain: float = -1.0
    anchor_phase_bias_weight: float = 0.03
    anchor_phase_bias_span: float = 8.0
    anchor_phase_bias_min_anchors: int = 3
    anchor_phase_bias_scale_pi: float = 0.35
    anchor_phase_bias_max_rmse_pi: float = 0.35
    anchor_phase_bias_trim_frac: float = 0.20
    anchor_slope_weight: float = 0.05
    anchor_slope_scale_pi: float = 0.45
    anchor_slope_min_anchors: int = 4
    anchor_slope_max_rmse_pi: float = 0.25
    bidirectional_anchor_min_count: int = 4
    bidirectional_anchor_margin_db: float = 2.50
    bidirectional_anchor_peak_to_median_db: float = 12.0
    bidirectional_anchor_min_coherence: float = 0.90
    bidirectional_anchor_prune_enabled: bool = True
    bidirectional_anchor_prune_second_diff_pi: float = 0.80
    bidirectional_energy_weight: float = 0.85
    bidirectional_coherence_weight: float = 0.05
    bidirectional_phase_weight: float = 0.10
    bidirectional_phase_scale_pi: float = 0.50
    bidirectional_max_energy_drop_db: float = 24.0
    bidirectional_min_switch_gain: float = 0.04
    bidirectional_lock_high_confidence: bool = False
    island_anchor_min_count: int = 2
    island_anchor_margin_db: float = 2.50
    island_anchor_peak_to_median_db: float = 12.0
    island_anchor_min_coherence: float = 0.90
    island_anchor_prune_enabled: bool = True
    island_anchor_prune_second_diff_pi: float = 0.80
    island_top_l: int = 0
    island_energy_weight: float = 0.75
    island_coherence_weight: float = 0.10
    island_profile_weight: float = 0.15
    island_phase_scale_pi: float = 0.50
    island_transition_weight: float = 1.0
    island_boundary_weight: float = 1.0
    island_top1_bonus: float = 0.08
    island_max_energy_drop_db: float = 24.0
    island_allow_unanchored_viterbi: bool = False
    adaptive_aggressive_enabled: bool = True
    adaptive_aggressive_anchor_ratio_threshold: float = 0.03
    adaptive_aggressive_mean_margin_db: float = 0.0
    adaptive_aggressive_mean_peak_to_median_db: float = 0.0
    adaptive_aggressive_arbitration_enabled: bool = True
    adaptive_aggressive_max_change_count: int = 4
    adaptive_aggressive_max_change_ratio: float = 0.12
    aggressive_anchor_margin_db: float = 3.25
    aggressive_anchor_peak_to_median_db: float = 14.0
    aggressive_anchor_min_coherence: float = 0.94
    aggressive_top_l: int = 24
    aggressive_energy_weight: float = 0.55
    aggressive_coherence_weight: float = 0.15
    aggressive_profile_weight: float = 0.30
    aggressive_phase_scale_pi: float = 0.45
    aggressive_transition_weight: float = 1.25
    aggressive_boundary_weight: float = 1.50
    aggressive_top1_bonus: float = 0.00
    aggressive_max_energy_drop_db: float = 30.0
    rewrite_top_l: int = 40
    rewrite_energy_weight: float = 0.45
    rewrite_coherence_weight: float = 0.15
    rewrite_profile_weight: float = 0.40
    rewrite_phase_scale_pi: float = 0.40
    rewrite_transition_weight: float = 1.40
    rewrite_boundary_weight: float = 1.20
    rewrite_top1_bonus: float = 0.00
    rewrite_max_energy_drop_db: float = 36.0
    risk_arbiter_v1_trajectory_threshold: float = 0.80
    risk_arbiter_crc_first: bool = True

    def first_order_scale_rad(self) -> float:
        return max(1e-6, float(self.first_order_scale_pi) * math.pi)

    def second_order_scale_rad(self) -> float:
        return max(1e-6, float(self.second_order_scale_pi) * math.pi)

    def header_slope_scale_rad(self) -> float:
        return max(1e-6, float(self.header_slope_scale_pi) * math.pi)

    def anchor_slope_scale_rad(self) -> float:
        return max(1e-6, float(self.anchor_slope_scale_pi) * math.pi)

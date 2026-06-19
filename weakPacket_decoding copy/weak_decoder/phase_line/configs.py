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
    phase_order: int = 2
    energy_weight: float = 0.35
    coherence_weight: float = 0.05
    rank_weight: float = 0.05
    first_order_weight: float = 0.00
    second_order_weight: float = 0.45
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

    def first_order_scale_rad(self) -> float:
        return max(1e-6, float(self.first_order_scale_pi) * math.pi)

    def second_order_scale_rad(self) -> float:
        return max(1e-6, float(self.second_order_scale_pi) * math.pi)

    def header_slope_scale_rad(self) -> float:
        return max(1e-6, float(self.header_slope_scale_pi) * math.pi)

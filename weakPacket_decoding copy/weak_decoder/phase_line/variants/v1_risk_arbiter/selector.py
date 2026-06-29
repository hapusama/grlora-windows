"""Path arbitration variants built from existing Stage-2 selectors."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ....symbol_phase_two_stage import SymbolPhaseResult
from ...configs import PhasePathSelectorConfig
from ..bidirectional_rerank.selector import select_anchor_bounded_bidirectional_rerank_path
from ..stage2_common.types import Stage2VariantDecision
from ..v1_one_order_dp.selector import select_phase_viterbi_path


def select_v1_risk_arbiter_path(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PhasePathSelectorConfig | None = None,
    offset_coherences: Sequence[np.ndarray] | None = None,
) -> tuple[SymbolPhaseResult, Stage2VariantDecision]:
    """Use forward Viterbi unless its own trajectory score is too weak."""

    cfg = config or PhasePathSelectorConfig()
    v1 = select_phase_viterbi_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=cfg,
        offset_coherences=offset_coherences,
    )
    threshold = float(getattr(cfg, "risk_arbiter_v1_trajectory_threshold", 0.80))
    if float(v1.trajectory_score) > threshold:
        return v1, Stage2VariantDecision(
            "v1_risk_arbiter",
            float("nan"),
            float("nan"),
            float("nan"),
            "v1_accepted",
        )

    fallback = select_anchor_bounded_bidirectional_rerank_path(
        center_spectra=center_spectra,
        evidence_powers=evidence_powers,
        abs_indices=abs_indices,
        config=cfg,
        offset_coherences=offset_coherences,
    )
    return fallback, Stage2VariantDecision(
        "v1_risk_arbiter",
        float("nan"),
        float("nan"),
        float("nan"),
        "v2_fallback_low_v1_trajectory",
    )


__all__ = ["select_v1_risk_arbiter_path"]

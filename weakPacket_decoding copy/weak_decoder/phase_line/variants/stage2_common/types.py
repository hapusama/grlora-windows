"""Shared Stage-2 variant type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Stage2VariantName = Literal[
    "bidirectional",
    "island_stable",
    "island_aggressive",
    "island_adaptive",
    "island_rewrite",
    "v1_risk_arbiter",
]


@dataclass(frozen=True)
class Stage2VariantDecision:
    """Packet-level mode decision used by adaptive Stage-2 variants."""

    variant: Stage2VariantName
    anchor_ratio: float
    mean_margin_db: float
    mean_peak_to_median_db: float
    reason: str
    change_count: int = 0
    change_ratio: float = 0.0


__all__ = ["Stage2VariantDecision", "Stage2VariantName"]

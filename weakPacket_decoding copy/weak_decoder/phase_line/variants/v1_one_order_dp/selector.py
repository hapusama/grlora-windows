"""Savaux Stage-1 + forward one-order phase Viterbi selector.

This is the current strongest Stage-2 baseline from the 2026-06-27 handoff.
The code is still implemented in the legacy core; this file gives the winning
variant its own stable import path.
"""

from .._legacy_core.selector import select_phase_viterbi_path

__all__ = ["select_phase_viterbi_path"]


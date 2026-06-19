"""Version label for the symbol-level two-stage phase selector branch."""

from .. import candidate_pruning
from .. import phase_guided_demod
from .. import symbol_phase_two_stage

VERSION = "v2"
STATUS = "historical-phy-baseline"
DESCRIPTION = "Symbol-level two-stage Top-L selector with phase-line rerank."

__all__ = [
    "DESCRIPTION",
    "STATUS",
    "VERSION",
    "candidate_pruning",
    "phase_guided_demod",
    "symbol_phase_two_stage",
]

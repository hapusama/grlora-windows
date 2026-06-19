"""Version label for the current phase-assisted PHY selector branch."""

from .. import candidate_pruning
from .. import phase_guided_demod
from .. import symbol_phase_two_stage

VERSION = "v3"
STATUS = "active-research-line"
DESCRIPTION = (
    "PHY-only Top-L candidate sequence selection with energy, offset "
    "coherence, and packet-local phase trajectory evidence."
)

__all__ = [
    "DESCRIPTION",
    "STATUS",
    "VERSION",
    "candidate_pruning",
    "phase_guided_demod",
    "symbol_phase_two_stage",
]

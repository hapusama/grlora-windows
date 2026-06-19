"""Version label for the legacy phase/codec beam weak decoder branch."""

from .. import blind_payload_decoder
from .. import blind_payload_search
from .. import phase_guided_demod
from .. import two_stage_weak_decoder

VERSION = "v1"
STATUS = "legacy-diagnostic"
DESCRIPTION = "Legacy phase-guided, blind-search, and codec/CRC beam branch."

__all__ = [
    "DESCRIPTION",
    "STATUS",
    "VERSION",
    "blind_payload_decoder",
    "blind_payload_search",
    "phase_guided_demod",
    "two_stage_weak_decoder",
]

"""Weak-packet LoRa demodulation prototypes.

This package is intentionally independent from GNU Radio at import time.  It
can reuse gr-lora_sdr packet metadata when available, then works on raw
complex64 IQ windows with NumPy.
"""

from .chirp import build_upchirp, dechirp_fft
from .observations import PacketObservation, SymbolObservation, extract_packet_observation
from .phase_model import PhaseRerankConfig, PhaseRerankResult, rerank_observation

__all__ = [
    "PacketObservation",
    "PhaseRerankConfig",
    "PhaseRerankResult",
    "SymbolObservation",
    "build_upchirp",
    "dechirp_fft",
    "extract_packet_observation",
    "rerank_observation",
]

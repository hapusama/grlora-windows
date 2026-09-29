"""Bounded residual-CFO recovery at the decoder's final timing coordinate.

The preamble ranking uses received known chirps only. It is an experimental
control, not a replacement for acquisition or a claim of improved sensitivity.
Neither candidate generation nor CRC arbitration accepts expected payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ...baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)
from .decoder_aware_crc import DecoderAwarePacketResult
from .soft_hamming_crc import decode_soft_hamming_sync_candidate


@dataclass(frozen=True)
class ResidualCfoCandidate:
    """Decoder inputs; timing and sampling-clock estimates are held fixed."""

    fine_payload_start_sample: int
    cfo_int_est: int
    cfo_frac_est: float
    sfo_hat: float
    sfo_cum_initial: float
    valid: bool
    delta_cfo_bins: int
    preamble_score: float | None = None
    preamble_symbols_used: int = 0


@dataclass(frozen=True)
class ResidualCfoAttempt:
    candidate: ResidualCfoCandidate
    decode: DecoderAwarePacketResult
    crc_accepted: bool


@dataclass(frozen=True)
class ResidualCfoResult:
    attempts: tuple[ResidualCfoAttempt, ...]

    @property
    def selected(self) -> ResidualCfoAttempt | None:
        return next((item for item in self.attempts if item.crc_accepted), None)


def build_residual_cfo_candidates(
    frame_sync: Any | None,
) -> tuple[ResidualCfoCandidate, ...]:
    """Keep the original estimate first, then CFO -1 and +1 bin."""
    if frame_sync is None:
        return ()
    return tuple(
        ResidualCfoCandidate(
            fine_payload_start_sample=int(frame_sync.fine_payload_start_sample),
            cfo_int_est=int(frame_sync.cfo_int_est) + delta,
            cfo_frac_est=float(frame_sync.cfo_frac_est),
            sfo_hat=float(frame_sync.sfo_hat),
            sfo_cum_initial=float(frame_sync.sfo_cum_initial),
            valid=bool(frame_sync.valid),
            delta_cfo_bins=delta,
        )
        for delta in (0, -1, 1)
    )


def rank_residual_cfo_from_preamble(
    samples: np.ndarray,
    candidates: tuple[ResidualCfoCandidate, ...],
    *,
    sf: int,
    os_factor: int,
    preamble_symbols: int,
) -> tuple[ResidualCfoCandidate, ...]:
    """Rank three CFOs by known-bin power at the final payload time origin.

Use up to eight middle preamble chirps, excluding four at each edge. Nominal
symbol spacing is used for this local diagnostic; large clock drift and wrong
whole-chirp boundaries can invalidate it. Scores are not calibrated detection
probabilities. Unavailable preambles retain the original candidate order.
"""
    from dataclasses import replace

    if int(sf) < 5 or int(os_factor) < 1 or int(preamble_symbols) < 0:
        raise ValueError("invalid SF, oversampling factor, or preamble length")
    if not candidates:
        return ()
    n_bins = 1 << int(sf)
    symbol_samples = n_bins * int(os_factor)
    header_start = candidates[0].fine_payload_start_sample
    starts = [
        header_start
        - int((int(preamble_symbols) + 4.25 - index) * symbol_samples)
        + int(os_factor) // 2
        for index in range(4, min(12, int(preamble_symbols) - 4))
    ]
    if not starts or any(start < 0 or start + symbol_samples > len(samples) for start in starts):
        return candidates
    ranked = []
    for candidate in candidates:
        known_power = 0.0
        total_power = 0.0
        for start in starts:
            spectrum, _, _ = paper_oversampled_spectrum(
                samples,
                start,
                int(sf),
                int(os_factor),
                cfo_int=candidate.cfo_int_est,
                cfo_frac=candidate.cfo_frac_est,
                header_start_sample=header_start + int(os_factor) // 2,
                cfo_correction_mode="continuous",
            )
            power = np.abs(spectrum).astype(np.float64) ** 2
            known_power += float(power[0])
            total_power += float(np.sum(power))
        score = known_power / max(total_power, np.finfo(float).tiny)
        if not math.isfinite(score):
            return candidates
        ranked.append(replace(candidate, preamble_score=score, preamble_symbols_used=len(starts)))
    # Stable sorting preserves the original-first tie break, including zero IQ.
    return tuple(sorted(ranked, key=lambda item: -float(item.preamble_score)))


def decode_residual_cfo_candidates(
    samples: np.ndarray,
    candidates: tuple[ResidualCfoCandidate, ...],
    *,
    sf: int,
    bw_hz: float,
    os_factor: int,
    max_attempts: int = 3,
    ldro_mode: int = 2,
    crc_mode: str = "grlora",
    allow_gate_failed_candidate: bool = False,
) -> ResidualCfoResult:
    """Stop at the first valid header with CRC enabled and payload CRC pass."""
    if int(max_attempts) not in (1, 2, 3):
        raise ValueError("max_attempts must be 1, 2, or 3")
    attempts = []
    for candidate in candidates[:int(max_attempts)]:
        decoded = decode_soft_hamming_sync_candidate(
            samples,
            candidate,
            sf=sf,
            bw_hz=bw_hz,
            os_factor=os_factor,
            ldro_mode=ldro_mode,
            crc_mode=crc_mode,
            allow_gate_failed_candidate=allow_gate_failed_candidate,
        )
        accepted = bool(decoded.header_valid and decoded.header.has_crc and decoded.crc_valid)
        attempts.append(ResidualCfoAttempt(candidate, decoded, accepted))
        if accepted:
            break
    return ResidualCfoResult(tuple(attempts))

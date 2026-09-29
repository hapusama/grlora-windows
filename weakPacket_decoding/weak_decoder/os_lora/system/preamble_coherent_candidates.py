"""PC-Ridge Stage A/B: coherent preamble candidate synchronization.

Implements the first slice of
``doc/preamble_coherent_list_sync_design_20260914.md``: segment-coherent
accumulation over the repeated preamble upchirps, SFD downchirp centring of a
bounded STO enumeration along the chirp coupling line, and a ranked candidate
list consumable by the existing decode/CRC arbitration path.

Calibration facts measured on the 8 clean OTA captures (2026-09-15):
* the 16-symbol segment-coherent upchirp statistic pins b_u = CFO + STO
  to within 0.3 bins of every packet's clean CFO+STO (peak/second 50-284x);
* the 2-symbol conjugated SFD pair carries a +0.5 bin interpolation bias and
  occasional 1-2 bin outliers, so it only CENTRES the STO search, never
  decides.

Branches are never treated as diversity (July negative result): the Savaux
branch-combined periodogram is used purely as the per-chirp front-end.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from ...baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)

SEGMENT_CHIRPS = 4
PREAMBLE_TO_PAYLOAD = 18.0 + 2.25  # upchirps(16) + sync(2) + SFD(2.25)


@dataclass(frozen=True)
class CoherentCandidate:
    """One (CFO, STO) hypothesis ranked by preamble evidence."""

    cfo_total_bins: float
    sto_chips: float
    payload_start_sample: int
    up_score: float
    down_score: float
    source: str
    sfo_hat: float = 0.0
    sfo_cum_initial: float = 0.0

    def as_frame_sync(self) -> SimpleNamespace:
        """Adapt to the decoder's FrameSync field contract."""

        cfo_int = int(math.floor(self.cfo_total_bins + 0.5))
        return SimpleNamespace(
            valid=True,
            fine_payload_start_sample=int(self.payload_start_sample),
            cfo_int_est=int(cfo_int),
            cfo_frac_est=float(self.cfo_total_bins - cfo_int),
            sfo_hat=float(self.sfo_hat),
            sfo_cum_initial=float(self.sfo_cum_initial),
        )


def _chirp_spectra(
    samples: np.ndarray,
    starts: Sequence[int],
    *,
    sf: int,
    os_factor: int,
    conjugate: bool = False,
) -> list[np.ndarray]:
    values = np.conjugate(samples) if conjugate else samples
    return [
        paper_oversampled_spectrum(
            values,
            int(start),
            sf=int(sf),
            os_factor=int(os_factor),
        )[0]
        for start in starts
    ]


def _segment_power(
    spectra: Sequence[np.ndarray], segment: int = SEGMENT_CHIRPS
) -> np.ndarray:
    """Coherent inside segments, noncoherent across segments."""

    if not spectra:
        raise ValueError("no spectra")
    total = None
    for index in range(0, len(spectra), int(segment)):
        block = spectra[index : index + int(segment)]
        seg = np.sum(block, axis=0)
        power = np.abs(seg) ** 2
        total = power if total is None else total + power
    return np.asarray(total, dtype=np.float64)


def _chirp_axis_accumulate(
    spectra: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fully coherent chirp-axis accumulation robust to fractional CFO.

    For every bin, the per-chirp complex values are derotated by a 16-point
    FFT along the chirp axis: a tone at fractional bin nu peaks at FFT index
    k = L*nu, so segment-cancellation from fractional CFO (up to -14 dB for
    L=4 segments at nu~0.27, measured 2026-09-15) disappears.  Returns the
    accumulated power per bin and the fractional index of its maximum.
    """

    stacked = np.stack(
        [np.asarray(s, dtype=np.complex128) for s in spectra], axis=0
    )
    rotated = np.fft.fft(stacked, axis=0)
    power = np.abs(rotated) ** 2  # (n_chirps, n_bins)
    best_axis = np.argmax(power, axis=0)
    best_power = power[best_axis, np.arange(power.shape[1])]
    return best_power, best_axis


def _top_peaks(
    power: np.ndarray, count: int, exclude_radius: int = 2
) -> list[tuple[float, float]]:
    """Top peaks with quadratic fractional-bin interpolation."""

    work = np.asarray(power, dtype=np.float64).copy()
    n_bins = work.size
    peaks: list[tuple[float, float]] = []
    for _ in range(int(count)):
        b = int(np.argmax(work))
        value = float(work[b])
        if value <= 0.0:
            break
        ym = float(work[(b - 1) % n_bins])
        yp = float(work[(b + 1) % n_bins])
        denom = ym - 2.0 * value + yp
        frac = 0.5 * (ym - yp) / denom if denom != 0.0 else 0.0
        frac = max(-0.5, min(0.5, frac))
        peaks.append((float(b) + frac, value))
        for k in range(-int(exclude_radius), int(exclude_radius) + 1):
            work[(b + k) % n_bins] = 0.0
    return peaks


def _wrap_half(value: float, n_bins: int) -> float:
    return (float(value) + n_bins / 2.0) % float(n_bins) - n_bins / 2.0


def _header_concentration(
    samples: np.ndarray,
    *,
    start: int,
    cfo_int: int,
    cfo_frac: float,
    sf: int,
    os_factor: int,
    n_symbols: int = 4,
) -> float:
    """Mean peak-to-total concentration of the first header symbols.

    Data-aided refinement score: the dechirp window/CFO split is invisible in
    the preamble (coupling invariant), but header symbols concentrate their
    energy in one bin only for the correct (cfo_frac, start) pair.
    """

    n_s = (1 << int(sf)) * int(os_factor)
    origin = int(os_factor) // 2
    scores: list[float] = []
    for index in range(int(n_symbols)):
        s0 = int(start) + index * n_s + origin
        if s0 < 0 or s0 + n_s > samples.size:
            return -1.0
        spectrum = paper_oversampled_spectrum(
            samples,
            s0,
            sf=int(sf),
            os_factor=int(os_factor),
            cfo_int=int(cfo_int),
            cfo_frac=float(cfo_frac),
            cfo_correction_mode="symbol",
        )[0]
        power = np.abs(spectrum) ** 2
        total = float(np.sum(power))
        scores.append(float(np.max(power)) / total if total > 0 else 0.0)
    return float(np.mean(scores))


def coherent_preamble_complex_stack(
    samples: np.ndarray,
    *,
    sf: int,
    os_factor: int,
    payload_start_hint: int,
    preamble_symbols: int = 16,
) -> tuple[np.ndarray, int]:
    """Complex chirp-axis FFT stack (n_chirps x n_bins) of the preamble.

    The value at [k, b] is the coherent 16-chirp sum of the upchirp tone at
    bin b derotated by e^{-j2*pi*(k/n_chirps)*s}; its angle at the shared
    peak gives each copy's carrier phase reference for cross-copy COHERENT
    combining (relative phase between copies is what matters and survives
    even when each copy's peak-picking fails).
    """

    values = np.asarray(samples, dtype=np.complex64)
    n_s = (1 << int(sf)) * int(os_factor)
    preamble_symbols = int(preamble_symbols)
    w0 = int(payload_start_hint) - int(round(PREAMBLE_TO_PAYLOAD * n_s))
    if w0 < 0:
        raise ValueError("preamble window starts before the capture")
    up_starts = [w0 + s * n_s for s in range(preamble_symbols)]
    up_spectra = _chirp_spectra(values, up_starts, sf=sf, os_factor=os_factor)
    stacked = np.stack(
        [np.asarray(s, dtype=np.complex128) for s in up_spectra], axis=0
    )
    return np.fft.fft(stacked, axis=0), w0


def coherent_preamble_axes_power(
    samples: np.ndarray,
    *,
    sf: int,
    os_factor: int,
    payload_start_hint: int,
    preamble_symbols: int = 16,
) -> tuple[np.ndarray, int]:
    """Chirp-axis power (n_chirps x n_bins) and the window start.

    Averaging this 2-D statistic across K independent copies of the same
    packet before peak-finding pushes the per-copy sync floor down by about
    10*log10(K) dB: the copies share the same b_u but not the same noise.
    """

    values = np.asarray(samples, dtype=np.complex64)
    n_s = (1 << int(sf)) * int(os_factor)
    preamble_symbols = int(preamble_symbols)
    w0 = int(payload_start_hint) - int(round(PREAMBLE_TO_PAYLOAD * n_s))
    if w0 < 0:
        raise ValueError("preamble window starts before the capture")
    up_starts = [w0 + s * n_s for s in range(preamble_symbols)]
    up_spectra = _chirp_spectra(values, up_starts, sf=sf, os_factor=os_factor)
    stacked = np.stack(
        [np.asarray(s, dtype=np.complex128) for s in up_spectra], axis=0
    )
    power = np.abs(np.fft.fft(stacked, axis=0)) ** 2
    return power, w0


def shared_bu_from_axes_power(
    axes_power: np.ndarray,
) -> float:
    """Peak bin (with fractional axis) of one already-averaged 2-D power."""

    per_bin = np.max(axes_power, axis=0)  # (n_bins,)
    per_bin_axis = np.argmax(axes_power, axis=0)
    b = int(np.argmax(per_bin))
    ym = float(per_bin[(b - 1) % per_bin.size])
    yp = float(per_bin[(b + 1) % per_bin.size])
    value = float(per_bin[b])
    denom = ym - 2.0 * value + yp
    frac = 0.5 * (ym - yp) / denom if denom != 0.0 else 0.0
    frac = max(-0.5, min(0.5, frac))
    return float(b) + frac + float(per_bin_axis[b]) / float(axes_power.shape[0])


def coherent_preamble_candidates(
    samples: np.ndarray,
    *,
    sf: int,
    os_factor: int,
    payload_start_hint: int,
    center_frequency_hz: float = 487_700_000.0,
    bw_hz: float = 125_000.0,
    preamble_symbols: int = 16,
    sto_search_chips: float = 2.0,
    sto_step_chips: float = 0.5,
    max_candidates: int = 9,
) -> list[CoherentCandidate]:
    """Build a ranked (CFO, STO) candidate list from one packet's IQ.

    The 16-symbol coherent upchirp accumulation yields b_u = CFO + STO to
    well under a bin.  The measured decoder tolerance along the coupling
    line is about 1.5 chips wide (clean-capture sweep 2026-09-15:
    d in [-1, +0.5] chips), so members at ``sto_step_chips`` spacing around
    the coarse anchor cover it; the anchor itself comes from packet
    detection.  The weak SFD downchirp pair is deliberately NOT used for
    centring: its noise at low SNR (several chips) exceeds the tolerance
    window and derails the enumeration.  SFO follows the CFO-induced model
    ``sfo = cfo_total * bw / fc`` used by the ridge builder.
    """

    values = np.asarray(samples, dtype=np.complex64)
    n_bins = 1 << int(sf)
    n_s = n_bins * int(os_factor)
    preamble_symbols = int(preamble_symbols)
    w0 = int(payload_start_hint) - int(round(PREAMBLE_TO_PAYLOAD * n_s))
    if w0 < 0:
        raise ValueError("preamble window starts before the capture")

    up_starts = [w0 + s * n_s for s in range(preamble_symbols)]
    up_spectra = _chirp_spectra(values, up_starts, sf=sf, os_factor=os_factor)
    up_power, up_frac_axis = _chirp_axis_accumulate(up_spectra)
    up_peaks = _top_peaks(up_power, 1)
    if not up_peaks:
        return []
    b_u_int, _up_peak = up_peaks[0]
    b_u = float(b_u_int) + float(
        up_frac_axis[int(round(b_u_int)) % n_bins]
    ) / float(len(up_spectra))

    step = float(sto_step_chips)
    span = int(round(float(sto_search_chips) / step))
    offsets = [0.0] + [
        sign * step * k for k in range(1, span + 1) for sign in (+1.0, -1.0)
    ]
    candidates: list[CoherentCandidate] = []
    for rank, sto in enumerate(offsets):
        cfo = (float(b_u) - sto) % float(n_bins)
        # CFO-induced SFO model (matches the ridge builder and the clean
        # audit: sfo_hat = cfo_total * bw / fc, residual cum at payload
        # start ~0 because the preamble drift is already folded into the
        # payload-start anchor).  Without it the payload cursor drifts
        # ~2.5 chips over the frame and CRC always fails at low SNR.
        # The SFO model needs the SIGNED CFO; ``cfo`` is wrapped into
        # [0, n_bins) and would flip the SFO sign and scale.
        sfo_hat = (
            _wrap_half(float(cfo), n_bins) * float(bw_hz) / float(center_frequency_hz)
        )
        candidates.append(
            CoherentCandidate(
                cfo_total_bins=cfo,
                sto_chips=float(sto),
                payload_start_sample=int(payload_start_hint)
                + int(round(sto * os_factor)),
                up_score=1.0,
                down_score=1.0 / (1.0 + rank),
                source="anchor_line" if rank == 0 else "anchor_offset",
                sfo_hat=sfo_hat,
                sfo_cum_initial=0.0,
            )
        )
    return candidates[: int(max_candidates)]

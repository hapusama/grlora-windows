"""Single-symbol LoRa detector using nested sample-rate structure.

The input is one perfectly aligned symbol sampled at ``source_os_factor * BW``.
Lower-rate views are deterministic phase-zero decimations of that same noisy
symbol.  No interpolation, prediction, denoising, or independent observations
are introduced.

For a candidate bin ``m`` and an integer oversampling factor ``q > 1``, the
full-rate dechirped FFT has two candidate components at ``m`` and ``m - N``
modulo ``qN``.  Their ideal amplitudes are proportional to ``N - m`` and ``m``.
At ``q = 1`` the components alias into the single bin ``m``.  This module makes
that invariant explicit and combines it across nested rates.

The multi-rate sum is a structured heuristic, not a likelihood formed from
independent measurements: all views contain reused samples and correlated
noise.  ``coherent_ml_scores`` is provided as the full-input AWGN upper control.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal, Sequence

import numpy as np

from ...chirp import build_upchirp
from .oversampled_glrt import fold_pair_steering


@dataclass(frozen=True)
class MultiRateSpectra:
    """Dechirped spectra derived from one high-rate symbol."""

    sf: int
    source_os_factor: int
    rates: tuple[int, ...]
    spectra: tuple[np.ndarray, ...]

    @property
    def n_bins(self) -> int:
        return 1 << int(self.sf)

    def spectrum(self, os_factor: int) -> np.ndarray:
        """Return the spectrum for one requested oversampling factor."""

        rate = int(os_factor)
        try:
            index = self.rates.index(rate)
        except ValueError as exc:
            raise KeyError(f"rate q={rate} is not present") from exc
        return self.spectra[index]


@dataclass(frozen=True)
class MultiRateDetectionResult:
    """Hard decision and diagnostics from the multi-rate structure score."""

    selected_bin: int
    scores: np.ndarray
    rate_scores: tuple[np.ndarray, ...]
    ratio_consistency: tuple[np.ndarray, ...]
    observation: MultiRateSpectra
    fusion_mode: str


def _validate_rates(
    source_os_factor: int,
    rates: Sequence[int],
) -> tuple[int, tuple[int, ...]]:
    source = int(source_os_factor)
    if source <= 0:
        raise ValueError("source_os_factor must be positive")
    values = tuple(int(value) for value in rates)
    if not values:
        raise ValueError("at least one rate is required")
    if len(set(values)) != len(values):
        raise ValueError("rates must be unique")
    for value in values:
        if value <= 0 or source % value:
            raise ValueError(
                f"rate q={value} must be a positive divisor of source q={source}"
            )
    return source, values


@lru_cache(maxsize=64)
def _downchirp(sf: int, os_factor: int) -> np.ndarray:
    reference = build_upchirp(int(sf), symbol_id=0, os_factor=int(os_factor))
    downchirp = np.conjugate(reference).astype(np.complex64)
    downchirp.flags.writeable = False
    return downchirp


@lru_cache(maxsize=64)
def _reference_spectrum(sf: int, os_factor: int) -> np.ndarray:
    reference = build_upchirp(int(sf), symbol_id=0, os_factor=int(os_factor))
    spectrum = np.fft.fft(reference)
    spectrum.flags.writeable = False
    return spectrum


def build_multirate_spectra(
    samples: np.ndarray,
    sf: int,
    source_os_factor: int = 8,
    rates: Sequence[int] = (8, 4, 2, 1),
) -> MultiRateSpectra:
    """Build normalized dechirped FFTs from nested phase-zero decimations.

    If ``source_os_factor=8``, the ``q=4,2,1`` views use strides ``2,4,8``.
    This intentionally performs no anti-alias filtering: every view remains a
    deterministic function of the exact same high-rate noisy input.
    """

    sf_value = int(sf)
    if sf_value <= 0:
        raise ValueError("sf must be positive")
    source, rate_values = _validate_rates(source_os_factor, rates)
    n_bins = 1 << sf_value
    values = np.asarray(samples, dtype=np.complex64)
    expected = n_bins * source
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"samples must have shape ({expected},), got {values.shape}")

    spectra: list[np.ndarray] = []
    for rate in rate_values:
        stride = source // rate
        view = np.asarray(values[::stride], dtype=np.complex64)
        if view.size != n_bins * rate:
            raise RuntimeError("nested decimation produced an unexpected symbol length")
        dechirped = view * _downchirp(sf_value, rate)
        spectrum = np.fft.fft(dechirped) / math.sqrt(float(dechirped.size))
        spectra.append(spectrum.astype(np.complex64))
    return MultiRateSpectra(sf_value, source, rate_values, tuple(spectra))


def fold_pair_components(
    spectrum: np.ndarray,
    sf: int,
    os_factor: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return candidate-aligned components separated by exactly ``N`` bins."""

    n_bins = 1 << int(sf)
    rate = int(os_factor)
    values = np.asarray(spectrum, dtype=np.complex64)
    expected = n_bins * rate
    if rate <= 0 or values.ndim != 1 or values.size != expected:
        raise ValueError(f"spectrum must have shape ({expected},)")
    primary = values[:n_bins]
    if rate == 1:
        return primary, None
    # m - N modulo qN is m + qN - N for m in [0, N).
    secondary = values[expected - n_bins : expected]
    return primary, secondary


def mapped_fft_argmax_scores(
    spectrum: np.ndarray,
    sf: int,
    os_factor: int,
) -> np.ndarray:
    """Naive FFT score: use only the stronger member of each legal pair."""

    primary, secondary = fold_pair_components(spectrum, sf, os_factor)
    primary_power = np.abs(primary).astype(np.float64) ** 2
    if secondary is None:
        return primary_power
    secondary_power = np.abs(secondary).astype(np.float64) ** 2
    return np.maximum(primary_power, secondary_power)


def fold_pair_energy_scores(
    spectrum: np.ndarray,
    sf: int,
    os_factor: int,
) -> np.ndarray:
    """Incoherently sum the two bins whose separation is the LoRa invariant."""

    primary, secondary = fold_pair_components(spectrum, sf, os_factor)
    scores = np.abs(primary).astype(np.float64) ** 2
    if secondary is not None:
        scores += np.abs(secondary).astype(np.float64) ** 2
    return scores


@lru_cache(maxsize=128)
def _fold_steering_bank(
    sf: int,
    os_factor: int,
    timing_offset_chips: float,
) -> np.ndarray:
    n_bins = 1 << int(sf)
    rate = int(os_factor)
    if rate <= 1:
        return np.ones((n_bins, 1), dtype=np.complex128)
    timing = float(timing_offset_chips)
    if timing == 0.0:
        candidate = np.arange(n_bins, dtype=np.float64)
        steering = np.stack((n_bins - candidate, candidate), axis=1)
        norms = np.linalg.norm(steering, axis=1, keepdims=True)
        return (steering / np.maximum(norms, 1e-30)).astype(np.complex128)
    return np.stack(
        [fold_pair_steering(candidate, sf, rate, timing) for candidate in range(n_bins)]
    ).astype(np.complex128)


def fold_profile_scores(
    spectrum: np.ndarray,
    sf: int,
    os_factor: int,
    timing_offsets_chips: Sequence[float] = (0.0,),
) -> np.ndarray:
    """Score the expected fold-length ratio and relative phase.

    The unknown common symbol phase is eliminated by magnitude-squaring the
    complex projection.  A small timing grid may be supplied to maximize over
    fractional STO; the perfect-synchronization AWGN control uses only zero.
    """

    offsets = tuple(float(value) for value in timing_offsets_chips)
    if not offsets:
        raise ValueError("timing_offsets_chips must not be empty")
    primary, secondary = fold_pair_components(spectrum, sf, os_factor)
    if secondary is None:
        return np.abs(primary).astype(np.float64) ** 2
    pairs = np.stack((primary, secondary), axis=1).astype(np.complex128)
    best = np.full(primary.size, -np.inf, dtype=np.float64)
    for timing in offsets:
        steering = _fold_steering_bank(int(sf), int(os_factor), timing)
        projection = np.sum(np.conjugate(steering) * pairs, axis=1)
        best = np.maximum(best, np.abs(projection).astype(np.float64) ** 2)
    return best


def fold_ratio_consistency(
    spectrum: np.ndarray,
    sf: int,
    os_factor: int,
    timing_offset_chips: float = 0.0,
) -> np.ndarray:
    """Return amplitude-profile cosine similarity in ``[0, 1]``.

    This diagnostic isolates the proposed second evidence: the observed peak
    amplitude ratio versus the ratio predicted by the symbol wrap position.
    It is not added as an independent likelihood term by itself.
    """

    primary, secondary = fold_pair_components(spectrum, sf, os_factor)
    if secondary is None:
        return np.ones(primary.size, dtype=np.float64)
    observed = np.stack((np.abs(primary), np.abs(secondary)), axis=1).astype(
        np.float64
    )
    expected = np.abs(
        _fold_steering_bank(int(sf), int(os_factor), float(timing_offset_chips))
    )
    numerator = np.sum(observed * expected, axis=1) ** 2
    denominator = np.sum(observed * observed, axis=1) * np.sum(
        expected * expected, axis=1
    )
    return np.clip(numerator / np.maximum(denominator, 1e-30), 0.0, 1.0)


def multirate_structure_scores(
    observation: MultiRateSpectra,
    timing_offsets_chips: Sequence[float] = (0.0,),
    rate_weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Combine per-rate fold-profile scores without assuming independence."""

    if rate_weights is None:
        weights = np.ones(len(observation.rates), dtype=np.float64)
    else:
        weights = np.asarray(tuple(float(value) for value in rate_weights), dtype=np.float64)
        if weights.ndim != 1 or weights.size != len(observation.rates):
            raise ValueError("rate_weights length must match observation.rates")
        if np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("rate_weights must be non-negative with at least one positive")
    rate_scores = tuple(
        fold_profile_scores(
            spectrum,
            observation.sf,
            rate,
            timing_offsets_chips=timing_offsets_chips,
        )
        for rate, spectrum in zip(
            observation.rates, observation.spectra, strict=True
        )
    )
    combined = np.zeros(observation.n_bins, dtype=np.float64)
    for weight, scores in zip(weights, rate_scores, strict=True):
        combined += float(weight) * scores
    return combined, rate_scores


def awgn_multirate_glrt_scores(
    observation: MultiRateSpectra,
    timing_offsets_chips: Sequence[float] = (0.0,),
) -> np.ndarray:
    """Return the covariance-aware nested-rate GLRT under source-rate AWGN.

    The lower-rate FFT components reuse source-rate samples.  If their full
    cross-rate covariance is whitened, the conditional lower-rate residuals
    contain no candidate signal beyond the source-rate legal fold pair.  The
    stacked multi-rate GLRT therefore collapses exactly to the source-rate
    fold-profile projection; summing per-rate powers would double-count noise.

    Keeping this closed form in the public API makes the no-information-gain
    boundary executable while retaining ``O(N)`` scoring complexity.
    """

    source = int(observation.source_os_factor)
    if source not in observation.rates:
        raise ValueError("observation must include its source_os_factor rate")
    return fold_profile_scores(
        observation.spectrum(source),
        observation.sf,
        source,
        timing_offsets_chips=timing_offsets_chips,
    )


def detect_multirate_structure(
    samples: np.ndarray,
    sf: int,
    source_os_factor: int = 8,
    rates: Sequence[int] = (8, 4, 2, 1),
    timing_offsets_chips: Sequence[float] = (0.0,),
    rate_weights: Sequence[float] | None = None,
    fusion_mode: Literal["awgn_glrt", "equal_sum"] = "awgn_glrt",
) -> MultiRateDetectionResult:
    """Detect one symbol using a low-complexity nested-rate score.

    ``awgn_glrt`` is the safe default and avoids double-counting correlated
    views.  ``equal_sum`` is retained only for the explicit fusion ablation.
    """

    observation = build_multirate_spectra(
        samples,
        sf,
        source_os_factor=source_os_factor,
        rates=rates,
    )
    equal_scores, rate_scores = multirate_structure_scores(
        observation,
        timing_offsets_chips=timing_offsets_chips,
        rate_weights=rate_weights,
    )
    mode = str(fusion_mode)
    if mode == "awgn_glrt":
        scores = awgn_multirate_glrt_scores(
            observation,
            timing_offsets_chips=timing_offsets_chips,
        )
    elif mode == "equal_sum":
        scores = equal_scores
    else:
        raise ValueError(f"unknown fusion_mode: {fusion_mode}")
    ratio = tuple(
        fold_ratio_consistency(spectrum, sf, rate)
        for rate, spectrum in zip(
            observation.rates, observation.spectra, strict=True
        )
    )
    return MultiRateDetectionResult(
        selected_bin=int(np.argmax(scores)),
        scores=scores,
        rate_scores=rate_scores,
        ratio_consistency=ratio,
        observation=observation,
        fusion_mode=mode,
    )


def coherent_ml_scores(
    samples: np.ndarray,
    sf: int,
    os_factor: int,
) -> np.ndarray:
    """Return full-symbol coherent AWGN ML scores using circular correlation.

    LoRa candidates are cyclic shifts of the reference chirp by ``m*q``
    samples, up to an irrelevant common phase.  One circular correlation thus
    evaluates the complete template bank in ``O(qN log(qN))`` time.
    """

    n_bins = 1 << int(sf)
    rate = int(os_factor)
    length = n_bins * rate
    values = np.asarray(samples, dtype=np.complex64)
    if rate <= 0 or values.ndim != 1 or values.size != length:
        raise ValueError(f"samples must have shape ({length},)")
    correlation = np.fft.ifft(
        np.fft.fft(values) * np.conjugate(_reference_spectrum(int(sf), rate))
    )
    locations = (-np.arange(n_bins, dtype=np.int64) * rate) % length
    return (np.abs(correlation[locations]).astype(np.float64) ** 2) / float(length)

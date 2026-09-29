"""Low-complexity robust LoRa support detection for sparse corruption.

The ordinary Savaux receiver remains the first-stage detector.  This module
only revisits a small Top-K candidate set when the best Savaux candidate leaves
an implausibly large fraction of sample-domain residual outliers.  For every
candidate, the exact oversampled LoRa phase law is removed and the remaining
common complex amplitude is fitted with a few Huber IRLS updates.

Under ordinary AWGN the outlier gate stays closed and the hard decision is
therefore *exactly* Savaux.  Under bursty or impulsive interference, the Huber
fit downweights the corrupted samples and can rerank the candidate support
without reconstructing a high-rate waveform or running a general LASSO.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)
from ..os_lora.system.nonuniform_sampling import prepare_dechirped_symbol
from .phase_templates import dechirped_candidate_template


@dataclass(frozen=True)
class RobustSparseConfig:
    """Bounded-complexity settings for robust Top-K support detection."""

    candidate_count: int = 16
    huber_delta: float = 2.5
    irls_iterations: int = 3
    min_outlier_fraction: float = 0.02
    min_robust_gain_db: float = 3.0

    def validate(self) -> None:
        if int(self.candidate_count) <= 0:
            raise ValueError("candidate_count must be positive")
        if not math.isfinite(float(self.huber_delta)) or float(self.huber_delta) <= 0.0:
            raise ValueError("huber_delta must be positive")
        if int(self.irls_iterations) <= 0:
            raise ValueError("irls_iterations must be positive")
        if not 0.0 <= float(self.min_outlier_fraction) <= 1.0:
            raise ValueError("min_outlier_fraction must be in [0, 1]")
        if not math.isfinite(float(self.min_robust_gain_db)):
            raise ValueError("min_robust_gain_db must be finite")


@dataclass(frozen=True)
class RobustSparseResult:
    """One robust decision plus diagnostics needed by experiment runners."""

    selected_bin: int
    savaux_bin: int
    changed_from_savaux: bool
    gate_triggered: bool
    outlier_fraction: float
    noise_scale: float
    robust_gain_db: float
    candidate_bins: tuple[int, ...]
    robust_scores: tuple[float, ...]


def _median_complex(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values)
    return np.median(array.real, axis=axis) + 1j * np.median(array.imag, axis=axis)


def _huber_objective(normalized_magnitude: np.ndarray, delta: float) -> np.ndarray:
    magnitude = np.asarray(normalized_magnitude, dtype=np.float64)
    threshold = float(delta)
    terms = np.where(
        magnitude <= threshold,
        0.5 * magnitude * magnitude,
        threshold * (magnitude - 0.5 * threshold),
    )
    return np.sum(terms, axis=-1, dtype=np.float64)


def _fit_locations(
    derotated: np.ndarray,
    noise_scale: float,
    delta: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one common complex amplitude per candidate with Huber IRLS."""

    values = np.asarray(derotated, dtype=np.complex128)
    locations = np.asarray(_median_complex(values, axis=1), dtype=np.complex128)
    threshold = max(float(delta) * float(noise_scale), 1e-15)
    weights = np.ones(values.shape, dtype=np.float64)
    for _ in range(int(iterations)):
        residual = np.abs(values - locations[:, None])
        weights = np.minimum(1.0, threshold / np.maximum(residual, 1e-30))
        denominator = np.sum(weights, axis=1, dtype=np.float64)
        locations = np.sum(weights * values, axis=1) / np.maximum(denominator, 1e-30)
    return locations, weights


def robust_sparse_rerank(
    dechirped: np.ndarray,
    savaux_spectrum: np.ndarray,
    sf: int,
    os_factor: int,
    config: RobustSparseConfig | None = None,
) -> RobustSparseResult:
    """Rerank Savaux Top-K candidates when sparse residual corruption is present.

    Ground truth, payload bytes, FEC and CRC are deliberately absent from this
    API.  The only inputs are the current symbol samples and Savaux spectrum.
    """

    cfg = config or RobustSparseConfig()
    cfg.validate()
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    symbol = np.asarray(dechirped, dtype=np.complex128)
    spectrum = np.asarray(savaux_spectrum)
    if symbol.ndim != 1 or symbol.size != n_bins * os_value:
        raise ValueError("dechirped symbol length mismatch")
    if spectrum.ndim != 1 or spectrum.size != n_bins:
        raise ValueError("savaux_spectrum length mismatch")

    power = np.abs(spectrum).astype(np.float64) ** 2
    savaux_bin = int(np.argmax(power))
    keep = min(n_bins, int(cfg.candidate_count))
    candidate_bins = np.argsort(power)[::-1][:keep].astype(np.int64)
    # np.argsort already places the argmax first, but keep the invariant
    # explicit because all gate diagnostics are defined on candidate zero.
    if int(candidate_bins[0]) != savaux_bin:
        candidate_bins = np.concatenate(
            (np.asarray([savaux_bin], dtype=np.int64), candidate_bins[candidate_bins != savaux_bin])
        )[:keep]

    # The gate is intentionally evaluated before constructing or fitting the
    # remaining candidates.  Normal AWGN therefore pays only one O(N) robust
    # fit; the O(K*N*R) reranker is activated solely for sparse corruption.
    base_template = dechirped_candidate_template(
        int(sf), os_value, savaux_bin
    ).astype(
        np.complex128, copy=False
    )
    base_derotated = symbol * np.conjugate(base_template)
    base_center = complex(_median_complex(base_derotated))
    base_residual = np.abs(base_derotated - base_center)
    # For circular complex Gaussian noise, median(|w|) = sigma*sqrt(log(2))
    # when sigma^2 is E|w|^2.  A tiny signal-relative floor handles clean IQ.
    median_radius = float(np.median(base_residual))
    signal_rms = float(np.sqrt(np.mean(np.abs(symbol) ** 2, dtype=np.float64)))
    noise_scale = max(
        median_radius / math.sqrt(math.log(2.0)),
        signal_rms * 1e-6,
        1e-12,
    )

    base_locations, _base_weights = _fit_locations(
        base_derotated[None, :],
        noise_scale=noise_scale,
        delta=float(cfg.huber_delta),
        iterations=int(cfg.irls_iterations),
    )
    base_final_residual = np.abs(base_derotated - base_locations[0])
    outlier_fraction = float(
        np.mean(base_final_residual > float(cfg.huber_delta) * noise_scale)
    )
    gate_triggered = bool(outlier_fraction >= float(cfg.min_outlier_fraction))
    if not gate_triggered:
        return RobustSparseResult(
            selected_bin=savaux_bin,
            savaux_bin=savaux_bin,
            changed_from_savaux=False,
            gate_triggered=False,
            outlier_fraction=outlier_fraction,
            noise_scale=noise_scale,
            robust_gain_db=0.0,
            candidate_bins=tuple(int(v) for v in candidate_bins),
            # Empty means the adaptive gate skipped Top-K scoring.
            robust_scores=(),
        )

    templates = np.stack(
        [
            dechirped_candidate_template(int(sf), os_value, int(k))
            for k in candidate_bins
        ]
    ).astype(np.complex128, copy=False)
    derotated = symbol[None, :] * np.conjugate(templates)
    locations, _weights = _fit_locations(
        derotated,
        noise_scale=noise_scale,
        delta=float(cfg.huber_delta),
        iterations=int(cfg.irls_iterations),
    )
    normalized_null = np.abs(derotated) / noise_scale
    normalized_fit = np.abs(derotated - locations[:, None]) / noise_scale
    null_objective = _huber_objective(normalized_null, float(cfg.huber_delta))
    fit_objective = _huber_objective(normalized_fit, float(cfg.huber_delta))
    scores = np.maximum(0.0, null_objective - fit_objective)

    best_index = int(np.argmax(scores))
    robust_gain_db = float(
        10.0
        * math.log10(
            (float(scores[best_index]) + 1e-30) / (float(scores[0]) + 1e-30)
        )
    )
    accept = bool(
        gate_triggered
        and best_index != 0
        and robust_gain_db >= float(cfg.min_robust_gain_db)
    )
    selected_bin = int(candidate_bins[best_index]) if accept else savaux_bin
    return RobustSparseResult(
        selected_bin=selected_bin,
        savaux_bin=savaux_bin,
        changed_from_savaux=bool(selected_bin != savaux_bin),
        gate_triggered=gate_triggered,
        outlier_fraction=outlier_fraction,
        noise_scale=noise_scale,
        robust_gain_db=robust_gain_db,
        candidate_bins=tuple(int(v) for v in candidate_bins),
        robust_scores=tuple(float(v) for v in scores),
    )


def demod_robust_sparse_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: str = "none",
    config: RobustSparseConfig | None = None,
) -> RobustSparseResult:
    """Compute Savaux evidence and apply the gated robust Top-K detector."""

    spectrum, _branches, _phase = paper_oversampled_spectrum(
        samples=samples,
        start_sample=int(start_sample),
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    dechirped = prepare_dechirped_symbol(
        samples=samples,
        start_sample=int(start_sample),
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    return robust_sparse_rerank(
        dechirped=dechirped,
        savaux_spectrum=spectrum,
        sf=int(sf),
        os_factor=int(os_factor),
        config=config,
    )


__all__ = [
    "RobustSparseConfig",
    "RobustSparseResult",
    "demod_robust_sparse_symbol",
    "robust_sparse_rerank",
]

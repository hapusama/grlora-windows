"""ChirpSVD experiments for oversampled LoRa symbols.

The core object is one dechirped oversampled symbol reshaped as an
``N x R`` matrix:

    X[p, q] = z[p * R + q]

For an ideal dechirped LoRa tone this matrix has a strong low-rank structure.
This module exposes two conservative ways to use that structure:

* score the leading left singular vector as a chip-time tone;
* reconstruct a rank-r denoised matrix, then run the existing Savaux
  oversampled spectrum on the denoised branches.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..baselines.savaux_oversampled.paper_oversampled_demod import (
    _paper_branch_spectrum,
    combine_paper_branch_spectra,
)


@dataclass(frozen=True)
class ChirpSVDSpectra:
    """Spectra derived from one SVD of a dechirped oversampled symbol."""

    singular_values: tuple[float, ...]
    rank1_ratio: float
    rank2_ratio: float
    svd_left_spectrum: np.ndarray
    rank_savaux_spectra: dict[int, np.ndarray]


def _validate_shape(dechirped: np.ndarray, sf: int, os_factor: int) -> tuple[np.ndarray, int, int]:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    if os_value <= 0:
        raise ValueError(f"os_factor must be positive, got {os_factor}")
    symbol = np.asarray(dechirped, dtype=np.complex64)
    expected = n_bins * os_value
    if symbol.size != expected:
        raise ValueError(f"dechirped symbol has {symbol.size} samples, expected {expected}")
    return symbol, n_bins, os_value


def chirp_matrix(dechirped: np.ndarray, sf: int, os_factor: int) -> np.ndarray:
    """Return the ``N x R`` ChirpSVD matrix for one dechirped symbol."""

    symbol, n_bins, os_value = _validate_shape(dechirped, sf, os_factor)
    return np.asarray(symbol.reshape(n_bins, os_value), dtype=np.complex128)


def low_rank_chirp_matrix(matrix: np.ndarray, rank: int) -> np.ndarray:
    """Return the rank-r truncated SVD reconstruction of ``matrix``."""

    mat = np.asarray(matrix, dtype=np.complex128)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2-D")
    max_rank = min(mat.shape)
    keep = max(1, min(int(rank), max_rank))
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    return np.asarray((u[:, :keep] * s[:keep]) @ vh[:keep, :], dtype=np.complex128)


def savaux_spectrum_from_dechirped_matrix(matrix: np.ndarray, sf: int, os_factor: int) -> np.ndarray:
    """Run the existing Savaux branch spectrum on an already-dechirped matrix."""

    mat = np.asarray(matrix, dtype=np.complex128)
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    if mat.shape != (n_bins, os_value):
        raise ValueError(f"matrix shape {mat.shape} does not match {(n_bins, os_value)}")
    branch_spectra = tuple(
        _paper_branch_spectrum(
            dechirped_branch=np.asarray(mat[:, q], dtype=np.complex64),
            sf=int(sf),
            os_factor=os_value,
            branch_index=q,
        )
        for q in range(os_value)
    )
    return combine_paper_branch_spectra(branch_spectra, os_factor=os_value).astype(np.complex64)


def chirp_svd_spectra(
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
    ranks: Sequence[int] = (1, 2),
) -> ChirpSVDSpectra:
    """Compute ChirpSVD spectra for one dechirped oversampled symbol."""

    matrix = chirp_matrix(dechirped, sf, os_factor)
    n_bins = 1 << int(sf)
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    singular = np.asarray(s, dtype=np.float64)
    total_energy = float(np.sum(singular ** 2))
    rank1_ratio = float((singular[0] ** 2) / max(total_energy, 1e-30)) if singular.size else 0.0
    rank2_ratio = (
        float(np.sum(singular[:2] ** 2) / max(total_energy, 1e-30))
        if singular.size
        else 0.0
    )

    if singular.size:
        # u[:, 0] is unit-norm, so scale by sigma_1.  The normalization keeps
        # the spectrum comparable across SF while preserving argmax.
        svd_left = (singular[0] * np.fft.fft(u[:, 0]) / math.sqrt(float(n_bins))).astype(np.complex64)
    else:
        svd_left = np.zeros(n_bins, dtype=np.complex64)

    rank_spectra: dict[int, np.ndarray] = {}
    for rank in ranks:
        keep = max(1, min(int(rank), min(matrix.shape)))
        reconstructed = np.asarray((u[:, :keep] * s[:keep]) @ vh[:keep, :], dtype=np.complex128)
        rank_spectra[int(rank)] = savaux_spectrum_from_dechirped_matrix(
            reconstructed,
            sf=int(sf),
            os_factor=int(os_factor),
        )

    return ChirpSVDSpectra(
        singular_values=tuple(float(v) for v in singular),
        rank1_ratio=rank1_ratio,
        rank2_ratio=rank2_ratio,
        svd_left_spectrum=svd_left,
        rank_savaux_spectra=rank_spectra,
    )


__all__ = [
    "ChirpSVDSpectra",
    "chirp_matrix",
    "chirp_svd_spectra",
    "low_rank_chirp_matrix",
    "savaux_spectrum_from_dechirped_matrix",
]

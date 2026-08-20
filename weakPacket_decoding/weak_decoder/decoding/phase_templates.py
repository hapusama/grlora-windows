"""Exact sample-domain phase templates shared by weak demodulators."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1024)
def dechirped_candidate_template(
    sf: int,
    os_factor: int,
    raw_fft_bin: int,
) -> np.ndarray:
    """Return one exact oversampled dechirped LoRa candidate.

    The suffix phase accounts for the LoRa frequency wrap.  A plain complex
    tone is insufficient for non-zero oversampling phases near that wrap.
    """

    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    if os_value <= 0:
        raise ValueError("os_factor must be positive")
    k = int(raw_fft_bin) % n_bins
    p = np.repeat(np.arange(n_bins, dtype=np.float64), os_value)
    q = np.tile(np.arange(os_value, dtype=np.float64), n_bins)
    phase = np.exp(
        2j * np.pi * float(k) * (os_value * p + q) / float(n_bins * os_value)
    )
    if k != 0:
        wrapped = p >= float(n_bins - k)
        phase[wrapped] *= np.exp(-2j * np.pi * q[wrapped] / float(os_value))
    output = np.asarray(phase, dtype=np.complex64)
    output.flags.writeable = False
    return output


__all__ = ["dechirped_candidate_template"]

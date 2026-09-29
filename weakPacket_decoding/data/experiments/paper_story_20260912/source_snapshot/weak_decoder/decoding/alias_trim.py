"""Low-complexity removal of aliased narrowband trajectories for LoRa.

After LoRa dechirping, a desired symbol is a horizontal tone while a sampled
narrowband blocker follows the conjugate chirp trajectory.  Rather than form
and edit a dense STFT image, AliasTrim applies the matched inverse trajectory:
it removes the provisional LoRa tone, rotates the residual by the conjugate
downchirp, and detects the resulting stationary blocker with a zero-padded
FFT.  Verified tones are least-squares subtracted and Savaux is rerun.

This operates entirely at the available sample rate.  It cannot undo folded
wideband noise and it does not infer the blocker's pre-alias RF frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..baselines.savaux_oversampled.paper_oversampled_demod import (
    _paper_branch_spectrum,
    combine_paper_branch_spectra,
    paper_oversampled_spectrum,
)
from ..chirp import build_upchirp
from ..os_lora.system.nonuniform_sampling import prepare_dechirped_symbol
from .phase_templates import dechirped_candidate_template


@dataclass(frozen=True)
class AliasTrimConfig:
    """Conservative detector and cancellation settings."""

    max_blockers: int = 2
    fft_oversampling: int = 4
    min_tone_prominence_db: float = 18.0
    min_tone_power_fraction: float = 0.02
    min_clean_gain_db: float = 3.0

    def validate(self) -> None:
        if int(self.max_blockers) <= 0:
            raise ValueError("max_blockers must be positive")
        if int(self.fft_oversampling) <= 0:
            raise ValueError("fft_oversampling must be positive")
        if not math.isfinite(float(self.min_tone_prominence_db)):
            raise ValueError("min_tone_prominence_db must be finite")
        if not 0.0 <= float(self.min_tone_power_fraction) <= 1.0:
            raise ValueError("min_tone_power_fraction must be in [0, 1]")
        if not math.isfinite(float(self.min_clean_gain_db)):
            raise ValueError("min_clean_gain_db must be finite")


@dataclass(frozen=True)
class AliasBlocker:
    """One detected conjugate-chirp trajectory and its sampled tone model."""

    normalized_frequency: float
    amplitude_real: float
    amplitude_imag: float
    prominence_db: float
    residual_power_fraction: float


@dataclass(frozen=True)
class AliasTrimResult:
    """AliasTrim decision and gate diagnostics for one LoRa symbol."""

    selected_bin: int
    savaux_bin: int
    cleaned_bin: int
    changed_from_savaux: bool
    gate_triggered: bool
    clean_gain_db: float
    strongest_tone_prominence_db: float
    strongest_tone_power_fraction: float
    blockers: tuple[AliasBlocker, ...]


def _complex_median(values: np.ndarray) -> complex:
    array = np.asarray(values)
    return complex(float(np.median(array.real)), float(np.median(array.imag)))


def _oversampled_downchirp(
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    sample_count = n_bins * os_value
    indexes = np.arange(sample_count, dtype=np.float64)
    reference = build_upchirp(
        sf=int(sf), symbol_id=int(cfo_int), os_factor=os_value
    )
    fractional = np.exp(
        -2j * np.pi * float(cfo_frac) * indexes / float(sample_count)
    )
    return (np.conjugate(reference) * fractional).astype(np.complex64)


def _savaux_spectrum_from_dechirped(
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    symbol = np.asarray(dechirped, dtype=np.complex64)
    if symbol.ndim != 1 or symbol.size != n_bins * os_value:
        raise ValueError("dechirped symbol length mismatch")
    branches = tuple(
        _paper_branch_spectrum(
            dechirped_branch=symbol[q::os_value],
            sf=int(sf),
            os_factor=os_value,
            branch_index=q,
        )
        for q in range(os_value)
    )
    return combine_paper_branch_spectra(branches, os_factor=os_value)


def _refined_tone(
    residual: np.ndarray,
    fft_oversampling: int,
) -> tuple[float, complex, float, float]:
    """Return frequency, LS amplitude, FFT prominence, and power fraction."""

    values = np.asarray(residual, dtype=np.complex128)
    sample_count = int(values.size)
    fft_size = int(fft_oversampling) * sample_count
    spectrum = np.fft.fft(values, n=fft_size)
    power = np.abs(spectrum).astype(np.float64) ** 2
    peak = int(np.argmax(power))
    left = (peak - 1) % fft_size
    right = (peak + 1) % fft_size
    log_power = np.log(power + 1e-30)
    denominator = float(
        log_power[left] - 2.0 * log_power[peak] + log_power[right]
    )
    delta = 0.0
    if abs(denominator) > 1e-15:
        delta = float(
            0.5 * (log_power[left] - log_power[right]) / denominator
        )
        delta = float(np.clip(delta, -0.5, 0.5))
    frequency = float((float(peak) + delta) / float(fft_size))
    if frequency >= 0.5:
        frequency -= 1.0
    indexes = np.arange(sample_count, dtype=np.float64)
    tone = np.exp(2j * np.pi * frequency * indexes)
    amplitude = complex(np.mean(values * np.conjugate(tone)))
    residual_power = float(np.mean(np.abs(values) ** 2, dtype=np.float64))
    power_fraction = float(abs(amplitude) ** 2 / (residual_power + 1e-30))
    prominence_db = float(
        10.0
        * math.log10(
            (float(power[peak]) + 1e-30) / (float(np.median(power)) + 1e-30)
        )
    )
    return frequency, amplitude, prominence_db, power_fraction


def alias_trim_rerank(
    dechirped: np.ndarray,
    savaux_spectrum: np.ndarray,
    downchirp: np.ndarray,
    sf: int,
    os_factor: int,
    config: AliasTrimConfig | None = None,
) -> AliasTrimResult:
    """Detect conjugate-chirp blocker trajectories and rerun Savaux."""

    cfg = config or AliasTrimConfig()
    cfg.validate()
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    sample_count = n_bins * os_value
    symbol = np.asarray(dechirped, dtype=np.complex128)
    spectrum = np.asarray(savaux_spectrum)
    local_downchirp = np.asarray(downchirp, dtype=np.complex128)
    if symbol.ndim != 1 or symbol.size != sample_count:
        raise ValueError("dechirped symbol length mismatch")
    if spectrum.ndim != 1 or spectrum.size != n_bins:
        raise ValueError("savaux_spectrum length mismatch")
    if local_downchirp.ndim != 1 or local_downchirp.size != sample_count:
        raise ValueError("downchirp length mismatch")

    savaux_power = np.abs(spectrum).astype(np.float64) ** 2
    savaux_bin = int(np.argmax(savaux_power))
    desired_template = dechirped_candidate_template(
        int(sf), os_value, savaux_bin
    ).astype(np.complex128, copy=False)
    desired_amplitude = _complex_median(
        symbol * np.conjugate(desired_template)
    )
    residual_dechirped = symbol - desired_amplitude * desired_template
    raw_residual = residual_dechirped * np.conjugate(local_downchirp)
    indexes = np.arange(sample_count, dtype=np.float64)
    removed_raw = np.zeros(sample_count, dtype=np.complex128)
    working = np.asarray(raw_residual, dtype=np.complex128).copy()
    blockers: list[AliasBlocker] = []
    strongest_prominence_db = 0.0
    strongest_power_fraction = 0.0

    for blocker_index in range(int(cfg.max_blockers)):
        frequency, amplitude, prominence_db, power_fraction = _refined_tone(
            working, fft_oversampling=int(cfg.fft_oversampling)
        )
        if blocker_index == 0:
            strongest_prominence_db = prominence_db
            strongest_power_fraction = power_fraction
        if (
            prominence_db < float(cfg.min_tone_prominence_db)
            or power_fraction < float(cfg.min_tone_power_fraction)
        ):
            break
        tone = amplitude * np.exp(2j * np.pi * frequency * indexes)
        working -= tone
        removed_raw += tone
        blockers.append(
            AliasBlocker(
                normalized_frequency=frequency,
                amplitude_real=float(amplitude.real),
                amplitude_imag=float(amplitude.imag),
                prominence_db=prominence_db,
                residual_power_fraction=power_fraction,
            )
        )

    if not blockers:
        return AliasTrimResult(
            selected_bin=savaux_bin,
            savaux_bin=savaux_bin,
            cleaned_bin=savaux_bin,
            changed_from_savaux=False,
            gate_triggered=False,
            clean_gain_db=0.0,
            strongest_tone_prominence_db=strongest_prominence_db,
            strongest_tone_power_fraction=strongest_power_fraction,
            blockers=(),
        )

    cleaned_dechirped = symbol - removed_raw * local_downchirp
    cleaned_spectrum = _savaux_spectrum_from_dechirped(
        cleaned_dechirped, sf=int(sf), os_factor=os_value
    )
    cleaned_power = np.abs(cleaned_spectrum).astype(np.float64) ** 2
    cleaned_bin = int(np.argmax(cleaned_power))
    clean_gain_db = float(
        10.0
        * math.log10(
            (float(cleaned_power[cleaned_bin]) + 1e-30)
            / (float(cleaned_power[savaux_bin]) + 1e-30)
        )
    )
    accept = bool(
        cleaned_bin != savaux_bin
        and clean_gain_db >= float(cfg.min_clean_gain_db)
    )
    selected_bin = cleaned_bin if accept else savaux_bin
    return AliasTrimResult(
        selected_bin=int(selected_bin),
        savaux_bin=savaux_bin,
        cleaned_bin=cleaned_bin,
        changed_from_savaux=bool(selected_bin != savaux_bin),
        gate_triggered=True,
        clean_gain_db=clean_gain_db,
        strongest_tone_prominence_db=strongest_prominence_db,
        strongest_tone_power_fraction=strongest_power_fraction,
        blockers=tuple(blockers),
    )


def demod_alias_trim_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: str = "none",
    config: AliasTrimConfig | None = None,
) -> AliasTrimResult:
    """Run Savaux followed by the gated AliasTrim cancellation path."""

    mode = str(cfo_correction_mode)
    if mode not in {"none", "symbol", "continuous"}:
        raise ValueError(f"unknown CFO correction mode: {mode}")
    use_cfo_int = int(cfo_int) if mode in {"symbol", "continuous"} else 0
    use_cfo_frac = float(cfo_frac) if mode in {"symbol", "continuous"} else 0.0
    spectrum, _branches, _common_phase = paper_oversampled_spectrum(
        samples=samples,
        start_sample=int(start_sample),
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=mode,
    )
    dechirped = prepare_dechirped_symbol(
        samples=samples,
        start_sample=int(start_sample),
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=mode,
    )
    downchirp = _oversampled_downchirp(
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=use_cfo_int,
        cfo_frac=use_cfo_frac,
    )
    return alias_trim_rerank(
        dechirped=dechirped,
        savaux_spectrum=spectrum,
        downchirp=downchirp,
        sf=int(sf),
        os_factor=int(os_factor),
        config=config,
    )


__all__ = [
    "AliasBlocker",
    "AliasTrimConfig",
    "AliasTrimResult",
    "alias_trim_rerank",
    "demod_alias_trim_symbol",
]

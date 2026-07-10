"""Non-uniform oversampling-pattern evidence for LoRa demodulation.

The experiment here stays at the demodulation/evidence layer.  For every
candidate raw FFT bin k, it evaluates a small bank of deterministic
non-uniform sampling patterns c_b[p] over the oversampled, dechirped symbol:

    n[p] = R * p + c_b[p]

Each pattern produces one matched DFT value for k.  The caller can inspect
individual pattern energy, an equal-phase coherent sum, or a covariance-whitened
coherent sum.  No CRC, payload prior, or cross-packet template is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Literal, Sequence

import numpy as np

from ..chirp import build_upchirp
from ..baselines.savaux_oversampled.paper_oversampled_demod import (
    _oversampled_downchirp,
    paper_oversampled_spectrum,
)


CfoCorrectionMode = Literal["none", "symbol", "continuous"]


@dataclass(frozen=True)
class NonuniformPatternBank:
    """A deterministic bank of non-uniform OSR sampling patterns."""

    names: tuple[str, ...]
    offsets: tuple[np.ndarray, ...]
    os_factor: int
    sf: int
    kind: str = "custom"


@dataclass(frozen=True)
class NonuniformScoreResult:
    """Per-candidate evidence from a non-uniform pattern bank."""

    raw_fft_bin: int
    savaux_power: float
    best_pattern_power: float
    mean_pattern_power: float
    coherent_pattern_power: float
    whitened_pattern_power: float
    stable_pattern_power: float
    hybrid_mean_power: float
    best_pattern_name: str
    best_pattern_gain_vs_savaux: float
    coherent_gain_vs_savaux: float
    whitened_gain_vs_savaux: float
    stable_gain_vs_savaux: float
    hybrid_gain_vs_savaux: float
    pattern_values: tuple[complex, ...]


@dataclass(frozen=True)
class MatrixFreeGLSResult:
    """Cross-fit GLS power with conjugate-gradient diagnostics."""

    power: np.ndarray
    iterations: tuple[int, ...]
    relative_residuals: tuple[float, ...]
    inverse_targets: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PatternSubsetSelection:
    """Covariance-aware greedy pattern selection diagnostics."""

    bank: NonuniformPatternBank
    indices: tuple[int, ...]
    marginal_information: tuple[float, ...]


@dataclass(frozen=True)
class ConditionalLoRaGLSResult:
    """Decision and stage diagnostics from the conditional LoRa detector."""

    raw_fft_bin: int
    savaux_bin: int
    savaux_margin_db: float
    branch_color_mismatch: float
    screened: bool
    backend_ran: bool
    cg_iterations: tuple[int, ...]


def _validate_os_factor(os_factor: int) -> int:
    value = int(os_factor)
    if value <= 0:
        raise ValueError(f"os_factor must be positive, got {os_factor}")
    return value


def _top_bins(power: np.ndarray, top_k: int) -> tuple[int, ...]:
    values = np.asarray(power, dtype=np.float64)
    if values.size == 0:
        return tuple()
    k = min(max(1, int(top_k)), int(values.size))
    if k >= values.size:
        return tuple(int(v) for v in np.argsort(values)[::-1])
    partial = np.argpartition(values, -k)[-k:]
    order = partial[np.argsort(values[partial])[::-1]]
    return tuple(int(v) for v in order)


def _pattern_key(offsets: np.ndarray) -> bytes:
    return np.asarray(offsets, dtype=np.int16).tobytes()


@lru_cache(maxsize=64)
def build_pattern_bank(
    sf: int,
    os_factor: int,
    kind: str = "basic",
    random_count: int = 64,
    seed: int = 0,
) -> NonuniformPatternBank:
    """Build a compact deterministic non-uniform sampling-pattern bank.

    `fixed` is the classical Savaux branch set.  Other banks include the fixed
    branches plus deterministic non-uniform candidates, so their matrix score
    can never lose the classical observations unless the caller asks for a
    custom bank.
    """

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    p = np.arange(n_bins, dtype=np.int64)
    requested_mode = str(kind).strip().lower()
    include_fixed = not requested_mode.endswith("_only")
    mode = requested_mode[:-5] if requested_mode.endswith("_only") else requested_mode
    names: list[str] = []
    patterns: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add(name: str, offsets: np.ndarray) -> None:
        vec = np.mod(np.asarray(offsets, dtype=np.int64), os_value).astype(np.int16)
        if vec.size != n_bins:
            raise ValueError("pattern length mismatch")
        key = _pattern_key(vec)
        if key in seen:
            return
        seen.add(key)
        names.append(str(name))
        patterns.append(vec.astype(np.int64))

    if include_fixed or mode == "fixed":
        for q in range(os_value):
            add(f"fixed_{q}", np.full(n_bins, q, dtype=np.int64))

    if mode == "fixed":
        return NonuniformPatternBank(
            names=tuple(names),
            offsets=tuple(patterns),
            os_factor=os_value,
            sf=int(sf),
            kind=requested_mode,
        )

    if mode in {"basic", "search", "super_search"}:
        for step in range(1, os_value):
            for start in range(os_value):
                add(f"linear_s{step}_b{start}", start + step * p)

        for q0 in range(os_value):
            for q1 in range(os_value):
                if q0 != q1:
                    add(f"alt_{q0}{q1}", np.where((p & 1) == 0, q0, q1))

        for split_num, split_den in ((1, 4), (1, 2), (3, 4)):
            split = int(round(n_bins * split_num / split_den))
            for q0 in range(os_value):
                for q1 in range(os_value):
                    if q0 == q1:
                        continue
                    vec = np.empty(n_bins, dtype=np.int64)
                    vec[:split] = q0
                    vec[split:] = q1
                    add(f"piece_{split_num}_{split_den}_{q0}{q1}", vec)

    if mode in {"periodic", "search", "super_search"}:
        for period in (4, 8, 16, 32):
            if period > n_bins:
                continue
            for step in range(1, os_value):
                for start in range(os_value):
                    motif = np.mod(start + step * np.arange(period, dtype=np.int64), os_value)
                    add(f"periodic_p{period}_s{step}_b{start}", motif[p % period])
            for width in range(1, period):
                motif = np.zeros(period, dtype=np.int64)
                motif[:width] = 1
                add(f"periodic_p{period}_duty{width}", motif[p % period])

    if mode in {"dither", "search", "super_search"}:
        for base in range(os_value):
            for period in (4, 8, 16, 32, 64):
                if period > n_bins:
                    continue
                for duty in (1, 2, max(1, period // 4), max(1, period // 2)):
                    for delta in range(1, os_value):
                        motif = np.full(period, base, dtype=np.int64)
                        motif[: min(duty, period)] = base + delta
                        add(f"dither_b{base}_p{period}_d{duty}_x{delta}", motif[p % period])

    if mode in {"multiscale", "super_search"}:
        widths = tuple(width for width in (2, 4, 8, 16, 32, 64, 128, 256) if width <= n_bins)
        for width in widths:
            block = p // int(width)
            within = p % int(width)
            fine_divisor = max(1, int(width) // os_value)
            for step in range(1, os_value):
                for start in range(os_value):
                    add(f"block_w{width}_s{step}_b{start}", start + step * block)
                    add(
                        f"multiscale_w{width}_s{step}_b{start}",
                        start + step * block + within // fine_divisor,
                    )

    if mode in {"quadratic", "super_search"}:
        triangular = p * (p + 1) // 2
        for curvature in range(1, os_value):
            for slope in range(os_value):
                for start in range(os_value):
                    add(
                        f"quadratic_a{curvature}_s{slope}_b{start}",
                        start + slope * p + curvature * p * p,
                    )
                    add(
                        f"triangular_a{curvature}_s{slope}_b{start}",
                        start + slope * p + curvature * triangular,
                    )

    if mode in {"bitmix", "super_search"}:
        gray = np.bitwise_xor(p, p >> 1)
        for step in range(1, os_value):
            for start in range(os_value):
                add(f"gray_s{step}_b{start}", start + step * gray)
        for shift in (1, 2, 3, 4, 5, 7, 8, 9):
            if shift >= int(sf):
                continue
            mixed = np.bitwise_xor(p, p >> shift)
            folded = np.bitwise_xor(gray, p >> shift)
            for step in range(1, os_value):
                for start in range(os_value):
                    add(f"xor_h{shift}_s{step}_b{start}", start + step * mixed)
                    add(f"fold_h{shift}_s{step}_b{start}", start + step * folded)

        p_u64 = p.astype(np.uint64)
        for idx, constant in enumerate((0x9E3779B185EBCA87, 0xD6E8FEB86659FD93, 0xA24BAED4963EE407)):
            mixed_u64 = p_u64 * np.uint64(constant)
            mixed_u64 = np.bitwise_xor(mixed_u64, mixed_u64 >> np.uint64(29 + idx))
            mixed_u64 = np.bitwise_xor(mixed_u64, mixed_u64 >> np.uint64(17 - 2 * idx))
            mixed = mixed_u64.astype(np.int64)
            for start in range(os_value):
                add(f"hash_{idx}_b{start}", mixed + start)

    if mode in {"random", "balanced_random", "search", "super_search"}:
        rng = np.random.default_rng(int(seed))
        count = max(0, int(random_count))
        for idx in range(count):
            if mode == "balanced_random" or (mode in {"search", "super_search"} and idx % 2 == 1):
                repeats = int(math.ceil(n_bins / float(os_value)))
                vec = np.tile(np.arange(os_value, dtype=np.int64), repeats)[:n_bins]
                rng.shuffle(vec)
            else:
                vec = rng.integers(0, os_value, size=n_bins, dtype=np.int64)
            add(f"{mode}_{idx:03d}", vec)

    if mode not in {
        "fixed",
        "basic",
        "periodic",
        "dither",
        "multiscale",
        "quadratic",
        "bitmix",
        "random",
        "balanced_random",
        "search",
        "super_search",
    }:
        raise ValueError(f"unknown non-uniform pattern-bank kind: {kind}")

    return NonuniformPatternBank(
        names=tuple(names),
        offsets=tuple(patterns),
        os_factor=os_value,
        sf=int(sf),
        kind=requested_mode,
    )


def select_pattern_subset(
    bank: NonuniformPatternBank,
    max_patterns: int,
    strategy: str = "diverse",
) -> NonuniformPatternBank:
    """Select a deterministic compact subset from a large pattern bank."""

    keep = int(max_patterns)
    if keep <= 0 or len(bank.offsets) <= keep:
        return bank
    mode = str(strategy).strip().lower()
    if mode not in {"head", "diverse"}:
        raise ValueError(f"unknown pattern subset strategy: {strategy}")
    if mode == "head":
        selected = list(range(keep))
    else:
        offsets = np.stack(bank.offsets).astype(np.int16, copy=False)
        selected = [idx for idx, name in enumerate(bank.names) if name.startswith("fixed_")][:keep]
        if not selected:
            selected = [0]
        available = np.ones(len(bank.offsets), dtype=bool)
        available[np.asarray(selected, dtype=np.int64)] = False
        min_distance = np.ones(len(bank.offsets), dtype=np.float64)
        for idx in selected:
            distance = np.mean(offsets != offsets[idx], axis=1)
            min_distance = np.minimum(min_distance, distance)
        while len(selected) < keep and np.any(available):
            score = np.where(available, min_distance, -1.0)
            idx = int(np.argmax(score))
            selected.append(idx)
            available[idx] = False
            distance = np.mean(offsets != offsets[idx], axis=1)
            min_distance = np.minimum(min_distance, distance)

    return NonuniformPatternBank(
        names=tuple(bank.names[idx] for idx in selected),
        offsets=tuple(bank.offsets[idx] for idx in selected),
        os_factor=int(bank.os_factor),
        sf=int(bank.sf),
        kind=f"{bank.kind}_{mode}{keep}",
    )


def select_pattern_subset_by_information(
    bank: NonuniformPatternBank,
    covariance: np.ndarray,
    max_patterns: int,
    diagonal_loading: float = 1e-3,
    target_response: np.ndarray | None = None,
) -> PatternSubsetSelection:
    """Greedily maximize the Schur-complement gain in a.H C^-1 a."""

    pattern_count = len(bank.offsets)
    keep = min(max(1, int(max_patterns)), pattern_count)
    cov = np.asarray(covariance, dtype=np.complex128)
    if cov.shape != (pattern_count, pattern_count):
        raise ValueError("covariance shape mismatch")
    cov = (cov + cov.conj().T) * 0.5
    mean_power = float(np.real(np.trace(cov)) / max(1, pattern_count))
    load = max(float(diagonal_loading), 0.0) * max(mean_power, 1e-30)
    loaded = cov + np.eye(pattern_count, dtype=np.complex128) * max(load, mean_power * 1e-12)
    target = (
        np.ones(pattern_count, dtype=np.complex128)
        if target_response is None
        else np.asarray(target_response, dtype=np.complex128)
    )
    if target.size != pattern_count:
        raise ValueError("target_response shape mismatch")

    selected: list[int] = []
    marginal_information: list[float] = []
    available = np.ones(pattern_count, dtype=bool)
    inverse = np.zeros((0, 0), dtype=np.complex128)
    for _ in range(keep):
        best_index = -1
        best_gain = -1.0
        for candidate in np.flatnonzero(available):
            candidate_index = int(candidate)
            if not selected:
                conditional_power = float(np.real(loaded[candidate_index, candidate_index]))
                residual_target = complex(target[candidate_index])
            else:
                cross_row = loaded[candidate_index, selected]
                cross_column = loaded[selected, candidate_index]
                conditional_power = float(
                    np.real(loaded[candidate_index, candidate_index] - cross_row @ inverse @ cross_column)
                )
                residual_target = complex(
                    target[candidate_index] - cross_row @ inverse @ target[np.asarray(selected, dtype=np.int64)]
                )
            gain = float(abs(residual_target) ** 2 / max(conditional_power, 1e-30))
            if gain > best_gain:
                best_gain = gain
                best_index = candidate_index
        if best_index < 0:
            break

        if not selected:
            inverse = np.asarray([[1.0 / max(float(np.real(loaded[best_index, best_index])), 1e-30)]], dtype=np.complex128)
        else:
            cross_column = loaded[selected, best_index]
            projected = inverse @ cross_column
            schur = float(
                np.real(loaded[best_index, best_index] - loaded[best_index, selected] @ projected)
            )
            schur = max(schur, 1e-30)
            old_size = len(selected)
            updated = np.empty((old_size + 1, old_size + 1), dtype=np.complex128)
            updated[:old_size, :old_size] = inverse + np.outer(projected, projected.conj()) / schur
            updated[:old_size, old_size] = -projected / schur
            updated[old_size, :old_size] = -projected.conj() / schur
            updated[old_size, old_size] = 1.0 / schur
            inverse = updated
        selected.append(best_index)
        marginal_information.append(max(best_gain, 0.0))
        available[best_index] = False

    subset = NonuniformPatternBank(
        names=tuple(bank.names[index] for index in selected),
        offsets=tuple(bank.offsets[index] for index in selected),
        os_factor=int(bank.os_factor),
        sf=int(bank.sf),
        kind=f"{bank.kind}_info{len(selected)}",
    )
    return PatternSubsetSelection(
        bank=subset,
        indices=tuple(selected),
        marginal_information=tuple(marginal_information),
    )


def prepare_dechirped_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "continuous",
) -> np.ndarray:
    """Return one oversampled symbol after optional CFO correction and dechirp."""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    start = int(start_sample)
    stop = start + n_bins * os_value
    if start < 0 or stop > np.asarray(samples).size:
        raise ValueError(f"symbol at {start_sample} exceeds input sample range")

    mode = str(cfo_correction_mode)
    if mode not in {"none", "symbol", "continuous"}:
        raise ValueError(f"unknown CFO correction mode: {cfo_correction_mode}")
    symbol = np.asarray(samples[start:stop], dtype=np.complex64)
    use_cfo_int = int(cfo_int) if mode in {"symbol", "continuous"} else 0
    use_cfo_frac = float(cfo_frac) if mode in {"symbol", "continuous"} else 0.0
    if mode == "continuous":
        if header_start_sample is None:
            raise ValueError("header_start_sample is required for continuous CFO correction")
        cfo_total = float(cfo_int) + float(cfo_frac)
        relative_chip_start = float(start - int(header_start_sample)) / float(os_value)
        cfo_common_phase_rad = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
        symbol = (symbol * np.exp(-1j * cfo_common_phase_rad)).astype(np.complex64)

    downchirp = _oversampled_downchirp(
        sf=int(sf),
        os_factor=os_value,
        cfo_int=use_cfo_int,
        cfo_frac=use_cfo_frac,
    )
    return (symbol * downchirp).astype(np.complex64)


def pattern_bin_value(
    dechirped: np.ndarray,
    raw_fft_bin: int,
    pattern_offsets: np.ndarray,
    sf: int,
    os_factor: int,
) -> complex:
    """Evaluate one non-uniform pattern for one candidate raw FFT bin."""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    if np.asarray(dechirped).size != n_bins * os_value:
        raise ValueError("dechirped symbol length mismatch")
    q = np.asarray(pattern_offsets, dtype=np.int64)
    if q.size != n_bins:
        raise ValueError("pattern length mismatch")
    q = np.mod(q, os_value)
    k = int(raw_fft_bin) % n_bins
    p = np.arange(n_bins, dtype=np.float64)
    indexes = os_value * np.arange(n_bins, dtype=np.int64) + q
    picked = np.asarray(dechirped[indexes], dtype=np.complex64)

    kernel = np.exp(-2j * np.pi * float(k) * p / float(n_bins))
    if k != 0:
        tail_mask = p >= float(n_bins - k)
        tail_phase = np.ones(n_bins, dtype=np.complex128)
        tail_phase[tail_mask] = np.exp(2j * np.pi * q[tail_mask].astype(np.float64) / float(os_value))
    else:
        tail_phase = 1.0
    branch_weight = np.exp(
        -2j * np.pi * q.astype(np.float64) * float(k) / float(n_bins * os_value)
    )
    return complex(np.sum(picked * kernel * tail_phase * branch_weight) / math.sqrt(float(n_bins)))


def pattern_bin_values(
    dechirped: np.ndarray,
    raw_fft_bin: int,
    bank: NonuniformPatternBank,
) -> np.ndarray:
    """Evaluate all patterns in a bank for one candidate raw FFT bin."""

    n_bins = 1 << int(bank.sf)
    os_value = _validate_os_factor(bank.os_factor)
    symbol = np.asarray(dechirped, dtype=np.complex64)
    if symbol.size != n_bins * os_value:
        raise ValueError("dechirped symbol length mismatch")
    if not bank.offsets:
        return np.zeros(0, dtype=np.complex128)

    offsets = np.stack([np.mod(np.asarray(item, dtype=np.int64), os_value) for item in bank.offsets])
    if offsets.shape[1] != n_bins:
        raise ValueError("pattern length mismatch")
    k = int(raw_fft_bin) % n_bins
    p_int = np.arange(n_bins, dtype=np.int64)
    p = p_int.astype(np.float64)
    indexes = os_value * p_int[None, :] + offsets
    picked = symbol[indexes]

    kernel = np.exp(-2j * np.pi * float(k) * p / float(n_bins))
    if k != 0:
        tail_mask = p >= float(n_bins - k)
        tail_phase = np.ones(offsets.shape, dtype=np.complex128)
        tail_phase[:, tail_mask] = np.exp(
            2j * np.pi * offsets[:, tail_mask].astype(np.float64) / float(os_value)
        )
    else:
        tail_phase = 1.0
    branch_weight = np.exp(
        -2j * np.pi * offsets.astype(np.float64) * float(k) / float(n_bins * os_value)
    )
    values = np.sum(picked * kernel[None, :] * tail_phase * branch_weight, axis=1) / math.sqrt(float(n_bins))
    return np.asarray(values, dtype=np.complex128)


def pattern_sample_matrix(dechirped: np.ndarray, bank: NonuniformPatternBank) -> np.ndarray:
    """Return one N-sample branch per non-uniform pattern."""

    n_bins = 1 << int(bank.sf)
    os_value = _validate_os_factor(bank.os_factor)
    symbol = np.asarray(dechirped, dtype=np.complex64)
    if symbol.size != n_bins * os_value:
        raise ValueError("dechirped symbol length mismatch")
    if not bank.offsets:
        return np.zeros((0, n_bins), dtype=np.complex64)
    offsets = np.stack([np.mod(np.asarray(item, dtype=np.int64), os_value) for item in bank.offsets])
    if offsets.shape[1] != n_bins:
        raise ValueError("pattern length mismatch")
    p_int = np.arange(n_bins, dtype=np.int64)
    indexes = os_value * p_int[None, :] + offsets
    return np.asarray(symbol[indexes], dtype=np.complex64)


def plain_pattern_fft_spectra(dechirped: np.ndarray, bank: NonuniformPatternBank) -> np.ndarray:
    """Return plain FFT spectra after taking each non-uniform N-sample branch."""

    n_bins = 1 << int(bank.sf)
    samples = pattern_sample_matrix(dechirped=dechirped, bank=bank)
    if samples.size == 0:
        return np.zeros((0, n_bins), dtype=np.complex128)
    return (np.fft.fft(samples, axis=1) / math.sqrt(float(n_bins))).astype(np.complex128)


def pattern_bank_split_spectra(
    dechirped: np.ndarray,
    bank: NonuniformPatternBank,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total, pre-wrap, and post-wrap LoRa-corrected spectra."""

    n_bins = 1 << int(bank.sf)
    os_value = _validate_os_factor(bank.os_factor)
    symbol = np.asarray(dechirped, dtype=np.complex64)
    if symbol.size != n_bins * os_value:
        raise ValueError("dechirped symbol length mismatch")
    if not bank.offsets:
        empty = np.zeros((0, n_bins), dtype=np.complex128)
        return empty, empty.copy(), empty.copy()

    offsets = np.stack([np.mod(np.asarray(item, dtype=np.int64), os_value) for item in bank.offsets])
    if offsets.shape[1] != n_bins:
        raise ValueError("pattern length mismatch")
    p_int = np.arange(n_bins, dtype=np.int64)
    indexes = os_value * p_int[None, :] + offsets
    picked = np.asarray(symbol[indexes], dtype=np.complex128)
    pattern_count = offsets.shape[0]
    branch_sequences = np.zeros((pattern_count, os_value, n_bins), dtype=np.complex128)
    for q in range(os_value):
        branch_sequences[:, q, :] = np.where(offsets == q, picked, 0.0)

    base = np.fft.fft(branch_sequences, axis=2)

    # The wrap correction is a causal chirp convolution:
    # sum_{r=1}^k u[r] exp(j 2 pi k r / N).
    r = np.arange(n_bins, dtype=np.float64)
    reversed_tail = np.zeros_like(branch_sequences)
    reversed_tail[:, :, 1:] = branch_sequences[:, :, n_bins - 1:0:-1]
    positive_chirp = np.exp(1j * np.pi * r * r / float(n_bins))
    negative_chirp = np.exp(-1j * np.pi * r * r / float(n_bins))
    fft_len = 2 * n_bins
    convolution = np.fft.ifft(
        np.fft.fft(reversed_tail * positive_chirp[None, None, :], n=fft_len, axis=2)
        * np.fft.fft(negative_chirp, n=fft_len)[None, None, :],
        axis=2,
    )[:, :, :n_bins]
    tail = positive_chirp[None, None, :] * convolution

    q = np.arange(os_value, dtype=np.float64)[:, None]
    k = np.arange(n_bins, dtype=np.float64)[None, :]
    branch_phase = np.exp(-2j * np.pi * q * k / float(n_bins * os_value))
    tail_factor = (np.exp(2j * np.pi * np.arange(os_value, dtype=np.float64) / float(os_value)) - 1.0)[
        None, :, None
    ]
    head = np.sum(branch_phase[None, :, :] * (base - tail), axis=1)
    corrected_tail = np.sum(branch_phase[None, :, :] * ((tail_factor + 1.0) * tail), axis=1)
    scale = math.sqrt(float(n_bins))
    head = np.asarray(head / scale, dtype=np.complex128)
    corrected_tail = np.asarray(corrected_tail / scale, dtype=np.complex128)
    return head + corrected_tail, head, corrected_tail


def pattern_bank_spectra(dechirped: np.ndarray, bank: NonuniformPatternBank) -> np.ndarray:
    """Return LoRa/Savaux-corrected spectra for every pattern and every bin."""

    spectra, _head, _tail = pattern_bank_split_spectra(dechirped, bank)
    return spectra


def lora_wrap_consistency_power(
    head_spectra: np.ndarray,
    tail_spectra: np.ndarray,
    base_power: np.ndarray | None = None,
    exponent: float = 1.0,
    minimum_segment: int = 16,
) -> np.ndarray:
    """Weight a spectrum by LoRa pre-wrap/post-wrap amplitude consistency."""

    head = np.asarray(head_spectra, dtype=np.complex128)
    tail = np.asarray(tail_spectra, dtype=np.complex128)
    if head.shape != tail.shape or head.ndim not in {1, 2}:
        raise ValueError("head_spectra and tail_spectra must have matching bin dimensions")
    n_bins = int(head.shape[-1])
    if n_bins == 0:
        return np.zeros(0, dtype=np.float64)
    coherent_head = head if head.ndim == 1 else np.mean(head, axis=0)
    coherent_tail = tail if tail.ndim == 1 else np.mean(tail, axis=0)
    raw_bins = np.arange(n_bins, dtype=np.float64)
    head_length = n_bins - raw_bins
    tail_length = raw_bins
    common = np.abs(coherent_head + coherent_tail).astype(np.float64) ** 2 / float(n_bins)
    separate = np.zeros(n_bins, dtype=np.float64)
    valid_head = head_length > 0.0
    valid_tail = tail_length > 0.0
    separate[valid_head] += np.abs(coherent_head[valid_head]).astype(np.float64) ** 2 / head_length[valid_head]
    separate[valid_tail] += np.abs(coherent_tail[valid_tail]).astype(np.float64) ** 2 / tail_length[valid_tail]
    consistency = np.clip(common / (separate + 1e-30), 0.0, 1.0)
    short_segment = np.minimum(head_length, tail_length) < max(0, int(minimum_segment))
    consistency[short_segment] = 1.0
    if float(exponent) != 1.0:
        consistency = consistency ** float(exponent)
    base = common if base_power is None else np.asarray(base_power, dtype=np.float64)
    if base.shape != consistency.shape:
        raise ValueError("base_power shape mismatch")
    return np.asarray(base * consistency, dtype=np.float64)


@lru_cache(maxsize=256)
def _lora_dechirped_template(sf: int, os_factor: int, raw_fft_bin: int) -> np.ndarray:
    """Return the exact oversampled LoRa phase law after reference dechirping."""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    k = int(raw_fft_bin) % n_bins
    p = np.repeat(np.arange(n_bins, dtype=np.float64), os_value)
    q = np.tile(np.arange(os_value, dtype=np.float64), n_bins)
    phase = np.exp(2j * np.pi * float(k) * (os_value * p + q) / float(n_bins * os_value))
    if k != 0:
        wrapped = p >= float(n_bins - k)
        phase[wrapped] *= np.exp(-2j * np.pi * q[wrapped] / float(os_value))
    template = np.asarray(phase, dtype=np.complex64)
    template.flags.writeable = False
    return template


def lora_phase_law_consistency(
    dechirped: np.ndarray,
    candidate_bins: Sequence[int],
    sf: int,
    os_factor: int,
    segment_count: int = 8,
) -> np.ndarray:
    """Test whether each candidate leaves one common LoRa symbol amplitude.

    The candidate-dependent LoRa phase law, including its cyclic frequency
    wrap, is removed from every oversampled sample.  The remaining symbol is
    split into equal cyclic segments beginning at that candidate's wrap point.
    The returned common-amplitude GLRT ratio is in [0, 1].
    """

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    symbol = np.asarray(dechirped, dtype=np.complex128)
    if symbol.size != n_bins * os_value:
        raise ValueError("dechirped symbol length mismatch")
    bins = tuple(int(value) % n_bins for value in candidate_bins)
    if not bins:
        return np.zeros(0, dtype=np.float64)
    segments = min(max(1, int(segment_count)), n_bins)
    boundaries = np.linspace(0, n_bins, num=segments + 1, dtype=np.int64)
    output = np.empty(len(bins), dtype=np.float64)
    for index, k in enumerate(bins):
        template = _lora_dechirped_template(int(sf), os_value, k)
        flattened = symbol * np.conjugate(template)
        wrap_chip = 0 if k == 0 else n_bins - k
        aligned = np.roll(flattened, -wrap_chip * os_value)
        common = float(np.abs(np.sum(aligned, dtype=np.complex128)) ** 2 / aligned.size)
        separate = 0.0
        for begin_chip, end_chip in zip(boundaries[:-1], boundaries[1:], strict=True):
            begin = int(begin_chip) * os_value
            end = int(end_chip) * os_value
            length = end - begin
            if length > 0:
                segment_sum = np.sum(aligned[begin:end], dtype=np.complex128)
                separate += float(np.abs(segment_sum) ** 2 / length)
        output[index] = float(np.clip(common / (separate + 1e-30), 0.0, 1.0))
    return output


def crossfit_weighted_spectrum(
    pattern_spectra: np.ndarray,
    inverse_targets: Sequence[np.ndarray],
) -> np.ndarray:
    """Apply fold-specific inverse-covariance target vectors to a spectrum."""

    spectra = np.asarray(pattern_spectra, dtype=np.complex128)
    if spectra.ndim != 2:
        raise ValueError("pattern_spectra must have shape (patterns, bins)")
    fold_count = len(inverse_targets)
    if fold_count <= 0:
        return np.zeros(spectra.shape[1], dtype=np.complex128)
    output = np.zeros(spectra.shape[1], dtype=np.complex128)
    all_bins = np.arange(spectra.shape[1], dtype=np.int64)
    for fold, inverse_target in enumerate(inverse_targets):
        weights = np.asarray(inverse_target, dtype=np.complex128)
        if weights.size != spectra.shape[0]:
            raise ValueError("inverse target shape mismatch")
        score_bins = all_bins[np.mod(all_bins, fold_count) == fold]
        output[score_bins] = weights.conj().T @ spectra[:, score_bins]
    return output


def pattern_coherence_weighted_power(
    pattern_spectra: np.ndarray,
    reference_power: np.ndarray | None = None,
    exponent: float = 1.0,
) -> np.ndarray:
    """Return a spectrum weighted by cross-pattern phase consistency.

    The ratio is one when all pattern outputs for a bin are identical and tends
    toward 1 / B for pattern-incoherent noise.  Multiplying a reference spectrum
    by this ratio turns the pattern bank into a false-peak consistency check.
    """

    spectra = np.asarray(pattern_spectra, dtype=np.complex128)
    if spectra.ndim != 2:
        raise ValueError("pattern_spectra must have shape (patterns, bins)")
    if spectra.size == 0:
        return np.zeros(spectra.shape[-1] if spectra.ndim == 2 else 0, dtype=np.float64)
    coherent_power = np.abs(np.mean(spectra, axis=0)).astype(np.float64) ** 2
    mean_power = np.mean(np.abs(spectra).astype(np.float64) ** 2, axis=0)
    consistency = np.clip(coherent_power / (mean_power + 1e-30), 0.0, 1.0)
    if float(exponent) != 1.0:
        consistency = consistency ** float(exponent)
    base = coherent_power if reference_power is None else np.asarray(reference_power, dtype=np.float64)
    if base.shape != consistency.shape:
        raise ValueError("reference_power shape mismatch")
    return np.asarray(base * consistency, dtype=np.float64)


def adaptive_gls_spectrum_power(
    pattern_spectra: np.ndarray,
    covariance_bins: Sequence[int] | np.ndarray | None = None,
    diagonal_loading: float = 0.05,
    target_response: np.ndarray | None = None,
) -> np.ndarray:
    """Return adaptive matched-filter power over a pattern spectrum.

    This is the pattern-domain analogue of Capon/AMF scoring: estimate a
    pattern-output covariance from background bins, then coherently combine the
    bin under test with weights proportional to R^{-1} a.
    """

    spectra = np.asarray(pattern_spectra, dtype=np.complex128)
    if spectra.ndim != 2:
        raise ValueError("pattern_spectra must have shape (patterns, bins)")
    pattern_count, n_bins = spectra.shape
    if pattern_count == 0 or n_bins == 0:
        return np.zeros(n_bins, dtype=np.float64)
    if covariance_bins is None:
        background = spectra
    else:
        indices = np.asarray(covariance_bins, dtype=np.int64)
        if indices.size == 0:
            background = spectra
        else:
            background = spectra[:, np.mod(indices, n_bins)]

    covariance = (background @ background.conj().T) / float(max(1, background.shape[1]))
    trace_power = float(np.real(np.trace(covariance)) / float(max(1, pattern_count)))
    load = max(float(diagonal_loading), 0.0) * max(trace_power, 1e-30)
    covariance = covariance + np.eye(pattern_count, dtype=np.complex128) * load

    target = (
        np.ones(pattern_count, dtype=np.complex128)
        if target_response is None
        else np.asarray(target_response, dtype=np.complex128)
    )
    if target.size != pattern_count:
        raise ValueError("target_response shape mismatch")
    try:
        inv_cov_target = np.linalg.solve(covariance, target)
    except np.linalg.LinAlgError:
        inv_cov_target = np.linalg.pinv(covariance, rcond=1e-10) @ target
    denom = max(float(np.real(target.conj().T @ inv_cov_target)), 1e-30)
    projection = inv_cov_target.conj().T @ spectra
    return np.asarray(np.abs(projection).astype(np.float64) ** 2 / denom, dtype=np.float64)


def crossfit_gls_spectrum_power(
    pattern_spectra: np.ndarray,
    covariance_bins: Sequence[int] | np.ndarray | None = None,
    diagonal_loading: float = 0.05,
    folds: int = 4,
    target_response: np.ndarray | None = None,
) -> np.ndarray:
    """Score each frequency fold with covariance estimated from other folds."""

    spectra = np.asarray(pattern_spectra, dtype=np.complex128)
    if spectra.ndim != 2:
        raise ValueError("pattern_spectra must have shape (patterns, bins)")
    pattern_count, n_bins = spectra.shape
    if pattern_count == 0 or n_bins == 0:
        return np.zeros(n_bins, dtype=np.float64)
    fold_count = min(max(2, int(folds)), n_bins)
    background_indices = (
        np.arange(n_bins, dtype=np.int64)
        if covariance_bins is None
        else np.mod(np.asarray(covariance_bins, dtype=np.int64), n_bins)
    )
    target = (
        np.ones(pattern_count, dtype=np.complex128)
        if target_response is None
        else np.asarray(target_response, dtype=np.complex128)
    )
    if target.size != pattern_count:
        raise ValueError("target_response shape mismatch")

    output = np.zeros(n_bins, dtype=np.float64)
    all_bins = np.arange(n_bins, dtype=np.int64)
    for fold in range(fold_count):
        train_bins = background_indices[np.mod(background_indices, fold_count) != fold]
        if train_bins.size == 0:
            train_bins = background_indices
        background = spectra[:, train_bins]
        covariance = (background @ background.conj().T) / float(max(1, background.shape[1]))
        trace_power = float(np.real(np.trace(covariance)) / float(max(1, pattern_count)))
        load = max(float(diagonal_loading), 0.0) * max(trace_power, 1e-30)
        covariance += np.eye(pattern_count, dtype=np.complex128) * load
        try:
            inverse_target = np.linalg.solve(covariance, target)
        except np.linalg.LinAlgError:
            inverse_target = np.linalg.pinv(covariance, rcond=1e-10) @ target
        denom = max(float(np.real(target.conj().T @ inverse_target)), 1e-30)
        score_bins = all_bins[np.mod(all_bins, fold_count) == fold]
        projection = inverse_target.conj().T @ spectra[:, score_bins]
        output[score_bins] = np.abs(projection).astype(np.float64) ** 2 / denom
    return output


def _conjugate_gradient(
    operator: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    initial: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int, float]:
    x = np.asarray(initial, dtype=np.complex128).copy()
    b = np.asarray(rhs, dtype=np.complex128)
    residual = b - operator(x)
    direction = residual.copy()
    residual_power = max(float(np.real(np.vdot(residual, residual))), 0.0)
    rhs_norm = max(float(np.linalg.norm(b)), 1e-30)
    relative_residual = math.sqrt(residual_power) / rhs_norm
    if relative_residual <= float(tolerance):
        return x, 0, relative_residual
    completed = 0
    for iteration in range(1, max(1, int(max_iterations)) + 1):
        applied = operator(direction)
        denominator = float(np.real(np.vdot(direction, applied)))
        if denominator <= 1e-30:
            break
        step = residual_power / denominator
        x += step * direction
        residual -= step * applied
        next_power = max(float(np.real(np.vdot(residual, residual))), 0.0)
        completed = iteration
        relative_residual = math.sqrt(next_power) / rhs_norm
        if relative_residual <= float(tolerance):
            residual_power = next_power
            break
        direction = residual + (next_power / max(residual_power, 1e-30)) * direction
        residual_power = next_power
    return x, completed, relative_residual


def matrix_free_crossfit_gls_spectrum_power(
    pattern_spectra: np.ndarray,
    covariance_bins: Sequence[int] | np.ndarray | None = None,
    diagonal_loading: float = 0.05,
    folds: int = 4,
    max_iterations: int = 12,
    tolerance: float = 1e-4,
    target_response: np.ndarray | None = None,
    prior_covariance: np.ndarray | None = None,
    prior_weight: float = 0.0,
) -> MatrixFreeGLSResult:
    """Cross-fit GLS without forming or factorizing a pattern covariance."""

    spectra = np.asarray(pattern_spectra, dtype=np.complex128)
    if spectra.ndim != 2:
        raise ValueError("pattern_spectra must have shape (patterns, bins)")
    pattern_count, n_bins = spectra.shape
    if pattern_count == 0 or n_bins == 0:
        return MatrixFreeGLSResult(
            power=np.zeros(n_bins, dtype=np.float64),
            iterations=tuple(),
            relative_residuals=tuple(),
            inverse_targets=tuple(),
        )
    fold_count = min(max(2, int(folds)), n_bins)
    background_indices = (
        np.arange(n_bins, dtype=np.int64)
        if covariance_bins is None
        else np.mod(np.asarray(covariance_bins, dtype=np.int64), n_bins)
    )
    target = (
        np.ones(pattern_count, dtype=np.complex128)
        if target_response is None
        else np.asarray(target_response, dtype=np.complex128)
    )
    if target.size != pattern_count:
        raise ValueError("target_response shape mismatch")
    prior = None if prior_covariance is None else np.asarray(prior_covariance, dtype=np.complex128)
    if prior is not None:
        if prior.shape != (pattern_count, pattern_count):
            raise ValueError("prior_covariance shape mismatch")
        prior = (prior + prior.conj().T) * 0.5
        prior_mean_power = float(np.real(np.trace(prior)) / max(1, pattern_count))
    else:
        prior_mean_power = 0.0

    output = np.zeros(n_bins, dtype=np.float64)
    all_bins = np.arange(n_bins, dtype=np.int64)
    iteration_counts: list[int] = []
    residuals: list[float] = []
    inverse_targets: list[np.ndarray] = []
    for fold in range(fold_count):
        train_bins = background_indices[np.mod(background_indices, fold_count) != fold]
        if train_bins.size == 0:
            train_bins = background_indices
        background = np.asarray(spectra[:, train_bins], dtype=np.complex128)
        background_h = background.conj().T
        sample_count = max(1, int(background.shape[1]))
        trace_power = float(np.mean(np.abs(background).astype(np.float64) ** 2))
        load = max(float(diagonal_loading), 0.0) * max(trace_power, 1e-30)
        load = max(load, max(trace_power, 1e-30) * 1e-12)
        scaled_prior = (
            None
            if prior is None or float(prior_weight) <= 0.0
            else prior * (float(prior_weight) * trace_power / max(prior_mean_power, 1e-30))
        )

        def covariance_product(vector: np.ndarray) -> np.ndarray:
            product = background @ (background_h @ vector) / float(sample_count) + load * vector
            if scaled_prior is not None:
                product += scaled_prior @ vector
            return product

        initial_scale = trace_power + load
        if scaled_prior is not None:
            initial_scale += float(prior_weight) * trace_power
        initial = target / max(initial_scale, 1e-30)
        inverse_target, iterations, relative_residual = _conjugate_gradient(
            covariance_product,
            target,
            initial,
            max_iterations=int(max_iterations),
            tolerance=float(tolerance),
        )
        iteration_counts.append(int(iterations))
        residuals.append(float(relative_residual))
        inverse_targets.append(np.asarray(inverse_target, dtype=np.complex128).copy())
        denom = max(float(np.real(target.conj().T @ inverse_target)), 1e-30)
        score_bins = all_bins[np.mod(all_bins, fold_count) == fold]
        projection = inverse_target.conj().T @ spectra[:, score_bins]
        output[score_bins] = np.abs(projection).astype(np.float64) ** 2 / denom
    return MatrixFreeGLSResult(
        power=output,
        iterations=tuple(iteration_counts),
        relative_residuals=tuple(residuals),
        inverse_targets=tuple(inverse_targets),
    )


def pattern_noise_signatures(bank: NonuniformPatternBank, raw_fft_bin: int) -> np.ndarray:
    """Return linear noise signatures for every pattern at one candidate bin.

    Each row is the vector `s_b` such that a pattern output is `<s_b, noise>`.
    The row covariance for white sample noise is therefore `S @ S.H`.
    """

    os_value = _validate_os_factor(bank.os_factor)
    n_bins = 1 << int(bank.sf)
    if not bank.offsets:
        return np.zeros((0, n_bins * os_value), dtype=np.complex128)
    offsets = np.stack([np.mod(np.asarray(item, dtype=np.int64), os_value) for item in bank.offsets])
    if offsets.shape[1] != n_bins:
        raise ValueError("pattern length mismatch")
    k = int(raw_fft_bin) % n_bins
    p_int = np.arange(n_bins, dtype=np.int64)
    p = p_int.astype(np.float64)
    indexes = os_value * p_int[None, :] + offsets

    kernel = np.exp(-2j * np.pi * float(k) * p / float(n_bins))
    if k != 0:
        tail_mask = p >= float(n_bins - k)
        tail_phase = np.ones(offsets.shape, dtype=np.complex128)
        tail_phase[:, tail_mask] = np.exp(
            2j * np.pi * offsets[:, tail_mask].astype(np.float64) / float(os_value)
        )
    else:
        tail_phase = 1.0
    branch_weight = np.exp(
        -2j * np.pi * offsets.astype(np.float64) * float(k) / float(n_bins * os_value)
    )
    signatures = np.zeros((offsets.shape[0], n_bins * os_value), dtype=np.complex128)
    signatures[np.arange(offsets.shape[0])[:, None], indexes] = (
        kernel[None, :] * tail_phase * branch_weight / math.sqrt(float(n_bins))
    )
    return signatures


def target_response_vector(bank: NonuniformPatternBank) -> np.ndarray:
    """Return the ideal LoRa target response for this corrected pattern bank."""

    n_bins = 1 << int(bank.sf)
    return np.full(len(bank.offsets), math.sqrt(float(n_bins)), dtype=np.complex128)


def lora_interbin_leakage_covariance(
    bank: NonuniformPatternBank,
    raw_bins: Sequence[int] | None = None,
    bin_offsets: Sequence[int] = (-8, -4, -2, -1, 1, 2, 4, 8),
) -> np.ndarray:
    """Return ideal LoRa wrong-bin leakage covariance in pattern space."""

    n_bins = 1 << int(bank.sf)
    os_factor = _validate_os_factor(bank.os_factor)
    bins = (
        tuple(int(value) % n_bins for value in raw_bins)
        if raw_bins is not None
        else tuple(int(value) for value in np.linspace(0, n_bins, num=8, endpoint=False, dtype=np.int64))
    )
    offsets = tuple(int(value) for value in bin_offsets if int(value) % n_bins != 0)
    if not bins or not offsets:
        return np.zeros((len(bank.offsets), len(bank.offsets)), dtype=np.complex128)
    downchirp = _oversampled_downchirp(int(bank.sf), os_factor, 0, 0.0)
    responses: list[np.ndarray] = []
    for raw_bin in bins:
        for delta in offsets:
            wrong_symbol = build_upchirp(int(bank.sf), (int(raw_bin) + int(delta)) % n_bins, os_factor)
            dechirped = np.asarray(wrong_symbol * downchirp, dtype=np.complex64)
            responses.append(pattern_bin_values(dechirped, int(raw_bin), bank))
    matrix = np.asarray(responses, dtype=np.complex128)
    covariance = matrix.T @ matrix.conj() / float(max(1, matrix.shape[0]))
    return (covariance + covariance.conj().T) * 0.5


def matched_gls_power(
    values: np.ndarray,
    covariance: np.ndarray,
    target_response: np.ndarray | None = None,
) -> float:
    """Matched generalized-least-squares energy for correlated pattern outputs."""

    vec = np.asarray(values, dtype=np.complex128)
    cov = np.asarray(covariance, dtype=np.complex128)
    target = np.ones(vec.size, dtype=np.complex128) if target_response is None else np.asarray(
        target_response, dtype=np.complex128
    )
    if cov.shape != (vec.size, vec.size):
        raise ValueError("covariance shape mismatch")
    if target.size != vec.size:
        raise ValueError("target_response shape mismatch")
    try:
        inv_cov_target = np.linalg.solve(cov, target)
        inv_cov_vec = np.linalg.solve(cov, vec)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(cov)
        inv_cov_target = pinv @ target
        inv_cov_vec = pinv @ vec
    denom = complex(target.conj().T @ inv_cov_target)
    if abs(denom) <= 1e-30:
        return 0.0
    numerator = complex(target.conj().T @ inv_cov_vec)
    return float(abs(numerator) ** 2 / max(float(np.real(denom)), 1e-30))


def effective_replica_count(
    bank: NonuniformPatternBank,
    raw_fft_bin: int,
    regularization: float = 0.0,
) -> float:
    """Return equivalent independent OSR replicas implied by the matrix bank."""

    signatures = pattern_noise_signatures(bank, int(raw_fft_bin))
    if signatures.size == 0:
        return 0.0
    covariance = signatures @ signatures.conj().T
    if regularization > 0.0:
        covariance = covariance + np.eye(covariance.shape[0], dtype=np.complex128) * float(regularization)
    target = target_response_vector(bank)
    try:
        inv_cov_target = np.linalg.solve(covariance, target)
    except np.linalg.LinAlgError:
        inv_cov_target = np.linalg.pinv(covariance, rcond=1e-10) @ target
    n_bins = 1 << int(bank.sf)
    return float(np.real(target.conj().T @ inv_cov_target) / float(n_bins))


def estimate_pattern_noise_covariance(
    bank: NonuniformPatternBank,
    sample_bins: Sequence[int],
) -> np.ndarray:
    """Estimate pattern-output noise covariance from pattern overlap.

    For white input noise, two pattern sums are correlated only when they reuse
    the same oversampled sample at the same chip index.  At every such overlap
    their candidate-dependent LoRa phase factors are identical and cancel in
    the covariance, so the result is exactly the normalized overlap matrix.
    """

    os_value = _validate_os_factor(bank.os_factor)
    n_bins = 1 << int(bank.sf)
    bins = tuple(int(raw_bin) for raw_bin in sample_bins)
    pattern_count = len(bank.offsets)
    if not bins or pattern_count == 0:
        return np.eye(pattern_count, dtype=np.complex128) * 1e-6
    offsets = np.stack(
        [np.mod(np.asarray(item, dtype=np.int64), os_value) for item in bank.offsets]
    )
    cov = np.mean(offsets[:, None, :] == offsets[None, :, :], axis=2).astype(np.complex128)
    cov += np.eye(pattern_count, dtype=np.complex128) * 1e-6
    return cov


def pattern_covariance_color_mismatch(
    pattern_spectra: np.ndarray,
    covariance_bins: Sequence[int] | np.ndarray,
    bank: NonuniformPatternBank,
) -> float:
    """Measure departure from the bank's exact LoRa/AWGN covariance shape."""

    spectra = np.asarray(pattern_spectra, dtype=np.complex128)
    if spectra.ndim != 2 or spectra.shape[0] != len(bank.offsets):
        raise ValueError("pattern_spectra shape mismatch")
    indices = np.unique(np.mod(np.asarray(covariance_bins, dtype=np.int64), spectra.shape[1]))
    if indices.size == 0:
        return 0.0
    background = spectra[:, indices]
    empirical = background @ background.conj().T / float(indices.size)
    white = estimate_pattern_noise_covariance(bank, indices)
    empirical_trace = max(float(np.real(np.trace(empirical))), 1e-30)
    white_trace = max(float(np.real(np.trace(white))), 1e-30)
    scaled_white = white * (empirical_trace / white_trace)
    denominator = max(float(np.linalg.norm(scaled_white, ord="fro")), 1e-30)
    return float(np.linalg.norm(empirical - scaled_white, ord="fro") / denominator)


def lora_branch_color_mismatch(
    branch_spectra: Sequence[np.ndarray],
    covariance_bins: Sequence[int] | np.ndarray,
) -> float:
    """Measure departure of disjoint Savaux LoRa branches from an AWGN null."""

    if len(branch_spectra) == 0:
        return 0.0
    spectra = np.stack([np.asarray(item, dtype=np.complex128) for item in branch_spectra])
    indices = np.unique(np.mod(np.asarray(covariance_bins, dtype=np.int64), spectra.shape[1]))
    if indices.size == 0:
        return 0.0
    background = spectra[:, indices]
    empirical = background @ background.conj().T / float(indices.size)
    mean_power = max(float(np.real(np.trace(empirical))) / spectra.shape[0], 1e-30)
    white = np.eye(spectra.shape[0], dtype=np.complex128) * mean_power
    denominator = max(float(np.linalg.norm(white, ord="fro")), 1e-30)
    return float(np.linalg.norm(empirical - white, ord="fro") / denominator)


def conditional_lora_gls_detect(
    dechirped: np.ndarray | Callable[[], np.ndarray],
    savaux_spectrum: np.ndarray,
    savaux_branch_spectra: Sequence[np.ndarray],
    bank: NonuniformPatternBank,
    covariance_bins: Sequence[int] | np.ndarray,
    savaux_margin_db: float = 0.5,
    branch_color_threshold: float = 0.12,
    diagonal_loading: float = 0.05,
    folds: int = 2,
    max_iterations: int = 4,
    tolerance: float = 0.0,
    wrap_consistency_exponent: float = 0.5,
    wrap_minimum_segment: int = 16,
) -> ConditionalLoRaGLSResult:
    """Run the low-complexity LoRa detector with two physical early exits."""

    savaux_power = np.abs(np.asarray(savaux_spectrum, dtype=np.complex128)) ** 2
    if savaux_power.size == 0:
        raise ValueError("savaux_spectrum must not be empty")
    savaux_bin = int(np.argmax(savaux_power))
    if savaux_power.size >= 2:
        top_two = np.partition(savaux_power, -2)[-2:]
        margin = float(10.0 * np.log10((float(np.max(top_two)) + 1e-30) / (float(np.min(top_two)) + 1e-30)))
    else:
        margin = float("inf")
    if margin > float(savaux_margin_db):
        return ConditionalLoRaGLSResult(
            raw_fft_bin=savaux_bin,
            savaux_bin=savaux_bin,
            savaux_margin_db=margin,
            branch_color_mismatch=float("nan"),
            screened=False,
            backend_ran=False,
            cg_iterations=tuple(),
        )

    color_mismatch = lora_branch_color_mismatch(savaux_branch_spectra, covariance_bins)
    if color_mismatch < float(branch_color_threshold):
        return ConditionalLoRaGLSResult(
            raw_fft_bin=savaux_bin,
            savaux_bin=savaux_bin,
            savaux_margin_db=margin,
            branch_color_mismatch=color_mismatch,
            screened=True,
            backend_ran=False,
            cg_iterations=tuple(),
        )

    symbol = dechirped() if callable(dechirped) else dechirped
    spectra, head_spectra, tail_spectra = pattern_bank_split_spectra(symbol, bank)
    gls_result = matrix_free_crossfit_gls_spectrum_power(
        spectra,
        covariance_bins=covariance_bins,
        diagonal_loading=float(diagonal_loading),
        folds=int(folds),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
    )
    weighted_head = crossfit_weighted_spectrum(head_spectra, gls_result.inverse_targets)
    weighted_tail = crossfit_weighted_spectrum(tail_spectra, gls_result.inverse_targets)
    score = lora_wrap_consistency_power(
        weighted_head,
        weighted_tail,
        base_power=gls_result.power,
        exponent=float(wrap_consistency_exponent),
        minimum_segment=int(wrap_minimum_segment),
    )
    return ConditionalLoRaGLSResult(
        raw_fft_bin=int(np.argmax(score)),
        savaux_bin=savaux_bin,
        savaux_margin_db=margin,
        branch_color_mismatch=color_mismatch,
        screened=True,
        backend_ran=True,
        cg_iterations=gls_result.iterations,
    )


def _whitened_sum_power(values: np.ndarray, covariance: np.ndarray) -> float:
    return matched_gls_power(values=values, covariance=covariance)


def _gls_target_projection(covariance: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    target = np.ones(int(size), dtype=np.complex128)
    cov = np.asarray(covariance, dtype=np.complex128)
    try:
        inv_cov_target = np.linalg.solve(cov, target)
    except np.linalg.LinAlgError:
        inv_cov_target = np.linalg.pinv(cov) @ target
    denom = float(np.real(target.conj().T @ inv_cov_target))
    return inv_cov_target, max(denom, 1e-30)


def score_nonuniform_candidates(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    candidate_bins: Sequence[int],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "continuous",
    bank: NonuniformPatternBank | None = None,
    savaux_power: np.ndarray | None = None,
    stable_exponent: float = 1.0,
    hybrid_mean_beta: float = 0.75,
    hybrid_consistency_alpha: float = 0.0,
    noise_covariance: np.ndarray | None = None,
) -> tuple[NonuniformScoreResult, ...]:
    """Score candidate bins using a non-uniform pattern bank."""

    os_value = _validate_os_factor(os_factor)
    pattern_bank = bank or build_pattern_bank(int(sf), os_value)
    dechirped = prepare_dechirped_symbol(
        samples=samples,
        start_sample=int(start_sample),
        sf=int(sf),
        os_factor=os_value,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    bins = tuple(int(v) for v in candidate_bins)
    covariance = (
        estimate_pattern_noise_covariance(pattern_bank, bins)
        if noise_covariance is None
        else np.asarray(noise_covariance, dtype=np.complex128)
    )
    pattern_count = len(pattern_bank.offsets)
    if covariance.shape != (pattern_count, pattern_count):
        raise ValueError("noise_covariance shape mismatch")
    inv_cov_target, gls_denom = _gls_target_projection(covariance, len(pattern_bank.offsets))
    power_lookup = None if savaux_power is None else np.asarray(savaux_power, dtype=np.float64)

    out: list[NonuniformScoreResult] = []
    for raw_bin in bins:
        values = pattern_bin_values(dechirped=dechirped, raw_fft_bin=int(raw_bin), bank=pattern_bank)
        powers = np.abs(values).astype(np.float64) ** 2
        best_idx = int(np.argmax(powers)) if powers.size else 0
        best_power = float(powers[best_idx]) if powers.size else 0.0
        mean_power = float(np.mean(powers)) if powers.size else 0.0
        coherent_power = float(abs(np.mean(values)) ** 2) if values.size else 0.0
        consistency = float(np.clip(coherent_power / (mean_power + 1e-30), 0.0, 1.0)) if values.size else 0.0
        whitened_power = (
            float(abs(complex(inv_cov_target.conj().T @ values)) ** 2 / gls_denom)
            if values.size
            else 0.0
        )
        savaux_bin_power = (
            float(power_lookup[int(raw_bin)])
            if power_lookup is not None
            else mean_power
        )
        mean_gain = float(mean_power / (savaux_bin_power + 1e-30))
        stable_power = float(savaux_bin_power * (consistency ** float(stable_exponent)))
        hybrid_power = float(
            savaux_bin_power
            * (mean_gain ** float(hybrid_mean_beta))
            * (consistency ** float(hybrid_consistency_alpha))
        )
        out.append(
            NonuniformScoreResult(
                raw_fft_bin=int(raw_bin),
                savaux_power=savaux_bin_power,
                best_pattern_power=best_power,
                mean_pattern_power=mean_power,
                coherent_pattern_power=coherent_power,
                whitened_pattern_power=whitened_power,
                stable_pattern_power=stable_power,
                hybrid_mean_power=hybrid_power,
                best_pattern_name=pattern_bank.names[best_idx] if pattern_bank.names else "",
                best_pattern_gain_vs_savaux=float(best_power / (savaux_bin_power + 1e-30)),
                coherent_gain_vs_savaux=float(coherent_power / (savaux_bin_power + 1e-30)),
                whitened_gain_vs_savaux=float(whitened_power / (savaux_bin_power + 1e-30)),
                stable_gain_vs_savaux=float(stable_power / (savaux_bin_power + 1e-30)),
                hybrid_gain_vs_savaux=float(hybrid_power / (savaux_bin_power + 1e-30)),
                pattern_values=tuple(complex(v) for v in values),
            )
        )
    return tuple(out)


def savaux_top_bins_for_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    top_k: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "continuous",
) -> tuple[tuple[int, ...], np.ndarray]:
    """Return Savaux Top-K candidate bins and its power vector."""

    spectrum, _branches, _phase = paper_oversampled_spectrum(
        samples=samples,
        start_sample=int(start_sample),
        sf=int(sf),
        os_factor=_validate_os_factor(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    power = np.abs(spectrum).astype(np.float64) ** 2
    return _top_bins(power, int(top_k)), power


__all__ = [
    "NonuniformPatternBank",
    "NonuniformScoreResult",
    "MatrixFreeGLSResult",
    "PatternSubsetSelection",
    "ConditionalLoRaGLSResult",
    "build_pattern_bank",
    "conditional_lora_gls_detect",
    "crossfit_gls_spectrum_power",
    "crossfit_weighted_spectrum",
    "effective_replica_count",
    "estimate_pattern_noise_covariance",
    "adaptive_gls_spectrum_power",
    "matched_gls_power",
    "matrix_free_crossfit_gls_spectrum_power",
    "lora_interbin_leakage_covariance",
    "lora_branch_color_mismatch",
    "lora_phase_law_consistency",
    "lora_wrap_consistency_power",
    "pattern_bin_value",
    "pattern_bin_values",
    "pattern_bank_spectra",
    "pattern_bank_split_spectra",
    "pattern_coherence_weighted_power",
    "pattern_covariance_color_mismatch",
    "pattern_noise_signatures",
    "pattern_sample_matrix",
    "plain_pattern_fft_spectra",
    "prepare_dechirped_symbol",
    "savaux_top_bins_for_symbol",
    "score_nonuniform_candidates",
    "select_pattern_subset",
    "select_pattern_subset_by_information",
    "target_response_vector",
]

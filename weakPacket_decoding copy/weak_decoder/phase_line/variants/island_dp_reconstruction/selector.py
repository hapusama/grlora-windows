"""Anchor-locked island DP with branch-aware coherent reconstruction.

This variant implements the 2026-06-27 direction:

* lock high-confidence Stage-1 symbols as immutable anchors;
* split the remaining low-confidence symbols into islands;
* run Viterbi only inside those islands;
* score states in the two-dimensional space ``(raw_bin, branch)``;
* use candidate-specific two-segment coherent reconstruction when Stage-1
  retained the oversampled dechirped IQ.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ....candidate_pruning import top_bins, wrap_phase
from ....phase_guided_demod import PhaseLine
from ....symbol_phase_two_stage import (
    SymbolEvidence,
    SymbolPhaseConfig,
    SymbolPhaseResult,
    build_symbol_evidences,
)
from ...configs import PhasePathSelectorConfig
from ...trajectory import fit_selected_phase_line
from .._legacy_core.selector import _coherence_score, _energy_score


@dataclass(frozen=True)
class IslandReconstructionConfig:
    """Runtime parameters for the experimental island DP."""

    top_l: int = 40
    anchor_margin_db: float = 1.5
    anchor_peak_to_median_db: float = 7.0
    anchor_min_coherence: float = 0.88
    branch_variance_max: float = 0.16
    energy_weight: float = 0.24
    coherence_weight: float = 0.06
    reconstruction_weight: float = 0.58
    phase_profile_weight: float = 0.04
    branch_profile_weight: float = 0.08
    transition_weight: float = 0.45
    boundary_weight: float = 1.10
    branch_transition_weight: float = 1.20
    phase_scale_pi: float = 0.45
    branch_profile_scale: float = 1.25
    branch_transition_scale: float = 1.00
    max_energy_drop_db: float = 36.0
    branch_top_k: int = 0
    sto_phase_sign: str = "plus"
    require_both_anchors: bool = True
    use_baseline_prior: bool = False
    baseline_bonus: float = 0.0
    island_accept_margin: float = 0.0
    third_bin_penalty: float = 0.0
    allow_third_bin_when_baseline_differs: bool = True
    return_to_hard_min_margin_db: float = 0.80
    auxiliary_top_l: int = 0
    auxiliary_extra_k: int = 0
    auxiliary_energy_weight: float = 0.0


@dataclass(frozen=True)
class _AnchorObservation:
    index: int
    abs_symbol_index: float
    raw_bin: int
    branch: int
    phase: float


@dataclass(frozen=True)
class _IslandDpCandidate:
    raw_bin: int
    branch: int
    phase: float
    local_score: float
    energy_score: float
    coherence_score: float
    reconstruction_score: float
    phase_profile_score: float
    branch_profile_score: float
    auxiliary_energy_score: float


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(10.0 * math.log10((float(numerator) + 1e-30) / (float(denominator) + 1e-30)))


def _huber_normalized(value: float, scale: float, delta: float) -> float:
    x = abs(float(value)) / max(1e-12, float(scale))
    d = max(1e-12, float(delta))
    if x <= d:
        return float(0.5 * x * x)
    return float(d * (x - 0.5 * d))


def _wrap_branch_delta(value: float, os_factor: int) -> float:
    os_value = max(1, int(os_factor))
    return float((float(value) + 0.5 * os_value) % float(os_value) - 0.5 * os_value)


def _anchor_locked(ev: SymbolEvidence, config: IslandReconstructionConfig) -> bool:
    if float(ev.top1_margin_db) < float(config.anchor_margin_db):
        return False
    if float(ev.top1_peak_to_median_db) < float(config.anchor_peak_to_median_db):
        return False
    if ev.offset_coherence is not None and float(config.anchor_min_coherence) > 0.0:
        if float(ev.top1_coherence_score) < float(config.anchor_min_coherence):
            return False
        top = max(1e-12, float(ev.offset_coherence[int(ev.top1_bin)]))
        nearby_bins = [int(v) for v in ev.top_bins[:4] if 0 <= int(v) < ev.offset_coherence.size]
        nearby = np.asarray(ev.offset_coherence[nearby_bins], dtype=np.float64)
        if nearby.size >= 2:
            normalized_var = float(np.var(nearby / top))
            if normalized_var > float(config.branch_variance_max):
                return False
    return True


def _island_ranges(locked_mask: Sequence[bool]) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for idx, locked in enumerate(tuple(bool(v) for v in locked_mask) + (True,)):
        if not locked and start is None:
            start = idx
        elif locked and start is not None:
            ranges.append((int(start), int(idx)))
            start = None
    return tuple(ranges)


def _interp_profile(
    observations: Sequence[tuple[float, float]],
    all_x: np.ndarray,
) -> np.ndarray | None:
    if len(observations) < 2:
        return None
    ordered = sorted((float(x), float(y)) for x, y in observations)
    xs = np.asarray([item[0] for item in ordered], dtype=np.float64)
    ys = np.asarray([item[1] for item in ordered], dtype=np.float64)
    profile = np.interp(all_x, xs, ys)
    left = all_x < xs[0]
    if np.any(left):
        slope = float((ys[1] - ys[0]) / max(1e-9, xs[1] - xs[0]))
        profile[left] = ys[0] + slope * (all_x[left] - xs[0])
    right = all_x > xs[-1]
    if np.any(right):
        slope = float((ys[-1] - ys[-2]) / max(1e-9, xs[-1] - xs[-2]))
        profile[right] = ys[-1] + slope * (all_x[right] - xs[-1])
    return np.asarray(profile, dtype=np.float64)


def _unwrap_branch_observations(
    observations: Sequence[_AnchorObservation],
    os_factor: int,
) -> tuple[tuple[float, float], ...]:
    if not observations:
        return ()
    ordered = sorted(observations, key=lambda item: item.abs_symbol_index)
    out: list[tuple[float, float]] = []
    prev: float | None = None
    os_value = max(1, int(os_factor))
    for item in ordered:
        value = float(item.branch)
        if prev is not None:
            value += round((float(prev) - value) / float(os_value)) * float(os_value)
        out.append((float(item.abs_symbol_index), float(value)))
        prev = value
    return tuple(out)


def _aligned_branch_spectrum_value(
    branch_spectra: Sequence[np.ndarray] | None,
    raw_bin: int,
    branch: int,
    os_factor: int,
) -> complex | None:
    if branch_spectra is None:
        return None
    q = int(branch)
    if q < 0 or q >= len(branch_spectra):
        return None
    spectrum = np.asarray(branch_spectra[q], dtype=np.complex128)
    if spectrum.size == 0:
        return None
    n_bins = int(spectrum.size)
    b = int(raw_bin) % n_bins
    weight = complex(np.exp(-2j * np.pi * float(q * b) / float(n_bins * max(1, int(os_factor)))))
    return complex(spectrum[b]) * weight


def _two_segment_branch_value(
    dechirped_symbol: np.ndarray,
    raw_bin: int,
    branch: int,
    os_factor: int,
    residual_sto_chips: float,
    *,
    phase_sign: str = "plus",
    extra_suffix_phase_rad: float = 0.0,
) -> tuple[complex, float, float]:
    os_value = max(1, int(os_factor))
    symbol = np.asarray(dechirped_symbol, dtype=np.complex128)
    if symbol.size == 0 or symbol.size % os_value != 0:
        raise ValueError("dechirped symbol length must be divisible by os_factor")
    n_bins = int(symbol.size // os_value)
    if n_bins <= 0:
        raise ValueError("empty dechirped symbol")
    q = int(branch)
    if q < 0 or q >= os_value:
        raise ValueError(f"branch must be in [0, {os_value}), got {branch}")

    b = int(raw_bin) % n_bins
    p = np.arange(n_bins, dtype=np.float64)
    branch_samples = np.asarray(symbol[q::os_value], dtype=np.complex128)
    if branch_samples.size != n_bins:
        raise ValueError("branch length mismatch")

    values = branch_samples * np.exp(-2j * np.pi * float(b) * p / float(n_bins))
    if b != 0:
        tail = p >= float(n_bins - b)
        if np.any(tail):
            values[tail] *= np.exp(2j * np.pi * float(q) / float(os_value))

    tau = float((float(residual_sto_chips) + 0.5) % 1.0 - 0.5)
    cut = int(math.ceil(float(n_bins - b) + tau - float(q) / float(os_value)))
    cut = max(0, min(n_bins, cut))
    front = complex(np.sum(values[:cut], dtype=np.complex128))
    back = complex(np.sum(values[cut:], dtype=np.complex128))

    sign_value = 1.0 if str(phase_sign).lower() != "minus" else -1.0
    suffix_phase = sign_value * 2.0 * math.pi * tau - float(extra_suffix_phase_rad)
    stitched_unweighted = front + back * complex(np.exp(1j * suffix_phase))
    branch_weight = complex(np.exp(-2j * np.pi * float(q * b) / float(n_bins * os_value)))
    stitched = branch_weight * stitched_unweighted / math.sqrt(float(n_bins))

    coherent = abs(stitched_unweighted) ** 2
    incoherent = abs(front) ** 2 + abs(back) ** 2 + 1e-30
    quality = float(max(0.0, min(1.0, coherent / (2.0 * incoherent))))
    return complex(stitched), quality, float(abs(stitched) ** 2)


def compute_two_segment_score(
    ev: SymbolEvidence,
    raw_bin: int,
    *,
    dechirped_symbol: np.ndarray | None = None,
    os_factor: int | None = None,
    local_drift_rad: float = 0.0,
    branch_index: int | None = None,
    residual_sto_chips: float = 0.0,
    phase_sign: str = "plus",
) -> float:
    """Score a candidate by symbol-internal two-piece reconstruction.

    Without retained dechirped IQ this falls back to the Stage-1 branch phase
    agreement proxy.  With retained IQ it splits at the candidate LoRa chirp
    wrap, phase-rotates the suffix according to the expected fractional STO,
    and returns a normalized coherent-combining score.
    """

    b = int(raw_bin)
    if dechirped_symbol is None or os_factor is None or int(os_factor) <= 1:
        return _coherence_score(ev, b)
    os_value = max(1, int(os_factor))
    branches = range(os_value) if branch_index is None else (int(branch_index),)
    scores: list[float] = []
    for q in branches:
        try:
            _value, quality, _power = _two_segment_branch_value(
                dechirped_symbol,
                b,
                int(q),
                os_value,
                float(residual_sto_chips),
                phase_sign=phase_sign,
                extra_suffix_phase_rad=float(local_drift_rad),
            )
        except ValueError:
            continue
        scores.append(float(quality))
    if not scores:
        return _coherence_score(ev, b)
    return float(max(scores))


def _candidate_bins(
    ev: SymbolEvidence,
    config: IslandReconstructionConfig,
    baseline_bin: int | None = None,
    auxiliary_powers: Sequence[np.ndarray] | None = None,
) -> tuple[int, ...]:
    power = np.asarray(ev.evidence_power, dtype=np.float64)
    if power.size == 0:
        return (int(ev.top1_bin),)
    bins: list[int] = []
    seen: set[int] = set()
    top1_power = float(power[int(ev.top1_bin)]) if 0 <= int(ev.top1_bin) < power.size else float(np.max(power))
    for raw_bin in top_bins(power, min(max(1, int(config.top_l)), power.size)):
        b = int(raw_bin)
        if b in seen:
            continue
        if b != int(ev.top1_bin) and _db_ratio(float(power[b]), top1_power) < -float(config.max_energy_drop_db):
            continue
        bins.append(b)
        seen.add(b)
    aux_top_l = int(config.auxiliary_top_l)
    if auxiliary_powers is not None and aux_top_l > 0:
        extra: dict[int, tuple[float, float]] = {}
        for aux in auxiliary_powers:
            aux_power = np.asarray(aux, dtype=np.float64)
            if aux_power.size != power.size or aux_power.size == 0:
                continue
            aux_peak = float(np.max(aux_power))
            if aux_peak <= 0.0:
                continue
            aux_bins = top_bins(aux_power, min(max(1, aux_top_l), aux_power.size))
            for rank, raw_bin in enumerate(aux_bins):
                b = int(raw_bin)
                if b in seen:
                    continue
                if _db_ratio(float(aux_power[b]), aux_peak) < -float(config.max_energy_drop_db):
                    continue
                normalized = float(max(0.0, min(1.0, float(aux_power[b]) / (aux_peak + 1e-30))))
                rank_score = 1.0 - float(rank) / float(max(1, aux_bins.size))
                score = 0.65 * normalized + 0.35 * rank_score
                if b not in extra or score > extra[b][0]:
                    extra[b] = (float(score), float(normalized))
        extra_items = sorted(extra.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
        extra_k = int(config.auxiliary_extra_k)
        if extra_k <= 0:
            extra_k = len(extra_items)
        for b, _score_pair in extra_items[:extra_k]:
            if b not in seen:
                bins.append(b)
                seen.add(b)
    if baseline_bin is not None:
        b = int(baseline_bin)
        if 0 <= b < power.size and b not in seen:
            bins.append(b)
    return tuple(bins) if bins else (int(ev.top1_bin),)


def _auxiliary_energy_score(
    raw_bin: int,
    auxiliary_powers: Sequence[np.ndarray] | None,
) -> float:
    if auxiliary_powers is None:
        return 0.0
    b = int(raw_bin)
    best = 0.0
    for aux in auxiliary_powers:
        power = np.asarray(aux, dtype=np.float64)
        if b < 0 or b >= power.size:
            continue
        peak = float(np.max(power)) if power.size else 0.0
        if peak <= 0.0:
            continue
        best = max(best, float(max(0.0, min(1.0, float(power[b]) / (peak + 1e-30)))))
    return float(best)


def _branch_candidates(
    raw_bin: int,
    os_factor: int,
    config: IslandReconstructionConfig,
    branch_spectra: Sequence[np.ndarray] | None,
) -> tuple[int, ...]:
    os_value = max(1, int(os_factor))
    if int(config.branch_top_k) <= 0 or int(config.branch_top_k) >= os_value:
        return tuple(range(os_value))
    scored: list[tuple[float, int]] = []
    for q in range(os_value):
        value = _aligned_branch_spectrum_value(branch_spectra, raw_bin, q, os_value)
        scored.append((0.0 if value is None else float(abs(value) ** 2), int(q)))
    scored.sort(reverse=True)
    return tuple(int(q) for _score, q in scored[: max(1, int(config.branch_top_k))])


def _branch_value_for_state(
    ev: SymbolEvidence,
    raw_bin: int,
    branch: int,
    os_factor: int,
    config: IslandReconstructionConfig,
    branch_spectra: Sequence[np.ndarray] | None,
    dechirped_symbol: np.ndarray | None,
    expected_branch: float | None,
    residual_sto_chips: float | None,
) -> tuple[complex, float, float]:
    os_value = max(1, int(os_factor))
    residual_sto = 0.0
    if residual_sto_chips is not None:
        residual_sto = float(residual_sto_chips)
    elif expected_branch is not None:
        residual_sto = _wrap_branch_delta(float(expected_branch) - float(branch), os_value) / float(os_value)
    if dechirped_symbol is not None:
        try:
            return _two_segment_branch_value(
                dechirped_symbol,
                raw_bin,
                branch,
                os_value,
                residual_sto,
                phase_sign=str(config.sto_phase_sign),
            )
        except ValueError:
            pass
    value = _aligned_branch_spectrum_value(branch_spectra, raw_bin, branch, os_value)
    if value is None:
        b = int(raw_bin)
        value = complex(ev.center_spectrum[b]) if 0 <= b < ev.center_spectrum.size else 0j
    return complex(value), _coherence_score(ev, raw_bin), float(abs(value) ** 2)


def _make_candidate_row(
    ev: SymbolEvidence,
    idx: int,
    path_config: PhasePathSelectorConfig,
    config: IslandReconstructionConfig,
    os_factor: int,
    branch_spectra: Sequence[np.ndarray] | None,
    dechirped_symbol: np.ndarray | None,
    phase_profile: np.ndarray | None,
    branch_profile: np.ndarray | None,
    baseline_bin: int | None,
    residual_sto_chip: float | None,
    branch_residual_sto_chips: Sequence[float] | None,
    auxiliary_powers: Sequence[np.ndarray] | None,
) -> tuple[_IslandDpCandidate, ...]:
    predicted_phase = float(phase_profile[idx]) if phase_profile is not None and idx < phase_profile.size else None
    expected_branch = (
        float(branch_profile[idx]) if branch_profile is not None and idx < branch_profile.size else None
    )
    pending: list[tuple[int, int, complex, float, float]] = []
    for raw_bin in _candidate_bins(ev, config, baseline_bin=baseline_bin, auxiliary_powers=auxiliary_powers):
        if (
            not bool(config.allow_third_bin_when_baseline_differs)
            and baseline_bin is not None
            and int(baseline_bin) != int(ev.top1_bin)
            and int(raw_bin) != int(baseline_bin)
            and int(raw_bin) != int(ev.top1_bin)
        ):
            continue
        if (
            baseline_bin is not None
            and int(baseline_bin) != int(ev.top1_bin)
            and int(raw_bin) == int(ev.top1_bin)
            and float(ev.top1_margin_db) < float(config.return_to_hard_min_margin_db)
        ):
            continue
        for branch in _branch_candidates(raw_bin, os_factor, config, branch_spectra):
            branch_residual = residual_sto_chip
            if branch_residual_sto_chips is not None and int(branch) < len(branch_residual_sto_chips):
                branch_residual = float(branch_residual_sto_chips[int(branch)])
            value, quality, branch_power = _branch_value_for_state(
                ev,
                raw_bin,
                branch,
                os_factor,
                config,
                branch_spectra,
                dechirped_symbol,
                expected_branch,
                branch_residual,
            )
            pending.append((int(raw_bin), int(branch), complex(value), float(quality), float(branch_power)))
    if not pending:
        return ()

    max_branch_power = max(1e-30, max(float(item[4]) for item in pending))
    weights = np.asarray(
        [
            max(0.0, float(config.energy_weight)),
            max(0.0, float(config.coherence_weight)),
            max(0.0, float(config.reconstruction_weight)),
            max(0.0, float(config.phase_profile_weight if predicted_phase is not None else 0.0)),
            max(0.0, float(config.branch_profile_weight if expected_branch is not None else 0.0)),
            max(0.0, float(config.auxiliary_energy_weight if auxiliary_powers is not None else 0.0)),
        ],
        dtype=np.float64,
    )
    total = float(np.sum(weights))
    weights = weights / total if total > 0.0 else np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    out: list[_IslandDpCandidate] = []
    for raw_bin, branch, value, quality, branch_power in pending:
        phase = float(np.angle(value)) if abs(value) > 0.0 else 0.0
        if abs(value) <= 0.0 and 0 <= int(raw_bin) < ev.center_spectrum.size:
            phase = float(np.angle(ev.center_spectrum[int(raw_bin)]))
        energy = _energy_score(ev, raw_bin)
        coherence = _coherence_score(ev, raw_bin)
        branch_energy = float(max(0.0, min(1.0, branch_power / max_branch_power)))
        reconstruction = float(max(0.0, min(1.0, 0.65 * branch_energy + 0.35 * float(quality))))
        phase_score = 0.0
        if predicted_phase is not None:
            residual = float(wrap_phase(phase - float(predicted_phase)))
            phase_score = float(math.exp(-((residual / max(1e-6, float(config.phase_scale_pi) * math.pi)) ** 2)))
        branch_score = 0.0
        if expected_branch is not None:
            residual_branch = _wrap_branch_delta(float(branch) - float(expected_branch), os_factor)
            branch_score = float(
                math.exp(-((residual_branch / max(1e-6, float(config.branch_profile_scale))) ** 2))
            )
        auxiliary_energy = _auxiliary_energy_score(raw_bin, auxiliary_powers)
        local = float(
            weights[0] * energy
            + weights[1] * coherence
            + weights[2] * reconstruction
            + weights[3] * phase_score
            + weights[4] * branch_score
            + weights[5] * auxiliary_energy
        )
        if baseline_bin is not None and int(raw_bin) == int(baseline_bin):
            local += float(config.baseline_bonus)
        if (
            baseline_bin is not None
            and int(baseline_bin) != int(ev.top1_bin)
            and int(raw_bin) != int(baseline_bin)
            and int(raw_bin) != int(ev.top1_bin)
        ):
            local -= float(config.third_bin_penalty)
        out.append(
            _IslandDpCandidate(
                raw_bin=int(raw_bin),
                branch=int(branch),
                phase=float(phase),
                local_score=local,
                energy_score=float(energy),
                coherence_score=float(coherence),
                reconstruction_score=float(reconstruction),
                phase_profile_score=float(phase_score),
                branch_profile_score=float(branch_score),
                auxiliary_energy_score=float(auxiliary_energy),
            )
        )
    return tuple(out)


def _best_anchor_observation(
    ev: SymbolEvidence,
    idx: int,
    os_factor: int,
    branch_spectra: Sequence[np.ndarray] | None,
    dechirped_symbol: np.ndarray | None,
    config: IslandReconstructionConfig,
) -> _AnchorObservation:
    b = int(ev.top1_bin)
    best_branch = 0
    best_value: complex | None = None
    best_power = -1.0
    for q in range(max(1, int(os_factor))):
        value, _quality, power = _branch_value_for_state(
            ev,
            b,
            q,
            os_factor,
            config,
            branch_spectra,
            dechirped_symbol,
            float(q),
            None,
        )
        if float(power) > best_power:
            best_power = float(power)
            best_branch = int(q)
            best_value = complex(value)
    if best_value is None or abs(best_value) <= 0.0:
        best_value = complex(ev.center_spectrum[b]) if 0 <= b < ev.center_spectrum.size else 0j
    return _AnchorObservation(
        index=int(idx),
        abs_symbol_index=float(ev.abs_symbol_index),
        raw_bin=b,
        branch=int(best_branch),
        phase=float(np.angle(best_value)),
    )


def _anchor_profiles(
    evidences: Sequence[SymbolEvidence],
    locked_mask: Sequence[bool],
    os_factor: int,
    branch_spectra: Sequence[Sequence[np.ndarray]] | None,
    dechirped_symbols: Sequence[np.ndarray | None] | None,
    config: IslandReconstructionConfig,
) -> tuple[tuple[_AnchorObservation, ...], np.ndarray | None, np.ndarray | None]:
    observations: list[_AnchorObservation] = []
    for idx, ev in enumerate(evidences):
        if idx >= len(locked_mask) or not bool(locked_mask[idx]):
            continue
        spectra = branch_spectra[idx] if branch_spectra is not None and idx < len(branch_spectra) else None
        dechirped = dechirped_symbols[idx] if dechirped_symbols is not None and idx < len(dechirped_symbols) else None
        observations.append(_best_anchor_observation(ev, idx, os_factor, spectra, dechirped, config))
    if len(observations) < 2:
        return tuple(observations), None, None

    ordered = sorted(observations, key=lambda item: item.abs_symbol_index)
    phase_x = [float(item.abs_symbol_index) for item in ordered]
    phase_y = np.unwrap(np.asarray([float(item.phase) for item in ordered], dtype=np.float64))
    all_x = np.asarray([float(ev.abs_symbol_index) for ev in evidences], dtype=np.float64)
    phase_profile = _interp_profile(tuple(zip(phase_x, phase_y)), all_x)
    branch_profile = _interp_profile(_unwrap_branch_observations(ordered, os_factor), all_x)
    return tuple(observations), phase_profile, branch_profile


def _anchor_candidate_from_observation(obs: _AnchorObservation) -> _IslandDpCandidate:
    return _IslandDpCandidate(
        raw_bin=int(obs.raw_bin),
        branch=int(obs.branch),
        phase=float(obs.phase),
        local_score=1.0,
        energy_score=1.0,
        coherence_score=1.0,
        reconstruction_score=1.0,
        phase_profile_score=1.0,
        branch_profile_score=1.0,
        auxiliary_energy_score=1.0,
    )


def _phase_transition_cost(
    prev: _IslandDpCandidate,
    curr: _IslandDpCandidate,
    config: PhasePathSelectorConfig,
) -> float:
    residual = float(wrap_phase(float(curr.phase) - float(prev.phase)))
    if float(config.first_order_weight) <= 0.0:
        return 0.0
    return float(
        float(config.first_order_weight)
        * _huber_normalized(residual, config.first_order_scale_rad(), float(config.huber_delta))
    )


def _branch_transition_cost(
    prev: _IslandDpCandidate,
    curr: _IslandDpCandidate,
    prev_idx: int,
    curr_idx: int,
    os_factor: int,
    branch_profile: np.ndarray | None,
    config: IslandReconstructionConfig,
) -> float:
    observed_delta = _wrap_branch_delta(float(curr.branch) - float(prev.branch), os_factor)
    expected_delta = 0.0
    if branch_profile is not None and prev_idx < branch_profile.size and curr_idx < branch_profile.size:
        expected_delta = float(branch_profile[curr_idx] - branch_profile[prev_idx])
    residual = _wrap_branch_delta(observed_delta - expected_delta, os_factor)
    return float(_huber_normalized(residual, float(config.branch_transition_scale), 1.0))


def _transition_cost(
    prev: _IslandDpCandidate,
    curr: _IslandDpCandidate,
    prev_idx: int,
    curr_idx: int,
    os_factor: int,
    branch_profile: np.ndarray | None,
    path_config: PhasePathSelectorConfig,
    config: IslandReconstructionConfig,
) -> float:
    return float(
        float(config.transition_weight) * _phase_transition_cost(prev, curr, path_config)
        + float(config.branch_transition_weight)
        * _branch_transition_cost(prev, curr, prev_idx, curr_idx, os_factor, branch_profile, config)
    )


def _score_fixed_path(
    path: Sequence[_IslandDpCandidate],
    lo: int,
    hi: int,
    left: _IslandDpCandidate | None,
    right: _IslandDpCandidate | None,
    os_factor: int,
    branch_profile: np.ndarray | None,
    path_config: PhasePathSelectorConfig,
    config: IslandReconstructionConfig,
) -> float:
    if not path:
        return -float("inf")
    score = float(path[0].local_score)
    if left is not None:
        score -= float(config.boundary_weight) * _transition_cost(
            left, path[0], lo - 1, lo, os_factor, branch_profile, path_config, config
        )
    for rel in range(1, len(path)):
        idx = lo + rel
        score += float(path[rel].local_score) - _transition_cost(
            path[rel - 1],
            path[rel],
            idx - 1,
            idx,
            os_factor,
            branch_profile,
            path_config,
            config,
        )
    if right is not None:
        score -= float(config.boundary_weight) * _transition_cost(
            path[-1], right, hi - 1, hi, os_factor, branch_profile, path_config, config
        )
    return float(score)


def _best_baseline_path(
    candidate_rows: Sequence[Sequence[_IslandDpCandidate]],
    lo: int,
    baseline_bins: Sequence[int] | None,
) -> tuple[_IslandDpCandidate, ...] | None:
    if baseline_bins is None:
        return None
    out: list[_IslandDpCandidate] = []
    for rel, row in enumerate(candidate_rows):
        idx = lo + rel
        if idx >= len(baseline_bins):
            return None
        baseline_bin = int(baseline_bins[idx])
        candidates = [cand for cand in row if int(cand.raw_bin) == baseline_bin]
        if not candidates:
            return None
        out.append(max(candidates, key=lambda cand: float(cand.local_score)))
    return tuple(out)


def _solve_island(
    evidences: Sequence[SymbolEvidence],
    lo: int,
    hi: int,
    locked_mask: Sequence[bool],
    anchor_by_index: dict[int, _AnchorObservation],
    phase_profile: np.ndarray | None,
    branch_profile: np.ndarray | None,
    path_config: PhasePathSelectorConfig,
    config: IslandReconstructionConfig,
    branch_spectra: Sequence[Sequence[np.ndarray]] | None,
    dechirped_symbols: Sequence[np.ndarray | None] | None,
    os_factor: int,
    baseline_bins: Sequence[int] | None,
    residual_sto_chips: Sequence[float] | None,
    branch_residual_sto_chips: Sequence[Sequence[float]] | None,
    auxiliary_evidence_powers: Sequence[Sequence[np.ndarray]] | None,
) -> tuple[tuple[int, ...], list[float], int]:
    left = (
        _anchor_candidate_from_observation(anchor_by_index[lo - 1])
        if lo > 0 and bool(locked_mask[lo - 1]) and (lo - 1) in anchor_by_index
        else None
    )
    right = (
        _anchor_candidate_from_observation(anchor_by_index[hi])
        if hi < len(evidences) and bool(locked_mask[hi]) and hi in anchor_by_index
        else None
    )
    if bool(config.require_both_anchors) and (left is None or right is None):
        return (), [], 0

    candidate_rows: list[tuple[_IslandDpCandidate, ...]] = []
    for idx in range(lo, hi):
        spectra = branch_spectra[idx] if branch_spectra is not None and idx < len(branch_spectra) else None
        dechirped = dechirped_symbols[idx] if dechirped_symbols is not None and idx < len(dechirped_symbols) else None
        baseline_bin = int(baseline_bins[idx]) if baseline_bins is not None and idx < len(baseline_bins) else None
        residual_sto_chip = (
            float(residual_sto_chips[idx])
            if residual_sto_chips is not None and idx < len(residual_sto_chips)
            else None
        )
        branch_residual_row = (
            branch_residual_sto_chips[idx]
            if branch_residual_sto_chips is not None and idx < len(branch_residual_sto_chips)
            else None
        )
        auxiliary_powers = (
            auxiliary_evidence_powers[idx]
            if auxiliary_evidence_powers is not None and idx < len(auxiliary_evidence_powers)
            else None
        )
        row = _make_candidate_row(
            evidences[idx],
            idx,
            path_config,
            config,
            os_factor,
            spectra,
            dechirped,
            phase_profile,
            branch_profile,
            baseline_bin,
            residual_sto_chip,
            branch_residual_row,
            auxiliary_powers,
        )
        if not row:
            return (), [], 0
        candidate_rows.append(row)
    if not candidate_rows:
        return (), [], 0

    states: dict[tuple[int, int], tuple[float, tuple[_IslandDpCandidate, ...]]] = {}
    for cand in candidate_rows[0]:
        score = float(cand.local_score)
        if left is not None:
            score -= float(config.boundary_weight) * _transition_cost(
                left, cand, lo - 1, lo, os_factor, branch_profile, path_config, config
            )
        states[(int(cand.raw_bin), int(cand.branch))] = (score, (cand,))

    for rel, row in enumerate(candidate_rows[1:], start=1):
        idx = lo + rel
        prev_idx = idx - 1
        next_states: dict[tuple[int, int], tuple[float, tuple[_IslandDpCandidate, ...]]] = {}
        for prev_score, prev_path in states.values():
            prev = prev_path[-1]
            for cand in row:
                step = float(cand.local_score) - _transition_cost(
                    prev, cand, prev_idx, idx, os_factor, branch_profile, path_config, config
                )
                next_score = float(prev_score + step)
                key = (int(cand.raw_bin), int(cand.branch))
                if key not in next_states or next_score > next_states[key][0]:
                    next_states[key] = (next_score, prev_path + (cand,))
        states = next_states

    best_score = -float("inf")
    best_path: tuple[_IslandDpCandidate, ...] = ()
    for score, path in states.values():
        final = float(score)
        if right is not None and path:
            final -= float(config.boundary_weight) * _transition_cost(
                path[-1], right, hi - 1, hi, os_factor, branch_profile, path_config, config
            )
        if final > best_score:
            best_score = final
            best_path = path
    baseline_path = _best_baseline_path(candidate_rows, lo, baseline_bins)
    if baseline_path is not None:
        baseline_score = _score_fixed_path(
            baseline_path,
            lo,
            hi,
            left,
            right,
            os_factor,
            branch_profile,
            path_config,
            config,
        )
        if float(best_score) < float(baseline_score) + float(config.island_accept_margin):
            return (
                tuple(int(item.raw_bin) for item in baseline_path),
                [float(item.local_score) for item in baseline_path],
                sum(len(row) for row in candidate_rows),
            )
    return (
        tuple(int(item.raw_bin) for item in best_path),
        [float(item.local_score) for item in best_path],
        sum(len(row) for row in candidate_rows),
    )


def select_island_reconstruction_viterbi_path(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PhasePathSelectorConfig | None = None,
    reconstruction_config: IslandReconstructionConfig | None = None,
    offset_coherences: Sequence[np.ndarray] | None = None,
    branch_spectra: Sequence[Sequence[np.ndarray]] | None = None,
    dechirped_symbols: Sequence[np.ndarray | None] | None = None,
    os_factor: int | None = None,
    baseline_bins: Sequence[int] | None = None,
    residual_sto_chips: Sequence[float] | None = None,
    branch_residual_sto_chips: Sequence[Sequence[float]] | None = None,
    auxiliary_evidence_powers: Sequence[Sequence[np.ndarray]] | None = None,
) -> SymbolPhaseResult:
    """Run anchor-locked island DP with branch-aware reconstruction scoring."""

    path_cfg = config or PhasePathSelectorConfig()
    cfg = reconstruction_config or IslandReconstructionConfig()
    os_value = max(1, int(os_factor or 1))
    stage1_config = SymbolPhaseConfig(
        top_l_low_confidence=max(int(path_cfg.top_l), int(cfg.top_l)),
        lock_margin_db=float("inf"),
        lock_peak_to_median_db=float("inf"),
        coherence_candidate_top_l=0,
    )
    evidences = tuple(build_symbol_evidences(center_spectra, evidence_powers, abs_indices, stage1_config, offset_coherences))
    if not evidences:
        return SymbolPhaseResult(
            selected_raw_bins=(),
            locked_mask=(),
            evidences=(),
            phase_line=PhaseLine(),
            trajectory_score=0.0,
            line_score=0.0,
            mean_phase_score=0.0,
            mean_amp_score=0.0,
            uncertain_count=0,
            locked_count=0,
            beam_final_size=0,
            error="no symbol evidence",
        )

    locked_mask = tuple(_anchor_locked(ev, cfg) for ev in evidences)
    anchor_observations, phase_profile, branch_profile = _anchor_profiles(
        evidences,
        locked_mask,
        os_value,
        branch_spectra,
        dechirped_symbols,
        cfg,
    )
    anchor_by_index = {int(item.index): item for item in anchor_observations}
    baseline_prior = baseline_bins if bool(cfg.use_baseline_prior) else None
    if baseline_prior is not None:
        selected = [
            int(baseline_prior[idx]) if idx < len(baseline_prior) else int(ev.top1_bin)
            for idx, ev in enumerate(evidences)
        ]
        for idx, locked in enumerate(locked_mask):
            if bool(locked):
                selected[idx] = int(evidences[idx].top1_bin)
    else:
        selected = [int(ev.top1_bin) for ev in evidences]
    chosen_local_scores: list[float] = []
    candidate_count = 0
    for lo, hi in _island_ranges(locked_mask):
        path, scores, count = _solve_island(
            evidences,
            lo,
            hi,
            locked_mask,
            anchor_by_index,
            phase_profile,
            branch_profile,
            path_cfg,
            cfg,
            branch_spectra,
            dechirped_symbols,
            os_value,
            baseline_prior,
            residual_sto_chips,
            branch_residual_sto_chips,
            auxiliary_evidence_powers,
        )
        for rel, raw_bin in enumerate(path):
            selected[lo + rel] = int(raw_bin)
        chosen_local_scores.extend(scores)
        candidate_count += int(count)

    selected_tuple = tuple(int(v) for v in selected)
    line = fit_selected_phase_line(
        center_spectra=[ev.center_spectrum for ev in evidences],
        selected_bins=selected_tuple,
        abs_indices=[ev.abs_symbol_index for ev in evidences],
        trim_frac=float(path_cfg.line_trim_frac),
    )
    amp_scores = [_energy_score(ev, raw_bin) for ev, raw_bin in zip(evidences, selected_tuple)]
    profile_ready = bool(phase_profile is not None and branch_profile is not None)
    return SymbolPhaseResult(
        selected_raw_bins=selected_tuple,
        locked_mask=locked_mask,
        evidences=evidences,
        phase_line=line,
        trajectory_score=float(np.mean(chosen_local_scores)) if chosen_local_scores else 0.0,
        line_score=0.0,
        mean_phase_score=1.0 if profile_ready else 0.0,
        mean_amp_score=float(np.mean(amp_scores)) if amp_scores else 0.0,
        uncertain_count=int(sum(not v for v in locked_mask)),
        locked_count=int(sum(locked_mask)),
        beam_final_size=int(candidate_count),
        error="" if profile_ready else "island reconstruction ran without enough locked anchors for phase/branch profiles",
    )


__all__ = [
    "IslandReconstructionConfig",
    "compute_two_segment_score",
    "select_island_reconstruction_viterbi_path",
]

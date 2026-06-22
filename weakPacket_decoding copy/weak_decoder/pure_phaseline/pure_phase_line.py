"""Direct phase-line payload FFT-bin selection.

This module is intentionally narrower than ``weak_decoder.phase_line``:

* no Savaux stage-1 evidence,
* no Top-K -> Viterbi two-stage path search,
* no CRC or codec feedback,
* no packet-template prior.

It takes payload FFT spectra, a phase-line reference, and selects each symbol bin
by phase residual plus a small optional amplitude term.  The default candidate
mask is an energy floor relative to each symbol argmax, which prevents the
trivial "some noise bin always has the right phase" failure while still avoiding
a compact Top-K candidate generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from ..phase_guided_demod import PhaseLine, fit_phase_line


CandidateMode = Literal["all", "energy_floor", "top_l"]
FallbackLineSource = Literal["argmax", "consensus", "none"]


@dataclass(frozen=True)
class PurePhaseLineConfig:
    """Parameters for direct phase-line selection."""

    candidate_mode: CandidateMode = "energy_floor"
    energy_floor_db: float = 24.0
    top_l: int = 128
    phase_weight: float = 1.0
    amplitude_weight: float = 0.05
    path_transition_weight: float = 0.65
    path_amplitude_weight: float = 0.35
    path_line_weight: float = 0.05
    path_slope_span_pi: float = 0.30
    path_slope_steps: int = 13
    refine_iterations: int = 1
    trim_frac: float = 0.20
    fallback_line_source: FallbackLineSource = "consensus"
    fallback_anchor_fraction: float = 0.50
    fallback_min_margin_db: float = 0.0
    consensus_slope_steps: int = 41
    consensus_intercept_steps: int = 96
    consensus_slope_span_pi: float = 0.25
    consensus_phase_weight: float = 0.75
    consensus_min_score: float = 0.0


@dataclass(frozen=True)
class PurePhaseLineResult:
    """Selected bins and diagnostics for one packet."""

    selected_raw_bins: tuple[int, ...]
    phase_line: PhaseLine
    reference_line: PhaseLine
    mean_phase_score: float
    mean_amp_score: float
    mean_candidate_count: float
    mean_selected_energy_drop_db: float
    refinement_count: int
    error: str = ""


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(10.0 * math.log10((float(numerator) + 1e-30) / (float(denominator) + 1e-30)))


def _wrap_phase(value: np.ndarray | float) -> np.ndarray | float:
    return np.angle(np.exp(1j * value))


def _candidate_indices(power: np.ndarray, config: PurePhaseLineConfig) -> np.ndarray:
    if power.size == 0:
        return np.asarray([], dtype=np.int64)
    mode = str(config.candidate_mode)
    if mode == "all":
        return np.arange(power.size, dtype=np.int64)

    order = np.argsort(power)[::-1].astype(np.int64)
    if mode == "top_l":
        return order[: max(1, min(int(config.top_l), power.size))]

    max_power = float(power[int(order[0])])
    rel_db = 10.0 * np.log10((power + 1e-30) / (max_power + 1e-30))
    selected = np.flatnonzero(rel_db >= -abs(float(config.energy_floor_db))).astype(np.int64)
    if selected.size == 0:
        return order[:1]
    return selected


def _fit_line_from_bins(
    spectra: Sequence[np.ndarray],
    selected_bins: Sequence[int],
    abs_indices: Sequence[float],
    trim_frac: float,
) -> PhaseLine:
    xs: list[float] = []
    phases: list[float] = []
    for spectrum, raw_bin, abs_index in zip(spectra, selected_bins, abs_indices):
        spec = np.asarray(spectrum, dtype=np.complex64)
        b = int(raw_bin)
        if b < 0 or b >= spec.size:
            continue
        xs.append(float(abs_index))
        phases.append(float(np.angle(spec[b])))
    if len(xs) < 2:
        return PhaseLine()
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    x_arr = np.asarray(xs, dtype=np.float64)[order]
    phase_arr = np.unwrap(np.asarray(phases, dtype=np.float64)[order])
    return fit_phase_line(x_arr, phase_arr, trim_frac=max(0.0, min(0.45, float(trim_frac))))


def _argmax_margin_db(power: np.ndarray) -> float:
    if power.size < 2:
        return 0.0
    order = np.argsort(power)[::-1]
    return _db_ratio(float(power[int(order[0])]), float(power[int(order[1])]))


def _fallback_argmax_phase_line(
    spectra: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PurePhaseLineConfig,
) -> PhaseLine:
    anchors: list[tuple[float, float, float]] = []
    for spectrum, abs_index in zip(spectra, abs_indices):
        spec = np.asarray(spectrum, dtype=np.complex64)
        if spec.size == 0:
            continue
        power = np.abs(spec).astype(np.float64) ** 2
        if power.size == 0:
            continue
        margin_db = _argmax_margin_db(power)
        if margin_db < float(config.fallback_min_margin_db):
            continue
        raw_bin = int(np.argmax(power))
        anchors.append((float(abs_index), float(np.angle(spec[raw_bin])), float(margin_db)))
    if len(anchors) < 2:
        return PhaseLine()

    anchors.sort(key=lambda item: item[2], reverse=True)
    keep_count = max(
        2,
        int(math.ceil(len(anchors) * max(0.05, min(1.0, float(config.fallback_anchor_fraction))))),
    )
    kept = anchors[:keep_count]
    xs = np.asarray([item[0] for item in kept], dtype=np.float64)
    phases = np.unwrap(np.asarray([item[1] for item in kept], dtype=np.float64))
    weights = np.asarray([max(0.1, item[2]) for item in kept], dtype=np.float64)
    return fit_phase_line(
        xs,
        phases,
        trim_frac=max(0.0, min(0.45, float(config.trim_frac))),
        weights=weights,
    )


def _fallback_consensus_phase_line(
    spectra: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    initial_line: PhaseLine,
    config: PurePhaseLineConfig,
) -> PhaseLine:
    xs: list[float] = []
    phase_rows: list[np.ndarray] = []
    amp_rows: list[np.ndarray] = []
    for spectrum, abs_index in zip(spectra, abs_indices):
        spec = np.asarray(spectrum, dtype=np.complex64)
        if spec.size == 0:
            continue
        power = np.abs(spec).astype(np.float64) ** 2
        candidates = _candidate_indices(power, config)
        if candidates.size == 0:
            continue
        max_power = float(np.max(power)) if power.size else 0.0
        xs.append(float(abs_index))
        phase_rows.append(np.angle(spec[candidates]).astype(np.float64))
        amp_rows.append(
            np.clip(power[candidates] / (max_power + 1e-30), 0.0, 1.0)
            if max_power > 0.0
            else np.zeros(candidates.size, dtype=np.float64)
        )
    if len(xs) < 2:
        return PhaseLine()

    slope_steps = max(3, int(config.consensus_slope_steps))
    intercept_steps = max(8, int(config.consensus_intercept_steps))
    if int(initial_line.anchor_count) >= 2:
        slope0 = float(initial_line.slope_rad)
        span = max(1e-6, float(config.consensus_slope_span_pi) * math.pi)
        slopes = np.linspace(slope0 - span, slope0 + span, slope_steps, dtype=np.float64)
    else:
        slopes = np.linspace(-math.pi, math.pi, slope_steps, endpoint=False, dtype=np.float64)
    intercepts = np.linspace(-math.pi, math.pi, intercept_steps, endpoint=False, dtype=np.float64)
    phase_weight = max(0.0, min(1.0, float(config.consensus_phase_weight)))

    best_score = -float("inf")
    best_slope = 0.0
    best_intercept = 0.0
    for slope in slopes:
        totals = np.zeros(intercept_steps, dtype=np.float64)
        for abs_index, phases, amps in zip(xs, phase_rows, amp_rows):
            predicted = float(slope) * float(abs_index) + intercepts[:, np.newaxis]
            residual = np.angle(np.exp(1j * (phases[np.newaxis, :] - predicted)))
            scores = phase_weight * (0.5 + 0.5 * np.cos(residual)) + (1.0 - phase_weight) * amps[np.newaxis, :]
            totals += np.max(scores, axis=1)
        pos = int(np.argmax(totals))
        score = float(totals[pos] / max(1, len(xs)))
        if score > best_score:
            best_score = score
            best_slope = float(slope)
            best_intercept = float(intercepts[pos])

    if best_score < float(config.consensus_min_score):
        return PhaseLine()

    selected_x: list[float] = []
    selected_phases: list[float] = []
    selected_weights: list[float] = []
    for abs_index, phases, amps in zip(xs, phase_rows, amp_rows):
        predicted = best_slope * float(abs_index) + best_intercept
        residual = np.angle(np.exp(1j * (phases - predicted)))
        phase_scores = 0.5 + 0.5 * np.cos(residual)
        scores = phase_weight * phase_scores + (1.0 - phase_weight) * amps
        pos = int(np.argmax(scores))
        selected_x.append(float(abs_index))
        selected_phases.append(float(predicted + float(residual[pos])))
        selected_weights.append(float(max(0.05, scores[pos])))
    order = np.argsort(np.asarray(selected_x, dtype=np.float64))
    x_arr = np.asarray(selected_x, dtype=np.float64)[order]
    phase_arr = np.unwrap(np.asarray(selected_phases, dtype=np.float64)[order])
    weights = np.asarray(selected_weights, dtype=np.float64)[order]
    return fit_phase_line(
        x_arr,
        phase_arr,
        trim_frac=max(0.0, min(0.45, float(config.trim_frac))),
        weights=weights,
    )


def _select_once(
    spectra: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    line: PhaseLine,
    config: PurePhaseLineConfig,
) -> tuple[tuple[int, ...], float, float, float, float]:
    selected: list[int] = []
    phase_scores: list[float] = []
    amp_scores: list[float] = []
    candidate_counts: list[int] = []
    energy_drops: list[float] = []

    phase_weight = float(config.phase_weight)
    amp_weight = float(config.amplitude_weight)
    for spectrum, abs_index in zip(spectra, abs_indices):
        spec = np.asarray(spectrum, dtype=np.complex64)
        if spec.size == 0:
            selected.append(0)
            continue
        power = np.abs(spec).astype(np.float64) ** 2
        max_power = float(np.max(power)) if power.size else 0.0
        candidates = _candidate_indices(power, config)
        if candidates.size == 0:
            candidates = np.asarray([int(np.argmax(power))], dtype=np.int64)

        predicted = float(line.predict(float(abs_index)))
        residual = _wrap_phase(np.angle(spec[candidates]) - predicted)
        phase_score = 0.5 + 0.5 * np.cos(residual)
        if max_power > 0.0:
            amp_score = np.clip(power[candidates] / (max_power + 1e-30), 0.0, 1.0)
        else:
            amp_score = np.zeros(candidates.size, dtype=np.float64)
        score = phase_weight * phase_score + amp_weight * amp_score
        pos = int(np.argmax(score))
        raw_bin = int(candidates[pos])
        selected.append(raw_bin)
        phase_scores.append(float(phase_score[pos]))
        amp_scores.append(float(amp_score[pos]))
        candidate_counts.append(int(candidates.size))
        energy_drops.append(_db_ratio(float(power[raw_bin]), max_power))

    return (
        tuple(selected),
        float(np.mean(phase_scores)) if phase_scores else 0.0,
        float(np.mean(amp_scores)) if amp_scores else 0.0,
        float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        float(np.mean(energy_drops)) if energy_drops else 0.0,
    )


def select_pure_phase_line_bins(
    spectra: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    reference_line: PhaseLine | None = None,
    config: PurePhaseLineConfig | None = None,
) -> PurePhaseLineResult:
    """Select raw FFT bins directly from a phase-line reference.

    ``spectra`` should normally be center-offset dechirp FFTs, or spectra whose
    complex phase is still the center-offset FFT phase.
    """

    cfg = config or PurePhaseLineConfig()
    count = min(len(spectra), len(abs_indices))
    spec_seq = tuple(np.asarray(spec, dtype=np.complex64) for spec in spectra[:count])
    abs_seq = tuple(float(v) for v in abs_indices[:count])
    if count <= 0:
        return PurePhaseLineResult(
            selected_raw_bins=(),
            phase_line=PhaseLine(),
            reference_line=PhaseLine(),
            mean_phase_score=0.0,
            mean_amp_score=0.0,
            mean_candidate_count=0.0,
            mean_selected_energy_drop_db=0.0,
            refinement_count=0,
            error="empty_input",
        )

    ref = reference_line if reference_line is not None else PhaseLine()
    if int(ref.anchor_count) < 2:
        if str(cfg.fallback_line_source) == "argmax":
            ref = _fallback_argmax_phase_line(spec_seq, abs_seq, cfg)
        elif str(cfg.fallback_line_source) == "consensus":
            ref = _fallback_consensus_phase_line(spec_seq, abs_seq, ref, cfg)
    if int(ref.anchor_count) < 2:
        bins = tuple(int(np.argmax(np.abs(spec).astype(np.float64) ** 2)) for spec in spec_seq)
        return PurePhaseLineResult(
            selected_raw_bins=bins,
            phase_line=ref,
            reference_line=ref,
            mean_phase_score=0.0,
            mean_amp_score=0.0,
            mean_candidate_count=1.0,
            mean_selected_energy_drop_db=0.0,
            refinement_count=0,
            error="no_phase_line",
        )

    line = ref
    selected: tuple[int, ...] = ()
    phase_score = 0.0
    amp_score = 0.0
    candidate_count = 0.0
    energy_drop = 0.0
    refinements = max(0, int(cfg.refine_iterations))
    for iteration in range(refinements + 1):
        selected, phase_score, amp_score, candidate_count, energy_drop = _select_once(
            spec_seq,
            abs_seq,
            line,
            cfg,
        )
        if iteration < refinements:
            refined = _fit_line_from_bins(spec_seq, selected, abs_seq, trim_frac=float(cfg.trim_frac))
            if int(refined.anchor_count) < 2:
                break
            line = refined

    return PurePhaseLineResult(
        selected_raw_bins=selected,
        phase_line=line,
        reference_line=ref,
        mean_phase_score=float(phase_score),
        mean_amp_score=float(amp_score),
        mean_candidate_count=float(candidate_count),
        mean_selected_energy_drop_db=float(energy_drop),
        refinement_count=int(refinements),
        error="",
    )


def _candidate_table(
    spectra: Sequence[np.ndarray],
    config: PurePhaseLineConfig,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    table: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for spectrum in spectra:
        spec = np.asarray(spectrum, dtype=np.complex64)
        if spec.size == 0:
            table.append(
                (
                    np.asarray([0], dtype=np.int64),
                    np.asarray([0.0], dtype=np.float64),
                    np.asarray([0.0], dtype=np.float64),
                )
            )
            continue
        power = np.abs(spec).astype(np.float64) ** 2
        bins = _candidate_indices(power, config)
        if bins.size == 0:
            bins = np.asarray([int(np.argmax(power))], dtype=np.int64)
        max_power = float(np.max(power)) if power.size else 0.0
        phases = np.angle(spec[bins]).astype(np.float64)
        amps = np.clip(power[bins] / (max_power + 1e-30), 0.0, 1.0) if max_power > 0.0 else np.zeros(bins.size)
        table.append((bins.astype(np.int64), phases, amps.astype(np.float64)))
    return table


def _run_phase_path_dp(
    table: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    abs_indices: Sequence[float],
    slope_rad: float,
    intercept_rad: float,
    config: PurePhaseLineConfig,
) -> tuple[tuple[int, ...], float, float, float]:
    if not table:
        return (), 0.0, 0.0, 0.0

    transition_w = max(0.0, float(config.path_transition_weight))
    amp_w = max(0.0, float(config.path_amplitude_weight))
    line_w = max(0.0, float(config.path_line_weight))
    bins0, phases0, amps0 = table[0]
    dp = amp_w * amps0
    parents: list[np.ndarray] = []
    for idx in range(1, len(table)):
        prev_phases = table[idx - 1][1]
        bins, phases, amps = table[idx]
        delta_x = float(abs_indices[idx]) - float(abs_indices[idx - 1])
        expected_delta = float(slope_rad) * delta_x
        residual = np.angle(np.exp(1j * ((phases[:, np.newaxis] - prev_phases[np.newaxis, :]) - expected_delta)))
        transition_score = 0.5 + 0.5 * np.cos(residual)
        scores = dp[np.newaxis, :] + transition_w * transition_score
        best_prev = np.argmax(scores, axis=1).astype(np.int64)
        best_scores = scores[np.arange(scores.shape[0]), best_prev]
        if line_w > 0.0:
            line_pred = float(slope_rad) * float(abs_indices[idx]) + float(intercept_rad)
            line_score = 0.5 + 0.5 * np.cos(np.angle(np.exp(1j * (phases - line_pred))))
        else:
            line_score = np.zeros_like(amps)
        dp = best_scores + amp_w * amps + line_w * line_score
        parents.append(best_prev)

    pos = int(np.argmax(dp))
    selected_positions = [pos]
    for parent in reversed(parents):
        pos = int(parent[pos])
        selected_positions.append(pos)
    selected_positions.reverse()
    selected_bins: list[int] = []
    phase_scores: list[float] = []
    amp_scores: list[float] = []
    energy_drops: list[float] = []
    prev_phase: float | None = None
    prev_abs: float | None = None
    for idx, pos in enumerate(selected_positions):
        bins, phases, amps = table[idx]
        p = int(pos)
        selected_bins.append(int(bins[p]))
        amp_scores.append(float(amps[p]))
        if prev_phase is not None and prev_abs is not None:
            residual = float(np.angle(np.exp(1j * ((float(phases[p]) - prev_phase) - slope_rad * (float(abs_indices[idx]) - prev_abs)))))
            phase_scores.append(float(0.5 + 0.5 * math.cos(residual)))
        prev_phase = float(phases[p])
        prev_abs = float(abs_indices[idx])
        energy_drops.append(float(10.0 * math.log10(float(amps[p]) + 1e-30)))
    return (
        tuple(selected_bins),
        float(np.mean(phase_scores)) if phase_scores else 0.0,
        float(np.mean(amp_scores)) if amp_scores else 0.0,
        float(np.mean(energy_drops)) if energy_drops else 0.0,
    )


def select_pure_phase_path_bins(
    spectra: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    reference_line: PhaseLine | None = None,
    config: PurePhaseLineConfig | None = None,
) -> PurePhaseLineResult:
    """Select bins with a pure phase-continuity path DP.

    This is still not the current two-stage decoder: candidates come directly
    from the given spectra, and the path score uses only local amplitude plus
    adjacent phase continuity.
    """

    cfg = config or PurePhaseLineConfig()
    count = min(len(spectra), len(abs_indices))
    spec_seq = tuple(np.asarray(spec, dtype=np.complex64) for spec in spectra[:count])
    abs_seq = tuple(float(v) for v in abs_indices[:count])
    if count <= 0:
        return PurePhaseLineResult(
            selected_raw_bins=(),
            phase_line=PhaseLine(),
            reference_line=PhaseLine(),
            mean_phase_score=0.0,
            mean_amp_score=0.0,
            mean_candidate_count=0.0,
            mean_selected_energy_drop_db=0.0,
            refinement_count=0,
            error="empty_input",
        )

    ref = reference_line if reference_line is not None else PhaseLine()
    if str(cfg.fallback_line_source) == "consensus":
        line_for_slope = _fallback_consensus_phase_line(spec_seq, abs_seq, ref, cfg)
    elif int(ref.anchor_count) >= 2:
        line_for_slope = ref
    else:
        line_for_slope = _fallback_argmax_phase_line(spec_seq, abs_seq, cfg)
    if int(line_for_slope.anchor_count) < 2:
        bins = tuple(int(np.argmax(np.abs(spec).astype(np.float64) ** 2)) for spec in spec_seq)
        return PurePhaseLineResult(
            selected_raw_bins=bins,
            phase_line=line_for_slope,
            reference_line=ref,
            mean_phase_score=0.0,
            mean_amp_score=1.0,
            mean_candidate_count=1.0,
            mean_selected_energy_drop_db=0.0,
            refinement_count=0,
            error="no_phase_line",
        )

    table = _candidate_table(spec_seq, cfg)
    slope0 = float(line_for_slope.slope_rad)
    span = max(0.0, float(cfg.path_slope_span_pi) * math.pi)
    if int(cfg.path_slope_steps) <= 1 or span <= 0.0:
        slopes = np.asarray([slope0], dtype=np.float64)
    else:
        slopes = np.linspace(slope0 - span, slope0 + span, max(3, int(cfg.path_slope_steps)), dtype=np.float64)

    best_bins: tuple[int, ...] = ()
    best_phase = 0.0
    best_amp = 0.0
    best_energy_drop = 0.0
    best_score = -float("inf")
    best_slope = slope0
    for slope in slopes:
        bins, phase_score, amp_score, energy_drop = _run_phase_path_dp(
            table,
            abs_seq,
            float(slope),
            float(line_for_slope.intercept_rad),
            cfg,
        )
        score = float(cfg.path_transition_weight) * phase_score + float(cfg.path_amplitude_weight) * amp_score
        if score > best_score:
            best_score = score
            best_bins = bins
            best_phase = phase_score
            best_amp = amp_score
            best_energy_drop = energy_drop
            best_slope = float(slope)

    fitted = _fit_line_from_bins(spec_seq, best_bins, abs_seq, trim_frac=float(cfg.trim_frac))
    if int(fitted.anchor_count) < 2:
        fitted = PhaseLine(slope_rad=best_slope, intercept_rad=0.0, anchor_count=len(best_bins))
    return PurePhaseLineResult(
        selected_raw_bins=best_bins,
        phase_line=fitted,
        reference_line=line_for_slope,
        mean_phase_score=float(best_phase),
        mean_amp_score=float(best_amp),
        mean_candidate_count=float(np.mean([len(item[0]) for item in table])) if table else 0.0,
        mean_selected_energy_drop_db=float(best_energy_drop),
        refinement_count=0,
        error="",
    )

"""Symbol-level two-stage phase-aware FFT-bin selection.

This module deliberately stays at the raw FFT-bin level.  It does not enumerate
payload bytes, does not use application templates, and does not combine packets.
The intended flow is:

1. Use oversampled / multi-offset evidence to lock high-confidence symbol bins.
2. Keep only Top-L candidates for low-confidence symbols.
3. Fit a packet-local phase line from locked bins, or build a local phase
   trajectory directly with a sliding window.
4. Select bins for low-confidence symbols with a small beam over symbol bins.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .candidate_pruning import top_bins, wrap_phase
from .phase_guided_demod import PhaseLine, fit_phase_line


@dataclass(frozen=True)
class PhaseCurve:
    """Polynomial packet-local phase predictor.

    Degree 1 is equivalent to the legacy PhaseLine.  Degree 2 is used to test
    whether weak-packet residual phase has a slow curvature inside the payload.
    """

    coefficients: tuple[float, ...] = ()
    degree: int = 0
    anchor_count: int = 0
    fit_r2: float = 0.0
    fit_rmse_pi: float = float("nan")
    residual_std_pi: float = float("nan")

    def predict(self, abs_symbol_index: float) -> float:
        if not self.coefficients:
            return 0.0
        return float(np.polyval(np.asarray(self.coefficients, dtype=np.float64), float(abs_symbol_index)))

    @property
    def slope_pi(self) -> float:
        if not self.coefficients:
            return 0.0
        if int(self.degree) == 1 and len(self.coefficients) >= 2:
            return float(self.coefficients[0] / math.pi)
        if int(self.degree) == 2 and len(self.coefficients) >= 3:
            # Report the local slope at x=0 for compact diagnostics.
            return float(self.coefficients[1] / math.pi)
        return 0.0


@dataclass(frozen=True)
class SymbolPhaseConfig:
    """Tunable parameters for symbol-level two-stage peak selection."""

    top_l_low_confidence: int = 24
    lock_margin_db: float = 1.5
    lock_peak_to_median_db: float = 5.0
    lock_phase_score: float = 0.35
    min_locked_for_line: int = 4
    line_trim_frac: float = 0.25
    phase_model: str = "linear"
    selection_mode: str = "coherence"
    beam_width: int = 128
    trajectory_rmse_scale_pi: float = 0.30
    phase_weight: float = 0.20
    line_weight: float = 0.00
    amp_weight: float = 0.80
    profile_weight: float = 0.00
    phase_override_min_gain: float = 0.15
    phase_override_max_drop_db: float = 0.60
    phase_override_score_margin: float = 0.06
    phase_override_min_line_anchors: int = 8
    phase_override_max_line_rmse_pi: float = 0.25
    coherence_weight: float = 0.0
    coherence_candidate_top_l: int = 0
    lock_min_coherence: float = 0.0
    smooth_phase_weight: float = 0.05
    smooth_amp_weight: float = 0.50
    smooth_coherence_weight: float = 0.90
    smooth_slope_penalty: float = 0.05
    smooth_curvature_penalty: float = 0.10
    smooth_max_energy_drop_db: float = 20.0
    smooth_min_line_anchors: int = 4
    smooth_min_locked_ratio: float = 0.0
    smooth_max_line_rmse_pi: float = float("inf")
    window_size: int = 5
    window_degree: int = 1
    window_phase_weight: float = 0.05
    window_amp_weight: float = 0.50
    window_coherence_weight: float = 0.90
    window_slope_weight: float = 0.00
    window_curvature_weight: float = 0.00
    window_phase_scale_pi: float = 0.25
    window_slope_scale_pi: float = 0.45
    window_curvature_scale_pi: float = 0.25
    window_recent_decay: float = 0.75
    window_anchor_span: float = 8.0
    window_anchor_min: int = 2
    window_anchor_max_rmse_pi: float = 0.40
    window_min_locked_ratio: float = 0.10
    window_guard_min_phase_gain: float = 0.10
    window_guard_max_energy_drop_db: float = 0.75
    window_guard_max_coherence_drop: float = 0.08


@dataclass(frozen=True)
class SymbolEvidence:
    """Per-symbol FFT evidence and Stage-1 candidate data."""

    symbol_index: int
    abs_symbol_index: float
    center_spectrum: np.ndarray
    evidence_power: np.ndarray
    top_bins: tuple[int, ...]
    top_scores: tuple[float, ...]
    top1_bin: int
    top1_margin_db: float
    top1_peak_to_median_db: float
    top1_amp_score: float
    top1_coherence_score: float
    confidence: float
    offset_coherence: np.ndarray | None = None
    locked: bool = False
    phase_score_to_line: float = 0.0


@dataclass(frozen=True)
class CandidateBin:
    """One candidate raw FFT bin for an uncertain symbol."""

    symbol_index: int
    raw_bin: int
    phase_score: float
    amp_score: float
    local_score: float


@dataclass(frozen=True)
class SymbolPhaseResult:
    """Selected raw bins and diagnostics for one packet."""

    selected_raw_bins: tuple[int, ...]
    locked_mask: tuple[bool, ...]
    evidences: tuple[SymbolEvidence, ...]
    phase_line: PhaseLine | PhaseCurve
    trajectory_score: float
    line_score: float
    mean_phase_score: float
    mean_amp_score: float
    uncertain_count: int
    locked_count: int
    beam_final_size: int
    error: str = ""


@dataclass(frozen=True)
class _BeamState:
    selected_bins: tuple[int, ...]
    selected_uncertain_indexes: tuple[int, ...]
    score: float
    trajectory_score: float
    line: PhaseLine | PhaseCurve


@dataclass(frozen=True)
class _SmoothBeamState:
    selected_bins: tuple[int, ...]
    selected_uncertain_indexes: tuple[int, ...]
    score: float
    last_residual: float | None = None
    last_slope: float | None = None
    last_abs_index: float | None = None


@dataclass(frozen=True)
class _WindowBeamState:
    selected_bins: tuple[int, ...]
    selected_uncertain_indexes: tuple[int, ...]
    phases: tuple[float, ...]
    score: float


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(10.0 * math.log10((float(numerator) + 1e-30) / (float(denominator) + 1e-30)))


def _fit_line_for_bins(
    evidences: Sequence[SymbolEvidence],
    selected_bins: Sequence[int],
    trim_frac: float,
    indexes: Sequence[int] | None = None,
) -> PhaseLine:
    xs: list[float] = []
    phases: list[float] = []
    use_indexes = list(range(min(len(evidences), len(selected_bins)))) if indexes is None else [int(v) for v in indexes]
    for idx in use_indexes:
        if idx < 0 or idx >= len(evidences) or idx >= len(selected_bins):
            continue
        ev = evidences[idx]
        raw_bin = selected_bins[idx]
        b = int(raw_bin)
        if b < 0 or b >= ev.center_spectrum.size:
            continue
        xs.append(float(ev.abs_symbol_index))
        phases.append(float(np.angle(ev.center_spectrum[b])))
    if len(xs) < 2:
        return PhaseLine()
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    x_arr = np.asarray(xs, dtype=np.float64)[order]
    phase_arr = np.unwrap(np.asarray(phases, dtype=np.float64)[order])
    return fit_phase_line(x_arr, phase_arr, trim_frac=float(trim_frac))


def _fit_phase_curve(
    xs: Sequence[float],
    phases: Sequence[float],
    trim_frac: float,
    phase_model: str,
) -> PhaseLine | PhaseCurve:
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.unwrap(np.asarray(list(phases), dtype=np.float64))
    if x.size < 2:
        return PhaseLine()
    model = str(phase_model).lower()
    if model not in {"linear", "quadratic"}:
        model = "linear"
    if model == "linear":
        return fit_phase_line(x, y, trim_frac=float(trim_frac))

    degree = 2 if x.size >= 3 else 1
    keep_x = x
    keep_y = y
    coef = np.polyfit(keep_x, keep_y, deg=degree)
    for iteration in range(2 if trim_frac > 0 else 1):
        pred = np.polyval(coef, keep_x)
        residuals = keep_y - pred
        if float(trim_frac) > 0.0 and iteration == 0:
            keep_count = max(degree + 1, int(keep_x.size * (1.0 - float(trim_frac))))
            keep_idx = np.argsort(np.abs(residuals))[:keep_count]
            keep_x = keep_x[keep_idx]
            keep_y = keep_y[keep_idx]
            degree = 2 if keep_x.size >= 3 else 1
            coef = np.polyfit(keep_x, keep_y, deg=degree)
        else:
            break
    pred = np.polyval(coef, keep_x)
    residuals = keep_y - pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((keep_y - float(np.mean(keep_y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    rmse = float(math.sqrt(np.mean(residuals**2)))
    rstd = float(np.std(residuals))
    return PhaseCurve(
        coefficients=tuple(float(v) for v in coef),
        degree=int(degree),
        anchor_count=int(keep_x.size),
        fit_r2=r2,
        fit_rmse_pi=rmse / math.pi,
        residual_std_pi=rstd / math.pi,
    )


def _fit_phase_curve_for_bins(
    evidences: Sequence[SymbolEvidence],
    selected_bins: Sequence[int],
    trim_frac: float,
    phase_model: str,
    indexes: Sequence[int] | None = None,
) -> PhaseLine | PhaseCurve:
    xs: list[float] = []
    phases: list[float] = []
    use_indexes = list(range(min(len(evidences), len(selected_bins)))) if indexes is None else [int(v) for v in indexes]
    for idx in use_indexes:
        if idx < 0 or idx >= len(evidences) or idx >= len(selected_bins):
            continue
        ev = evidences[idx]
        raw_bin = int(selected_bins[idx])
        if raw_bin < 0 or raw_bin >= ev.center_spectrum.size:
            continue
        xs.append(float(ev.abs_symbol_index))
        phases.append(float(np.angle(ev.center_spectrum[raw_bin])))
    order = np.argsort(np.asarray(xs, dtype=np.float64)) if xs else np.asarray([], dtype=np.int64)
    xs_sorted = [xs[int(i)] for i in order]
    phases_sorted = [phases[int(i)] for i in order]
    return _fit_phase_curve(xs_sorted, phases_sorted, trim_frac=float(trim_frac), phase_model=str(phase_model))


def _phase_score(center_spectrum: np.ndarray, raw_bin: int, line: PhaseLine | PhaseCurve, abs_index: float) -> float:
    b = int(raw_bin)
    if b < 0 or b >= center_spectrum.size:
        return 0.0
    residual = float(wrap_phase(float(np.angle(center_spectrum[b])) - line.predict(float(abs_index))))
    return float(0.5 + 0.5 * math.cos(residual))


def _amp_score(power: np.ndarray, raw_bin: int) -> float:
    b = int(raw_bin)
    if b < 0 or b >= power.size:
        return 0.0
    max_power = float(np.max(power)) if power.size else 0.0
    if max_power <= 0.0:
        return 0.0
    return float(power[b] / (max_power + 1e-30))


def _coherence_score(ev: SymbolEvidence, raw_bin: int) -> float:
    if ev.offset_coherence is None:
        return 0.0
    b = int(raw_bin)
    if b < 0 or b >= ev.offset_coherence.size:
        return 0.0
    value = float(ev.offset_coherence[b])
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _phase_residual(center_spectrum: np.ndarray, raw_bin: int, line: PhaseLine | PhaseCurve, abs_index: float) -> float:
    b = int(raw_bin)
    if b < 0 or b >= center_spectrum.size or int(line.anchor_count) < 2:
        return 0.0
    return float(wrap_phase(float(np.angle(center_spectrum[b])) - line.predict(float(abs_index))))


def _local_candidate_score(
    ev: SymbolEvidence,
    raw_bin: int,
    line: PhaseLine | PhaseCurve,
    config: SymbolPhaseConfig,
) -> tuple[float, float, float]:
    phase = (
        _phase_score(ev.center_spectrum, int(raw_bin), line, ev.abs_symbol_index)
        if int(line.anchor_count) >= 2
        else 0.0
    )
    amp = _amp_score(ev.evidence_power, int(raw_bin))
    coherence = _coherence_score(ev, int(raw_bin))
    score = (
        float(config.phase_weight) * phase
        + float(config.amp_weight) * amp
        + float(config.coherence_weight) * coherence
        + float(config.profile_weight) * 0.0
    )
    return float(score), float(phase), float(amp)


def _eligible_candidates(
    ev: SymbolEvidence,
    line: PhaseLine | PhaseCurve,
    config: SymbolPhaseConfig,
) -> tuple[int, ...]:
    """Keep Top-1 plus conservative phase-approved alternatives.

    The second stage is allowed to repair ambiguous symbols, but it should not
    rewrite reliable multi-offset evidence.  Alternatives must beat the Top-1
    phase score by a minimum amount while staying close in fused energy.
    """

    top1 = int(ev.top1_bin)
    if int(line.anchor_count) < max(2, int(config.phase_override_min_line_anchors)):
        return (top1,)
    if (
        math.isfinite(float(line.fit_rmse_pi))
        and float(line.fit_rmse_pi) > float(config.phase_override_max_line_rmse_pi)
    ):
        return (top1,)

    top1_score, top1_phase, _top1_amp = _local_candidate_score(ev, top1, line, config)
    top1_power = float(ev.evidence_power[top1]) if 0 <= top1 < ev.evidence_power.size else 0.0
    out: list[int] = [top1]
    seen = {top1}
    for raw_bin in ev.top_bins:
        b = int(raw_bin)
        if b in seen:
            continue
        if b < 0 or b >= ev.evidence_power.size:
            continue
        score, phase, _amp = _local_candidate_score(ev, b, line, config)
        phase_gain = float(phase - top1_phase)
        energy_drop_db = _db_ratio(float(ev.evidence_power[b]), top1_power)
        if phase_gain < float(config.phase_override_min_gain):
            continue
        if energy_drop_db < -float(config.phase_override_max_drop_db):
            continue
        if score <= top1_score + float(config.phase_override_score_margin):
            continue
        out.append(b)
        seen.add(b)
    return tuple(out)


def _smooth_candidates(
    ev: SymbolEvidence,
    locked: bool,
    config: SymbolPhaseConfig,
) -> tuple[int, ...]:
    top1 = int(ev.top1_bin)
    if bool(locked):
        return (top1,)
    top1_power = float(ev.evidence_power[top1]) if 0 <= top1 < ev.evidence_power.size else 0.0
    out: list[int] = []
    seen: set[int] = set()
    for raw_bin in ev.top_bins:
        b = int(raw_bin)
        if b in seen or b < 0 or b >= ev.evidence_power.size:
            continue
        drop_db = _db_ratio(float(ev.evidence_power[b]), top1_power)
        if b != top1 and drop_db < -float(config.smooth_max_energy_drop_db):
            continue
        out.append(b)
        seen.add(b)
    return tuple(out) if out else (top1,)


def _smooth_local_score(
    ev: SymbolEvidence,
    raw_bin: int,
    line: PhaseLine | PhaseCurve,
    config: SymbolPhaseConfig,
) -> tuple[float, float, float, float]:
    residual = _phase_residual(ev.center_spectrum, raw_bin, line, ev.abs_symbol_index)
    phase = float(0.5 + 0.5 * math.cos(residual))
    amp = _amp_score(ev.evidence_power, raw_bin)
    coherence = _coherence_score(ev, raw_bin)
    score = (
        float(config.smooth_phase_weight) * phase
        + float(config.smooth_amp_weight) * amp
        + float(config.smooth_coherence_weight) * coherence
    )
    return float(score), residual, phase, amp


def _unwrap_phase_against(reference: float, phase: float) -> float:
    """Return `phase` shifted by 2pi so it is closest to `reference`."""

    return float(float(reference) + wrap_phase(float(phase) - float(reference)))


def _window_weights(count: int, decay: float) -> np.ndarray:
    n = int(count)
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    d = float(decay)
    if d <= 0.0 or d >= 1.0:
        return np.ones(n, dtype=np.float64)
    # Older samples get smaller weights; the newest point has weight 1.
    return np.asarray([d ** float(n - 1 - idx) for idx in range(n)], dtype=np.float64)


def _predict_phase_from_window(
    abs_indices: Sequence[float],
    phases: Sequence[float],
    target_abs_index: float,
    config: SymbolPhaseConfig,
) -> tuple[float, float, float]:
    """Predict local phase and return predicted phase, slope, and curvature.

    This is deliberately local.  It avoids declaring one global line to be the
    packet truth, and instead asks whether the current candidate continues the
    recent smooth phase trajectory.
    """

    count = min(len(abs_indices), len(phases), max(1, int(config.window_size)))
    if count <= 0:
        return 0.0, 0.0, 0.0

    xs = np.asarray(list(abs_indices)[-count:], dtype=np.float64)
    ys = np.asarray(list(phases)[-count:], dtype=np.float64)
    target = float(target_abs_index)
    if count == 1:
        return float(ys[-1]), 0.0, 0.0

    degree = min(max(0, int(config.window_degree)), 2, count - 1)
    if degree <= 0:
        return float(np.average(ys, weights=_window_weights(count, config.window_recent_decay))), 0.0, 0.0

    x0 = float(xs[-1])
    x_rel = xs - x0
    target_rel = target - x0
    weights = _window_weights(count, config.window_recent_decay)
    try:
        coef = np.polyfit(x_rel, ys, deg=degree, w=weights)
        pred = float(np.polyval(coef, target_rel))
        deriv = np.polyder(coef, m=1)
        slope = float(np.polyval(deriv, target_rel)) if deriv.size else 0.0
        second = np.polyder(coef, m=2)
        curvature = float(np.polyval(second, target_rel)) if second.size else 0.0
    except np.linalg.LinAlgError:
        dx = max(1e-6, float(xs[-1] - xs[-2]))
        slope = float((ys[-1] - ys[-2]) / dx)
        pred = float(ys[-1] + slope * (target - float(xs[-1])))
        curvature = 0.0
    return pred, slope, curvature


def _local_anchor_phase_prediction(
    evidences: Sequence[SymbolEvidence],
    locked: Sequence[bool],
    selected: Sequence[int],
    target_index: int,
    fallback_line: PhaseLine | PhaseCurve,
    config: SymbolPhaseConfig,
) -> tuple[bool, float, float, float]:
    target_idx = int(target_index)
    if target_idx < 0 or target_idx >= len(evidences):
        return False, 0.0, 0.0, 0.0
    target_abs = float(evidences[target_idx].abs_symbol_index)
    anchors: list[tuple[float, float, float]] = []
    span = float(config.window_anchor_span)
    for idx, ev in enumerate(evidences):
        if idx >= len(locked) or idx >= len(selected) or not bool(locked[idx]):
            continue
        raw_bin = int(selected[idx])
        if raw_bin < 0 or raw_bin >= ev.center_spectrum.size:
            continue
        distance = abs(float(ev.abs_symbol_index) - target_abs)
        if distance > span:
            continue
        # Prefer nearby anchors, but keep a floor so two adjacent anchors do not
        # create an infinitely sharp local model.
        weight = math.exp(-0.5 * (distance / max(1e-6, span * 0.5)) ** 2)
        anchors.append((float(ev.abs_symbol_index), float(np.angle(ev.center_spectrum[raw_bin])), float(weight)))

    if len(anchors) < max(1, int(config.window_anchor_min)):
        if int(fallback_line.anchor_count) >= 2:
            return True, float(fallback_line.predict(target_abs)), float(getattr(fallback_line, "slope_rad", 0.0)), 0.0
        return False, 0.0, 0.0, 0.0

    anchors.sort(key=lambda item: item[0])
    xs = np.asarray([item[0] for item in anchors], dtype=np.float64)
    phases = np.unwrap(np.asarray([item[1] for item in anchors], dtype=np.float64))
    weights = np.asarray([item[2] for item in anchors], dtype=np.float64)
    degree = min(max(0, int(config.window_degree)), 2, len(anchors) - 1)
    if degree <= 0:
        pred = float(np.average(phases, weights=weights))
        return True, pred, 0.0, 0.0

    x0 = target_abs
    x_rel = xs - x0
    try:
        coef = np.polyfit(x_rel, phases, deg=degree, w=weights)
    except np.linalg.LinAlgError:
        if int(fallback_line.anchor_count) >= 2:
            return True, float(fallback_line.predict(target_abs)), float(getattr(fallback_line, "slope_rad", 0.0)), 0.0
        return False, 0.0, 0.0, 0.0
    fitted = np.polyval(coef, x_rel)
    rmse_pi = float(math.sqrt(float(np.average((phases - fitted) ** 2, weights=weights))) / math.pi)
    if math.isfinite(rmse_pi) and rmse_pi > float(config.window_anchor_max_rmse_pi):
        if int(fallback_line.anchor_count) >= 2:
            return True, float(fallback_line.predict(target_abs)), float(getattr(fallback_line, "slope_rad", 0.0)), 0.0
        return False, 0.0, 0.0, 0.0
    pred = float(np.polyval(coef, 0.0))
    deriv = np.polyder(coef, m=1)
    slope = float(np.polyval(deriv, 0.0)) if deriv.size else 0.0
    second = np.polyder(coef, m=2)
    curvature = float(np.polyval(second, 0.0)) if second.size else 0.0
    return True, pred, slope, curvature


def _window_local_score(
    ev: SymbolEvidence,
    raw_bin: int,
    predicted_phase: float,
    predicted_slope: float,
    predicted_curvature: float,
    has_phase_prediction: bool,
    previous_phase: float | None,
    previous_slope: float | None,
    previous_abs_index: float | None,
    config: SymbolPhaseConfig,
) -> tuple[float, float, float, float, float, float]:
    b = int(raw_bin)
    if b < 0 or b >= ev.center_spectrum.size:
        return -float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0

    observed_raw = float(np.angle(ev.center_spectrum[b]))
    reference = float(predicted_phase)
    if previous_phase is not None:
        reference = float(previous_phase)
    observed = _unwrap_phase_against(reference, observed_raw)
    residual = float(wrap_phase(observed - float(predicted_phase)))

    phase_scale = max(1e-6, float(config.window_phase_scale_pi) * math.pi)
    slope_scale = max(1e-6, float(config.window_slope_scale_pi) * math.pi)
    curvature_scale = max(1e-6, float(config.window_curvature_scale_pi) * math.pi)

    phase_score = float(math.exp(-((residual / phase_scale) ** 2))) if bool(has_phase_prediction) else 0.0
    amp = _amp_score(ev.evidence_power, b)
    coherence = _coherence_score(ev, b)

    slope_penalty = 0.0
    curvature_penalty = 0.0
    local_slope = float(predicted_slope)
    if previous_phase is not None and previous_abs_index is not None:
        dx = max(1e-6, float(ev.abs_symbol_index) - float(previous_abs_index))
        local_slope = float((observed - float(previous_phase)) / dx)
        slope_penalty = float((local_slope / slope_scale) ** 2)
        if previous_slope is not None:
            local_curvature = float(local_slope - float(previous_slope))
            curvature_penalty = float((local_curvature / curvature_scale) ** 2)
    elif bool(has_phase_prediction):
        slope_penalty = float((float(predicted_slope) / slope_scale) ** 2)
        curvature_penalty = float((float(predicted_curvature) / curvature_scale) ** 2)

    score = (
        (float(config.window_phase_weight) * phase_score if bool(has_phase_prediction) else 0.0)
        + float(config.window_amp_weight) * amp
        + float(config.window_coherence_weight) * coherence
        - float(config.window_slope_weight) * slope_penalty
        - float(config.window_curvature_weight) * curvature_penalty
    )
    return float(score), observed, residual, phase_score, amp, coherence


def _select_with_window_path(
    evidences: Sequence[SymbolEvidence],
    selected: Sequence[int],
    locked: Sequence[bool],
    anchor_line: PhaseLine | PhaseCurve,
    config: SymbolPhaseConfig,
) -> tuple[tuple[int, ...], int, float] | None:
    if not evidences:
        return None

    beam = [
        _WindowBeamState(
            selected_bins=(),
            selected_uncertain_indexes=(),
            phases=(),
            score=0.0,
        )
    ]
    for idx in range(len(evidences)):
        ev = evidences[idx]
        candidates = _smooth_candidates(ev, bool(locked[idx]), config)
        next_beam: list[_WindowBeamState] = []
        for state in beam:
            history_count = min(len(state.phases), max(1, int(config.window_size)))
            hist_indices = [float(evidences[j].abs_symbol_index) for j in range(len(state.phases) - history_count, len(state.phases))]
            hist_phases = list(state.phases[-history_count:])
            has_phase_prediction, predicted, pred_slope, pred_curvature = _local_anchor_phase_prediction(
                evidences=evidences,
                locked=locked,
                selected=selected,
                target_index=idx,
                fallback_line=anchor_line,
                config=config,
            )
            if not has_phase_prediction and hist_phases:
                has_phase_prediction = True
                predicted, pred_slope, pred_curvature = _predict_phase_from_window(
                    hist_indices,
                    hist_phases,
                    ev.abs_symbol_index,
                    config,
                )

            previous_phase = state.phases[-1] if state.phases else None
            previous_abs = evidences[len(state.phases) - 1].abs_symbol_index if state.phases else None
            previous_slope = None
            if len(state.phases) >= 2:
                dx = max(
                    1e-6,
                    float(evidences[len(state.phases) - 1].abs_symbol_index)
                    - float(evidences[len(state.phases) - 2].abs_symbol_index),
                )
                previous_slope = float((state.phases[-1] - state.phases[-2]) / dx)

            for raw_bin in candidates:
                local_score, observed, _residual, _phase_score, _amp, _coherence = _window_local_score(
                    ev=ev,
                    raw_bin=int(raw_bin),
                    predicted_phase=predicted,
                    predicted_slope=pred_slope,
                    predicted_curvature=pred_curvature,
                    previous_phase=previous_phase,
                    previous_slope=previous_slope,
                    previous_abs_index=previous_abs,
                    has_phase_prediction=has_phase_prediction,
                    config=config,
                )
                if not math.isfinite(local_score):
                    continue
                next_beam.append(
                    _WindowBeamState(
                        selected_bins=state.selected_bins + (int(raw_bin),),
                        selected_uncertain_indexes=(
                            state.selected_uncertain_indexes
                            if bool(locked[idx])
                            else state.selected_uncertain_indexes + (int(idx),)
                        ),
                        phases=state.phases + (float(observed),),
                        score=float(state.score + local_score),
                    )
                )
        next_beam.sort(key=lambda item: item.score, reverse=True)
        beam = next_beam[: max(1, int(config.beam_width))]
        if not beam:
            return None

    best = beam[0]
    if len(best.selected_bins) != len(evidences):
        return None
    return tuple(int(v) for v in best.selected_bins), int(len(beam)), float(best.score)


def _select_with_guarded_window(
    evidences: Sequence[SymbolEvidence],
    selected: Sequence[int],
    locked: Sequence[bool],
    anchor_line: PhaseLine | PhaseCurve,
    config: SymbolPhaseConfig,
) -> tuple[int, ...]:
    guarded = [int(v) for v in selected]
    for idx, ev in enumerate(evidences):
        if idx >= len(guarded) or (idx < len(locked) and bool(locked[idx])):
            continue
        has_pred, predicted, pred_slope, pred_curvature = _local_anchor_phase_prediction(
            evidences=evidences,
            locked=locked,
            selected=guarded,
            target_index=idx,
            fallback_line=anchor_line,
            config=config,
        )
        if not has_pred:
            continue
        base_bin = int(guarded[idx])
        base_score, _base_observed, _base_residual, base_phase, base_amp, base_coherence = _window_local_score(
            ev=ev,
            raw_bin=base_bin,
            predicted_phase=predicted,
            predicted_slope=pred_slope,
            predicted_curvature=pred_curvature,
            has_phase_prediction=True,
            previous_phase=None,
            previous_slope=None,
            previous_abs_index=None,
            config=config,
        )
        base_power = float(ev.evidence_power[base_bin]) if 0 <= base_bin < ev.evidence_power.size else 0.0
        best_bin = base_bin
        best_score = float(base_score)
        for raw_bin in _smooth_candidates(ev, False, config):
            b = int(raw_bin)
            if b == base_bin or b < 0 or b >= ev.evidence_power.size:
                continue
            score, _observed, _residual, phase_score, amp, coherence = _window_local_score(
                ev=ev,
                raw_bin=b,
                predicted_phase=predicted,
                predicted_slope=pred_slope,
                predicted_curvature=pred_curvature,
                has_phase_prediction=True,
                previous_phase=None,
                previous_slope=None,
                previous_abs_index=None,
                config=config,
            )
            phase_gain = float(phase_score - base_phase)
            energy_drop_db = _db_ratio(float(ev.evidence_power[b]), base_power)
            coherence_drop = float(base_coherence - coherence)
            if phase_gain < float(config.window_guard_min_phase_gain):
                continue
            if energy_drop_db < -float(config.window_guard_max_energy_drop_db):
                continue
            if coherence_drop > float(config.window_guard_max_coherence_drop):
                continue
            if score > best_score:
                best_score = float(score)
                best_bin = b
        guarded[idx] = int(best_bin)
    return tuple(int(v) for v in guarded)


def _select_with_local_coherence(
    evidences: Sequence[SymbolEvidence],
    selected: Sequence[int],
    locked: Sequence[bool],
    line: PhaseLine | PhaseCurve,
    config: SymbolPhaseConfig,
) -> tuple[int, ...]:
    coherence_selected = [int(v) for v in selected]
    for idx, ev in enumerate(evidences):
        if idx < len(locked) and bool(locked[idx]):
            continue
        candidates = _smooth_candidates(ev, False, config)
        best_bin = int(ev.top1_bin)
        best_score = -float("inf")
        for raw_bin in candidates:
            score, _residual, _phase, _amp = _smooth_local_score(ev, int(raw_bin), line, config)
            if score > best_score:
                best_score = float(score)
                best_bin = int(raw_bin)
        coherence_selected[idx] = int(best_bin)
    return tuple(int(v) for v in coherence_selected)


def _select_with_smooth_path(
    evidences: Sequence[SymbolEvidence],
    selected: Sequence[int],
    locked: Sequence[bool],
    anchor_line: PhaseLine | PhaseCurve,
    config: SymbolPhaseConfig,
) -> tuple[int, ...] | None:
    if int(anchor_line.anchor_count) < max(2, int(config.smooth_min_line_anchors)):
        return None
    locked_ratio = float(sum(bool(v) for v in locked) / max(1, len(locked)))
    if locked_ratio < float(config.smooth_min_locked_ratio):
        return None
    if (
        math.isfinite(float(anchor_line.fit_rmse_pi))
        and float(anchor_line.fit_rmse_pi) > float(config.smooth_max_line_rmse_pi)
    ):
        return None
    beam: list[_SmoothBeamState] = [
        _SmoothBeamState(
            selected_bins=(),
            selected_uncertain_indexes=(),
            score=0.0,
            last_residual=None,
            last_slope=None,
            last_abs_index=None,
        )
    ]
    for idx, ev in enumerate(evidences):
        candidates = _smooth_candidates(ev, bool(locked[idx]), config)
        next_beam: list[_SmoothBeamState] = []
        for state in beam:
            for raw_bin in candidates:
                local_score, residual_raw, _phase, _amp = _smooth_local_score(ev, int(raw_bin), anchor_line, config)
                residual = float(residual_raw)
                slope = None
                transition_penalty = 0.0
                if state.last_residual is not None:
                    residual = float(state.last_residual + wrap_phase(residual_raw - state.last_residual))
                    dx = max(1e-6, float(ev.abs_symbol_index) - float(state.last_abs_index))
                    slope = float((residual - float(state.last_residual)) / dx)
                    transition_penalty += float(config.smooth_slope_penalty) * ((slope / math.pi) ** 2)
                    if state.last_slope is not None:
                        curvature = float(slope - float(state.last_slope))
                        transition_penalty += float(config.smooth_curvature_penalty) * ((curvature / math.pi) ** 2)
                next_beam.append(
                    _SmoothBeamState(
                        selected_bins=state.selected_bins + (int(raw_bin),),
                        selected_uncertain_indexes=(
                            state.selected_uncertain_indexes
                            if bool(locked[idx])
                            else state.selected_uncertain_indexes + (int(idx),)
                        ),
                        score=float(state.score + local_score - transition_penalty),
                        last_residual=residual,
                        last_slope=slope if slope is not None else state.last_slope,
                        last_abs_index=float(ev.abs_symbol_index),
                    )
                )
        next_beam.sort(key=lambda item: item.score, reverse=True)
        beam = next_beam[: max(1, int(config.beam_width))]
        if not beam:
            return None
    best = beam[0]
    if len(best.selected_bins) != len(evidences):
        return None
    return tuple(int(v) for v in best.selected_bins)


def _trajectory_score(
    evidences: Sequence[SymbolEvidence],
    selected_bins: Sequence[int],
    score_indexes: Sequence[int],
    line: PhaseLine | PhaseCurve,
    config: SymbolPhaseConfig,
) -> tuple[float, float, float, float]:
    if int(line.anchor_count) < 2:
        return 0.0, 0.0, 0.0, 0.0
    if not math.isfinite(float(line.fit_rmse_pi)):
        line_score = 0.0
    else:
        scale = max(1e-6, float(config.trajectory_rmse_scale_pi))
        line_score = float(math.exp(-((float(line.fit_rmse_pi) / scale) ** 2)))

    indexes = [int(v) for v in score_indexes]
    if not indexes:
        indexes = list(range(len(evidences)))
    phase_scores: list[float] = []
    amp_scores: list[float] = []
    for idx in indexes:
        if idx < 0 or idx >= len(evidences) or idx >= len(selected_bins):
            continue
        ev = evidences[idx]
        raw_bin = int(selected_bins[idx])
        phase_scores.append(_phase_score(ev.center_spectrum, raw_bin, line, ev.abs_symbol_index))
        amp_scores.append(_amp_score(ev.evidence_power, raw_bin))

    mean_phase = float(np.mean(phase_scores)) if phase_scores else 0.0
    mean_amp = float(np.mean(amp_scores)) if amp_scores else 0.0
    mean_profile = 0.0
    score = (
        float(config.phase_weight) * mean_phase
        + float(config.line_weight) * line_score
        + float(config.amp_weight) * mean_amp
        + float(config.profile_weight) * mean_profile
    )
    return float(score), float(line_score), mean_phase, mean_amp


def build_symbol_evidences(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: SymbolPhaseConfig,
    offset_coherences: Sequence[np.ndarray] | None = None,
) -> tuple[SymbolEvidence, ...]:
    """Build per-symbol evidence records from spectra and multi-offset power."""

    out: list[SymbolEvidence] = []
    top_l = max(1, int(config.top_l_low_confidence))
    count = min(len(center_spectra), len(evidence_powers), len(abs_indices))
    for idx in range(count):
        center = np.asarray(center_spectra[idx], dtype=np.complex64)
        power = np.asarray(evidence_powers[idx], dtype=np.float64)
        if center.size == 0 or power.size == 0 or center.size != power.size:
            continue
        coherence_vec: np.ndarray | None = None
        if offset_coherences is not None and idx < len(offset_coherences):
            raw_coherence = np.asarray(offset_coherences[idx], dtype=np.float64)
            if raw_coherence.size == power.size:
                coherence_vec = np.clip(raw_coherence, 0.0, 1.0)
        energy_bins = top_bins(power, min(max(top_l, 2), power.size))
        candidate_bins: list[int] = []
        seen: set[int] = set()
        for raw_bin in energy_bins:
            b = int(raw_bin)
            if b not in seen:
                candidate_bins.append(b)
                seen.add(b)
        coherence_top_l = max(0, int(config.coherence_candidate_top_l))
        if coherence_vec is not None and coherence_top_l > 0:
            for raw_bin in top_bins(coherence_vec, min(coherence_top_l, coherence_vec.size)):
                b = int(raw_bin)
                if b not in seen:
                    candidate_bins.append(b)
                    seen.add(b)
        bins = np.asarray(candidate_bins, dtype=np.int64)
        top1 = int(bins[0])
        top2 = int(energy_bins[1]) if energy_bins.size > 1 else top1
        top1_power = float(power[top1])
        top2_power = float(power[top2])
        median_power = float(np.median(power)) if power.size else 0.0
        margin_db = _db_ratio(top1_power, top2_power)
        peak_to_median_db = _db_ratio(top1_power, median_power)
        amp = _amp_score(power, top1)
        top1_coherence = float(coherence_vec[top1]) if coherence_vec is not None else 0.0
        confidence = float(margin_db + 0.20 * peak_to_median_db)
        locked = (
            margin_db >= float(config.lock_margin_db)
            and peak_to_median_db >= float(config.lock_peak_to_median_db)
            and top1_coherence >= float(config.lock_min_coherence)
        )
        out.append(
            SymbolEvidence(
                symbol_index=int(idx),
                abs_symbol_index=float(abs_indices[idx]),
                center_spectrum=center,
                evidence_power=power,
                top_bins=tuple(int(v) for v in bins),
                top_scores=tuple(float(power[int(v)]) for v in bins),
                top1_bin=top1,
                top1_margin_db=float(margin_db),
                top1_peak_to_median_db=float(peak_to_median_db),
                top1_amp_score=float(amp),
                top1_coherence_score=float(top1_coherence),
                confidence=confidence,
                offset_coherence=coherence_vec,
                locked=bool(locked),
            )
        )
    return tuple(out)


def select_symbol_bins_two_stage(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: SymbolPhaseConfig | None = None,
    fallback_line: PhaseLine | None = None,
    offset_coherences: Sequence[np.ndarray] | None = None,
) -> SymbolPhaseResult:
    """Select one raw FFT bin per payload symbol with symbol-level phase search."""

    cfg = config or SymbolPhaseConfig()
    evidences = list(build_symbol_evidences(center_spectra, evidence_powers, abs_indices, cfg, offset_coherences))
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

    selected = [int(ev.top1_bin) for ev in evidences]
    locked = [bool(ev.locked) for ev in evidences]

    line = _fit_phase_curve_for_bins(
        [ev for ev, is_locked in zip(evidences, locked) if is_locked],
        [bin_value for bin_value, is_locked in zip(selected, locked) if is_locked],
        trim_frac=float(cfg.line_trim_frac),
        phase_model=str(cfg.phase_model),
    )
    if int(line.anchor_count) < max(2, int(cfg.min_locked_for_line)) and fallback_line is not None:
        line = fallback_line

    if int(line.anchor_count) >= max(2, int(cfg.min_locked_for_line)):
        refined_locked: list[bool] = []
        for ev, is_locked in zip(evidences, locked):
            if not is_locked:
                refined_locked.append(False)
                continue
            ps = _phase_score(ev.center_spectrum, ev.top1_bin, line, ev.abs_symbol_index)
            refined_locked.append(bool(ps >= float(cfg.lock_phase_score)))
        if sum(refined_locked) >= max(2, int(cfg.min_locked_for_line)):
            locked = refined_locked
            line = _fit_phase_curve_for_bins(
                [ev for ev, is_locked in zip(evidences, locked) if is_locked],
                [bin_value for bin_value, is_locked in zip(selected, locked) if is_locked],
                trim_frac=float(cfg.line_trim_frac),
                phase_model=str(cfg.phase_model),
            )

    refined_evidences: list[SymbolEvidence] = []
    for ev, is_locked in zip(evidences, locked):
        phase_score = (
            _phase_score(ev.center_spectrum, ev.top1_bin, line, ev.abs_symbol_index)
            if int(line.anchor_count) >= 2
            else 0.0
        )
        refined_evidences.append(
            SymbolEvidence(
                symbol_index=ev.symbol_index,
                abs_symbol_index=ev.abs_symbol_index,
                center_spectrum=ev.center_spectrum,
                evidence_power=ev.evidence_power,
                top_bins=ev.top_bins,
                top_scores=ev.top_scores,
                top1_bin=ev.top1_bin,
                top1_margin_db=ev.top1_margin_db,
                top1_peak_to_median_db=ev.top1_peak_to_median_db,
                top1_amp_score=ev.top1_amp_score,
                top1_coherence_score=ev.top1_coherence_score,
                confidence=ev.confidence,
                offset_coherence=ev.offset_coherence,
                locked=bool(is_locked),
                phase_score_to_line=float(phase_score),
            )
        )
    evidences = refined_evidences

    uncertain = [idx for idx, is_locked in enumerate(locked) if not is_locked]
    uncertain_sorted = sorted(uncertain, key=lambda idx: evidences[idx].confidence, reverse=True)
    coherence_selected = _select_with_local_coherence(
        evidences=evidences,
        selected=selected,
        locked=locked,
        line=line,
        config=cfg,
    )

    if str(cfg.selection_mode).lower() == "coherence":
        final_line = _fit_phase_curve_for_bins(
            evidences,
            coherence_selected,
            trim_frac=float(cfg.line_trim_frac),
            phase_model=str(cfg.phase_model),
        )
        if int(final_line.anchor_count) < 2:
            final_line = line
        score_indexes = uncertain if uncertain else list(range(len(evidences)))
        final_score, line_score, mean_phase, mean_amp = _trajectory_score(
            evidences,
            coherence_selected,
            score_indexes,
            final_line,
            cfg,
        )
        return SymbolPhaseResult(
            selected_raw_bins=tuple(int(v) for v in coherence_selected),
            locked_mask=tuple(bool(v) for v in locked),
            evidences=tuple(evidences),
            phase_line=final_line,
            trajectory_score=float(final_score),
            line_score=float(line_score),
            mean_phase_score=float(mean_phase),
            mean_amp_score=float(mean_amp),
            uncertain_count=int(len(uncertain)),
            locked_count=int(sum(locked)),
            beam_final_size=1,
            error="",
        )

    if str(cfg.selection_mode).lower() == "smooth":
        smooth_selected = _select_with_smooth_path(
            evidences=evidences,
            selected=selected,
            locked=locked,
            anchor_line=line,
            config=cfg,
        )
        if smooth_selected is not None:
            final_line = _fit_phase_curve_for_bins(
                evidences,
                smooth_selected,
                trim_frac=float(cfg.line_trim_frac),
                phase_model=str(cfg.phase_model),
            )
            if int(final_line.anchor_count) < 2:
                final_line = line
            score_indexes = uncertain if uncertain else list(range(len(evidences)))
            final_score, line_score, mean_phase, mean_amp = _trajectory_score(
                evidences,
                smooth_selected,
                score_indexes,
                final_line,
                cfg,
            )
            return SymbolPhaseResult(
                selected_raw_bins=tuple(int(v) for v in smooth_selected),
                locked_mask=tuple(bool(v) for v in locked),
                evidences=tuple(evidences),
                phase_line=final_line,
                trajectory_score=float(final_score),
                line_score=float(line_score),
                mean_phase_score=float(mean_phase),
                mean_amp_score=float(mean_amp),
                uncertain_count=int(len(uncertain)),
                locked_count=int(sum(locked)),
                beam_final_size=int(max(1, int(cfg.beam_width))),
                error="",
            )

    if str(cfg.selection_mode).lower() == "window":
        locked_ratio = float(sum(bool(v) for v in locked) / max(1, len(locked)))
        window_result = None
        if locked_ratio >= float(cfg.window_min_locked_ratio):
            window_result = _select_with_window_path(
                evidences=evidences,
                selected=coherence_selected,
                locked=locked,
                anchor_line=line,
                config=cfg,
            )
        if window_result is not None:
            window_selected, window_beam_size, window_score = window_result
            final_line = _fit_phase_curve_for_bins(
                evidences,
                window_selected,
                trim_frac=float(cfg.line_trim_frac),
                phase_model=str(cfg.phase_model),
            )
            if int(final_line.anchor_count) < 2:
                final_line = line
            score_indexes = uncertain if uncertain else list(range(len(evidences)))
            final_score, line_score, mean_phase, mean_amp = _trajectory_score(
                evidences,
                window_selected,
                score_indexes,
                final_line,
                cfg,
            )
            return SymbolPhaseResult(
                selected_raw_bins=tuple(int(v) for v in window_selected),
                locked_mask=tuple(bool(v) for v in locked),
                evidences=tuple(evidences),
                phase_line=final_line,
                trajectory_score=float(window_score + final_score),
                line_score=float(line_score),
                mean_phase_score=float(mean_phase),
                mean_amp_score=float(mean_amp),
                uncertain_count=int(len(uncertain)),
                locked_count=int(sum(locked)),
                beam_final_size=int(window_beam_size),
                error="",
            )
        final_line = _fit_phase_curve_for_bins(
            evidences,
            coherence_selected,
            trim_frac=float(cfg.line_trim_frac),
            phase_model=str(cfg.phase_model),
        )
        if int(final_line.anchor_count) < 2:
            final_line = line
        score_indexes = uncertain if uncertain else list(range(len(evidences)))
        final_score, line_score, mean_phase, mean_amp = _trajectory_score(
            evidences,
            coherence_selected,
            score_indexes,
            final_line,
            cfg,
        )
        return SymbolPhaseResult(
            selected_raw_bins=tuple(int(v) for v in coherence_selected),
            locked_mask=tuple(bool(v) for v in locked),
            evidences=tuple(evidences),
            phase_line=final_line,
            trajectory_score=float(final_score),
            line_score=float(line_score),
            mean_phase_score=float(mean_phase),
            mean_amp_score=float(mean_amp),
            uncertain_count=int(len(uncertain)),
            locked_count=int(sum(locked)),
            beam_final_size=1,
            error="window fallback to coherence: low locked ratio or empty beam",
        )

    if str(cfg.selection_mode).lower() == "window_guarded":
        guarded_selected = _select_with_guarded_window(
            evidences=evidences,
            selected=coherence_selected,
            locked=locked,
            anchor_line=line,
            config=cfg,
        )
        final_line = _fit_phase_curve_for_bins(
            evidences,
            guarded_selected,
            trim_frac=float(cfg.line_trim_frac),
            phase_model=str(cfg.phase_model),
        )
        if int(final_line.anchor_count) < 2:
            final_line = line
        score_indexes = uncertain if uncertain else list(range(len(evidences)))
        final_score, line_score, mean_phase, mean_amp = _trajectory_score(
            evidences,
            guarded_selected,
            score_indexes,
            final_line,
            cfg,
        )
        return SymbolPhaseResult(
            selected_raw_bins=tuple(int(v) for v in guarded_selected),
            locked_mask=tuple(bool(v) for v in locked),
            evidences=tuple(evidences),
            phase_line=final_line,
            trajectory_score=float(final_score),
            line_score=float(line_score),
            mean_phase_score=float(mean_phase),
            mean_amp_score=float(mean_amp),
            uncertain_count=int(len(uncertain)),
            locked_count=int(sum(locked)),
            beam_final_size=1,
            error="",
        )

    initial_selected = tuple(selected)
    initial_score, initial_line_score, initial_phase, initial_amp = _trajectory_score(
        evidences,
        initial_selected,
        [idx for idx, is_locked in enumerate(locked) if is_locked],
        line,
        cfg,
    )
    beam: list[_BeamState] = [
        _BeamState(
            selected_bins=initial_selected,
            selected_uncertain_indexes=(),
            score=float(initial_score),
            trajectory_score=float(initial_score),
            line=line,
        )
    ]

    anchor_line = line
    for idx in uncertain_sorted:
        ev = evidences[idx]
        next_beam: list[_BeamState] = []
        candidates = _eligible_candidates(ev, anchor_line, cfg)
        for state in beam:
            for raw_bin in candidates:
                new_selected = list(state.selected_bins)
                new_selected[idx] = int(raw_bin)
                candidate_selected = tuple(new_selected)
                score_indexes = tuple(sorted(set(state.selected_uncertain_indexes + (idx,))))
                score, _line_score, _mean_phase, _mean_amp = _trajectory_score(
                    evidences,
                    candidate_selected,
                    score_indexes,
                    anchor_line,
                    cfg,
                )
                next_beam.append(
                    _BeamState(
                        selected_bins=candidate_selected,
                        selected_uncertain_indexes=score_indexes,
                        score=float(score),
                        trajectory_score=float(score),
                        line=anchor_line,
                    )
                )
        next_beam.sort(key=lambda item: item.score, reverse=True)
        beam = next_beam[: max(1, int(cfg.beam_width))]
        if not beam:
            break

    best = beam[0] if beam else _BeamState(initial_selected, (), 0.0, 0.0, line)
    final_line = _fit_phase_curve_for_bins(
        evidences,
        best.selected_bins,
        trim_frac=float(cfg.line_trim_frac),
        phase_model=str(cfg.phase_model),
    )
    if int(final_line.anchor_count) < 2:
        final_line = best.line
    score_indexes = uncertain if uncertain else list(range(len(evidences)))
    final_score, line_score, mean_phase, mean_amp = _trajectory_score(
        evidences,
        best.selected_bins,
        score_indexes,
        final_line,
        cfg,
    )

    return SymbolPhaseResult(
        selected_raw_bins=tuple(int(v) for v in best.selected_bins),
        locked_mask=tuple(bool(v) for v in locked),
        evidences=tuple(evidences),
        phase_line=final_line,
        trajectory_score=float(final_score),
        line_score=float(line_score),
        mean_phase_score=float(mean_phase),
        mean_amp_score=float(mean_amp),
        uncertain_count=int(len(uncertain)),
        locked_count=int(sum(locked)),
        beam_final_size=int(len(beam)),
        error="",
    )


__all__ = [
    "CandidateBin",
    "SymbolEvidence",
    "SymbolPhaseConfig",
    "SymbolPhaseResult",
    "build_symbol_evidences",
    "select_symbol_bins_two_stage",
]

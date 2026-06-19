"""Local circular phase trajectory helpers."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..candidate_pruning import wrap_phase
from ..phase_guided_demod import PhaseLine, fit_phase_line
from .configs import PhaseLineSelectorConfig


def unwrap_against(reference: float, phase: float) -> float:
    """Shift `phase` by 2*pi so it is closest to `reference`."""

    return float(float(reference) + wrap_phase(float(phase) - float(reference)))


def recent_weights(count: int, decay: float) -> np.ndarray:
    n = max(0, int(count))
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    d = float(decay)
    if d <= 0.0 or d >= 1.0:
        return np.ones(n, dtype=np.float64)
    return np.asarray([d ** float(n - 1 - idx) for idx in range(n)], dtype=np.float64)


def predict_from_history(
    abs_indices: Sequence[float],
    phases: Sequence[float],
    target_abs_index: float,
    config: PhaseLineSelectorConfig,
) -> tuple[bool, float, float, float]:
    """Predict local phase, slope, and curvature from recent selected points."""

    count = min(len(abs_indices), len(phases), max(1, int(config.window_size)))
    if count <= 0:
        return False, 0.0, 0.0, 0.0

    xs = np.asarray(list(abs_indices)[-count:], dtype=np.float64)
    ys = np.asarray(list(phases)[-count:], dtype=np.float64)
    target = float(target_abs_index)
    if count == 1:
        return True, float(ys[-1]), 0.0, 0.0

    degree = min(max(0, int(config.window_degree)), 2, count - 1)
    weights = recent_weights(count, config.recent_decay)
    if degree <= 0:
        return True, float(np.average(ys, weights=weights)), 0.0, 0.0

    x0 = float(xs[-1])
    x_rel = xs - x0
    target_rel = target - x0
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
    return True, pred, slope, curvature


def mix_with_header_reference(
    has_history: bool,
    predicted: float,
    header_line: PhaseLine | None,
    target_abs_index: float,
    config: PhaseLineSelectorConfig,
) -> tuple[bool, float]:
    """Use header line as an initial weak reference before payload history exists."""

    weight = max(0.0, min(1.0, float(config.header_reference_weight)))
    if weight <= 0.0:
        return bool(has_history), float(predicted)
    if header_line is None or int(header_line.anchor_count) < 2:
        return bool(has_history), float(predicted)
    header_pred = float(header_line.predict(float(target_abs_index)))
    if not has_history:
        return True, header_pred
    delta = float(wrap_phase(header_pred - float(predicted)))
    return True, float(predicted + weight * delta)


def fit_selected_phase_line(
    center_spectra: Sequence[np.ndarray],
    selected_bins: Sequence[int],
    abs_indices: Sequence[float],
    trim_frac: float,
) -> PhaseLine:
    xs: list[float] = []
    phases: list[float] = []
    count = min(len(center_spectra), len(selected_bins), len(abs_indices))
    for idx in range(count):
        spectrum = np.asarray(center_spectra[idx], dtype=np.complex64)
        raw_bin = int(selected_bins[idx])
        if raw_bin < 0 or raw_bin >= spectrum.size:
            continue
        xs.append(float(abs_indices[idx]))
        phases.append(float(np.angle(spectrum[raw_bin])))
    if len(xs) < 2:
        return PhaseLine()
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    x_arr = np.asarray(xs, dtype=np.float64)[order]
    phase_arr = np.unwrap(np.asarray(phases, dtype=np.float64)[order])
    return fit_phase_line(x_arr, phase_arr, trim_frac=float(trim_frac))


def smoothness_penalty(
    observed_phase: float,
    current_abs_index: float,
    previous_phase: float | None,
    previous_abs_index: float | None,
    previous_slope: float | None,
    predicted_slope: float,
    predicted_curvature: float,
    config: PhaseLineSelectorConfig,
) -> tuple[float, float | None]:
    """Return transition penalty and the local slope to carry in the beam."""

    slope = None
    penalty = 0.0
    slope_scale = config.slope_scale_rad()
    curvature_scale = config.curvature_scale_rad()
    if previous_phase is not None and previous_abs_index is not None:
        dx = max(1e-6, float(current_abs_index) - float(previous_abs_index))
        slope = float((float(observed_phase) - float(previous_phase)) / dx)
        penalty += float(config.slope_penalty_weight) * ((slope - float(predicted_slope)) / slope_scale) ** 2
        if previous_slope is not None:
            curvature = float(slope - float(previous_slope))
            penalty += float(config.curvature_penalty_weight) * ((curvature - float(predicted_curvature)) / curvature_scale) ** 2
    return float(penalty), slope


def phase_likelihood(residual_rad: float, config: PhaseLineSelectorConfig) -> float:
    scale = config.phase_scale_rad()
    return float(math.exp(-((float(residual_rad) / scale) ** 2)))

"""Lightweight phase-aware FFT-bin candidate pruning.

This module is deliberately narrower than the payload decoders: it only ranks
raw FFT bins for each payload symbol and computes recall-oriented helper
statistics.  It does not use payload templates, byte priors, CRC feedback, or
cross-packet information.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .phase_guided_demod import PhaseLine, fit_phase_line


def wrap_phase(value: np.ndarray | float) -> np.ndarray | float:
    """Wrap phase angle(s) to [-pi, pi]."""

    return np.angle(np.exp(1j * value))


def circular_mean(phases: Sequence[float]) -> float:
    """Circular mean of phase angles."""

    arr = np.asarray(list(phases), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    z = np.mean(np.exp(1j * arr))
    if abs(z) <= 1e-30:
        return 0.0
    return float(np.angle(z))


def circular_std(phases: Sequence[float]) -> float:
    """Circular standard deviation in radians."""

    arr = np.asarray(list(phases), dtype=np.float64)
    if arr.size == 0:
        return float("inf")
    r = float(abs(np.mean(np.exp(1j * arr))))
    r = max(1e-12, min(1.0, r))
    return float(math.sqrt(-2.0 * math.log(r)))


def rank_of(scores: np.ndarray, target_bin: int) -> int:
    """1-based rank of target_bin under descending scores, or 0 if invalid."""

    target = int(target_bin)
    if target < 0 or target >= scores.size:
        return 0
    order = np.argsort(scores)[::-1]
    hit = np.where(order == target)[0]
    return int(hit[0] + 1) if hit.size else 0


def top_bins(scores: np.ndarray, count: int) -> np.ndarray:
    """Return bins with the largest scores, sorted by score descending."""

    n = int(scores.size)
    k = max(0, min(int(count), n))
    if k == 0:
        return np.array([], dtype=np.int64)
    if k == n:
        return np.argsort(scores)[::-1].astype(np.int64)
    partial = np.argpartition(scores, n - k)[n - k :]
    return partial[np.argsort(scores[partial])[::-1]].astype(np.int64)


@dataclass(frozen=True)
class PhaseTrend:
    """Packet-local phase predictor for payload FFT bins."""

    line: PhaseLine
    residual_offset_rad: float = 0.0
    residual_std_rad: float = float("inf")
    early_anchor_count: int = 0
    source: str = "header"

    def predict(self, abs_symbol_index: float) -> float:
        return float(self.line.predict(abs_symbol_index) + self.residual_offset_rad)

    @property
    def quality(self) -> float:
        """Conservative [0, 1] quality gate for phase bonus."""

        if self.line.anchor_count < 2:
            return 0.0
        anchor_q = min(1.0, float(self.line.anchor_count) / 8.0)
        if math.isfinite(self.line.fit_r2):
            r2_q = max(0.0, min(1.0, (float(self.line.fit_r2) + 0.25) / 1.25))
        else:
            r2_q = 0.5
        if self.early_anchor_count > 0 and math.isfinite(self.residual_std_rad):
            std_q = max(0.0, min(1.0, 1.0 - self.residual_std_rad / math.pi))
            early_q = min(1.0, float(self.early_anchor_count) / 4.0)
            return float(anchor_q * r2_q * (0.35 + 0.65 * early_q * std_q))
        return float(0.35 * anchor_q * r2_q)


@dataclass(frozen=True)
class PhaseGatedMetricConfig:
    """Parameters for phase-gated candidate ranking."""

    phase_bonus_weight: float = 0.05
    phase_gate_width_rad: float = math.pi / 2.0
    energy_preselect_count: int = 128
    energy_preselect_factor: int = 4
    noise_floor_percentile: float = 50.0
    rel_db_floor: float = -30.0
    amp_gate_floor_db: float = -18.0
    amp_gate_temp_db: float = 4.0


def score_energy(power: np.ndarray, config: PhaseGatedMetricConfig) -> np.ndarray:
    """Robust monotonic energy score for all bins."""

    pwr = np.asarray(power, dtype=np.float64)
    floor = float(np.percentile(pwr, float(config.noise_floor_percentile)))
    floor = max(floor, 1e-30)
    return np.log1p(pwr / floor)


def estimate_payload_residual_offset(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    phase_line: PhaseLine,
    max_symbols: int = 8,
    min_margin_db: float = 3.0,
) -> tuple[float, float, int]:
    """Estimate packet-local phase residual offset from early reliable symbols.

    This uses only low-SNR packet-local observations.  It does not use GT bins:
    early argmax bins are accepted as anchors only when their energy margin is
    large enough.
    """

    residuals: list[float] = []
    limit = min(len(center_spectra), len(evidence_powers), len(abs_indices), int(max_symbols))
    for idx in range(max(0, limit)):
        center = np.asarray(center_spectra[idx], dtype=np.complex64)
        power = np.asarray(evidence_powers[idx], dtype=np.float64)
        if center.size == 0 or power.size < 2:
            continue
        order = np.argsort(power)[::-1]
        best = int(order[0])
        second = int(order[1])
        margin_db = 10.0 * math.log10(float(power[best] + 1e-30) / float(power[second] + 1e-30))
        if margin_db < float(min_margin_db):
            continue
        phase = float(np.angle(center[best]))
        predicted = float(phase_line.predict(float(abs_indices[idx])))
        residuals.append(float(wrap_phase(phase - predicted)))

    if not residuals:
        return 0.0, float("inf"), 0
    return circular_mean(residuals), circular_std(residuals), len(residuals)


def fit_early_payload_phase_trend(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    fallback_line: PhaseLine,
    max_symbols: int = 8,
    min_margin_db: float = 0.0,
    trim_frac: float = 0.25,
) -> PhaseTrend:
    """Fit a payload-local phase trend from early energy-selected bins.

    The selected bins are not GT labels.  They are early argmax bins accepted
    only by an energy-margin check, so this remains an inference-side metric.
    """

    anchors_x: list[float] = []
    anchors_phase: list[float] = []
    limit = min(len(center_spectra), len(evidence_powers), len(abs_indices), int(max_symbols))
    for idx in range(max(0, limit)):
        center = np.asarray(center_spectra[idx], dtype=np.complex64)
        power = np.asarray(evidence_powers[idx], dtype=np.float64)
        if center.size == 0 or power.size < 2:
            continue
        order = np.argsort(power)[::-1]
        best = int(order[0])
        second = int(order[1])
        margin_db = 10.0 * math.log10(float(power[best] + 1e-30) / float(power[second] + 1e-30))
        if margin_db < float(min_margin_db):
            continue
        anchors_x.append(float(abs_indices[idx]))
        anchors_phase.append(float(np.angle(center[best])))

    if len(anchors_x) < 2:
        offset, offset_std, offset_count = estimate_payload_residual_offset(
            center_spectra=center_spectra,
            evidence_powers=evidence_powers,
            abs_indices=abs_indices,
            phase_line=fallback_line,
            max_symbols=max_symbols,
            min_margin_db=min_margin_db,
        )
        return PhaseTrend(
            line=fallback_line,
            residual_offset_rad=offset,
            residual_std_rad=offset_std,
            early_anchor_count=offset_count,
            source="header-offset",
        )

    order = np.argsort(np.asarray(anchors_x, dtype=np.float64))
    x = np.asarray(anchors_x, dtype=np.float64)[order]
    phases = np.unwrap(np.asarray(anchors_phase, dtype=np.float64)[order])
    line = fit_phase_line(x, phases, trim_frac=float(trim_frac))
    residuals = [float(wrap_phase(phase - line.predict(abs_idx))) for abs_idx, phase in zip(x, phases)]
    return PhaseTrend(
        line=line,
        residual_offset_rad=0.0,
        residual_std_rad=circular_std(residuals),
        early_anchor_count=len(anchors_x),
        source="early-payload",
    )


def phase_gated_scores(
    center_spectrum: np.ndarray,
    evidence_power: np.ndarray,
    predicted_phase_rad: float,
    phase_trend_quality: float,
    max_top_l: int,
    config: PhaseGatedMetricConfig,
) -> np.ndarray:
    """Score bins with energy plus a conservative phase bonus.

    Phase is evaluated only on an energy preselection set.  All other bins keep
    their energy-only score.
    """

    center = np.asarray(center_spectrum, dtype=np.complex64)
    power = np.asarray(evidence_power, dtype=np.float64)
    if center.size != power.size:
        raise ValueError("center_spectrum and evidence_power must have the same size")

    scores = score_energy(power, config)
    if center.size == 0 or float(config.phase_bonus_weight) <= 0.0:
        return scores

    if int(config.energy_preselect_count) <= 0 and int(config.energy_preselect_factor) <= 0:
        candidates = np.arange(power.size, dtype=np.int64)
    else:
        preselect = max(
            int(config.energy_preselect_count),
            int(config.energy_preselect_factor) * max(1, int(max_top_l)),
        )
        candidates = top_bins(power, min(preselect, power.size))
    if candidates.size == 0:
        return scores

    max_power = float(np.max(power)) if power.size else 0.0
    rel_db = 10.0 * np.log10((power[candidates] + 1e-30) / (max_power + 1e-30))
    amp_gate = 1.0 / (
        1.0
        + np.exp(
            -(
                rel_db - float(config.amp_gate_floor_db)
            )
            / max(1e-6, float(config.amp_gate_temp_db))
        )
    )

    residual = np.angle(
        np.exp(1j * (np.angle(center[candidates]) - float(predicted_phase_rad)))
    )
    gate_width = max(1e-6, min(math.pi, float(config.phase_gate_width_rad)))
    threshold = math.cos(gate_width)
    phase_cos = np.cos(residual)
    phase_bonus = np.maximum(0.0, phase_cos - threshold) / max(1e-12, 1.0 - threshold)

    rel_floor = float(config.rel_db_floor)
    phase_bonus = np.where(rel_db >= rel_floor, phase_bonus, 0.0)
    bonus = (
        float(config.phase_bonus_weight)
        * max(0.0, min(1.0, float(phase_trend_quality)))
        * amp_gate
        * phase_bonus
    )
    out = scores.copy()
    out[candidates] += bonus
    return out


def phase_rescue_scores(
    energy_scores: np.ndarray,
    phase_scores: np.ndarray,
    top_l: int,
    rescue_count: int,
) -> np.ndarray:
    """Build an exact Top-L rescue ranking as scores.

    The selected set is energy top (L-rescue_count) plus phase-gated top
    rescue_count from outside that kept energy core.  Returned scores only need
    to preserve that Top-L ordering for recall/rank evaluation.
    """

    n = int(energy_scores.size)
    keep = max(0, min(int(top_l), int(top_l) - int(rescue_count)))
    rescue = max(0, min(int(rescue_count), int(top_l) - keep))
    selected: list[int] = []
    for b in top_bins(energy_scores, keep):
        selected.append(int(b))
    selected_set = set(selected)
    for b in top_bins(phase_scores, n):
        ib = int(b)
        if ib in selected_set:
            continue
        selected.append(ib)
        selected_set.add(ib)
        if len(selected) >= int(top_l):
            break

    scores = np.full(n, -np.inf, dtype=np.float64)
    for rank, bin_index in enumerate(selected):
        scores[int(bin_index)] = float(int(top_l) - rank)
    return scores


__all__ = [
    "PhaseGatedMetricConfig",
    "PhaseTrend",
    "circular_mean",
    "circular_std",
    "estimate_payload_residual_offset",
    "fit_early_payload_phase_trend",
    "phase_gated_scores",
    "phase_rescue_scores",
    "rank_of",
    "score_energy",
    "top_bins",
    "wrap_phase",
]

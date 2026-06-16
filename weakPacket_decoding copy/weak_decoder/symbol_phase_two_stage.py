"""Symbol-level two-stage phase-aware FFT-bin selection.

This module deliberately stays at the raw FFT-bin level.  It does not enumerate
payload bytes, does not use application templates, and does not combine packets.
The intended flow is:

1. Use oversampled / multi-offset evidence to lock high-confidence symbol bins.
2. Keep only Top-L candidates for low-confidence symbols.
3. Fit a packet-local phase line from locked bins.
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
class SymbolPhaseConfig:
    """Tunable parameters for symbol-level two-stage peak selection."""

    top_l_low_confidence: int = 8
    lock_margin_db: float = 1.5
    lock_peak_to_median_db: float = 5.0
    lock_phase_score: float = 0.35
    min_locked_for_line: int = 4
    line_trim_frac: float = 0.25
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
    confidence: float
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
    phase_line: PhaseLine
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
    line: PhaseLine


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


def _phase_score(center_spectrum: np.ndarray, raw_bin: int, line: PhaseLine, abs_index: float) -> float:
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


def _local_candidate_score(
    ev: SymbolEvidence,
    raw_bin: int,
    line: PhaseLine,
    config: SymbolPhaseConfig,
) -> tuple[float, float, float]:
    phase = (
        _phase_score(ev.center_spectrum, int(raw_bin), line, ev.abs_symbol_index)
        if int(line.anchor_count) >= 2
        else 0.0
    )
    amp = _amp_score(ev.evidence_power, int(raw_bin))
    score = (
        float(config.phase_weight) * phase
        + float(config.amp_weight) * amp
        + float(config.profile_weight) * 0.0
    )
    return float(score), float(phase), float(amp)


def _eligible_candidates(
    ev: SymbolEvidence,
    line: PhaseLine,
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


def _trajectory_score(
    evidences: Sequence[SymbolEvidence],
    selected_bins: Sequence[int],
    score_indexes: Sequence[int],
    line: PhaseLine,
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
        bins = top_bins(power, min(max(top_l, 2), power.size))
        top1 = int(bins[0])
        top2 = int(bins[1]) if bins.size > 1 else top1
        top1_power = float(power[top1])
        top2_power = float(power[top2])
        median_power = float(np.median(power)) if power.size else 0.0
        margin_db = _db_ratio(top1_power, top2_power)
        peak_to_median_db = _db_ratio(top1_power, median_power)
        amp = _amp_score(power, top1)
        confidence = float(margin_db + 0.20 * peak_to_median_db)
        locked = (
            margin_db >= float(config.lock_margin_db)
            and peak_to_median_db >= float(config.lock_peak_to_median_db)
        )
        out.append(
            SymbolEvidence(
                symbol_index=int(idx),
                abs_symbol_index=float(abs_indices[idx]),
                center_spectrum=center,
                evidence_power=power,
                top_bins=tuple(int(v) for v in bins[:top_l]),
                top_scores=tuple(float(power[int(v)]) for v in bins[:top_l]),
                top1_bin=top1,
                top1_margin_db=float(margin_db),
                top1_peak_to_median_db=float(peak_to_median_db),
                top1_amp_score=float(amp),
                confidence=confidence,
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
) -> SymbolPhaseResult:
    """Select one raw FFT bin per payload symbol with symbol-level phase search."""

    cfg = config or SymbolPhaseConfig()
    evidences = list(build_symbol_evidences(center_spectra, evidence_powers, abs_indices, cfg))
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

    line = _fit_line_for_bins(
        [ev for ev, is_locked in zip(evidences, locked) if is_locked],
        [bin_value for bin_value, is_locked in zip(selected, locked) if is_locked],
        trim_frac=float(cfg.line_trim_frac),
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
            line = _fit_line_for_bins(
                [ev for ev, is_locked in zip(evidences, locked) if is_locked],
                [bin_value for bin_value, is_locked in zip(selected, locked) if is_locked],
                trim_frac=float(cfg.line_trim_frac),
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
                confidence=ev.confidence,
                locked=bool(is_locked),
                phase_score_to_line=float(phase_score),
            )
        )
    evidences = refined_evidences

    uncertain = [idx for idx, is_locked in enumerate(locked) if not is_locked]
    uncertain_sorted = sorted(uncertain, key=lambda idx: evidences[idx].confidence, reverse=True)

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
    final_line = _fit_line_for_bins(evidences, best.selected_bins, trim_frac=float(cfg.line_trim_frac))
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

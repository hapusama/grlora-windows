"""Phase-line dominated candidate path selector."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from ..candidate_pruning import top_bins, wrap_phase
from ..phase_guided_demod import PhaseLine, fit_phase_line
from ..symbol_phase_two_stage import (
    SymbolEvidence,
    SymbolPhaseConfig,
    SymbolPhaseResult,
    build_symbol_evidences,
    select_symbol_bins_two_stage,
)
from .configs import PhaseLineSelectorConfig, PhasePathSelectorConfig
from .trajectory import (
    fit_selected_phase_line,
    mix_with_header_reference,
    phase_likelihood,
    predict_from_history,
    recent_weights,
    smoothness_penalty,
    unwrap_against,
)


@dataclass(frozen=True)
class _PathState:
    selected_bins: tuple[int, ...]
    phases: tuple[float, ...]
    score: float
    last_slope: float | None = None


@dataclass(frozen=True)
class _ViterbiCandidate:
    raw_bin: int
    phase: float
    local_score: float
    energy_score: float
    coherence_score: float
    rank_score: float
    phase_local_score: float = 0.0


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(10.0 * math.log10((float(numerator) + 1e-30) / (float(denominator) + 1e-30)))


def _energy_score(ev: SymbolEvidence, raw_bin: int) -> float:
    b = int(raw_bin)
    if b < 0 or b >= ev.evidence_power.size:
        return 0.0
    max_power = float(np.max(ev.evidence_power)) if ev.evidence_power.size else 0.0
    if max_power <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, ev.evidence_power[b] / (max_power + 1e-30))))


def _coherence_score(ev: SymbolEvidence, raw_bin: int) -> float:
    if ev.offset_coherence is None:
        return 0.0
    b = int(raw_bin)
    if b < 0 or b >= ev.offset_coherence.size:
        return 0.0
    return float(max(0.0, min(1.0, ev.offset_coherence[b])))


def _is_high_confidence(ev: SymbolEvidence, config: PhasePathSelectorConfig) -> bool:
    coherence_ok = (
        ev.offset_coherence is None
        or float(config.high_confidence_min_coherence) <= 0.0
        or float(ev.top1_coherence_score) >= float(config.high_confidence_min_coherence)
    )
    return bool(
        float(ev.top1_margin_db) >= float(config.high_confidence_margin_db)
        and float(ev.top1_peak_to_median_db) >= float(config.high_confidence_peak_to_median_db)
        and coherence_ok
    )


def _is_hard_anchor(ev: SymbolEvidence, config: PhasePathSelectorConfig) -> bool:
    coherence_ok = (
        ev.offset_coherence is None
        or float(config.hard_anchor_min_coherence) <= 0.0
        or float(ev.top1_coherence_score) >= float(config.hard_anchor_min_coherence)
    )
    return bool(
        int(config.hard_anchor_top_k) > 0
        and float(ev.top1_margin_db) >= float(config.hard_anchor_margin_db)
        and float(ev.top1_peak_to_median_db) >= float(config.hard_anchor_peak_to_median_db)
        and coherence_ok
    )


def _confidence_blend(ev: SymbolEvidence, config: PhasePathSelectorConfig) -> float:
    margin_low = float(getattr(config, "adaptive_local_score_margin_low_db", 0.40))
    margin_high = max(margin_low + 1e-6, float(getattr(config, "adaptive_local_score_margin_high_db", 2.50)))
    peak_low = float(getattr(config, "adaptive_local_score_peak_low_db", 7.0))
    peak_high = max(peak_low + 1e-6, float(getattr(config, "adaptive_local_score_peak_high_db", 12.0)))
    margin_blend = (float(ev.top1_margin_db) - margin_low) / (margin_high - margin_low)
    peak_blend = (float(ev.top1_peak_to_median_db) - peak_low) / (peak_high - peak_low)
    value = min(margin_blend, peak_blend)
    return float(max(0.0, min(1.0, value)))


def _local_score_weights(ev: SymbolEvidence, config: PhasePathSelectorConfig) -> tuple[float, float, float, float]:
    high = (
        float(config.energy_weight),
        float(config.coherence_weight),
        float(getattr(config, "phase_local_weight", 0.0)),
        float(config.rank_weight),
    )
    if not bool(getattr(config, "adaptive_local_score_enabled", False)):
        return high
    low = (
        float(getattr(config, "adaptive_low_conf_energy_weight", high[0])),
        float(getattr(config, "adaptive_low_conf_coherence_weight", high[1])),
        float(getattr(config, "adaptive_low_conf_phase_local_weight", high[2])),
        float(getattr(config, "adaptive_low_conf_rank_weight", high[3])),
    )
    blend = _confidence_blend(ev, config)
    return tuple(float((1.0 - blend) * low[idx] + blend * high[idx]) for idx in range(4))  # type: ignore[return-value]


def _candidate_local_score(
    ev: SymbolEvidence,
    cand: _ViterbiCandidate,
    config: PhasePathSelectorConfig,
    *,
    high_confidence: bool,
    hard_anchor: bool,
) -> float:
    energy_w, coherence_w, phase_w, rank_w = _local_score_weights(ev, config)
    bonus = 0.0
    if int(cand.raw_bin) == int(ev.top1_bin) and (bool(high_confidence) or bool(hard_anchor)):
        bonus = float(config.top1_soft_bonus)
    return float(
        energy_w * float(cand.energy_score)
        + coherence_w * float(cand.coherence_score)
        + phase_w * float(cand.phase_local_score)
        + rank_w * float(cand.rank_score)
        + bonus
        - _noise_peak_penalty(ev, cand, config)
    )


def _huber_normalized(value: float, scale: float, delta: float) -> float:
    x = abs(float(value)) / max(1e-12, float(scale))
    d = max(1e-12, float(delta))
    if x <= d:
        return float(0.5 * x * x)
    return float(d * (x - 0.5 * d))


def _phase_proposal_prediction(
    evidences: Sequence[SymbolEvidence],
    anchor_mask: Sequence[bool],
    target_index: int,
    config: PhaseLineSelectorConfig | PhasePathSelectorConfig,
) -> tuple[bool, float]:
    idx_target = int(target_index)
    if idx_target < 0 or idx_target >= len(evidences):
        return False, 0.0
    target_abs = float(evidences[idx_target].abs_symbol_index)
    span = max(1e-6, float(getattr(config, "phase_proposal_anchor_span_symbols", 12.0)))
    anchors: list[tuple[float, float, float]] = []
    for idx, ev in enumerate(evidences):
        if idx >= len(anchor_mask) or not bool(anchor_mask[idx]):
            continue
        raw_bin = int(ev.top1_bin)
        if raw_bin < 0 or raw_bin >= ev.center_spectrum.size:
            continue
        distance = abs(float(ev.abs_symbol_index) - target_abs)
        if distance > span:
            continue
        margin_boost = max(0.0, min(2.0, float(ev.top1_margin_db) / 8.0))
        distance_weight = math.exp(-0.5 * (distance / max(1e-6, 0.5 * span)) ** 2)
        anchors.append(
            (
                float(ev.abs_symbol_index),
                float(np.angle(ev.center_spectrum[raw_bin])),
                float(distance_weight * (1.0 + 0.25 * margin_boost)),
            )
        )
    if len(anchors) < max(1, int(getattr(config, "phase_proposal_anchor_min_count", 2))):
        return False, 0.0
    anchors.sort(key=lambda item: item[0])
    xs = np.asarray([item[0] for item in anchors], dtype=np.float64)
    phases = np.unwrap(np.asarray([item[1] for item in anchors], dtype=np.float64))
    weights = np.asarray([item[2] for item in anchors], dtype=np.float64)
    degree = min(max(0, int(getattr(config, "window_degree", 1))), 2, len(anchors) - 1)
    if degree <= 0:
        pred = float(np.average(phases, weights=weights))
        return True, pred
    x_rel = xs - target_abs
    try:
        coef = np.polyfit(x_rel, phases, deg=degree, w=weights)
    except np.linalg.LinAlgError:
        return False, 0.0
    fitted = np.polyval(coef, x_rel)
    rmse_pi = float(math.sqrt(float(np.average((phases - fitted) ** 2, weights=weights))) / math.pi)
    if (
        math.isfinite(rmse_pi)
        and rmse_pi > float(getattr(config, "phase_proposal_anchor_max_rmse_pi", 0.45))
    ):
        return False, 0.0
    return True, float(np.polyval(coef, 0.0))


def _consensus_phase_line(
    evidences: Sequence[SymbolEvidence],
    config: PhaseLineSelectorConfig | PhasePathSelectorConfig,
) -> tuple[PhaseLine, float]:
    if not bool(getattr(config, "phase_proposal_consensus_enabled", False)) or not evidences:
        return PhaseLine(), 0.0

    xs: list[float] = []
    phase_rows: list[np.ndarray] = []
    amp_rows: list[np.ndarray] = []
    candidate_top_l = max(1, int(getattr(config, "phase_proposal_consensus_top_l", 12)))
    for ev in evidences:
        bins = [int(v) for v in ev.top_bins[:candidate_top_l]]
        if not bins:
            continue
        phases: list[float] = []
        amps: list[float] = []
        max_power = float(np.max(ev.evidence_power)) if ev.evidence_power.size else 0.0
        for b in bins:
            if b < 0 or b >= ev.center_spectrum.size or b >= ev.evidence_power.size:
                continue
            phases.append(float(np.angle(ev.center_spectrum[b])))
            amps.append(float(max(0.0, min(1.0, ev.evidence_power[b] / (max_power + 1e-30)))) if max_power > 0.0 else 0.0)
        if not phases:
            continue
        xs.append(float(ev.abs_symbol_index))
        phase_rows.append(np.asarray(phases, dtype=np.float64))
        amp_rows.append(np.asarray(amps, dtype=np.float64))
    if len(xs) < 3:
        return PhaseLine(), 0.0

    slope_steps = max(3, int(getattr(config, "phase_proposal_consensus_slope_steps", 25)))
    intercept_steps = max(8, int(getattr(config, "phase_proposal_consensus_intercept_steps", 48)))
    slopes = np.linspace(-math.pi, math.pi, slope_steps, endpoint=False, dtype=np.float64)
    intercepts = np.linspace(-math.pi, math.pi, intercept_steps, endpoint=False, dtype=np.float64)
    phase_w = max(0.0, min(1.0, float(getattr(config, "phase_proposal_consensus_phase_weight", 0.70))))

    best_score = -float("inf")
    best_slope = 0.0
    best_intercept = 0.0
    for slope in slopes:
        totals = np.zeros_like(intercepts)
        for abs_idx, phases, amps in zip(xs, phase_rows, amp_rows):
            pred = slope * float(abs_idx) + intercepts[:, np.newaxis]
            residual = np.angle(np.exp(1j * (phases[np.newaxis, :] - pred)))
            scores = phase_w * np.cos(residual) + (1.0 - phase_w) * amps[np.newaxis, :]
            totals += np.max(scores, axis=1)
        pos = int(np.argmax(totals))
        score = float(totals[pos] / max(1, len(xs)))
        if score > best_score:
            best_score = score
            best_slope = float(slope)
            best_intercept = float(intercepts[pos])

    if best_score < float(getattr(config, "phase_proposal_consensus_min_score", 0.20)):
        return PhaseLine(), float(best_score)

    sel_x: list[float] = []
    sel_phase: list[float] = []
    for abs_idx, phases, amps in zip(xs, phase_rows, amp_rows):
        pred = best_slope * float(abs_idx) + best_intercept
        residual = np.angle(np.exp(1j * (phases - pred)))
        scores = phase_w * np.cos(residual) + (1.0 - phase_w) * amps
        pos = int(np.argmax(scores))
        sel_x.append(float(abs_idx))
        sel_phase.append(float(pred + float(residual[pos])))
    try:
        line = fit_phase_line(
            np.asarray(sel_x, dtype=np.float64),
            np.unwrap(np.asarray(sel_phase, dtype=np.float64)),
            trim_frac=min(0.25, float(getattr(config, "line_trim_frac", 0.20))),
        )
    except (ValueError, np.linalg.LinAlgError):
        line = PhaseLine(slope_rad=best_slope, intercept_rad=best_intercept, anchor_count=len(sel_x))
    return line, float(best_score)


def _phase_augmented_bins(
    ev: SymbolEvidence,
    predicted_phase: float,
    config: PhaseLineSelectorConfig | PhasePathSelectorConfig,
) -> tuple[int, ...]:
    top_l = max(1, int(getattr(config, "top_l", 24)))
    power = np.asarray(ev.evidence_power, dtype=np.float64)
    center = np.asarray(ev.center_spectrum, dtype=np.complex64)
    if power.size == 0 or center.size != power.size:
        return tuple(int(v) for v in ev.top_bins[:top_l])

    rescue_count = max(0, min(int(getattr(config, "phase_proposal_rescue_count", 0)), top_l - 1))
    if rescue_count <= 0:
        return tuple(int(v) for v in ev.top_bins[:top_l])

    core_count = max(1, top_l - rescue_count)
    selected: list[int] = []
    seen: set[int] = set()
    for raw_bin in top_bins(power, min(core_count, power.size)):
        b = int(raw_bin)
        if b not in seen:
            selected.append(b)
            seen.add(b)

    preselect = max(
        int(getattr(config, "phase_proposal_energy_preselect_count", 128)),
        int(getattr(config, "phase_proposal_energy_preselect_factor", 6)) * top_l,
    )
    candidates = top_bins(power, min(max(top_l, preselect), power.size))
    top1_power = float(power[int(ev.top1_bin)]) if 0 <= int(ev.top1_bin) < power.size else float(np.max(power))
    max_drop = float(getattr(config, "phase_proposal_max_energy_drop_db", 24.0))
    min_phase = float(getattr(config, "phase_proposal_min_phase_score", 0.25))
    phase_scale = max(1e-6, float(getattr(config, "phase_proposal_phase_scale_pi", 0.32)) * math.pi)
    w_phase = max(0.0, float(getattr(config, "phase_proposal_phase_weight", 0.78)))
    w_energy = max(0.0, float(getattr(config, "phase_proposal_energy_weight", 0.18)))
    w_coh = max(0.0, float(getattr(config, "phase_proposal_coherence_weight", 0.04)))
    total = max(1e-12, w_phase + w_energy + w_coh)
    w_phase, w_energy, w_coh = w_phase / total, w_energy / total, w_coh / total

    scored: list[tuple[float, int]] = []
    max_power = float(np.max(power)) if power.size else 0.0
    for raw_bin in candidates:
        b = int(raw_bin)
        if b in seen or b < 0 or b >= power.size:
            continue
        drop_db = _db_ratio(float(power[b]), top1_power)
        if drop_db < -max_drop:
            continue
        residual = float(wrap_phase(float(np.angle(center[b])) - float(predicted_phase)))
        phase_score = float(math.exp(-((residual / phase_scale) ** 2)))
        if phase_score < min_phase:
            continue
        energy_score = float(max(0.0, min(1.0, float(power[b]) / (max_power + 1e-30)))) if max_power > 0.0 else 0.0
        coherence_score = _coherence_score(ev, b)
        score = w_phase * phase_score + w_energy * energy_score + w_coh * coherence_score
        scored.append((float(score), b))

    for _score, b in sorted(scored, key=lambda item: item[0], reverse=True):
        if b in seen:
            continue
        selected.append(b)
        seen.add(b)
        if len(selected) >= top_l:
            break
    if len(selected) < top_l:
        for raw_bin in top_bins(power, min(top_l, power.size)):
            b = int(raw_bin)
            if b in seen:
                continue
            selected.append(b)
            seen.add(b)
            if len(selected) >= top_l:
                break
    return tuple(int(v) for v in selected[:top_l])


def _coherence_augmented_bins(
    ev: SymbolEvidence,
    config: PhaseLineSelectorConfig | PhasePathSelectorConfig,
) -> tuple[int, ...]:
    top_l = max(1, int(getattr(config, "top_l", 24)))
    power = np.asarray(ev.evidence_power, dtype=np.float64)
    coherence = ev.offset_coherence
    if power.size == 0 or coherence is None or coherence.size != power.size:
        return tuple(int(v) for v in ev.top_bins[:top_l])

    rescue_count = max(0, min(int(getattr(config, "phase_proposal_coherence_rescue_count", 0)), top_l - 1))
    if rescue_count <= 0:
        return tuple(int(v) for v in ev.top_bins[:top_l])

    protected_count = max(1, top_l - rescue_count)
    selected: list[int] = []
    seen: set[int] = set()
    for raw_bin in top_bins(power, min(top_l, power.size)):
        b = int(raw_bin)
        if b not in seen:
            selected.append(b)
            seen.add(b)
    if len(selected) < top_l:
        return tuple(int(v) for v in selected[:top_l])

    preselect = max(top_l, rescue_count * max(1, int(getattr(config, "phase_proposal_coherence_preselect_factor", 4))))
    min_gain = float(getattr(config, "phase_proposal_coherence_min_gain", 0.15))
    for raw_bin in top_bins(coherence, min(preselect, coherence.size)):
        b = int(raw_bin)
        if b in seen:
            continue
        tail_indexes = list(range(min(protected_count, len(selected)), len(selected)))
        if not tail_indexes:
            break
        replace_pos = min(tail_indexes, key=lambda pos: float(coherence[int(selected[pos])]))
        if float(coherence[b]) <= float(coherence[int(selected[replace_pos])]) + min_gain:
            continue
        seen.remove(int(selected[replace_pos]))
        selected[replace_pos] = b
        seen.add(b)
    return tuple(int(v) for v in selected[:top_l])


def _augment_evidences_with_phase_proposals(
    evidences: Sequence[SymbolEvidence],
    anchor_mask: Sequence[bool],
    config: PhaseLineSelectorConfig | PhasePathSelectorConfig,
) -> tuple[SymbolEvidence, ...]:
    if not bool(getattr(config, "phase_proposal_enabled", False)):
        return tuple(evidences)
    out: list[SymbolEvidence] = []
    top_l = max(1, int(getattr(config, "top_l", 24)))
    consensus_line, _consensus_score = _consensus_phase_line(evidences, config)
    has_consensus = bool(int(consensus_line.anchor_count) >= 2)
    for idx, ev in enumerate(evidences):
        if idx < len(anchor_mask) and bool(anchor_mask[idx]):
            bins = tuple(int(v) for v in ev.top_bins[:top_l])
        else:
            has_pred, predicted = _phase_proposal_prediction(evidences, anchor_mask, idx, config)
            if not has_pred and has_consensus:
                has_pred = True
                predicted = float(consensus_line.predict(ev.abs_symbol_index))
            bins = (
                _phase_augmented_bins(ev, predicted, config)
                if has_pred
                else _coherence_augmented_bins(ev, config)
            )
        if tuple(int(v) for v in ev.top_bins[:top_l]) == bins:
            out.append(ev)
            continue
        out.append(
            replace(
                ev,
                top_bins=bins,
                top_scores=tuple(
                    float(ev.evidence_power[int(v)])
                    for v in bins
                    if 0 <= int(v) < ev.evidence_power.size
                ),
            )
        )
    return tuple(out)


def _candidate_drop_db(ev: SymbolEvidence, raw_bin: int) -> float:
    b = int(raw_bin)
    if b < 0 or b >= ev.evidence_power.size:
        return -float("inf")
    top = float(ev.evidence_power[int(ev.top1_bin)]) if 0 <= int(ev.top1_bin) < ev.evidence_power.size else 0.0
    return _db_ratio(float(ev.evidence_power[b]), top)


def _candidate_energy_gap_db(ev: SymbolEvidence, raw_bin: int) -> float:
    b = int(raw_bin)
    if b < 0 or b >= ev.evidence_power.size:
        return 0.0
    candidate_power = float(ev.evidence_power[b])
    if candidate_power <= 0.0:
        return 0.0
    competitor = 0.0
    for raw_other in ev.top_bins:
        other = int(raw_other)
        if other == b or other < 0 or other >= ev.evidence_power.size:
            continue
        competitor = max(competitor, float(ev.evidence_power[other]))
    if competitor <= 0.0:
        return float("inf")
    return _db_ratio(candidate_power, competitor)


def _noise_peak_penalty(ev: SymbolEvidence, cand: _ViterbiCandidate, config: PhasePathSelectorConfig) -> float:
    if not bool(getattr(config, "noise_peak_penalty_enabled", False)):
        return 0.0
    energy = float(cand.energy_score)
    min_energy = float(getattr(config, "noise_peak_penalty_min_energy", 0.80))
    if energy <= min_energy:
        return 0.0
    energy_excess = (energy - min_energy) / max(1e-6, 1.0 - min_energy)
    coherence_ref = float(getattr(config, "noise_peak_penalty_coherence_ref", 0.92))
    phase_ref = float(getattr(config, "noise_peak_penalty_phase_ref", 0.55))
    gap_ref = float(getattr(config, "noise_peak_penalty_gap_ref_db", 1.50))
    coherence_deficit = max(0.0, coherence_ref - float(cand.coherence_score)) / max(1e-6, coherence_ref)
    phase_deficit = max(0.0, phase_ref - float(cand.phase_local_score)) / max(1e-6, phase_ref)
    gap_db = _candidate_energy_gap_db(ev, int(cand.raw_bin))
    gap_deficit = max(0.0, gap_ref - gap_db) / max(1e-6, gap_ref)
    suspicion = max(coherence_deficit, 0.5 * coherence_deficit + 0.3 * phase_deficit + 0.2 * gap_deficit)
    return float(getattr(config, "noise_peak_penalty_weight", 0.0)) * float(energy_excess) * float(suspicion)


def _viterbi_candidates(ev: SymbolEvidence, config: PhasePathSelectorConfig) -> tuple[_ViterbiCandidate, ...]:
    top_l = max(1, int(config.top_l))
    bins = [int(v) for v in ev.top_bins[:top_l]]
    if not bins:
        bins = [int(ev.top1_bin)]
    hard_anchor = _is_hard_anchor(ev, config)
    if hard_anchor:
        keep = max(1, int(config.hard_anchor_top_k))
        soft_keep = max(1, int(getattr(config, "hard_anchor_soft_top_k", 1)))
        if (
            soft_keep > keep
            and float(ev.top1_margin_db) <= float(getattr(config, "hard_anchor_soft_max_margin_db", 0.0))
            and float(ev.top1_peak_to_median_db) <= float(
                getattr(config, "hard_anchor_soft_max_peak_to_median_db", 0.0)
            )
        ):
            keep = soft_keep
        bins = bins[:keep]
    elif _is_high_confidence(ev, config) and int(config.high_confidence_top_k) > 0:
        keep = max(1, int(config.high_confidence_top_k))
        bins = bins[:keep]
    elif (
        bool(getattr(config, "phase_proposal_keep_energy_backup", False))
        and (int(config.phase_order) < 2 or float(config.second_order_weight) <= 0.0)
    ):
        seen_backup = {int(v) for v in bins}
        for raw_bin in top_bins(np.asarray(ev.evidence_power, dtype=np.float64), min(top_l, ev.evidence_power.size)):
            b = int(raw_bin)
            if b in seen_backup:
                continue
            bins.append(b)
            seen_backup.add(b)
        extra_count = max(0, int(getattr(config, "extra_coherence_candidates", 0)))
        coherence = ev.offset_coherence
        if extra_count > 0 and coherence is not None and coherence.size == ev.evidence_power.size:
            preselect = max(
                top_l,
                extra_count * max(1, int(getattr(config, "extra_coherence_preselect_factor", 4))),
            )
            min_gain = float(getattr(config, "extra_coherence_min_gain", 0.05))
            tail_coherence = min((_coherence_score(ev, b) for b in seen_backup), default=0.0)
            added = 0
            for raw_bin in top_bins(coherence, min(preselect, coherence.size)):
                b = int(raw_bin)
                if b in seen_backup:
                    continue
                if _coherence_score(ev, b) <= tail_coherence + min_gain:
                    continue
                bins.append(b)
                seen_backup.add(b)
                added += 1
                if added >= extra_count:
                    break

    out: list[_ViterbiCandidate] = []
    seen: set[int] = set()
    rank_den = max(1, len(bins) - 1)
    for rank, raw_bin in enumerate(bins):
        b = int(raw_bin)
        if b in seen or b < 0 or b >= ev.center_spectrum.size:
            continue
        seen.add(b)
        if b != int(ev.top1_bin) and _candidate_drop_db(ev, b) < -float(config.max_energy_drop_db):
            continue
        energy = _energy_score(ev, b)
        coherence = _coherence_score(ev, b)
        rank_score = 1.0 - float(rank) / float(rank_den)
        high_confidence = _is_high_confidence(ev, config)
        local = (
            float(config.energy_weight) * energy
            + float(config.coherence_weight) * coherence
            + float(config.rank_weight) * rank_score
            + (
                float(config.top1_soft_bonus)
                if b == int(ev.top1_bin) and (high_confidence or hard_anchor)
                else 0.0
            )
        )
        out.append(
            _ViterbiCandidate(
                raw_bin=b,
                phase=float(np.angle(ev.center_spectrum[b])),
                local_score=float(local),
                energy_score=float(energy),
                coherence_score=float(coherence),
                rank_score=float(rank_score),
                phase_local_score=0.0,
            )
        )
    if not out:
        b = int(ev.top1_bin)
        out.append(
            _ViterbiCandidate(
                raw_bin=b,
                phase=float(np.angle(ev.center_spectrum[b])),
                local_score=float(_energy_score(ev, b)),
                energy_score=float(_energy_score(ev, b)),
                coherence_score=float(_coherence_score(ev, b)),
                rank_score=1.0,
                phase_local_score=0.0,
            )
        )
    return tuple(out)


def _transition_penalty(
    prevprev: _ViterbiCandidate | None,
    prev: _ViterbiCandidate,
    curr: _ViterbiCandidate,
    config: PhasePathSelectorConfig,
    reference_slope: float | None = None,
    apply_reference_slope: bool = False,
) -> float:
    penalty = 0.0
    r1 = float(wrap_phase(float(curr.phase) - float(prev.phase)))
    if float(config.first_order_weight) > 0.0:
        penalty += float(config.first_order_weight) * _huber_normalized(
            r1,
            config.first_order_scale_rad(),
            float(config.huber_delta),
        )
    if (
        apply_reference_slope
        and reference_slope is not None
        and float(config.header_slope_weight) > 0.0
    ):
        slope_residual = float(wrap_phase(r1 - float(reference_slope)))
        penalty += float(config.header_slope_weight) * _huber_normalized(
            slope_residual,
            config.header_slope_scale_rad(),
            float(config.huber_delta),
        )
    if (
        apply_reference_slope
        and reference_slope is not None
        and float(config.anchor_slope_weight) > 0.0
    ):
        slope_residual = float(wrap_phase(r1 - float(reference_slope)))
        penalty += float(config.anchor_slope_weight) * _huber_normalized(
            slope_residual,
            config.anchor_slope_scale_rad(),
            float(config.huber_delta),
        )
    if (
        int(config.phase_order) >= 2
        and prevprev is not None
        and float(config.second_order_weight) > 0.0
    ):
        r2 = float(wrap_phase(float(curr.phase) - 2.0 * float(prev.phase) + float(prevprev.phase)))
        penalty += float(config.second_order_weight) * _huber_normalized(
            r2,
            config.second_order_scale_rad(),
            float(config.huber_delta),
        )
    return float(penalty)


def _path_phase_abs_stats(
    evidences: Sequence[SymbolEvidence],
    selected: Sequence[int],
) -> tuple[float, float]:
    phases: list[float] = []
    count = min(len(evidences), len(selected))
    for idx in range(count):
        ev = evidences[idx]
        b = int(selected[idx])
        if b < 0 or b >= ev.center_spectrum.size:
            continue
        phases.append(float(np.angle(ev.center_spectrum[b])))
    if len(phases) < 2:
        return 0.0, 0.0
    first = [
        abs(float(wrap_phase(float(phases[idx]) - float(phases[idx - 1])))) / math.pi
        for idx in range(1, len(phases))
    ]
    second = [
        abs(float(wrap_phase(float(phases[idx]) - 2.0 * float(phases[idx - 1]) + float(phases[idx - 2])))) / math.pi
        for idx in range(2, len(phases))
    ]
    return (
        float(np.mean(first)) if first else 0.0,
        float(np.mean(second)) if second else 0.0,
    )


def _arbiter_v3_config(config: PhasePathSelectorConfig) -> SymbolPhaseConfig:
    return SymbolPhaseConfig(
        top_l_low_confidence=max(1, int(getattr(config, "path_arbiter_top_l", 24))),
        lock_margin_db=1.5,
        lock_peak_to_median_db=5.0,
        lock_phase_score=0.35,
        min_locked_for_line=4,
        line_trim_frac=0.25,
        phase_model="linear",
        selection_mode="coherence",
        beam_width=128,
        trajectory_rmse_scale_pi=0.30,
        phase_weight=0.20,
        line_weight=0.00,
        amp_weight=0.80,
        profile_weight=0.00,
        phase_override_min_gain=0.15,
        phase_override_max_drop_db=0.60,
        phase_override_score_margin=0.06,
        phase_override_min_line_anchors=8,
        phase_override_max_line_rmse_pi=0.25,
        coherence_weight=0.0,
        coherence_candidate_top_l=0,
        lock_min_coherence=0.0,
        smooth_phase_weight=0.05,
        smooth_amp_weight=0.50,
        smooth_coherence_weight=0.90,
        smooth_slope_penalty=0.05,
        smooth_curvature_penalty=0.10,
        smooth_max_energy_drop_db=20.0,
        smooth_min_line_anchors=4,
        smooth_min_locked_ratio=0.0,
        smooth_max_line_rmse_pi=float("inf"),
        window_size=5,
        window_degree=1,
        window_phase_weight=0.05,
        window_amp_weight=0.50,
        window_coherence_weight=0.90,
        window_slope_weight=0.00,
        window_curvature_weight=0.00,
        window_phase_scale_pi=0.25,
        window_slope_scale_pi=0.45,
        window_curvature_scale_pi=0.25,
        window_recent_decay=0.75,
        window_anchor_span=8.0,
        window_anchor_min=2,
        window_anchor_max_rmse_pi=0.40,
        window_min_locked_ratio=0.10,
        window_guard_min_phase_gain=0.10,
        window_guard_max_energy_drop_db=0.75,
        window_guard_max_coherence_drop=0.08,
    )


def _path_arbiter_selected(
    evidences: Sequence[SymbolEvidence],
    selected: Sequence[int],
    config: PhasePathSelectorConfig,
    offset_coherences: Sequence[np.ndarray] | None,
) -> tuple[int, ...]:
    if not bool(getattr(config, "path_arbiter_enabled", False)):
        return tuple(int(v) for v in selected)
    first_abs_pi, second_abs_pi = _path_phase_abs_stats(evidences, selected)
    if (
        first_abs_pi <= float(getattr(config, "path_arbiter_first_abs_pi_threshold", float("inf")))
        and second_abs_pi <= float(getattr(config, "path_arbiter_second_abs_pi_threshold", float("inf")))
    ):
        return tuple(int(v) for v in selected)
    alt = select_symbol_bins_two_stage(
        center_spectra=[ev.center_spectrum for ev in evidences],
        evidence_powers=[ev.evidence_power for ev in evidences],
        abs_indices=[ev.abs_symbol_index for ev in evidences],
        config=_arbiter_v3_config(config),
        offset_coherences=offset_coherences,
    )
    if len(alt.selected_raw_bins) != len(selected):
        return tuple(int(v) for v in selected)
    guarded = [int(v) for v in alt.selected_raw_bins]
    for idx, ev in enumerate(evidences):
        if idx >= len(guarded) or not _is_hard_anchor(ev, config):
            continue
        allowed = {int(cand.raw_bin) for cand in _viterbi_candidates(ev, config)}
        if guarded[idx] not in allowed:
            guarded[idx] = int(ev.top1_bin)
    return tuple(int(v) for v in guarded)


def _estimate_anchor_slope(
    evidences: Sequence[SymbolEvidence],
    anchor_mask: Sequence[bool],
    config: PhasePathSelectorConfig,
) -> float | None:
    if float(config.anchor_slope_weight) <= 0.0:
        return None
    xs: list[float] = []
    phases: list[float] = []
    for idx, ev in enumerate(evidences):
        if idx >= len(anchor_mask) or not bool(anchor_mask[idx]):
            continue
        b = int(ev.top1_bin)
        if b < 0 or b >= ev.center_spectrum.size:
            continue
        xs.append(float(ev.abs_symbol_index))
        phases.append(float(np.angle(ev.center_spectrum[b])))
    if len(xs) < max(2, int(config.anchor_slope_min_anchors)):
        return None
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    x_arr = np.asarray(xs, dtype=np.float64)[order]
    phase_arr = np.unwrap(np.asarray(phases, dtype=np.float64)[order])
    try:
        coef = np.polyfit(x_arr, phase_arr, deg=1)
    except np.linalg.LinAlgError:
        return None
    fitted = np.polyval(coef, x_arr)
    rmse_pi = float(math.sqrt(float(np.mean((phase_arr - fitted) ** 2))) / math.pi)
    if math.isfinite(rmse_pi) and rmse_pi > float(config.anchor_slope_max_rmse_pi):
        return None
    return float(coef[0])


def _anchor_phase_line_rmse_pi(
    evidences: Sequence[SymbolEvidence],
    anchor_mask: Sequence[bool],
) -> float:
    xs: list[float] = []
    phases: list[float] = []
    for idx, ev in enumerate(evidences):
        if idx >= len(anchor_mask) or not bool(anchor_mask[idx]):
            continue
        b = int(ev.top1_bin)
        if b < 0 or b >= ev.center_spectrum.size:
            continue
        xs.append(float(ev.abs_symbol_index))
        phases.append(float(np.angle(ev.center_spectrum[b])))
    if len(xs) < 2:
        return float("inf")
    order = np.argsort(np.asarray(xs, dtype=np.float64))
    x_arr = np.asarray(xs, dtype=np.float64)[order]
    phase_arr = np.unwrap(np.asarray(phases, dtype=np.float64)[order])
    try:
        coef = np.polyfit(x_arr, phase_arr, deg=1)
    except np.linalg.LinAlgError:
        return float("inf")
    fitted = np.polyval(coef, x_arr)
    return float(math.sqrt(float(np.mean((phase_arr - fitted) ** 2))) / math.pi)


def _sliding_window_refine_path(
    evidences: Sequence[SymbolEvidence],
    selected: Sequence[int],
    config: PhasePathSelectorConfig,
    anchor_mask: Sequence[bool] | None = None,
) -> tuple[int, ...]:
    if not bool(config.sliding_window_refine_enabled):
        return tuple(int(v) for v in selected)
    count = min(len(evidences), len(selected))
    if count <= 0:
        return tuple(int(v) for v in selected)

    refined = [int(v) for v in selected[:count]]
    anchors = tuple(bool(v) for v in anchor_mask[:count]) if anchor_mask is not None else tuple(False for _ in range(count))
    radius = max(1, int(config.sliding_window_radius))
    min_anchors = max(2, int(config.sliding_window_min_anchors))
    phase_scale = max(1e-6, float(config.sliding_window_phase_scale_pi) * math.pi)
    weights = (
        max(0.0, float(config.sliding_window_phase_weight)),
        max(0.0, float(config.sliding_window_energy_weight)),
        max(0.0, float(config.sliding_window_coherence_weight)),
    )
    total_w = max(1e-12, sum(weights))
    w_phase, w_energy, w_coh = (float(v / total_w) for v in weights)

    base_selected = list(refined)
    for idx, ev in enumerate(evidences[:count]):
        if idx < len(anchors) and bool(anchors[idx]):
            continue
        anchor_rows: list[tuple[float, float, float]] = []
        lo = max(0, idx - radius)
        hi = min(count, idx + radius + 1)
        for j in range(lo, hi):
            if j == idx:
                continue
            if anchor_mask is not None and (j >= len(anchors) or not bool(anchors[j])):
                continue
            anchor_bin = int(base_selected[j])
            anchor_ev = evidences[j]
            if anchor_bin < 0 or anchor_bin >= anchor_ev.center_spectrum.size:
                continue
            distance = abs(j - idx)
            weight = math.exp(-0.5 * (float(distance) / max(1e-6, 0.5 * float(radius))) ** 2)
            anchor_rows.append(
                (
                    float(anchor_ev.abs_symbol_index),
                    float(np.angle(anchor_ev.center_spectrum[anchor_bin])),
                    float(weight),
                )
            )
        if len(anchor_rows) < min_anchors:
            continue
        anchor_rows.sort(key=lambda item: item[0])
        xs = np.asarray([item[0] for item in anchor_rows], dtype=np.float64)
        phases = np.unwrap(np.asarray([item[1] for item in anchor_rows], dtype=np.float64))
        fit_w = np.asarray([item[2] for item in anchor_rows], dtype=np.float64)
        try:
            coef = np.polyfit(xs - float(ev.abs_symbol_index), phases, deg=1, w=fit_w)
        except np.linalg.LinAlgError:
            continue
        fitted = np.polyval(coef, xs - float(ev.abs_symbol_index))
        rmse_pi = float(math.sqrt(float(np.average((phases - fitted) ** 2, weights=fit_w))) / math.pi)
        if math.isfinite(rmse_pi) and rmse_pi > float(config.sliding_window_max_rmse_pi):
            continue
        predicted = float(np.polyval(coef, 0.0))
        top_power = float(ev.evidence_power[int(ev.top1_bin)]) if 0 <= int(ev.top1_bin) < ev.evidence_power.size else 0.0
        current_bin = int(base_selected[idx])
        current_power = (
            float(ev.evidence_power[current_bin])
            if 0 <= current_bin < ev.evidence_power.size
            else 0.0
        )

        def score_bin(raw_bin: int) -> tuple[float, float, float, float]:
            b = int(raw_bin)
            if b < 0 or b >= ev.center_spectrum.size or b >= ev.evidence_power.size:
                return -float("inf"), 0.0, 0.0, 0.0
            residual = float(wrap_phase(float(np.angle(ev.center_spectrum[b])) - predicted))
            phase_score = float(math.exp(-((residual / phase_scale) ** 2)))
            energy_score = _energy_score(ev, b)
            coherence_score = _coherence_score(ev, b)
            score = w_phase * phase_score + w_energy * energy_score + w_coh * coherence_score
            return float(score), float(phase_score), float(energy_score), float(coherence_score)

        current_score, _cur_phase, _cur_energy, current_coh = score_bin(current_bin)
        best_bin = current_bin
        best_score = current_score
        best_coh = current_coh
        for cand in _viterbi_candidates(ev, config):
            b = int(cand.raw_bin)
            if b == current_bin:
                continue
            if _db_ratio(float(ev.evidence_power[b]), top_power) < -float(config.sliding_window_max_energy_drop_db):
                continue
            candidate_score, _phase_score, _energy_score_value, candidate_coh = score_bin(b)
            if candidate_score > best_score:
                best_bin = b
                best_score = candidate_score
                best_coh = candidate_coh
        if best_bin == current_bin:
            continue
        switch_loss = _db_ratio(float(ev.evidence_power[best_bin]), current_power)
        coherence_gain = float(best_coh - current_coh)
        if (
            best_score >= current_score + float(config.sliding_window_min_gain)
            and (
                switch_loss >= -float(config.sliding_window_max_switch_loss_db)
                or coherence_gain >= float(config.sliding_window_min_coherence_gain)
            )
        ):
            refined[idx] = int(best_bin)
    if len(selected) > count:
        refined.extend(int(v) for v in selected[count:])
    return tuple(int(v) for v in refined)


def _anchor_phase_prediction_for_path(
    evidences: Sequence[SymbolEvidence],
    anchor_mask: Sequence[bool],
    phase_reference_mask: Sequence[bool],
    target_index: int,
    config: PhasePathSelectorConfig,
) -> tuple[bool, float]:
    idx_target = int(target_index)
    if idx_target < 0 or idx_target >= len(evidences):
        return False, 0.0
    target_abs = float(evidences[idx_target].abs_symbol_index)
    span = max(1e-6, float(config.anchor_phase_bias_span))
    anchors: list[tuple[float, float, float]] = []
    for idx, ev in enumerate(evidences):
        anchor_ref = idx < len(anchor_mask) and bool(anchor_mask[idx])
        phase_ref = idx < len(phase_reference_mask) and bool(phase_reference_mask[idx])
        if (not anchor_ref and not phase_ref) or idx == idx_target:
            continue
        raw_bin = int(ev.top1_bin)
        if raw_bin < 0 or raw_bin >= ev.center_spectrum.size:
            continue
        distance = abs(float(ev.abs_symbol_index) - target_abs)
        if distance > span:
            continue
        weight = math.exp(-0.5 * (distance / max(1e-6, 0.5 * span)) ** 2)
        anchors.append((float(ev.abs_symbol_index), float(np.angle(ev.center_spectrum[raw_bin])), float(weight)))
    if len(anchors) < max(2, int(config.anchor_phase_bias_min_anchors)):
        return False, 0.0
    anchors.sort(key=lambda item: item[0])
    xs = np.asarray([item[0] for item in anchors], dtype=np.float64)
    phases = np.unwrap(np.asarray([item[1] for item in anchors], dtype=np.float64))
    weights = np.asarray([item[2] for item in anchors], dtype=np.float64)
    trim_frac = max(0.0, min(0.45, float(getattr(config, "anchor_phase_bias_trim_frac", 0.20))))
    try:
        coef = np.polyfit(xs - target_abs, phases, deg=1, w=weights)
    except np.linalg.LinAlgError:
        return False, 0.0
    fitted = np.polyval(coef, xs - target_abs)
    residual_abs = np.abs(phases - fitted)
    keep_count = int(round(float(len(residual_abs)) * (1.0 - trim_frac)))
    keep_count = max(max(2, int(config.anchor_phase_bias_min_anchors)), min(len(residual_abs), keep_count))
    if keep_count < len(residual_abs):
        keep = np.argsort(residual_abs)[:keep_count]
        try:
            coef = np.polyfit(xs[keep] - target_abs, phases[keep], deg=1, w=weights[keep])
        except np.linalg.LinAlgError:
            return False, 0.0
        fitted = np.polyval(coef, xs[keep] - target_abs)
        phases_for_rmse = phases[keep]
        weights_for_rmse = weights[keep]
    else:
        phases_for_rmse = phases
        weights_for_rmse = weights
    rmse_pi = float(math.sqrt(float(np.average((phases_for_rmse - fitted) ** 2, weights=weights_for_rmse))) / math.pi)
    if math.isfinite(rmse_pi) and rmse_pi > float(config.anchor_phase_bias_max_rmse_pi):
        return False, 0.0
    return True, float(np.polyval(coef, 0.0))


def _apply_anchor_phase_bias(
    candidates: Sequence[tuple[_ViterbiCandidate, ...]],
    evidences: Sequence[SymbolEvidence],
    anchor_mask: Sequence[bool],
    phase_reference_mask: Sequence[bool],
    config: PhasePathSelectorConfig,
) -> tuple[tuple[_ViterbiCandidate, ...], ...]:
    weight = float(config.anchor_phase_bias_weight)
    if weight <= 0.0:
        return tuple(tuple(row) for row in candidates)
    phase_scale = max(1e-6, float(config.anchor_phase_bias_scale_pi) * math.pi)
    out: list[tuple[_ViterbiCandidate, ...]] = []
    for idx, row in enumerate(candidates):
        if idx >= len(evidences) or (idx < len(anchor_mask) and bool(anchor_mask[idx])):
            out.append(tuple(row))
            continue
        has_pred, predicted = _anchor_phase_prediction_for_path(evidences, anchor_mask, phase_reference_mask, idx, config)
        if not has_pred:
            out.append(tuple(row))
            continue
        adjusted: list[_ViterbiCandidate] = []
        for cand in row:
            residual = float(wrap_phase(float(cand.phase) - predicted))
            phase_score = float(math.exp(-((residual / phase_scale) ** 2)))
            candidate_with_phase = _ViterbiCandidate(
                raw_bin=int(cand.raw_bin),
                phase=float(cand.phase),
                local_score=float(cand.local_score),
                energy_score=float(cand.energy_score),
                coherence_score=float(cand.coherence_score),
                rank_score=float(cand.rank_score),
                phase_local_score=float(phase_score),
            )
            adjusted.append(
                _ViterbiCandidate(
                    raw_bin=int(cand.raw_bin),
                    phase=float(cand.phase),
                    local_score=float(
                        _candidate_local_score(
                            evidences[idx],
                            candidate_with_phase,
                            config,
                            high_confidence=_is_high_confidence(evidences[idx], config),
                            hard_anchor=idx < len(anchor_mask) and bool(anchor_mask[idx]),
                        )
                        if bool(getattr(config, "adaptive_local_score_enabled", False))
                        or float(getattr(config, "phase_local_weight", 0.0)) > 0.0
                        else cand.local_score + weight * phase_score
                    ),
                    energy_score=float(cand.energy_score),
                    coherence_score=float(cand.coherence_score),
                    rank_score=float(cand.rank_score),
                    phase_local_score=float(phase_score),
                )
            )
        out.append(tuple(adjusted))
    return tuple(out)


def _reliability(ev: SymbolEvidence, config: PhaseLineSelectorConfig) -> float:
    margin = (float(ev.top1_margin_db) - float(config.reliable_margin_db)) / max(1e-6, float(config.reliability_temperature))
    peak = (
        float(ev.top1_peak_to_median_db) - float(config.reliable_peak_to_median_db)
    ) / max(1e-6, 2.0 * float(config.reliability_temperature))
    value = 1.0 / (1.0 + math.exp(-(margin + peak)))
    return float(max(0.0, min(1.0, value)))


def _dynamic_weights(
    ev: SymbolEvidence,
    has_phase_prediction: bool,
    config: PhaseLineSelectorConfig,
) -> tuple[float, float, float]:
    rel = _reliability(ev, config)
    phase_w = (1.0 - rel) * float(config.phase_weight_low_conf) + rel * float(config.phase_weight_high_conf)
    energy_w = (1.0 - rel) * float(config.energy_weight_low_conf) + rel * float(config.energy_weight_high_conf)
    if not has_phase_prediction:
        if bool(config.fallback_to_energy_without_phase):
            phase_w = 0.0
            energy_w = 1.0 - max(0.0, float(config.coherence_weight))
        else:
            phase_w *= 0.25
    coh_w = max(0.0, float(config.coherence_weight))
    total = max(1e-12, phase_w + energy_w + coh_w)
    return float(phase_w / total), float(energy_w / total), float(coh_w / total)


def _max_energy_drop_db(ev: SymbolEvidence, config: PhaseLineSelectorConfig) -> float:
    rel = _reliability(ev, config)
    return float((1.0 - rel) * float(config.max_energy_drop_db_low_conf) + rel * float(config.max_energy_drop_db_high_conf))


def _min_phase_gain_to_switch(ev: SymbolEvidence, config: PhaseLineSelectorConfig) -> float:
    rel = _reliability(ev, config)
    return float(
        (1.0 - rel) * float(config.min_phase_gain_to_switch_low_conf)
        + rel * float(config.min_phase_gain_to_switch_high_conf)
    )


def _candidate_bins(ev: SymbolEvidence, config: PhaseLineSelectorConfig) -> tuple[int, ...]:
    bins = [int(v) for v in ev.top_bins[: max(1, int(config.top_l))]]
    if not bins:
        return (int(ev.top1_bin),)
    max_drop = _max_energy_drop_db(ev, config)
    top_power = float(ev.evidence_power[int(ev.top1_bin)]) if 0 <= int(ev.top1_bin) < ev.evidence_power.size else 0.0
    out: list[int] = []
    for b in bins:
        if b == int(ev.top1_bin):
            out.append(b)
            continue
        if b < 0 or b >= ev.evidence_power.size:
            continue
        drop = _db_ratio(float(ev.evidence_power[b]), top_power)
        if drop >= -max_drop:
            out.append(b)
    return tuple(out or [int(ev.top1_bin)])


def _should_hard_lock(ev: SymbolEvidence, config: PhaseLineSelectorConfig) -> bool:
    energy_lock = (
        float(ev.top1_margin_db) >= float(config.top1_lock_margin_db)
        and float(ev.top1_peak_to_median_db) >= float(config.top1_lock_peak_to_median_db)
    )
    coherence_lock = (
        ev.offset_coherence is not None
        and float(ev.top1_coherence_score) >= float(config.top1_lock_min_coherence)
        and float(ev.top1_margin_db) >= max(0.5, 0.45 * float(config.top1_lock_margin_db))
        and float(ev.top1_peak_to_median_db) >= max(3.0, 0.55 * float(config.top1_lock_peak_to_median_db))
    )
    return bool(energy_lock or coherence_lock)


def _score_candidate(
    ev: SymbolEvidence,
    raw_bin: int,
    predicted_phase: float,
    predicted_slope: float,
    predicted_curvature: float,
    has_phase_prediction: bool,
    previous_phase: float | None,
    previous_abs_index: float | None,
    previous_slope: float | None,
    config: PhaseLineSelectorConfig,
) -> tuple[float, float, float, float, float, float | None]:
    b = int(raw_bin)
    if b < 0 or b >= ev.center_spectrum.size:
        return -float("inf"), 0.0, 0.0, 0.0, 0.0, previous_slope
    reference = float(predicted_phase) if has_phase_prediction else (float(previous_phase) if previous_phase is not None else 0.0)
    raw_phase = float(np.angle(ev.center_spectrum[b]))
    observed = unwrap_against(reference, raw_phase)
    residual = float(wrap_phase(observed - float(predicted_phase))) if has_phase_prediction else 0.0
    phase_score = phase_likelihood(residual, config) if has_phase_prediction else 0.0
    energy_score = _energy_score(ev, b)
    coherence_score = _coherence_score(ev, b)
    phase_w, energy_w, coh_w = _dynamic_weights(ev, has_phase_prediction, config)
    if has_phase_prediction and phase_w > 1e-12:
        penalty, local_slope = smoothness_penalty(
            observed_phase=observed,
            current_abs_index=ev.abs_symbol_index,
            previous_phase=previous_phase,
            previous_abs_index=previous_abs_index,
            previous_slope=previous_slope,
            predicted_slope=predicted_slope,
            predicted_curvature=predicted_curvature,
            config=config,
        )
    else:
        penalty = 0.0
        local_slope = previous_slope
    score = phase_w * phase_score + energy_w * energy_score + coh_w * coherence_score - penalty
    return float(score), float(observed), float(phase_score), float(energy_score), float(coherence_score), local_slope


def _candidate_phase_score(
    ev: SymbolEvidence,
    raw_bin: int,
    predicted_phase: float,
    has_phase_prediction: bool,
    config: PhaseLineSelectorConfig,
) -> float:
    if not has_phase_prediction:
        return 0.0
    b = int(raw_bin)
    if b < 0 or b >= ev.center_spectrum.size:
        return 0.0
    residual = float(wrap_phase(float(np.angle(ev.center_spectrum[b])) - float(predicted_phase)))
    return phase_likelihood(residual, config)


def _local_anchor_prediction(
    evidences: Sequence[SymbolEvidence],
    locked_mask: Sequence[bool],
    target_index: int,
    config: PhaseLineSelectorConfig,
) -> tuple[bool, float, float, float]:
    idx_target = int(target_index)
    if idx_target < 0 or idx_target >= len(evidences):
        return False, 0.0, 0.0, 0.0
    target_abs = float(evidences[idx_target].abs_symbol_index)
    anchors: list[tuple[float, float, float]] = []
    span = max(1e-6, float(config.anchor_span_symbols))
    for idx, ev in enumerate(evidences):
        if idx >= len(locked_mask) or not bool(locked_mask[idx]):
            continue
        raw_bin = int(ev.top1_bin)
        if raw_bin < 0 or raw_bin >= ev.center_spectrum.size:
            continue
        distance = abs(float(ev.abs_symbol_index) - target_abs)
        if distance > span:
            continue
        weight = math.exp(-0.5 * (distance / max(1e-6, 0.5 * span)) ** 2)
        anchors.append((float(ev.abs_symbol_index), float(np.angle(ev.center_spectrum[raw_bin])), float(weight)))
    if len(anchors) < max(1, int(config.anchor_min_count)):
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
        return False, 0.0, 0.0, 0.0
    fitted = np.polyval(coef, x_rel)
    rmse_pi = float(math.sqrt(float(np.average((phases - fitted) ** 2, weights=weights))) / math.pi)
    if math.isfinite(rmse_pi) and rmse_pi > float(config.anchor_max_rmse_pi):
        return False, 0.0, 0.0, 0.0
    pred = float(np.polyval(coef, 0.0))
    deriv = np.polyder(coef, m=1)
    slope = float(np.polyval(deriv, 0.0)) if deriv.size else 0.0
    second = np.polyder(coef, m=2)
    curvature = float(np.polyval(second, 0.0)) if second.size else 0.0
    return True, pred, slope, curvature


def _select_with_local_anchor_lines(
    evidences: Sequence[SymbolEvidence],
    locked_mask: Sequence[bool],
    config: PhaseLineSelectorConfig,
) -> tuple[int, ...]:
    selected: list[int] = []
    for idx, ev in enumerate(evidences):
        if idx < len(locked_mask) and bool(locked_mask[idx]):
            selected.append(int(ev.top1_bin))
            continue
        has_pred, predicted, pred_slope, pred_curvature = _local_anchor_prediction(evidences, locked_mask, idx, config)
        if not has_pred:
            selected.append(int(ev.top1_bin))
            continue
        top1_phase = _candidate_phase_score(ev, ev.top1_bin, predicted, True, config)
        min_gain = _min_phase_gain_to_switch(ev, config)
        best_bin = int(ev.top1_bin)
        best_score, *_ = _score_candidate(
            ev=ev,
            raw_bin=best_bin,
            predicted_phase=predicted,
            predicted_slope=pred_slope,
            predicted_curvature=pred_curvature,
            has_phase_prediction=True,
            previous_phase=None,
            previous_abs_index=None,
            previous_slope=None,
            config=config,
        )
        for raw_bin in _candidate_bins(ev, config):
            b = int(raw_bin)
            if b == int(ev.top1_bin):
                continue
            cand_phase = _candidate_phase_score(ev, b, predicted, True, config)
            if cand_phase < top1_phase + min_gain:
                continue
            score, *_ = _score_candidate(
                ev=ev,
                raw_bin=b,
                predicted_phase=predicted,
                predicted_slope=pred_slope,
                predicted_curvature=pred_curvature,
                has_phase_prediction=True,
                previous_phase=None,
                previous_abs_index=None,
                previous_slope=None,
                config=config,
            )
            if score > best_score:
                best_score = float(score)
                best_bin = b
        selected.append(int(best_bin))
    return tuple(int(v) for v in selected)


def select_phase_smooth_path(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PhaseLineSelectorConfig | None = None,
    fallback_line: PhaseLine | None = None,
    offset_coherences: Sequence[np.ndarray] | None = None,
) -> SymbolPhaseResult:
    """Select a raw-bin path with phase smoothness as the second-stage driver."""

    cfg = config or PhaseLineSelectorConfig()
    stage1_config = SymbolPhaseConfig(
        top_l_low_confidence=int(cfg.top_l),
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

    locked_mask: list[bool] = [bool(_should_hard_lock(ev, cfg)) for ev in evidences]
    evidences = _augment_evidences_with_phase_proposals(evidences, locked_mask, cfg)
    locked_mask = [bool(_should_hard_lock(ev, cfg)) for ev in evidences]
    if sum(locked_mask) >= max(1, int(cfg.anchor_min_count)):
        selected = _select_with_local_anchor_lines(evidences, locked_mask, cfg)
        beam_size = 1
        error = ""
    else:
        selected = tuple(int(ev.top1_bin) for ev in evidences)
        beam_size = 1
        error = "phase-line fallback to multi-offset top1: insufficient anchors"

    if len(selected) != len(evidences):
        selected = tuple(int(ev.top1_bin) for ev in evidences)

    line = fit_selected_phase_line(
        center_spectra=[ev.center_spectrum for ev in evidences],
        selected_bins=selected,
        abs_indices=[ev.abs_symbol_index for ev in evidences],
        trim_frac=float(cfg.line_trim_frac),
    )
    phase_scores: list[float] = []
    amp_scores: list[float] = []
    for ev, raw_bin in zip(evidences, selected):
        predicted = float(line.predict(ev.abs_symbol_index)) if int(line.anchor_count) >= 2 else 0.0
        residual = float(wrap_phase(float(np.angle(ev.center_spectrum[int(raw_bin)])) - predicted))
        phase_scores.append(phase_likelihood(residual, cfg) if int(line.anchor_count) >= 2 else 0.0)
        amp_scores.append(_energy_score(ev, int(raw_bin)))
    mean_phase = float(np.mean(phase_scores)) if phase_scores else 0.0
    mean_amp = float(np.mean(amp_scores)) if amp_scores else 0.0
    if int(line.anchor_count) >= 2 and math.isfinite(float(line.fit_rmse_pi)):
        line_score = float(math.exp(-((float(line.fit_rmse_pi) / max(1e-6, float(cfg.phase_scale_pi))) ** 2)))
    else:
        line_score = 0.0
    return SymbolPhaseResult(
        selected_raw_bins=selected,
        locked_mask=tuple(bool(v) for v in locked_mask),
        evidences=evidences,
        phase_line=line,
        trajectory_score=float(mean_phase + line_score),
        line_score=float(line_score),
        mean_phase_score=float(mean_phase),
        mean_amp_score=float(mean_amp),
        uncertain_count=int(sum(not v for v in locked_mask)),
        locked_count=int(sum(locked_mask)),
        beam_final_size=int(beam_size),
        error=error,
    )


def select_phase_viterbi_path(
    center_spectra: Sequence[np.ndarray],
    evidence_powers: Sequence[np.ndarray],
    abs_indices: Sequence[float],
    config: PhasePathSelectorConfig | None = None,
    fallback_line: PhaseLine | None = None,
    offset_coherences: Sequence[np.ndarray] | None = None,
) -> SymbolPhaseResult:
    """Select a payload raw-bin path with phase-smooth Viterbi decoding.

    This is the payload-only second-stage proposed in the phase-line notes:
    Stage 1 builds Top-L candidates from multi-offset FFT evidence, and Stage 2
    searches the full payload sequence with first/second phase-difference
    penalties.  Header phase is intentionally not used here.
    """

    cfg = config or PhasePathSelectorConfig()
    stage1_top_l = max(
        int(cfg.top_l),
        int(getattr(cfg, "adaptive_top_l_high", 0))
        if int(getattr(cfg, "adaptive_top_l_anchor_threshold", 0)) > 0
        else int(cfg.top_l),
    )
    stage1_config = SymbolPhaseConfig(
        top_l_low_confidence=int(stage1_top_l),
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

    high_confidence_mask = [bool(_is_high_confidence(ev, cfg)) for ev in evidences]
    pre_hard_anchor_mask = tuple(bool(_is_hard_anchor(ev, cfg)) for ev in evidences)
    pre_hard_anchor_count = int(sum(pre_hard_anchor_mask))
    hard_anchor_rmse_pi = _anchor_phase_line_rmse_pi(evidences, pre_hard_anchor_mask)
    use_hard_for_proposal = bool(getattr(cfg, "phase_proposal_use_hard_anchors", False))
    if not use_hard_for_proposal and bool(getattr(cfg, "phase_proposal_gated_hard_anchors", False)):
        min_count = int(getattr(cfg, "phase_proposal_gated_hard_anchor_min_count", 0))
        max_count = int(getattr(cfg, "phase_proposal_gated_hard_anchor_max_count", 0))
        max_rmse = float(getattr(cfg, "phase_proposal_gated_hard_anchor_max_rmse_pi", float("inf")))
        use_hard_for_proposal = bool(
            pre_hard_anchor_count >= min_count
            and (max_count <= 0 or pre_hard_anchor_count <= max_count)
            and hard_anchor_rmse_pi <= max_rmse
        )
    proposal_anchor_mask = tuple(
        bool(high_confidence_mask[idx] or (use_hard_for_proposal and pre_hard_anchor_mask[idx]))
        for idx in range(len(evidences))
    )
    adaptive_top_l_high = int(getattr(cfg, "adaptive_top_l_high", 0))
    adaptive_anchor_threshold = int(getattr(cfg, "adaptive_top_l_anchor_threshold", 0))
    if (
        adaptive_top_l_high > int(cfg.top_l)
        and adaptive_anchor_threshold > 0
        and pre_hard_anchor_count >= adaptive_anchor_threshold
    ):
        cfg = replace(cfg, top_l=adaptive_top_l_high)
    else:
        adaptive_top_l_mid = int(getattr(cfg, "adaptive_top_l_mid", 0))
        adaptive_mid_threshold = int(getattr(cfg, "adaptive_top_l_mid_anchor_threshold", 0))
        adaptive_mid_rmse = float(getattr(cfg, "adaptive_top_l_mid_max_anchor_rmse_pi", float("inf")))
        if (
            adaptive_top_l_mid > int(cfg.top_l)
            and adaptive_mid_threshold > 0
            and pre_hard_anchor_count >= adaptive_mid_threshold
            and hard_anchor_rmse_pi <= adaptive_mid_rmse
        ):
            cfg = replace(cfg, top_l=adaptive_top_l_mid)
    adaptive_phase_threshold = int(getattr(cfg, "adaptive_phase_relax_anchor_threshold", 0))
    if (
        adaptive_phase_threshold > 0
        and pre_hard_anchor_count >= adaptive_phase_threshold
        and (int(cfg.phase_order) < 2 or float(cfg.second_order_weight) <= 0.0)
    ):
        cfg = replace(
            cfg,
            first_order_weight=float(getattr(cfg, "adaptive_phase_relax_first_order_weight", cfg.first_order_weight)),
            anchor_slope_weight=float(getattr(cfg, "adaptive_phase_relax_anchor_slope_weight", cfg.anchor_slope_weight)),
            anchor_slope_max_rmse_pi=float(
                getattr(cfg, "adaptive_phase_relax_anchor_slope_max_rmse_pi", cfg.anchor_slope_max_rmse_pi)
            ),
        )
    if bool(getattr(cfg, "adaptive_hard_anchor_softening_enabled", False)):
        soft_min_count = int(getattr(cfg, "adaptive_hard_anchor_softening_min_count", 0))
        soft_max_rmse = float(getattr(cfg, "adaptive_hard_anchor_softening_max_rmse_pi", float("inf")))
        if pre_hard_anchor_count >= soft_min_count and hard_anchor_rmse_pi <= soft_max_rmse:
            cfg = replace(
                cfg,
                hard_anchor_soft_top_k=max(
                    int(getattr(cfg, "hard_anchor_soft_top_k", 1)),
                    int(getattr(cfg, "adaptive_hard_anchor_softening_top_k", 1)),
                ),
                hard_anchor_soft_max_margin_db=float(
                    getattr(cfg, "adaptive_hard_anchor_softening_max_margin_db", 0.0)
                ),
                hard_anchor_soft_max_peak_to_median_db=float(
                    getattr(cfg, "adaptive_hard_anchor_softening_max_peak_to_median_db", 0.0)
                ),
    )
    evidences = _augment_evidences_with_phase_proposals(evidences, proposal_anchor_mask, cfg)
    hard_anchor_mask = tuple(bool(_is_hard_anchor(ev, cfg)) for ev in evidences)
    candidates = tuple(_viterbi_candidates(ev, cfg) for ev in evidences)
    phase_reference_mask = tuple(
        bool(high_confidence_mask[idx] or hard_anchor_mask[idx])
        for idx in range(len(evidences))
    )
    candidates = _apply_anchor_phase_bias(candidates, evidences, hard_anchor_mask, phase_reference_mask, cfg)
    anchor_slope = _estimate_anchor_slope(evidences, hard_anchor_mask, cfg)
    header_slope: float | None = None
    if fallback_line is not None and int(fallback_line.anchor_count) >= 2 and float(cfg.header_slope_weight) > 0.0:
        header_slope = float(fallback_line.slope_rad)
    reference_slope = anchor_slope if anchor_slope is not None else header_slope
    if len(candidates) == 1:
        best = max(candidates[0], key=lambda cand: cand.local_score)
        selected = (int(best.raw_bin),)
        final_score = float(best.local_score)
        final_size = len(candidates[0])
    elif int(cfg.phase_order) < 2 or float(cfg.second_order_weight) <= 0.0:
        states: dict[int, tuple[float, tuple[int, ...]]] = {
            idx: (float(cand.local_score), (idx,))
            for idx, cand in enumerate(candidates[0])
        }
        for t in range(1, len(candidates)):
            next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
            apply_ref = bool(
                header_slope is not None
                and (
                    (t == 1 and int(cfg.header_slope_span) >= 2)
                    or (t < int(cfg.header_slope_span))
                )
            )
            for prev_idx, (score, path) in states.items():
                prev = candidates[t - 1][prev_idx]
                for curr_idx, curr in enumerate(candidates[t]):
                    penalty = _transition_penalty(
                        None,
                        prev,
                        curr,
                        cfg,
                        reference_slope=reference_slope,
                        apply_reference_slope=bool(anchor_slope is not None or apply_ref),
                    )
                    next_score = float(score + curr.local_score - penalty)
                    if curr_idx not in next_states or next_score > next_states[curr_idx][0]:
                        next_states[curr_idx] = (next_score, path + (curr_idx,))
            states = next_states
        if not states:
            selected = tuple(int(ev.top1_bin) for ev in evidences)
            final_score = 0.0
            final_size = 0
        else:
            final_score, path = max(states.values(), key=lambda item: item[0])
            selected = tuple(int(candidates[t][path[t]].raw_bin) for t in range(len(path)))
            final_size = len(states)
    else:
        states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for i, cand0 in enumerate(candidates[0]):
            for j, cand1 in enumerate(candidates[1]):
                penalty = _transition_penalty(
                    None,
                    cand0,
                    cand1,
                    cfg,
                    reference_slope=reference_slope,
                    apply_reference_slope=bool(anchor_slope is not None or int(cfg.header_slope_span) >= 2),
                )
                score = float(cand0.local_score + cand1.local_score - penalty)
                states[(i, j)] = (score, (i, j))

        for t in range(2, len(candidates)):
            next_states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
            for (prevprev_idx, prev_idx), (score, path) in states.items():
                prevprev = candidates[t - 2][prevprev_idx]
                prev = candidates[t - 1][prev_idx]
                for curr_idx, curr in enumerate(candidates[t]):
                    penalty = _transition_penalty(
                        prevprev,
                        prev,
                        curr,
                        cfg,
                        reference_slope=reference_slope,
                        apply_reference_slope=bool(anchor_slope is not None or (header_slope is not None and t < int(cfg.header_slope_span))),
                    )
                    next_score = float(score + curr.local_score - penalty)
                    key = (prev_idx, curr_idx)
                    if key not in next_states or next_score > next_states[key][0]:
                        next_states[key] = (next_score, path + (curr_idx,))
            states = next_states

        if not states:
            selected = tuple(int(ev.top1_bin) for ev in evidences)
            final_score = 0.0
            final_size = 0
        else:
            final_score, path = max(states.values(), key=lambda item: item[0])
            selected = tuple(int(candidates[t][path[t]].raw_bin) for t in range(len(path)))
            final_size = len(states)

    selected = _sliding_window_refine_path(evidences, selected, cfg, hard_anchor_mask)
    selected = _path_arbiter_selected(evidences, selected, cfg, offset_coherences)

    line = fit_selected_phase_line(
        center_spectra=[ev.center_spectrum for ev in evidences],
        selected_bins=selected,
        abs_indices=[ev.abs_symbol_index for ev in evidences],
        trim_frac=float(cfg.line_trim_frac),
    )
    amp_scores: list[float] = []
    phase_penalties: list[float] = []
    for idx, (ev, raw_bin) in enumerate(zip(evidences, selected)):
        amp_scores.append(_energy_score(ev, int(raw_bin)))
        if idx >= 1:
            prev = _ViterbiCandidate(
                raw_bin=int(selected[idx - 1]),
                phase=float(np.angle(evidences[idx - 1].center_spectrum[int(selected[idx - 1])])),
                local_score=0.0,
                energy_score=0.0,
                coherence_score=0.0,
                rank_score=0.0,
                phase_local_score=0.0,
            )
            curr = _ViterbiCandidate(
                raw_bin=int(raw_bin),
                phase=float(np.angle(ev.center_spectrum[int(raw_bin)])),
                local_score=0.0,
                energy_score=0.0,
                coherence_score=0.0,
                rank_score=0.0,
                phase_local_score=0.0,
            )
            prevprev = None
            if idx >= 2:
                prevprev = _ViterbiCandidate(
                    raw_bin=int(selected[idx - 2]),
                    phase=float(np.angle(evidences[idx - 2].center_spectrum[int(selected[idx - 2])])),
                    local_score=0.0,
                    energy_score=0.0,
                    coherence_score=0.0,
                    rank_score=0.0,
                    phase_local_score=0.0,
                )
            phase_penalties.append(_transition_penalty(prevprev, prev, curr, cfg))
    mean_amp = float(np.mean(amp_scores)) if amp_scores else 0.0
    mean_penalty = float(np.mean(phase_penalties)) if phase_penalties else 0.0
    mean_phase = float(math.exp(-mean_penalty)) if math.isfinite(mean_penalty) else 0.0
    return SymbolPhaseResult(
        selected_raw_bins=selected,
        locked_mask=hard_anchor_mask,
        evidences=evidences,
        phase_line=line,
        trajectory_score=float(final_score / max(1, len(selected))),
        line_score=0.0,
        mean_phase_score=float(mean_phase),
        mean_amp_score=float(mean_amp),
        uncertain_count=int(len(evidences)),
        locked_count=int(sum(hard_anchor_mask)),
        beam_final_size=int(final_size),
        error="",
    )


__all__ = [
    "PhaseLineSelectorConfig",
    "PhasePathSelectorConfig",
    "select_phase_smooth_path",
    "select_phase_viterbi_path",
]

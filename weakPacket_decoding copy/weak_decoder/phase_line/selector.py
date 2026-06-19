"""Phase-line dominated candidate path selector."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ..candidate_pruning import top_bins, wrap_phase
from ..phase_guided_demod import PhaseLine
from ..symbol_phase_two_stage import SymbolEvidence, SymbolPhaseConfig, SymbolPhaseResult, build_symbol_evidences
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


def _huber_normalized(value: float, scale: float, delta: float) -> float:
    x = abs(float(value)) / max(1e-12, float(scale))
    d = max(1e-12, float(delta))
    if x <= d:
        return float(0.5 * x * x)
    return float(d * (x - 0.5 * d))


def _candidate_drop_db(ev: SymbolEvidence, raw_bin: int) -> float:
    b = int(raw_bin)
    if b < 0 or b >= ev.evidence_power.size:
        return -float("inf")
    top = float(ev.evidence_power[int(ev.top1_bin)]) if 0 <= int(ev.top1_bin) < ev.evidence_power.size else 0.0
    return _db_ratio(float(ev.evidence_power[b]), top)


def _viterbi_candidates(ev: SymbolEvidence, config: PhasePathSelectorConfig) -> tuple[_ViterbiCandidate, ...]:
    bins = [int(v) for v in ev.top_bins[: max(1, int(config.top_l))]]
    if not bins:
        bins = [int(ev.top1_bin)]
    if _is_high_confidence(ev, config) and int(config.high_confidence_top_k) > 0:
        keep = max(1, int(config.high_confidence_top_k))
        bins = bins[:keep]

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
        bonus = 0.0
        if b == int(ev.top1_bin) and _is_high_confidence(ev, config):
            bonus = float(config.top1_soft_bonus)
        local = (
            float(config.energy_weight) * energy
            + float(config.coherence_weight) * coherence
            + float(config.rank_weight) * rank_score
            + bonus
        )
        out.append(
            _ViterbiCandidate(
                raw_bin=b,
                phase=float(np.angle(ev.center_spectrum[b])),
                local_score=float(local),
                energy_score=float(energy),
                coherence_score=float(coherence),
                rank_score=float(rank_score),
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

    candidates = tuple(_viterbi_candidates(ev, cfg) for ev in evidences)
    header_slope: float | None = None
    if fallback_line is not None and int(fallback_line.anchor_count) >= 2 and float(cfg.header_slope_weight) > 0.0:
        header_slope = float(fallback_line.slope_rad)
    if len(candidates) == 1:
        best = max(candidates[0], key=lambda cand: cand.local_score)
        selected = (int(best.raw_bin),)
        final_score = float(best.local_score)
        final_size = len(candidates[0])
    else:
        states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for i, cand0 in enumerate(candidates[0]):
            for j, cand1 in enumerate(candidates[1]):
                penalty = _transition_penalty(
                    None,
                    cand0,
                    cand1,
                    cfg,
                    reference_slope=header_slope,
                    apply_reference_slope=int(cfg.header_slope_span) >= 2,
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
                        reference_slope=header_slope,
                        apply_reference_slope=bool(header_slope is not None and t < int(cfg.header_slope_span)),
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
            )
            curr = _ViterbiCandidate(
                raw_bin=int(raw_bin),
                phase=float(np.angle(ev.center_spectrum[int(raw_bin)])),
                local_score=0.0,
                energy_score=0.0,
                coherence_score=0.0,
                rank_score=0.0,
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
                )
            phase_penalties.append(_transition_penalty(prevprev, prev, curr, cfg))
    mean_amp = float(np.mean(amp_scores)) if amp_scores else 0.0
    mean_penalty = float(np.mean(phase_penalties)) if phase_penalties else 0.0
    mean_phase = float(math.exp(-mean_penalty)) if math.isfinite(mean_penalty) else 0.0
    return SymbolPhaseResult(
        selected_raw_bins=selected,
        locked_mask=tuple(False for _ in evidences),
        evidences=evidences,
        phase_line=line,
        trajectory_score=float(final_score / max(1, len(selected))),
        line_score=0.0,
        mean_phase_score=float(mean_phase),
        mean_amp_score=float(mean_amp),
        uncertain_count=int(len(evidences)),
        locked_count=0,
        beam_final_size=int(final_size),
        error="",
    )


__all__ = [
    "PhaseLineSelectorConfig",
    "PhasePathSelectorConfig",
    "select_phase_smooth_path",
    "select_phase_viterbi_path",
]

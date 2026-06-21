"""Savaux-style Stage-1 evidence for phase-line FFT-bin selection.

This module keeps the implementation local to ``weak_decoder.phase_line`` while
reusing the paper OSR spectrum primitive from the existing baseline package.
It deliberately stops at raw FFT-bin selection: no codec search, CRC feedback,
payload template, or cross-packet prior is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from ..baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)
from ..candidate_pruning import top_bins
from ..phase_guided_demod import PhaseLine
from ..symbol_phase_two_stage import SymbolPhaseResult
from .configs import PhasePathSelectorConfig
from .selector import select_phase_viterbi_path
from .trajectory import fit_selected_phase_line


CfoCorrectionMode = Literal["none", "symbol", "continuous"]


@dataclass(frozen=True)
class SavauxStage1Config:
    """Parameters for synchronized oversampled Stage-1 evidence."""

    cfo_correction_mode: CfoCorrectionMode = "continuous"
    origin_shift_samples: int | None = None
    top_k: int = 24
    branch_agreement_power: float = 0.0
    normalize_power: bool = False


@dataclass(frozen=True)
class SavauxSymbolEvidence:
    """Per-symbol Savaux Stage-1 observation."""

    symbol_index: int
    start_sample: int
    paper_start_sample: int
    abs_symbol_index: float
    combined_spectrum: np.ndarray
    evidence_power: np.ndarray
    branch_phase_agreement: np.ndarray
    branch_spectra: tuple[np.ndarray, ...]
    top_bins: tuple[int, ...]
    top1_bin: int
    top1_margin_db: float
    top1_peak_to_median_db: float


@dataclass(frozen=True)
class SavauxStage1PacketEvidence:
    """Payload-level Stage-1 evidence ready for phase-path selection."""

    symbols: tuple[SavauxSymbolEvidence, ...]
    center_spectra: tuple[np.ndarray, ...]
    evidence_powers: tuple[np.ndarray, ...]
    abs_indices: tuple[float, ...]
    branch_phase_agreements: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SavauxPhaseGuardConfig:
    """Runtime guard for accepting phase-line corrections over Savaux hard bins."""

    enabled: bool = True
    min_changed_symbols: int = 1
    max_changed_symbols: int = 2
    min_eval_rmse_gain_pi: float = 0.0
    min_mean_phase_score: float = 0.990
    min_changed_mean_energy_drop_db: float = -2.0
    line_trim_frac: float = 0.20


@dataclass(frozen=True)
class SavauxPhaseGuardDecision:
    """Decision and diagnostics for guarded phase-line takeover."""

    accept_phase: bool
    guarded_bins: tuple[int, ...]
    phase_bins: tuple[int, ...]
    hard_bins: tuple[int, ...]
    reason: str
    change_count: int
    eval_rmse_gain_pi: float
    hard_eval_rmse_pi: float
    phase_eval_rmse_pi: float
    changed_mean_energy_drop_db: float
    mean_phase_score: float


def default_savaux_phase_path_config(top_l: int = 16) -> PhasePathSelectorConfig:
    """Return the conservative phase selector used with Savaux Stage 1."""

    return PhasePathSelectorConfig(
        top_l=int(top_l),
        phase_order=1,
        energy_weight=0.70,
        coherence_weight=0.20,
        rank_weight=0.0,
        first_order_weight=0.08,
        second_order_weight=0.0,
        hard_anchor_top_k=1,
        hard_anchor_margin_db=0.40,
        hard_anchor_peak_to_median_db=7.0,
        hard_anchor_min_coherence=0.80,
        high_confidence_top_k=1,
        high_confidence_margin_db=1.20,
        high_confidence_peak_to_median_db=9.0,
        high_confidence_min_coherence=0.80,
        sliding_window_refine_enabled=False,
        path_arbiter_enabled=False,
        phase_proposal_enabled=False,
    )


def _line_eval_rmse_pi(
    center_spectra: Sequence[np.ndarray],
    selected_bins: Sequence[int],
    abs_indices: Sequence[float],
    line: PhaseLine,
) -> float:
    residuals: list[float] = []
    count = min(len(center_spectra), len(selected_bins), len(abs_indices))
    for idx in range(count):
        spectrum = np.asarray(center_spectra[idx], dtype=np.complex64)
        raw_bin = int(selected_bins[idx])
        if raw_bin < 0 or raw_bin >= spectrum.size:
            continue
        observed = float(np.angle(spectrum[raw_bin]))
        predicted = float(line.predict(float(abs_indices[idx])))
        residuals.append(float(np.angle(np.exp(1j * (observed - predicted)))) / math.pi)
    if not residuals:
        return float("inf")
    values = np.asarray(residuals, dtype=np.float64)
    return float(math.sqrt(float(np.mean(values * values))))


def _selected_energy_drop_db(
    powers: Sequence[np.ndarray],
    hard_bins: Sequence[int],
    phase_bins: Sequence[int],
) -> tuple[float, int]:
    drops: list[float] = []
    count = min(len(powers), len(hard_bins), len(phase_bins))
    for idx in range(count):
        hard_bin = int(hard_bins[idx])
        phase_bin = int(phase_bins[idx])
        if hard_bin == phase_bin:
            continue
        power = np.asarray(powers[idx], dtype=np.float64)
        if hard_bin < 0 or hard_bin >= power.size or phase_bin < 0 or phase_bin >= power.size:
            continue
        drops.append(_db_ratio(float(power[phase_bin]), float(power[hard_bin])))
    if not drops:
        return 0.0, 0
    return float(np.mean(drops)), int(len(drops))


def _guard_failure_reason(
    change_count: int,
    rmse_gain: float,
    changed_energy_drop_db: float,
    mean_phase_score: float,
    config: SavauxPhaseGuardConfig,
) -> str:
    if int(change_count) < int(config.min_changed_symbols):
        return "no_phase_change"
    if int(change_count) > int(config.max_changed_symbols):
        return "too_many_changes"
    if float(rmse_gain) < float(config.min_eval_rmse_gain_pi):
        return "no_rmse_gain"
    if float(mean_phase_score) < float(config.min_mean_phase_score):
        return "low_phase_score"
    if float(changed_energy_drop_db) < float(config.min_changed_mean_energy_drop_db):
        return "energy_drop"
    return "accepted"


def evaluate_savaux_phase_guard(
    stage1: SavauxStage1PacketEvidence,
    phase_result: SymbolPhaseResult,
    hard_bins: Sequence[int] | None = None,
    config: SavauxPhaseGuardConfig | None = None,
) -> SavauxPhaseGuardDecision:
    """Return a runtime-only guard decision for phase-line corrections."""

    cfg = config or SavauxPhaseGuardConfig()
    phase_bins = tuple(int(v) for v in phase_result.selected_raw_bins)
    if hard_bins is None:
        hard = tuple(int(symbol.top1_bin) for symbol in stage1.symbols)
    else:
        hard = tuple(int(v) for v in hard_bins)
    count = min(len(hard), len(phase_bins), len(stage1.center_spectra), len(stage1.abs_indices))
    hard = hard[:count]
    phase_bins = phase_bins[:count]
    change_count = int(sum(int(a) != int(b) for a, b in zip(hard, phase_bins)))

    hard_line = fit_selected_phase_line(
        center_spectra=stage1.center_spectra[:count],
        selected_bins=hard,
        abs_indices=stage1.abs_indices[:count],
        trim_frac=float(cfg.line_trim_frac),
    )
    phase_line = fit_selected_phase_line(
        center_spectra=stage1.center_spectra[:count],
        selected_bins=phase_bins,
        abs_indices=stage1.abs_indices[:count],
        trim_frac=float(cfg.line_trim_frac),
    )
    hard_eval_rmse = _line_eval_rmse_pi(stage1.center_spectra[:count], hard, stage1.abs_indices[:count], hard_line)
    phase_eval_rmse = _line_eval_rmse_pi(
        stage1.center_spectra[:count], phase_bins, stage1.abs_indices[:count], phase_line
    )
    rmse_gain = float(hard_eval_rmse - phase_eval_rmse)
    changed_energy_drop_db, changed_count = _selected_energy_drop_db(stage1.evidence_powers[:count], hard, phase_bins)
    if changed_count <= 0:
        changed_energy_drop_db = 0.0
    reason = "accepted" if not bool(cfg.enabled) else _guard_failure_reason(
        change_count=change_count,
        rmse_gain=rmse_gain,
        changed_energy_drop_db=changed_energy_drop_db,
        mean_phase_score=float(phase_result.mean_phase_score),
        config=cfg,
    )
    accept = bool((not bool(cfg.enabled)) or reason == "accepted")
    return SavauxPhaseGuardDecision(
        accept_phase=accept,
        guarded_bins=phase_bins if accept else hard,
        phase_bins=phase_bins,
        hard_bins=hard,
        reason=reason,
        change_count=int(change_count),
        eval_rmse_gain_pi=float(rmse_gain),
        hard_eval_rmse_pi=float(hard_eval_rmse),
        phase_eval_rmse_pi=float(phase_eval_rmse),
        changed_mean_energy_drop_db=float(changed_energy_drop_db),
        mean_phase_score=float(phase_result.mean_phase_score),
    )


def _db_ratio(numerator: float, denominator: float) -> float:
    return float(10.0 * math.log10((float(numerator) + 1e-30) / (float(denominator) + 1e-30)))


def _validate_os_factor(os_factor: int) -> int:
    value = int(os_factor)
    if value <= 0:
        raise ValueError(f"os_factor must be positive, got {os_factor}")
    return value


def savaux_branch_phase_agreement(
    branch_spectra: Sequence[np.ndarray],
    os_factor: int,
) -> np.ndarray:
    """Return Eq. (37)-aligned branch phase agreement for every raw FFT bin."""

    spectra = [np.asarray(item, dtype=np.complex64) for item in branch_spectra]
    if not spectra:
        raise ValueError("at least one branch spectrum is required")
    n_bins = int(spectra[0].size)
    if n_bins <= 0 or any(item.size != n_bins for item in spectra):
        raise ValueError("all branch spectra must have the same non-zero length")
    os_value = _validate_os_factor(os_factor)
    if len(spectra) != os_value:
        raise ValueError(f"got {len(spectra)} branch spectra, expected {os_value}")

    bins = np.arange(n_bins, dtype=np.float64)
    aligned = np.empty((os_value, n_bins), dtype=np.complex128)
    for q, spectrum in enumerate(spectra):
        weight = np.exp(-2j * np.pi * float(q) * bins / float(n_bins * os_value))
        aligned[q, :] = spectrum.astype(np.complex128) * weight
    agreement = np.abs(np.sum(aligned, axis=0)) / (np.sum(np.abs(aligned), axis=0) + 1e-30)
    return np.clip(agreement.astype(np.float64), 0.0, 1.0)


def _power_from_combined(
    combined_spectrum: np.ndarray,
    agreement: np.ndarray,
    config: SavauxStage1Config,
) -> np.ndarray:
    power = np.abs(np.asarray(combined_spectrum, dtype=np.complex64)).astype(np.float64) ** 2
    if float(config.branch_agreement_power) > 0.0:
        if agreement.size != power.size:
            raise ValueError("branch agreement and spectrum length mismatch")
        power = power * np.power(np.clip(agreement, 1e-6, 1.0), float(config.branch_agreement_power))
    if bool(config.normalize_power):
        max_power = float(np.max(power)) if power.size else 0.0
        if max_power > 0.0:
            power = power / max_power
    return power.astype(np.float64, copy=False)


def build_savaux_symbol_evidence(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    abs_symbol_index: float,
    symbol_index: int = 0,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    config: SavauxStage1Config | None = None,
) -> SavauxSymbolEvidence:
    """Build one synchronized Savaux Stage-1 symbol observation."""

    cfg = config or SavauxStage1Config()
    os_value = _validate_os_factor(os_factor)
    origin_shift = int(os_value // 2 if cfg.origin_shift_samples is None else cfg.origin_shift_samples)
    paper_start = int(start_sample) + origin_shift
    paper_header_start = None if header_start_sample is None else int(header_start_sample) + origin_shift
    combined, branches, _common_phase = paper_oversampled_spectrum(
        samples=np.asarray(samples, dtype=np.complex64),
        start_sample=paper_start,
        sf=int(sf),
        os_factor=os_value,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=paper_header_start,
        cfo_correction_mode=str(cfg.cfo_correction_mode),
    )
    agreement = savaux_branch_phase_agreement(branches, os_factor=os_value)
    power = _power_from_combined(combined, agreement, cfg)
    bins = top_bins(power, min(max(1, int(cfg.top_k)), power.size))
    top1 = int(bins[0]) if bins.size else 0
    top2 = int(bins[1]) if bins.size > 1 else top1
    top1_power = float(power[top1]) if 0 <= top1 < power.size else 0.0
    top2_power = float(power[top2]) if 0 <= top2 < power.size else 0.0
    median_power = float(np.median(power)) if power.size else 0.0
    return SavauxSymbolEvidence(
        symbol_index=int(symbol_index),
        start_sample=int(start_sample),
        paper_start_sample=int(paper_start),
        abs_symbol_index=float(abs_symbol_index),
        combined_spectrum=np.asarray(combined, dtype=np.complex64),
        evidence_power=power,
        branch_phase_agreement=agreement,
        branch_spectra=tuple(np.asarray(item, dtype=np.complex64) for item in branches),
        top_bins=tuple(int(v) for v in bins),
        top1_bin=top1,
        top1_margin_db=_db_ratio(top1_power, top2_power),
        top1_peak_to_median_db=_db_ratio(top1_power, median_power),
    )


def build_savaux_stage1_packet_evidence(
    samples: np.ndarray,
    start_samples: Sequence[int],
    sf: int,
    os_factor: int,
    abs_indices: Sequence[float],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    config: SavauxStage1Config | None = None,
) -> SavauxStage1PacketEvidence:
    """Build payload evidence arrays accepted by ``select_phase_viterbi_path``."""

    cfg = config or SavauxStage1Config()
    count = min(len(start_samples), len(abs_indices))
    symbols: list[SavauxSymbolEvidence] = []
    for idx in range(count):
        symbols.append(
            build_savaux_symbol_evidence(
                samples=samples,
                start_sample=int(start_samples[idx]),
                sf=int(sf),
                os_factor=int(os_factor),
                abs_symbol_index=float(abs_indices[idx]),
                symbol_index=int(idx),
                cfo_int=int(cfo_int),
                cfo_frac=float(cfo_frac),
                header_start_sample=header_start_sample,
                config=cfg,
            )
        )
    return SavauxStage1PacketEvidence(
        symbols=tuple(symbols),
        center_spectra=tuple(item.combined_spectrum for item in symbols),
        evidence_powers=tuple(item.evidence_power for item in symbols),
        abs_indices=tuple(item.abs_symbol_index for item in symbols),
        branch_phase_agreements=tuple(item.branch_phase_agreement for item in symbols),
    )


def payload_abs_indices(
    payload_symbol_indexes: Sequence[int],
    preamble_len: float = 8.0,
) -> tuple[float, ...]:
    """Return the absolute symbol indexes used by the existing phase-line tests."""

    return tuple(float(preamble_len) + 12.25 + float(idx) for idx in payload_symbol_indexes)


def select_savaux_phase_viterbi_path(
    samples: np.ndarray,
    start_samples: Sequence[int],
    sf: int,
    os_factor: int,
    abs_indices: Sequence[float],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    stage1_config: SavauxStage1Config | None = None,
    selector_config: PhasePathSelectorConfig | None = None,
    fallback_line: PhaseLine | None = None,
) -> SymbolPhaseResult:
    """Run Savaux Stage 1 followed by the phase-line Viterbi selector."""

    stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        config=stage1_config,
    )
    return select_phase_viterbi_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=selector_config or default_savaux_phase_path_config(),
        fallback_line=fallback_line,
        offset_coherences=stage1.branch_phase_agreements,
    )


__all__ = [
    "SavauxPhaseGuardConfig",
    "SavauxPhaseGuardDecision",
    "SavauxStage1Config",
    "SavauxStage1PacketEvidence",
    "SavauxSymbolEvidence",
    "build_savaux_stage1_packet_evidence",
    "build_savaux_symbol_evidence",
    "default_savaux_phase_path_config",
    "evaluate_savaux_phase_guard",
    "payload_abs_indices",
    "savaux_branch_phase_agreement",
    "select_savaux_phase_viterbi_path",
]

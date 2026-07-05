"""为 phase-line 峰选择准备 Savaux 风格的第一阶段证据。

这个模块放在 ``weak_decoder.phase_line`` 内部实现，同时复用 baseline
目录里已经实现的论文版过采样频谱计算。这里刻意只做到 raw FFT bin
选择，不做 codec 搜索、CRC 反馈、payload 模板匹配，也不使用跨包先验。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from ..baselines.savaux_oversampled.paper_oversampled_demod import (
    _oversampled_downchirp,
    paper_oversampled_spectrum,
)
from ..candidate_pruning import top_bins
from ..phase_guided_demod import PhaseLine
from ..symbol_phase_two_stage import SymbolPhaseResult
from .configs import PhasePathSelectorConfig
from .selector import (
    select_anchor_bounded_bidirectional_rerank_path,
    select_anchor_bounded_island_viterbi_path,
    select_phase_viterbi_path,
)
from .stage2_arbiter import select_v1_risk_arbiter_path
from .stage2_variants import (
    Stage2VariantDecision,
    select_adaptive_aggressive_island_viterbi_path,
    select_rewrite_island_viterbi_path,
)
from .trajectory import fit_selected_phase_line
from .variants.island_dp_reconstruction import (
    IslandReconstructionConfig,
    select_island_reconstruction_viterbi_path,
)


CfoCorrectionMode = Literal["none", "symbol", "continuous"]


@dataclass(frozen=True)
class SavauxStage1Config:
    """同步后的过采样第一阶段证据参数。"""

    cfo_correction_mode: CfoCorrectionMode = "continuous"
    origin_shift_samples: int | None = None
    top_k: int = 24
    branch_agreement_power: float = 0.0
    normalize_power: bool = False
    residual_sto_phase_correction: bool = False
    residual_sto_phase_sign: Literal["plus", "minus"] = "plus"
    residual_sto_preselect_factor: int = 4
    residual_sto_update_power: bool = False
    retain_dechirped_symbols: bool = False


@dataclass(frozen=True)
class SavauxSymbolEvidence:
    """单个 symbol 的 Savaux 第一阶段观测结果。"""

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
    dechirped_oversampled: np.ndarray | None = None


@dataclass(frozen=True)
class SavauxStage1PacketEvidence:
    """整段 payload 的第一阶段证据，供后续 phase-line 路径选择使用。"""

    symbols: tuple[SavauxSymbolEvidence, ...]
    center_spectra: tuple[np.ndarray, ...]
    evidence_powers: tuple[np.ndarray, ...]
    abs_indices: tuple[float, ...]
    branch_phase_agreements: tuple[np.ndarray, ...]
    branch_spectra: tuple[tuple[np.ndarray, ...], ...]
    dechirped_symbols: tuple[np.ndarray | None, ...]


@dataclass(frozen=True)
class SavauxPhaseGuardConfig:
    """运行时 guard 参数，用来判断是否接受 phase-line 对 Savaux hard bin 的修正。"""

    enabled: bool = True
    min_changed_symbols: int = 1
    max_changed_symbols: int = 2
    min_eval_rmse_gain_pi: float = 0.0
    min_mean_phase_score: float = 0.990
    min_changed_mean_energy_drop_db: float = -2.0
    line_trim_frac: float = 0.20


@dataclass(frozen=True)
class SavauxPhaseGuardDecision:
    """guard 判断结果以及相关诊断信息。"""

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
    """返回配合 Savaux 第一阶段使用的保守 phase-line 选择器配置。"""

    return PhasePathSelectorConfig(
        top_l=int(top_l),
        phase_order=1,
        energy_weight=0.65,
        coherence_weight=0.15,
        phase_local_weight=0.15,
        rank_weight=0.0,
        first_order_weight=0.08,
        second_order_weight=0.0,
        top1_soft_bonus=0.0,
        hard_anchor_top_k=1,
        hard_anchor_margin_db=2.50,
        hard_anchor_peak_to_median_db=12.0,
        hard_anchor_min_coherence=0.90,
        hard_anchor_soft_top_k=4,
        hard_anchor_soft_max_margin_db=99.0,
        hard_anchor_soft_max_peak_to_median_db=99.0,
        high_confidence_top_k=4,
        high_confidence_margin_db=1.20,
        high_confidence_peak_to_median_db=9.0,
        high_confidence_min_coherence=0.80,
        anchor_phase_bias_weight=0.16,
        anchor_phase_bias_span=10.0,
        anchor_phase_bias_min_anchors=4,
        anchor_phase_bias_scale_pi=0.32,
        anchor_phase_bias_max_rmse_pi=0.40,
        anchor_phase_bias_trim_frac=0.25,
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
    """根据运行时诊断决定是否采纳 phase-line 的修正结果。"""

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


def _wrap_half(value: float) -> float:
    return float((float(value) + 0.5) % 1.0 - 0.5)


def _prepare_dechirped_oversampled_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    header_start_sample: int | None,
    cfo_correction_mode: CfoCorrectionMode,
) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    start = int(start_sample)
    stop = start + n_bins * os_value
    if start < 0 or stop > np.asarray(samples).size:
        raise ValueError(f"symbol at {start_sample} exceeds input sample range")

    mode = str(cfo_correction_mode)
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
        sf=sf,
        os_factor=os_value,
        cfo_int=use_cfo_int,
        cfo_frac=use_cfo_frac,
    )
    return (symbol * downchirp).astype(np.complex64)


def _sto_phase_corrected_bin_value(
    dechirped: np.ndarray,
    raw_bin: int,
    os_factor: int,
    residual_sto_chips: float,
    phase_sign: Literal["plus", "minus"],
) -> complex:
    os_value = _validate_os_factor(os_factor)
    full = np.asarray(dechirped, dtype=np.complex64)
    if full.size % os_value != 0:
        raise ValueError("dechirped symbol length must be divisible by os_factor")

    n_bins = int(full.size // os_value)
    k = int(raw_bin) % n_bins
    tau = _wrap_half(float(residual_sto_chips))
    sign_value = 1.0 if str(phase_sign) == "plus" else -1.0
    residual_phase = complex(np.exp(1j * sign_value * 2.0 * math.pi * tau))
    p = np.arange(n_bins, dtype=np.float64)
    kernel = np.exp(-2j * np.pi * float(k) * p / float(n_bins))

    combined = 0.0j
    for q in range(os_value):
        branch = np.asarray(full[q::os_value], dtype=np.complex128)
        values = branch * kernel
        if k != 0:
            base_tail = p >= float(n_bins - k)
            if np.any(base_tail):
                values[base_tail] *= np.exp(2j * np.pi * float(q) / float(os_value))
        if abs(tau) > 1e-12:
            cut = int(math.ceil(float(n_bins - k) + tau - float(q) / float(os_value)))
            cut = max(0, min(n_bins, cut))
            if cut < n_bins:
                values[cut:] *= residual_phase
        branch_value = complex(np.sum(values, dtype=np.complex128) / math.sqrt(float(n_bins)))
        branch_weight = complex(np.exp(-2j * np.pi * float(q * k) / float(n_bins * os_value)))
        combined += branch_weight * branch_value
    return complex(combined)


def savaux_branch_phase_agreement(
    branch_spectra: Sequence[np.ndarray],
    os_factor: int,
) -> np.ndarray:
    """计算每个 raw FFT bin 在 Eq. (37) 相位对齐后的 branch 相位一致性。"""

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
    residual_sto_chips: float = 0.0,
    config: SavauxStage1Config | None = None,
) -> SavauxSymbolEvidence:
    """构建一个同步后的 Savaux 第一阶段 symbol 观测。"""

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
    retained_dechirped: np.ndarray | None = None
    if bool(cfg.residual_sto_phase_correction) or bool(cfg.retain_dechirped_symbols):
        dechirped = _prepare_dechirped_oversampled_symbol(
            samples=np.asarray(samples, dtype=np.complex64),
            start_sample=paper_start,
            sf=int(sf),
            os_factor=os_value,
            cfo_int=int(cfo_int),
            cfo_frac=float(cfo_frac),
            header_start_sample=paper_header_start,
            cfo_correction_mode=cfg.cfo_correction_mode,
        )
        if bool(cfg.retain_dechirped_symbols):
            retained_dechirped = np.asarray(dechirped, dtype=np.complex64)
    if bool(cfg.residual_sto_phase_correction):
        original_power = np.asarray(power, dtype=np.float64).copy()
        preselect = min(
            max(1, int(cfg.top_k) * max(1, int(cfg.residual_sto_preselect_factor))),
            int(power.size),
        )
        candidate_bins = top_bins(power, preselect)
        corrected = np.asarray(combined, dtype=np.complex64).copy()
        for raw_bin in candidate_bins:
            b = int(raw_bin)
            corrected[b] = np.complex64(
                _sto_phase_corrected_bin_value(
                    dechirped=dechirped,
                    raw_bin=b,
                    os_factor=os_value,
                    residual_sto_chips=float(residual_sto_chips),
                    phase_sign=cfg.residual_sto_phase_sign,
                )
            )
        combined = corrected
        if bool(cfg.residual_sto_update_power):
            power = _power_from_combined(combined, agreement, cfg)
        else:
            power = original_power
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
        dechirped_oversampled=retained_dechirped,
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
    residual_sto_chips: Sequence[float] | None = None,
    config: SavauxStage1Config | None = None,
) -> SavauxStage1PacketEvidence:
    """构建 ``select_phase_viterbi_path`` 需要的 payload 证据数组。"""

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
                residual_sto_chips=(
                    float(residual_sto_chips[idx])
                    if residual_sto_chips is not None and idx < len(residual_sto_chips)
                    else 0.0
                ),
                config=cfg,
            )
        )
    return SavauxStage1PacketEvidence(
        symbols=tuple(symbols),
        center_spectra=tuple(item.combined_spectrum for item in symbols),
        evidence_powers=tuple(item.evidence_power for item in symbols),
        abs_indices=tuple(item.abs_symbol_index for item in symbols),
        branch_phase_agreements=tuple(item.branch_phase_agreement for item in symbols),
        branch_spectra=tuple(item.branch_spectra for item in symbols),
        dechirped_symbols=tuple(item.dechirped_oversampled for item in symbols),
    )


def payload_abs_indices(
    payload_symbol_indexes: Sequence[int],
    preamble_len: float = 8.0,
) -> tuple[float, ...]:
    """返回现有 phase-line 测试使用的绝对 symbol 序号。"""

    return tuple(float(preamble_len) + 12.25 + float(idx) for idx in payload_symbol_indexes)


def branch_residual_sto_chips_from_sync_estimates(
    branch_sync_estimates: Sequence[object],
    payload_symbol_indexes: Sequence[int],
    os_factor: int,
    header_count: int = 8,
) -> tuple[tuple[float, ...], ...]:
    """Expand per-sampling-phase framesync offsets to payload-symbol rows.

    ``grlora_frame_sync`` obtains one timing estimate per downsample phase by
    rerunning the framesync estimation path with ``sample_phase=0..R-1``.  The
    island JTRD selector needs the instantaneous residual STO for every payload
    symbol and every branch, so this helper mirrors the SFO cursor update used
    by ``header_first_demod.advance_symbol_cursor``.
    """

    os_value = _validate_os_factor(os_factor)
    estimates = tuple(branch_sync_estimates)
    if not estimates:
        return ()
    max_payload_index = max((int(v) for v in payload_symbol_indexes), default=-1)
    if max_payload_index < 0:
        return ()

    branch_values: list[list[float]] = []
    threshold = 1.0 / (2.0 * os_value)
    total_symbols = int(header_count) + int(max_payload_index) + 1
    for item in estimates:
        sfo_cum = float(getattr(item, "sfo_cum_initial", 0.0))
        sfo_hat = float(getattr(item, "sfo_hat", 0.0))
        per_payload: list[float] = []
        wanted = {int(idx): pos for pos, idx in enumerate(payload_symbol_indexes)}
        values = [0.0 for _ in payload_symbol_indexes]
        for frame_symbol_index in range(total_symbols):
            payload_index = int(frame_symbol_index) - int(header_count)
            if payload_index in wanted:
                values[wanted[payload_index]] = float(sfo_cum)
            if abs(float(sfo_cum)) > threshold:
                sign = -1 if math.copysign(1.0, float(sfo_cum)) < 0.0 else 1
                sfo_cum -= sign * (1.0 / os_value)
            sfo_cum += sfo_hat
        per_payload.extend(values)
        branch_values.append(per_payload)

    rows: list[tuple[float, ...]] = []
    for symbol_pos in range(len(payload_symbol_indexes)):
        row = [
            float(branch_values[branch][symbol_pos])
            for branch in range(min(os_value, len(branch_values)))
        ]
        if len(row) < os_value:
            row.extend([row[-1] if row else 0.0] * (os_value - len(row)))
        rows.append(tuple(row[:os_value]))
    return tuple(rows)


def select_savaux_phase_viterbi_path(
    samples: np.ndarray,
    start_samples: Sequence[int],
    sf: int,
    os_factor: int,
    abs_indices: Sequence[float],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    residual_sto_chips: Sequence[float] | None = None,
    stage1_config: SavauxStage1Config | None = None,
    selector_config: PhasePathSelectorConfig | None = None,
    fallback_line: PhaseLine | None = None,
) -> SymbolPhaseResult:
    """先运行 Savaux 第一阶段，再运行 phase-line 路径选择器。"""

    stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
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


def select_savaux_bidirectional_rerank_path(
    samples: np.ndarray,
    start_samples: Sequence[int],
    sf: int,
    os_factor: int,
    abs_indices: Sequence[float],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    residual_sto_chips: Sequence[float] | None = None,
    stage1_config: SavauxStage1Config | None = None,
    selector_config: PhasePathSelectorConfig | None = None,
    fallback_line: PhaseLine | None = None,
) -> SymbolPhaseResult:
    """Run Savaux Stage-1, then bidirectional anchor-bounded reranking."""

    stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
        config=stage1_config,
    )
    return select_anchor_bounded_bidirectional_rerank_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=selector_config or default_savaux_phase_path_config(),
        fallback_line=fallback_line,
        offset_coherences=stage1.branch_phase_agreements,
    )


def select_savaux_island_viterbi_path(
    samples: np.ndarray,
    start_samples: Sequence[int],
    sf: int,
    os_factor: int,
    abs_indices: Sequence[float],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    residual_sto_chips: Sequence[float] | None = None,
    stage1_config: SavauxStage1Config | None = None,
    selector_config: PhasePathSelectorConfig | None = None,
    fallback_line: PhaseLine | None = None,
) -> SymbolPhaseResult:
    """Run Savaux Stage-1, then hard-anchor island-bounded Viterbi."""

    stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
        config=stage1_config,
    )
    return select_anchor_bounded_island_viterbi_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=selector_config or default_savaux_phase_path_config(),
        fallback_line=fallback_line,
        offset_coherences=stage1.branch_phase_agreements,
    )


def select_savaux_adaptive_aggressive_island_viterbi_path(
    samples: np.ndarray,
    start_samples: Sequence[int],
    sf: int,
    os_factor: int,
    abs_indices: Sequence[float],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    residual_sto_chips: Sequence[float] | None = None,
    stage1_config: SavauxStage1Config | None = None,
    selector_config: PhasePathSelectorConfig | None = None,
    fallback_line: PhaseLine | None = None,
) -> tuple[SymbolPhaseResult, Stage2VariantDecision]:
    """Run Savaux Stage-1, then adaptive stable/aggressive island Viterbi."""

    del fallback_line
    stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
        config=stage1_config,
    )
    return select_adaptive_aggressive_island_viterbi_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=selector_config or default_savaux_phase_path_config(),
        offset_coherences=stage1.branch_phase_agreements,
    )


def select_savaux_rewrite_island_viterbi_path(
    samples: np.ndarray,
    start_samples: Sequence[int],
    sf: int,
    os_factor: int,
    abs_indices: Sequence[float],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    residual_sto_chips: Sequence[float] | None = None,
    stage1_config: SavauxStage1Config | None = None,
    selector_config: PhasePathSelectorConfig | None = None,
    fallback_line: PhaseLine | None = None,
) -> SymbolPhaseResult:
    """Run Savaux Stage-1, then permissive rewrite island Viterbi."""

    del fallback_line
    stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
        config=stage1_config,
    )
    return select_rewrite_island_viterbi_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=selector_config or default_savaux_phase_path_config(),
        offset_coherences=stage1.branch_phase_agreements,
    )


def select_savaux_v1_risk_arbiter_path(
    samples: np.ndarray,
    start_samples: Sequence[int],
    sf: int,
    os_factor: int,
    abs_indices: Sequence[float],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    residual_sto_chips: Sequence[float] | None = None,
    stage1_config: SavauxStage1Config | None = None,
    selector_config: PhasePathSelectorConfig | None = None,
    fallback_line: PhaseLine | None = None,
) -> tuple[SymbolPhaseResult, Stage2VariantDecision]:
    """Run Savaux Stage-1, then v1 trajectory-risk arbitration."""

    del fallback_line
    stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
        config=stage1_config,
    )
    return select_v1_risk_arbiter_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=selector_config or default_savaux_phase_path_config(),
        offset_coherences=stage1.branch_phase_agreements,
    )


def select_savaux_island_reconstruction_viterbi_path(
    samples: np.ndarray,
    start_samples: Sequence[int],
    sf: int,
    os_factor: int,
    abs_indices: Sequence[float],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    residual_sto_chips: Sequence[float] | None = None,
    branch_residual_sto_chips: Sequence[Sequence[float]] | None = None,
    branch_sync_estimates: Sequence[object] | None = None,
    payload_symbol_indexes: Sequence[int] | None = None,
    stage1_config: SavauxStage1Config | None = None,
    selector_config: PhasePathSelectorConfig | None = None,
    reconstruction_config: IslandReconstructionConfig | None = None,
) -> SymbolPhaseResult:
    """Run Savaux Stage-1, then experimental island reconstruction DP."""

    cfg = stage1_config or SavauxStage1Config(retain_dechirped_symbols=True)
    stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
        config=cfg,
    )
    path_cfg = selector_config or default_savaux_phase_path_config()
    branch_residual = branch_residual_sto_chips
    if branch_residual is None and branch_sync_estimates is not None:
        indexes = tuple(
            int(v)
            for v in (
                payload_symbol_indexes
                if payload_symbol_indexes is not None
                else range(len(stage1.abs_indices))
            )
        )
        branch_residual = branch_residual_sto_chips_from_sync_estimates(
            branch_sync_estimates,
            indexes,
            os_factor=int(os_factor),
        )
    return select_island_reconstruction_viterbi_path(
        center_spectra=stage1.center_spectra,
        evidence_powers=stage1.evidence_powers,
        abs_indices=stage1.abs_indices,
        config=path_cfg,
        reconstruction_config=reconstruction_config,
        offset_coherences=stage1.branch_phase_agreements,
        branch_spectra=stage1.branch_spectra,
        dechirped_symbols=stage1.dechirped_symbols,
        os_factor=int(os_factor),
        residual_sto_chips=residual_sto_chips,
        branch_residual_sto_chips=branch_residual,
    )


__all__ = [
    "SavauxPhaseGuardConfig",
    "SavauxPhaseGuardDecision",
    "SavauxStage1Config",
    "SavauxStage1PacketEvidence",
    "SavauxSymbolEvidence",
    "build_savaux_stage1_packet_evidence",
    "build_savaux_symbol_evidence",
    "branch_residual_sto_chips_from_sync_estimates",
    "default_savaux_phase_path_config",
    "evaluate_savaux_phase_guard",
    "payload_abs_indices",
    "savaux_branch_phase_agreement",
    "select_savaux_adaptive_aggressive_island_viterbi_path",
    "select_savaux_bidirectional_rerank_path",
    "select_savaux_island_viterbi_path",
    "select_savaux_island_reconstruction_viterbi_path",
    "select_savaux_phase_viterbi_path",
    "select_savaux_rewrite_island_viterbi_path",
    "select_savaux_v1_risk_arbiter_path",
]

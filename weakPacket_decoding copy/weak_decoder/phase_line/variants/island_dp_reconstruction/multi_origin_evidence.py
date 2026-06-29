"""Multi-origin Stage-1 FFT-bin evidence fusion.

This experiment generalizes the original frame-sync downsample mouthfeel:
instead of using only the center sampling phase, build Stage-1 evidence for
origin shifts ``0..os_factor-1`` and fuse their FFT-bin powers.  It remains an
FFT-bin selector feature only; no payload codec or CRC feedback is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np

from ...configs import PhasePathSelectorConfig
from ...savaux_stage1 import (
    SavauxStage1Config,
    SavauxStage1PacketEvidence,
    build_savaux_stage1_packet_evidence,
    default_savaux_phase_path_config,
)
from ....symbol_phase_two_stage import SymbolPhaseResult
from ..v1_one_order_dp import select_phase_viterbi_path
from .selector import IslandReconstructionConfig, select_island_reconstruction_viterbi_path


MultiOriginFusionMode = Literal[
    "sum_norm",
    "product_norm",
    "max_norm",
    "median_norm",
    "top2_mean_norm",
    "trimmed_mean_norm",
]
MultiOriginPhaseMode = Literal["fixed_origin", "weighted_unit", "weighted_complex", "max_bin", "best_symbol_origin"]


@dataclass(frozen=True)
class MultiOriginFusionConfig:
    """Parameters for origin-shift FFT-bin power fusion."""

    mode: MultiOriginFusionMode = "sum_norm"
    phase_mode: MultiOriginPhaseMode = "fixed_origin"
    stage1_top_k: int = 40
    origins: tuple[int, ...] | None = None
    phase_origin: int | None = None
    corrected: bool = False
    residual_sto_phase_sign: Literal["plus", "minus"] = "plus"
    residual_sto_preselect_factor: int = 4
    retain_dechirped_symbols: bool = True


@dataclass(frozen=True)
class MultiOriginGateConfig:
    """Conservative selector gate between dual fusion and multi-origin fusion."""

    enabled: bool = False
    min_trajectory_gain: float = 0.003267
    min_amp_gain: float | None = None
    max_changes_vs_dual: int | None = None
    symbol_gate_enabled: bool = False
    symbol_min_trajectory_gain: float = 0.005618281741596176
    symbol_min_old_power_margin: float = -0.3055904873940537
    symbol_min_old_multi_norm: float | None = None


@dataclass(frozen=True)
class MultiOriginStage1PacketEvidence:
    """Stage-1 evidence with fused powers across sampling origins."""

    origin_stage1: tuple[SavauxStage1PacketEvidence, ...]
    origin_shifts: tuple[int, ...]
    phase_origin: int
    phase_stage1: SavauxStage1PacketEvidence
    center_spectra: tuple[np.ndarray, ...]
    evidence_powers: tuple[np.ndarray, ...]
    abs_indices: tuple[float, ...]
    branch_phase_agreements: tuple[np.ndarray, ...]
    branch_spectra: tuple[tuple[np.ndarray, ...], ...]
    dechirped_symbols: tuple[np.ndarray | None, ...]


@dataclass(frozen=True)
class MultiOriginGateDecision:
    """Decision metadata for dual-vs-multi path arbitration."""

    use_multi: bool
    reason: str
    trajectory_gain: float
    amp_gain: float
    changes_vs_dual: int


@dataclass(frozen=True)
class MultiOriginArbiterResult:
    """Selected path plus the two candidate paths and gate decision."""

    selected: SymbolPhaseResult
    dual: SymbolPhaseResult
    multi: SymbolPhaseResult
    decision: MultiOriginGateDecision


def _normalize_power(power: np.ndarray) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    peak = float(np.max(values))
    if peak <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(values / peak, dtype=np.float64)


def _unit_complex(values: np.ndarray) -> np.ndarray:
    complex_values = np.asarray(values, dtype=np.complex128)
    return complex_values / (np.abs(complex_values) + 1e-30)


def _origin_power_stack(powers: Sequence[np.ndarray]) -> np.ndarray:
    rows = tuple(_normalize_power(np.asarray(power, dtype=np.float64)) for power in powers)
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    shape = rows[0].shape
    if any(row.shape != shape for row in rows):
        raise ValueError("all origin power rows must have the same shape")
    return np.stack(rows, axis=0)


def _symbol_origin_confidence(power: np.ndarray) -> float:
    row = _normalize_power(np.asarray(power, dtype=np.float64))
    if row.size < 2:
        return 0.0
    top2 = np.partition(row, -2)[-2:]
    top1_power = float(top2[1])
    top2_power = float(top2[0])
    median_power = float(np.median(row)) if row.size else 0.0
    margin_db = float(10.0 * np.log10((top1_power + 1e-30) / (top2_power + 1e-30)))
    peak_db = float(10.0 * np.log10((top1_power + 1e-30) / (median_power + 1e-30)))
    return float(margin_db + 0.20 * peak_db)


def fuse_origin_phase_spectrum(
    spectra: Sequence[np.ndarray],
    powers: Sequence[np.ndarray],
    config: MultiOriginFusionConfig | None = None,
) -> np.ndarray:
    """Fuse one symbol's complex spectra across sampling origins for phase DP."""

    cfg = config or MultiOriginFusionConfig()
    mode = str(cfg.phase_mode)
    spectrum_rows = tuple(np.asarray(spectrum, dtype=np.complex128) for spectrum in spectra)
    if not spectrum_rows:
        return np.asarray((), dtype=np.complex64)
    shape = spectrum_rows[0].shape
    if any(row.shape != shape for row in spectrum_rows):
        raise ValueError("all origin spectra must have the same shape")
    if mode == "fixed_origin":
        origin = int(cfg.phase_origin if cfg.phase_origin is not None else len(spectrum_rows) // 2)
        origin = min(len(spectrum_rows) - 1, max(0, origin))
        return np.asarray(spectrum_rows[origin], dtype=np.complex64)

    power_stack = _origin_power_stack(powers)
    if power_stack.shape != (len(spectrum_rows), int(shape[0])):
        raise ValueError("origin spectra and power rows must have matching shapes")
    spectrum_stack = np.stack(spectrum_rows, axis=0)
    if mode == "weighted_unit":
        phase_sum = np.sum(_unit_complex(spectrum_stack) * power_stack, axis=0)
        amp = np.sqrt(np.mean(power_stack, axis=0))
        fused = amp * _unit_complex(phase_sum)
    elif mode == "weighted_complex":
        fused = np.sum(spectrum_stack * power_stack, axis=0) / (np.sum(power_stack, axis=0) + 1e-30)
    elif mode == "max_bin":
        origin_pos = np.argmax(power_stack, axis=0)
        bins = np.arange(int(shape[0]), dtype=np.int64)
        fused = spectrum_stack[origin_pos, bins]
    elif mode == "best_symbol_origin":
        confidences = [_symbol_origin_confidence(power) for power in powers]
        origin = int(np.argmax(np.asarray(confidences, dtype=np.float64)))
        fused = spectrum_rows[origin]
    else:
        raise ValueError(f"unsupported multi-origin phase mode: {cfg.phase_mode!r}")
    return np.asarray(fused, dtype=np.complex64)


def fuse_origin_power_rows(
    powers: Sequence[np.ndarray],
    config: MultiOriginFusionConfig | None = None,
) -> np.ndarray:
    """Fuse one symbol's power rows from multiple sampling origins."""

    cfg = config or MultiOriginFusionConfig()
    stack = _origin_power_stack(powers)
    if stack.size == 0:
        return np.asarray((), dtype=np.float64)
    if str(cfg.mode) == "sum_norm":
        fused = np.mean(stack, axis=0)
    elif str(cfg.mode) == "product_norm":
        fused = np.exp(np.mean(np.log(stack + 1e-30), axis=0))
    elif str(cfg.mode) == "max_norm":
        fused = np.max(stack, axis=0)
    elif str(cfg.mode) == "median_norm":
        fused = np.median(stack, axis=0)
    elif str(cfg.mode) == "top2_mean_norm":
        top_count = min(2, int(stack.shape[0]))
        fused = np.mean(np.sort(stack, axis=0)[-top_count:, :], axis=0)
    elif str(cfg.mode) == "trimmed_mean_norm":
        if int(stack.shape[0]) >= 3:
            ordered = np.sort(stack, axis=0)
            fused = np.mean(ordered[1:-1, :], axis=0)
        else:
            fused = np.mean(stack, axis=0)
    else:
        raise ValueError(f"unsupported multi-origin fusion mode: {cfg.mode!r}")
    peak = float(np.max(fused)) if fused.size else 0.0
    if peak > 0.0:
        fused = fused / peak
    return np.asarray(fused, dtype=np.float64)


def _origin_list(os_factor: int, config: MultiOriginFusionConfig) -> tuple[int, ...]:
    os_value = max(1, int(os_factor))
    if config.origins is None:
        return tuple(range(os_value))
    out: list[int] = []
    seen: set[int] = set()
    for item in config.origins:
        q = int(item)
        if q < 0 or q >= os_value:
            raise ValueError(f"origin shift must be in [0, {os_value}), got {item}")
        if q not in seen:
            out.append(q)
            seen.add(q)
    return tuple(out) if out else tuple(range(os_value))


def _stage1_config_for_origin(
    base_config: SavauxStage1Config | None,
    origin: int,
    config: MultiOriginFusionConfig,
) -> SavauxStage1Config:
    base = base_config or SavauxStage1Config()
    return replace(
        base,
        origin_shift_samples=int(origin),
        top_k=max(1, int(config.stage1_top_k)),
        residual_sto_phase_correction=bool(config.corrected),
        residual_sto_phase_sign=config.residual_sto_phase_sign,
        residual_sto_preselect_factor=max(1, int(config.residual_sto_preselect_factor)),
        residual_sto_update_power=bool(config.corrected),
        retain_dechirped_symbols=bool(config.retain_dechirped_symbols),
    )


def build_multi_origin_stage1_packet_evidence(
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
    fusion_config: MultiOriginFusionConfig | None = None,
) -> MultiOriginStage1PacketEvidence:
    """Build Stage-1 evidence fused across all requested sampling origins."""

    cfg = fusion_config or MultiOriginFusionConfig()
    origins = _origin_list(int(os_factor), cfg)
    origin_stage1 = tuple(
        build_savaux_stage1_packet_evidence(
            samples=samples,
            start_samples=start_samples,
            sf=int(sf),
            os_factor=int(os_factor),
            abs_indices=abs_indices,
            cfo_int=int(cfo_int),
            cfo_frac=float(cfo_frac),
            header_start_sample=header_start_sample,
            residual_sto_chips=residual_sto_chips,
            config=_stage1_config_for_origin(stage1_config, origin, cfg),
        )
        for origin in origins
    )
    if not origin_stage1:
        raise ValueError("no origin Stage-1 evidence built")

    phase_origin = int(cfg.phase_origin if cfg.phase_origin is not None else int(os_factor) // 2)
    if phase_origin not in origins:
        phase_origin = origins[min(len(origins) - 1, max(0, len(origins) // 2))]
    phase_pos = origins.index(phase_origin)
    phase_stage1 = origin_stage1[phase_pos]

    count = min(len(stage1.evidence_powers) for stage1 in origin_stage1)
    fused_powers = tuple(
        fuse_origin_power_rows([stage1.evidence_powers[idx] for stage1 in origin_stage1], cfg)
        for idx in range(count)
    )
    if str(cfg.phase_mode) == "fixed_origin":
        center_spectra = tuple(phase_stage1.center_spectra[:count])
    else:
        center_spectra = tuple(
            fuse_origin_phase_spectrum(
                [stage1.center_spectra[idx] for stage1 in origin_stage1],
                [stage1.evidence_powers[idx] for stage1 in origin_stage1],
                cfg,
            )
            for idx in range(count)
        )
    return MultiOriginStage1PacketEvidence(
        origin_stage1=origin_stage1,
        origin_shifts=origins,
        phase_origin=phase_origin,
        phase_stage1=phase_stage1,
        center_spectra=center_spectra,
        evidence_powers=fused_powers,
        abs_indices=tuple(phase_stage1.abs_indices[:count]),
        branch_phase_agreements=tuple(phase_stage1.branch_phase_agreements[:count]),
        branch_spectra=tuple(phase_stage1.branch_spectra[:count]),
        dechirped_symbols=tuple(phase_stage1.dechirped_symbols[:count]),
    )


def select_multi_origin_viterbi_path(
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
    fusion_config: MultiOriginFusionConfig | None = None,
) -> SymbolPhaseResult:
    """Run one-order phase DP on multi-origin fused FFT-bin powers."""

    evidence = build_multi_origin_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
        stage1_config=stage1_config,
        fusion_config=fusion_config,
    )
    return select_phase_viterbi_path(
        center_spectra=evidence.center_spectra,
        evidence_powers=evidence.evidence_powers,
        abs_indices=evidence.abs_indices,
        config=selector_config or default_savaux_phase_path_config(),
        offset_coherences=evidence.branch_phase_agreements,
    )


def _path_change_count(left: Sequence[int], right: Sequence[int]) -> int:
    count = min(len(left), len(right))
    return int(sum(int(left[idx]) != int(right[idx]) for idx in range(count)))


def _normalized_bin_power(power: np.ndarray, raw_bin: int) -> float:
    values = np.asarray(power, dtype=np.float64)
    if values.size == 0:
        return 0.0
    peak = float(np.max(values))
    if peak <= 0.0:
        return 0.0
    b = int(raw_bin)
    if b < 0 or b >= values.size:
        return 0.0
    return float(values[b] / (peak + 1e-30))


def _symbol_old_power_margin(
    old_powers: Sequence[np.ndarray],
    symbol_index: int,
    dual_bin: int,
    multi_bin: int,
) -> float:
    idx = int(symbol_index)
    if idx < 0 or idx >= len(old_powers):
        return -float("inf")
    power = np.asarray(old_powers[idx], dtype=np.float64)
    multi_power = _normalized_bin_power(power, int(multi_bin))
    dual_power = _normalized_bin_power(power, int(dual_bin))
    return float(multi_power - dual_power)


def _symbol_old_multi_norm(
    old_powers: Sequence[np.ndarray],
    symbol_index: int,
    multi_bin: int,
) -> float:
    idx = int(symbol_index)
    if idx < 0 or idx >= len(old_powers):
        return 0.0
    return _normalized_bin_power(np.asarray(old_powers[idx], dtype=np.float64), int(multi_bin))


def _symbol_gated_bins(
    dual: SymbolPhaseResult,
    multi: SymbolPhaseResult,
    old_powers: Sequence[np.ndarray] | None,
    gate_config: MultiOriginGateConfig,
) -> tuple[int, ...]:
    dual_bins = tuple(int(v) for v in dual.selected_raw_bins)
    multi_bins = tuple(int(v) for v in multi.selected_raw_bins)
    count = min(len(dual_bins), len(multi_bins))
    if count == 0:
        return dual_bins
    selected = list(dual_bins[:count])
    trajectory_gain = float(multi.trajectory_score - dual.trajectory_score)
    if trajectory_gain < float(gate_config.symbol_min_trajectory_gain):
        return tuple(selected)
    for idx in range(count):
        if int(dual_bins[idx]) == int(multi_bins[idx]):
            continue
        margin = _symbol_old_power_margin(
            old_powers or (),
            idx,
            int(dual_bins[idx]),
            int(multi_bins[idx]),
        )
        multi_norm = _symbol_old_multi_norm(old_powers or (), idx, int(multi_bins[idx]))
        if (
            margin >= float(gate_config.symbol_min_old_power_margin)
            and (
                gate_config.symbol_min_old_multi_norm is None
                or multi_norm >= float(gate_config.symbol_min_old_multi_norm)
            )
        ):
            selected[idx] = int(multi_bins[idx])
    return tuple(selected)


def _result_with_selected_bins(
    template: SymbolPhaseResult,
    selected_bins: Sequence[int],
    *,
    error: str = "",
) -> SymbolPhaseResult:
    return SymbolPhaseResult(
        selected_raw_bins=tuple(int(v) for v in selected_bins),
        locked_mask=template.locked_mask,
        evidences=template.evidences,
        phase_line=template.phase_line,
        trajectory_score=float(template.trajectory_score),
        line_score=float(template.line_score),
        mean_phase_score=float(template.mean_phase_score),
        mean_amp_score=float(template.mean_amp_score),
        uncertain_count=int(template.uncertain_count),
        locked_count=int(template.locked_count),
        beam_final_size=int(template.beam_final_size),
        error=str(error or template.error),
    )


def arbitrate_dual_vs_multi_origin_path(
    dual: SymbolPhaseResult,
    multi: SymbolPhaseResult,
    gate_config: MultiOriginGateConfig | None = None,
    old_evidence_powers: Sequence[np.ndarray] | None = None,
) -> MultiOriginArbiterResult:
    """Choose between dual-evidence and multi-origin paths using path metrics only."""

    cfg = gate_config or MultiOriginGateConfig()
    trajectory_gain = float(multi.trajectory_score - dual.trajectory_score)
    amp_gain = float(multi.mean_amp_score - dual.mean_amp_score)
    changes = _path_change_count(multi.selected_raw_bins, dual.selected_raw_bins)
    use_multi = True
    reason = "accepted"
    if not bool(cfg.enabled):
        use_multi = True
        reason = "gate_disabled"
    elif bool(cfg.symbol_gate_enabled):
        use_multi = False
        reason = "symbol_gate"
    elif trajectory_gain < float(cfg.min_trajectory_gain):
        use_multi = False
        reason = "trajectory_gain"
    elif cfg.min_amp_gain is not None and amp_gain < float(cfg.min_amp_gain):
        use_multi = False
        reason = "amp_gain"
    elif cfg.max_changes_vs_dual is not None and changes > int(cfg.max_changes_vs_dual):
        use_multi = False
        reason = "change_count"
    decision = MultiOriginGateDecision(
        use_multi=bool(use_multi),
        reason=str(reason),
        trajectory_gain=float(trajectory_gain),
        amp_gain=float(amp_gain),
        changes_vs_dual=int(changes),
    )
    if bool(cfg.enabled) and bool(cfg.symbol_gate_enabled):
        selected_bins = _symbol_gated_bins(dual, multi, old_evidence_powers, cfg)
        selected = _result_with_selected_bins(dual, selected_bins, error="symbol_gate")
    else:
        selected = multi if use_multi else dual
    return MultiOriginArbiterResult(
        selected=selected,
        dual=dual,
        multi=multi,
        decision=decision,
    )


def select_multi_origin_island_reconstruction_path(
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
    reconstruction_config: IslandReconstructionConfig | None = None,
    fusion_config: MultiOriginFusionConfig | None = None,
) -> SymbolPhaseResult:
    """Run island reconstruction DP with multi-origin fused powers."""

    evidence = build_multi_origin_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
        stage1_config=stage1_config,
        fusion_config=fusion_config,
    )
    path_cfg = selector_config or default_savaux_phase_path_config()
    baseline = select_phase_viterbi_path(
        center_spectra=evidence.center_spectra,
        evidence_powers=evidence.evidence_powers,
        abs_indices=evidence.abs_indices,
        config=path_cfg,
        offset_coherences=evidence.branch_phase_agreements,
    )
    auxiliary = tuple(
        tuple(np.asarray(stage1.evidence_powers[idx], dtype=np.float64) for stage1 in evidence.origin_stage1)
        for idx in range(min(len(stage1.evidence_powers) for stage1 in evidence.origin_stage1))
    )
    return select_island_reconstruction_viterbi_path(
        center_spectra=evidence.center_spectra,
        evidence_powers=evidence.evidence_powers,
        abs_indices=evidence.abs_indices,
        config=path_cfg,
        reconstruction_config=reconstruction_config,
        offset_coherences=evidence.branch_phase_agreements,
        branch_spectra=evidence.branch_spectra,
        dechirped_symbols=evidence.dechirped_symbols,
        os_factor=int(os_factor),
        baseline_bins=baseline.selected_raw_bins,
        residual_sto_chips=residual_sto_chips,
        auxiliary_evidence_powers=auxiliary,
    )


__all__ = [
    "MultiOriginFusionConfig",
    "MultiOriginGateConfig",
    "MultiOriginGateDecision",
    "MultiOriginArbiterResult",
    "MultiOriginStage1PacketEvidence",
    "arbitrate_dual_vs_multi_origin_path",
    "build_multi_origin_stage1_packet_evidence",
    "fuse_origin_power_rows",
    "select_multi_origin_island_reconstruction_path",
    "select_multi_origin_viterbi_path",
]

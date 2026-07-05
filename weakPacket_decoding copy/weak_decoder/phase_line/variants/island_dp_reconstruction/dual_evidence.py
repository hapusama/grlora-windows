"""Dual Stage-1 FFT-bin evidence fusion for island-DP experiments.

This module stays local to the island reconstruction variant.  It does not
decode payload bits or consult CRC; it only builds alternate FFT-bin power
evidence for the existing phase Viterbi and island DP selectors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np

from ...savaux_stage1 import (
    SavauxStage1Config,
    SavauxStage1PacketEvidence,
    build_savaux_stage1_packet_evidence,
    default_savaux_phase_path_config,
)
from ...configs import PhasePathSelectorConfig
from ..v1_one_order_dp import select_phase_viterbi_path
from .selector import IslandReconstructionConfig, select_island_reconstruction_viterbi_path
from ....symbol_phase_two_stage import SymbolPhaseResult


FusionMode = Literal["sum_norm", "product_norm", "max_norm", "old", "corrected"]


@dataclass(frozen=True)
class DualEvidenceFusionConfig:
    """Parameters for old/new Stage-1 bin-power fusion."""

    mode: FusionMode = "product_norm"
    corrected_weight: float = 0.50
    stage1_top_k: int = 40
    residual_sto_phase_sign: Literal["plus", "minus"] = "plus"
    residual_sto_preselect_factor: int = 4
    retain_dechirped_symbols: bool = True


@dataclass(frozen=True)
class DualSavauxStage1PacketEvidence:
    """Old-center phase evidence plus fused FFT-bin powers."""

    old_stage1: SavauxStage1PacketEvidence
    corrected_stage1: SavauxStage1PacketEvidence
    center_spectra: tuple[np.ndarray, ...]
    evidence_powers: tuple[np.ndarray, ...]
    abs_indices: tuple[float, ...]
    branch_phase_agreements: tuple[np.ndarray, ...]
    branch_spectra: tuple[tuple[np.ndarray, ...], ...]
    dechirped_symbols: tuple[np.ndarray | None, ...]


def _normalize_power(power: np.ndarray) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    peak = float(np.max(values))
    if peak <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(values / peak, dtype=np.float64)


def fuse_power_rows(
    old_power: np.ndarray,
    corrected_power: np.ndarray,
    config: DualEvidenceFusionConfig | None = None,
) -> np.ndarray:
    """Fuse two per-symbol FFT-bin power vectors."""

    cfg = config or DualEvidenceFusionConfig()
    old = np.asarray(old_power, dtype=np.float64)
    corrected = np.asarray(corrected_power, dtype=np.float64)
    if old.shape != corrected.shape:
        raise ValueError("old and corrected power vectors must have the same shape")
    mode = str(cfg.mode)
    if mode == "old":
        return old.copy()
    if mode == "corrected":
        return corrected.copy()

    a = _normalize_power(old)
    b = _normalize_power(corrected)
    w = float(max(0.0, min(1.0, float(cfg.corrected_weight))))
    if mode == "sum_norm":
        fused = (1.0 - w) * a + w * b
    elif mode == "product_norm":
        fused = np.exp((1.0 - w) * np.log(a + 1e-30) + w * np.log(b + 1e-30))
    elif mode == "max_norm":
        fused = np.maximum(a, b)
    else:
        raise ValueError(f"unsupported fusion mode: {cfg.mode!r}")

    peak = float(np.max(fused)) if fused.size else 0.0
    if peak > 0.0:
        fused = fused / peak
    return np.asarray(fused, dtype=np.float64)


def fuse_packet_powers(
    old_stage1: SavauxStage1PacketEvidence,
    corrected_stage1: SavauxStage1PacketEvidence,
    config: DualEvidenceFusionConfig | None = None,
) -> tuple[np.ndarray, ...]:
    """Fuse all per-symbol power rows from two Stage-1 packets."""

    count = min(len(old_stage1.evidence_powers), len(corrected_stage1.evidence_powers))
    return tuple(
        fuse_power_rows(old_stage1.evidence_powers[idx], corrected_stage1.evidence_powers[idx], config)
        for idx in range(count)
    )


def _stage1_configs(
    base_config: SavauxStage1Config | None,
    fusion_config: DualEvidenceFusionConfig,
) -> tuple[SavauxStage1Config, SavauxStage1Config]:
    base = base_config or SavauxStage1Config()
    top_k = max(1, int(fusion_config.stage1_top_k))
    old_cfg = replace(
        base,
        top_k=top_k,
        residual_sto_phase_correction=False,
        residual_sto_update_power=False,
        retain_dechirped_symbols=bool(fusion_config.retain_dechirped_symbols),
    )
    corrected_cfg = replace(
        base,
        top_k=top_k,
        residual_sto_phase_correction=True,
        residual_sto_phase_sign=fusion_config.residual_sto_phase_sign,
        residual_sto_preselect_factor=max(1, int(fusion_config.residual_sto_preselect_factor)),
        residual_sto_update_power=True,
        retain_dechirped_symbols=bool(fusion_config.retain_dechirped_symbols),
    )
    return old_cfg, corrected_cfg


def build_dual_savaux_stage1_packet_evidence(
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
    fusion_config: DualEvidenceFusionConfig | None = None,
) -> DualSavauxStage1PacketEvidence:
    """Build old-center spectra with fused old/corrected FFT-bin powers."""

    fusion_cfg = fusion_config or DualEvidenceFusionConfig()
    old_cfg, corrected_cfg = _stage1_configs(stage1_config, fusion_cfg)
    old_stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
        config=old_cfg,
    )
    corrected_stage1 = build_savaux_stage1_packet_evidence(
        samples=samples,
        start_samples=start_samples,
        sf=int(sf),
        os_factor=int(os_factor),
        abs_indices=abs_indices,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        residual_sto_chips=residual_sto_chips,
        config=corrected_cfg,
    )
    fused_powers = fuse_packet_powers(old_stage1, corrected_stage1, fusion_cfg)
    count = min(len(old_stage1.center_spectra), len(fused_powers))
    return DualSavauxStage1PacketEvidence(
        old_stage1=old_stage1,
        corrected_stage1=corrected_stage1,
        center_spectra=tuple(old_stage1.center_spectra[:count]),
        evidence_powers=tuple(fused_powers[:count]),
        abs_indices=tuple(old_stage1.abs_indices[:count]),
        branch_phase_agreements=tuple(old_stage1.branch_phase_agreements[:count]),
        branch_spectra=tuple(old_stage1.branch_spectra[:count]),
        dechirped_symbols=tuple(old_stage1.dechirped_symbols[:count]),
    )


def select_dual_evidence_viterbi_path(
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
    fusion_config: DualEvidenceFusionConfig | None = None,
) -> SymbolPhaseResult:
    """Run one-order phase DP on fused FFT-bin evidence."""

    dual = build_dual_savaux_stage1_packet_evidence(
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
        center_spectra=dual.center_spectra,
        evidence_powers=dual.evidence_powers,
        abs_indices=dual.abs_indices,
        config=selector_config or default_savaux_phase_path_config(),
        offset_coherences=dual.branch_phase_agreements,
    )


def select_dual_evidence_island_reconstruction_path(
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
    stage1_config: SavauxStage1Config | None = None,
    selector_config: PhasePathSelectorConfig | None = None,
    reconstruction_config: IslandReconstructionConfig | None = None,
    fusion_config: DualEvidenceFusionConfig | None = None,
) -> SymbolPhaseResult:
    """Run island reconstruction DP using fused FFT-bin power evidence."""

    dual = build_dual_savaux_stage1_packet_evidence(
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
    return select_island_reconstruction_viterbi_path(
        center_spectra=dual.center_spectra,
        evidence_powers=dual.evidence_powers,
        abs_indices=dual.abs_indices,
        config=path_cfg,
        reconstruction_config=reconstruction_config,
        offset_coherences=dual.branch_phase_agreements,
        branch_spectra=dual.branch_spectra,
        dechirped_symbols=dual.dechirped_symbols,
        os_factor=int(os_factor),
        residual_sto_chips=residual_sto_chips,
        branch_residual_sto_chips=branch_residual_sto_chips,
        auxiliary_evidence_powers=tuple(
            (
                np.asarray(dual.old_stage1.evidence_powers[idx], dtype=np.float64),
                np.asarray(dual.corrected_stage1.evidence_powers[idx], dtype=np.float64),
            )
            for idx in range(min(len(dual.old_stage1.evidence_powers), len(dual.corrected_stage1.evidence_powers)))
        ),
    )


__all__ = [
    "DualEvidenceFusionConfig",
    "DualSavauxStage1PacketEvidence",
    "build_dual_savaux_stage1_packet_evidence",
    "fuse_packet_powers",
    "fuse_power_rows",
    "select_dual_evidence_island_reconstruction_path",
    "select_dual_evidence_viterbi_path",
]

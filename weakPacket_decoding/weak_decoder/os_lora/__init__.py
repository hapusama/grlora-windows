"""OS-LoRa 基于过采样与非均匀 pattern 的弱包解调接口。

系统实现位于 :mod:`weak_decoder.os_lora.system`，离线实验入口位于
:mod:`weak_decoder.os_lora.experiments`。本文件继续导出主要系统 API，保证
现有 ``from weak_decoder.os_lora import ...`` 调用不受目录重构影响。
"""

from .system.chirp_svd import (
    ChirpSVDSpectra,
    chirp_matrix,
    chirp_svd_spectra,
    low_rank_chirp_matrix,
    savaux_spectrum_from_dechirped_matrix,
)
from .system.nonuniform_sampling import (
    NonuniformPatternBank,
    NonuniformScoreResult,
    MatrixFreeGLSResult,
    PatternSubsetSelection,
    ConditionalLoRaGLSResult,
    build_pattern_bank,
    conditional_lora_gls_detect,
    crossfit_gls_spectrum_power,
    crossfit_weighted_spectrum,
    lora_branch_color_mismatch,
    lora_interbin_leakage_covariance,
    lora_phase_law_consistency,
    lora_wrap_consistency_power,
    matrix_free_crossfit_gls_spectrum_power,
    pattern_bank_split_spectra,
    pattern_covariance_color_mismatch,
    score_nonuniform_candidates,
    select_pattern_subset,
    select_pattern_subset_by_information,
)

__all__ = [
    "ChirpSVDSpectra",
    "NonuniformPatternBank",
    "NonuniformScoreResult",
    "MatrixFreeGLSResult",
    "PatternSubsetSelection",
    "ConditionalLoRaGLSResult",
    "build_pattern_bank",
    "chirp_matrix",
    "chirp_svd_spectra",
    "conditional_lora_gls_detect",
    "crossfit_gls_spectrum_power",
    "crossfit_weighted_spectrum",
    "low_rank_chirp_matrix",
    "lora_branch_color_mismatch",
    "lora_interbin_leakage_covariance",
    "lora_phase_law_consistency",
    "lora_wrap_consistency_power",
    "matrix_free_crossfit_gls_spectrum_power",
    "pattern_bank_split_spectra",
    "pattern_covariance_color_mismatch",
    "savaux_spectrum_from_dechirped_matrix",
    "score_nonuniform_candidates",
    "select_pattern_subset",
    "select_pattern_subset_by_information",
]

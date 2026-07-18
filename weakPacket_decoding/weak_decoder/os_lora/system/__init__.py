"""OS-LoRa 可复用系统实现。

该包只包含能够被在线解调器或其他模块直接复用的算法，不包含数据集路径、
Monte Carlo 扫描、绘图或结果 CSV 编排等实验逻辑。
"""

from .chirp_svd import (
    ChirpSVDSpectra,
    chirp_matrix,
    chirp_svd_spectra,
    low_rank_chirp_matrix,
    savaux_spectrum_from_dechirped_matrix,
)
from .noise import select_background_bins
from .nonuniform_sampling import (
    ConditionalLoRaGLSResult,
    MatrixFreeGLSResult,
    NonuniformPatternBank,
    NonuniformScoreResult,
    PatternSubsetSelection,
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
    "ConditionalLoRaGLSResult",
    "MatrixFreeGLSResult",
    "NonuniformPatternBank",
    "NonuniformScoreResult",
    "PatternSubsetSelection",
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
    "select_background_bins",
    "select_pattern_subset",
    "select_pattern_subset_by_information",
]

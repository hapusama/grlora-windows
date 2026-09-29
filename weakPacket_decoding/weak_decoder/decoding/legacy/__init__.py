"""[2026-09-28 结构整理] 退役的自研解调探索模块收纳。

这些模块是 2026-06~07 期间围绕 Savaux Top-K 的路径/稳健/修剪类二次证据实验
（adaptive_path / structured_path / timing_path / alias_trim / robust_sparse /
phase_templates），已被 ambiguity-ridge 主线与 2026-09-28 的细网格 GLRT 链
（见 data/experiments/ablation_battle_20260928/）取代。
保留供历史实验复现；新代码请勿依赖。
"""

from .alias_trim import (
    AliasBlocker,
    AliasTrimConfig,
    AliasTrimResult,
    alias_trim_rerank,
    demod_alias_trim_symbol,
)
from .robust_sparse_demod import (
    RobustSparseConfig,
    RobustSparseResult,
    demod_robust_sparse_symbol,
    robust_sparse_rerank,
)

__all__ = [
    "AliasBlocker",
    "AliasTrimConfig",
    "AliasTrimResult",
    "RobustSparseConfig",
    "RobustSparseResult",
    "alias_trim_rerank",
    "demod_alias_trim_symbol",
    "demod_robust_sparse_symbol",
    "robust_sparse_rerank",
]

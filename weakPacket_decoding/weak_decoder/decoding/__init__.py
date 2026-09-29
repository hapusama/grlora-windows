"""LoRa 帧编解码层（活跃部分）+ 退役解调探索的向后兼容再导出。

[2026-09-28 结构整理] 活跃模块：header_first_demod / payload_codec。
退役探索（alias_trim / robust_sparse / adaptive_path / structured_path /
timing_path / phase_templates）已迁入 legacy/，此处仅保留符号级再导出。
"""

from .legacy.alias_trim import (
    AliasBlocker,
    AliasTrimConfig,
    AliasTrimResult,
    alias_trim_rerank,
    demod_alias_trim_symbol,
)
from .legacy.robust_sparse_demod import (
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

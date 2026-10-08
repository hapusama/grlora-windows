# -*- coding: utf-8 -*-
"""κ 格解调器（我方方法正式模块）：fine_grid + trellis + demodulator。

[D2F 副本 2026-10-05] 源头 = weak_decoder/decoding/kappa_trellis/。
导入方式（D2F 线内）::

    sys.path.insert(0, r"<...>\weak_decoder\D2F_20261005")
    from decode_trellis import KappaTrellisDemodulator
"""

from .fine_grid import FineGrid, build_rot_columns, build_bandpass_mask
from .trellis import (emission_prominence, emission_max, forward_backward,
                      per_symbol_posterior, viterbi)
from .demodulator import KappaTrellisDemodulator, READOUTS

__all__ = [
    "FineGrid",
    "build_rot_columns",
    "build_bandpass_mask",
    "emission_prominence",
    "emission_max",
    "forward_backward",
    "per_symbol_posterior",
    "viterbi",
    "KappaTrellisDemodulator",
    "READOUTS",
]

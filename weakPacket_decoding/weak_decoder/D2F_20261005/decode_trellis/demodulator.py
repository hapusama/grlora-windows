# -*- coding: utf-8 -*-
"""KappaTrellisDemodulator：κ 格解调器高层入口（与 baseline 同级的正式模块）。

[D2F 副本 2026-10-05] 源头 = weak_decoder/decoding/kappa_trellis/demodulator.py
（冻结快照，数值逐位一致，无改动）。D2F 线的 trellis 演进（去网格化/
连续化/与精确模板对接）在此副本进行，不影响共享原件。

用法（与 sav_demod / trim_demod 等 baseline 对等的证据行接口）::

    from decode_trellis import KappaTrellisDemodulator   # D2F 副本
    demod = KappaTrellisDemodulator(sf=10, os_factor=4)
    rows = demod.demod_payload(samples, start_symbol=16, psym=35,
                               readout="viterbi")   # "grid0"/"viterbi"/"bcjr"
    # rows: (psym, N) bin 域能量行；判据侧按值映射消费（见 value_from_bin）

值映射（SER/判据共用）::

    value = demod.value_from_bin(bin_argmax, delta, ldro)
    # 非 LDRO: ((bin - delta - 1) % N) % N        —— value = bin-1
    # LDRO  : ((bin - delta - 1) % N) // 4        —— 值 v 占 bins 4v+1..4v+4

δ（整 bin 残余映射）按既有协议在干净信号上以 mode(argmax − 期望bin) 冻结，
属同步先验一部分（见 README.md「约定与判据」）。
"""
from __future__ import annotations

import numpy as np

from .fine_grid import FineGrid
from . import trellis


READOUTS = ("grid0", "viterbi", "bcjr")


class KappaTrellisDemodulator:
    """κ 细格 + 跨符号格架解调器（我方方法，TREL/BCJR 两读出）。"""

    def __init__(self, sf: int, os_factor: int = 4, n_cols: int = 5,
                 radius: int = 1):
        self.grid = FineGrid(sf, os_factor, n_cols)
        self.radius = int(radius)

    @property
    def n_bins(self) -> int:
        return self.grid.n

    def grids(self, samples: np.ndarray, start_symbol: int, psym: int):
        """逐符号 (N, n_cols) 细格谱列表。"""
        return [self.grid.symbol_spectrum(samples, start_symbol + k)
                for k in range(psym)]

    def demod_payload(self, samples: np.ndarray, start_symbol: int, psym: int,
                      readout: str = "viterbi") -> np.ndarray:
        """返回 (psym, N) bin 域能量行（软证据，供 SymFEC 风格判据）。"""
        if readout not in READOUTS:
            raise ValueError(f"readout 必须是 {READOUTS}，got {readout}")
        ms = self.grids(samples, start_symbol, psym)
        if readout == "grid0":
            return np.stack([m[:, 2] for m in ms])
        lam = trellis.emission_prominence(ms)
        if readout == "viterbi":
            path = trellis.viterbi(lam, self.radius)
            return np.stack([ms[k][:, path[k]] for k in range(psym)])
        post = trellis.forward_backward(lam, self.radius)
        return np.stack([ms[k] @ post[k] for k in range(psym)])

    def value_from_bin(self, bin_index: int, delta: int = 0,
                       ldro: bool = False) -> int:
        """bin 级 argmax（扣除 δ 后）→ 符号值。

        约定与战 runner 一致：非 LDRO 音在 bin v（value=argmax−δ），
        LDRO 音在 bin 4v+1（value=((argmax−δ−1) mod N)//4）。
        """
        n = self.grid.n
        b = (int(bin_index) - int(delta)) % n
        return (b - 1) % n // 4 if ldro else b

    def freeze_delta(self, samples: np.ndarray, start_symbol: int, psym: int,
                     known_values, ldro: bool = False) -> int:
        """δ 冻结：mode over 符号 of (argmax(col2) − 期望 bin)。

        期望 bin = 值 v 的音位（非 LDRO: v；LDRO: 4v+1）。
        known_values 为干净信号上的已知真值（GT/同步先验约定）。
        """
        ms = self.grids(samples, start_symbol, psym)
        rows = np.stack([m[:, 2] for m in ms])
        d = []
        for k, v in enumerate(known_values):
            expect = (4 * int(v) + 1) if ldro else int(v)
            d.append((int(np.argmax(rows[k])) - expect) % self.grid.n)
        return int(np.bincount(d).argmax())

# -*- coding: utf-8 -*-
"""κ 细网格谱：过采样相位 chip 抽取 + κ 频率旋转列。

提取自 dera_battle_20260929/battle_runner.py 的 wm()（数值逐位一致，
见 tests/test_kappa_trellis.py 的等价校验）。列 d ↔ κ=(2-d)/4，
即 5 列覆盖 κ∈{-0.5,-0.25,0,+0.25,+0.5}（OS=4 时 0.25 bin 步进）。
"""
from __future__ import annotations

import numpy as np

from ...chirp import build_upchirp


def build_rot_columns(n_bins: int, os_factor: int, n_cols: int = 5) -> np.ndarray:
    """(n_cols, n_bins) 旋转矩阵：列 d 的 κ=(2-d)/4 个 bin。"""
    return np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(n_bins) / n_bins)
                     for d in range(n_cols)])


def build_bandpass_mask(nf: int) -> np.ndarray:
    """符号窗频域带通掩膜（|f| ≤ BW/2 + 40/NF，与 wm 同式）。"""
    return np.abs(np.fft.fftfreq(nf)) <= 0.125 + 40.0 / nf


class FineGrid:
    """一个 (SF, OS) 组合的预计算参考与网格（可跨帧复用）。"""

    def __init__(self, sf: int, os_factor: int = 4, n_cols: int = 5):
        self.sf = int(sf)
        self.os = int(os_factor)
        self.n = 1 << self.sf
        self.nf = self.n * self.os
        self.n_cols = int(n_cols)
        self.ref = np.conj(build_upchirp(self.sf, symbol_id=0, os_factor=self.os))
        self.rot = build_rot_columns(self.n, self.os, self.n_cols)
        self.mask = build_bandpass_mask(self.nf)

    def symbol_spectrum(self, samples: np.ndarray, k: int) -> np.ndarray:
        """第 k 个符号的 (N, n_cols) 窗口能量；行=bin，列=κ 假设。"""
        seg = np.fft.ifft(np.fft.fft(samples[k * self.nf:(k + 1) * self.nf]) * self.mask)
        y = seg * self.ref
        chips = np.stack([y[p:p + self.nf:self.os] for p in range(self.os)])
        inp = (chips[None, :, :] * self.rot[:, None, :]).reshape(
            self.n_cols * self.os, self.n)
        P = np.abs(np.fft.fft(inp, axis=1)) ** 2
        return P.reshape(self.n_cols, self.os, self.n).mean(axis=1).T

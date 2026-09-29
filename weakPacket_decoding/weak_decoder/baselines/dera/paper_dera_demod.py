# -*- coding: utf-8 -*-
"""DeRa (MobiCom'26) 解码级复现（decode-only port）。

依据：论文 Algorithm 1 Stage 1-2（Phase-compensated magnitude decode +
CRC-guided bounded iteration）与仓库 gen_signal.py 的信号模型。上游主系统
代码截至 2026-09-29 未发布（repo 仅含 gen_signal.py），本 port 只覆盖
解码阶段（我们对比的 perfect-sync/decode-only 领域）；检测与 20 候选
重试不在 decode-only 比较范围。

机制（论文 3.3 Two-Section Windowing）：CFO/TO 下接收窗跨两个相邻符号
（当前符号尾 + 下符号头各半能量）。每符号窗拆两段，各自以全符号参考
下啁啾去斜后补零细 FFT；两段音对候选值 v 有确定性相位关系
Δφ(v)=2π·f_v·(NF/2)；相干合并 coh[v]=X1[2v]+e^{-jΔφ(v)}·X2[2v]。
Stage-2：第一遍幅度解出临时符号后，ML 估计残余公共相位
φ0=angle(Σ X2[2ŝ]·conj(X1[2ŝ])·e^{+jΔφ(ŝ)})，再相干重组合。

输入约定：samples 已完成整 bin+分数 CFO 纠偏与 STO 亚 chip 对齐
（与全部 baseline 相同的公平输入）；残余分数偏移由其自身机制处理。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...chirp import build_upchirp


@dataclass(frozen=True)
class DeRaDemodResult:
    """单符号解调结果（行=值域上的相干合并幅度谱）。"""
    metric: np.ndarray          # (N,) |coh[v]|^2，Stage-2 相干行
    stage1_metric: np.ndarray   # (N,) Stage-1 行（无残余相位校正）
    argmax_symbol_value: int


class DeRaDemodulator:
    """跨符号保持 Stage-2 ML 相位估计状态的 DeRa 解调器。"""

    def __init__(self, sf: int, os_factor: int):
        self.sf = int(sf)
        self.os = int(os_factor)
        self.n = 1 << int(sf)
        self.nf = self.n * self.os
        self.ref = np.conj(build_upchirp(self.sf, symbol_id=0, os_factor=self.os))
        # 细 FFT：半窗(NF/2 样本)补零到 2*NF；值 v 的音在细 bin 2v
        self.fine = 2 * self.nf
        # 两段相位关系 Δφ(v)=2π·(v/(4N))·(NF/2)=πv（OS=4 约定下）
        self.dphi = np.pi * np.arange(self.n)
        self.bin_of_value = (2 * np.arange(self.n)) % self.fine

    def _section_fine_fft(self, samples: np.ndarray, start: int, half: int,
                          ref_offset: int = 0):
        seg = samples[start:start + half] * self.ref[ref_offset:ref_offset + half]
        buf = np.zeros(self.fine, dtype=np.complex128)
        buf[:half] = seg
        return np.fft.fft(buf)

    def demod_symbol(self, samples: np.ndarray, start_sample: int,
                     phi0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """返回 (stage1_row, coherent_row)，均为值域 (N,) 幅度谱。

        段1=窗前半 x ref 前半；段2=窗后半 x ref 后半（参考啁啾连续，
        两段去斜后值 v 的音都落在细 bin 2v，仅相位差 Δφ(v)=πv）。
        """
        half = self.nf // 2
        X1 = self._section_fine_fft(samples, start_sample, half, 0)
        X2 = self._section_fine_fft(samples, start_sample + half, half, half)
        c1 = X1[self.bin_of_value]
        c2 = X2[self.bin_of_value]
        s1 = np.abs(c1 + np.exp(-1j * self.dphi) * c2) ** 2
        s2 = np.abs(c1 + np.exp(-1j * (self.dphi + phi0)) * c2) ** 2
        return s1, s2

    @staticmethod
    def ml_phase(c1_all, c2_all, dphi, symbols_hat):
        """Eq.7 风格 ML 公共相位：临时符号上的平均残差相位。"""
        idx = np.asarray(symbols_hat, dtype=int)
        z = c2_all[np.arange(len(idx)), idx] * np.conj(c1_all[np.arange(len(idx)), idx])
        z = z * np.exp(1j * dphi[idx])
        s = z.sum()
        return float(np.angle(s)) if abs(s) > 0 else 0.0

    def demod_payload(self, samples: np.ndarray, start_symbol: int, psym: int,
                      ) -> tuple[np.ndarray, np.ndarray]:
        """整段 payload：返回 (stage1_rows, coherent_rows)，(psym, N)。"""
        half = self.nf // 2
        c1s, c2s = [], []
        for k in range(psym):
            st = (start_symbol + k) * self.nf
            X1 = self._section_fine_fft(samples, st, half, 0)
            X2 = self._section_fine_fft(samples, st + half, half, half)
            c1s.append(X1[self.bin_of_value])
            c2s.append(X2[self.bin_of_value])
        c1_all = np.stack(c1s)
        c2_all = np.stack(c2s)
        stage1 = np.abs(c1_all + np.exp(-1j * self.dphi)[None, :] * c2_all) ** 2
        # Stage-1 幅度解出临时符号（两段能量和，避免相位耦合初始歧义）
        prov = np.argmax(np.abs(c1_all) ** 2 + np.abs(c2_all) ** 2, axis=1)
        phi0 = self.ml_phase(c1_all, c2_all, self.dphi, prov)
        coh = np.abs(c1_all + np.exp(-1j * (self.dphi + phi0))[None, :] * c2_all) ** 2
        return stage1, coh

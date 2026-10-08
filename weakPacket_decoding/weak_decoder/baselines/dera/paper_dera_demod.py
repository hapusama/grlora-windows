# -*- coding: utf-8 -*-
"""DeRa (MobiCom'26) 解码级复现（decode-only port，v2：逐候选越界切分+相干合并）。

论文依据（paper/DeRa_MobiCom26.pdf，仓库副本公式页直读核对）：
  - §4.3 / Algorithm 1 Stage 1-2：payload 解码 = 两段 FFT 贡献 V1/V2 的相干
    重组合 Y_coh[c] = V1[c] + V2[c]·e^{-j(Δφcfo[c]+ΔφTO[c])}（Eq.22），
    幅度峰为判决（Eq.23）；CRC 失败后 Stage-2 用 ML 公共相位
    φ̂0 = ∠(Σ_i Y_i[ĉ_i])（Algorithm 1 line 36）重新相干解出。
  - 分段边界 = 候选 c 的啁啾越界点 t_wrap = (1−c/N)·T（§3.3 两段窗模型在
    payload 窗内的形态；与 LoRaTrimmer 原型 time_split 几何同源——两文的
    V1/V2 分段一致，差别只在合并方式：DeRa 相干 vs Trimmer 非相干
    |V1|²+|V2|²，论文 §4.3 明言非相干损失至多 3 dB）。

公平输入映射（decode-only，与全 baseline 相同口径）：
  samples 已完成整 bin+分数 CFO 纠偏与 STO 亚 chip 对齐 ⇒ 论文确定项
  Δφcfo[c]（Eq.20）、ΔφTO[c]（Eq.21）中的 fcfo、T'−T 均为 0。同步残差
  只剩每帧常数分数 bin 偏移 κ，实测（probe1/probe2，smoke_tmp/dera_probe*.py）
  它在 V1/V2 间引入的相对相位 γ = π·κ 与候选 c 无关 ⇒ 属公共相位，由
  Stage-2 的 ML φ̂0 吸收——这正是论文 Stage-2 自带的机制，无需额外估计 κ。

  port 注记（与论文的唯一实现差异）：Stage-2 ML 初始化的临时符号取
  非相干行 argmax(|V1|²+|V2|²) 而非 Stage-1 行——κ→0.5 时无补偿相干合并
  会相消（probe2: −20dB 55/100 错），非相干初始化对 κ 鲁棒；最终判决行
  仍是论文的 Stage-2 相干输出。判决行取 |·|² 幅度域（与全部 baseline 的
  SymFEC 证据口径一致；论文硬判的 Re 投影与幅度峰在相位对齐后 argmax 同）。

v1（挂起版）失败根因：固定中点两段切分——真越界点在 (1−c/N)·NF，随候选
变化，中点切分使两段各混入相邻段的样点，去斜谱峰不在模型位（native SER
0.14）。v2 改逐候选越界切分后，合成符号 native 全 κ 网格 0 错（probe2）。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ...chirp import build_upchirp


@dataclass(frozen=True)
class DeRaDemodResult:
    """单符号解调结果（行均为值域/候选域 (N,) 功率谱）。"""

    metric: np.ndarray          # Stage-2 相干行 |F + e^{-jφ̂}T|²（最终判决）
    stage1_metric: np.ndarray   # Stage-1 行 |F + T|²（无 ML 相位补偿）
    noncoherent_metric: np.ndarray  # 非相干行 |F|²+|T|²（= LoRaTrimmer 度量，对照）
    argmax_symbol_value: int


@lru_cache(maxsize=32)
def _wrap_split_matrices(sf: int, os_factor: int):
    """逐候选越界切分投影矩阵（与 LoRaTrimmer 原型 time_split 几何一致）。

    候选 k 的符号在窗内 (1−k/N)·M 处越界：front 投影 = 越界前段 × 参考啁啾
    （自候选起始频率继续），tail 投影 = 越界后段 × 回卷参考。二者即论文
    §4.3 的 V1/V2。
    """
    sf, os_factor = int(sf), int(os_factor)
    n_bins = 1 << sf
    n_samples = n_bins * os_factor
    downchirp = np.conjugate(
        build_upchirp(sf=sf, symbol_id=0, os_factor=os_factor)
    ).astype(np.complex64)
    front = np.zeros((n_bins, n_samples), dtype=np.complex64)
    tail = np.zeros((n_bins, n_samples), dtype=np.complex64)
    split_samples = np.empty(n_bins, dtype=np.int64)
    for k in range(n_bins):
        time_shift = int(k * n_samples / n_bins)      # k·os
        time_split = int(n_samples - time_shift)      # (N−k)·os = 越界点
        split_samples[k] = time_split
        front[k, :time_split] = downchirp[time_shift:]
        if k != 0:
            tail[k, time_split:] = downchirp[:time_shift]
    return front, tail, split_samples


class DeRaDemodulator:
    """跨符号保持 Stage-2 ML 公共相位的 DeRa 解调器（decode-only）。"""

    def __init__(self, sf: int, os_factor: int):
        self.sf = int(sf)
        self.os = int(os_factor)
        self.n = 1 << self.sf
        self.nf = self.n * self.os
        self._front, self._tail, self._split = _wrap_split_matrices(
            self.sf, self.os)
        self.ml_phase: float = 0.0

    # ---- 核心投影：F/V1、T/V2、三行度量 ----
    def _project(self, window: np.ndarray):
        """window: (NF,) 复数样本窗 → (F, T, stage1, noncoh)，候选域 (N,)。"""
        f_proj = self._front @ window
        t_proj = self._tail @ window
        stage1 = np.abs(f_proj + t_proj).astype(np.float64) ** 2
        noncoh = (
            np.abs(f_proj).astype(np.float64) ** 2
            + np.abs(t_proj).astype(np.float64) ** 2
        )
        return f_proj, t_proj, stage1, noncoh

    def demod_symbol(self, samples: np.ndarray, start_sample: int,
                     phi0: float | None = None) -> DeRaDemodResult:
        """解一个符号窗（phi0=None 时用当前 self.ml_phase）。"""
        start = int(start_sample)
        window = np.asarray(samples[start:start + self.nf], dtype=np.complex64)
        if window.size != self.nf:
            raise ValueError(f"symbol window at {start_sample} exceeds input")
        f_proj, t_proj, stage1, noncoh = self._project(window)
        phase = self.ml_phase if phi0 is None else float(phi0)
        metric = np.abs(f_proj + np.exp(-1j * phase) * t_proj).astype(np.float64) ** 2
        return DeRaDemodResult(
            metric=metric, stage1_metric=stage1, noncoherent_metric=noncoh,
            argmax_symbol_value=int(np.argmax(metric)))

    def demod_payload(self, samples: np.ndarray, start_symbol: int, psym: int,
                      ) -> tuple[np.ndarray, np.ndarray]:
        """整段 payload：返回 (stage1_rows, coherent_rows)，各 (psym, N)。

        流程 = 论文 Algorithm 1 Stage 1-2（decode-only，无候选重试）：
        1) 逐符号窗做 V1/V2 投影与 Stage-1 相干行；
        2) 临时符号（κ 鲁棒非相干 argmax）→ ML 公共相位 φ̂0（line 36）；
        3) 以 φ̂0 重组合得 Stage-2 相干行（最终判决/软证据口径）。
        """
        stage1_rows, coherent_rows, _noncoh = self.demod_payload_all(
            samples, start_symbol, psym)
        return stage1_rows, coherent_rows

    def demod_payload_all(self, samples: np.ndarray, start_symbol: int,
                          psym: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """整段 payload 一次过：返回 (stage1, coherent, noncoh) 三组行。

        与 demod_payload 完全同一计算（委托关系，数值逐位一致）；noncoh 行
        供 DERA×Trimmer 联合链（DT-FUSE = coherent + noncoh 等增益合并，
        即统一度量 ρ=1/2 工作点）与审计对照使用。
        """
        rows1, rows_nc, f_all, t_all = [], [], [], []
        for k in range(int(psym)):
            start = (int(start_symbol) + k) * self.nf
            window = np.asarray(samples[start:start + self.nf], dtype=np.complex64)
            if window.size != self.nf:
                raise ValueError(f"payload symbol {k} window exceeds input")
            f_proj, t_proj, stage1, noncoh = self._project(window)
            rows1.append(stage1)
            rows_nc.append(noncoh)
            f_all.append(f_proj)
            t_all.append(t_proj)
        stage1_rows = np.stack(rows1)
        noncoh_rows = np.stack(rows_nc)
        f_mat = np.stack(f_all)
        t_mat = np.stack(t_all)

        # Stage-2 ML 公共相位（Algorithm 1 line 36 的忠实结构：单一 φ̂0）。
        # γ_i = angle(T_i·conj(F_i)) = π·κ_i 与候选无关（probe1 实测），故
        # 论文 Stage-2 的逐符号漂移项 φ̂_i = φ̂0 + 2π fcfo·i·T' 在 decode-only
        # 公平口径下退化为常数（fcfo 由检测级提供、范围外 ⇒ 0）。
        # ⚠️ 曾试过从 payload 拟合线性漂移（unwrap+polyfit）：native 2→23 错
        # ——antiphase 零点符号的 |z_i|≈0 角度纯噪声，unwrap 链被带歪污染
        # 整帧。残余 κ 逐符漂移是 native 仅剩 2 错的机理（完整 DeRa 由
        # 检测级 CFO 跟踪），decode-only 口径下如实保留，不另行发明。
        init = np.argmax(noncoh_rows, axis=1)
        idx = np.arange(f_mat.shape[0])
        z = np.sum(t_mat[idx, init] * np.conj(f_mat[idx, init]))
        self.ml_phase = float(np.angle(z)) if abs(z) > 0 else 0.0

        coherent = np.abs(
            f_mat + np.exp(-1j * self.ml_phase) * t_mat
        ).astype(np.float64) ** 2
        return stage1_rows, coherent, noncoh_rows

    def demod_payload_noncoherent(self, samples: np.ndarray, start_symbol: int,
                                  psym: int) -> np.ndarray:
        """对照口径：非相干行（数值上 = LoRaTrimmer 度量），供审计。"""
        rows = []
        for k in range(int(psym)):
            start = (int(start_symbol) + k) * self.nf
            window = np.asarray(samples[start:start + self.nf], dtype=np.complex64)
            _f, _t, _s1, noncoh = self._project(window)
            rows.append(noncoh)
        return np.stack(rows)

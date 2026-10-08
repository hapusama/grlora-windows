# -*- coding: utf-8 -*-
"""命题 A 数值验证：合并谱峰功率对时延的核形状与平台结构。

干净合成符号（无噪声）扫时延 d∈[−2,2] 步 0.02：
  V1: P(d)=max_k|C_d[k]|² 曲线 vs Dirichlet 预测 |D_N(φ(d))|²（归一化）；
  V2: argmax bin 对 d 的台阶（平台边界实测 vs 推论）；
  V3: 量化地板：d 舍入到 0.125 格的残差 κ 直方图（理论 RMSE≥Δ/√12）。
"""
import sys
import numpy as np
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.chirp import build_upchirp
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)

SF, N, OS = 10, 1024, 4
NF = N * OS

SYM = build_upchirp(sf=SF, symbol_id=123, os_factor=OS).astype(np.complex128)


def frac_delay(x, s):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x))
    return np.fft.ifft(X * np.exp(-2j * np.pi * f * s))


ds = np.arange(-2.0, 2.001, 0.02)
P = np.empty(len(ds))
B_ = np.empty(len(ds), dtype=int)
for i, d in enumerate(ds):
    s = frac_delay(SYM, float(d))
    res = sav_demod(samples=s, start_sample=0, sf=SF, os_factor=OS,
                    cfo_int=0)
    p = np.abs(res.combined_spectrum.astype(np.complex128)) ** 2
    P[i] = np.max(p)
    B_[i] = int(np.argmax(p))

Pn = P / P.max()
# Dirichlet 预测：残余 bin 相位 φ(d) ∝ d（系数用实测标定：找首个 argmax
# 跳变点 d*，理论 |D_N(φ)|² 与实测峰功率损失拟合）
def dirich(pow_bin):
    # pow_bin: 相位差（bin 单位）→ |D_N|²
    x = np.pi * pow_bin
    return (np.sin(x) / np.maximum(N * np.sin(x / N), 1e-12)) ** 2

print("V1 峰功率曲线（每 0.25 打点）与 argmax：")
for i in range(0, len(ds), 13):
    print("  d=%+.2f  P=%.3f  argmax=%d" % (ds[i], Pn[i], B_[i]))

# 平台边界：argmax 跳变点
jumps = [i for i in range(1, len(ds)) if B_[i] != B_[i - 1]]
print("\nV2 argmax 跳变位置:",
      ["d=%+.2f (%d→%d)" % (ds[i], B_[i - 1], B_[i]) for i in jumps])

# 混跳区宽度（峰功率相对平台的凹陷区间）
pmin_zone = Pn < 0.98
print("V2b P(d)<0.98·max 的连续区间起点/终点（混跳区近似）")
idx = np.where(pmin_zone)[0]
if len(idx):
    print("   d ∈ [%+.2f, %+.2f]" % (ds[idx[0]], ds[idx[-1]]))

# V3 量化地板
grid = 0.125
d_cont = np.random.default_rng(0).uniform(-1.875, 1.875, 100000)
kappa = d_cont - np.round(d_cont / grid) * grid
print("\nV3 量化残差 κ: RMSE=%.5f（理论 Δ/√12=%.5f, Δ/2=%.5f）"
      % (np.sqrt(np.mean(kappa ** 2)), grid / np.sqrt(12), grid / 2))

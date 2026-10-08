# -*- coding: utf-8 -*-
"""D2F 精确模板相关器（ref_template，2026-10-05）：原始采样完整波形匹配。

用户方案（05_docs/D2F下一代接收机设计_用户方案_20261005.md）验证表
第 1/2 行的参考实现：真值/给定参数 (ν0, ε) 下，对 payload 每符号按
连续相位公式做逐样本模板相关（含回绕分段、时间伸缩、载波 CFO），
与 DeRa port 的两段合并统计量做**同信息同噪声口径的输出 SNR 对比**。

模板相位 = gr-lora build_upchirp 的连续化（chirp.py 逐字核对）：
  φ_c(u) = 2π[ u²/(2N·os²) + (c/N − 1/2)·u/os ]        u < fold=(N−c)os
         = 2π[ u²/(2N·os²) + (c/N − 3/2)·u/os ]        u ≥ fold
频率在 fold 处跳 −B、相位连续（mod 2π）——用户模型 φ_c(u) 的离散等价。
时间伸缩：u_n = (m−M_k) + s_k，s_k = ε·(M_k−origin)（起点漂移，样本）；
own-support = u_n∈[0,NF) 的样本（边缘 ±s_k 样本归属邻符号，如实剔除）。
载波：e^{−j2πν0·m/NF}（ν0 bins，1 bin = 1/NF cycles/sample）。

输出 SNR 口径（两边同 σ̂²=N0，每样本单位模权重）：
  full: |Z_k|²/(σ̂²·M_sup)   port: |metric[k, gt+1]|²/(σ̂²·NF)
差距 = 合并结构损失（行1）+ 漂移近似损失（行2），dB。
"""
import numpy as np

N = 1024
OS = 4
NF = N * OS


def chirp_eval(c, u):
    """gr-lora 上啁啾连续相位（build_upchirp 的解析版），u 可为分数样本。
    u ≥ NF 的回绕分支由公式自带；u<0 或 u≥NF 的 own-support 由调用方裁。"""
    c = c % N
    fold = (N - c) * OS
    lin = np.where(u < fold, (c / N - 0.5) * u / OS,
                   (c / N - 1.5) * u / OS)
    return np.exp(2j * np.pi * (u * u / (2.0 * N * OS * OS) + lin))


def z_full(seg, m0, codes, nu0, eps, origin, tau0=0.0):
    """精确模板相关。seg：原始段（未旋转未重采样）；m0：payload 首窗
    绝对样本；codes：真值码值+1 (psym,)；ν0 bins；ε 时间伸缩；origin：
    注入/漂移历元（绝对样本）；tau0：sub-sample STO（样本，Eq.21 TO 项
    的来源——两段频率差 B ⇒ 相对相位 2π·τ0/os，必须进模板）。
    返回 (Z (psym,), M_sup (psym,))。"""
    psym = len(codes)
    Z = np.zeros(psym, dtype=np.complex128)
    Msup = np.zeros(psym)
    n_rel = np.arange(NF)
    for k in range(psym):
        Mk = m0 + k * NF
        s_k = eps * (Mk - origin) - tau0
        u = n_rel + s_k
        own = (u >= 0) & (u < NF)
        uu = u[own]
        m = Mk + n_rel[own]
        q = chirp_eval(codes[k], uu) * np.exp(
            +2j * np.pi * nu0 * (m - m0) / NF)
        Z[k] = np.sum(seg[m] * np.conj(q))
        Msup[k] = own.sum()
    return Z, Msup


def est_tau0(seg, m0, codes, nu0, eps, origin, n0):
    """τ0 一维搜索：max Σ_k |Z_k(τ0)|²/(M_sup)（干净信号上估计后冻结）。
    搜索域 ±2·os 采样（对称——负 STO 存在，f9 实测 −1.15）。"""
    grid = np.arange(-2.0 * OS, 2.0 * OS, 0.05)
    best, btau = None, 0.0
    for t in grid:
        Z, Msup = z_full(seg, m0, codes, nu0, eps, origin, tau0=float(t))
        v = float(np.sum(np.abs(Z) ** 2 / Msup))
        if best is None or v > best:
            best, btau = v, float(t)
    return btau


def snr_gap_experiment(seg, f, nu0, eps, origin, dd, n0):
    """行1/2 主对比：同帧上 full-template vs DeRa port 合并的逐符号
    输出 SNR（dB）。seg：已按 (δ) 注入的段；nu0/eps：本实验参数；
    返回 dict(gap_db 数组, snr_full, snr_port)。"""
    gt = np.asarray(f["gt"])
    psym = len(gt)
    m0 = f["hs"] + 8 * NF
    Z, Msup = z_full(seg, m0, gt, nu0, eps, origin)
    snr_full = np.abs(Z) ** 2 / (n0 * Msup)
    # port：同信息（ν0 旋掉）+ 自己的 ML φ̂0 合并；真值 bin 处取值
    pre = f["pre"]
    pad = (pre + 5) * NF
    seg_p = np.concatenate((np.zeros(pad, dtype=np.complex128),
                            seg[m0:]))
    base = seg_p * np.exp(-2j * np.pi * nu0 * np.arange(len(seg_p)) / NF)
    _s1, coh = dd.demod_payload(base, pre + 5, psym)
    snr_port = coh[np.arange(psym), (gt + 1) % N] / (n0 * NF)
    gap = 10 * np.log10(np.maximum(snr_full, 1e-30)) - \
        10 * np.log10(np.maximum(snr_port, 1e-30))
    return dict(gap_db=gap, snr_full=snr_full, snr_port=snr_port)

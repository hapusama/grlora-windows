# -*- coding: utf-8 -*-
"""DeRa port 修复探针 2：验证"逐候选越界切分 + 相干合并 + ML 公共相位"全设计。

设计（对照论文 Algorithm 1 Stage 1-2，映射到公平输入口径）：
  Stage-1 行:  |F(k) + T(k)|^2            （确定相位项=0：CFO/TO 已校平）
  初始化:      argmax(|F|^2 + |T|^2)      （kappa 鲁棒的临时符号，仅用于 ML 初始化）
  Stage-2 ML:  phi0 = angle(sum_i T_i[s_i] conj(F_i[s_i]))   （论文 line 36 结构）
  最终行:      |F(k) + e^{-j phi0} T(k)|^2
对照：TRIMMER 非相干行 |F|^2+|T|^2。
"""
import sys
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.chirp import build_upchirp
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    build_loratrimmer_matrices)

SF, N, OS = 10, 1024, 4
NF = N * OS
mat = build_loratrimmer_matrices(SF, OS)
F_MAT, T_MAT = mat.front_matrix.astype(np.complex128), mat.tail_matrix.astype(np.complex128)


def rows_for_window(win, phi0=None):
    F = F_MAT @ win
    T = T_MAT @ win
    stage1 = np.abs(F + T) ** 2
    noncoh = np.abs(F) ** 2 + np.abs(T) ** 2
    if phi0 is None:
        return stage1, noncoh, F, T
    coh = np.abs(F + np.exp(-1j * phi0) * T) ** 2
    return stage1, coh, noncoh, F, T


def run_batch(kappa, snr_db=None, n_sym=256, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, N, size=n_sym)
    st1_err = coh_err = tri_err = 0
    phi0_errs = []
    for c in codes:
        win = build_upchirp(SF, int(c), OS).astype(np.complex128)
        n = np.arange(NF)
        win = win * np.exp(2j * np.pi * kappa * n / NF)
        if snr_db is not None:
            p = float(np.mean(np.abs(win) ** 2))
            noise = (rng.standard_normal(NF) + 1j * rng.standard_normal(NF)) * np.sqrt(p / 2.0 / 10 ** (snr_db / 10.0))
            win = win + noise
        stage1, noncoh, F, T = rows_for_window(win)
        tri_err += int(np.argmax(noncoh) != c)
        st1_err += int(np.argmax(stage1) != c)
        # ---- 整 payload 的 ML 在批内做：先收集，再统一估 phi0 ----
        probe = (F, T, int(c))
        phi0_errs.append(probe)
    # ML phi0（用非相干临时符号）
    z = 0j
    for F, T, c_hat_guess in phi0_errs:
        s_hat = int(np.argmax(np.abs(F) ** 2 + np.abs(T) ** 2))
        z += T[s_hat] * np.conj(F[s_hat])
    phi0 = float(np.angle(z))
    for F, T, c in phi0_errs:
        coh = np.abs(F + np.exp(-1j * phi0) * T) ** 2
        coh_err += int(np.argmax(coh) != c)
    return st1_err, coh_err, tri_err, phi0


print("== 高SNR(none) 下各 kappa：Stage1 / Stage2(ML) / TRIMMER 错误数 / phi0(应为 pi*kappa) ==")
for kap in [0.0, 0.13, 0.25, 0.4, 0.5]:
    s1, c2, tr, phi0 = run_batch(kap)
    print("kappa=%.2f: Stage1 %3d/256  Stage2 %3d/256  TRIMMER %3d/256  phi0=%+.3f (pi*k=%+.3f)"
          % (kap, s1, c2, tr, phi0, np.pi * kap))

print()
print("== SNR=-20dB, kappa=0.25：三法对比（100 符号）==")
for kap in [0.0, 0.25, 0.5]:
    s1, c2, tr, phi0 = run_batch(kap, snr_db=-20.0, n_sym=100, seed=7)
    print("kappa=%.2f SNR=-20: Stage1 %3d/100  Stage2 %3d/100  TRIMMER %3d/100" % (kap, s1, c2, tr))

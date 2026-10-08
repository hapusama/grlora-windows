# -*- coding: utf-8 -*-
"""M2d/dep5 核心库（2026-10-04 夜）：搜索税消解版可部署检测器。

设计文档：doc/dep5_搜索税消解_20261004.md。相对 dep4（m2c_core）的三处替换：
  ① δ：21 格(0.01 步) → 细轴 129 点(±0.25 步 1/256)。离格残差相位
     2π·0.002·6.75≈0.085 rad ⇒ 损失 ~0.01dB ⇒ 量化性免疫"压格点"审计
     （S1 验证）。与 keystone 剪切形态等价（d1 U3 ≤0.12dB 先例），实现
     取便宜者：dep4 骨架 + 行 FFT 吸收 (θ,κ)。
  ② 锚：提名列 + 抛物线精化（C.interp_peak）→ 连续 ν̂；aoff{±0.25}×
     osfd{5} bank 全部删除（镜像律中心由 ν̂ 连续给出）。
  ③ K：主臂按 smoke 实测召回降档（k_list 嵌套 2/8/16），K16 为消融。
统计量（同 dep4 两群结构，无 bank）：
  score = (max_κ|FFT_{rows}(√b·X_A)| + max_κ|FFT_{rows}(√b·X_B)|)²
          / (σ̂² · K_ks)，K_ks = pre+4.0（1/4 窗弃，d1 keystone 先例）。
门限：解析参考 = cfar_threshold(FAR/K, c_δ·c_κ·2群)；部署 = 经验 H0
  池化精确分位（协议 v1.1 §5A/Tier3，与 dep4/dera 同待遇）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_core as C
import m2b_core as WB
import m2c_core as MC

SF, N, OS, NF = 10, 1024, 4, 4096
NFFT = 2 * NF
N_KAP = 128                                   # κ-FFT 长度（1/128 分辨率，损失≤0.05dB）
K_MAX = 16
DGRID5 = (np.arange(129) - 64) / 256.0        # δ ∈ [−0.25, 0.25] 步 1/256


# ---------------------------------------------------------------- 布局
def ks_layout(pre):
    """全场 12.0 行（前导+2 sync+2 SFD 整窗；1/4 窗弃）。"""
    idx_all, b_all = C.frame_slots(pre)
    rows = list(range(pre + 4))
    idx = idx_all[rows]
    b = b_all[rows]
    o = np.zeros(len(rows))
    o[pre:pre + 2] = WB.O_SYNC
    sgn = np.array([1.0 if j < pre + 2 else -1.0 for j in rows])
    c_e = (pre - 1) / 2.0
    return dict(rows=rows, idx=idx, b=b, o=o, sgn=sgn,
                e=sgn * (idx - c_e), K_ks=float(b.sum()),
                selA=np.arange(pre + 2), selB=np.arange(pre + 2, pre + 4))


# ---------------------------------------------------------------- 提名+精化
def nominate_refine(segs, hs, pre, k_max=K_MAX):
    """Stage-1：池化幅度谱 top-K 列 + 抛物线精化 → 连续 ν̂（0.5 格→亚 0.1）。"""
    wins = C.field_windows(hs, pre)
    B = segs.shape[0]
    acc = np.zeros((B, 2 * N), dtype=np.float64)
    for j in range(pre):
        s, r, l = wins[j]
        X = np.fft.fft((segs[:, s:s + l] * r[None, :l]).astype(np.complex64),
                       NFFT, axis=1)
        Xc = np.concatenate((X[:, NFFT - N:], X[:, :N]), axis=1)
        acc += np.abs(Xc) ** 2
    kpart = np.argpartition(acc, -k_max, axis=1)[:, -k_max:]
    vals = acc[np.arange(B)[:, None], kpart]
    order = np.argsort(vals, axis=1)[:, ::-1]
    cols = kpart[np.arange(B)[:, None], order]
    amp = np.sqrt(acc)
    nu_ref = np.empty((B, k_max))
    for i in range(B):
        for k in range(k_max):
            nu_ref[i, k] = (C.interp_peak(amp[i], int(cols[i, k])) - N) / 2.0
    return nu_ref, cols


# ---------------------------------------------------------------- 走动 twiddle
_TWD = {}


def _twd(pre):
    """exp(−2πi·e_j·δ·n/NF)，(pre, j) → (l, nd) 缓存。"""
    if pre in _TWD:
        return
    L = ks_layout(pre)
    wins = C.field_windows(0, pre)
    n_arr = np.arange(NF)
    for j, rw in enumerate(L["rows"]):
        l = wins[rw][2]
        _TWD[(pre, j)] = np.exp(
            np.outer(-2j * np.pi * n_arr[:l] / NF, L["e"][j] * DGRID5)
        ).astype(np.complex64)


# ---------------------------------------------------------------- σ̂²（m2c 同款）
def est_sigma2(segs, wins, rows, nu_all, pre):
    subs = C._far_subgrid([])
    TWf = MC._twf()
    kc = np.rint(2.0 * nu_all + N).astype(int)
    km = np.rint(2.0 * MC.sfd_mirror_pred(nu_all, pre) + N).astype(int)
    kc = np.concatenate([kc, km], axis=1)
    B = segs.shape[0]
    mask = np.ones((B, len(subs)), dtype=bool)
    for i in range(kc.shape[1]):
        mask &= np.abs(subs[None, :] - kc[:, i:i + 1]) > C.NOISE_GUARD
    neff = mask.sum(axis=1).astype(float)
    sig2 = np.zeros(B)
    for rw in rows:
        s, r, l = wins[rw]
        if l != NF:
            continue
        Wm = (segs[:, s:s + l] * r[None, :l]).astype(np.complex64)
        sig2 += np.sum(np.abs(Wm @ TWf) ** 2 * mask, axis=1)
    return sig2 / (len(rows) * neff)


# ---------------------------------------------------------------- 验证
def verify5(segs, hs, pre, nu_ref, k_list=(8, 16), ret_diag=False):
    """Stage-2：无 bank 两群统计。nu_ref (B,K) 连续锚（nominate_refine 出）。"""
    _twd(pre)
    L = ks_layout(pre)
    wins = C.field_windows(hs, pre)
    B, K_nom = nu_ref.shape
    nd = len(DGRID5)
    n_arr = np.arange(NF)
    selA, selB = L["selA"], L["selB"]
    bA = np.sqrt(L["b"][selA])
    bB = np.sqrt(L["b"][selB])
    sig2 = est_sigma2(segs, wins, L["rows"], nu_ref, pre)

    Wrows = [segs[:, wins[rw][0]:wins[rw][0] + wins[rw][2]]
             * wins[rw][1][None, :wins[rw][2]] for rw in L["rows"]]

    best_per_nom = np.zeros((B, K_nom))
    bd_nom = np.zeros((B, K_nom))
    for r in range(K_nom):
        nu = nu_ref[:, r]
        baseA = np.exp(-2j * np.pi * nu[:, None] * n_arr[None, :] / NF
                       ).astype(np.complex64)
        sfd_c = MC.sfd_mirror_pred(nu, pre)
        baseB = np.exp(-2j * np.pi * (nu + sfd_c)[:, None]
                       * n_arr[None, :] / NF).astype(np.complex64)
        XA = np.empty((B, len(selA), nd), dtype=np.complex128)
        for jj, j in enumerate(selA):
            l = Wrows[j].shape[1]
            m = (Wrows[j] * baseA[:, :l]).astype(np.complex64)
            m *= np.exp(-2j * np.pi * L["o"][j] * n_arr[:l] / NF
                        ).astype(np.complex64)[None, :]
            XA[:, jj] = m @ _TWD[(pre, j)]
        FA = np.abs(np.fft.fft((XA * bA[None, :, None]).astype(np.complex64),
                               N_KAP, axis=1)).max(axis=1)      # (B,nd)
        XB = np.empty((B, len(selB), nd), dtype=np.complex128)
        for jj, j in enumerate(selB):
            l = Wrows[j].shape[1]
            m = ((Wrows[j] * baseB[:, :l]).astype(np.complex64))
            XB[:, jj] = m @ _TWD[(pre, j)]
        FB = np.abs(np.fft.fft((XB * bB[None, :, None]).astype(np.complex64),
                               N_KAP, axis=1)).max(axis=1)
        S = FA + FB
        d_i = S.argmax(axis=1)
        best_per_nom[:, r] = S[np.arange(B), d_i]
        bd_nom[:, r] = DGRID5[d_i]

    out = {}
    for k in k_list:
        kk = min(k, K_nom)
        sel = best_per_nom[:, :kk]
        am = sel.argmax(axis=1)
        out["score_k%d" % k] = (sel[np.arange(B), am] ** 2
                                / (sig2 * L["K_ks"]))
        if ret_diag:
            out["dhat_k%d" % k] = bd_nom[np.arange(B), am]
    if ret_diag:
        out["best_per_nom"] = best_per_nom
        out["sig2"] = sig2
        out["nu_ref"] = nu_ref.copy()
    return out


def score_dep5(segs, hs, pre, k_list=(2, 8, 16), ret_diag=False):
    nu_ref, cols = nominate_refine(segs, hs, pre)
    out = verify5(segs, hs, pre, nu_ref, k_list=k_list, ret_diag=ret_diag)
    if ret_diag:
        out["cols"] = cols
        out["K_ks"] = ks_layout(pre)["K_ks"]
    return out


# ------------------------------------------------------------ 解析门限（参考）
def thr_analytic_dep5(pre, far, k_nom=8):
    """c_δ（上穿，δ 轴 Dirichlet 相关）× c_κ（128 FFT 上穿）× 2 群 × K 联邦。
    声明：两群/细轴相关为近似 ⇒ 仅作上界参考；部署门限=经验 H0 分位。"""
    L = ks_layout(pre)
    c_d = C.upcrossing_c(L["idx"], L["b"], 256)       # δ 轴（分辨率 1/256）
    c_k = C.upcrossing_c(L["idx"], L["b"], N_KAP)     # κ 轴
    c_tot = c_d * c_k * 2.0
    g = C.cfar_threshold(far / k_nom, c_tot)
    g1 = C.cfar_threshold(far, c_tot)
    return g, dict(c_delta=c_d, c_kappa=c_k, c_total=c_tot,
                   thr_union_db=10 * np.log10(g),
                   thr_single_db=10 * np.log10(g1))

# -*- coding: utf-8 -*-
"""M2c/dep4 核心库（2026-10-04）：两级可部署检测器——非相干提名 + 相干验证。

================================================================================
【架构与证书论证（耦合税定律视角）】

dep2/dep3 的结构性亏损（M2b 实测核实）：拆分证书要求估计集 E 与确认集 C
不交（E∩C=∅ ⇒ 条件 H0 = 无条件 H0），P=8 时 12.25 场被拆 6(E)+6.25(C)，
确认场能量低于 DeRa port 的 8 前导行（10log10(8/6.25)=1.07dB），外加采集
墙（相干 6 行 vs 2048 列 Gumbel 极值的深端截断）。M2b 战表原始交点：
dep3a/b 在 Pd09 上落后 DeRa 1.0~1.3dB（δ≤0.02 全 P 组）——拆分路线的
能量账封顶。

dep4 把"锚"从能量账里拿掉：
  Stage-1（提名，非相干）：前导行 |X_j(k)|² 平均 → 列能量图 → top-K 列
    （K=16 计算，K=8 为嵌套子臂）。无 θ、无 κ̂、无相干锚、无 walk bank——
    成本 = P 个 FFT（与 DeRa 的提名图同源，可比性干净）。
  Stage-2（验证，相干）：对每个提名列 ν_c=(k−N)/2，全场 pre+4.25 行做
    dep3b 式两组边缘化相干统计（A=前导+sync 组 κ-FFT；B=SFD 组独立
    κ_s-FFT，镜像律 o_sfd=K̄−2ν_c 中心 bank），固定格网
    aoff(±0.25,3) × osfd(±0.5,5) × δ(±0.10 步 0.01,21) × κ(256)，
    score=(max|A|+max|B|)²/(σ̂²·K_c)，K_c=pre+4.25（P=8 即 12.25——
    全场完整进验证，行不再被估计集消耗）。aoff 在 A/B 两组间绑定
    （同一锚频偏，与 dep3b 一致）。

证书分层（同 DeRa 待遇 + 解析上界）：
  Tier1（逐格精确）：给定提名人选（离散列指标）+ 固定格网，验证统计量是
    数据无关的固定线性变换族 ⇒ 逐格 Exp(1) 精确——耦合税定律的"固定族"侧
    （权重⊥样本在"给定提名"条件意义下成立）。
  Tier2（提名 union）：K 个提名的多重性 ⇒ Bonferroni 解析上界
    thr(far) = cfar_threshold(far/K, c_tot_单提名)。
  Tier3（选择耦合残差）：Stage-1 能量选择与 Stage-2 幅度共享行——该分量
    无解析闭式，与 DeRa port 自身的 argmax 列选择同构（存在性先例：DeRa
    全系战表均以 battle-H0 池化精确分位为其门限）。部署门限 = 各臂自己的
    H0 池精确分位（协议标准，四臂一致）；解析值作上界申报。
================================================================================
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C
import m2_core as M
import m2b_core as W

SF, N, OS, NF = 10, 1024, 4, 4096
NFFT = 2 * NF
N_FINE = 256
K_MAX = 16                                  # 提名数上限（K=8 嵌套子臂）
OSFD_BANK = np.arange(-0.5, 0.5001, 0.25)   # 镜像律中心 bank（5 值）
DGRID4 = np.round(np.arange(-0.10, 0.1001, 0.01), 4)   # 验证 δ 格（21）
AOFFS = W.ANCHOR_OFF                        # (−0.25, 0, +0.25)


def kap_of_q(q):
    return M.kap_of_q(q)


# ---------------------------------------------------------------- 布局
def full_layout(pre):
    """验证场 = 全部行（前导 + 2 sync + SFD 2.25）。c_e 取前导中心
    （固定常数——任何与数据无关的参考都保持固定变换性质）。"""
    idx_all, b_all = C.frame_slots(pre)
    rows = list(range(pre + 5))
    idx = idx_all[rows]
    b = b_all[rows]
    o = np.zeros(len(rows))
    o[pre:pre + 2] = W.O_SYNC
    sgn = np.array([1.0 if j < pre + 2 else -1.0 for j in rows])
    return dict(rows=rows, idx=idx, b=b, o=o, sgn=sgn,
                c_e=(pre - 1) / 2.0, K_c=float(b.sum()),
                selA=np.arange(pre + 2), selB=np.arange(pre + 2, pre + 5))


def sfd_mirror_pred(nu0h, pre):
    """镜像律 o_sfd ≈ K̄ − 2ν̂0（wrap 到 (−N/2, N/2]）。"""
    return ((W.KMIRROR[pre] - 2.0 * np.asarray(nu0h) + W.SFD_HALF_SPAN)
            % N) - W.SFD_HALF_SPAN


# ------------------------------------------------------------ 全局缓存
_TWF = None
_TWA = {}      # {(pre, j, ia): (l, nd) A 组复合 twiddle（walk+aoff+o_j）}
_TWB = {}      # {(pre, j, ia): (l, nd) B 组复合 twiddle（walk+aoff）}
_EOS = None    # osfd bank 的时域相位向量（5, NF）


def _twf():
    global _TWF
    if _TWF is None:
        subs = C._far_subgrid([])
        n = np.arange(NF)
        _TWF = np.exp(-2j * np.pi * (subs - N)[:, None] * n[None, :] / NF
                      / 2.0).astype(np.complex64).T
    return _TWF


def _tw_cache(pre):
    """复合 twiddle：exp(−2πi(walk_δ + Δ)·n/NF)，Δ = aoff(+o_j)。

    将锚/偏移相位折进逐列 twiddle ⇒ 每提名只需一个基带 exp。"""
    if pre in _TWA:
        return
    L = full_layout(pre)
    wins = C.field_windows(0, pre)
    n_arr = np.arange(NF)
    for j, rw in enumerate(L["rows"]):
        l = wins[rw][2]
        n4 = n_arr[:l]
        w = L["sgn"][j] * (L["idx"][j] - L["c_e"]) * DGRID4   # (nd,)
        inB = j >= pre + 2
        for ia, aoff in enumerate(AOFFS):
            d = w + (aoff if inB else aoff + L["o"][j])
            m = np.exp(np.outer(-2j * np.pi * n4 / NF, d)) \
                .astype(np.complex64)                        # (l, nd)
            if inB:
                _TWB[(pre, j, ia)] = m
            else:
                _TWA[(pre, j, ia)] = m


def _eos():
    global _EOS
    if _EOS is None:
        n = np.arange(NF)
        _EOS = [np.exp(-2j * np.pi * oS * n / NF).astype(np.complex64)
                for oS in OSFD_BANK]
    return _EOS


# ---------------------------------------------------------------- 提名
def nominate(segs, hs, pre, k_max=K_MAX):
    """Stage-1：前导行非相干列能量 → top-K 提名列（能量降序）。

    返回 nu_c (B,K)、cols (B,K)。成本 = pre 个 8192-FFT（与 DeRa 的
    列图同源，无相干操作、无 walk bank）。"""
    wins = C.field_windows(hs, pre)
    B = segs.shape[0]
    acc = np.zeros((B, 2 * N), dtype=np.float32)
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
    nu_c = (cols.astype(np.float64) - N) / 2.0
    return nu_c, cols


# ---------------------------------------------------------------- 验证
def verify(segs, hs, pre, nu_c, k_list=(8, 16), ret_diag=False):
    """Stage-2：逐提名相干验证（全场行，dep3b 式两组边缘化，aoff 绑定）。

    返回 dict（每 k_list 档位一个 score_kK 键 = 前 K 提名的 max 分数）。"""
    _tw_cache(pre)
    L = full_layout(pre)
    wins = C.field_windows(hs, pre)
    B = segs.shape[0]
    selA, selB = L["selA"], L["selB"]
    nd = len(DGRID4)
    n_arr = np.arange(NF)
    K_nom = nu_c.shape[1]
    eos = _eos()

    # ---- σ̂²：全窗行远子格，护带围绕全部提名列簇 + 其 SFD 镜像簇 ----
    subs = C._far_subgrid([])
    TWf = _twf()
    nfull = sum(1 for rw in L["rows"] if wins[rw][2] == NF)
    k_nom_cols = np.rint(2.0 * nu_c + N).astype(int)
    k_mir = np.rint(2.0 * sfd_mirror_pred(nu_c, pre) + N).astype(int)
    kc = np.concatenate([k_nom_cols, k_mir], axis=1)
    mask = np.ones((B, len(subs)), dtype=bool)
    for i in range(kc.shape[1]):
        mask &= np.abs(subs[None, :] - kc[:, i:i + 1]) > C.NOISE_GUARD
    neff_cells = mask.sum(axis=1).astype(float)
    sig2 = np.zeros(B)
    for rw in L["rows"]:
        s, r, l = wins[rw]
        if l != NF:
            continue
        Wm = (segs[:, s:s + l] * r[None, :l]).astype(np.complex64)
        sig2 += np.sum(np.abs(Wm @ TWf) ** 2 * mask, axis=1)
    sig2 /= (nfull * neff_cells)

    Wrows = [segs[:, wins[rw][0]:wins[rw][0] + wins[rw][2]]
             * wins[rw][1][None, :wins[rw][2]] for rw in L["rows"]]
    sb = np.sqrt(L["b"])

    best_per_nom = np.zeros((B, K_nom))
    bd_nom = np.zeros((B, K_nom))
    for r in range(K_nom):
        nu = nu_c[:, r]
        # 基带相位（每提名 2 个 exp）：A 组 ν_c、B 组 ν_c+o_sfd(镜像律)
        baseA = np.exp(-2j * np.pi * nu[:, None]
                       * n_arr[None, :NF] / NF).astype(np.complex64)
        sfd_c = sfd_mirror_pred(nu, pre)
        baseB = np.exp(-2j * np.pi * (nu + sfd_c)[:, None]
                       * n_arr[None, :NF] / NF).astype(np.complex64)
        WA = [(Wrows[j] * baseA[:, :Wrows[j].shape[1]]).astype(np.complex64)
              for j in selA]
        WB = [(Wrows[j] * baseB[:, :Wrows[j].shape[1]]).astype(np.complex64)
              for j in selB]
        Amax_ia = []
        for ia in range(len(AOFFS)):
            Xpre = np.empty((B, len(selA), nd), dtype=np.complex128)
            for jj, j in enumerate(selA):
                Xpre[:, jj] = WA[jj] @ _TWA[(pre, j, ia)]
            Fpre = np.abs(np.fft.fft(
                (Xpre * sb[None, selA, None]).astype(np.complex64),
                N_FINE, axis=1))
            Amax_ia.append(Fpre.max(axis=1))                # (B,nd)
        Ssum_best = np.zeros((B, nd))
        for ia in range(len(AOFFS)):
            for iS in range(len(OSFD_BANK)):
                Xsfd = np.empty((B, len(selB), nd), dtype=np.complex128)
                for jj, j in enumerate(selB):
                    Xsfd[:, jj] = (WB[jj] * eos[iS][:WB[jj].shape[1]]) \
                        @ _TWB[(pre, j, ia)]
                Fsfd = np.abs(np.fft.fft(
                    (Xsfd * sb[None, selB, None]).astype(np.complex64),
                    N_FINE, axis=1))
                Ssum_best = np.maximum(Ssum_best,
                                       Amax_ia[ia] + Fsfd.max(axis=1))
        d_i = Ssum_best.argmax(axis=1)
        best_per_nom[:, r] = Ssum_best[np.arange(B), d_i]
        bd_nom[:, r] = DGRID4[d_i]

    out = {}
    for k in k_list:
        kk = min(k, K_nom)
        sel = best_per_nom[:, :kk]
        am = sel.argmax(axis=1)
        out["score_k%d" % k] = (sel[np.arange(B), am] ** 2
                                / (sig2 * L["K_c"]))
        if ret_diag:
            out["dhat_k%d" % k] = bd_nom[np.arange(B), am]
    if ret_diag:
        out["best_per_nom"] = best_per_nom
        out["sig2"] = sig2
        out["nu_c"] = nu_c.copy()
        out["dhat_nom"] = bd_nom
    return out


def score_dep4(segs, hs, pre, k_list=(8, 16), ret_diag=False):
    """dep4 全链（提名 → 验证）。"""
    nu_c, cols = nominate(segs, hs, pre)
    out = verify(segs, hs, pre, nu_c, k_list=k_list, ret_diag=ret_diag)
    if ret_diag:
        out["cols"] = cols
        out["K_c"] = full_layout(pre)["K_c"]
    return out


# ------------------------------------------------------------ 解析门限
def thr_analytic_dep4(pre, far, k_nom=8):
    """Bonferroni-union 解析上界：单提名乘积闭式（全场布局）+ K 项 union。

    声明：乘积组合在多维下偏乐观（M2 §4 实测 −0.6~−0.9dB）且不含 Tier3
    选择耦合残差 ⇒ 本值仅为上界参考；部署门限 = battle-H0 池化精确分位。"""
    L = full_layout(pre)
    idx, b, c_e = L["idx"], L["b"], L["c_e"]
    sgn = L["sgn"]
    cup = C.upcrossing_c(idx, b, N_FINE)

    rho_d = np.zeros(len(DGRID4))
    rho_d[0] = 1.0
    for k in range(1, len(DGRID4)):
        dpj = sgn * (idx - c_e) * DGRID4[k]
        rho_d[k] = np.abs(np.sum(b * M.rho_dirichlet(dpj)) / L["K_c"])
    neff_d = M.n_eff_from_rho(rho_d)

    def shift_neff(grid):
        rk = np.abs(M.rho_dirichlet(np.asarray(grid)[1:] - grid[0]))
        return M.n_eff_from_rho(np.concatenate(([1.0], rk)))

    neff_s = shift_neff(OSFD_BANK)
    neff_a = shift_neff(AOFFS)
    # 两组边缘化：A/B 组内相干、组间相位偏移由幅度相加吸收
    bA = float(b[L["selA"]].sum())
    bB = float(b[L["selB"]].sum())
    rhoA = (bA / L["K_c"]) ** 2
    neff_gA = 2.0 / (1.0 + rhoA ** 2)
    rhoB = (bB / L["K_c"]) ** 2
    neff_gB = 2.0 / (1.0 + rhoB ** 2)

    c_tot = cup * neff_d * neff_s * neff_a * (neff_gA + neff_gB)
    g = C.cfar_threshold(far / k_nom, c_tot)
    g1 = C.cfar_threshold(far, c_tot)          # K=1 参照（union 前）
    return g, dict(c_kappa=cup, neff_delta=neff_d, neff_sfd=neff_s,
                   neff_anchor=neff_a, neff_gA=neff_gA, neff_gB=neff_gB,
                   c_total=c_tot, thr_union_db=10 * np.log10(g),
                   thr_single_db=10 * np.log10(g1),
                   union_shift_db=10 * np.log10(g / g1))

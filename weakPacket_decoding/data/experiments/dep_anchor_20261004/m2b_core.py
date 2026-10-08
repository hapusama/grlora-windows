# -*- coding: utf-8 -*-
"""M2b 核心库（2026-10-04）：dep3 = dep2 的三税修复版（拆分证书保持）。

================================================================================
【三税与修复（M2b 诊断 m2b_probe0/1/1b 驱动）】

税 1 能量转移税：dep2 P=8 的 12.25 场拆 6(E)+6.25(C)，C 里 2 行是 sync——
  诊断：sync 行是确认场最差相干行（组相位 ±51~72° 散布），前导行相互
  对齐。dep3a：E = {前 Kp 前导 + 2 sync}（sync 已知偏移 +24/+32 整数 bin
  ⇒ 列折叠精确，"双段采集"）——采集行数不变（6/12/12），C 换成
  {其余前导 + SFD 2.25}——行数不变（6.25/8.25/24.25），sync 换前导
  ⇒ 干净域相干度 +0.3~0.4dB（P8/P16δ0），采集基线加长（δ 可辨）。
税 2 采集墙：M2 报告的"δ=0.082 采集 67%@−30"是度量伪象（m2_battle 的
  nu_err 用未注入模板当 GT；真注入 GT 下 99%@−30）。双段采集真实收益在
  漂移域锚精度（δ=.082 native p50 0.137→0.062）与深端（−36：44→54%）。
税 3 类型失配：无 θ 时确认相干度 0.69-0.91（probe1b）。ψ 行间步进律
  （2π·frac(ν)）实测仅 ~45% 命中 ⇒ 弃预测路线；dep3b 改**类型相位
  边缘化 ML**（ff1 FF_ml 的拆分认证版）：确认场拆 pre 组（κ-FFT 相干）
  与 SFD 组（独立 κ_s-FFT 相干），统计量 (max|A| + max|B|)²/(σ̂²K_c)——
  组内全相干（组内线性相位自由度吸收 intra-sfd 步进）、组间偏移由
  幅度相加吸收（max_α|A+e^{jα}B| = |A|+|B| 闭式）。o_sfd 用镜像律
  （o_sfd = K̄−2ν0，per-capture 干净域拟合常量，与 O_SYNC 同级声明）
  中心化细格 bank（±0.75 步 0.25，7 值）。

【H0 证书】dep3a：与 dep2 同构（逐格 Exp(1) 精确 + 解析 max 族）。
dep3b：逐格分布 = max|A| + max|B|（两组独立复高斯场的相关最大值之和，
无初等闭式）⇒ Tier1 降级为"合成域精确 MC"（协方差 Dirichlet 核特征
分解 + 同款格网搜索，确定性 seed 冻结）——与 M2 部署门限同等级声明。
证书核心（E∩C=∅ ⇒ 条件 H0 = 无条件 H0、采集失败只伤 H1）两版均保持。
================================================================================
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C
import m2_core as M

SF, N, OS, NF = 10, 1024, 4, 4096
NFFT = 2 * NF
N_FINE = 256
O_SYNC = M.O_SYNC                       # (24.0, 32.0) 整数 bin（协议常量）
DGRID = M.DGRID                         # ±0.10 步 0.005（41 格）
ACQ_P0_OFF = M.ACQ_P0_OFF
ACQ_DGRID = M.ACQ_DGRID
ANCHOR_OFF = M.ANCHOR_OFF               # ±0.25 步 0.25（3）
CONJ_TRY = (False, True)
_KAP_AXIS = np.fft.fftfreq(N_FINE)

# 双段估计集：前 Kp 前导 + 2 sync（行数 = dep2 的 K_a：6/12/12）
ACQ_PRE = {8: 4, 16: 10, 32: 10}
# per-capture 镜像常数 K̄（o_sfd ≈ K̄ − 2·ν0；28 帧干净域拟合。声明：协议
# 常量级（跨帧拟合残差见表），与 O_SYNC=24/32 / SFD_OFFS 盲域设计同级）
KMIRROR = {8: -41.05, 16: -47.10, 32: -48.89}
SFD_HALF_SPAN = 0.5 * N                 # wrap 半域（o_sfd 实测 [0.4,3.0]）


def kap_of_q(q):
    return M.kap_of_q(q)


# ------------------------------------------------------------ 全局缓存
_TWF = None            # σ̂² 远子格 DTFT 表（常量）
_TWW = {}              # 走动 twiddle {(pre): [TWw_j]}


def _twf():
    global _TWF
    if _TWF is None:
        subs = C._far_subgrid([])
        n = np.arange(NF)
        _TWF = np.exp(-2j * np.pi * (subs - N)[:, None] * n[None, :] / NF
                      / 2.0).astype(np.complex64).T        # (NF, n_sub)
    return _TWF


def _tww(pre):
    if pre not in _TWW:
        L = confirm_layout(pre)
        wins = C.field_windows(0, pre)          # 长度/参考与 start 无关
        idx, c_e = L["idx"], L["c_e"]
        n_arr = np.arange(NF)
        out = []
        for j in range(len(L["rows"])):
            sgn_j = -1.0 if j in L["sfd_sel"] else 1.0
            w = sgn_j * (idx[j] - c_e) * DGRID
            l = wins[L["rows"][j]][2]
            out.append(np.exp(np.outer(-2j * np.pi * n_arr[:l] / NF, w))
                       .astype(np.complex64))
        _TWW[pre] = out
    return _TWW[pre]


# ------------------------------------------------------------------ 布局
def est_layout(pre):
    """估计集 E：前 Kp 前导 + 2 sync。返回 (rows, idx_e, c_e, o_e)。"""
    kp = ACQ_PRE[pre]
    rows = list(range(kp)) + [pre, pre + 1]
    idx_e = np.array(rows, dtype=float)
    o_e = np.zeros(len(rows))
    o_e[kp:] = O_SYNC
    return rows, idx_e, float(idx_e.mean()), o_e


def confirm_layout(pre):
    """确认集 C：其余前导 + SFD(2+0.25)。无 sync 行（dep3a 核心）。"""
    kp = ACQ_PRE[pre]
    idx_all, b_all = C.frame_slots(pre)
    rows = list(range(kp, pre)) + [pre + 2, pre + 3, pre + 4]
    idx = idx_all[rows]
    b = b_all[rows]
    o = np.zeros(len(rows))
    _, _, c_e, _ = est_layout(pre)
    return dict(kp=kp, rows=rows, idx=idx, b=b, o=o, c_e=c_e,
                K_c=float(b.sum()),
                sfd_sel=np.array([j for j, rw in enumerate(rows)
                                  if rw >= pre + 2]),
                pre_sel=np.array([j for j, rw in enumerate(rows)
                                  if rw < pre]))


def sfd_mirror_pred(nu0h, pre):
    """o_sfd 镜像律预测（wrap 到 (−N/2, N/2]）：K̄ − 2ν̂0。"""
    return ((KMIRROR[pre] - 2.0 * np.asarray(nu0h) + SFD_HALF_SPAN)
            % N) - SFD_HALF_SPAN


# ------------------------------------------------------------------ 采集
def acquire(segs, hs, pre):
    """双段采集（无门限）：Kp 前导 + 2 sync 联合相干列搜索 + DTFT 细化。

    sync 行谱按 2·o_sync 整数列 roll（精确）并入；走动 bank + κ-FFT 同
    dep2。ν̂0 以 E 质心 c_e 为参考（走动对称吸收）。"""
    rows, idx_e, c_e, o_e = est_layout(pre)
    wins = C.field_windows(hs, pre)
    B = segs.shape[0]
    ka = len(rows)
    Xs = np.empty((B, ka, 2 * N), dtype=np.complex64)
    sig2 = np.zeros(B)
    n_arr = np.arange(NF)
    TWf = _twf()                                   # (NF, n_sub) 全局缓存
    for q, j in enumerate(rows):
        s, r, l = wins[j]
        W = (segs[:, s:s + l] * r[None, :l]).astype(np.complex64)
        X = np.fft.fft(W, NFFT, axis=1)
        Xc = np.concatenate((X[:, NFFT - N:], X[:, :N]), axis=1)
        Xs[:, q] = np.roll(Xc, -int(round(2.0 * o_e[q])), axis=1)
        sig2 += np.sum(np.abs(W @ TWf) ** 2, axis=1) / TWf.shape[1]
    sig2 /= ka

    Ekap = np.exp(-2j * np.pi * np.outer(_KAP_AXIS, idx_e - c_e)) \
        .astype(np.complex64)
    cols = np.arange(2 * N)
    best = np.full(B, -1.0)
    k_pre = np.zeros(B, dtype=np.int64)
    d_pre = np.zeros(B)
    for dg in ACQ_DGRID:
        sh = np.rint(2.0 * (idx_e - c_e) * float(dg)).astype(int)
        Xsh = np.empty_like(Xs)
        for q in range(ka):
            Xsh[:, q] = Xs[:, q][:, (cols + sh[q]) % (2 * N)]
        T = np.abs(np.matmul(Ekap, Xsh)) ** 2      # (B,256,2048)
        flat = T.reshape(B, -1)
        kf = np.argmax(flat, axis=1)
        v = flat[np.arange(B), kf]
        upd = v > best
        best = np.where(upd, v, best)
        k_pre = np.where(upd, (kf % (2 * N)).astype(np.int64), k_pre)
        d_pre = np.where(upd, float(dg), d_pre)

    k0 = (k_pre.astype(float) - N) / 2.0
    nu0h = k0.copy()
    khat = np.zeros(B)
    Wrows = [segs[:, wins[j][0]:wins[j][0] + wins[j][2]]
             * wins[j][1][None, :wins[j][2]] for j in rows]
    for poff in ACQ_P0_OFF:
        p0 = k0 + poff
        Xd = np.empty((B, ka), dtype=np.complex128)
        for q, j in enumerate(rows):
            pj = p0 + o_e[q] + (idx_e[q] - c_e) * d_pre
            tw = np.exp(-2j * np.pi * pj[:, None] * n_arr[None, :NF] / NF)
            Xd[:, q] = np.einsum("bl,bl->b", Wrows[q], tw)
        F = np.abs(np.fft.fft(Xd, N_FINE, axis=1))
        q_ = np.argmax(F, axis=1)
        v = F[np.arange(B), q_] ** 2
        upd = v > best
        best = np.where(upd, v, best)
        nu0h = np.where(upd, p0, nu0h)
        khat = np.where(upd, kap_of_q(q_), khat)
    return dict(nu0h=nu0h, khat=khat, acq_score=best / (ka * sig2),
                c_e=c_e, ka=ka, sig2_a=sig2, d_pre=d_pre)


# ------------------------------------------------------------------ 确认
def confirm(segs, hs, pre, nu0h, mode="dep3a", ret_diag=False):
    """dep3 确认。mode：
      'dep3a'：单和（前导+SFD 相干合并），SFD blind bank [0.5,3.5] 步 0.5；
      'dep3b'：两组边缘化 (max|A|+max|B|)²/K_c，SFD 镜像律中心 bank
               ±0.75 步 0.25；A=前导组 κ-FFT，B=SFD组独立 κ_s-FFT。

    score 逐格（dep3a）Exp(1) 精确；dep3b 逐格 = 相关 max 之和（无初等
    闭式，合成域 MC 精确，见 field_mc）。"""
    L = confirm_layout(pre)
    wins = C.field_windows(hs, pre)
    B = segs.shape[0]
    rows, idx, b, c_e = L["rows"], L["idx"], L["b"], L["c_e"]
    nw = len(rows)
    nd = len(DGRID)
    n_arr = np.arange(NF)
    sfd_sel, pre_sel = L["sfd_sel"], L["pre_sel"]

    if mode == "dep3b":
        sfd_c = sfd_mirror_pred(nu0h, pre)
        off_grid = np.arange(-0.75, 0.751, 0.25)
        osfd_list = [sfd_c + o for o in off_grid]
    else:
        osfd_list = [np.full(B, o) for o in M.SFD_OFFS]

    # ---- σ̂²：确认整窗行远子格，护带围绕锚/SFD 列簇 ----
    subs = C._far_subgrid([])
    TWf = _twf()                                   # 全局缓存 (NF, n_sub)
    nfull = sum(1 for rw in rows if wins[rw][2] == NF)
    k_nu = np.rint(2.0 * nu0h + N).astype(int)
    k_sfd = np.rint(2.0 * (nu0h + 3.0) + N).astype(int)
    mask = (np.abs(subs[None, :] - k_nu[:, None]) > C.NOISE_GUARD) \
        & (np.abs(subs[None, :] - k_sfd[:, None]) > C.NOISE_GUARD)
    neff_cells = mask.sum(axis=1).astype(float)
    sig2 = np.zeros(B)
    for rw in rows:
        s, r, l = wins[rw]
        if l != NF:
            continue
        Wm = (segs[:, s:s + l] * r[None, :l]).astype(np.complex64)
        sig2 += np.sum(np.abs(Wm @ TWf) ** 2 * mask, axis=1)
    sig2 /= (nfull * neff_cells)

    # ---- 走动 twiddle：全局缓存（只依赖 pre）----
    TWw = _tww(pre)
    Wrows = [segs[:, wins[rw][0]:wins[rw][0] + wins[rw][2]]
             * wins[rw][1][None, :wins[rw][2]] for rw in rows]
    sb = np.sqrt(b)
    npre = len(pre_sel)

    bestS = np.zeros(B)
    bd = np.zeros(B)
    bk = np.zeros(B)
    bks = np.zeros(B)
    bconj = np.zeros(B, dtype=bool)
    # κ-FFT 行向线性 ⇒ 前导组谱/FFT 仅依赖 aoff（3 组合），SFD 组按
    # (conj,aoff,osfd)。dep3b 的 conj 维冗余（|FFT(conj x)| = |FFT(x)|
    # 的 q 镜像 ⇒ max_κs 不变）⇒ 跳过 conj（21 组合）——解析恒等式，
    # 非近似。dep3a 保留 conj。
    conj_list = CONJ_TRY if mode == "dep3a" else (False,)
    for aoff in ANCHOR_OFF:
        Xpre = np.empty((B, npre, nd), dtype=np.complex128)
        for jj, j in enumerate(pre_sel):
            bp = np.exp(-2j * np.pi * (nu0h + aoff)[:, None]
                        * n_arr[None, :Wrows[j].shape[1]] / NF)
            Xpre[:, jj] = (Wrows[j] * bp).astype(np.complex64) @ TWw[j]
        Fpre = np.fft.fft((Xpre * sb[None, pre_sel, None])
                          .astype(np.complex64), N_FINE, axis=1)
        Apre = np.abs(Fpre)                                   # (B,256,nd)
        Sa_g = Apre.max(axis=1) if mode == "dep3b" else None   # (B,nd)
        qa_g = Apre.argmax(axis=1) if mode == "dep3b" else None
        for conj in conj_list:
            for iS in range(len(osfd_list)):
                osfd = osfd_list[iS]
                Xsfd = np.empty((B, len(sfd_sel), nd), dtype=np.complex128)
                for jj, j in enumerate(sfd_sel):
                    bp = np.exp(-2j * np.pi
                                * (nu0h + aoff + osfd)[:, None]
                                * n_arr[None, :Wrows[j].shape[1]] / NF)
                    Xsfd[:, jj] = (Wrows[j] * bp) \
                        .astype(np.complex64) @ TWw[j]
                if conj:
                    Xsfd = np.conj(Xsfd)
                Fsfd = np.fft.fft((Xsfd * sb[None, sfd_sel, None])
                                  .astype(np.complex64), N_FINE, axis=1)
                # SFD 行在全场栈中的位置偏移（npre..）：行向 FFT 的相位
                # e^{−2πiq·npre/NF} 补回（dep3a 单和需要；dep3b 的独立
                # max 对群内相位斜移不变，但统一补上保持 κ̂s 语义）
                Fsfd = Fsfd * np.exp(-2j * np.pi * npre
                                     * np.arange(N_FINE) / N_FINE)[None, :,
                                                                   None]
                Asfd = np.abs(Fsfd)
                if mode == "dep3a":
                    F = np.abs(Fpre + Fsfd)       # 单和：复数 FFT 相加后取模
                    af = F.reshape(B, -1)
                    kfl = af.argmax(axis=1)
                    S = af[np.arange(B), kfl]
                    q_i, d_i = np.unravel_index(kfl, (N_FINE, nd))
                    kfs = q_i
                else:
                    Sb = Asfd.max(axis=1)
                    Ssum = Sa_g + Sb
                    S = Ssum.max(axis=1)
                    d_i = Ssum.argmax(axis=1)
                    q_i = qa_g[np.arange(B), d_i]
                    kfs = Asfd.argmax(axis=1)[np.arange(B), d_i]
                upd = S > bestS
                if not upd.any():
                    continue
                for i in np.nonzero(upd)[0]:
                    bestS[i] = S[i]
                    bd[i] = DGRID[d_i[i]]
                    bk[i] = kap_of_q(q_i[i])
                    bconj[i] = conj
                    bks[i] = kap_of_q(kfs[i])
    score = bestS ** 2 / (sig2 * L["K_c"])
    if ret_diag:
        return score, bd, bk, bks, dict(sig2=sig2, layout=L,
                                        conj=bconj, nu0h=nu0h.copy())
    return score, bd, bk, bks


def score_dep3(segs, hs, pre, mode="dep3a", ret_diag=False):
    """dep3 全链（双段采集 → 确认）。mode='dep3a'/'dep3b'。"""
    acq = acquire(segs, hs, pre)
    out = confirm(segs, hs, pre, acq["nu0h"], mode=mode, ret_diag=ret_diag)
    if ret_diag:
        sc, bd, bk, bks, diag = out
        diag.update(acq)
        return sc, bd, bk, bks, diag
    return out + (acq,)


# ------------------------------------------------------------ 门限族（dep3a）
def thr_analytic_dep3a(pre, far):
    """dep3a 解析门限（同 m2_core.thr_analytic 结构，布局换 dep3 确认集）。"""
    L = confirm_layout(pre)
    idx, b, c_e = L["idx"], L["b"], L["c_e"]
    sgn = np.where(np.isin(np.arange(len(idx)), L["sfd_sel"]), -1.0, 1.0)
    cup = C.upcrossing_c(idx, b, N_FINE)

    rho_d = np.zeros(len(DGRID))
    rho_d[0] = 1.0
    for k in range(1, len(DGRID)):
        dpj = sgn * (idx - c_e) * DGRID[k]
        rho_d[k] = np.abs(np.sum(b * M.rho_dirichlet(dpj)) / L["K_c"])
    neff_d = M.n_eff_from_rho(rho_d)

    def shift_neff(grid):
        rk = np.abs(M.rho_dirichlet(np.asarray(grid)[1:] - grid[0]))
        return M.n_eff_from_rho(np.concatenate(([1.0], rk)))

    neff_s = shift_neff(M.SFD_OFFS)
    neff_a = shift_neff(ANCHOR_OFF)
    b_main = float(b[L["pre_sel"]].sum())
    rho_c = (b_main / L["K_c"]) ** 2
    neff_c = 2.0 / (1.0 + rho_c ** 2)

    c_tot = cup * neff_d * neff_s * neff_a * neff_c
    g = C.cfar_threshold(far, c_tot)
    return g, dict(c_kappa=cup, neff_delta=neff_d, neff_sfd=neff_s,
                   neff_anchor=neff_a, neff_conj=neff_c, c_total=c_tot,
                   thr_db=10 * np.log10(g))


# ------------------------------------------------------------ 合成域 MC
_MC_CACHE = {}


def _field_positions3(pre, mode):
    """每确认行位置数组（相对锚；per (anchor,sfd) 块 × δ 全格）。"""
    key = ("pos3", pre, mode)
    if key in _MC_CACHE:
        return _MC_CACHE[key]
    L = confirm_layout(pre)
    rows, idx, sfd_sel, c_e = (L["rows"], L["idx"], L["sfd_sel"],
                               L["c_e"])
    offs = M.SFD_OFFS if mode == "dep3a" else np.arange(-0.75, 0.751, 0.25)
    pos = []
    for j, rw in enumerate(rows):
        ps = []
        for aoff in ANCHOR_OFF:
            for oS in offs:
                if j in sfd_sel:
                    ps.append(aoff + oS - (idx[j] - c_e) * DGRID)
                else:
                    ps.append(aoff + (idx[j] - c_e) * DGRID)
        pos.append(np.concatenate([np.asarray(p) for p in ps]))
    _MC_CACHE[key] = pos
    return pos


def _field_factor3(pre, j, mode, rank_tol=2e-3):
    key = ("fac3", pre, j, mode)
    if key in _MC_CACHE:
        return _MC_CACHE[key]
    pos = _field_positions3(pre, mode)[j]
    d = pos[:, None] - pos[None, :]
    Cov = M.rho_dirichlet(d)
    w, V = np.linalg.eigh(Cov)
    r = int(np.sum(w > rank_tol * w[-1]))
    fac = V[:, -r:] * np.sqrt(w[-r:])
    _MC_CACHE[key] = (fac, r)
    return fac, r


def field_mc3(pre, n_units, mode="dep3a", seed=20261004):
    """dep3 确认统计量 H0 合成域 MC（协方差精确 + 格网搜索同真实实现）。"""
    rng = np.random.default_rng(seed)
    L = confirm_layout(pre)
    rows = L["rows"]
    nw = len(rows)
    nd = len(DGRID)
    npre = len(L["pre_sel"])
    facs = [_field_factor3(pre, j, mode) for j in range(nw)]
    sb = np.sqrt(L["b"])
    scores = np.empty(n_units)
    chunk = 400
    done = 0
    while done < n_units:
        m = min(chunk, n_units - done)
        Xrow = []
        for j in range(nw):
            fac, r = facs[j]
            z = (rng.standard_normal((m, r)) + 1j
                 * rng.standard_normal((m, r))) * np.sqrt(0.5)
            Xrow.append(z @ fac.T)
        best = np.zeros(m)
        n_off = len(M.SFD_OFFS) if mode == "dep3a" else 7
        for conj in CONJ_TRY:
            for ia in range(len(ANCHOR_OFF)):
                for iS in range(n_off):
                    Xall = np.empty((m, nw, nd), dtype=np.complex128)
                    for j, rw in enumerate(rows):
                        base = ((ia * n_off + iS) if j in L["sfd_sel"]
                                else ia) * nd
                        Xall[:, j] = Xrow[j][:, base:base + nd]
                    if conj:
                        Xall[:, L["sfd_sel"]] = np.conj(
                            Xall[:, L["sfd_sel"]])
                    Xall = Xall * sb[None, :, None]
                    if mode == "dep3a":
                        F = np.abs(np.fft.fft(Xall, N_FINE, axis=1))
                        best = np.maximum(best, F.max(axis=(1, 2)))
                    else:
                        Fa = np.abs(np.fft.fft(Xall[:, :npre], N_FINE,
                                               axis=1))
                        Fb = np.abs(np.fft.fft(Xall[:, npre:], N_FINE,
                                               axis=1))
                        best = np.maximum(best,
                                          Fa.max(axis=(1, 2))
                                          + Fb.max(axis=(1, 2)))
        scores[done:done + m] = best ** 2 / L["K_c"]
        done += m
    return scores

# -*- coding: utf-8 -*-
"""M2 核心库（2026-10-04）：dep2 = 拆分认证可部署检测器（前导相干锚 + 解析门限）。

任务来源：决战 DeRa D2 遗留 #1/#4。dep 旧臂 ~3dB H0 折扣全在门限侧
（13.5dB vs cert 10.5dB@1e-3），来源 = ①带噪选列锚与确认和同样本的耦合
（ff3 耦合税定律）②锚/δ×κ 格搜索膨胀 + 连续功率质心选列。

================================================================================
【设计：拆分认证（ff3 耦合税定律的检测级整装）】

  估计集 E = 前 K_a=6 个前导 chirp（采集，无门限）：
    ① 相干列搜索：T(k,κ)=Σ_j X_j(k)e^{−2πiκ(j−c_e)}（2048 半 bin 列 ×
       κ-FFT 256）——"前导相干平均锚"：K_a 行相干合成，锚方差 ÷K_a
       （任务 #1；δ∈{0}：6 行 walk ≤0.5 bin，相干损失 −0.26dB@δ=0.082，
       且 δ 的走动斜率被 κ 完全吸收 ⇒ 采集不估 δ——c_e 对称参考吸收走动）；
    ② DTFT 细化：p0 ∈ k*/2 ±1.25（步 0.25）× κ 256 → ν̂0（c_e 参考）。
    采集失败只伤 H1（锚错→无检出），不伤 H0。
  确认集 C = 其余前导 + 2 sync + 2.25 SFD（K_c = P+4.25−K_a）：
    score = max_{conj(2)×锚bank(3)×SFD镜像bank(7)×δ格(41 固定)×κ(256)}
            |Σ_{j∈C} √b_j·x_j(p_j)|² / (σ̂²·K_c)
    conj 与 SFD 镜像锚 per-capture 未知 ⇒ 固定格 bank（可部署且解析）；
    δ 用与 dep 旧臂相同的固定 DGRID（±0.10 步 0.005）。

【H0 证书（精确逐格 + 解析 max 族）】
给定锚（E 的函数），C 行谱为未条件化复高斯，T 为固定线性变换 ⇒ 逐格
score ~ Exp(1) 精确（含 √b 不等权）。公共移位 ν̂0 不改变格间相关（白噪
DTFT 平稳）⇒ 条件 H0 = 无条件 H0：
    P(max ≤ γ) ≈ exp(−c_tot·√γ·e^{−γ})，
    c_tot = c_κ(Cramér–Lindgren 上穿，d1 同款闭式) × N_eff(δ) × N_eff(SFD)
            × N_eff(锚) × N_eff(conj)   （各维谱相关 → 周期图方差公式）
合成域 MC 验证 <5%（任务 #2）。δ̂ 输出 = 确认格 argmax（任务 #3 检验）。
================================================================================
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C

SF, N, OS, NF = 10, 1024, 4, 4096
NFFT = 2 * NF
N_FINE = 256
O_SYNC = (24.0, 32.0)
DGRID = C.DGRID                          # 固定 δ 格（±0.10 步 0.005，41 格）

ACQ_K = {8: 6, 16: 12, 32: 12}   # 自适应估计集（acq95/confirm95 预算交点；见 RESULTS §2）
ACQ_DGRID = np.array([-0.082, -0.055, -0.028, 0.0, 0.028, 0.055, 0.082])  # 采集 walk bank（7 格）
ACQ_P0_OFF = np.arange(-1.25, 1.26, 0.25)
SFD_OFFS = np.arange(0.5, 3.5001, 0.5)
ANCHOR_OFF = np.array([-0.25, 0.0, 0.25])
CONJ_TRY = (False, True)
_KAP_AXIS = np.fft.fftfreq(N_FINE)


def kap_of_q(q):
    return ((np.asarray(q) / N_FINE + 0.5) % 1.0) - 0.5


# ------------------------------------------------------------------ 布局
def _confirm_layout(pre):
    ka = min(ACQ_K[pre], pre)
    idx_all, b_all = C.frame_slots(pre)
    rows = list(range(ka, pre + 5))
    idx = idx_all[rows]
    b = b_all[rows]
    o = np.zeros(len(rows))
    for j, rw in enumerate(rows):
        if pre <= rw < pre + 2:
            o[j] = O_SYNC[rw - pre]
    sgn = np.array([1.0 if rw < pre + 2 else -1.0 for rw in rows])
    b_main = float(sum(b[j] for j, rw in enumerate(rows) if rw < pre + 2))
    return dict(ka=ka, rows=rows, idx=idx, b=b, o=o, sgn=sgn,
                c_e=float(np.arange(ka).mean()), K_c=float(b.sum()),
                b_main=b_main)


# ------------------------------------------------------------------ 采集
def acquire(segs, hs, pre):
    """批量采集（无门限）。返回 dict(nu0h, khat, acq_score, c_e, ka, sig2_a)。

    ν̂0 以估计集质心 c_e 为参考（走动对称吸收）。acq_score=|T|²/(K_a·σ̂²)。"""
    wins = C.field_windows(hs, pre)
    B = segs.shape[0]
    ka = min(ACQ_K[pre], pre)
    idx_e = np.arange(ka, dtype=float)
    c_e = float(idx_e.mean())
    n_arr = np.arange(NF)

    Xs = np.empty((B, ka, 2 * N), dtype=np.complex128)
    for j in range(ka):
        s, r, l = wins[j]
        W = segs[:, s:s + l] * r[None, :l]
        X = np.fft.fft(W, NFFT, axis=1)
        Xs[:, j] = np.concatenate((X[:, NFFT - N:], X[:, :N]), axis=1)

    subs = C._far_subgrid([])
    TWf = np.exp(-2j * np.pi * (subs - N)[:, None] * n_arr[None, :] / (2.0 * NF))
    sig2 = np.zeros(B)
    for j in range(ka):
        s, r, l = wins[j]
        W = segs[:, s:s + l] * r[None, :l]
        sig2 += np.sum(np.abs(W @ TWf.T) ** 2, axis=1) / subs.size
    sig2 /= ka

    # ---- ① 相干 walk-bank 列搜索（K_a=12 时 walk 达 1.0 bin，必须补偿；
    # K_a=6 时 walk≤0.5 bin 退化无害）----
    Ekap = np.exp(-2j * np.pi * np.outer(_KAP_AXIS, idx_e - c_e))
    cols = np.arange(2 * N)
    best = np.full(B, -1.0)
    k_pre = np.zeros(B, dtype=np.int64)
    d_pre = np.zeros(B)
    for dg in ACQ_DGRID:
        sh = np.rint(2.0 * (idx_e - c_e) * float(dg)).astype(int)
        Xsh = np.empty((B, ka, 2 * N), dtype=np.complex128)
        for j in range(ka):
            Xsh[:, j] = Xs[:, j][:, (cols + sh[j]) % (2 * N)]
        T = np.abs(np.einsum("qk,bkn->bqn", Ekap, Xsh)) ** 2
        flat = T.reshape(B, -1)
        kf = np.argmax(flat, axis=1)
        v = flat[np.arange(B), kf]
        upd = v > best
        best = np.where(upd, v, best)
        k_pre = np.where(upd, (kf % (2 * N)).astype(np.int64), k_pre)
        d_pre = np.where(upd, float(dg), d_pre)

    # ---- ② DTFT 细化（p0 × κ）----
    k0 = (k_pre.astype(float) - N) / 2.0
    nu0h = k0.copy()
    khat = np.zeros(B)
    Wrows = [segs[:, wins[j][0]:wins[j][0] + wins[j][2]]
             * wins[j][1][None, :wins[j][2]] for j in range(ka)]
    for poff in ACQ_P0_OFF:
        p0 = k0 + poff
        Xd = np.empty((B, ka), dtype=np.complex128)
        for j in range(ka):
            pj = p0 + (idx_e[j] - c_e) * d_pre
            tw = np.exp(-2j * np.pi * pj[:, None] * n_arr[None, :NF] / NF)
            Xd[:, j] = np.einsum("bl,bl->b", Wrows[j], tw)
        F = np.abs(np.fft.fft(Xd, N_FINE, axis=1))
        q = np.argmax(F, axis=1)
        v = F[np.arange(B), q] ** 2
        upd = v > best
        best = np.where(upd, v, best)
        nu0h = np.where(upd, p0, nu0h)
        khat = np.where(upd, kap_of_q(q), khat)
    return dict(nu0h=nu0h, khat=khat, acq_score=best / (ka * sig2),
                c_e=c_e, ka=ka, sig2_a=sig2)


# ------------------------------------------------------------------ 确认
def confirm(segs, hs, pre, nu0h, ret_diag=False):
    """批量确认统计量。score = max 格 |T|²/(σ̂²·K_c)（逐格 Exp(1) 精确）。"""
    L = _confirm_layout(pre)
    wins = C.field_windows(hs, pre)
    B = segs.shape[0]
    rows, idx, b, o, sgn = L["rows"], L["idx"], L["b"], L["o"], L["sgn"]
    nw, c_e = len(rows), L["c_e"]
    nd = len(DGRID)
    n_arr = np.arange(NF)

    # ---- σ̂²：确认整窗行远子格，per-unit 护带围绕两锚列簇 ----
    subs = C._far_subgrid([])
    TWf = np.exp(-2j * np.pi * (subs - N)[:, None] * n_arr[None, :NF]
                 / (2.0 * NF))
    nfull = sum(1 for rw in rows if wins[rw][2] == NF)
    k_nu = np.rint(2.0 * nu0h + N).astype(int)
    k_sfd = np.rint(2.0 * (nu0h + 2.0) + N).astype(int)
    mask = (np.abs(subs[None, :] - k_nu[:, None]) > C.NOISE_GUARD) \
        & (np.abs(subs[None, :] - k_sfd[:, None]) > C.NOISE_GUARD)
    neff_cells = mask.sum(axis=1).astype(float)
    sig2 = np.zeros(B)
    for rw in rows:
        s, r, l = wins[rw]
        if l != NF:
            continue
        W = segs[:, s:s + l] * r[None, :l]
        sig2 += np.sum(np.abs(W @ TWf.T) ** 2 * mask, axis=1)
    sig2 /= (nfull * neff_cells)

    # ---- 走动 twiddle 缓存（per 行：walk = s_j(idx−c_e)·δ 格）----
    TWw = []
    for j, rw in enumerate(rows):
        l = wins[rw][2]
        w = sgn[j] * (idx[j] - c_e) * DGRID
        TWw.append(np.exp(np.outer(-2j * np.pi * n_arr[:l] / NF, w))
                   .astype(np.complex64))
    Wrows = [segs[:, wins[rw][0]:wins[rw][0] + wins[rw][2]]
             * wins[rw][1][None, :wins[rw][2]] for rw in rows]
    sb = np.sqrt(b)

    bestT = np.zeros(B, dtype=np.complex128)      # 最优格复数 T（供 φ̂0）
    bd = np.zeros(B)
    bk = np.zeros(B)
    bconj = np.zeros(B, dtype=bool)
    for conj in CONJ_TRY:
        for aoff in ANCHOR_OFF:
            for osfd in SFD_OFFS:
                Xall = np.empty((B, nw, nd), dtype=np.complex128)
                for j, rw in enumerate(rows):
                    base = (nu0h + aoff + osfd) if rw >= pre + 2 \
                        else (nu0h + aoff + o[j])
                    bp = np.exp(-2j * np.pi * base[:, None]
                                * n_arr[None, :Wrows[j].shape[1]] / NF)
                    Xall[:, j] = (Wrows[j] * bp).astype(np.complex64) @ TWw[j]
                if conj:
                    for j, rw in enumerate(rows):
                        if rw >= pre + 2:
                            Xall[:, j] = np.conj(Xall[:, j])
                Xall = Xall * sb[None, :, None]
                F = np.fft.fft(Xall, N_FINE, axis=1)
                af = np.abs(F).reshape(B, -1)
                kf = np.argmax(af, axis=1)
                v = af[np.arange(B), kf]
                q_i, d_i = np.unravel_index(kf, (N_FINE, nd))
                upd = v > np.abs(bestT)
                for i in np.nonzero(upd)[0]:
                    bestT[i] = F[i, q_i[i], d_i[i]]
                    bd[i] = DGRID[d_i[i]]
                    bk[i] = kap_of_q(q_i[i])
                    bconj[i] = conj
    score = np.abs(bestT) ** 2 / (sig2 * L["K_c"])
    if ret_diag:
        return score, bd, bk, np.angle(bestT), dict(
            sig2=sig2, layout=L, conj=bconj, nu0h=nu0h.copy())
    return score, bd, bk, np.angle(bestT)


def score_dep2(segs, hs, pre, ret_diag=False):
    """dep2 全链（采集 → 确认）。"""
    acq = acquire(segs, hs, pre)
    out = confirm(segs, hs, pre, acq["nu0h"], ret_diag=ret_diag)
    if ret_diag:
        sc, bd, bk, ph, diag = out
        diag.update(acq)
        return sc, bd, bk, ph, diag
    return out + (acq,)


# ------------------------------------------------------------ 谱相关核
def rho_dirichlet(dp):
    """NF 窗 DTFT 谱相关（dp 单位 bin，复）。|D|=|sin(πdp)/(NF·sin(πdp/NF))|。"""
    dp = np.asarray(dp, dtype=float)
    out = np.empty(dp.shape, dtype=np.complex128)
    zero = np.abs(dp - np.round(dp)) < 1e-12
    kz = np.abs(dp) < 1e-9
    out[zero] = np.where(kz[zero], 1.0, 0.0)
    nz = ~zero
    d = dp[nz]
    out[nz] = (np.exp(-1j * np.pi * d * (NF - 1) / NF)
               * np.sin(np.pi * d) / (NF * np.sin(np.pi * d / NF)))
    return out


def n_eff_from_rho(rho_mag):
    """周期图方差公式：B/(1+2Σ_{k≥1}(1−k/B)ρ_k²)。"""
    Bc = len(rho_mag)
    s = np.sum([(1 - k / Bc) * rho_mag[k] ** 2 for k in range(1, Bc)])
    return Bc / (1.0 + 2.0 * s)


def thr_analytic(pre, far):
    """dep2 确认统计量解析门限 γ（线性）+ 因子分解。"""
    L = _confirm_layout(pre)
    idx, b, sgn, c_e = L["idx"], L["b"], L["sgn"], L["c_e"]
    cup = C.upcrossing_c(idx, b, N_FINE)

    rho_d = np.zeros(len(DGRID))
    rho_d[0] = 1.0
    for k in range(1, len(DGRID)):
        dpj = sgn * (idx - c_e) * DGRID[k]
        rho_d[k] = np.abs(np.sum(b * rho_dirichlet(dpj)) / L["K_c"])
    neff_d = n_eff_from_rho(rho_d)

    def shift_neff(grid):
        rk = np.abs(rho_dirichlet(np.asarray(grid)[1:] - grid[0]))
        return n_eff_from_rho(np.concatenate(([1.0], rk)))

    neff_s = shift_neff(SFD_OFFS)
    neff_a = shift_neff(ANCHOR_OFF)
    rho_c = (L["b_main"] / L["K_c"]) ** 2
    neff_c = 2.0 / (1.0 + rho_c ** 2)

    c_tot = cup * neff_d * neff_s * neff_a * neff_c
    g = C.cfar_threshold(far, c_tot)
    return g, dict(c_kappa=cup, neff_delta=neff_d, neff_sfd=neff_s,
                   neff_anchor=neff_a, neff_conj=neff_c, c_total=c_tot,
                   thr_db=10 * np.log10(g))


# ------------------------------------------------------------ 合成域 MC
_MC_CACHE = {}


def _field_positions(pre):
    """每确认行位置数组（相对锚 ν̂0；δ 全格连续排布 per (anchor,sfd) 块）。"""
    key = ("pos", pre)
    if key in _MC_CACHE:
        return _MC_CACHE[key]
    L = _confirm_layout(pre)
    rows, idx, o, sgn, c_e = L["rows"], L["idx"], L["o"], L["sgn"], L["c_e"]
    pos = []
    for j, rw in enumerate(rows):
        ps = []
        for aoff in ANCHOR_OFF:
            if rw >= pre + 2:
                for osfd in SFD_OFFS:
                    ps.append(aoff + osfd - (idx[j] - c_e) * DGRID)
            else:
                ps.append(aoff + o[j] + (idx[j] - c_e) * DGRID)
        pos.append(np.concatenate([np.asarray(p) for p in ps]))
    _MC_CACHE[key] = pos
    return pos


def _field_factor(pre, j, rank_tol=2e-3):
    """行 j 位置集合协方差的特征截断因子 (n_pos×r)：x = z@Lᵀ 精确到 tol。"""
    key = ("fac", pre, j)
    if key in _MC_CACHE:
        return _MC_CACHE[key]
    pos = _field_positions(pre)[j]
    d = pos[:, None] - pos[None, :]
    Cov = rho_dirichlet(d)
    w, V = np.linalg.eigh(Cov)
    r = int(np.sum(w > rank_tol * w[-1]))
    fac = V[:, -r:] * np.sqrt(w[-r:])
    _MC_CACHE[key] = (fac, r)
    return fac, r


def field_mc(pre, n_units, seed=20261004):
    """确认统计量 H0 合成域 MC（协方差精确 + κ 维 256-FFT 同真实实现）。

    行 DTFT 值 = z@Lᵀ（协方差 Dirichlet 核，谱截断 tol 2e-3 ⇒ 幅值误差
    <0.1%）；格网组装/搜索与 confirm() 同款（conj×锚×SFD×δ×κ）。"""
    rng = np.random.default_rng(seed)
    L = _confirm_layout(pre)
    rows = L["rows"]
    nw = len(rows)
    nd = len(DGRID)
    facs = [_field_factor(pre, j) for j in range(nw)]
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
        for conj in CONJ_TRY:
            for ia in range(len(ANCHOR_OFF)):
                for iS in range(len(SFD_OFFS)):
                    Xall = np.empty((m, nw, nd), dtype=np.complex128)
                    for j, rw in enumerate(rows):
                        base = ((ia * len(SFD_OFFS) + iS) if rw >= pre + 2
                                else ia) * nd
                        Xall[:, j] = Xrow[j][:, base:base + nd]
                    if conj:
                        for j, rw in enumerate(rows):
                            if rw >= pre + 2:
                                Xall[:, j] = np.conj(Xall[:, j])
                    Xall = Xall * sb[None, :, None]
                    F = np.abs(np.fft.fft(Xall, N_FINE, axis=1))
                    best = np.maximum(best, F.max(axis=(1, 2)) ** 2)
        scores[done:done + m] = best / L["K_c"]
        done += m
    return scores


# ---------------------------------------------------- 耦合变体 dep2c（对照臂）
def score_dep2c(segs, hs, pre, ret_diag=False):
    """dep2c：采集 = 全前导相干（K_a=pre）；确认 = 全场 P+4.25（耦合——
    锚来自同样本 ⇒ H0 无闭式，门限 = battle H0 精确分位[与 DeRa 同等待遇]）。
    单锚（相干锚 σ≈0.05 bin，无锚 bank）× SFD bank(7) × δ(41) × κ(256)
    × conj(2)。返回 dict(score, dhat, khat, nu0h, phat0, acq)。"""
    saved = ACQ_K[pre]
    ACQ_K[pre] = pre
    acq = acquire(segs, hs, pre)
    ACQ_K[pre] = saved
    nu0h = acq["nu0h"]
    wins = C.field_windows(hs, pre)
    idx_all, b_all = C.frame_slots(pre)
    rows = list(range(0, pre + 5))
    idx = idx_all[rows]
    b = b_all[rows]
    o = np.zeros(len(rows))
    for j, rw in enumerate(rows):
        if pre <= rw < pre + 2:
            o[j] = O_SYNC[rw - pre]
    sgn = np.array([1.0 if rw < pre + 2 else -1.0 for rw in rows])
    nw = len(rows)
    c_e = acq["c_e"]
    nd = len(DGRID)
    n_arr = np.arange(NF)
    B = segs.shape[0]

    subs = C._far_subgrid([])
    TWf = np.exp(-2j * np.pi * (subs - N)[:, None] * n_arr[None, :] / (2.0 * NF))
    k_nu = np.rint(2.0 * nu0h + N).astype(int)
    k_sfd = np.rint(2.0 * (nu0h + 2.0) + N).astype(int)
    mask = (np.abs(subs[None, :] - k_nu[:, None]) > C.NOISE_GUARD) \
        & (np.abs(subs[None, :] - k_sfd[:, None]) > C.NOISE_GUARD)
    nfull = sum(1 for rw in rows if wins[rw][2] == NF)
    sig2 = np.zeros(B)
    for rw in rows:
        s, r, l = wins[rw]
        if l != NF:
            continue
        W = segs[:, s:s + l] * r[None, :l]
        sig2 += np.sum(np.abs(W @ TWf.T) ** 2 * mask, axis=1)
    sig2 /= (nfull * mask.sum(axis=1))

    TWw = []
    for j, rw in enumerate(rows):
        l = wins[rw][2]
        w = sgn[j] * (idx[j] - c_e) * DGRID
        TWw.append(np.exp(np.outer(-2j * np.pi * n_arr[:l] / NF, w))
                   .astype(np.complex64))
    Wrows = [segs[:, wins[rw][0]:wins[rw][0] + wins[rw][2]]
             * wins[rw][1][None, :wins[rw][2]] for rw in rows]
    sb = np.sqrt(b)
    bestT = np.zeros(B, dtype=np.complex128)
    bd = np.zeros(B)
    bk = np.zeros(B)
    for conj in CONJ_TRY:
        for osfd in SFD_OFFS:
            Xall = np.empty((B, nw, nd), dtype=np.complex128)
            for j, rw in enumerate(rows):
                base = (nu0h + osfd) if rw >= pre + 2 else (nu0h + o[j])
                bp = np.exp(-2j * np.pi * base[:, None]
                            * n_arr[None, :Wrows[j].shape[1]] / NF)
                Xall[:, j] = (Wrows[j] * bp).astype(np.complex64) @ TWw[j]
                if conj and rw >= pre + 2:
                    Xall[:, j] = np.conj(Xall[:, j])
            Xall = Xall * sb[None, :, None]
            F = np.fft.fft(Xall, N_FINE, axis=1)
            af = np.abs(F).reshape(B, -1)
            kf = np.argmax(af, axis=1)
            v = af[np.arange(B), kf]
            q_i, d_i = np.unravel_index(kf, (N_FINE, nd))
            upd = v > np.abs(bestT)
            for i in np.nonzero(upd)[0]:
                bestT[i] = F[i, q_i[i], d_i[i]]
                bd[i] = DGRID[d_i[i]]
                bk[i] = kap_of_q(q_i[i])
    score = np.abs(bestT) ** 2 / (sig2 * float(b.sum()))
    return dict(score=score, dhat=bd, khat=bk, nu0h=nu0h.copy(),
                phat0=np.angle(bestT), acq_score=acq["acq_score"],
                c_e=c_e, sig2=sig2)

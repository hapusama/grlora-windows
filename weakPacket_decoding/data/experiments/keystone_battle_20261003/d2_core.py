# -*- coding: utf-8 -*-
"""D2 核心库（2026-10-03 第二仗）：induced-SFO 注入器 + cert-q 模板 + dep 统计量。

任务：Battle B（漂移域检测）+ Battle C（端到端解码）。
复用第一仗 d1_core（帧结构/DTFT/CFAR 闭式/合成器）与 d1_battle（帧集/加噪/H0）。

================================================================================
【induced-SFO（E2，已获用户批准）】物理等价实现 = IQ 级分数重采样：
    y[m] = x(origin + (m - origin)·(1+eps))
窗口加窗 sinc（Lanczos-3 × sinc，17 抽头，纯 numpy）插值；信号带限 ≤ Fs/4
（OS=4 过采样）⇒ 插值误差 < -70dB（probe 自校验）。物理对应：接收机采样钟
偏 ε ⇒ 去斜 tone 每符走动 δ = eps·N bin（probe 实测校准），且跨符累积相位
含二次项 π·δ·idx²（物理律#5 的"相位二次化"）——dep 统计量必须补偿之。

【cert-q】cert 模板在干净注入信号上冻结（GT 锚定，实验B 哲学）；相对 d1 的
差异：前导相位拟合升为二次（πδ idx² 吸收），否则物理二次相位使 cert 在
δ=0.082 丢 ~2.5dB（机制级不允许）。θ 类型相位 = 二次斜坡后的类型均值角。

【dep（可部署）】无相位约定、锚列带噪、δ 网格（DGRID 41 格）+ 二次相位
补偿（经验律 q2，probe 校准 ≈1）+ κ-FFT：
    score = max_{δc∈DGRID, q} |FFT_j(√b_j·c_j·x_j(p_j(δc))·e^{∓jπq2δc·idx_j²})|²
          / (σ̂²·Σb)
锚：前导 b 加权功率 argmax（列 0.5-bin 格 + 抛物线内插→连续）ν̂0，轨迹以前导
质心 c_pre 为参考；SFD 族自己的锚 m̂（UP 去斜谱）+ 下行走动。选列事件数 = 2
（两锚）+ 41 δ 格（固定网格）+ 256 q —— H0 无闭式 ⇒ MC 门限（与 d1 声明同）。
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
DGRID = C.DGRID                    # ±0.10 步 0.005 = 41 格（§5 建议）
O_SYNC = (24.0, 32.0)              # sync-word 已知偏移（协议常量，SF10/0x34）
Q2 = 0.0                           # 二次相位律系数。probe 实测：时钟 SFO 无
                                   # 相位二次项（净 c2 ≤0.024 rad/idx² ≈ 噪声
                                   # 级，native c2=−0.019 同量级）⇒ Q2=0。
                                   # （πδ idx² 律属去斜域频率斜坡模型，见
                                   # RESULTS_B 定律5 裁决节。）
_TAPS = 8                          # 每侧 sinc 抽头数（17 抽头）


# ------------------------------------------------------------ SFO 分数重采样
def _sinc_windowed(frac, taps=_TAPS):
    """窗 sinc 分数延迟核：h(d) = sinc(d)·Lanczos(d/taps)，|d|<taps。"""
    d = (np.arange(-taps, taps + 1)[:, None] - frac[None, :])   # (2T+1, n_frac)
    h = np.sinc(d) * np.sinc(d / taps)
    return h


def resample_sfo(seg, origin, eps):
    """y[m] = x(origin + (m-origin)(1+eps))，窗 sinc 插值（eps=0 → 原样拷贝）。

    origin：走动累积起点（场首窗，本库取 field_windows 的 base）——SFO 的
    物理历元选择（ε 从 origin 起累积）。边界（首尾 taps 样本）零填充。"""
    if eps == 0.0:
        return np.array(seg, dtype=np.complex128, copy=True)
    seg = np.asarray(seg, dtype=np.complex128)
    n = np.arange(len(seg), dtype=np.float64)
    t = origin + (n - origin) * (1.0 + eps)
    base = np.floor(t).astype(np.int64)
    frac = t - base                                               # (L,)
    h = _sinc_windowed(frac, _TAPS)                               # (2T+1, L)
    pad = np.zeros(_TAPS, dtype=np.complex128)
    xs = np.concatenate((pad, seg, pad))
    idx = base[None, :] + np.arange(-_TAPS, _TAPS + 1)[:, None] + _TAPS
    idx = np.clip(idx, 0, len(xs) - 1)
    y = np.sum(xs[idx] * h, axis=0)
    out_of = (base < 0) | (base > len(seg) - 1)
    y[out_of] = 0.0
    return y


def eps_of_delta(delta):
    """δ(bin/符) → ε（一阶映射 δ = eps·N；probe 实测校准）。"""
    return float(delta) / N


# ------------------------------------------------------------ cert-q 模板
def clean_template_q(seg, hs, pre, walk_sign_dn=-1.0):
    """cert 模板（GT 锚定，含二次相位）。与 C.clean_template 的差异：
    前导相位 unwrap 后 polyfit deg=2（吸收 πδ idx²），类型 θ 为二次斜坡后
    的类型均值角。返回 dict 兼容 d1 统计量的键 + ramp2 系数。"""
    wins = C.field_windows(hs, pre)
    pos = []
    for s, ref, ln in wins:
        m = np.abs(C.row_spectrum(seg, s, ref, ln))
        pos.append((C.interp_peak(m, int(np.argmax(m))) - N) / 2.0)
    nu0_med = float(np.median(pos[:pre]))
    idx, b = C.frame_slots(pre)
    delta = float(np.polyfit(idx[:pre], pos[:pre], 1)[0])
    nu0 = float(np.mean(pos[:pre]) - delta * (pre - 1) / 2.0)
    o = np.zeros(pre + 5)
    for j in range(pre, pre + 5):
        o[j] = pos[j] - nu0 - _sgn(j, pre, walk_sign_dn) * idx[j] * delta
    sgn = np.ones(pre + 5)
    sgn[pre + 2:] = walk_sign_dn

    vals = []
    for j in range(pre + 5):
        pj = nu0 + o[j] + sgn[j] * idx[j] * delta
        vals.append(C.row_dtft(seg, wins[j][0], wins[j][1], wins[j][2], pj)[0])
    vals = np.array(vals)
    ph_pre = np.unwrap(np.angle(vals[:pre]))
    c2, c1, c0 = np.polyfit(idx[:pre], ph_pre, 2)
    ramp = c2 * idx ** 2 + c1 * idx + c0
    kappa_raw = c1 / (2 * np.pi)
    kap = kappa_raw - round(kappa_raw)

    def type_theta(conj_dn):
        th = np.zeros(pre + 5)
        tot = [float(np.abs(np.sum(vals[:pre] * np.exp(-1j * ramp[:pre]))))]
        for lo, hi, is_dn in ((pre, pre + 2, False), (pre + 2, pre + 5, True)):
            z = vals[lo:hi].copy()
            if conj_dn and is_dn:
                z = np.conj(z)
            z = z * np.exp(-1j * ramp[lo:hi])
            th[lo:hi] = np.angle(np.sum(z))
            tot.append(float(np.abs(np.sum(z))))
        return th, sum(tot)
    th_c, cc = type_theta(True)
    th_n, cn = type_theta(False)
    conj_dn = cc > cn
    theta = th_c if conj_dn else th_n
    return dict(nu0=nu0, o=o, delta=delta, conj_dn=conj_dn, theta=theta,
                pos=pos, sgn=sgn, kappa=kap, c2=float(c2))


def _sgn(j, pre, walk_sign_dn=-1.0):
    return 1.0 if j < pre + 2 else walk_sign_dn


# ------------------------------------------------------------ dep 统计量
def _centroids(pre):
    """锚参考点：c_pre=(pre−1)/2（前导质心）；c_sfd=pre+2.5（两整 SFD 窗
    中点——镜像族锚的 idx 参考）。"""
    return (pre - 1) / 2.0, pre + 2.5


def _polyfit_rows(x, y):
    """逐行一次拟合（x,y 同形 (B,n)）→ (slope, intercept) 各 (B,)。"""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    sl = np.sum(xc * yc, axis=1) / np.sum(xc * xc, axis=1)
    return sl, y.mean(axis=1) - sl * x.mean(axis=1)


# δ 网格走动 twittle 缓存（进程级，(pre,j) 常量）
_TW_WALK = {}


def _tw_walk(pre):
    """{(j): (l, n_dg) complex64 走动相位 exp(−2πj·δc·w_j·n/NF)}。"""
    if pre not in _TW_WALK:
        idx, b = C.frame_slots(pre)
        c_pre, c_sfd = _centroids(pre)
        n_arr = np.arange(NF)
        out = {}
        for j in range(len(b)):
            w_j = (idx[j] - c_pre) if j < pre + 2 else -(idx[j] - c_sfd)
            l = NF if j < pre + 4 else NF // 4
            out[j] = np.exp(np.outer(-2j * np.pi * n_arr[:l] / NF,
                                     DGRID * w_j)).astype(np.complex64)
        _TW_WALK[pre] = out
    return _TW_WALK[pre]


# σ̂² 固定远子格（d1 同口径：每 8 列取 1；锚护带列事后扣除）
_SUBS_FAR = C._far_subgrid([])
_TW_FAR = np.exp(-2j * np.pi * (_SUBS_FAR[None, :] - N)
                 * np.arange(NF)[:, None] / (2.0 * NF))       # (NF, n_sub)


def score_dep_batch(segs, hs, pre, tmpl, dgrid=DGRID, q2=Q2,
                    ret_anchor=False):
    """可部署统计量（批量）。锚带噪、无 θ、δ 网格 + 二次相位补偿 + κ-FFT。

    返回 (score, delta_hat, kappa_hat, sig2[, anchor_dict])。σ̂² = 整窗
    远列子格均值（排除两锚列，护带 ±16 列，d1 同口径）。"""
    wins = C.field_windows(hs, pre)
    idx, b = C.frame_slots(pre)
    conj = tmpl["conj_dn"]
    B = segs.shape[0]
    nw = len(b)
    c_pre, c_sfd = _centroids(pre)

    # ---- 行谱（锚 + σ̂² 用）----
    Xs = np.empty((B, nw, 2 * N), dtype=np.complex128)
    for j, (s, r, l) in enumerate(wins):
        W = segs[:, s:s + l] * r[None, :l]
        X = np.fft.fft(W, NFFT, axis=1)
        Xs[:, j] = np.concatenate((X[:, NFFT - N:], X[:, :N]), axis=1)

    pw2 = np.abs(Xs) ** 2                                     # (B, nw, 2N)

    # ---- 粗捕获 = 走动补偿非相干 bank（非相干 keystone）----
    # 诊断（d2 探针）：逐窗 argmax 在战区 SNR 不可用（−30dB 时 tone 列 4.1×
    # vs 最大噪声列 7.6×）；宽走动簇（P=32/δ=0.082）时簇功率和 argmax 也被
    # 噪声最大列压过（锚 RMS 误差 11~235 bin）。走动补偿后逐列求和把 P 窗
    # 能量聚回单列 ⇒ 非相干 (δ, k) 联合 argmax 同时给锚与 δ 先验。
    # bank 用稀疏 δ 子格（步 0.02，锚质心对残余走动 ≤±0.5 bin 对称不偏）。
    bg = np.asarray(dgrid)[::4]
    cols = np.arange(2 * N)
    pw_rows = pw2[:, :pre]                                     # (B, pre, 2N)
    best_pw = np.full(B, -1.0)
    k_pre = np.zeros(B, dtype=np.int64)
    d_pre = np.zeros(B)
    for dg in bg:
        pw = np.zeros((B, 2 * N))
        for j in range(pre):
            sh = int(round(2.0 * (idx[j] - c_pre) * float(dg)))
            pw += pw_rows[:, j][:, (cols - sh) % (2 * N)]
        k = np.argmax(pw, axis=1)
        v = pw[np.arange(B), k]
        upd = v > best_pw
        best_pw = np.where(upd, v, best_pw)
        k_pre = np.where(upd, k, k_pre)
        d_pre = np.where(upd, float(dg), d_pre)
    # 锚精化：δ=d_pre 的补偿和功率在 k_pre ±3 列内取功率质心（亚列精度）
    pw = np.zeros((B, 2 * N))
    for j in range(pre):
        sh = np.rint(2.0 * (idx[j] - c_pre) * d_pre).astype(int)
        for i in range(B):
            pw[i] += pw2[i, j][(cols - sh[i]) % (2 * N)]
    nu0h = np.empty(B)
    for i in range(B):
        lo = max(0, int(k_pre[i]) - 3)
        hi = min(2 * N, int(k_pre[i]) + 4)
        ks = np.arange(lo, hi)
        wgt = pw[i, lo:hi]
        nu0h[i] = (float(np.sum(ks * wgt) / np.sum(wgt)) - N) / 2.0

    # ---- SFD 锚：镜像走动补偿（斜率 = −d_pre）后 b 加权功率质心 ----
    pw_s = np.zeros((B, 2 * N))
    for j in range(pre + 2, nw):
        sh = np.rint(-2.0 * (idx[j] - c_sfd) * d_pre).astype(int)
        for i in range(B):
            pw_s[i] += b[j] * pw2[i, j][(cols - sh[i]) % (2 * N)]
    k_sfd = np.argmax(pw_s, axis=1)
    mh = np.empty(B)
    for i in range(B):
        lo = max(0, int(k_sfd[i]) - 3)
        hi = min(2 * N, int(k_sfd[i]) + 4)
        ks = np.arange(lo, hi)
        wgt = pw_s[i, lo:hi]
        mh[i] = (float(np.sum(ks * wgt) / np.sum(wgt)) - N) / 2.0

    # ---- σ̂²：固定远列子格（缓存表）+ 锚护带列掩码扣除（零额外建表）----
    guard_mask = np.ones((B, _SUBS_FAR.size), dtype=bool)
    for i in range(B):
        for kc in (int(k_pre[i]), int(k_sfd[i])):
            guard_mask[i] &= np.abs(_SUBS_FAR - kc) > C.NOISE_GUARD
    n_eff = guard_mask.sum(axis=1).astype(float)
    sig2 = np.zeros(B)
    sig_w = list(range(0, nw - 1, 2))
    for j in sig_w:
        s, r, l = wins[j]
        W = segs[:, s:s + l] * r[None, :l]
        Pw = np.abs(W @ _TW_FAR) ** 2                          # (B, n_sub)
        sig2 += np.sum(Pw * guard_mask, axis=1)
    sig2 /= len(sig_w) * n_eff

    # ---- δ 网格搜索（轨迹 DTFT：锚相位 exp per-unit + 走动相位缓存 TW）----
    # TW_j[n, δc] = exp(−2πj·δc·w_j·n/NF) 只依赖 (pre, j) ⇒ 进程级缓存
    # （complex64，精度 1e-7 ≪ 战区分辨率）；per-unit 只剩锚相位 exp +
    # (B,l)@(l,41) matmul。cumprod/G-查表两案已试毙（慢/无周期性）。
    o_known = np.zeros(nw)
    o_known[pre] = O_SYNC[0]
    o_known[pre + 1] = O_SYNC[1]
    n_arr = np.arange(NF)
    dga = np.asarray(dgrid, dtype=float)
    TWc = _tw_walk(pre)
    Xall = np.empty((B, nw, len(dgrid)), dtype=np.complex128)
    for j, (s, r, l) in enumerate(wins):
        base_j = (nu0h + o_known[j]) if j < pre + 2 else mh
        bp = np.exp(-2j * np.pi * base_j[:, None] * n_arr[None, :l] / NF)
        Wj = (segs[:, s:s + l] * r[None, :l] * bp).astype(np.complex64)
        Xall[:, j] = Wj @ TWc[j]
    if conj:
        Xall = Xall.copy()
        Xall[:, pre + 2:] = np.conj(Xall[:, pre + 2:])
    Xall = Xall * np.sqrt(b)[None, :, None]
    if q2 != 0.0:
        quad = np.pi * q2 * dga[None, None, :] * idx[None, :, None] ** 2
        comp = np.exp(-1j * quad)
        comp[:, pre + 2:, :] = np.conj(comp[:, pre + 2:, :])
        Xall = Xall * comp
    F = np.abs(np.fft.fft(Xall, N_FINE, axis=1))               # (B, nq, n_dg)
    flat = F.reshape(B, -1)
    k_flat = np.argmax(flat, axis=1)
    best = flat[np.arange(B), k_flat] ** 2
    q_i, d_i = np.unravel_index(k_flat, (N_FINE, len(dgrid)))
    bdk = dga[d_i]
    bkh = ((q_i / N_FINE + 0.5) % 1.0) - 0.5
    if ret_anchor:
        anch = dict(nu0h=nu0h.copy(), mh=mh.copy(), d_pre=d_pre.copy(),
                    k_pre=k_pre.copy())
        return (best / (sig2 * float(b.sum())), bdk, bkh, sig2, anch)
    return best / (sig2 * float(b.sum())), bdk, bkh, sig2


# ------------------------------------------------------------ 定律5 闭式
def dera_defocus_dirichlet(delta, pre):
    """时钟 SFO（分数重采样）模型下 DeRa 单列统计量的散焦闭式：
    固定列 k*（≈质心）上各窗幅度 |D(δ(j−(P−1)/2))|，相位线性可搜 ⇒
        L = [Σ_j |D(δ(j−(P−1)/2))| / P]²，D(x)=sinc(x)（NF 窗 Dirichlet）。
    probe P7 实测裁决：时钟 SFO 走此律（P=8/δ=.082→−0.33dB 实测 vs 本式
    −0.44dB；P=32→−5.75 vs −5.3），相位二次律不适用于时钟 SFO。"""
    j = np.arange(pre, dtype=np.float64) - (pre - 1) / 2.0
    amp = np.abs(np.sinc(float(delta) * j))
    return float(np.sum(amp) / pre) ** 2


def dera_defocus_loss(delta, pre):
    """去斜域频率斜坡模型（tone 位置走 + 持续指数相位累积）下 DeRa 散焦：
    L = max_lin |Σ_j e^{j(πδ j² − lin·j)}|² / pre²（FFT max = 最优线性移除）。
    即物理律#5 原始闭式——probe 裁决：它描述频率斜坡损伤，时钟 SFO 不走此律
    （时钟 SFO 的二次项系数 πδ²/N → 0，见 d2_core 模块注释）。"""
    j = np.arange(pre, dtype=np.float64)
    quad = np.pi * float(delta) * j ** 2
    lin = np.linspace(-np.pi, np.pi, 4001) * (pre - 1)
    ph = quad[None, :] - lin[:, None]
    amp = np.abs(np.exp(1j * ph).sum(axis=1)).max() / pre
    return float(amp ** 2)

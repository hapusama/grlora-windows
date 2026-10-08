# -*- coding: utf-8 -*-
"""D1 核心库：12.25 全已知场 keystone 相干检测器 + CFAR 门限（2026-10-03）。

蓝图：doc/去斜与采样挖掘_20261003.md §3.1-3.3（复合检测统计量）；
物理律#5（DeRa 列向 FFT=线性相位匹配滤波器，δ 散焦）；方案挖掘#12
（同 FAR 用精确分位数）。

================================================================================
【统计量】12.25 = P 前导 + 2 sync-word upchirp + 2.25 SFD downchirp
（P=8 → 13 窗 = 8+2+2 整窗 + 1 个 1/4 窗；符号时间索引 idx_j=j，末窗 12.25）

  x_j(p) = <y_j·ref_j, exp(-2πj·p·n/NF)>   （整窗精确 DTFT，p=bin 域连续位置）
  评估位置 p_j(δ) = ν0 + o_j + s_j·idx_j·δ（o_j=已知偏移：sync +24/+32、
  SFD 实测镜像、1/4 窗同 SFD 音；s_j=走动符号 ±1：upchirp +1，SFD 镜像 −1）

  T(q,δ) = Σ_j √b_j·e^{-jθ_j}·x_j(p_j(δ))·exp(-2πj·q·idx_j/N_FINE)
  score  = max_{q∈256, δ∈网格} |T|² / (σ̂²·Σb_j)
  b_j：整窗 1、1/4 窗 0.25（信号幅度∝b、噪声方差∝b ⇒ 等权 √b 求和为 ML，
  K_eff=Σb=12.25 恰为相干符号数）；θ_j=类型相位约定（协议常量级；
  battle 在干净 native 冻结——与 ff_runner 同哲学，双方同等信息）。

================================================================================
【CFAR 证书推导】（§3.2：免估计相干积累 ⇒ 虚警可证）

H0 下 y_j 复白噪 ⇒ x_j(p)=<y_j,固定向量> ~ CN(0, NF·b_j·σ²)。
给定冻结 o_j/θ_j/p_j，T(q,δ) 是 13 个独立复高斯的【固定线性组合】：
  var T = NF·σ²·Σ_j b_j·|√b_j 归一...| = NF·σ²·Σb_j（|相位因子|全为 1）
⇒ score_cell = |T|²/(σ̂²·Σb) ~ Exp(1)（(1/2)χ²₂）【逐格精确，含 1/4 窗
的加权——不等权 b_j 不破坏单格指数性，只改 K_eff】。σ² 精确时单格门限
γ=-ln(FAR)。搜索修正（max over q/δ/列）：格间相关
  ρ(Δq) = |Σ_j b_j·e^{-2πjΔq·idx_j/N_FINE}| / Σb_j
已知 ⇒ 等效独立格数（周期图方差公式，解析）
  N_eff = Q / (1 + 2·Σ_{Δ≥1}(1-Δ/Q)|ρ(Δ)|²)
闭式门限 γ(FAR) = -ln(1-FAR^(1/N_eff))，MC(1e5) 验证 <5%（d1_unit）。
σ̂² = 12 整窗远列 |X|² 均值（~2.3e4 独立格，相对误差<1%）。
DeRa 统计量（带噪 argmax 选列+排序候选）无此闭式 ⇒ 证书不对称性定量化。

================================================================================
【keystone】（§3.1）(i,n) 域相位 e^{j2πκi}·e^{j2πδ·i·n/NF}：双线性耦合项
δ·i·n 经剪切 i'=i·n/NF → 公共斜坡 e^{j2πδi'}（δ 与 n 解耦，一次 FFT 搜全
部 δ）；κ(CFO) 与 δ(SFO) 在 LoRa 中物理独立（物理律#13）≠ 雷达同源情形，
纯剪切使 κ-斜坡列化（速率 κNF/n）⇒ κ-bank(步1/16) 补偿后精确，
|n|≥NF/16 列残余速率≤1/16 无混叠（低列弃）。d1_unit 验证与 δ 网格一致。
================================================================================
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.chirp import build_upchirp

SF, N, OS, NF = 10, 1024, 4, 4096
NFFT = 2 * NF
N_FINE = 256
UP = build_upchirp(SF, 0, OS).astype(np.complex128)
DOWN = np.conj(UP)
NOISE_GUARD = 16
DGRID = np.round(np.arange(-0.10, 0.1001, 0.005), 4)


# ---------------------------------------------------------------- 帧结构与提取
def field_windows(hs, pre):
    """[(start, ref, length)]；j=0..pre-1 前导, pre..pre+1 sync（DOWN 参考）,
    pre+2..pre+3 SFD 整窗（UP 参考）, pre+4 SFD 1/4 窗（UP 参考前段）。"""
    base = hs - int((pre + 4.25) * NF)
    out = [(base + j * NF, DOWN if j < pre + 2 else UP, NF)
           for j in range(pre + 4)]
    out.append((base + (pre + 4) * NF, UP, NF // 4))
    return out


def row_spectrum(seg, start, ref, length):
    w = np.asarray(seg[start:start + length], dtype=np.complex128)
    X = np.fft.fft(w * ref[:length], NFFT)
    return np.concatenate((X[NFFT - N:], X[:N]))


def row_dtft(seg, start, ref, length, p_bins):
    """整窗精确 DTFT @ 连续 bin 位置（可数组）。"""
    w = np.asarray(seg[start:start + length], dtype=np.complex128)
    n = np.arange(length)
    tw = np.exp(-2j * np.pi * np.atleast_1d(p_bins)[:, None] * n[None, :] / NF)
    return tw @ (w * ref[:length])


def interp_peak(mag, idx):
    if idx <= 0 or idx >= len(mag) - 1:
        return float(idx)
    a, b, c = mag[idx - 1], mag[idx], mag[idx + 1]
    d = a - 2.0 * b + c
    return idx + 0.5 * (a - c) / d if d != 0 else float(idx)


def frame_slots(pre):
    idx = np.arange(pre + 5, dtype=float)
    idx[-1] = pre + 4.25
    b = np.ones(pre + 5)
    b[-1] = 0.25
    return idx, b


# ------------------------------------------------------------ 干净冻结模板
def _sgn_of(j, pre, walk_sign_dn=-1.0):
    return 1.0 if j < pre + 2 else walk_sign_dn


def clean_template(seg, hs, pre, walk_sign_dn=-1.0):
    """干净信号冻结：ν0、逐符号偏移 o_j、SFD 共轭约定、类型相位 θ、δ̂、κ̂。"""
    wins = field_windows(hs, pre)
    pos = []
    for s, ref, ln in wins:
        m = np.abs(row_spectrum(seg, s, ref, ln))
        pos.append((interp_peak(m, int(np.argmax(m))) - N) / 2.0)
    nu0_med = float(np.median(pos[:pre]))
    idx, b = frame_slots(pre)
    # 走动 δ̂：前导线性拟合；ν0 = idx0 外推截距
    delta = float(np.polyfit(idx[:pre], pos[:pre], 1)[0])
    nu0 = float(np.mean(pos[:pre]) - delta * (pre - 1) / 2.0)
    o = np.zeros(pre + 5)
    for j in range(pre, pre + 5):
        o[j] = pos[j] - nu0 - _sgn_of(j, pre, walk_sign_dn) * idx[j] * delta
    sgn = np.ones(pre + 5)
    sgn[pre + 2:] = walk_sign_dn

    # ---- 相位：前导斜坡稳健拟合 + 类型均值角（无跨跳 unwrap）----
    vals = []
    for j in range(pre + 5):
        pj = nu0 + o[j] + sgn[j] * idx[j] * delta
        vals.append(row_dtft(seg, wins[j][0], wins[j][1], wins[j][2],
                             pj)[0])
    vals = np.array(vals)
    ph_pre = np.unwrap(np.angle(vals[:pre]))
    sl, ic = np.polyfit(idx[:pre], ph_pre, 1)
    ramp = sl * idx + ic
    kappa_raw = sl / (2 * np.pi)
    kap = kappa_raw - round(kappa_raw)

    def type_theta(conj_dn):
        th = np.zeros(pre + 5)
        coh_pre = float(np.abs(np.sum(vals[:pre] * np.exp(-1j * ramp[:pre]))))
        tot = [coh_pre]
        for lo, hi, is_dn in ((pre, pre + 2, False), (pre + 2, pre + 5, True)):
            z = vals[lo:hi].copy()
            if conj_dn and is_dn:
                z = np.conj(z)
            z = z * np.exp(-1j * ramp[lo:hi])
            th[lo:hi] = np.angle(np.sum(z))       # 类型均值角
            tot.append(float(np.abs(np.sum(z))))
        return th, sum(tot)
    th_c, cc = type_theta(True)
    th_n, cn = type_theta(False)
    conj_dn = cc > cn
    theta = th_c if conj_dn else th_n
    return dict(nu0=nu0, o=o, delta=delta, conj_dn=conj_dn, theta=theta,
                pos=pos, sgn=sgn, kappa=kap)


# ---------------------------------------------------------------- 噪声估计
def _far_mask(cols, k_list):
    m = np.ones(cols, dtype=bool)
    for k in k_list:
        m[max(0, k - NOISE_GUARD):k + NOISE_GUARD + 1] = False
    return m


FAR_STEP = 8        # 远列子格步进（0.5-bin 列）：~2952 独立格/单元


def _far_subgrid(k_list, step=FAR_STEP):
    msk = _far_mask(2 * N, k_list)
    return np.nonzero(msk)[0][::step]


def _tw_of(p_bins, length):
    n = np.arange(length)
    return np.exp(-2j * np.pi * np.atleast_1d(p_bins)[:, None]
                  * n[None, :] / NF).T            # (length, n_p)


def eval_noise(seg, wins, k_list, n_rows=None):
    """σ̂²：整窗远列 |X|² 子格均值（每 8 列取 1，~246 列/窗）。

    固定二次型（给定 k_list）：cert 闭式的 σ̂² 口径；DeRa 臂传 n_rows=pre。
    护带 ±16 列（±8 bin）：Dirichlet 旁瓣尾（<1.3% 符号能量）对称计入两臂
    （native ~14dB 档 ~0.1dB 级，battle 档 ≤0.13% 忽略）。"""
    if n_rows is None:
        n_rows = len(wins) - 1
    subs = _far_subgrid(k_list)
    TW = _tw_of((subs - N) / 2.0, NF)
    tot, cnt = 0.0, 0
    for s, r, l in [(s, r, l) for s, r, l in wins[:-1]][:n_rows]:
        W = np.asarray(seg[s:s + l], dtype=np.complex128) * r[:NF]
        tot += float(np.sum(np.abs(W @ TW) ** 2))
        cnt += subs.size
    return tot / max(cnt, 1)


# ---------------------------------------------------------------- 主统计量
def _kap_of_q(q):
    return ((q / N_FINE + 0.5) % 1.0) - 0.5


def score_ours(seg, hs, pre, tmpl, variant="cert"):
    """OURS 两变体；variant='cert'（冻结位置/相位，固定线性变换，闭式 CFAR）
    或 'dep'（带噪 argmax 选列±1、无类型相位——可部署下界，门限需 MC）。"""
    wins = field_windows(hs, pre)
    idx, b = frame_slots(pre)
    nu0, o, delta = tmpl["nu0"], tmpl["o"], tmpl["delta"]
    sgn = tmpl["sgn"]
    conj = tmpl["conj_dn"]
    if variant == "cert":
        p = nu0 + o + sgn * idx * delta
        k_list = [int(round(p[j] * 2 + N)) % (2 * N)
                  for j in range(len(b))]
        sig2 = eval_noise(seg, wins, k_list)
        x = np.array([row_dtft(seg, wins[j][0], wins[j][1], wins[j][2],
                               p[j])[0] for j in range(len(b))])
        if conj:
            x = x.copy()
            x[pre + 2:] = np.conj(x[pre + 2:])
        x = np.sqrt(b) * x * np.exp(-1j * tmpl["theta"])
        F = np.abs(np.fft.fft(x, N_FINE))
        q = int(np.argmax(F))
        return dict(score_db=10 * np.log10(float(F[q] ** 2)
                                           / (sig2 * float(b.sum()))),
                    kappa=_kap_of_q(q), sigma2=sig2, n_cells=N_FINE)
    # dep
    Xs = np.stack([row_spectrum(seg, s, r, l) for s, r, l in wins])
    mag = np.sum(b[:, None] * np.abs(Xs) ** 2, axis=0)
    k_star = int(np.argmax(mag))
    sig2 = eval_noise(seg, wins, [k_star])
    best, kap = 0.0, 0.0
    for k in (k_star - 1, k_star, k_star + 1):
        x = np.array([Xs[j][k] for j in range(len(b))])
        if conj:
            x = x.copy()
            x[pre + 2:] = np.conj(x[pre + 2:])
        F = np.abs(np.fft.fft(np.sqrt(b) * x, N_FINE))
        q = int(np.argmax(F))
        if F[q] ** 2 > best:
            best, kap = float(F[q] ** 2), _kap_of_q(q)
    return dict(score_db=10 * np.log10(best / (sig2 * float(b.sum()))),
                kappa=kap, sigma2=sig2, n_cells=3 * N_FINE)


def score_dera(seg, hs, pre):
    """DeRa 臂（port Eq.47 原样统计量，GT 锚定窗，仅 8 前导整窗）。"""
    wins = field_windows(hs, pre)
    k_rows = np.stack([row_spectrum(seg, wins[j][0], wins[j][1], NF)
                       for j in range(pre)])
    mag = np.sqrt(np.mean(np.abs(k_rows) ** 2, axis=0))
    k_star = int(np.argmax(mag))
    sig2 = eval_noise(seg, wins[:pre], [k_star], n_rows=pre)
    best, kap = 0.0, 0.0
    for k in (k_star - 1, k_star, k_star + 1):
        x = np.array([row_dtft(seg, wins[j][0], wins[j][1], NF,
                               (k - N) / 2.0)[0] for j in range(pre)])
        F = np.abs(np.fft.fft(x, N_FINE))
        q = int(np.argmax(F))
        if F[q] ** 2 > best:
            best, kap = float(F[q] ** 2), _kap_of_q(q)
    return dict(score_db=10 * np.log10(best / (pre * sig2)), kappa=kap,
                sigma2=sig2, n_cells=3 * N_FINE)


# ------------------------------------------------------- δ 网格参考（精确）
def score_grid(seg, hs, pre, tmpl, delta_list=DGRID):
    """显式 δ 网格相干和（keystone 的精确慢速对照）。"""
    wins = field_windows(hs, pre)
    idx, b = frame_slots(pre)
    nu0, o, th, conj, sgn = (tmpl["nu0"], tmpl["o"], tmpl["theta"],
                             tmpl["conj_dn"], tmpl["sgn"])
    k0 = int(round((nu0 + o[0]) * 2 + N)) % (2 * N)
    sig2 = eval_noise(seg, wins, [k0])
    best, bd, bk = -1e18, 0.0, 0.0
    for dg in delta_list:
        p = nu0 + o + sgn * idx * dg
        x = np.array([row_dtft(seg, wins[j][0], wins[j][1], wins[j][2],
                               p[j])[0] for j in range(len(b))])
        if conj:
            x = x.copy()
            x[pre + 2:] = np.conj(x[pre + 2:])
        x = np.sqrt(b) * x * np.exp(-1j * th)
        F = np.abs(np.fft.fft(x, N_FINE))
        q = int(np.argmax(F))
        s = float(F[q] ** 2)
        if s > best:
            best, bd, bk = s, float(dg), _kap_of_q(q)
    return dict(score_db=10 * np.log10(best / (sig2 * float(b.sum()))),
                delta=bd, kappa=bk, sigma2=sig2)


# --------------------------------------------------- keystone 快速版（κ-bank）
def score_keystone(seg, hs, pre, tmpl, col_step=4, kap_bank_step=1.0 / 16.0,
                   n_fine=N_FINE):
    """κ-bank × 每列符号轴剪切重标度 + 单一第二级 FFT（§3.1 形态）。

    精确等价（全部列时）：T_ks(f_q,κ_h) = s·Σ_n e^{-2πjν0n/NF}·Σ_j √b_j
    e^{-jθ_j}e^{-2πjκ_h s_j idx_j}Y[j,n]·e^{-2πj(q/n_fine)·idx_j·n/NF}
    = T_grid(δ=f_q, κ=κ_h)（剪切定义 i'=idx·n/NF；列步 s 补偿 s²·n_cols=NF）。
    SFD 行：conj（镜像→+ν0 轨迹、走动符号翻正）+ 频移 + κ-补偿符号 s_j=-1。
    κ-bank 残差 ≤1/32 ⇒ 符号和损失 ≤0.1dB；δ 网格 = 1/n_fine ≈ 0.004。
    """
    wins = field_windows(hs, pre)
    idx, b = frame_slots(pre)
    nu0, o, th, conj = tmpl["nu0"], tmpl["o"], tmpl["theta"], tmpl["conj_dn"]
    pos, delta = tmpl["pos"], tmpl["delta"]
    n_sym = len(b)
    k0 = int(round((nu0 + o[0]) * 2 + N)) % (2 * N)
    sig2 = eval_noise(seg, wins, [k0])
    n_arr = np.arange(NF)
    # (j,n) 矩阵：去斜；upchirp 行去已知偏移 e^{-2πjo_jn/NF}；SFD 行 conj 后
    # 频移 -(ν0+pos_j)（镜像位置 pos_j → 共同轨迹 ν0+iδ）
    Y = np.zeros((n_sym, NF), dtype=np.complex128)
    for j, (s, r, l) in enumerate(wins):
        w = np.asarray(seg[s:s + l], dtype=np.complex128) * r[:l]
        if conj and j >= pre + 2:
            # conj 行 rate=-(K-ν0-idxδ)（K=镜像常数=pos_j+ν0+idx_jδ̂）：
            # 乘 e^{+j2πKn/NF} 使其落在共同轨迹 ν0+idxδ
            kk = pos[j] + nu0 + idx[j] * delta
            Y[j, :l] = np.conj(w) * np.exp(2j * np.pi * kk * n_arr[:l] / NF)
        else:
            Y[j, :l] = w * np.exp(-2j * np.pi * o[j] * n_arr[:l] / NF)
    sqb = np.sqrt(b) * np.exp(-1j * th)
    sk = np.ones(n_sym)   # conj 后 SFD 行 ramp 与 upchirp 同向（probe0/合成
                          # 均验证），κ-补偿全行同号
    cols = np.arange(int(NF * 0.25), NF, col_step)   # 低列弃：κ 残余放大 NF/n
    rot = np.exp(-2j * np.pi * nu0 * cols / NF)
    qs = np.arange(n_fine)
    best, bdk, bkh = -1e18, 0.0, 0.0

    def run_bank(kaps):
        nonlocal best, bdk, bkh
        for kh in kaps:
            w_j = sqb * np.exp(-2j * np.pi * kh * sk * idx)
            Yw = Y * w_j[:, None]
            Tq = np.zeros(n_fine, dtype=np.complex128)
            for ci, n in enumerate(cols):
                tw = np.exp((-2j * np.pi / (n_fine * NF))
                            * np.outer(qs, idx * n))
                Tq += (tw @ Yw[:, n]) * rot[ci]
            Tq *= col_step * NF / (cols[-1] - cols[0] + col_step)
            a = int(np.argmax(np.abs(Tq)))
            sv = float(np.abs(Tq[a]) ** 2)
            if sv > best:
                best = sv
                bdk = ((a / n_fine + 0.5) % 1.0) - 0.5
                bkh = float(kh)
        return
    # 两级自适应 κ-bank：粗 1/16 全程 → 细 1/128 在最优 ±1/16
    run_bank((np.arange(16) + 0.5) / 16.0 - 0.5)
    kh0 = bkh
    run_bank(kh0 + (np.arange(65) - 32) / 128.0)
    b_ks = float(b[:-1].sum())          # 1/4 窗列 ∈[0,NF/4) 已弃 ⇒ 12.0
    return dict(score_db=10 * np.log10(best / (sig2 * b_ks)),
                delta=bdk, kappa=bkh, sigma2=sig2)


# ---------------------------------------------------------------- CFAR 闭式
def rho_deltaq(idx, b, N_fine=N_FINE):
    d = np.arange(1, N_fine)
    ph = np.abs(np.exp(-2j * np.pi * np.outer(d, idx) / N_fine) @ b) \
        / float(b.sum())
    return ph


def n_eff(idx, b, N_fine=N_FINE):
    """等效独立格数（解析，周期图方差公式）。"""
    ph2 = rho_deltaq(idx, b, N_fine) ** 2
    d = np.arange(1, N_fine)
    return N_fine / (1.0 + 2.0 * float(np.sum((1 - d / N_fine) * ph2)))


def upcrossing_c(idx, b, N_fine=N_FINE):
    """Cramér–Lindgren 上穿率常数 c = 2Q·√(−r″(0))/π^{3/2}。

    r(Δ)=|ρ(Δ)|² 为 |T(q)|² 过程的归一化自相关（解析：b 加权 Dirichlet），
    r″(0) 用二阶差分 r(2)−2r(1)+1。⇒ H0 下
        P(max_q score ≤ γ) ≈ exp(−c·√γ·e^{−γ})
    （χ²₂ 平稳过程上穿渐近，Lindgren；无拟合参数）。"""
    ph = np.concatenate(([1.0], rho_deltaq(idx, b, N_fine)))
    r2 = ph ** 2
    d2 = float(r2[2] - 2 * r2[1] + 1.0)
    return 2.0 * N_fine * np.sqrt(-d2) / (np.pi ** 1.5)


def cfar_threshold(far, c_upx):
    """闭式门限：解 c·√γ·e^{−γ} = FAR（γ − ½lnγ = ln(c/FAR)，牛顿 2 步）。"""
    L = float(np.log(c_upx / far))
    g = L + 0.5 * np.log(max(L, 1.0))
    for _ in range(3):
        g = g - (g - 0.5 * np.log(max(g, 1e-9)) - L)             / (1.0 - 0.5 / max(g, 1e-9))
    return float(g)


def cell_threshold(far):
    """单格精确门限（σ² 已知）：score_cell ~ Exp(1)。"""
    return float(-np.log(far))


# ------------------------------------------------------------ 合成帧生成器
SYN = dict(nu0=-21.7, o_sync=(24.0, 32.0), k_mirror=-40.67,
           theta_sync=(2.8, 3.2), theta_sfd=(1.5, 1.7, 1.9))


def synth_frame(pre=8, delta=0.0, kappa=0.21, snr_db=-18.0, seed=0,
                nu0=None, A=1.0, noise=True):
    """SF10 12.25 场合成帧（去斜域直接生成，物理结构对齐 OTA probe0）。

    upchirp 行（前导 P + sync2）音位 ν0+o_j+idx_j·δ（o_pre=0，
    o_sync=+24/+32）；SFD 行（含 1/4 窗）镜像音位 K−ν0−idx_j·δ（走动
    符号 −1，K 冻结使 o_sfd≈+2.7）；跨符相位 φ_j=φ0+2πκ·idx_j+θ_type。
    生成段 seg（长 (P+6)·NF，field 起点 hs=(P+5.25)·NF…由 field_windows
    取窗），SNR=A²/σ_t²（整带口径，chirp |·|=1）。
    """
    if nu0 is None:
        nu0 = SYN["nu0"]
    rng = np.random.default_rng(seed)
    hs = int((pre + 5.25) * NF)
    seg = np.zeros((pre + 6) * NF, dtype=np.complex128)
    wins = field_windows(hs, pre)
    idx, b = frame_slots(pre)
    th = np.zeros(pre + 5)
    th[pre:pre + 2] = SYN["theta_sync"]
    th[pre + 2:] = SYN["theta_sfd"]
    phi0 = 0.7
    o = np.zeros(pre + 5)
    o[pre], o[pre + 1] = SYN["o_sync"]
    for j, (s, r, l) in enumerate(wins):
        n = np.arange(l)
        if j >= pre + 2:
            tone = SYN["k_mirror"] - nu0 - idx[j] * delta      # 镜像
        else:
            tone = nu0 + o[j] + idx[j] * delta
        ph = phi0 + 2 * np.pi * kappa * idx[j] + th[j]
        # 检测器参考 DOWN(=conj UP) for upchirp 行 ⇒ 段内放 UP×tone；
        # SFD 行检测器参考 UP ⇒ 段内放 DOWN×tone（物理上下行啁啾），且
        # 相位共轭（镜像 SFD——OTA probe0 实测 conj 提升相干性，同构）
        chirp = UP[:l] if j < pre + 2 else DOWN[:l]
        if j >= pre + 2:
            ph = -ph
        seg[s:s + l] = A * chirp * np.exp(1j * ph) \
            * np.exp(2j * np.pi * tone * n / NF)
    if noise:
        snr_lin = 10 ** (snr_db / 10.0)
        st2 = A * A / snr_lin
        seg += (rng.standard_normal(seg.size)
                + 1j * rng.standard_normal(seg.size)) * np.sqrt(st2 / 2)
    return seg, hs


# ------------------------------------------------- cert 统计量批量快路径
def score_cert_batch(segs, hs, pre, tmpl):
    """score_ours(variant='cert') 的批量版（同一估计器逐位等价，自校验
    5e-15，protocol §6）。"""
    wins = field_windows(hs, pre)
    idx, b = frame_slots(pre)
    nu0, o, delta, th = tmpl["nu0"], tmpl["o"], tmpl["delta"], tmpl["theta"]
    sgn, conj = tmpl["sgn"], tmpl["conj_dn"]
    B = segs.shape[0]
    p = nu0 + o + sgn * idx * delta
    k_list = [int(round(p[j] * 2 + N)) % (2 * N) for j in range(len(b))]
    subs = _far_subgrid(k_list)
    TW = _tw_of((subs - N) / 2.0, NF)
    sig2 = np.zeros(B)
    x = np.empty((B, len(b)), dtype=np.complex128)
    for j, (s, r, l) in enumerate(wins):
        W = segs[:, s:s + l] * r[None, :l]
        x[:, j] = (W @ _tw_of(p[j], l))[:, 0]
        if l == NF and j < len(wins) - 1:
            sig2 += np.sum(np.abs(W @ TW) ** 2, axis=1) / subs.size
    sig2 /= (len(wins) - 1)
    if conj:
        x[:, pre + 2:] = np.conj(x[:, pre + 2:])
    x = x * (np.sqrt(b) * np.exp(-1j * th))[None, :]
    F = np.abs(np.fft.fft(x, N_FINE, axis=1))
    q = np.argmax(F, axis=1)
    score = F[np.arange(B), q] ** 2 / (sig2 * float(b.sum()))
    kap = ((q / N_FINE + 0.5) % 1.0) - 0.5
    return score, kap, sig2


def score_dera_batch(segs, hs, pre, tmpl):
    """DeRa 臂批量（port Eq.47 统计量：P 前导整窗、带噪 argmax 列±1、
    κ-FFT、σ̂²=前导窗远子格均值[冻结列版，声明：稳定化方向，对其有利
    无害]）。"""
    wins = field_windows(hs, pre)
    B = segs.shape[0]
    k0 = int(round(tmpl["nu0"] * 2 + N)) % (2 * N)
    subs = _far_subgrid([k0])
    TW = _tw_of((subs - N) / 2.0, NF)
    sig2 = np.zeros(B)
    Xk = np.empty((B, pre, 2 * N), dtype=np.complex128)
    Ws = []
    for j in range(pre):
        s, r, l = wins[j]
        W = segs[:, s:s + l] * r[None, :l]
        Ws.append(W)
        X = np.fft.fft(W, NFFT, axis=1)
        Xk[:, j] = np.concatenate((X[:, NFFT - N:], X[:, :N]), axis=1)
        sig2 += np.sum(np.abs(W @ TW) ** 2, axis=1) / subs.size
    sig2 /= pre
    k_star = np.argmax(np.sqrt(np.mean(np.abs(Xk) ** 2, axis=1)), axis=1)
    qi = np.arange(B)
    n_lin = np.arange(NF)
    best = np.zeros(B)
    kap = np.zeros(B)
    for dOff in (-1, 0, 1):
        x = np.empty((B, pre), dtype=np.complex128)
        pcol = ((k_star + dOff) - N) / 2.0
        for j in range(pre):
            tw = np.exp(-2j * np.pi * pcol[:, None] * n_lin[None, :] / NF)
            x[:, j] = np.einsum("bl,bl->b", Ws[j], tw)
        F = np.abs(np.fft.fft(x, N_FINE, axis=1))
        q = np.argmax(F, axis=1)
        upd = F[qi, q] ** 2 > best
        best = np.where(upd, F[qi, q] ** 2, best)
        kap = np.where(upd, ((q / N_FINE + 0.5) % 1.0) - 0.5, kap)
    return best / (pre * sig2), kap, sig2

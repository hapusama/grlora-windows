# -*- coding: utf-8 -*-
"""E3（2026-10-02 第三轮）：软判决引导（turbo）相位读出 vs E2 三臂遗产。

任务：E2 证伪了全盲相位 HMM（臂 C：Φ 后验均匀，RMSE≈π/√3）。本轮把"盲
边缘化"换成"第一遍解码的符号后验当软导频"（turbo），测它能否兑现 E2 量化
的相位红利（genie B−A ≥1.5dB 窗界、公平域 PER 131:2）。

臂（数据/口径/种子与 E2 完全一致，噪声实现逐字节复现，A/B 复算做配对验证）：
  A   = E2 臂 A（非相干 κ-walk 孪生）——import e2_arm 复算；
  B   = E2 臂 B（genie）——复算，phi_hat 作 Φ̂ 健康度参照；
  D   = turbo 相位：Pass1=A 行 → q_i(v)=softmax(rows) → 软 κ-line 拟合
        （top-M 值能量格，B 同网格、v 上软加权）→ τ̂ 三支{−1.1,0,+1.05}+细化
        （平滑残差 SSE 择优）→ 软相位测量 u_k=Σ_v q̃Z e^{−jd(v)}（混合浓度
        r_k=|u|/E|Z|）→ 增量 π 外点门控 → RTS Wiener 平滑（σ_w=0.08，观测
        精度 λ_k=(2a_k r_k)²）→ 方差阻尼相干读出（臂 B 公式×e^{−var/2}）→
        软 Hamming+CRC，CRC 过即停，≤3 轮；
  DNP = D-noPhase 对照：同 turbo 循环但读出 logI₀(2a|Ys[j_k,v]|)（只用
        κ-line 不用 Φ̂）——"κ 修正收益 vs 相位收益"归因分解；
  E8/E16 = 块部分相干对照（无逐符相位跟踪）：块内公共相位边缘化
        score_k(v)=logI₀(|Σ_{j≠k}2a_jC_j(v̂_j)+2a_kC_k(v)|)，判决反馈块 MRC
        块内 2 轮；τ̂ 由块内边缘化浓度三支择优。

偏差声明：解码器不回吐符号后验，pass-1 的 q 取 softmax(A 行)（FEC 只作
停止判据与最终判决）——见 RESULTS 偏差声明。
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
import e2_arm as E2

KAP_GRID = E2.KAP_GRID
NK = len(KAP_GRID)
N = 1024
TAU_BRANCH = (-1.1, 0.0, 1.05)      # E2 probe5 capture 聚类三支
SIGMA_W = 0.08                       # E2 L3 / 任务书指定
TOPV = 16                            # 软观测 / κ 拟合 top-M 值截断
K0S = np.round(np.arange(-0.45, 0.451, 0.025), 4)
ENUS = np.round(np.arange(-0.02, 0.0201, 0.002), 4)

# probe5（干净信号 + GT 条件化）的逐帧 τ̂ 真值参照——仅作诊断，不入解码器
TAU_GT = {0: -1.132, 1: -1.100, 2: -1.064, 3: -1.115, 4: -1.128, 5: 1.197,
          6: -0.038, 7: 1.057, 8: 1.099, 9: 1.047, 10: -1.091, 11: -1.019,
          12: -0.996, 13: -1.003, 14: -1.046, 15: -0.993, 16: -1.132,
          17: 0.123, 18: 1.026, 19: 1.005, 20: -1.006, 21: 0.003, 22: -0.998,
          23: -0.996, 24: -1.091, 25: -0.019, 26: 1.135, 27: 1.068}
# probe5 条件化增量 σ（rad）——帧相位 jitter 参照（诊断用）
INC_COR = {0: 0.432, 1: 0.299, 2: 0.195, 3: 0.296, 4: 0.430, 5: 0.633,
           6: 0.152, 7: 0.189, 8: 0.264, 9: 0.165, 10: 0.235, 11: 0.135,
           12: 0.128, 13: 0.134, 14: 0.164, 15: 0.139, 16: 0.556,
           17: 0.589, 18: 0.190, 19: 0.159, 20: 0.137, 21: 0.190, 22: 0.142,
           23: 0.160, 24: 0.339, 25: 0.164, 26: 0.635, 27: 0.240}


def wrap(x):
    return np.angle(np.exp(1j * np.asarray(x, dtype=float)))


def softmax_rows(rows):
    r = rows - rows.max(axis=1, keepdims=True)
    q = np.exp(r)
    return q / q.sum(axis=1, keepdims=True)


def _top_vals(q, m=TOPV):
    top = np.argpartition(-q, m - 1, axis=1)[:, :m]
    w = np.take_along_axis(q, top, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-30)
    return top, w


def soft_kappa_line(mt, q):
    """软 κ-line 拟合：E(κ₀,εν)=Σ_k Σ_v q̃_k(v)|Z_k(κ_line_k,v)|²（top-M 截断）。
    Z 沿 κ 网格**复数线性插值**（= 臂 B zval 同式）——插 |Z|² 会把 wrap/mask
    纹波的副瓣当主峰（E2 L4 教训，e3_dbg 复盘：|Z|² 插值在 κ₀=−0.275 出假峰）。"""
    P, psym = mt["P"], mt["psym"]
    Ys = mt["Ys"][P + 2:]
    top, w = _top_vals(q)
    kk = np.arange(psym)
    sis = np.arange(P + 2, P + 2 + psym, dtype=float)
    best = None
    for enu in ENUS:
        fj = ((K0S[:, None] + enu * sis[None, :] - KAP_GRID[0]) % 1.0) * 16.0
        j0 = fj.astype(int)
        fr = (fj - j0)[:, :, None]
        g0 = Ys[kk[None, :, None], j0[:, :, None] % NK, top[None, :, :]]
        g1 = Ys[kk[None, :, None], ((j0 + 1) % NK)[:, :, None], top[None, :, :]]
        e = (np.abs(g0 * (1 - fr) + g1 * fr) ** 2 * w[None, :, :]).sum((1, 2))
        jj = int(np.argmax(e))
        if best is None or e[jj] > best[0]:
            best = (float(e[jj]), float(K0S[jj]), float(enu))
    _, kap0, enu = best
    kap_line = kap0 + enu * sis
    jstar = (np.round(((kap_line - KAP_GRID[0]) % 1.0) * 16.0).astype(int)) % NK
    return kap0, enu, kap_line, jstar


def soft_meas(mt, q, kap_line, tau, jstar):
    """τ 支路软相位测量：u_k=Σ_v q̃ Z e^{−jd(v)}，d=2πτ((v+κ)mod N)/N；
    Z 取 κ-line 最近网格切片（κ 量化对 d 的影响 ≤2e-4 rad，可忽略）。"""
    P, psym = mt["P"], mt["psym"]
    Ys = mt["Ys"][P + 2:]
    top, w = _top_vals(q)
    y = np.empty(psym)
    r = np.empty(psym)
    for k in range(psym):
        Z = Ys[k, jstar[k], top[k]]                            # (M,)
        d = 2 * np.pi * tau * ((top[k] + kap_line[k]) % N) / N
        u = np.sum(w[k] * Z * np.exp(-1j * d))
        y[k] = np.angle(u)
        r[k] = abs(u) / max(float(np.sum(w[k] * np.abs(Z))), 1e-30)
    return y, r


SIG_OM = 0.01          # 相位漂移率 ω 的游走 σ（rad/symbol²）


def _rts(y, lam, sig_w):
    """二态（相位 φ + 漂移率 ω）RTS 平滑，圆周更新。

    为什么二态：payload 相位轨迹带真实线性漂移（≈2π·εν rad/symbol，SFO/CFO
    残差；e3 调试 gid13：±1 rad/symbol 抖动段里一维 walk 滤波滞后 →
    wrap 支路翻转 → LOO 输出逐符号跳 >π）。恒速相位模型原生跟 ramp，
    后向平滑不再跨支路。返回 (φ̂, var_φ)。"""
    m = len(y)
    # 状态 [φ, ω]
    x = np.array([float(y[0]), 0.0])
    P = np.array([[2.0 / max(lam[0], 1e-6), 0.0],
                  [0.0, 0.04]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[sig_w ** 2, 0.0], [0.0, SIG_OM ** 2]])
    H = np.array([[1.0, 0.0]])
    ms = np.empty((m, 2))
    Ps = np.empty((m, 2, 2))
    pms = np.empty((m, 2))
    Pps = np.empty((m, 2, 2))
    ms[0] = x
    Ps[0] = P
    pms[0] = x
    Pps[0] = P
    for k in range(1, m):
        xp_ = F @ x
        Pp = F @ P @ F.T + Q
        r = max(1.0 / max(lam[k], 1e-9), 1e-9)
        S = Pp[0, 0] + r
        K = Pp[:, 0] / S
        innov = wrap(y[k] - xp_[0])
        x = xp_ + K * innov
        P = Pp - np.outer(K, Pp[0, :])
        ms[k] = x
        Ps[k] = P
        pms[k] = xp_
        Pps[k] = Pp
    xs_ = np.empty((m, 2))
    Pv = np.empty((m, 2, 2))
    xs_[m - 1] = ms[m - 1]
    Pv[m - 1] = Ps[m - 1]
    for k in range(m - 2, -1, -1):
        G = Ps[k] @ F.T @ np.linalg.inv(Pps[k + 1])
        xs_[k] = ms[k] + G @ (xs_[k + 1] - pms[k + 1])
        Pv[k] = Ps[k] + G @ (Pv[k + 1] - Pps[k + 1]) @ G.T
    return xs_[:, 0], np.maximum(Pv[:, 0, 0], 1e-9)


def track_phase(mt, q, kap_line, jstar, tau_branches=TAU_BRANCH, loo=True,
                adaptive=False):
    """τ 支+细化 + 门控 RTS 平滑；输出 LOO 版 φ̂/var。

    - 择支 SSE 用**无门控** RTS + Huber 封顶（门控会偏袒多门控的支）。
    - adaptive=True：σ_w 从增量 MAD 自估（probe5：帧间条件化增量 σ 0.13-0.63，
      固定 0.08 对 jitter 帧欠拟合）；否则用预注册 σ_w=0.08。
    - 门控两遍：增量 MAD 阈 + 平滑残差 >2.2 兜底（均匀错测）。
    - LOO：符号 k 的 φ̂_k 由其余符号给出（turbo 外信息，防自我实现）。"""
    P, psym = mt["P"], mt["psym"]
    amp = mt["amp"][P + 2:]

    def meas_lam(tau):
        y, r = soft_meas(mt, q, kap_line, tau, jstar)
        lam = np.minimum(2.0 * (amp * np.maximum(r, 1e-3)) ** 2, 2500.0)
        return y, lam, r

    def sel_sse(y, lam, sw):
        xs, _ = _rts(y, lam, sw)
        res = wrap(y - xs)
        return float(np.sum(np.minimum(lam * res ** 2, lam * 2.25)))

    cands = {}
    for t in tau_branches:
        y, lam, r = meas_lam(t)
        cands[t] = (y, lam, r, sel_sse(y, lam, SIGMA_W))
    best_t = min(cands, key=lambda t: cands[t][3])
    for t in np.round(np.arange(best_t - 0.15, best_t + 0.151, 0.03), 3):
        if abs(t - best_t) < 1e-9 or t in cands:
            continue
        y, lam, r = meas_lam(float(t))
        cands[float(t)] = (y, lam, r, sel_sse(y, lam, SIGMA_W))
        if cands[float(t)][3] < cands[best_t][3]:
            best_t = float(t)
    y, lam, r, _ = cands[best_t]
    # σ_w（自适应可选）
    inc = wrap(np.diff(y))
    mad = float(np.median(np.abs(wrap(inc - np.median(inc)))))
    var_inc = (1.4826 * mad) ** 2
    lam_med = float(np.median(lam))
    sig_w = SIGMA_W
    if adaptive:
        sig_w = float(np.sqrt(np.clip(var_inc / 2.0 - 1.0 / max(lam_med, 1e-9),
                                      0.0064, 0.36)))
    # 门控两遍
    med = float(np.median(inc))
    thr = float(np.clip(2.5 * mad, 1.0, 2.0))
    gate = np.zeros(psym, bool)
    for i in np.where(np.abs(wrap(inc - med)) > thr)[0]:
        gate[i if lam[i] <= lam[i + 1] else i + 1] = True
    lam_g = lam * np.where(gate, 0.01, 1.0)
    xs, xv = _rts(y, lam_g, sig_w)
    res = wrap(y - xs)
    gate2 = np.abs(res) > 2.2
    if gate2.any():
        gate = gate | gate2
        lam_g = lam * np.where(gate, 0.01, 1.0)
        xs, xv = _rts(y, lam_g, sig_w)
        res = wrap(y - xs)
    out = dict(tau=float(best_t), y=y, lam=lam_g, phi=wrap(xs), var=xv,
               sse=float(np.sum(lam_g * res ** 2)),
               gate_rate=float(np.mean(gate)), r=r,
               sig_w=sig_w,
               inc_sigma=float(np.std(inc[np.abs(wrap(inc - med)) < 1.0]))
               if len(inc) else None)
    if loo:
        loo_phi = np.empty(psym)
        loo_var = np.empty(psym)
        for k in range(psym):
            lam2 = lam_g.copy()
            lam2[k] = min(lam2[k], 1e-4)
            xs2, xv2 = _rts(y, lam2, sig_w)
            loo_phi[k] = xs2[k]
            loo_var[k] = xv2[k]
        out.update(phi=wrap(loo_phi), var=np.maximum(loo_var, 1e-9),
                   loo_res=float(np.median(np.abs(wrap(y - wrap(loo_phi))))))
    return out


def coherent_rows(mt, phi, var, tau, kap_line, jstar):
    """VM 卷积边缘化读出（精确式）：

        rows[k][v] = log I₀( | 2a·Z_k(v)·e^{−jd(v)} + κ_p·e^{jφ̂_k} | )

    von Mises 卷积恒等式：∫exp(s·cos(α−φ))·exp(κ_p cos(φ−μ))dφ = 2π·I₀(判据
    如上)。κ_p=1/var_k（RTS LOO 方差）：κ_p→∞ 退化为臂 B 相干式；κ_p→0 退化
    为 logI₀(2a|Z|)（=DNP 非相干）——相位跟踪失效时**优雅回退**而非翻转。
    （旧版 e^{−var/2} 阻尼只整行缩放、不改 argmax，坏 φ̂ 无法回退——pilot 教训。）"""
    P, psym = mt["P"], mt["psym"]
    amp = mt["amp"][P + 2:]
    Ys = mt["Ys"][P + 2:]
    v = np.arange(N)
    rows = np.empty((psym, N))
    for k in range(psym):
        d = 2 * np.pi * tau * ((v + kap_line[k]) % N) / N
        kap_p = 1.0 / max(var[k], 1e-3)
        s = (2.0 * amp[k]) * Ys[k, jstar[k], :] * np.exp(-1j * d)
        rows[k] = E2.log_i0(np.abs(s + kap_p * np.exp(1j * phi[k])))
        rows[k] -= rows[k].max()
    return rows


def noncoh_rows(mt, jstar):
    """D-noPhase 读出：logI₀(2a|Ys[j_k,v]|)（κ-line 修正切片，无相位）。"""
    P, psym = mt["P"], mt["psym"]
    amp = mt["amp"][P + 2:]
    Ys = mt["Ys"][P + 2:]
    rows = np.empty((psym, N))
    for k in range(psym):
        rows[k] = E2.log_i0(2.0 * amp[k] * np.abs(Ys[k, jstar[k], :]))
        rows[k] -= rows[k].max()
    return rows


def block_rows(mt, q, kap_line, jstar, tau, blk):
    """臂 E：块内公共相位边缘化（I₀(|Σ|) 精确式）+ **软反馈**块 MRC。

    反馈项 = pass-1 后验软混合 Σ_v q̃_j(v)C_j(v)（硬判决反馈在 −24 会发散，
    pilot 教训：E8 SER 0.44 > A 0.24）。"""
    P, psym = mt["P"], mt["psym"]
    amp = mt["amp"][P + 2:]
    Ys = mt["Ys"][P + 2:]
    C = np.empty((psym, N), np.complex64)
    v = np.arange(N)
    for k in range(psym):
        d = 2 * np.pi * tau * ((v + kap_line[k]) % N) / N
        C[k] = Ys[k, jstar[k], :] * np.exp(-1j * d)
    top, w = _top_vals(q)
    Cf = np.empty(psym, np.complex64)          # 软反馈复均值
    for j in range(psym):
        z = w[j] @ C[j][top[j]]
        Cf[j] = z / max(float(np.sum(w[j] * np.abs(C[j][top[j]]))), 1e-30)
    rows = np.empty((psym, N))
    for k in range(psym):
        lo = (k // blk) * blk
        hi = min(lo + blk, psym)
        acc = np.zeros(N, np.complex64)
        for j in range(lo, hi):
            if j != k:
                acc += (2.0 * amp[j]) * Cf[j]
        S = np.abs(acc + (2.0 * amp[k]) * C[k])
        rows[k] = E2.log_i0(S)
        rows[k] -= rows[k].max()
    return rows


def pick_tau_block(mt, q, kap_line, jstar, blk):
    """臂 E 的 τ 选择：块内边缘化浓度 Σ_blk|Σu|/Σ|u| 三支择优（无相位跟踪）。"""
    psym = mt["psym"]
    best, bv = None, -1.0
    for t in TAU_BRANCH:
        y, r = soft_meas(mt, q, kap_line, t, jstar)
        u = r * np.exp(1j * y)
        conc = 0.0
        for lo in range(0, psym, blk):
            seg = u[lo:lo + blk]
            conc += abs(seg.sum()) / max(float(np.abs(seg).sum()), 1e-9)
        if conc > bv:
            best, bv = float(t), conc
    return best


def arm_D_turbo(mt, fr, genie, rowsA):
    """臂 D。genie = 臂 B 复算 dict（phi_hat / tau_hat 作参照）。

    τ 支选择：SSE 排序后依次过 CRC——首个 CRC 通过者胜（协议 §3.5 的
    CRC 仲裁原语，同 DeRa；2⁻¹⁶/支 撞检风险可忽略）；全不过则用 SSE 最优支。
    φ̂ 用 LOO 版（外信息）。返回 (rows, diag)。"""
    gt = np.asarray(fr["gt"])
    q = softmax_rows(rowsA)
    rounds = []
    last_tr = None
    rows = rowsA
    for it in range(3):
        if it == 0:
            extra = dict(src="A")
        else:
            kap0, enu, kap_line, jstar = soft_kappa_line(mt, q)
            # τ 三支全跑（LOO φ̂）；排序 = CRC 通过优先，否则 loo_res（LOO 拟合
            # 残差——实测右支 0.12-0.33 vs 错支 1.05-1.22，比 SSE 判别力强）
            cands = [track_phase_at(mt, q, kap_line, jstar, t)
                     for t in TAU_BRANCH]
            cands.sort(key=lambda c: c["loo_res"])
            pick, crc_pick = None, False
            for c in cands:
                rws = coherent_rows(mt, c["phi"], c["var"], c["tau"],
                                    kap_line, jstar)
                if E2.judge(rws, fr["gt_hdr"], fr["plen"], fr["cr"]):
                    pick, rows, crc_pick = c, rws, True
                    break
            if pick is None:
                pick = cands[0]
                rows = coherent_rows(mt, pick["phi"], pick["var"], pick["tau"],
                                     kap_line, jstar)
            last_tr = pick
            gphi = np.asarray(genie["phi_hat"])
            rmse_g = (float(np.sqrt(np.mean(wrap(pick["phi"] - gphi) ** 2)))
                      if abs(pick["tau"] - genie["tau_hat"]) <= 0.3 else None)
            extra = dict(src="coh%d" % it, kap0=float(kap0), enu=float(enu),
                         tau=pick["tau"], tau_crc=crc_pick,
                         gate_rate=pick["gate_rate"],
                         inc_sigma=pick["inc_sigma"], loo_res=pick["loo_res"],
                         r_med=float(np.median(pick["r"])),
                         phi_rmse_g=rmse_g)
        hard = np.argmax(rows, 1)
        crc_ok = bool(E2.judge(rows, fr["gt_hdr"], fr["plen"], fr["cr"]))
        rounds.append(dict(it=it, SER=float(np.mean(hard != gt)),
                           crc_ok=crc_ok, **extra))
        if crc_ok or it == 2:
            break
        q = softmax_rows(rows)
    diag = dict(rounds=rounds, final_it=len(rounds) - 1)
    if last_tr is not None:
        diag.update(gate_rate=last_tr["gate_rate"], tau=float(last_tr["tau"]),
                    r_med=float(np.median(last_tr["r"])),
                    loo_res=float(last_tr["loo_res"]))
    return rows, diag


def track_phase_at(mt, q, kap_line, jstar, tau):
    """指定 τ（含细化）跑 track_phase：细化在支内做，返回最优细化点。"""
    c = track_phase(mt, q, kap_line, jstar, tau_branches=(tau,))
    return c


def arm_D_nophase(mt, fr, rowsA):
    """D-noPhase 对照：同 turbo 循环但读出只用 κ-line（非相干 logI₀）。"""
    gt = np.asarray(fr["gt"])
    q = softmax_rows(rowsA)
    rounds = []
    rows = rowsA
    for it in range(3):
        extra = {}
        if it > 0:
            k0, enu, _kl, jstar = soft_kappa_line(mt, q)
            rows = noncoh_rows(mt, jstar)
            extra = dict(kap0=float(k0), enu=float(enu))
        hard = np.argmax(rows, 1)
        crc_ok = bool(E2.judge(rows, fr["gt_hdr"], fr["plen"], fr["cr"]))
        rounds.append(dict(it=it, SER=float(np.mean(hard != gt)),
                           crc_ok=crc_ok, **extra))
        if crc_ok or it == 2:
            break
        q = softmax_rows(rows)
    return rows, rounds


def arm_E(mt, fr, rowsA, blk):
    """臂 E：块部分相干（B∈{8,16}），τ̂ 块浓度三支择优。"""
    gt = np.asarray(fr["gt"])
    q = softmax_rows(rowsA)
    kap0, enu, kap_line, jstar = soft_kappa_line(mt, q)
    tau = pick_tau_block(mt, q, kap_line, jstar, blk)
    rows = block_rows(mt, q, kap_line, jstar, tau, blk)
    hard = np.argmax(rows, 1)
    crc_ok = bool(E2.judge(rows, fr["gt_hdr"], fr["plen"], fr["cr"]))
    return rows, dict(SER=float(np.mean(hard != gt)), crc_ok=crc_ok,
                      tau=float(tau), kap0=float(kap0), enu=float(enu))

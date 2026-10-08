# -*- coding: utf-8 -*-
"""D2 Battle C：端到端解码战（漂移域）——OURS 全链 vs DeRa 全链。

链 ① OURS-dep（系统臂，论文数字）：盲门扫（单窗峰/中位 >4dB，hop 512）
    → 产线 locate（1-sample 栅格，与 DeRa 前端同一原语——对称公平）
    → dep 统计量（锚带噪 + δ 网格 + κ-FFT）过 MC 门限 → (start, κ̂, δ̂)
    → γ-链（前导导频）精化 (κ₀,δ) → 统一度量行 + TREL-5 行 → 逐符整数
      走动反映射（round(m̂_i)+Δ0，Δ0∈{0,±1,±2} CRC 仲裁）→ CRC。
链 ①' OURS-cert（机制参考）：GT 锚定 cert 检出（闭式门限）+ 同一解码。
链 ② DeRa：paper_dera_detector（自带盲扫+coherent+括号+locate）top-5
    → paper_dera_demod + δ∈{0,±1,±2} CRC fallback（front_runner 原样）。
场景：δ ∈ {0(报告), 0.02, 0.082(判定)}；SNR −18..−28；5 种子；28 帧。
公平红线：同一注入+加噪 IQ 喂两链；κ̂/δ̂ 各用各的；GT 只作计分。
预注册判定：≥1 漂移场景 OURS-dep×γ 链 PER 优于 DeRa 全链 ≥1.5dB
（10% PER 门限）= 端到端战胜；δ=0 场景报告不计胜负。
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\dera_front_battle_20260930")
import d1_core as C
import d1_battle as A
import d2_core as D
import front_runner as FR
from weak_decoder.decoding.gamma_link import GammaLinkDemodulator
from weak_decoder.decoding.kappa_trellis import KappaTrellisDemodulator

SF, N, OS, NF = 10, 1024, 4, 4096
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "d2_battleC_checkpoint.jsonl")
SEED_CONST = 20261003
DELTAS_C = (0.0, 0.02, 0.082)
LEVELS_C = [-20, -22, -24, -26, -28]
N_SEEDS_C = 3
SCAN_HOP = NF // 8
SCAN_THRESH_DB = 4.0
GATE_EVENTS = 3

G = {}
KT = None
GL = None


def gate_scan(seg):
    """单窗门扫（DeRa scan 同款原语）：峰/中位 >4dB 的 hop → 周期事件分组。"""
    n = len(seg)
    peaks = []
    for s in range(0, n - NF + 1, SCAN_HOP):
        X = np.fft.fft(np.asarray(seg[s:s + NF]) * C.DOWN)
        row = np.abs(np.concatenate((X[NF - N // 2:], X[:N // 2]))) ** 2
        med = float(np.median(row))
        if med <= 0:
            continue
        k = int(np.argmax(row))
        r = 10 * np.log10(row[k] / med)
        if r > SCAN_THRESH_DB:
            peaks.append((s, k, r))
    events, used = [], [False] * len(peaks)
    for i, p in enumerate(peaks):
        if used[i]:
            continue
        grp, used[i] = [p], True
        for j in range(i + 1, len(peaks)):
            if used[j]:
                continue
            q = peaks[j]
            db = min(abs(q[1] - grp[-1][1]), N - abs(q[1] - grp[-1][1]))
            if db <= 4 and 0 < q[0] - grp[-1][0] <= 2 * NF:
                grp.append(q)
                used[j] = True
        if len(grp) >= 3:
            events.append(dict(start=min(g[0] for g in grp),
                               power=sum(10 ** (g[2] / 10) for g in grp)))
    events.sort(key=lambda e: -e["power"])
    return events[:GATE_EVENTS]


def ours_dep_detect(seg, pre, thr):
    """盲检出：门扫事件 → locate 精化 → dep 分数择优 → ±64/16 细对齐。

    locate 可能锁 +192 样本的伪对齐（结构判据宽容）；dep 分数在真对齐处
    显著占优（native 152k vs ≤107k），且对 ±8 样本残余敏感（1/4 窗）——
    故对所有 (事件, s_try) 的 locate 成功逐个打分取最大，再做细对齐。"""
    def _sc_of(hs_t):
        sc, dhat, khat, _, anch = D.score_dep_batch(
            seg[None, :], hs_t, pre, G["conj"][pre], ret_anchor=True)
        # hs_t = header 起点（front_runner pay0 = hs//NF+8 同构）
        return dict(score=float(sc[0]), pay0=int(hs_t) + 8 * NF,
                    dhat=float(dhat[0]),
                    khat=float(khat[0]), nu0h=float(anch["nu0h"][0]))

    def _ok(hs_t):
        return int((pre + 4.25) * NF) <= hs_t             and hs_t + 10 * NF <= len(seg)

    best = None
    for ev in gate_scan(seg)[:2]:
        # 粗对齐（±512 步 128）→ 细对齐（±32 步 8；±6-8 样本残余落入
        # sync o_known 的 Dirichlet 零点，必须细扫）。对齐原语 = dep 分数
        # 自身（对称：DeRa 用其 locate，我们用我们的分数扫描）。
        c_best = None
        hs0 = ev["start"] + int((pre + 4.25) * NF)   # 门事件 → header 起点
        for off in range(-512, 513, 128):
            hs_t = hs0 + off
            if not _ok(hs_t):
                continue
            r = _sc_of(hs_t)
            if c_best is None or r["score"] > c_best["score"]:
                c_best = r
        if c_best is None:
            continue
        base_best = c_best["pay0"] - 8 * NF
        for off in range(-32, 33, 8):
            hs_t = base_best + off
            if not _ok(hs_t):
                continue
            r = _sc_of(hs_t)
            if r["score"] > c_best["score"]:
                c_best = r
                base_best = hs_t
        if best is None or c_best["score"] > best["score"]:
            best = c_best
    if best is None or best["score"] < thr:
        return None
    return best


def demap_judge(rows, ints, f):
    """逐符整数走动反映射 + Δ0 CRC 仲裁。

    payload 音位 p_i = ν̂0_abs + walk_i（绝对 bin 域，走动从场首累积）
    ⇒ 行 argmax = (v_i + p_i) mod N；反映射后 argmax 应落在 v+1
    （fast_evidence 值映射 (bin−1)%N）。ints_i = round(p_i)（检测锚
    绝对 bin + γ-链漂移精化），残差常数 Δ0∈{0,±1,±2} CRC 仲裁。
    返回 (ok, ser, delta0)。"""
    # 轨迹选择（无 GT）：正确整数轨迹下 (argmax − ints) 应逐符近常数；
    # 错误轨迹在走动过半处出 ±1 跳。取绕众数的残差能量最小者。
    am = np.argmax(rows, axis=1)
    best_ints, best_score = None, None
    for cand in (ints, ints + 1):
        c = (am - cand) % N
        mode = np.bincount(c, minlength=N).argmax()
        dev = np.minimum((c - mode) % N, (mode - c) % N)
        score = float(np.sum(dev ** 2))
        if best_score is None or score < best_score:
            best_ints, best_score = cand, score
    ints = best_ints
    for d0 in (0, 1, -1, 2, -2, 3, -3):
        rows_dm = np.stack([np.roll(rows[k], -(ints[k] + d0))
                            for k in range(rows.shape[0])])
        if FR.judge_crc(rows_dm, 1, f["gt_hdr"], f["plen"], f["cr"]):
            hard = [(int(np.argmax(r)) - 1) % N for r in rows_dm]
            ser = int(sum(int(h != g) for h, g in zip(hard, f["gt"])))
            return True, ser, d0
    return False, -1, 0


class DriftInitGammaLink(GammaLinkDemodulator):
    """γ-链 + 检测器 δ̂ 初始化（链规范：检测器出 δ̂ → γ-链 κ̂ 精化）。

    原 γ-链 WLS 的合角初始化在走动跨 ≥1 圈（δ·psym>1）时落到平坦分支
    （native δ=0.082 实测 drift̂=−0.0035 vs 真 +0.082）。本变体把 y 拟合
    域平移到 wrap(y − δ̂·x)（残差走动 ≤0.1 圈，安全），拟合 (κ₀,δ_res)
    后恢复 δ̂_tot = δ̂_det + δ_res。度量/权重/导频/统一行逐字复用
    gamma_link（front/tail 投影、|z| 加权、mod-1 残差卷绕精化）。"""

    drift_init = 0.0
    prior_slope = None        # y 域斜率先验 = −δ̂_bin（检测器/模板）

    def demod_payload(self, samples, start_symbol, psym, pilots=None):
        d0 = float(self.drift_init)
        self.n = self.n
        idx = np.arange(int(psym))
        f_mat = np.empty((psym, self.n), dtype=np.complex128)
        t_mat = np.empty((psym, self.n), dtype=np.complex128)
        for k in idx:
            st = (int(start_symbol) + k) * self.nf
            w = np.asarray(samples[st:st + self.nf], dtype=np.complex64)
            if w.size != self.nf:
                raise ValueError(f"symbol {k} window exceeds input")
            f_mat[k] = self._front @ w
            t_mat[k] = self._tail @ w
        noncoh = np.abs(f_mat) ** 2 + np.abs(t_mat) ** 2
        chat = np.argmax(noncoh, axis=1)
        z = t_mat[idx, chat] * np.conj(f_mat[idx, chat])
        gamma = np.angle(z)
        y = gamma / np.pi
        w_i = np.abs(z).astype(np.float64)

        y_all = list(y)
        w_all = list(w_i)
        x_all = [float(k) for k in idx]
        for p_start, p_x in (pilots or []):
            st = int(p_start)
            w_p = np.asarray(samples[st:st + self.nf], dtype=np.complex64)
            if w_p.size != self.nf:
                continue
            f_p = self._front @ w_p
            t_p = self._tail @ w_p
            c_p = int(np.argmax(np.abs(f_p) ** 2 + np.abs(t_p) ** 2))
            z_p = t_p[c_p] * np.conj(f_p[c_p])
            y_all.append(float(np.angle(z_p)) / np.pi)
            w_all.append(float(np.abs(z_p)))
            x_all.append(float(p_x))
        n_all = len(y_all)
        w_arr = np.asarray(w_all, dtype=np.float64)
        w_arr /= max(float(w_arr.sum()), 1e-30)
        x_arr = np.asarray(x_all, dtype=np.float64)
        # 初值阶梯拟合（σ² 择优）：合角初始化只在 |δ|·span < ~1 圈安全，
        # 检测 δ̂ 又可能偏 ±0.08 —— 对 {0, ±0.05, ±0.1, δ̂_det} 各拟合一
        # 次，取加权残差 σ² 最小者（投影只算一次，拟合是 O(n) 的）。
        H = np.stack([np.ones(n_all), x_arr], axis=1)
        Wd = np.diag(w_arr)
        HW = H.T @ Wd
        y_raw = np.asarray(y_all)

        def _wls(yv):
            try:
                return np.linalg.solve(HW @ H, HW @ yv)
            except np.linalg.LinAlgError:
                return np.array([float(np.average(yv, weights=w_arr)), 0.0])

        def _fit(d_init):
            yr = ((y_raw - d_init * x_arr + 0.5) % 1.0) - 0.5
            beta = np.array([float(np.angle(np.sum(
                w_arr * np.exp(1j * np.pi * yr))) / np.pi), 0.0])
            for _ in range(3):
                r = yr - H @ beta
                r = ((r + 0.5) % 1.0) - 0.5
                beta = _wls(H @ beta + r)
            res = yr - H @ beta
            res = ((res + 0.5) % 1.0) - 0.5
            sig2 = float(np.sum(w_arr * res ** 2) / max(n_all - 2.0, 1.0))
            return beta, d_init, sig2

        # 先验（y 域斜率 = −δ_bin；检测器/模板给出）+ 阶梯；选择规则：
        # σ² < 3×min 者中取 |drift − prior| 最小（纯 σ² 会锁错分支，
        # δ=0.02 实测 σ² 近平时漂到 −0.0097 vs 真 −0.017）。
        prior = self.prior_slope
        inits = [0.0, 0.05, -0.05, 0.1, -0.1]
        for extra in ((-d0,) if abs(d0) > 1e-6 else ()):
            if all(abs(extra - v) > 0.01 for v in inits):
                inits.append(extra)
        fits = [_fit(di) for di in inits]
        sig_min = min(f[2] for f in fits)
        elig = [f for f in fits if f[2] < 3.0 * sig_min + 1e-9]
        if prior is not None:
            beta, d_used, sigma2 = min(
                elig, key=lambda f: abs((f[0][1] + f[1]) - prior))
        else:
            beta, d_used, sigma2 = min(elig, key=lambda f: f[2])
        beta = np.array([beta[0], beta[1]])
        drift = float(beta[1]) + d_used
        # κ0 在【未平移域】按已定 drift 重估（合角保 π 周期奇偶性）：
        # 平移域合角会破坏 e^{jπκ} 的分支（κ0 落 κ+1 ⇒ 相干项反号，
        # native 实测行镜像化）。y − drift·x ≈ κ0 常数 ⇒ 合角给出 κ0 mod 2。
        y_res = y_raw - drift * x_arr
        order = np.argsort(x_arr)
        phi = np.unwrap(np.pi * y_res[order])
        ph_full = np.empty_like(phi)
        ph_full[order] = phi
        kappa0 = float(np.angle(np.sum(w_arr * np.exp(1j * ph_full)))
                       / np.pi)
        m_hat = kappa0 + drift * idx.astype(np.float64)
        try:
            cov = np.linalg.inv(HW @ H)
            lev_all = np.einsum("ij,jk,ik->i", H, cov, H)
        except np.linalg.LinAlgError:
            lev_all = np.full(n_all, 1.0 / max(n_all, 1))
        lev = lev_all[:psym]
        s2 = sigma2 * (lev + 1.0 / max(n_all, 1))
        rho = np.exp(-np.pi ** 2 * s2 / 2.0)
        # 相干参考精化：S = Σ w·z·e^{−jπm̂} 的角给出 m̂ 的整体相位差
        #（含 π 奇偶错位——mod-2 拟合盲区，漂移帧实测错侧时 cross 项反噬）
        z_all = t_mat[idx, chat] * np.conj(f_mat[idx, chat])
        S = np.sum(w_i * z_all * np.exp(-1j * np.pi * m_hat))
        ph_fix = float(np.angle(S))
        rows = noncoh.copy()
        for k in idx:
            theta = np.pi * m_hat[k] + ph_fix
            cross = 2.0 * rho[k] * np.real(
                np.conj(f_mat[k]) * t_mat[k] * np.exp(-1j * theta))
            rows[k] = rows[k] + cross
        self.last_fit = dict(kappa0=kappa0, drift=drift, sigma2=sigma2,
                             rho_min=float(rho.min()),
                             rho_mean=float(rho.mean()), n_meas=int(n_all))
        return rows


def payload_ints(kappa0, drift, psym, pre):
    """payload 逐符整数走动反映射：**零走动**（实测裁决）。

    合成（naive 频率走动注入）am−v 每 1/δ 符步进；而【重采样注入的 OTA】
    am−v 在 δ=0.082 全 payload 只走 ~0.3 bin（~0.009/符）——理想栅格重启
   窗使整数 bin 走动被窗时漂移近似抵消（γ 序列斜率 −δ 是 V1/V2 分裂效应，
    非整数 bin 位移）。⇒ 整数反映射只需常数 Δ0（δ=0 已验证路径）；
    γ 拟合的价值在统一度量的相位补偿项（e^{−jπm̂}，已生效）。
    残余 ~0.3 bin 走动在 0.5 边界处翻 ±1 → demap_judge 的 ±1 轨迹候选
    与 CRC Δ0 吸收。本函数保留签名，返回 0。"""
    return np.zeros(psym, dtype=int)


def run_unit(u):
    di, level, seed, fi = u
    f = G["frames"][fi]
    pre = f["pre"]
    psym = f["psym"]
    lead = pre + 6
    tail = (8 + psym + 2) * NF + 64
    key = (fi, di)
    cch = G.setdefault("inj", {})
    if key not in cch:
        seg0 = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + tail],
                          dtype=np.complex128)
        origin = (lead - (pre + 4.25)) * NF
        inj = D.resample_sfo(seg0, origin, D.eps_of_delta(DELTAS_C[di]))
        tm_nat = D.clean_template_q(seg0, lead * NF, pre)
        tm_d = D.clean_template_q(inj, lead * NF, pre)
        snr = A.snr_parts(inj[: lead * NF + 8 * NF])
        cch[key] = (inj, tm_nat, tm_d, snr)
        if len(cch) > 24:
            for k in list(cch)[:len(cch) - 24]:
                del cch[k]
    inj, tm_nat, tm_d, (S, N0) = cch[key]

    if level is None:
        seg = inj
    else:
        rng = np.random.default_rng((SEED_CONST * 7919
                                     + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919
                                     + di * 104729) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = inj + ((rng.standard_normal(len(inj))
                      + 1j * rng.standard_normal(len(inj)))
                     * np.sqrt(p_add / 2.0))

    out = dict(kind="c", di=di, delta=DELTAS_C[di], level=level, seed=seed,
               frame=fi, pre=pre)

    # ---- 链① OURS-dep（系统臂）----
    try:
        det = ours_dep_detect(seg, pre, G["dep_thr"])
        out["dep_det"] = det is not None
        if det is not None:
            pay0 = det["pay0"]                    # locate 的 payload 起点
            pad = int((pre + 5) * NF)
            seg_p = np.concatenate((np.zeros(pad, dtype=np.complex128),
                                    seg[pay0:]))
            n_rot = np.arange(len(seg_p))
            seg_p = seg_p * np.exp(-2j * np.pi * det["nu0h"] * n_rot / NF)
            pilots = [(pad - (pre + 12.25 - j) * NF,
                       j - (pre + 12.25)) for j in range(pre)]
            GL.drift_init = det["dhat"] if abs(det["dhat"]) < 0.2 else 0.0
            GL.prior_slope = -det["dhat"] if abs(det["dhat"]) < 0.2 else None
            rows = GL.demod_payload(seg_p, pre + 5, psym, pilots=pilots)
            lf = GL.last_fit
            rows_t = KT.demod_payload(seg_p, pre + 5, psym, readout="viterbi")
            ints = payload_ints(lf["kappa0"], lf["drift"], psym, pre)
            ok, ser, _d0 = demap_judge(rows_t, ints, f)
            out["dep_per"] = int(not ok)
            out["dep_ser"] = ser
            out["dep_dhat"] = det["dhat"]
            out["gfit_d"] = GL.last_fit["drift"]
            out["gfit_k0"] = GL.last_fit["kappa0"]

    except Exception as ex:
        out["dep_err"] = repr(ex)[:120]

    # ---- 链①' OURS-cert（机制参考：GT 锚定检出 + 同一解码）----
    try:
        sc, _, _ = C.score_cert_batch(seg[None, :], lead * NF, pre, tm_d)
        out["cert_det"] = bool(sc[0] > G["cert_thr"])
        if out["cert_det"]:
            pay0 = (lead + 8) * NF                # GT 锚定（段内 hs + 8 头符）
            pad = int((pre + 5) * NF)
            seg_p = np.concatenate((np.zeros(pad, dtype=np.complex128),
                                    seg[pay0:]))
            c_pre0 = (pre - 1) / 2.0
            nu_rot = tm_d["nu0"] + c_pre0 * tm_d["delta"]
            n_rot = np.arange(len(seg_p))
            seg_p = seg_p * np.exp(-2j * np.pi * nu_rot * n_rot / NF)
            pilots = [(pad + lead * NF - (pre + 4.25 - j) * NF,
                       j - (pre + 12.25)) for j in range(pre)]
            GL.drift_init = tm_d["delta"] if abs(tm_d["delta"]) < 0.2 else 0.0
            GL.prior_slope = -tm_d["delta"] if abs(tm_d["delta"]) < 0.2                 else None
            rows = GL.demod_payload(seg_p, pre + 5, psym, pilots=pilots)
            lf = GL.last_fit
            rows_t = KT.demod_payload(seg_p, pre + 5, psym, readout="viterbi")
            ints = payload_ints(lf["kappa0"], lf["drift"], psym, pre)
            ok, ser, _ = demap_judge(rows_t, ints, f)
            out["cert_per"] = int(not ok)
            out["cert_ser"] = ser
    except Exception as ex:
        out["cert_err"] = repr(ex)[:120]

    # ---- 链② DeRa 全链（front_runner 原样）----
    try:
        cands, seg_as, pay0s = FR.dera_sync(seg, pre)
        out["dera_det"] = bool(cands)
        if cands:
            top5_rows = []
            for seg_a, pay0 in zip(seg_as, pay0s):
                if pay0 + psym + 1 > len(seg_a) // NF:
                    continue
                _s1, rows = FR.DERA_DEC.demod_payload(seg_a, pay0, psym)
                top5_rows.append(rows)
            if top5_rows:
                _d, ok, rows = FR.decode_chain(None, f, top5=top5_rows)
                hard = [(int(np.argmax(rows[k])) - _d) % N
                        for k in range(psym)]
                out["dera_per"] = int(not ok)
                out["dera_ser"] = int(sum(int(h != g)
                                          for h, g in zip(hard, f["gt"])))
    except Exception as ex:
        out["dera_err"] = str(ex)[:80]
    return out


def init_worker():
    global KT, GL
    G["frames"] = FR.build_frames()
    G["conj"] = {}
    for pre in (8, 16, 32):
        f = [fr for fr in G["frames"] if fr["pre"] == pre][0]
        seg0 = np.asarray(f["iq"][f["hs"] - (pre + 6) * NF:
                                  f["hs"] + 8 * NF], dtype=np.complex128)
        G["conj"][pre] = D.clean_template_q(seg0, (pre + 6) * NF, pre)
    KT = KappaTrellisDemodulator(SF, OS)
    GL = DriftInitGammaLink(SF, OS)


def calibrate_thresholds():
    """dep 盲扫系统门限（合成噪声 MC）+ cert 闭式门限（含 2D 膨胀）。"""
    thr_file = os.path.join(EXP_DIR, "d2_battleC_thr.json")
    if os.path.exists(thr_file):
        return json.load(open(thr_file))
    init_worker()
    f = G["frames"][0]
    pre = f["pre"]
    lead = pre + 6
    L = (lead + 8) * NF
    scores = []
    rng = np.random.default_rng(999)
    for t in range(60):
        seg = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) \
            * np.sqrt(0.5)
        det = ours_dep_detect(seg, pre, -1.0)
        if det is not None:
            scores.append(det["score"])
        if (t + 1) % 20 == 0:
            print("  thr MC %d/60 n_det=%d" % (t + 1, len(scores)),
                  flush=True)
    idx, bb = C.frame_slots(8)
    cup = C.upcrossing_c(idx, bb)
    clo = C.cfar_threshold(1e-2, cup)
    try:
        c2d = json.load(open(os.path.join(EXP_DIR, "d2_cfar2d.json")))
        infl = c2d.get("FAR=0.01", {}).get("inflation_db", 0.0)
    except Exception:
        infl = 0.0
    dep_thr = float(np.max(scores)) if scores else 16.0
    res = dict(dep_thr=dep_thr, n_mc=60,
               cert_thr=float(clo * 10 ** (infl / 10.0)),
               cert_thr_1d=float(clo), infl_db=infl,
               dep_scores_max=[float(s) for s in sorted(scores)[-5:]])
    json.dump(res, open(thr_file, "w"), indent=1)
    print("门限：", res, flush=True)
    return res


def _pool_init(dep_thr, cert_thr):
    init_worker()
    G["dep_thr"] = dep_thr
    G["cert_thr"] = cert_thr


def main():
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        thr = dict(dep_thr=0.0, cert_thr=0.0, infl_db=0.0)
        G.update(dep_thr=0.0, cert_thr=0.0)
        init_worker()
        for di in (0,):
            for u in [(di, None, 1, fi) for fi in range(4)]:
                r = run_unit(u)
                print(json.dumps(r)[:400], flush=True)
        return
    thr = calibrate_thresholds()
    print("Battle C：δ %s × 档 %s × 种子 %d，门限 %s"
          % (list(DELTAS_C), LEVELS_C, N_SEEDS_C,
             {k: thr[k] for k in ("dep_thr", "cert_thr")}), flush=True)

    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["di"], r["level"], r["seed"], r["frame"]))
            except Exception:
                pass
        print("断点恢复：%d 单元" % len(done), flush=True)
    units = ([(di, None, 0, fi) for di in range(len(DELTAS_C))
              for fi in range(28)]
             + [(di, lv, sd, fi) for di in range(len(DELTAS_C))
                for lv in LEVELS_C for sd in range(N_SEEDS_C)
                for fi in range(28)])
    units = [u for u in units if u not in done]
    print("待跑 %d 单元" % len(units), flush=True)

    ctx = mp.get_context("spawn")

    n = 0
    with open(CKPT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=6, initializer=_pool_init,
                      initargs=(thr["dep_thr"], thr["cert_thr"])) as pool:
            for r in pool.imap_unordered(run_unit, units, chunksize=1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n += 1
                if n % 28 == 0:
                    print("  %d/%d (%.0fs)" % (n, len(units),
                                               time.time() - t0), flush=True)
    print("完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

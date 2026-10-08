# -*- coding: utf-8 -*-
"""E2 三臂实验（2026-10-02 rev3）：数据条件化相位轨迹 HMM 读出 vs 非相干 vs genie。

协议：EXPERIMENT_PROTOCOL.md 实验B（先验给定型）：同步先验冻结（e1_native，
干净信号），整包 AWGN（§2 口径，S/N0 冻结值），种子派生与 E1 相同
→ 噪声实现与 E1_snr_units 逐字节一致。GT 只来自干净原生解。判据 =
decode_symfec_payload_from_evidences（软 Hamming+CRC16，GT header 同权）；
启动负控：8/16 符号损坏 + 全随机行必须被拒（单符号损坏可被 Hamming(8,4)+
对角交织纠正，属 FEC 正常工作，不属旁路）。

OTA 实证相位/音位律（e2_probe1-5 + 本轮补充探针，SF10 28 帧）：
  L1 payload 音位 ν_i = v_i + κ₀ + εν·i；κ₀ ≈ −0.05（帧间稳定，同步链以
     payload 为对齐锚），εν ∈ [−0.017, 0]（SFO 漂移，17/28 帧 |>0.003|）；
     preamble κ ≈ κ_pay + 0.24..0.61（帧间变）——payload 切片格自含。
  L2 峰 bin 相位 θ_i(v_i) = Φ_i + 2π·τ·ν_i/N mod 2π；τ = 每帧残余 STO 滑动
     系数，按 capture 聚类 {−1.1, 0, +1.05}（探针5：条件化增量 1.75→0.19 rad，
     浓度 0.982；τ̂∈[−1.13,1.20]）。奇偶律/二次/φ_H(c) 均被排除（探针2/3/4）。
  L3 Φ_i：Wiener 游走（前导 σ 0.083，payload 条件化隐含 ~0.13）。
  L4 |Z| 在真音 ±0.1 bin 内平顶（Dirichlet 核平坦顶），且 wrap/mask 分裂
     造成纹波——逐符号 |Z| argmax 不可作 κ̂（探针教训）。
域 = 混叠 N 域（e1_common 抽取）；det 精确性由 e2_unit 单元测试背书。

臂：A0 纯能量 |X₀[v]|²；A 相位盲孪生（I₀ 发射 + κ-walk 前后向，同机器）；
   B genie（GT 条件化 (κ₀,εν) 池化格拟合 + τ̂ 扫描 + Φ̂_i）相干读出；
   C 机制 HMM：状态 τ(33)×κ(24,warp±0.75)×Φ(24)，转移 = Φ Wiener(σ_w=0.10)
   + κ 慢走(σ_κ=0.015) + τ 恒定；SFD+header 桥 = Φ 方差 11.25σ_w²+0.6²、
   κ 放宽（σ 0.4，吸收 pre→pay κ 跳变）；发射对 v 全边缘化（top-K）。
指标：SER/PER（CRC 仲裁）/C 健康度（相位后验 RMSE vs genie Φ̂）。
输出 e2_units.jsonl（断点续跑）。
"""
import sys
import os
import json
import time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C

ROOT = C.ROOT
IN = os.path.join(ROOT, "e1_native.jsonl")
OUTU = os.path.join(ROOT, "e2_units.jsonl")

from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, SymFECSymbolEvidence, decode_symfec_payload_from_evidences)

FEC_CFG = SymFECConfig()
SIGMA_W = 0.10
SIGMA_K = 0.015
SIGMA_KGAP = 0.4
GAP_VAR = 11.25 * SIGMA_W ** 2 + 0.6 ** 2
TAU_GRID = np.round(np.arange(-1.6, 1.61, 0.1), 2)
KAP_GRID = np.round(np.arange(-0.75, 0.75, 1.0 / 16), 4)
PHI_NB = 24
TOPK = 8
NOISE_MAX = np.log(len(KAP_GRID) * 1024) + 0.5772
SEEDS = list(range(C.N_SEEDS))

_W = {}
ST = {}


def _walk_matrix(sigma, step, n):
    p = 0.5 * (sigma / step) ** 2
    if p > 0.25:
        p = 0.25
    W = np.zeros((n, n))
    for a in range(n):
        if a > 0:
            W[a, a - 1] = p
        if a < n - 1:
            W[a, a + 1] = p
        W[a, a] = 1.0 - W[a].sum()
    return W


def build_static():
    n = 1024
    v = np.arange(n)
    phi_c = (np.arange(PHI_NB) + 0.5) * 2 * np.pi / PHI_NB
    EDT = np.angle(np.exp(-2j * np.pi * TAU_GRID[:, None]
                          * ((v[None, :] + 0.0) % n) / n))        # (T,N) = −det_v
    ROT = np.exp(-2j * np.pi * KAP_GRID[:, None] * v[None, :] / n).astype(np.complex64)
    d = np.angle(np.exp(1j * (phi_c[:, None] - phi_c[None, :])))
    W = np.exp(-0.5 * (d / SIGMA_W) ** 2)
    W /= W.sum(1, keepdims=True)
    Wg = np.exp(-0.5 * (d / np.sqrt(GAP_VAR)) ** 2)
    Wg /= Wg.sum(1, keepdims=True)
    WK = _walk_matrix(SIGMA_K, 1.0 / 16, len(KAP_GRID))
    WKG = _walk_matrix(SIGMA_KGAP, 1.0 / 16, len(KAP_GRID))
    ST.update(n=n, EDT=EDT, ROT=ROT, phi_c=phi_c,
              logW=np.log(W), logWgap=np.log(Wg),
              logWK=np.log(WK), logWKG=np.log(WKG),
              K=len(KAP_GRID), T=len(TAU_GRID), F=PHI_NB,
              rk=np.arange(len(KAP_GRID)))


def lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def log_i0(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    sm = x < 8.0
    out[sm] = np.log(np.i0(x[sm]))
    xl = x[~sm]
    out[~sm] = xl - 0.5 * np.log(2 * np.pi * np.maximum(xl, 1e-30))
    return out


def unit_measure(seg, fr, ds):
    n = ds.n
    K = ST["K"]
    P, psym = fr["P"], fr["psym"]
    ntot = P + 2 + psym
    offs = ([1.0 + j - 0.25 for j in range(P)] + [P + 0.75, P + 1.75]
            + [P + 13.0 + k for k in range(psym)])
    z0 = np.empty((ntot, n), np.complex64)
    Ys = np.empty((ntot, K, n), np.complex64)
    for si, off in enumerate(offs):
        m = C.extract_raw(seg, off, ds, ds.ref_dn, c_nom=0)
        if m is None:
            return None
        dr = m["dr"].astype(np.complex64)
        X = np.fft.fft(dr)
        p2 = np.abs(X) ** 2
        keep = np.ones(n, bool)
        top = np.argsort(p2)[::-1][:6]
        for t in top:
            keep[t] = keep[(t + 1) % n] = keep[(t - 1) % n] = False
        s = float(np.sqrt(max(np.mean(p2[keep]), 1e-30)))
        z0[si] = (X / s).astype(np.complex64)
        Ys[si] = (np.fft.fft(dr[None, :] * ST["ROT"], axis=1) / s).astype(np.complex64)
    peaks = (np.abs(Ys) ** 2).max(axis=(1, 2))
    amp = np.sqrt(np.maximum(peaks - NOISE_MAX, 0.3))
    amp = np.clip(amp, 0.3, 30.0)
    return dict(z0=z0, Ys=Ys, amp=amp, P=P, psym=psym, ntot=ntot)


def _emission_payload(Yrow, a):
    mag = np.abs(Yrow)
    top = np.argpartition(-mag, TOPK, axis=1)[:, :TOPK]
    m = np.take_along_axis(mag, top, 1)
    ang = np.take_along_axis(np.angle(Yrow), top, 1)
    det_v = ST["EDT"][:, top]                               # (T,K,TK) = −2πτv/N
    r = ang[None] + det_v                                   # ang − det
    cosr = np.cos(ST["phi_c"][None, None, None, :] - r[:, :, :, None])
    return top, m, ang, 2.0 * a * m[None, :, :, None] * cosr


def _emission_known(Yrow, a, v):
    zv = Yrow[:, v]
    det_v = ST["EDT"][:, v]                                 # (T,)
    r = np.angle(zv)[None, :] + det_v[:, None]              # (T,K)
    cosr = np.cos(ST["phi_c"][None, None, :] - r[:, :, None])
    return 2.0 * a * np.abs(zv)[None, :, None] * cosr       # (T,K,F)


def kappa_step(a, logWK_):
    """a: (T,K,F) → 对 κ 源做 LSE 转移：out[t,kd,f] = LSE_ks(logWK[kd,ks]+a[t,ks,f])。"""
    tmp = a[:, None, :, :] + logWK_.T[None, :, :, None]   # (T,Kd,Ks,F)
    return lse(tmp, 2)


def hmm_run(mt):
    Ys, amp, P, psym, ntot = mt["Ys"], mt["amp"], mt["P"], mt["psym"], mt["ntot"]
    K, T, F, n = ST["K"], ST["T"], ST["F"], 1024
    known = [n - 1] * P + [23, 31]
    emis = np.empty((ntot, T, K, F))
    tops = [None] * ntot
    for si in range(ntot):
        if si < P + 2:
            emis[si] = _emission_known(Ys[si], amp[si], known[si] % n)
        else:
            top, m, ang, terms = _emission_payload(Ys[si], amp[si])
            tops[si] = (top, m, ang)
            emis[si] = lse(terms, 2)
    prior = np.zeros((T, K, F)) - float(np.log(T * K * F))
    alpha = prior + emis[0]
    alphas = np.empty((ntot, T, K, F))
    alphas[0] = alpha

    def step(a, W, WK_):
        a = kappa_step(a, WK_)
        return lse(a[:, :, :, None] + W[None, None, :, :], 2)

    for si in range(1, ntot):
        gap = (si == P + 2)
        alpha = step(alpha, ST["logWgap"] if gap else ST["logW"],
                     ST["logWKG"] if gap else ST["logWK"]) + emis[si]
        alphas[si] = alpha
    betas = np.empty((ntot, T, K, F))
    betas[ntot - 1] = 0.0
    for si in range(ntot - 2, -1, -1):
        gap = (si + 1 == P + 2)
        m2 = emis[si + 1] + betas[si + 1]
        b = lse(m2[:, :, :, None]
                + (ST["logWgap"] if gap else ST["logW"])[None, None, :, :], 2)
        betas[si] = kappa_step(b, ST["logWKG"] if gap else ST["logWK"])
    rows = np.zeros((psym, n))
    ph_post = np.zeros((psym, 2))
    for k in range(psym):
        si = P + 2 + k
        g = alphas[si] + betas[si]
        g = np.exp(g - g.max())
        g /= g.sum()
        lg = np.log(np.maximum(g, 1e-300)) - (emis[si] + np.log(n))
        top, m, ang = tops[si]
        a2 = amp[si]
        det_v = ST["EDT"][:, top]
        r = ang[None] + det_v
        terms = 2.0 * a2 * m[None, :, :, None] * np.cos(
            ST["phi_c"][None, None, None, :] - r[:, :, :, None])
        lt = lg[:, :, None, :] + terms
        m2 = lt.max(axis=(0, 3))
        lc2 = m2 + np.log(np.sum(np.exp(lt - m2[None, :, :, None]), axis=(0, 3)))
        rowslog = np.full(n, np.log(1.0 / n))
        for ki in range(K):
            for j, vv in enumerate(top[ki]):
                rowslog[vv] = np.logaddexp(rowslog[vv], lc2[ki, j])
        rows[k] = rowslog - rowslog.max()
        pc = (g.sum((0, 1)) * np.exp(1j * ST["phi_c"])).sum()
        ph_post[k] = [float(np.angle(pc)), float(abs(pc))]
    g_last = alphas[-1] + betas[-1]
    g_last = np.exp(g_last - g_last.max())
    g_last /= g_last.sum()
    diag = dict(tau_hat=float(TAU_GRID[int(g_last.sum((1, 2)).argmax())]),
                kap_hat=float(KAP_GRID[int(g_last.sum((0, 2)).argmax())]),
                phi_post=ph_post.tolist())
    return rows, diag


def arm_A_twin(mt):
    """相位盲孪生：κ-walk 前后向（能量 I₀ 发射），逐符号 p(v) 行。"""
    Ys, amp, P, psym, ntot, n = mt["Ys"], mt["amp"], mt["P"], mt["psym"], \
        mt["ntot"], 1024
    K = ST["K"]
    known = [n - 1] * P + [23, 31]
    emis = np.empty((ntot, K))
    for si in range(ntot):
        li = log_i0(2.0 * amp[si] * np.abs(Ys[si]))
        if si < P + 2:
            emis[si] = li[:, known[si] % n]
        else:
            emis[si] = lse(li, 1) - np.log(n)
    alpha = np.zeros((ntot, K))
    alpha[0] = emis[0]
    for si in range(1, ntot):
        WK_ = ST["logWKG"] if si == P + 2 else ST["logWK"]
        prev = lse(WK_.T + alpha[si - 1][:, None], 0)     # (K_dst,)
        alpha[si] = prev + emis[si]
    beta = np.zeros((ntot, K))
    for si in range(ntot - 2, -1, -1):
        WK_ = ST["logWKG"] if si + 1 == P + 2 else ST["logWK"]
        beta[si] = lse(WK_ + (emis[si + 1] + beta[si + 1])[:, None], 1)
    rows = np.zeros((psym, n))
    for k in range(psym):
        si = P + 2 + k
        g = alpha[si] + beta[si]
        g = np.exp(g - g.max())
        g /= g.sum()
        li = log_i0(2.0 * amp[si] * np.abs(Ys[si]))
        rows[k] = lse(np.log(g)[:, None] + li, 0)
        rows[k] -= rows[k].max()
    return rows, float(KAP_GRID[int(np.argmax(alpha[-1] + beta[-1]))])


def arm_B_genie(mt, gt):
    """genie：GT 条件化 (κ₀,εν) 池化能量格拟合（纹波免疫）→ τ̂ 扫描 → Φ̂。"""
    Ys, amp, P, psym, n = mt["Ys"], mt["amp"], mt["P"], mt["psym"], 1024
    K = ST["K"]
    gt = np.asarray(gt)
    sis = np.arange(P + 2, P + 2 + psym)
    k0s = np.arange(-0.45, 0.451, 0.025)
    enus = np.arange(-0.02, 0.0201, 0.001)
    # Z(v+κ) 从切片网格线性插值（|D| 平顶 + 相位 π/16 内，误差可忽略）
    def zval(si, kappa, v):
        rho = (kappa - KAP_GRID[0]) % 1.0
        fj = rho * 16.0
        j0 = int(np.floor(fj)) % K
        frac = fj - np.floor(fj)
        return Ys[si, j0, v] * (1 - frac) + Ys[si, (j0 + 1) % K, v] * frac
    best = None
    for enu in enus:
        kap_line = k0s[:, None] + enu * sis[None, :]
        e = np.zeros(len(k0s))
        for k in range(psym):
            for j in range(len(k0s)):
                e[j] += abs(zval(sis[k], kap_line[j, k], gt[k])) ** 2
        j = int(np.argmax(e))
        if best is None or e[j] > best[0]:
            best = (e[j], float(k0s[j]), float(enu))
    _, kap0, enu = best
    kap_line = kap0 + enu * sis
    nu = (gt + kap_line) % n
    Zg = np.array([zval(sis[k], kap_line[k], gt[k]) for k in range(psym)])
    av = np.angle(Zg)
    taus = np.arange(-1.8, 1.81, 0.05)
    conc = np.array([abs(np.mean(np.exp(1j * (av - 2 * np.pi * t * nu / n))))
                     for t in taus])
    tau = float(taus[int(np.argmax(conc))])
    phi_hat = av - 2 * np.pi * tau * nu / n
    rows = np.zeros((psym, n))
    for k, si in enumerate(sis):
        rho = (kap_line[k] - KAP_GRID[0]) % 1.0
        jstar = int(np.round(rho * 16.0)) % K
        d = 2 * np.pi * tau * ((np.arange(n) + kap_line[k]) % n) / n
        rows[k] = 2.0 * amp[si] * np.real(
            np.exp(-1j * (phi_hat[k] + d)) * Ys[si, jstar, :])
        rows[k] -= rows[k].max()
    return rows, dict(kap_hat=kap0, enu=enu, tau_hat=tau,
                      conc=float(conc.max()), phi_hat=phi_hat.tolist())


def judge(rows, gt_hdr, plen, cr):
    n = rows.shape[1]
    evs = []
    for k in range(rows.shape[0]):
        r = np.maximum(rows[k] - rows[k].max(), -30.0)
        order = np.argsort(r)[::-1][:TOPK]
        evs.append(SymFECSymbolEvidence(
            symbol_index=k, raw_scores=r, score_by_value=r,
            best_raw_bin_by_value=np.arange(n),
            argmax_raw_bin=int(order[0]), argmax_symbol_value=int(order[0]),
            peak_margin_db=float(r[order[0]] - r[order[1]]),
            top_raw_bins=tuple(int(b) for b in order),
            top_symbol_values=tuple(int(b) for b in order)))
    try:
        res = decode_symfec_payload_from_evidences(
            evidences=evs, sf=10, cr=int(cr), ldro=False, config=FEC_CFG,
            header_symbol_values=list(gt_hdr), payload_len=int(plen),
            has_crc=True, crc_mode="grlora")
        return bool(res.payload_decode is not None
                    and res.payload_decode.crc_valid)
    except Exception:
        return False


def run_unit(u):
    gid, snr, seed = u["gid"], u["snr"], u["seed"]
    fr, seg0, ds = _W["frames"][gid], _W["segs"][gid], _W["ds"][gid]
    seg = C.inject_noise(seg0, snr, seed, gid, fr["S"], fr["N0"]) \
        if snr is not None else seg0
    mt = unit_measure(seg, fr, ds)
    if mt is None:
        return dict(u, skip="extract_fail")
    gt = np.array(fr["gt"])
    rowsA, kA = arm_A_twin(mt)
    rowsB, dB = arm_B_genie(mt, gt)
    rowsC, dC = hmm_run(mt)
    ph = np.array(dC["phi_post"])
    e = C.wrap(ph[:, 0] - np.array(dB["phi_hat"]))
    out = dict(gid=gid, snr=snr, seed=seed, sym_tot=mt["psym"],
               phase_rmse=float(np.sqrt(np.mean(e ** 2))),
               phase_post_conc=float(np.median(ph[:, 1])),
               tau_hat=dC["tau_hat"], tau_hat_B=dB["tau_hat"],
               kap_hat_A=kA, kap_hat_B=dB["kap_hat"], enu=dB["enu"],
               genie_conc=dB["conc"])
    rowsA0 = 20.0 * np.log10(np.maximum(
        np.abs(mt["z0"][mt["P"] + 2:, :]) ** 2, 1e-12))
    for name, rows in (("A0", rowsA0), ("A", rowsA), ("B", rowsB), ("C", rowsC)):
        hard = np.argmax(rows, 1)
        out[name] = dict(sym_err=int((hard != gt).sum()),
                         crc_fail=int(not judge(rows, fr["gt_hdr"],
                                                fr["plen"], fr["cr"])))
    return out


def init_worker(gids):
    build_static()
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    keep = [fr for fr in frames if fr["gid"] in gids]
    caps = sorted(set(fr["cap"] for fr in keep))
    _W["frames"] = {fr["gid"]: fr for fr in keep}
    _W["segs"] = {}
    _W["ds"] = {}
    for cap in caps:
        src = next(s for s in C.SF10_SOURCES + C.SF11_SOURCES if s[0] == cap)
        ds = C.DS(src[3])
        iq = np.memmap(src[1], dtype=np.complex64, mode="r")
        for fr in (f for f in keep if f["cap"] == cap):
            r = dict(header_start_sample=fr["hs"],
                     source_grlora_cfo_int=fr["cfo_int"],
                     source_grlora_cfo_frac=str(fr["cfo_frac"]),
                     source_grlora_payload_sto_frac=str(fr["sto_frac"]))
            seg, _i0, _bo = C.align_seg(iq, r, ds, fr["P"], fr["psym"])
            _W["segs"][fr["gid"]] = seg
            _W["ds"][fr["gid"]] = ds
        del iq


def negative_control():
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    fr = frames[0]
    n, psym = 1024, fr["psym"]
    rows = np.full((psym, n), -30.0)
    for k, v in enumerate(fr["gt"]):
        rows[k, v] = 0.0
    assert judge(rows, fr["gt_hdr"], fr["plen"], fr["cr"]), "GT 行必须过 CRC"
    rng = np.random.default_rng(0)
    for nbad in (8, 16):
        bad = 0
        for trial in range(6):
            rows_bad = rows.copy()
            idx = rng.choice(psym, nbad, replace=False)
            for k in idx:
                rows_bad[k, fr["gt"][k]] = -30.0
                rows_bad[k, int(rng.integers(0, n))] = 0.0
            bad += int(judge(rows_bad, fr["gt_hdr"], fr["plen"], fr["cr"]))
        assert bad == 0, "%d 符号损坏存在未被拒绝（%d/6）" % (nbad, bad)
    rows_rand = np.full((psym, n), -30.0)
    for k in range(psym):
        rows_rand[k, int(rng.integers(0, n))] = 0.0
    assert not judge(rows_rand, fr["gt_hdr"], fr["plen"], fr["cr"]), \
        "全随机行不得过 CRC"
    print("判据负控通过：GT 全对过 CRC；8/16 符号损坏与全随机行全部被拒")


def main():
    t0 = time.time()
    build_static()
    negative_control()
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    frames = [f for f in frames if f["sf"] == 10]
    by_cap = {}
    for fr in frames:
        by_cap.setdefault(fr["cap"], []).append(fr["gid"])
    units = []
    for fr in frames:
        units.append(dict(gid=fr["gid"], snr=None, seed=None))
        for snr in C.SNR_LEVELS:
            for sd in SEEDS:
                units.append(dict(gid=fr["gid"], snr=snr, seed=sd))
    done = set()
    if os.path.exists(OUTU):
        for l in open(OUTU, encoding="utf-8"):
            rec = json.loads(l)
            done.add((rec["gid"], rec["snr"], rec["seed"]))
        print("断点续跑: 已完成 %d 单元" % len(done), flush=True)
    todo = [u for u in units if (u["gid"], u["snr"], u["seed"]) not in done]
    print("总单元 %d, 待跑 %d" % (len(units), len(todo)), flush=True)
    n = 0
    fout = open(OUTU, "a", encoding="utf-8")
    for cap, gids in sorted(by_cap.items()):
        cu = [u for u in todo if u["gid"] in set(gids)]
        if not cu:
            continue
        with Pool(processes=6, initializer=init_worker, initargs=(set(gids),)) as pool:
            for res in pool.imap_unordered(run_unit, cu, chunksize=4):
                fout.write(json.dumps(res) + "\n")
                n += 1
                if n % 200 == 0:
                    fout.flush()
                    print("  %d/%d (%.0fs)" % (n, len(todo), time.time() - t0),
                          flush=True)
    fout.close()
    print("完成 %d 单元 (%.0fs)" % (n, time.time() - t0))


if __name__ == "__main__":
    main()

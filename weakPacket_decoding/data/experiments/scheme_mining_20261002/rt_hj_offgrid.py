# -*- coding: utf-8 -*-
"""红队攻击 (h)(j)：margF ML-tightness 的网格敏感性 + A4 增益的 κ 剖面。

(h) t1_ml_gap 的 κ* 全在 0.05 网格上。真 κ* 取 0.225/0.375（格点中间，
0.025 偏置）重跑三臂（marg/margF/jml）：tightness ≤0.01dB 是否保持？
off-grid 量化惩罚多大？
(j) A4 "①+⑤ 合计 2.2dB" 取在 κ=0.4。补 κ=0.2/0.35，报 plain→margF/jml
的增益剖面，检验"软接口为主炮、重 bin 坍缩 ~0.1dB"是否 κ=0.4 特供。

模型与 t1_ml_gap.py 相同（SF7 N=128, M=8 未编码符号, 网格 ±0.5 步 0.05）。
运行：py -3.12 rt_hj_offgrid.py
"""
import numpy as np
import time

rng = np.random.default_rng(20261003)
N = 128
M = 8
V0, V1 = 16, N - 16
W0, W1 = 8, N - 8
T = np.arange(N)
GRID = np.round(np.arange(-0.5, 0.5001, 0.05), 2)
G = GRID.size


def log_i0(x):
    x = np.abs(x)
    return np.where(x < 300.0, np.log(np.i0(np.minimum(x, 300.0)) + 1e-300),
                    x - 0.5 * np.log(2 * np.pi * np.maximum(x, 300.0)))


def lse(a, axis):
    am = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(am + np.log(np.sum(np.exp(a - am), axis=axis, keepdims=True)), axis=axis)


def gen(kappa, snr_db, n):
    v = rng.integers(V0, V1, size=(n, M))
    ph = rng.uniform(0, 2 * np.pi, size=(n, M))
    y = np.exp(2j * np.pi * (v[..., None] + kappa) * T / N + ph[..., None])
    sg = 10 ** (-snr_db / 10.0)
    y += np.sqrt(sg / 2) * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
    return y, v


def receivers(y, snr_db):
    n = y.shape[0]
    Fs = np.empty((G,) + y.shape, dtype=np.complex128)
    for g in range(G):
        tw = np.exp(-2j * np.pi * GRID[g] * T / N)
        Fs[g] = np.fft.fft(y * tw[None, None, :], axis=-1)
    g0 = int(np.argmin(np.abs(GRID)))
    rho = 10 ** (snr_db / 10.0)
    lp = log_i0(2.0 * rho * np.abs(Fs))
    err = {}
    vh = np.argmax(np.abs(Fs[g0][..., W0:W1]), axis=-1) + W0
    err['plain'] = (vh != _v).sum()
    Lm = lse(lp, axis=0)
    vh = np.argmax(Lm[..., W0:W1], axis=-1) + W0
    err['marg'] = (vh != _v).sum()
    lzv = lse(lp, axis=3)
    sc = lzv.sum(axis=2)
    logw = sc - lse(sc, axis=0)[None, :]
    Lf = lse(lp + logw[:, :, None, None], axis=0)
    vh = np.argmax(Lf[..., W0:W1], axis=-1) + W0
    err['margF'] = (vh != _v).sum()
    gML = np.argmax(np.max(lp[..., W0:W1], axis=3).sum(axis=2), axis=0)
    sel = lp[gML, np.arange(n)]
    vh = np.argmax(sel[..., W0:W1], axis=-1) + W0
    err['jml'] = (vh != _v).sum()
    return err


def ser_all(kappa, snr_db, n):
    global _v
    tot = M * n
    acc = {}
    for s in range(0, n, 50):
        m = min(50, n - s)
        y, _v = gen(kappa, snr_db, m)
        e = receivers(y, snr_db)
        for k in e:
            acc[k] = acc.get(k, 0) + e[k]
    return {k: acc[k] / tot for k in acc}


def crossing(snrs, sers, target=0.10):
    sers = np.asarray(sers)
    ok = np.isfinite(sers) & (sers > 0)
    idx = np.where(ok)[0]
    below = idx[sers[idx] <= target]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return snrs[0]
    j = i - 1
    l0, l1 = np.log10(sers[j]), np.log10(sers[i])
    if l0 == l1:
        return snrs[i]
    return snrs[j] + (np.log10(target) - l0) * (snrs[i] - snrs[j]) / (l1 - l0)


def threshold(kappa, n_coarse=120, n_fine=260):
    coarse = np.arange(-24.0, -14.0, 1.0)
    cs = {r: [] for r in ('plain', 'marg', 'margF', 'jml')}
    for s in coarse:
        e = ser_all(kappa, s, n_coarse)
        for r in cs:
            cs[r].append(e[r])
    lo, hi = np.inf, -np.inf
    for r in cs:
        c = crossing(coarse, cs[r])
        if np.isfinite(c):
            lo = min(lo, c - 1.25)
            hi = max(hi, c + 1.25)
    fine = np.arange(np.floor(lo * 4) / 4, hi, 0.25)
    allsnr = np.concatenate([coarse, fine])
    allser = {r: list(cs[r]) for r in cs}
    for s in fine:
        e = ser_all(kappa, s, n_fine)
        for r in cs:
            allser[r].append(e[r])
    order = np.argsort(allsnr)
    return {r: crossing(allsnr[order], np.asarray(allser[r])[order]) for r in cs}


t0 = time.time()
print("SF7 N=%d M=%d, grid ±0.5 step 0.05 (21 pts)" % (N, M))
print("%8s | %7s %7s %7s %7s | %9s %9s | %10s %10s"
      % ("kappa*", "plain", "marg", "margF", "jml", "margF-jml", "marg-jml",
         "margF-plain", "jml-plain"))
res = {}
for kappa in (0.20, 0.35, 0.225, 0.375):
    th = threshold(kappa)
    res[kappa] = th
    dj = th['margF'] - th['jml']
    dm = th['marg'] - th['jml']
    print("%+8.3f | %+7.2f %+7.2f %+7.2f %+7.2f | %+9.2f %+9.2f | %+10.2f %+10.2f"
          % (kappa, th['plain'], th['marg'], th['margF'], th['jml'], dj, dm,
             th['plain'] - th['margF'], th['plain'] - th['jml']), flush=True)
print("\n(h) off-grid 偏置惩罚估计: margF@0.225 vs margF@0.20 = %+.2f dB ; jml 同差 = %+.2f dB"
      % (res[0.225]['margF'] - res[0.20]['margF'], res[0.225]['jml'] - res[0.20]['jml']))
print("(j) 增益剖面 (plain->margF): κ=0.2: %+.2f | 0.35: %+.2f | Round1: 0.3: %+.2f, 0.4: %+.2f dB"
      % (res[0.20]['plain'] - res[0.20]['margF'],
         res[0.35]['plain'] - res[0.35]['margF'], -19.06 - (-20.59), -17.00 - (-20.29)))
print("elapsed %.1f s" % (time.time() - t0))

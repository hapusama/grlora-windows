# -*- coding: utf-8 -*-
"""任务一：TREL 距联合 ML 天花板多远 —— 无码小规模可穷举代理。

设定：SF7 (N=128)、M=10 个未编码符号、共享分数偏移 κ（真值在已知网格上）、
每符号独立随机相位、纯 AWGN（逐样本 CN(0, sigma^2)，信号幅度 1）。
非相干精确似然：log p(y|v,κ) = log I0( 2|F(v+κ)|/σ² )，F = Σ y e^{-2πiνn/N}。

接收机（同一帧数据、共随机数）：
 (i)   plain   : κ=0 假设，argmax_v |F(v)|            （默认链硬读出）
 (ii)  plugin  : 帧池化比值 κ̂（中值）插件补偿 argmax   （est-compensate 族代理）
 (iii) marg    : 逐符号对 κ 网格求和的软判决          （TREL 无码简化）
 (iii-b) margF : 帧池化 κ 后验加权的边缘化            （和积精确符号 MAP）
 (iv)  jml     : max_κ Σ_syms max_v log p（无码可分解，网格上精确）

输出：各接收机 SER=10% 的逐样本 SNR 门限(dB) 与 (iii)-(iv) 门限差。
门限 = 粗扫 1dB + 联合细化 0.25dB + log10(SER) 线性插值。
"""
import numpy as np
import time

rng = np.random.default_rng(20261002)
N = 128                      # SF7
M = 8                        # 帧内符号数
V0, V1 = 16, N - 16          # 真值符号窗
W0, W1 = 8, N - 8            # 候选判决窗（防卷绕伪影，所有接收机一致）
T = np.arange(N)
GRID = None
G = 0


def log_i0(x):
    x = np.abs(x)
    return np.where(x < 300.0,
                    np.log(np.i0(np.minimum(x, 300.0)) + 1e-300),
                    x - 0.5 * np.log(2 * np.pi * np.maximum(x, 300.0)))


def lse(a, axis):
    am = np.max(a, axis=axis, keepdims=True)
    out = am + np.log(np.sum(np.exp(a - am), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def gen(kappa, snr_db, n):
    v = rng.integers(V0, V1, size=(n, M))
    ph = rng.uniform(0, 2 * np.pi, size=(n, M))
    y = np.exp(2j * np.pi * (v[..., None] + kappa) * T / N + ph[..., None])
    sg = 10 ** (-snr_db / 10.0)
    y += np.sqrt(sg / 2) * (rng.standard_normal(y.shape)
                            + 1j * rng.standard_normal(y.shape))
    return y, v


def one_batch(y, v, snr_db, kappa):
    """返回 dict: 各接收机错符号计数。"""
    n = y.shape[0]
    # --- κ 网格上的 F: [G, n, M, N] ---
    Fs = np.empty((G,) + y.shape, dtype=np.complex128)
    for g in range(G):
        tw = np.exp(-2j * np.pi * GRID[g] * T / N)
        Fs[g] = np.fft.fft(y * tw[None, None, :], axis=-1)
    g0 = int(np.argmin(np.abs(GRID)))
    rho = 10 ** (snr_db / 10.0)
    lp = log_i0(2.0 * rho * np.abs(Fs))          # log p(y_i | v, κ_g)
    err = {}
    # (i) plain
    vh = np.argmax(np.abs(Fs[g0][..., W0:W1]), axis=-1) + W0
    err['plain'] = (vh != v).sum()
    # (iii) marg: 逐符号对 g 求和
    Lm = lse(lp, axis=0)                          # [n, M, N]
    vh = np.argmax(Lm[..., W0:W1], axis=-1) + W0
    err['marg'] = (vh != v).sum()
    # (iii-b) margF: 帧 κ 后验（和积）
    lzv = lse(lp, axis=3)                         # [G, n, M]
    sc = lzv.sum(axis=2)                          # [G, n]
    logw = sc - lse(sc, axis=0)[None, :]          # [G, n] 归一化对数后验
    Lf = lse(lp + logw[:, :, None, None], axis=0)  # [n, M, N]
    vh = np.argmax(Lf[..., W0:W1], axis=-1) + W0
    err['margF'] = (vh != v).sum()
    # (iv) jml: max_κ Σ_i max_v
    gML = np.argmax(np.max(lp[..., W0:W1], axis=3).sum(axis=2), axis=0)  # [n]
    sel = lp[gML, np.arange(n)]                  # [n, M, N]
    vh = np.argmax(sel[..., W0:W1], axis=-1) + W0
    err['jml'] = (vh != v).sum()
    # (ii) plugin: 帧池化比值 κ̂（中值）→ 该偏移下 argmax |F|
    X0 = Fs[g0]
    j0 = np.argmax(np.abs(X0), axis=-1)
    aj = np.take_along_axis(np.abs(X0), j0[..., None], -1)[..., 0]
    jp = (j0 + 1) % N
    jm = (j0 - 1) % N
    ap = np.take_along_axis(np.abs(X0), jp[..., None], -1)[..., 0]
    am = np.take_along_axis(np.abs(X0), jm[..., None], -1)[..., 0]
    side = ap >= am
    r = np.where(side, ap, am) / np.maximum(aj, 1e-12)
    kh = np.where(side, r / (1 + r), -r / (1 + r))
    kf = np.median(kh, axis=1)                    # [n]
    for f in range(n):                            # 每帧一次偏移 FFT
        tw = np.exp(-2j * np.pi * kf[f] * T / N)
        Fp = np.abs(np.fft.fft(y[f] * tw[None, :], axis=-1))
        vhf = np.argmax(Fp[:, W0:W1], axis=-1) + W0
        err['plugin'] = err.get('plugin', 0) + (vhf != v[f]).sum()
    return err, float(np.mean(np.abs(((kf - kappa + 0.5) % 1.0) - 0.5)))


def ser_all(kappa, snr_db, n):
    tot = M * n
    acc = {}
    ke = 0.0
    nb = 50
    for s in range(0, n, nb):
        m = min(nb, n - s)
        y, v = gen(kappa, snr_db, m)
        e, d = one_batch(y, v, snr_db, kappa)
        for k in e:
            acc[k] = acc.get(k, 0) + e[k]
        ke += d * m
    return {k: acc[k] / tot for k in acc}, ke / n


def crossing(snrs, sers, target=0.10):
    sers = np.asarray(sers)
    if len(sers) == 0 or np.nanmin(sers) > 0.4:
        return np.nan            # 50% 地板（信息墙）
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


REC = ['plain', 'plugin', 'marg', 'margF', 'jml']
KAPPAS = [0.0, 0.2, 0.3, 0.4, -0.5]   # -0.5 ≡ +0.5（信息墙点，网格上）


def run_set(half, n_coarse=240, n_fine=500):
    global GRID, G
    GRID = np.round(np.arange(-half, half + 0.001, 0.05), 2)
    G = GRID.size
    print("==== grid = [%.2f, %.2f], %d pts ====" % (GRID[0], GRID[-1], G))
    print("threshold = per-sample SNR (dB) @ SER=10%; 'floor'=50% info wall")
    print("%6s | %8s %8s %8s %8s %8s | %9s %9s %8s"
          % ("kappa", "plain", "plugin", "marg", "margF", "jml",
             "marg-jml", "margF-jml", "E|khat|"))
    for kappa in KAPPAS:
        if abs(kappa) > half + 1e-9:
            print("%+6.2f | (true kappa outside this grid, skipped)" % kappa)
            continue
        coarse = np.arange(-24.0, -2.0, 1.0)
        cs = {r: [] for r in REC}
        for s in coarse:
            e, _ = ser_all(kappa, s, n_coarse)
            for r in REC:
                cs[r].append(e[r])
        lo, hi = np.inf, -np.inf
        for r in REC:
            c = crossing(coarse, cs[r])
            if np.isfinite(c):
                lo = min(lo, c - 1.25)
                hi = max(hi, c + 1.25)
        fine = np.arange(np.floor(lo * 4) / 4, hi, 0.25)
        allsnr = np.concatenate([coarse, fine])
        allser = {r: list(cs[r]) for r in REC}
        ksum, kn = 0.0, 0
        for s in fine:
            e, d = ser_all(kappa, s, n_fine)
            for r in REC:
                allser[r].append(e[r])
            ksum += d
            kn += 1
        order = np.argsort(allsnr)
        thr = {}
        for r in REC:
            thr[r] = crossing(allsnr[order], np.asarray(allser[r])[order])
        dj = thr['marg'] - thr['jml'] if np.isfinite(thr['marg']) else np.nan
        df = thr['margF'] - thr['jml'] if np.isfinite(thr['margF']) else np.nan
        print("%+6.2f | %8s %8s %8s %8s %8s | %9s %9s %8.3f"
              % (kappa,
                 *["%.2f" % thr[r] if np.isfinite(thr[r]) else "floor"
                   for r in REC],
                 "%.2f" % dj if np.isfinite(dj) else "--",
                 "%.2f" % df if np.isfinite(df) else "--",
                 ksum / max(kn, 1)))


t0 = time.time()
print("SF7 N=%d, M=%d uncoded syms, step 0.05" % (N, M))
run_set(0.5, n_coarse=150, n_fine=350)   # 全宽网格（±0.5，盲部署）
KAPPAS[:] = [0.0, 0.2]
run_set(0.2, n_coarse=150, n_fine=350)   # 窄网格（±0.2，γ-链先验收窄后）
print("elapsed %.1f s" % (time.time() - t0))

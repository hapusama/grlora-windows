# -*- coding: utf-8 -*-
"""任务二：增益组合算术。
Part A: SF8 代理上 2x2 矩阵 = {plain 整数 bin | PCM 零参数重 bin} x {硬判决 | 软判决}。
        Hamming(8,4) 码本（16 码字 <-> 1 个 SF8 符号，Gray 映射），κ=0.4，8 符号/帧。
        软 = 码字后验精确边缘化；硬 = 全字母 argmax -> Gray 反映射 -> 最近码字（Hamming）。
        PCM 软证据模型 Z_v ~ CN(β e^{jφ}, 2Nσ²)（错设：忽略跨泄漏），β̂ 可缩放（标定敏感性）。
Part B: 全前导折叠 8 -> 12.25 已知符号，非相干统计量 T=Σ|r_l|² 的门限兑现率
        （相干参考 10log10(12.25/8)=1.86dB；精确 MC + deflection 高斯近似对照）。
"""
import numpy as np
import time

rng = np.random.default_rng(20261002)
N = 256                      # SF8
M = 8                        # 帧符号数
KAP = 0.4
T = np.arange(N)

# ---------- Hamming(8,4) 码本 + Gray 映射 ----------
Gm = np.array([[1, 0, 0, 0, 1, 1, 1, 0],
               [0, 1, 0, 0, 1, 1, 0, 1],
               [0, 0, 1, 0, 1, 0, 1, 1],
               [0, 0, 0, 1, 0, 1, 1, 1]])
info = np.arange(16)
cb = (((info[:, None] >> np.arange(4)[::-1]) & 1) @ Gm) % 2  # [16,8] 码字
cbi = [int(c) for c in (cb @ (1 << np.arange(7, -1, -1)))]   # 码字 bit 模式整数
dmin = min(bin(a ^ b).count('1') for a in cbi for b in cbi if a != b)
gray = lambda b: b ^ (b >> 1)
sym = np.array([gray(c) for c in cbi])                       # 码字 -> 符号值
# gray 逆映射表
ginv = np.zeros(256, dtype=np.int64)
for b in range(256):
    ginv[gray(b)] = b
# 距离矩阵: 符号 v -> gray 逆映射 bit 模式 -> 到各码字 Hamming 距离
dist = np.zeros((256, 16), dtype=np.int64)
for v in range(256):
    dist[v] = [bin(int(ginv[v]) ^ int(c)).count('1') for c in cbi]
nearest = np.argmin(dist, axis=1)   # [256] 符号 -> 码字索引（平局取首个）


def log_i0(x):
    x = np.abs(x)
    return np.where(x < 300.0,
                    np.log(np.i0(np.minimum(x, 300.0)) + 1e-300),
                    x - 0.5 * np.log(2 * np.pi * np.maximum(x, 300.0)))


def lse(a, axis=-1):
    am = np.max(a, axis=axis, keepdims=True)
    return (am + np.log(np.sum(np.exp(a - am), axis=axis, keepdims=True)))[..., 0]


# PCM 抽头 β(κ) = d0 - e^{-jπ/N} d1（真 κ，标定上限）
d0 = np.exp(1j * np.pi * KAP * (N - 1) / N) * np.sin(np.pi * KAP) / np.sin(np.pi * KAP / N)
d1 = np.exp(1j * np.pi * (KAP - 1) * (N - 1) / N) * np.sin(np.pi * (KAP - 1)) \
     / np.sin(np.pi * (KAP - 1) / N)
BETA = abs(d0 - np.exp(-1j * np.pi / N) * d1)
print("Hamming(8,4) dmin=%d; |d0|^2/N=%.1f |d1|^2/N=%.1f |beta|^2/N=%.1f  (PCM 相干合并)"
      % (dmin, abs(d0) ** 2 / N, abs(d1) ** 2 / N, BETA ** 2 / N))


def gen(n, snr_db):
    ci = rng.integers(0, 16, size=(n, M))
    v = sym[ci]
    ph = rng.uniform(0, 2 * np.pi, size=(n, M))
    y = np.exp(2j * np.pi * (v[..., None] + KAP) * T / N + ph[..., None])
    sg = 10 ** (-snr_db / 10.0)
    y += np.sqrt(sg / 2) * (rng.standard_normal(y.shape)
                            + 1j * rng.standard_normal(y.shape))
    return y, ci


def cer_cell(y, ci, snr_db, reb, soft, sfac=1.0):
    """reb: 0=plain 1=PCM; soft: 0/1; sfac: |β̂| 缩放。返回 CER。"""
    n = y.shape[0]
    X = np.fft.fft(y, axis=-1)                     # [n,M,N]
    rho = 10 ** (snr_db / 10.0)
    sg2 = 1.0 / rho                                 # 每样本噪声方差
    if reb == 0:
        m = log_i0(2.0 * rho * np.abs(X))           # [n,M,N]（κ=0 模型 LLR-vs-噪声）
    else:
        s2 = 2.0 * N * sg2                          # Z 噪声方差
        Z = X - np.exp(-1j * np.pi / N) * np.roll(X, -1, axis=-1)
        # 与 plain 行同构的 LLR-vs-噪声形式（标定参数 |β̂|=sfac·BETA）
        m = log_i0(2.0 * sfac * BETA * np.abs(Z) / s2)
    if soft == 0:
        vh = np.argmax(np.abs(X) if reb == 0 else np.abs(Z), axis=-1)
        cdec = nearest[vh]
        err = (cdec != ci).sum()
    else:
        mc = m[:, :, sym]                           # [n,M,16]
        pz = np.exp(mc - mc.max(axis=-1, keepdims=True))
        pz /= pz.sum(axis=-1, keepdims=True)
        p1 = pz @ cb                                # bit=1 后验 [n,M,8]
        bdec = (p1 > 0.5).astype(np.int64)
        err = np.sum(np.any(bdec != cb[ci], axis=-1))
    return err / (n * M)


def threshold(fn, lo=-26.0, hi=-6.0, coarse=1.0, fine=0.25, target=0.10,
              nc=150, nf=400):
    snrs = np.arange(lo, hi + 1e-9, coarse)
    sers = []
    for s in snrs:
        y, ci = gen(nc, s)
        sers.append(fn(y, ci, s))
    sers = np.array(sers)
    i = np.where(sers <= target)[0]
    if len(i) == 0 or i[0] == 0:
        return np.nan, snrs, sers
    c = snrs[i[0] - 1]
    fs = np.arange(c, c + coarse + 1e-9, fine)
    fser = []
    for s in fs:
        y, ci = gen(nf, s)
        fser.append(fn(y, ci, s))
    snrs2 = np.concatenate([snrs, fs])
    sers2 = np.concatenate([sers, fser])
    o = np.argsort(snrs2)
    sers2, snrs2 = sers2[o], snrs2[o]
    l = np.log10(np.maximum(sers2, 1e-6))
    j = np.where(sers2 <= target)[0][0]
    k = j - 1
    return snrs2[k] + (np.log10(target) - l[k]) * (snrs2[j] - snrs2[k]) / (l[j] - l[k]), \
        snrs2, sers2


t0 = time.time()
print("== Part A: 2x2 matrix, SF8, kappa=0.4, Hamming(8,4)/Gray, M=%d ==" % M)
cells = {}
for reb in (0, 1):
    for soft in (0, 1):
        th, _, _ = threshold(lambda y, c, s, r=reb, so=soft: cer_cell(y, c, s, r, so))
        cells[(reb, soft)] = th
        print("  reb=%d soft=%d : CER10%% threshold = %+.2f dB"
              % (reb, soft, th))
d1g = cells[(0, 0)] - cells[(1, 0)]     # 重 bin 增益（硬行）
d5g = cells[(0, 0)] - cells[(0, 1)]     # 软增益（plain 列）
tot = cells[(0, 0)] - cells[(1, 1)]
print("  Δrebin(hard)=%.2f dB  Δsoft(plain)=%.2f dB  total=%.2f dB  协同=%.2f dB"
      % (d1g, d5g, tot, tot - d1g - d5g))

# 标定敏感性: (PCM, soft) 单元在固定 SNR 下扫 sfac
sth = cells[(1, 1)]
if np.isfinite(sth):
    s0 = sth + 1.0
    print("  -- LLR 标定敏感性 (PCM,soft) @ SNR=%+.2f dB --" % s0)
    for sf in (0.7, 0.85, 1.0, 1.15, 1.3):
        e = 0.0
        nt = 0
        for _ in range(3):
            y, ci = gen(200, s0)
            e += cer_cell(y, ci, s0, 1, 1, sfac=sf) * 200 * M
            nt += 200 * M
        print("    sfac=%.2f : CER=%.4f" % (sf, e / nt))

# ---------- Part B: 前导折叠非相干兑现率 ----------
print("== Part B: preamble fold 8 -> 12.25, T=sum|r|^2 ==")
print("   coherent ref = 10log10(12.25/8) = %.3f dB" % (10 * np.log10(12.25 / 8)))


def stat_T(L, sig_db=None, nd=2000000):
    """T = sum_{l<Lc-1}|r_l|^2 + 0.25*|r_last|^2（L=12.25 的折叠统计量）。
    sig_db=None -> H0 纯噪声；否则 r = sqrt(E)+cn(0,1)。"""
    Lc = int(np.ceil(L))
    w = np.ones(Lc)
    if L != Lc:
        w[-1] = L - (Lc - 1)
    if sig_db is None:
        r = (rng.standard_normal((nd, Lc)) + 1j * rng.standard_normal((nd, Lc))) \
            * np.sqrt(0.5)
    else:
        E = 10 ** (sig_db / 10.0)
        r = np.sqrt(E) + (rng.standard_normal((nd, Lc))
                          + 1j * rng.standard_normal((nd, Lc))) * np.sqrt(0.5)
    return np.sum(w * np.abs(r) ** 2, axis=1)


def pd_at(L, E_db, thr, nd=300000):
    return float(np.mean(stat_T(L, sig_db=E_db, nd=nd) > thr))


def thr_of(L, pfa, nd=2000000):
    return float(np.quantile(stat_T(L, sig_db=None, nd=nd), 1 - pfa))


def snr_pd90(L, pfa):
    thr = thr_of(L, pfa)
    lo, hi = -12.0, 6.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if pd_at(L, mid, thr) < 0.9:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), thr


def deflect(L, pfa, pd=0.9):
    from math import sqrt
    zfa = {1e-3: 3.0902, 1e-2: 2.3263}[pfa]
    zd = 1.2816
    # deflection 判据: (LE)^2/(L(1+2E)) = ((zfa sqrt(1) + zd sqrt(1+2E))... 简化高斯:
    # t = L + zfa*sqrt(L); Pd: L(1+E)+zd*sqrt(L(1+2E)) = t
    lo, hi = 0.001, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if L * (1 + mid) + zd * sqrt(L * (1 + 2 * mid)) < L + zfa * sqrt(L):
            lo = mid
        else:
            hi = mid
    return 10 * np.log10(0.5 * (lo + hi))


for pfa in (1e-3, 1e-2):
    s8, t8 = snr_pd90(8.0, pfa)
    s12, t12 = snr_pd90(12.25, pfa)
    gain = s8 - s12
    ref = 10 * np.log10(12.25 / 8)
    print("  Pfa=%.0e: SNR@Pd90: L=8 %+.2f dB, L=12.25 %+.2f dB -> 增益 %.2f dB"
          " (兑现率 %.0f%%)" % (pfa, s8, s12, gain, 100 * gain / ref))
    d8 = deflect(8.0, pfa)
    d12 = deflect(12.25, pfa)
    print("    deflection 高斯近似: L=8 %+.2f, L=12.25 %+.2f -> 增益 %.2f dB"
          " (精确 %.2f dB)" % (d8, d12, d8 - d12, gain))
print("elapsed %.1f s" % (time.time() - t0))

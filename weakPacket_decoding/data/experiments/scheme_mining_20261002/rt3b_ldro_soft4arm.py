# -*- coding: utf-8 -*-
"""rt3b 任务一：LDRO 行校验消歧——正确映射下的软先验四臂全链量化。

红队勘误语义（gray_mapping_impl.cc L70 单步 fold；interleaver_impl.cc
L203-207 校验位在 2^1、LSB 恒 0）：相邻 bin 候选差 = 单 bit 2^t（t=尾随1数），
t=0 踩恒零 LSB、t=1 单翻校验位、t>=2 单翻数据位 → 行校验(数据⊕校验)对
相邻 bin 混淆对 100% 确定性击杀（每 coarse 组 4 个精细 bin 恰 1 个合法；
本脚本开头做全空间结构断言）。

基线 A = 4-bin 相干块（定律3机制）：候选 tone ν = bin(w)+κ̂，对 straddle
窗口 bin-1..bin+2 做 κ̂ 条件化共轭 Dirichlet 4 抽头相干合并（pcm_adjudicate2
零参数 2 抽头的 κ̂ 条件化推广），帧公共 κ̂。

臂（同噪声，SF9-LDRO 代理、CR4/8 Hamming(8,4)、NB=8 块/帧=64 符号，
帧公共 κ；A/B/C 用 genie κ̂，D 用帧池化 κ̂）：
  A   基线 + 软判决
  B   A + 行校验软先验（非法行 LLR 罚 W nat，W∈{2,4,8}）
  C   A + 硬剪枝（W=∞，红队 A1 对照）
  D   B(W=4) + 帧池化 κ̂（γ-链代理：跨踞两 bin 比值估计 × 64 符号圆均值）
  附 P0 plain argmax|X|（group-max 参照，定律3 锚点复核）。
κ 扫描 {0.30..0.50} + (-0.40) 符号抽查；SER10/CER10 双门限。
运行：py -3.12 rt3b_ldro_soft4arm.py
"""
import numpy as np
import time

SF, N = 9, 512
NSB, SFA, NB = 8, 7, 8          # cw_len=8 符号/块, 7 码字/块, 8 块/帧
Gm = np.array([[1, 0, 0, 0, 1, 1, 1, 0],
               [0, 1, 0, 0, 1, 1, 0, 1],
               [0, 0, 1, 0, 1, 0, 1, 1],
               [0, 0, 0, 1, 0, 1, 1, 1]])
cb = (((np.arange(16)[:, None] >> np.arange(4)[::-1]) & 1) @ Gm) % 2   # [16,8]

def cumfold(v):
    g = v.copy()
    for sh in range(1, SF):
        g ^= v >> sh
    return g

Wall = np.arange(N)                                    # 全部 9-bit w
db9 = ((Wall[:, None] >> np.arange(SF - 1, -1, -1)) & 1)        # [512,9] MSB-first
LEGAL = ((Wall & 1) == 0) & (((Wall >> 1) & 1) == (db9[:, :7].sum(1) % 2))
BINw = (cumfold(Wall) + 1) % N                         # TX bin
assert LEGAL.sum() == 128

# ---- 红队勘误结构断言：相邻 bin 候选差恒单 bit 且行校验 100% 击杀 ----
b = np.arange(N)
w1 = b ^ (b >> 1)                                      # RX 单步 fold(bin)
w2m = ((b + 1) % N)
w2 = w2m ^ (w2m >> 1)
dbits = w1 ^ w2
_t = []
_kill = 0
for i in range(N):
    d = int(dbits[i])
    assert d & (d - 1) == 0, "相邻 bin 候选差非单 bit"
    # 单 bit 2^t: t=0 踩 LSB, t=1 踩校验位, t>=2 踩数据位 -> 至多 1 个合法
    if LEGAL[w1[i]] and LEGAL[w2[i]]:
        _kill += 1
assert _kill == 0, "存在相邻 bin 对双双合法，击杀非 100%"
KILL100 = True

IDX = np.array([[(i - 1 - m) % SFA for i in range(NSB)] for m in range(SFA)])  # [m,i]
U4 = np.arange(4)

def dirich(delta):
    d = np.asarray(delta, dtype=np.complex128)
    out = np.exp(1j * np.pi * d * (N - 1) / N) * np.sin(np.pi * d) \
        / np.sin(np.pi * d / N)
    return np.where(np.abs(d) < 1e-12, N + 0j, out)

def taps(khat):
    """H[u] = D(khat+1-u)，窗口 bins {b-1..b+2}，u=0..3."""
    H = dirich(khat + 1 - U4)
    return H, float((np.abs(H) ** 2).sum())

def logi0(x):
    x = np.abs(x)
    return np.where(x > 30, x - 0.5 * np.log(2 * np.pi * np.maximum(x, 30)),
                    np.log(np.i0(np.minimum(x, 30)) + 1e-300))

def gen(kappa, gamma_db, n, seed):
    rng = np.random.default_rng(seed)
    ci = rng.integers(0, 16, size=(n, NB, SFA))
    cwb = cb[ci]
    wt = np.zeros((n, NB, NSB), dtype=np.int64)
    row = np.zeros((n, NB, SF), dtype=np.int8)
    for i in range(NSB):
        for j in range(SFA):
            row[:, :, j] = cwb[:, :, (i - j - 1) % SFA, i]
        row[:, :, SFA] = row[:, :, :SFA].sum(2) % 2
        wt[:, :, i] = row @ (1 << np.arange(SF - 1, -1, -1))
    bins = BINw[wt]
    sig = np.exp(2j * np.pi * (bins[..., None] + kappa) * np.arange(N) / N)
    sg = 10.0 ** (-gamma_db / 10.0)
    y = sig + np.sqrt(sg / 2) * (rng.standard_normal(sig.shape)
                                 + 1j * rng.standard_normal(sig.shape))
    return y.reshape(n, NB * NSB, N), wt.reshape(n, NB * NSB), ci

def pooled_khat(Xa):
    """跨踞比值 κ̂ × 帧内 64 符号圆均值（γ-链 WLS 的轻量代理）."""
    p = Xa.argmax(-1)                                   # [n,64]
    n_, s_ = Xa.shape[0], Xa.shape[1]
    fr = np.arange(n_)[:, None]
    sy = np.arange(s_)[None, :]
    P0 = Xa[fr, sy, p]
    R = Xa[fr, sy, (p + 1) % N]
    Lf = Xa[fr, sy, (p - 1) % N]
    sdir = np.where(R >= Lf, 1.0, -1.0)
    big = np.maximum(R, Lf)
    ks = sdir * big / np.maximum(P0 + big, 1e-9)
    ks = ((ks + 0.5) % 1.0) - 0.5
    phv = (P0 ** 2 * np.exp(2j * np.pi * ks)).mean(axis=1)   # 能量加权圆均值
    kp = np.angle(phv) / (2 * np.pi)
    return np.where(np.abs(phv) < 1e-9, 0.0, kp)

def mf_logL(X, khat, sigma2):
    """X [n,64,N] -> L [n*64,512]。khat 标量(genie) 或 [n] 帧向量(池化)."""
    nf = X.shape[0]
    if np.ndim(khat) == 0:
        Hlist = [taps(float(khat))[0]] * nf
        Sarr = np.full(nf, taps(float(khat))[1])
    else:
        Hlist = [taps(k)[0] for k in np.atleast_1d(khat)]
        Sarr = np.array([taps(k)[1] for k in np.atleast_1d(khat)])
    Xc = X.astype(np.complex64)
    rolls = [np.roll(Xc, 1 - u, axis=-1) for u in range(4)]   # bins b-1+u
    out = np.empty((nf, 64, N), dtype=np.float32)
    for c0 in range(0, nf, 64):
        sl = slice(c0, min(c0 + 64, nf))
        mf = None
        for u in range(4):
            if np.ndim(khat) == 0:
                add = np.conj(Hlist[0][u]) * rolls[u][sl]
            else:
                hh = np.array([np.conj(Hlist[c][u]) for c in range(sl.start, sl.stop)])
                add = hh[:, None, None] * rolls[u][sl]
            mf = add if mf is None else mf + add
        lam = (Sarr[sl] / (N * sigma2))[:, None, None]
        out[sl] = (logi0(2 * np.abs(mf) / (N * sigma2)) - lam).astype(np.float32)
    return out.reshape(nf * 64, N)[:, BINw]             # -> w 字母表序

def decide(Lg, Lp, wt_flat, ci, arms):
    out = {}
    n = wt_flat.shape[0] // (NB * NSB)
    for name, pw, tag in arms:
        Lx = (Lg if tag == 'g' else Lp) + pw[None, :]
        mx = Lx.max(-1, keepdims=True)
        pz = np.exp(Lx - mx)
        pz /= pz.sum(-1, keepdims=True)
        ser = float((Lx.argmax(-1) != wt_flat).mean())
        p1 = pz @ db9
        pb = np.clip(p1[:, :7], 1e-12, 1 - 1e-12).reshape(n, NB, NSB, 7)
        pv = pb[:, :, np.arange(NSB)[None, :], IDX]      # [n,NB,m,i]
        cbb = cb.astype(np.float64)
        ll = np.einsum('nbmi,ci->nbmc', np.log(pv), cbb) \
            + np.einsum('nbmi,ci->nbmc', np.log1p(-pv), 1 - cbb)
        cer = float((ll.argmax(-1) != ci).mean())
        out[name] = (ser, cer)
    return out

ARMS = [('A', None, 'g'), ('B2', 2.0, 'g'), ('B4', 4.0, 'g'), ('B8', 8.0, 'g'),
        ('C', np.inf, 'g'), ('D2', 2.0, 'p'), ('D4', 4.0, 'p'),
        ('Dh', np.inf, 'p'), ('Ap', None, 'p')]

def eval_all(kappa, gamma_db, n, seed, khat_bias=0.0):
    y, wt, ci = gen(kappa, gamma_db, n, seed)
    sigma2 = 10.0 ** (-gamma_db / 10.0)
    X = np.fft.fft(y, axis=-1)
    Xa = np.abs(X)
    kpool = pooled_khat(Xa)
    Lg = mf_logL(X, kappa + khat_bias, sigma2)
    Lp = mf_logL(X, kpool, sigma2)
    arms = [(nm, np.zeros(512, np.float32) if w is None
             else np.where(LEGAL, 0.0, -w).astype(np.float32), tg)
            for nm, w, tg in ARMS]
    res = decide(Lg, Lp, wt.reshape(-1), ci, arms)
    bhat = Xa.argmax(-1).reshape(-1)
    wpl = (bhat - 1) % N
    wpl = wpl ^ (wpl >> 1)                               # RX 单步 fold
    res['P0'] = (float((wpl != wt.reshape(-1)).mean()), np.nan)
    kerr = ((kpool - kappa + 0.5) % 1.0) - 0.5
    res['_khat'] = (float(np.sqrt((kerr ** 2).mean())), np.nan)
    return res

CACHE = {}

def cev(kappa, gamma_db, n, seed):
    key = (kappa, round(gamma_db, 3), n, seed)
    if key not in CACHE:
        CACHE[key] = eval_all(kappa, gamma_db, n, seed)
    return CACHE[key]

def crossing(pts, target=0.10):
    pts = sorted([p for p in pts if np.isfinite(p[1]) and p[1] > 0])
    for (g0, v0), (g1, v1) in zip(pts[:-1], pts[1:]):
        if (v0 - target) * (v1 - target) <= 0 and v0 != v1:
            return g0 + (np.log10(target) - np.log10(v0)) * (g1 - g0) \
                / (np.log10(v1) - np.log10(v0))
    return np.nan

def bisect_cer(kappa, arm, lo, hi, n, seed, iters=6, target=0.10):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cev(kappa, mid, n, seed + 1)[arm][1] > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

if __name__ == '__main__':
    t0 = time.time()
    print("结构断言通过: 相邻 bin 差恒单 bit, 行校验击杀 100%")
    KAPS = [0.30, 0.35, 0.40, 0.45, 0.50, -0.40]
    seedc = lambda kap: 70000 + int(round(kap * 100)) * 977
    seedf = lambda kap: 90000 + int(round(kap * 100)) * 131
    RES = {}
    for kap in KAPS:
        grid = np.arange(-24.0, -10.9, 1.0)
        for g in grid:
            cev(kap, g, 140, seedc(kap))
        for arm in [a[0] for a in ARMS]:
            pts = [(g, CACHE[(kap, round(g, 3), 140, seedc(kap))][arm][1])
                   for g in grid]
            v = [p[1] for p in pts]
            lo = hi = None
            for i in range(len(v) - 1):
                if v[i] >= 0.1 > v[i + 1]:
                    lo, hi = grid[i] - 1.0, grid[i + 1] + 0.5
                    break
            if lo is None:
                lo, hi = grid[0], grid[-1]
            bisect_cer(kap, arm, lo, hi, 250, seedf(kap), iters=5)
        row = {}
        ptsS, ptsC = {}, {}
        for (k, g, n, s), r in CACHE.items():
            if k != kap:
                continue
            for arm in [a[0] for a in ARMS] + ['P0']:
                ptsS.setdefault(arm, []).append((g, r[arm][0]))
                if arm != 'P0':
                    ptsC.setdefault(arm, []).append((g, r[arm][1]))
        for arm in ptsS:
            row[arm] = (crossing(ptsS[arm]),
                        crossing(ptsC[arm]) if arm in ptsC else np.nan)
        RES[kap] = row
        kr = [r['_khat'][0] for (k, g, n, s), r in CACHE.items() if k == kap]
        names = ['P0', 'A', 'B2', 'B4', 'B8', 'C', 'D2', 'D4', 'Dh', 'Ap']
        print("kappa=%+5.2f SER10:" % kap + "".join(
            " %s=%+6.2f" % (a, row[a][0]) for a in names)
            + " | khat_rmse=%.4f" % np.mean(kr), flush=True)
        print("            CER10:" + "".join(
            " %s=%+6.2f" % (a, row[a][1]) for a in
            ['A', 'B2', 'B4', 'B8', 'C', 'D2', 'D4', 'Dh', 'Ap']), flush=True)
    DARM = ['B2', 'B4', 'B8', 'C', 'D2', 'D4', 'Dh', 'Ap']
    print("\n== ΔdB(SER10, 相对 A；正=更深) ==")
    for kap in KAPS:
        r = RES[kap]
        print("%+5.2f" % kap + "".join(" %+7.2f" % (r['A'][0] - r[a][0])
                                       for a in DARM))
    print("\n== ΔdB(CER10, 相对 A) ==")
    for kap in KAPS:
        r = RES[kap]
        print("%+5.2f" % kap + "".join(" %+7.2f" % (r['A'][1] - r[a][1])
                                       for a in DARM))
    print("\n== 定律3 锚点: P0(group-max) vs A(4-bin 相干块) SER10 ==")
    for kap in KAPS:
        print("  kappa=%+5.2f : %+0.2f dB" % (kap, RES[kap]['P0'][0] - RES[kap]['A'][0]))
    print("elapsed %.1f s, cache evals=%d" % (time.time() - t0, len(CACHE)))

    # ---- 阶段2: κ̂ 偏差 x 先验权重 交互（门控/退火依据）----
    print("\n== κ̂ 偏差敏感性 (κ=0.45 与 0.35, 固定 γ 近门限, n=400) ==")
    for kap, gg in ((0.45, -17.2), (0.35, -16.6)):
        print(" kappa=%+.2f @ gamma=%.1f" % (kap, gg))
        for dlt in (0.0, 0.02, 0.05, 0.10, 0.20):
            r = eval_all(kap, gg, 400, 555, khat_bias=dlt)
            r0 = eval_all(kap, gg, 400, 555, khat_bias=0.0)
            print("   delta=%+.2f : A %.3f | B2 %.3f B4 %.3f C %.3f "
                  "(ΔSER vs A: %+.3f %+.3f %+.3f)"
                  % (dlt, r['A'][0], r['B2'][0], r['B4'][0], r['C'][0],
                     r['A'][0] - r['B2'][0], r['A'][0] - r['B4'][0],
                     r['A'][0] - r['C'][0]), flush=True)
    print("total elapsed %.1f s" % (time.time() - t0))

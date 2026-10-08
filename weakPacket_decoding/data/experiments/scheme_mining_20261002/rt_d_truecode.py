# -*- coding: utf-8 -*-
"""红队攻击 (d)：真码 GLRT vs 8 符号前导 —— A1 支配论证的 H0 侧漏洞检验。

A1 论证：T3c（无约束序列相干 GLRT）统计量 >= 任何码约束序列统计量
→ T3c 输前导 ~5dB ⇒ 全家族封死。漏洞：统计量逐点支配 ≠ ROC 支配，
码约束同时收紧 H0 分布（候选集 2^32 vs 2^40 每块）→ 同 FAR 门限更低。

真码版（真实 codec 约定，SF8 CR4/5，30 符号 payload = 6 块×5 符号，
每块 8 个 (5,4) 单校验码字 + 对角交织；TX bin = 累积fold(w)+1；
RX 候选值差 = 单 bit（Gray 性质））：
  T1    8 符号前导相干 GLRT（κ 16 格 max）
  T3c   无约束序列相干 GLRT（top-J=12，自适应 φ 三点）
  T3cod 码约束序列相干 GLRT —— 块内精确解：
        块 = 符号 i=0..4，符号 i 的 8 bit = 各码字的 bit i 向量 x_i
        （i=0..3 自由 = d3..d0；i=4 = x0⊕x1⊕x2⊕x3 校验）
        blockmax = max_t [ XORconv(XORconv(h0,h1), XORconv(h2,h3))[t] + h4[t] ]
        XORconv(A,B)[t] = max_u A[u]+B[u^t]  （精确，非近似）
  同帧同噪声喂三检测器；FAR 1e-2 frontier (Pd=0.9) 对比。

运行：py -3.12 rt_d_truecode.py
"""
import numpy as np
import json
import time

M, NP, NS, K, NB = 256, 8, 30, 16, 6          # NB: 6 blocks x 5 symbols
BLK = 5
kap = np.arange(K) / K
mm = np.arange(M)
TW = np.exp(-2j * np.pi * np.outer(kap, mm) / M)
XOR = np.array([[t ^ u for u in range(256)] for t in range(256)])  # [t,u]

# ---- 真实 codec 查找表：x(8bit 码字位向量) -> TX bin ----
def cumfold(v):
    v = int(v)
    g = v
    for sh in range(1, 8):
        g ^= v >> sh
    return g

def make_luts():
    luts = []
    for i in range(BLK):
        lut = np.zeros(256, dtype=np.int64)
        for x in range(256):
            w = 0
            for m in range(8):
                if (x >> m) & 1:
                    w |= 1 << (7 - ((i - 1 - m) % 8))
            lut[x] = (cumfold(w) + 1) & 255
        luts.append(lut)
    return luts

LUT = make_luts()

def encode54(d):        # d: [...,4] -> 5bit cw
    p4 = d[..., 0] ^ d[..., 1] ^ d[..., 2] ^ d[..., 3]
    return (d @ (1 << np.array([3, 2, 1, 0]))) | p4  # 低 5 bit: d3 d2 d1 d0 p4

def gen_payload(rng, n):
    """随机 nibble -> 30 个 TX bin [n,30]"""
    out = np.zeros((n, NS), dtype=np.int64)
    for b in range(NB):
        d = rng.integers(0, 2, size=(n, 8, 4))
        cw = encode54(d)                                   # [n,8] 5bit
        xb = ((cw[:, :, None] >> np.arange(4, -1, -1)) & 1)  # [n,8,5] bit i
        for i in range(BLK):
            x = 0
            for m in range(8):
                x |= xb[:, m, i].astype(np.int64) << m     # x_i 8bit 向量
            out[:, b * BLK + i] = LUT[i][x]
    return out

def gen_noise(rng, n, sigma):
    return sigma * ((rng.standard_normal((n, NP + NS, M))
                     + 1j * rng.standard_normal((n, NP + NS, M))) / np.sqrt(2))

def xorconv(A, B):
    """out[f,t] = max_u A[f,u] + B[f, u^t] ; A,B [n,256] float32"""
    return (A[:, None, :] + B[:, XOR]).max(axis=2)      # [n,256]

def stats(y, sigma, J=12, dphi=np.pi / 12):
    """返回 T1, T3c, T3cod（同帧）。"""
    n = y.shape[0]
    sc = 1.0 / (sigma * np.sqrt(M))
    T1 = np.zeros(n)
    T3c = np.zeros(n)
    T3d = np.zeros(n)
    for g in range(K):
        zp = np.einsum('psm,gm->psg', y[:, :NP, :], TW[g:g + 1])[:, :, 0] * sc
        T1 = np.maximum(T1, np.abs(zp.sum(1)) ** 2)
        Z = np.fft.fft(y[:, NP:, :] * TW[g], axis=-1) * sc   # [n,30,256]
        mag = np.abs(Z)
        idx = np.argpartition(mag, -J, axis=-1)[..., -J:]     # top-J bins
        Zt = np.take_along_axis(Z, idx, axis=-1)
        ph = np.angle((Zt.mean(axis=1)).sum(axis=1))          # 自适应相位锚
        for k in (-1, 0, 1):
            e = np.exp(-1j * (ph + k * dphi))
            proj = (Zt * e[:, None, None]).real               # [n,30,J]
            T3c = np.maximum(T3c, proj.max(axis=-1).sum(axis=1) ** 2)
            gg = (Z * e[:, None, None]).real.astype(np.float32)  # [n,30,256]
            tot = np.zeros(n, np.float32)
            for b in range(NB):
                h = [gg[:, b * BLK + i, LUT[i]] for i in range(BLK)]
                blk = xorconv(xorconv(h[0], h[1]), xorconv(h[2], h[3]))
                tot += (blk + h[4]).max(axis=1)
            T3d = np.maximum(T3d, tot ** 2)
    return T1, T3c, T3d

def run(gamma_db, n0, n1, seed, heavy=True):
    rng = np.random.default_rng(seed)
    sigma = 10.0 ** (-gamma_db / 20.0)
    res = {}
    for tag, n in (('h0', n0), ('h1', n1)):
        if n == 0:
            continue
        a1, a2, a3 = [], [], []
        for i in range(0, n, 32):
            c = min(32, n - i)
            w = gen_noise(rng, c, 1.0 if tag == 'h0' else sigma)
            if tag == 'h1':
                nu = np.concatenate([np.zeros((c, NP), int),
                                     gen_payload(rng, c)], axis=1)
                phi = rng.uniform(0, 2 * np.pi, c)
                y = w + np.exp(2j * np.pi * nu[..., None] * mm[None, None, :] / M
                               + 1j * phi[:, None, None])
                t1, t2, t3 = stats(y, sigma)
            else:
                t1, t2, t3 = stats(w, 1.0)
            a1.append(t1); a2.append(t2); a3.append(t3)
        res[tag] = (np.concatenate(a1), np.concatenate(a2), np.concatenate(a3))
    return res

def frontier(f, det_i, lo, hi, iters, thr):
    rng = np.random.default_rng(0)
    for it in range(iters):
        mid = 0.5 * (lo + hi)
        sigma = 10.0 ** (-mid / 20.0)
        a = [[], [], []]
        for i in range(0, 300, 32):
            c = min(32, 300 - i)
            w = gen_noise(rng, c, sigma)
            nu = np.concatenate([np.zeros((c, NP), int), gen_payload(rng, c)], axis=1)
            phi = rng.uniform(0, 2 * np.pi, c)
            y = w + np.exp(2j * np.pi * nu[..., None] * mm[None, None, :] / M
                           + 1j * phi[:, None, None])
            tt = stats(y, sigma)
            for j in range(3):
                a[j].append(tt[j])
        pd = float(np.mean(np.concatenate(a[det_i]) > thr))
        print('    g=%+.2f Pd=%.3f' % (mid, pd), flush=True)
        if pd >= 0.9:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def main():
    t0 = time.time()
    # H0 门限（统计量与 sigma 无关，单点即可）
    r = run(-30.0, 1600, 0, 777)
    h0 = r['h0']
    thr = {}
    for far in (3e-2, 1e-2, 3e-3):
        thr[far] = [float(np.quantile(h0[i], 1 - far)) for i in range(3)]
    print('H0 thresholds (T1,T3c,T3cod):', {k: [round(x, 1) for x in v] for k, v in thr.items()}, flush=True)
    bias = [[float(np.sqrt(h0[i]).mean()) for i in range(3)]]
    print('E[sqrt stat] H0 (T1,T3c,T3cod):', [round(x, 2) for x in bias[0]], flush=True)

    far = 1e-2
    f1 = frontier(far, 0, -24.6, -22.4, 5, thr[far][0])
    f2 = frontier(far, 1, -19.6, -17.4, 5, thr[far][1])
    f3 = frontier(far, 2, -23.5, -16.5, 6, thr[far][2])
    print('FAR=1e-2 frontiers: T1=%+.2f T3c=%+.2f T3cod=%+.2f dB'
          % (f1, f2, f3), flush=True)
    print('gaps: T3c-T1=%+.2f  T3cod-T1=%+.2f  (T3cod 追回 %.2f dB)'
          % (f2 - f1, f3 - f1, f2 - f3), flush=True)
    rep = dict(thr={str(k): v for k, v in thr.items()},
               sqrt_mean_h0=bias[0],
               frontier=dict(T1=f1, T3c=f2, T3cod=f3),
               gap_T3c_T1=f2 - f1, gap_T3cod_T1=f3 - f1, recovery=f2 - f3,
               elapsed=time.time() - t0)
    with open('rt_d_truecode_results.json', 'w') as f:
        json.dump(rep, f, indent=1, default=float)
    print('saved rt_d_truecode_results.json (%.1f s)' % (time.time() - t0))

if __name__ == '__main__':
    main()

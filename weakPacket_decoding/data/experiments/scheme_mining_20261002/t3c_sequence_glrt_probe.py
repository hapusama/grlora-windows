# -*- coding: utf-8 -*-
"""T3c = 无约束 payload 序列相干 GLRT（coding-as-detection 的最强天花板）。

T3c = max_κ max_φ [Σ_n max_v Re(e^{-jφ} ζ_n(v+κ))]²
    = max_κ max_{v-seq} |Σ_n ζ_n(v_n+κ)|²   （逐符号 argmax 分解精确成立）

它 >= 任何码约束序列统计量（码只会缩小候选集）。参数无关（不需 λ0）。
若 T3c 仍大幅输给 8 符号相干前导 T1，则 coding-as-detection 全家族（含
码约束 GLRT、BCJR-Λ 检测读出）的"相干增益口径"被数值封死。

共享 coding_as_detection_probe 的模型与常数（同噪声实现同 σ 口径）。
"""
import numpy as np
import time
import coding_as_detection_probe as P

M, NS, NP, K = P.M, P.NS, P.NP, P.K
TW = P.TW
PHI = np.linspace(0, 2 * np.pi, 48, endpoint=False)

def t3c_stats(y, sigma):
    sc = 1.0 / (sigma * np.sqrt(M))
    n = y.shape[0]
    best = np.full(n, -np.inf)
    for g in range(K):
        Z = np.fft.fft(y[:, NP:, :] * TW[g], axis=-1) * sc   # (n,NS,M)
        # top-J bins per symbol by |zeta| (J=12): argmax over v of Re for any
        # phi lies in top-J with overwhelming prob; slightly pessimistic if not
        mag = np.abs(Z)
        J = 12
        idx = np.argpartition(mag, -J, axis=-1)[..., -J:]    # (n,NS,J)
        Zt = np.take_along_axis(Z, idx, axis=-1)
        acc = np.zeros((n, len(PHI)))
        for i, ph in enumerate(PHI):
            proj = (Zt * np.exp(-1j * ph)).real              # (n,NS,J)
            acc[:, i] = proj.max(axis=-1).sum(axis=-1)
        best = np.maximum(best, acc.max(axis=-1) ** 2)
    return best

def run_block(gamma_db, n0, n1, seed):
    rng = np.random.default_rng(seed)
    sigma = 10.0 ** (-gamma_db / 20.0)
    out = {}
    for tag, n in (('h0', n0), ('h1', n1)):
        acc = []
        for i in range(0, n, 64):
            c = min(64, n - i)
            w = P.gen_noise(rng, c, NP + NS, sigma)
            vn = rng.integers(0, M, (c, NS))
            if tag == 'h1':
                nu = np.concatenate([np.zeros((c, NP)), vn], axis=1)
                phi = rng.uniform(0, 2 * np.pi, c)
                y = w + P.signal(rng, c, NP + NS, nu, phi)
            else:
                y = w
            acc.append(t3c_stats(y, sigma))
        out[tag] = np.concatenate(acc)
    return out

def frontier(far, lo, hi, iters=7, n0=1800, n1=400, seed0=503):
    for it in range(iters):
        mid = 0.5 * (lo + hi)
        b = run_block(mid, n0, n1, seed0 + 6151 * it)
        thr = np.quantile(b['h0'], 1 - far)
        pd = float(np.mean(b['h1'] > thr))
        print('    [T3c] gamma=%+.2f dB  thr=%.4g  Pd=%.3f' % (mid, thr, pd), flush=True)
        if pd >= 0.9:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

if __name__ == '__main__':
    t0 = time.time()
    rep = {}
    for far in (1e-2, 1e-3):
        g = frontier(far, -16.0, -22.0)
        rep['far%g' % far] = g
        print(' FAR=%.0e  T3c frontier=%+.2f dB' % (far, g), flush=True)
    import json
    rep['elapsed_s'] = time.time() - t0
    with open('t3c_probe_results.json', 'w') as f:
        json.dump(rep, f, indent=1, default=float)
    print('saved t3c_probe_results.json (%.1f s)' % rep['elapsed_s'])

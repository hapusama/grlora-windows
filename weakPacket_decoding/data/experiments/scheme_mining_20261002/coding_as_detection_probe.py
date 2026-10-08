# -*- coding: utf-8 -*-
"""coding-as-detection 数值核查探针（A1 视角方案挖掘，2026-10-02）。

问题："payload 的 FEC 冗余是最长的前导"——全帧边缘化 GLRT（③，无码约束
悲观版）能否比 8 符号相干前导（①）在同 FAR / 同 Pd 下赢 >=3dB
（闭合 4.5dB detection cascade 缺口的前提）？

三检测器（同一噪声实现喂所有检测器；SF8 M=256；30 符号 payload 值均匀；
κ 共格未知，16 点网格，κ*=0 在格上；γ = 每样本 SNR，λ0 = Mγ = 峰 bin
后 SNR）：
  T1  ①  8 符号已知前导相干 GLRT(φ)：max_κ |Σ_p ζ_p(κ)|²   （DeRa 式理想化）
  T2  ②  30 符号 genie 数据 + 共 φ 相干（上界）：|Σ_n ζ_n(v_n+κ*)|²
  T3  ③  无约束数据混合 LLR：max_κ Σ_n log[(1/M)Σ_v e^{-λ0} I0(2√λ0|ζ_n(v+κ)|)]
        ——检测器拿真 λ0（genie 加持：若 genie 版仍输，结论更强）

另附 T1 的 FAR 曲线（1e-1..1e-4 MC + 1e-6 尾外推）→ 两段式 FAR 再分配
的 dB 预算。

运行：py -3.12 coding_as_detection_probe.py
"""
import numpy as np
import json
import time

M, NS, NP, K = 256, 30, 8, 16
kap = np.arange(K) / K
mm = np.arange(M)
TW = np.exp(-2j * np.pi * np.outer(kap, mm) / M)      # (K, M)
LN2M = np.log(2) * M   # not used, kept for clarity
CHUNK = 64

def logi0(x):
    """stable log I0"""
    xa = np.abs(np.asarray(x, dtype=np.float64))
    out = np.empty_like(xa)
    big = xa > 30.0
    out[big] = xa[big] - 0.5 * np.log(2 * np.pi * xa[big])
    nb = ~big
    out[nb] = np.log(np.i0(xa[nb]))
    return out

def gen_noise(rng, n, s, sigma):
    return sigma * ((rng.standard_normal((n, s, M))
                     + 1j * rng.standard_normal((n, s, M))) / np.sqrt(2.0))

def signal(rng, n, s, nu, phi):
    return np.exp(2j * np.pi * nu[..., None] * mm[None, None, :] / M
                  + 1j * phi[:, None, None])

def light_stats(y, sigma, vn=None):
    """T1 (preamble coherent GLRT) and T2 (genie coherent, needs vn)."""
    sc = 1.0 / (sigma * np.sqrt(M))
    zp = np.einsum('psm,gm->psg', y[:, :NP, :], TW) * sc        # (n,NP,K)
    T1 = np.max(np.abs(zp.sum(1)) ** 2, axis=-1)
    if vn is None:
        return T1, None
    Ev = np.exp(-2j * np.pi * vn[..., None] * mm[None, None, :] / M)  # (n,NS,M)
    zn = np.einsum('nsm,nsm->ns', y[:, NP:, :], Ev) * sc
    T2 = np.abs(zn.sum(1)) ** 2
    return T1, T2

def t3_stats(y, sigma, lam0):
    """unconstrained-data mixture LLR over (v, kappa)."""
    sc = 1.0 / (sigma * np.sqrt(M))
    t = 2.0 * np.sqrt(lam0)
    n = y.shape[0]
    R = np.empty((n, NS, K, M), np.float32)
    for g in range(K):
        Z = np.fft.fft(y[:, NP:, :] * TW[g], axis=-1)
        R[:, :, g, :] = (np.abs(Z) * sc).astype(np.float32)
    s = logi0(t * R) - lam0
    mx = s.max(-1, keepdims=True)
    lse = mx[..., 0] + np.log(np.mean(np.exp(s - mx), axis=-1))   # (n,NS,K)
    return lse.sum(1).max(-1)

def run_block(gamma_db, n0, n1, seed, heavy=True):
    rng = np.random.default_rng(seed)
    sigma = 10.0 ** (-gamma_db / 20.0)
    lam0 = M * 10.0 ** (gamma_db / 10.0)
    out = {}
    for tag, n in (('h0', n0), ('h1', n1)):
        acc = {'T1': [], 'T2': [], 'T3': []}
        for i in range(0, n, CHUNK):
            c = min(CHUNK, n - i)
            w = gen_noise(rng, c, NP + NS, sigma)
            vn = rng.integers(0, M, (c, NS))
            if tag == 'h1':
                nu = np.concatenate([np.zeros((c, NP)), vn], axis=1)
                phi = rng.uniform(0, 2 * np.pi, c)
                y = w + signal(rng, c, NP + NS, nu, phi)
            else:
                y = w
            T1, T2 = light_stats(y, sigma, vn)
            acc['T1'].append(T1)
            acc['T2'].append(T2)
            if heavy:
                acc['T3'].append(t3_stats(y, sigma, lam0))
        o = {'T1': np.concatenate(acc['T1']),
             'T2': np.concatenate(acc['T2'])}
        if heavy:
            o['T3'] = np.concatenate(acc['T3'])
        out[tag] = o
    return out

def run_light_T1_only(gamma_db, n0, n1, seed):
    """cheap path for T1 FAR sweep (S=NP rows only)."""
    rng = np.random.default_rng(seed)
    sigma = 10.0 ** (-gamma_db / 20.0)
    res = {}
    for tag, n in (('h0', n0), ('h1', n1)):
        acc = []
        for i in range(0, n, 512):
            c = min(512, n - i)
            w = gen_noise(rng, c, NP, sigma)
            if tag == 'h1':
                nu = np.zeros((c, NP))
                phi = rng.uniform(0, 2 * np.pi, c)
                y = w + signal(rng, c, NP, nu, phi)
            else:
                y = w
            T1, _ = light_stats(np.concatenate([y, np.zeros((c, 1, M), np.complex128)], axis=1), sigma)
            acc.append(T1)
        res[tag] = np.concatenate(acc)
    return res

def pd_at(stats, thr):
    return float(np.mean(stats > thr))

def frontier(det, far, lo, hi, iters, n0, n1, seed0, heavy=True):
    """bisection on gamma for Pd>=0.9 at fixed FAR; T3 re-draws H0 per gamma."""
    for it in range(iters):
        mid = 0.5 * (lo + hi)
        b = run_block(mid, n0, n1, seed0 + 7919 * it, heavy=heavy)
        thr = np.quantile(b['h0'][det], 1 - far)
        pd = pd_at(b['h1'][det], thr)
        if pd >= 0.9:
            lo = mid
        else:
            hi = mid
        print('    [%s] gamma=%+.2f dB  thr=%.4g  Pd=%.3f' % (det, mid, thr, pd), flush=True)
    return 0.5 * (lo + hi)

def main():
    t0 = time.time()
    rep = {}
    print('=== Part 1: 三检测器 frontier（Pd=0.9）===')
    for far in (1e-2, 1e-3):
        g1 = frontier('T1', far, -20.0, -30.0, 8, 3000, 600, 101, heavy=False)
        g2 = frontier('T2', far, -26.0, -36.0, 8, 3000, 600, 202, heavy=False)
        g3 = frontier('T3', far, -10.0, -24.0, 7, 2000, 400, 303, heavy=True)
        rep['far%g' % far] = {'T1': g1, 'T2': g2, 'T3': g3,
                              'T2_minus_T1': g2 - g1, 'T3_minus_T1': g3 - g1}
        print(' FAR=%.0e  T1=%+.2f  T2=%+.2f  T3=%+.2f dB | T2-T1=%+.2f (th. 5.75)  T3-T1=%+.2f'
              % (far, g1, g2, g3, g2 - g1, g3 - g1), flush=True)

    print('=== Part 2: T1 FAR 曲线（两段式 FAR 再分配预算）===')
    b = run_light_T1_only(-30.0, 100000, 0, 777)
    t1h0 = b['h0']
    thrs = {f: float(np.quantile(t1h0, 1 - f))
            for f in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4)}
    # tail extrapolation to 1e-6: ln P = a + b*thr on (1e-3, 1e-4) pair
    import math
    q3, q4 = np.log(1e-3), np.log(1e-4)
    slope = (q4 - q3) / (math.log(thrs[1e-4]) - math.log(thrs[1e-3]))
    # use log-thr linear fit: lnP = a + b*ln(thr) (power-law tail, conservative)
    bln = (q4 - q3) / (math.log(thrs[1e-4]) - math.log(thrs[1e-3]))
    aln = q3 - bln * math.log(thrs[1e-3])
    thr6 = math.exp((math.log(1e-6) - aln) / bln)
    thrs['1e-6(extrap)'] = float(thr6)
    print(' T1 thresholds:', {k: round(v, 2) for k, v in thrs.items()}, flush=True)

    # frontier per FAR (cheap, T1 only)
    far_front = {}
    for f in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4):
        lo, hi = -18.0, -30.0
        for it in range(8):
            mid = 0.5 * (lo + hi)
            bb = run_light_T1_only(mid, 2000, 600, 888 + 31 * it)
            pd = pd_at(bb['h1']['T1'] if isinstance(bb['h1'], dict) else bb['h1'], thrs[f])
            if pd >= 0.9:
                lo = mid
            else:
                hi = mid
        far_front[f] = 0.5 * (lo + hi)
        print('  FAR=%.0e  T1 frontier=%+.2f dB' % (f, far_front[f]), flush=True)
    # extrap 1e-6 frontier via same sweep with fixed thr6
    lo, hi = -18.0, -34.0
    for it in range(8):
        mid = 0.5 * (lo + hi)
        bb = run_light_T1_only(mid, 2000, 600, 999 + 37 * it)
        pd = pd_at(bb['h1'], thr6)
        if pd >= 0.9:
            lo = mid
        else:
            hi = mid
    far_front['1e-6(extrap)'] = 0.5 * (lo + hi)
    print('  FAR=1e-6(extrap)  T1 frontier=%+.2f dB' % far_front['1e-6(extrap)'], flush=True)
    rep['T1_far_curve'] = {str(k): v for k, v in far_front.items()}
    rep['far_budget_1e-6_to_1e-2'] = far_front['1e-6(extrap)'] - far_front[1e-2]

    rep['elapsed_s'] = time.time() - t0
    with open('coding_as_detection_probe_results.json', 'w') as f:
        json.dump(rep, f, indent=1, default=float)
    print('saved coding_as_detection_probe_results.json  (%.1f s)' % rep['elapsed_s'])

    # analytic cross-prints
    print('theory: 10log10(30/8)=%.2f dB; 2lnM(SF8)=%.1f dB; 2ln4096(SF12)=%.1f dB'
          % (10 * np.log10(30.0 / 8.0), 10 * np.log10(2 * np.log(M)), 10 * np.log10(2 * np.log(4096))))

if __name__ == '__main__':
    main()

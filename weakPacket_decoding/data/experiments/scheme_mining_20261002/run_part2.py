# -*- coding: utf-8 -*-
"""Part 2 standalone：T1 FAR 曲线 + 两段式 FAR 再分配预算。
修复：跳过 n=0 的分支。Part 1 数字取自 2026-10-02 完成的 probe.log 运行。
"""
import numpy as np
import json
import math
import coding_as_detection_probe as P


def run_T1(gamma_db, n0, n1, seed):
    rng = np.random.default_rng(seed)
    sigma = 10.0 ** (-gamma_db / 20.0)
    res = {}
    for tag, n in (('h0', n0), ('h1', n1)):
        if n == 0:
            continue
        acc = []
        for i in range(0, n, 512):
            c = min(512, n - i)
            w = P.gen_noise(rng, c, P.NP, sigma)
            if tag == 'h1':
                nu = np.zeros((c, P.NP))
                phi = rng.uniform(0, 2 * np.pi, c)
                y = w + P.signal(rng, c, P.NP, nu, phi)
            else:
                y = w
            T1, _ = P.light_stats(np.concatenate(
                [y, np.zeros((c, 1, P.M), np.complex128)], axis=1), sigma)
            acc.append(T1)
        res[tag] = np.concatenate(acc)
    return res


def main():
    b = run_T1(-22.0, 100000, 0, 777)
    t1h0 = b['h0']
    thrs = {f: float(np.quantile(t1h0, 1 - f))
            for f in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4)}
    q3, q4 = np.log(1e-3), np.log(1e-4)
    bln = (q4 - q3) / (math.log(thrs[1e-4]) - math.log(thrs[1e-3]))
    aln = q3 - bln * math.log(thrs[1e-3])
    thr6 = math.exp((math.log(1e-6) - aln) / bln)
    thrs['1e-6(extrap)'] = float(thr6)
    print('T1 thresholds:', {k: round(v, 2) for k, v in thrs.items()}, flush=True)

    far_front = {}
    for f in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4):
        lo, hi = -18.0, -30.0
        for it in range(8):
            mid = 0.5 * (lo + hi)
            bb = run_T1(mid, 2000, 600, 888 + 31 * it)
            pd = float(np.mean(bb['h1'] > thrs[f]))
            if pd >= 0.9:
                lo = mid
            else:
                hi = mid
        far_front[f] = 0.5 * (lo + hi)
        print('  FAR=%.0e  T1 frontier=%+.2f dB' % (f, far_front[f]), flush=True)
    lo, hi = -18.0, -34.0
    for it in range(9):
        mid = 0.5 * (lo + hi)
        bb = run_T1(mid, 2000, 600, 999 + 37 * it)
        pd = float(np.mean(bb['h1'] > thr6))
        if pd >= 0.9:
            lo = mid
        else:
            hi = mid
    far_front['1e-6(extrap)'] = 0.5 * (lo + hi)
    print('  FAR=1e-6(extrap)  T1 frontier=%+.2f dB' % far_front['1e-6(extrap)'], flush=True)

    rep = {
        'part1_from_probe_log': {
            'far1e-2': {'T1': -23.18, 'T2': -29.73, 'T3': -17.49},
            'far1e-3': {'T1': -21.93, 'T2': -28.13, 'T3': -16.95},
            'T3c_from_t3c_probe': {'far1e-2': -18.23, 'far1e-3': -17.85},
        },
        'T1_thresholds': thrs,
        'T1_far_curve': {str(k): v for k, v in far_front.items()},
        'far_budget_1e-6_to_1e-2': far_front['1e-6(extrap)'] - far_front[1e-2],
    }
    with open('coding_as_detection_probe_results.json', 'w') as f:
        json.dump(rep, f, indent=1, default=float)
    print('far budget 1e-6 -> 1e-2: %+.2f dB'
          % rep['far_budget_1e-6_to_1e-2'])
    print('theory: 10log10(30/8)=%.2f dB; 2lnM(SF8)=%.1f; 2ln4096(SF12)=%.1f'
          % (10 * np.log10(30.0 / 8.0), 10 * np.log10(2 * np.log(P.M)),
             10 * np.log10(2 * np.log(4096))))


if __name__ == '__main__':
    main()

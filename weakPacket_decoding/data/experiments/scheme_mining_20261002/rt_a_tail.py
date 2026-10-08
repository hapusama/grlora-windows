# -*- coding: utf-8 -*-
"""红队攻击 (a)：T1 尾外推稳健性 —— 指数尾 vs 幂律尾，4.10dB 预算的置信区间。

复刻 Round1 coding_as_detection_probe 的 T1（8 符号前导相干 GLRT，
SF8 M=256，κ 16 点网格，统计量归一化后与 σ 无关）。
  1) 2e5 帧 H0 样本 → 1e-1..1e-4 分位数 + bootstrap CI
  2) 在 (1e-3,1e-4) 段拟合两种尾模型，外推 thr(1e-6)；逐段斜率表
  3) 两个 thr(1e-6) 模型分别跑 frontier(Pd=0.9)，得预算 1e-6→1e-2 的
     模型分歧与 CI；与 Round1 的 4.10 dB（幂律）对照。

运行：py -3.12 rt_a_tail.py
"""
import numpy as np
import json
import time

M, NP, K = 256, 8, 16
kap = np.arange(K) / K
mm = np.arange(M)
TW = np.exp(-2j * np.pi * np.outer(kap, mm) / M)


def t1_h0(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(0, n, 512):
        c = min(512, n - i)
        w = (rng.standard_normal((c, NP, M)) + 1j * rng.standard_normal((c, NP, M))) / np.sqrt(2)
        sc = 1.0 / np.sqrt(M)          # sigma=1
        zp = np.einsum('psm,gm->psg', w, TW) * sc
        out.append(np.max(np.abs(zp.sum(1)) ** 2, axis=-1))
    return np.concatenate(out)


def t1_h1(gamma_db, n, seed):
    rng = np.random.default_rng(seed)
    sigma = 10.0 ** (-gamma_db / 20.0)
    out = []
    for i in range(0, n, 512):
        c = min(512, n - i)
        w = sigma * (rng.standard_normal((c, NP, M)) + 1j * rng.standard_normal((c, NP, M))) / np.sqrt(2)
        phi = rng.uniform(0, 2 * np.pi, c)
        sig = np.exp(1j * phi[:, None, None]) * np.ones((c, NP, M))
        y = w + sig
        sc = 1.0 / (sigma * np.sqrt(M))
        zp = np.einsum('psm,gm->psg', y, TW) * sc
        out.append(np.max(np.abs(zp.sum(1)) ** 2, axis=-1))
    return np.concatenate(out)


def frontier_at_thr(thr, lo, hi, iters=8, n1=600, seed0=4242):
    for it in range(iters):
        mid = 0.5 * (lo + hi)
        pd = float(np.mean(t1_h1(mid, n1, seed0 + 7919 * it) > thr))
        if pd >= 0.9:
            hi = mid          # 已达标 -> frontier 在更深(更负)侧
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    t0 = time.time()
    print("== (a) T1 tail extrapolation: exponential vs power-law ==")
    x = t1_h0(200000, 777)
    fars = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4])
    q = np.quantile(x, 1 - fars)

    # 逐段 dB/decade 斜率（用 Round1 的 frontier 曲线做对照打印）
    r1_curve = {1e-1: -25.05, 3e-2: -24.02, 1e-2: -23.27, 3e-3: -22.29,
                1e-3: -21.82, 1e-4: -20.79}
    print("Round1 frontier per-decade slope (dB/dec):")
    fs = sorted(r1_curve)
    for a, b in zip(fs[:-1], fs[1:]):
        dec = np.log10(b / a)
        print("  %.0e->%.0e : %+.2f dB/dec" % (a, b, (r1_curve[b] - r1_curve[a]) / dec))

    # bootstrap thr CI（在 1e-3/1e-4 分位上）
    rng = np.random.default_rng(999)
    boots = {"q3": [], "q4": [], "thr6_exp": [], "thr6_pow": []}
    for _ in range(500):
        xb = rng.choice(x, size=len(x), replace=True)
        q3, q4 = np.quantile(xb, 1 - 1e-3), np.quantile(xb, 1 - 1e-4)
        boots["q3"].append(q3); boots["q4"].append(q4)
        d = q4 - q3
        boots["thr6_exp"].append(q4 + 2 * d)                       # lnP = a + b*thr
        r = q4 / q3
        boots["thr6_pow"].append(q4 * r * r)                       # lnP = a + b*ln thr
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    print("\nH0 quantiles (2e5 frames):")
    for f, qq in zip(fars, q):
        print("  FAR=%.0e  thr=%.2f" % (f, qq))
    thr6_exp = float(np.mean(boots["thr6_exp"]))
    print("  thr(1e-6) exp-fit  = %.2f  CI(95%%)=[%.1f, %.1f]"
          % (thr6_exp, *ci(boots["thr6_exp"])))
    thr6_pow = float(np.mean(boots["thr6_pow"]))
    print("  thr(1e-6) pow-fit  = %.2f  CI(95%%)=[%.1f, %.1f]  (Round1 用 140.5)"
          % (thr6_pow, *ci(boots["thr6_pow"])))

    # 理论对照：max of ~N_eff 个 Exp(mean NP) → 纯指数尾
    d_dec = q[5] - q[4]
    print("  measured d(thr)/decade on (1e-3,1e-4) = %.2f ; pure-exp prediction 8*ln10 = %.2f"
          % (d_dec, 8 * np.log(10)))

    # frontier 重算：1e-2（实测分位）、1e-4（实测）、1e-6 两种模型
    rep = {}
    thr02 = float(q[2]); thr04 = float(q[5])
    f02 = frontier_at_thr(thr02, -26, -20)
    f04 = frontier_at_thr(thr04, -24, -18)
    f6e = frontier_at_thr(thr6_exp, -23, -17)
    f6p = frontier_at_thr(thr6_pow, -22, -16)
    print("\nfrontiers (Pd=0.9): 1e-2=%+.2f  1e-4=%+.2f  1e-6(exp)=%+.2f  1e-6(pow)=%+.2f dB"
          % (f02, f04, f6e, f6p))
    bud_exp = f6e - f02
    bud_pow = f6p - f02
    print("budget 1e-6->1e-2 : exp-tail = %+.2f dB ; pow-tail = %+.2f dB ; Round1 claimed = +4.10 dB"
          % (bud_exp, bud_pow))
    print("model divergence = %.2f dB" % abs(bud_exp - bud_pow))
    rep.update(thr6_exp=thr6_exp, thr6_pow=thr6_pow, frontier_1e2=f02,
               frontier_1e4=f04, frontier_1e6_exp=f6e, frontier_1e6_pow=f6p,
               budget_exp=bud_exp, budget_pow=bud_pow,
               budget_1e4_to_1e2=f04 - f02,
               divergence=abs(bud_exp - bud_pow))
    with open("rt_a_tail_results.json", "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("saved rt_a_tail_results.json (%.1f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()

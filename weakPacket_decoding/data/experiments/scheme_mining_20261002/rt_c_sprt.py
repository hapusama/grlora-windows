# -*- coding: utf-8 -*-
"""红队攻击 (c)：假候选的早停 —— LLR 漂移 vs 码校验早停的算术。

A' 声称复杂度 = 1+W*FAR_front 个候选 × 早停 ≈ ×2-3。
两问：
 1) LLR/Λ 型 SPRT 早停：纯噪声候选的逐符号 mixture-LLR 漂移 μ0 是多少？
    若 |μ0| 极小，部分和是零漂移随机游走 → 30 符号内早停基本不触发。
    μ0 = E_0[ log (1/M) Σ_v e^{-λ0} I0(2√λ0 R_v) ], R_v=|ζ_v|, ζ~CN(0,1)。
 2) 码校验早停（逐 block 硬校验）：噪声块过全校验概率 q_blk = 2^{-sf_app*(n-k)}，
    E[符号消耗] = cw_len/(1-q_blk)。逐 CR 报。

运行：py -3.12 rt_c_sprt.py
"""
import numpy as np

rng = np.random.default_rng(20261002)
M = 256
NS = 30

def mu0_mc(lam0, n_sym=200000):
    """E_0[per-symbol mixture LLR] 与部分和穿越 -a 的时间分布（纯噪声）。"""
    def sym_llr(lam, batches):
        out = []
        t = 2.0 * np.sqrt(lam)
        for _ in range(batches):
            R = np.abs((rng.standard_normal((1000, M)) + 1j * rng.standard_normal((1000, M)))) / np.sqrt(2)
            x = t * R
            # log I0 stable
            li = np.where(x > 30, x - 0.5 * np.log(2 * np.pi * np.maximum(x, 30)),
                          None)
            small = x[x <= 30]
            l0 = np.zeros_like(x)
            l0[x > 30] = x[x > 30] - 0.5 * np.log(2 * np.pi * x[x > 30])
            if small.size:
                l0[x <= 30] = np.log(np.i0(small))
            s = l0 - lam
            mx = s.max(1, keepdims=True)
            out.append((mx[:, 0] + np.log(np.mean(np.exp(s - mx), axis=1))))
        return np.concatenate(out)
    return sym_llr(lam0, n_sym // 1000)

print("== (c) 假候选早停：LLR 漂移 vs 码校验 ==")
for gamma_db in (-23.27, -21.0, -19.0):
    lam0 = M * 10 ** (gamma_db / 10.0)
    s = mu0_mc(lam0)
    mu, sd = s.mean(), s.std()
    # 30 符号部分和最劣值分布 → 若 -a 取 |sum30| 的 5% 分位，H0 早停率仅 5%
    sums = np.add.reduceat(s[:1000 * (len(s) // 1000)].reshape(-1, 30), np.arange(0, len(s) - 29, 30), axis=1) \
        if False else np.cumsum(s[:30 * (len(s) // 30)].reshape(-1, 30), axis=1)
    mn = sums.min(1)
    print(" gamma=%+.2f dB (λ0=%.2f): μ0=%+.5f nats/sym, σ=%.4f "
          "| P(min partial-sum < -1)=%.3f, < -3)=%.3f, < -10)=%.4f"
          % (gamma_db, lam0, mu, sd,
             float(np.mean(mn < -1)), float(np.mean(mn < -3)), float(np.mean(mn < -10))))
    a_wald = 1.0 / max(abs(mu), 1e-12)
    print("   → SPRT 平均停留 (Wald a/|μ0|, a=1 nat) ≈ %.0f 符号（帧长 30）" % a_wald)

print("\n码校验早停算术（硬序列校验，逐 interleave block）：")
print(" %4s %5s | q_blk(过全校验) | E[块] | E[符号] | ×N_dec(10 假候选, 30 符号帧)")
for cr_name, n, k, sf_app, cw_len in (("CR4/5", 5, 4, 8, 5), ("CR4/6", 6, 4, 8, 6),
                                       ("CR4/7", 7, 4, 8, 7), ("CR4/8", 8, 4, 8, 8),
                                       ("LDRO CR4/8 SF11", 8, 4, 9, 8)):
    q = 2.0 ** (-sf_app * (n - k))
    eb = 1.0 / (1.0 - q)
    es = eb * cw_len
    mult = 1 + 10 * es / 30.0
    print(" %12s | %.2e | %.3f | %.2f | ×%.2f" % (cr_name, q, eb, es, mult))

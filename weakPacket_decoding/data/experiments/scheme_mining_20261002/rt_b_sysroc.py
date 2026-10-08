# -*- coding: utf-8 -*-
"""红队攻击 (b)：A' 两段式获取的记账口径、top-K 枚举损失、payload 门控。

三问：
 1) 记账：A' 的 "FAR_front=1e-2 + CRC16 → 系统 FAR 1e-6" 在 W~10^3 搜索窗下
    是否自洽？逐口径算允许的 FAR_front 与预算。
 2) top-K：门限放松到 1e-2（每 cell），真 cell 排名进不了 top-K 的损失
    （K=10/50/100），Pd_sys@0.9 的净 dB。
 3) payload 门控：系统 Pd = P(检出)×P(解码+CRC 过)。SF8/CR4/5/30 符号
    硬链 (1-SER)^30=0.9 的门限 γ_gate —— A' 的 stage-1 frontier 深过 γ_gate
    的部分全部被吃掉。

运行：py -3.12 rt_b_sysroc.py
"""
import numpy as np
import json
import rt_a_tail as A

M, NP, K = A.M, A.NP, A.K
W = 1000

# ---- 指数尾门限函数（由 rt_a 的 H0 分位标定：thr(f) 线性于 ln f）----
q1, q2 = 65.62, 84.84            # 1e-3, 1e-4
b_exp = (q2 - q1) / np.log(10.0)  # thr per decade
def thr_exp(far):
    return q2 + b_exp * np.log(1e-4 / far)

def pd_curve(thr, gammas, n=3000, seed=31337):
    return np.array([float(np.mean(A.t1_h1(g, n, seed + 11 * int((g + 30) * 10)) > thr))
                     for g in gammas])

def frontier_thr(thr, lo, hi, iters=6, n=1500, seed=555):
    for it in range(iters):
        mid = 0.5 * (lo + hi)
        pd = float(np.mean(A.t1_h1(mid, n, seed + 7919 * it) > thr))
        if pd >= 0.9:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

print("== (b1) 记账口径 ==")
crc16 = 2.0 ** -16
f_front_window = 1e-6 / (W * crc16)          # 窗口 FAR 1e-6 -> 每 cell
f_base_window = 1e-6 / W                      # 无 CRC 基线每 cell
print(" 窗口记账: 基线 per-cell FAR=%.1e (thr=%.1f) ; A' 允许 per-cell FAR=%.2e (thr=%.1f)"
      % (f_base_window, thr_exp(f_base_window), f_front_window, thr_exp(f_front_window)))
print(" per-cell 记账 (A' 原始): 基线 1e-6 (thr=%.1f) ; A' 1e-2 (thr=%.1f, 实测)"
      % (thr_exp(1e-6), 46.04))
print(" A' 在窗口记账下若仍用 FAR_front=1e-2: 系统 FAR = %.1e (超标 %d 倍)"
      % (W * 1e-2 * crc16, int(round(W * 1e-2 * crc16 / 1e-6))))

f_win_base = frontier_thr(thr_exp(f_base_window), -24, -14)
f_win_a = frontier_thr(thr_exp(f_front_window), -24, -14)
f_cell_base = frontier_thr(thr_exp(1e-6), -24, -14)
f_cell_1e2 = frontier_thr(46.04, -24, -14)
print(" frontier: 窗口记账 基线=%+.2f A'=%+.2f -> 增益 %+.2f dB | per-cell 记账 基线(1e-6)=%+.2f A'(1e-2)=%+.2f -> %+.2f dB"
      % (f_win_base, f_win_a, f_win_base - f_win_a, f_cell_base, f_cell_1e2, f_cell_base - f_cell_1e2))

print("\n== (b2) top-K 枚举损失 (thr=1e-2 实测 46.04, 该配置系统 FAR 超标, 诊断用) ==")
rng = np.random.default_rng(2026)
noise_pool = A.t1_h0(60000, 31)               # H0 cell 统计量池
res = {}
for Kc in (10, 50, 100):
    for gamma_db in (-23.1, -22.5, -22.0, -21.5):
        t = A.t1_h1(gamma_db, 2000, 71 + Kc)
        idx = rng.integers(0, len(noise_pool), size=(2000, W - 1))
        cells = noise_pool[idx]
        n_above = np.sum(cells > t[:, None], axis=1)   # 排在真 cell 之前的噪声数
        pd_sys = float(np.mean((n_above < Kc) & (t > 46.04)))
        pd_pure = float(np.mean(t > 46.04))
        res[(Kc, gamma_db)] = (pd_pure, pd_sys)
        print(" K=%3d gamma=%+.2f: 纯前导 Pd=%.3f -> Pd_sys=%.3f (损失 %.3f)"
              % (Kc, gamma_db, pd_pure, pd_sys, pd_pure - pd_sys))

print("\n== (b3) payload 门控 (SF8 CR4/5 30 符号, 硬链, κ=0 最优情形) ==")
def ser_noncoh(gamma_db, n=400000):
    lam = M * 10 ** (gamma_db / 10.0)
    r = np.sqrt(lam) + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    mx = np.zeros(n)
    for _ in range(255):
        mx = np.maximum(mx, np.abs((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)))
    return float(np.mean(mx > np.abs(r)))

gate = None
prev_g, prev_p = None, None
for gamma_db in np.arange(-21, -9.01, 0.25):
    s = ser_noncoh(gamma_db, 60000)
    pframe = (1 - s) ** 30
    if prev_p is not None and prev_p < 0.9 <= pframe:
        gate = prev_g + 0.25 * (0.9 - prev_p) / max(pframe - prev_p, 1e-12)
        break
    prev_g, prev_p = gamma_db, pframe
if gate is not None:
    print("  γ_gate((1-SER)^30=0.9) ≈ %+.2f dB" % gate)
    print("  A' 系统 frontier = max(解码门, stage-1 frontier)（dB, 浅者绑定）:")
    print("    窗口记账: max(%+.2f, %+.2f) = %+.2f  vs 纯前导基线 %+.2f -> 净 %+.2f dB"
          % (gate, f_win_a, max(gate, f_win_a), f_win_base, f_win_base - max(gate, f_win_a)))
    print("    per-cell 记账: max(%+.2f, %+.2f) = %+.2f  vs 基线 %+.2f -> 净 %+.2f dB"
          % (gate, f_cell_1e2, max(gate, f_cell_1e2), f_cell_base, f_cell_base - max(gate, f_cell_1e2)))
with open("rt_b_sysroc_results.json", "w") as f:
    json.dump(dict(f_win_base=f_win_base, f_win_a=f_win_a,
                   f_cell_base=f_cell_base, f_cell_1e2=f_cell_1e2,
                   gate=gate, topk={str(k): v for k, v in res.items()}),
              f, indent=1, default=float)
print("saved rt_b_sysroc_results.json")

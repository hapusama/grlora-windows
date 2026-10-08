# -*- coding: utf-8 -*-
"""rt3b 任务二补：分位数级 frontier 分解（Pd=0.9 工作在 Q10，不在均值）。

均值版（rt3b_h0lemma.py）T3c frontier 复现 −17.99（rt_d −17.98）但 cod 侧
ε=0.01dB——因 mean-ρ≈0.97 高估了错候选在 Q10 处的胜率。frontier 条件是
P(√T>√thr)=0.9 ⇔ Q10(√T)=√thr。本脚本：
  1) 4 个 γ 点 × n=400 重跑 H1，取 Q10(√T3c), Q10(√T3cod), Q10(√S)；
  2) Pd 插值 frontier（对照 rt_d 实测 −17.98/−18.20/0.21dB）；
  3) 闭式：√thr_i − δ_i(γ) = qS(γ)，δ_i = Q10(√T_i)−Q10(√S)（Q10 级
     错候选抬升，预计 ≈0）⇒ ε = qS⁻¹(√thr_full) − qS⁻¹(√thr_cod) −修正。
H0 sqrt 门限沿用 rt3b_h0lemma 实测：T3c 65.69 / T3cod 61.44 @FAR 1e-2。
运行：py -3.12 rt3b_h0lemma_q10.py
"""
import numpy as np
import json
import time
import rt3b_h0lemma as L

THR_SQRT = dict(full=65.69, cod=61.44)     # @FAR 1e-2 (rt3b_h0lemma.log)

def main():
    t0 = time.time()
    gammas = [-17.7, -17.95, -18.2, -18.45]
    res = {}
    for g in gammas:
        T1, Tc, Td, S = L.run_h1(g, 400, 4242 + int(round((g + 20) * 100)))
        rc, rd, rs = np.sqrt(Tc), np.sqrt(Td), np.sqrt(S)
        res[g] = dict(
            q10c=float(np.quantile(rc, 0.10)), q10d=float(np.quantile(rd, 0.10)),
            q10s=float(np.quantile(rs, 0.10)),
            pd_c=float(np.mean(rc > THR_SQRT['full'])),
            pd_d=float(np.mean(rd > THR_SQRT['cod'])),
            dc=float(np.quantile(rc, 0.10) - np.quantile(rs, 0.10)),
            dd=float(np.quantile(rd, 0.10) - np.quantile(rs, 0.10)))
        print("γ=%+.2f: Q10 √T3c=%.2f √T3cod=%.2f √S=%.2f | δ_full=%+.2f "
              "δ_cod=%+.2f | Pd_full=%.3f Pd_cod=%.3f"
              % (g, res[g]['q10c'], res[g]['q10d'], res[g]['q10s'],
                 res[g]['dc'], res[g]['dd'], res[g]['pd_c'], res[g]['pd_d']),
              flush=True)

    def cross(g1, g2, p1, p2, target=0.9):
        return g1 + (target - p1) * (g2 - g1) / (p2 - p1)

    # Pd 插值 frontier
    gs = gammas
    f_full = f_cod = np.nan
    for i in range(len(gs) - 1):
        if (res[gs[i]]['pd_c'] - 0.9) * (res[gs[i + 1]]['pd_c'] - 0.9) <= 0:
            f_full = cross(gs[i], gs[i + 1], res[gs[i]]['pd_c'],
                           res[gs[i + 1]]['pd_c'])
        if (res[gs[i]]['pd_d'] - 0.9) * (res[gs[i + 1]]['pd_d'] - 0.9) <= 0:
            f_cod = cross(gs[i], gs[i + 1], res[gs[i]]['pd_d'],
                          res[gs[i + 1]]['pd_d'])
    print("\nPd 插值 frontier: T3c=%+.2f (rt_d -17.98)  T3cod=%+.2f (rt_d -18.20)"
          "  -> ε=%+.2f dB (实测 0.21)" % (f_full, f_cod, f_cod - f_full))

    # Q10 分解: qS(γ) 曲线 + δ 修正; frontier 条件 qS(γ)+δ_i(γ)=√thr_i
    qS = np.array([res[g]['q10s'] for g in gs])
    d_f = np.array([res[g]['dc'] for g in gs])
    d_c = np.array([res[g]['dd'] for g in gs])
    def solve_i(thr, dvec):
        for i in range(len(gs) - 1):
            f1 = qS[i] + dvec[i] - thr
            f2 = qS[i + 1] + dvec[i + 1] - thr
            if f1 * f2 <= 0 and f1 != f2:
                return gs[i] + (0 - f1) * (gs[i + 1] - gs[i]) / (f2 - f1)
        return np.nan
    ff = solve_i(THR_SQRT['full'], d_f)
    fc = solve_i(THR_SQRT['cod'], d_c)
    print("Q10 分解 frontier: T3c=%+.2f  T3cod=%+.2f -> ε=%+.2f dB"
          % (ff, fc, fc - ff))
    print("δ(Q10 级错候选抬升): full %.2f→%.2f  cod %.2f→%.2f (γ 深处)"
          % (d_f[0], d_f[-1], d_c[0], d_c[-1]))
    # 纯门限位移上界(δ=0): qS 曲线在 thr 两点间的 γ 差
    def qS_solve(thr):
        for i in range(len(gs) - 1):
            f1, f2 = qS[i] - thr, qS[i + 1] - thr
            if f1 * f2 <= 0 and f1 != f2:
                return gs[i] + (0 - f1) * (gs[i + 1] - gs[i]) / (f2 - f1)
        return np.nan
    print("纯 Δ_thr 位移(δ=0 上界): %+.2f dB" % (qS_solve(THR_SQRT['cod'])
                                                 - qS_solve(THR_SQRT['full'])))
    json.dump({str(k): v for k, v in res.items()} |
              dict(f_full=f_full, f_cod=f_cod, eps=f_cod - f_full,
                   ff=ff, fc=fc, eps_q10=fc - ff,
                   thr_sqrt=THR_SQRT),
              open('rt3b_h0lemma_q10_results.json', 'w'), indent=1)
    print("saved rt3b_h0lemma_q10_results.json (%.1f s)" % (time.time() - t0))

if __name__ == '__main__':
    main()

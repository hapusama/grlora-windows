# -*- coding: utf-8 -*-
"""M2 门限验证：解析族（上穿 × N_eff 乘积）vs 合成域 MC（<5% 击杀开关）。

输出 m2_thr.json：各 P 的解析门限/因子分解、MC 分位、偏差（dB 与 %）、
经验上穿 c 拟合 vs c_tot。
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2_core as M
import d1_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
FARS = (1e-2, 1e-3)
N_MC = {8: 24000, 16: 24000, 32: 8000}


def fit_upcross_c(samples, far_lo=0.02):
    """从 MC 上尾拟合上穿常数：P(max≤γ)=exp(−c√γ e^{−γ})。
    对 y=−ln(F̂(γ)) 取 ln(y/√γ) 的截距（尾段最小二乘）。"""
    qs = np.linspace(1 - far_lo, 1 - 1.5 / len(samples), 40)
    g = np.quantile(samples, qs)
    Fhat = qs
    y = -np.log(Fhat)
    lhs = np.log(y / np.sqrt(g))
    # ln c = mean(lhs)（γ 足够大时 e^{-γ} 主导 ⇒ y ≈ c√γ e^{−γ}）
    return float(np.exp(np.mean(lhs)))


def main():
    t0 = time.time()
    res = {}
    for pre in (8, 16, 32):
        row = {}
        for far in FARS:
            g, fac = M.thr_analytic(pre, far)
            row["closed_%g" % far] = dict(gamma=float(g),
                                          db=10 * np.log10(g), fac=fac)
        n = N_MC[pre]
        t1 = time.time()
        s = M.field_mc(pre, n, seed=20261004)
        mc_t = time.time() - t1
        entry = dict(n_mc=n, mc_sec=mc_t,
                     mc=dict(q50=float(np.median(s)),
                             mean_db=10 * float(np.log10(s.mean()))))
        for far in FARS:
            emp = float(np.quantile(s, 1 - far))
            clo = row["closed_%g" % far]["gamma"]
            entry["FAR=%g" % far] = dict(
                mc_gamma=emp, mc_db=10 * np.log10(emp),
                closed_db=10 * np.log10(clo),
                dev_db=10 * np.log10(emp / clo),
                dev_pct=100.0 * (emp / clo - 1.0),
                far_at_closed=float(np.mean(s > clo)))
        # 1D 参照（仅 κ 维，cert 同款）与经验 c
        L = M._confirm_layout(pre)
        cup = C.upcrossing_c(L["idx"], L["b"], M.N_FINE)
        entry["c_kappa_only"] = float(cup)
        entry["c_emp_tailfit"] = fit_upcross_c(s)
        entry["c_tot_analytic"] = row["closed_%g" % FARS[0]]["fac"]["c_total"]
        res["P=%d" % pre] = entry
        row["mc_summary"] = entry
        res["P=%d_closed" % pre] = {k: v for k, v in row.items()
                                    if k.startswith("closed")}
        print("P=%d done (%.0fs mc): dev@1e-2=%+.3fdB dev@1e-3=%+.3fdB "
              "c_emp=%.1f c_tot=%.1f" % (
                  pre, mc_t,
                  entry["FAR=0.01"]["dev_db"],
                  entry["FAR=0.001"]["dev_db"],
                  entry["c_emp_tailfit"], entry["c_tot_analytic"]),
              flush=True)
    json.dump(res, open(os.path.join(HERE, "m2_thr.json"), "w"), indent=1,
              default=str)
    print("→ m2_thr.json (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()

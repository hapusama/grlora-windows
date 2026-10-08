# -*- coding: utf-8 -*-
"""M2b Tier3 修正（dep3c）：乘积组合闭式的跨维上穿修正。

dep2/M2 的 Tier3 = c_κ(1D 上穿) × N_eff(δ) × N_eff(SFD) × N_eff(锚) ×
N_eff(conj)（独立维乘积）——合成域 MC 实测低估复合门限 14.6~22.6%
（跨维上穿分量缺失，与 d2_cfar2d 的 2D +0.84dB 同因）。

dep3c 修正：把 (κ,δ) 二维格的跨维上穿显式化——**蛇形链 CL 复合**：
  P(max ≤ u) ≈ exp( − c_serp·√u·e^{−u} − M_blk·e^{−u} )
  c_serp = (2/π^{3/2})·Σ_chain √(max(0, −d²r_k))：蛇形链（δ 行内 κ 扫，
  行间跳）上 |ρ|² 序列的三点二阶差的局部上穿率求和（1D Cramér–
  Lindgren 的非平稳链推广——d1 已验 1.12%@1e-3 的同族公式）；
  M_blk = 锚×SFD×conj 维的等效独立块数 − 1（块间近独立 ⇒ ρ_0 项
  M·e^{−u}，EC 展开的 0 维项）。
门限：解 c_serp√u e^{−u} + M_blk e^{−u} = FAR（牛顿）。
验证：vs field_mc3（合成域精确 MC，协方差 Dirichlet 核 + 同款格网搜索）。
→ m2b_thr2d.json
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_core as C
import m2_core as M
import m2b_core as W

HERE = os.path.dirname(os.path.abspath(__file__))
FARS = (1e-2, 1e-3)


def cell_rho(pre, dq, dd):
    """(κ,δ) 格单元间复相关：ρ((q,δ),(q+dq,δ+dd))（同 (锚,SFD,conj) 块内，
    j 对角近似——跨行位置差 ≫ Dirichlet 宽 ⇒ j≠j' 项为 0）。返回复数。"""
    L = W.confirm_layout(pre)
    idx, b, c_e = L["idx"], L["b"], L["c_e"]
    sgn = np.where(np.isin(np.arange(len(idx)), L["sfd_sel"]), -1.0, 1.0)
    dp = sgn * (idx - c_e) * dd                       # 各行位置差（bin）
    ker = M.rho_dirichlet(dp)
    ph = np.exp(-2j * np.pi * dq * idx / W.N_FINE)
    return complex(np.sum(b * ker * ph) / L["K_c"])


def c_serp(pre, mode="dep3a"):
    """蛇形链上穿常数：δ 行内 κ 0..255 正扫/反扫交替，行间跳接。

    逐链点的局部曲率（CL 非平稳链推广）：点 k 的三点集
      r_l=|ρ(k−1,k)|², r_r=|ρ(k,k+1)|², r_2=|ρ(k−1,k+1)|²
      d²_k = 1 − r_l − r_r + r_2 （r(0)=1 的二阶差分近似）
      c = (2/π^{3/2})·Σ_k √(max(0, −d²_k))"""
    nq, nd = W.N_FINE, len(W.DGRID)
    coords = []
    for k in range(nd):
        qs = range(nq) if k % 2 == 0 else range(nq - 1, -1, -1)
        for q in qs:
            coords.append((q, k))

    def rho_c(m1, m2):
        (q1, k1), (q2, k2) = coords[m1], coords[m2]
        dq = ((q2 - q1 + nq // 2) % nq) - nq // 2      # 环形 κ 差
        return abs(cell_rho(pre, dq, k2 - k1)) ** 2

    n = len(coords)
    csum = 0.0
    for k in range(n):
        r_l = rho_c(k - 1, k) if k >= 1 else 0.0       # 链首：左侧独立
        r_r = rho_c(k, k + 1) if k <= n - 2 else 0.0
        r_2 = rho_c(k - 1, k + 1) if 1 <= k <= n - 2 else 0.0
        d2 = 1.0 - r_l - r_r + r_2
        if d2 < 0:
            csum += np.sqrt(-d2)
    return float(2.0 / np.pi ** 1.5 * csum), None


def thr_2d(far, c_s, m_blk):
    """解 c_s·√u·e^{−u} + m_blk·e^{−u} = FAR（牛顿，u 从 1D 闭式起）。"""
    u = C.cfar_threshold(far, max(c_s, 1e-9))

    def f(u):
        e = np.exp(-u)
        return c_s * np.sqrt(u) * e + m_blk * e - far

    def fp(u):
        e = np.exp(-u)
        return e * (c_s * (0.5 / np.sqrt(u) - np.sqrt(u)) - m_blk)
    for _ in range(50):
        step = f(u) / fp(u) if fp(u) != 0 else 0
        u = max(u - step, 1e-6)
    return float(u)


def m_blocks(pre, mode="dep3a"):
    """等效独立块数（锚×SFD×conj 维，乘积 N_eff，块间近独立）。"""
    g, fac = W.thr_analytic_dep3a(pre, FARS[0])       # 借因子分解
    return float(fac["neff_anchor"] * fac["neff_sfd"] * fac["neff_conj"])


def main():
    t0 = time.time()
    res = {}
    n_mc = 24000
    for pre in (8, 16, 32):
        row = dict()
        cs, _ = c_serp(pre)
        m_eff = m_blocks(pre)
        row["c_serp_perblock"] = cs
        row["m_eff_blocks"] = m_eff
        g_prod, fac = W.thr_analytic_dep3a(pre, FARS[0])
        row["fac_prod"] = fac
        # 复合：M_eff 个等效块各贡献蛇形链上穿 + (M_eff−1) 个块界 ρ_0 项
        row["c_serp_total"] = m_eff * cs
        for mode in ("dep3a", "dep3b"):
            t1 = time.time()
            sc = W.field_mc3(pre, n_mc, mode=mode)
            row[mode] = dict(mc_sec=time.time() - t1,
                             q50=float(np.quantile(sc, .5)))
            for far in FARS:
                mc_g = float(np.quantile(sc, 1 - far))
                if mode == "dep3a":
                    clo_prod = 10 * np.log10(
                        C.cfar_threshold(far, fac["c_total"]))
                    u2 = thr_2d(far, m_eff * cs, m_eff - 1.0)
                    row[mode]["FAR=%g" % far] = dict(
                        mc_db=10 * np.log10(mc_g),
                        closed_prod_db=clo_prod,
                        closed_2d_db=10 * np.log10(u2),
                        dev_prod_db=10 * np.log10(mc_g) - clo_prod,
                        dev_prod_pct=100 * (mc_g / C.cfar_threshold(
                            far, fac["c_total"]) - 1),
                        dev_2d_db=10 * np.log10(mc_g) - 10 * np.log10(u2),
                        dev_2d_pct=100 * (mc_g / u2 - 1),
                        far_at_2d=float(np.mean(sc > u2)))
                else:
                    u2 = thr_2d(far, m_eff * cs, m_eff + 1.0)  # 参照级
                    row[mode]["FAR=%g" % far] = dict(
                        mc_db=10 * np.log10(mc_g),
                        ref_2d_db=10 * np.log10(u2),
                        far_at_ref=float(np.mean(sc > u2)))
        res["P=%d" % pre] = row
        json.dump(res, open(os.path.join(HERE, "m2b_thr2d.json"), "w"),
                  indent=1)
        print("P=%d done (%.0fs)" % (pre, time.time() - t0), flush=True)
    print("%.0fs → m2b_thr2d.json" % (time.time() - t0))


def _unused():
    return json.dumps({k: {m: v.get(m) for m in ("c_serp_perblock",
                                                 "m_eff_blocks")}
                       for k, v in {}.items()})


if __name__ == "__main__":
    main()

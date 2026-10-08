# -*- coding: utf-8 -*-
"""E1 探针3：滑动定律基扩展——SFO 残余交叉项 i·c/N 的解释力 + 与 εν 的一致性检验。

物理推导：窗口相对真符号边界漂移 D(i)=δ·i（抽取域样本），θ += 2π·ν_i·D(i)/N
≈ 2π·δ·i·c_i/N。预测：基列 [i·c/N] 系数 ≈ 2π·εν（ν̂ 滑动律独立测得）。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_common import ROOT, wrap

IN = os.path.join(ROOT, "e1_native.jsonl")


def gn_fit(th, B, iters=8):
    x = np.zeros(B.shape[1])
    for _ in range(iters):
        r = wrap(th - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x += dx
    return x, wrap(th - B @ x)


def build(syms, nb, terms):
    th = np.array([s["theta"] for s in syms])
    c = np.array([s["c"] for s in syms], dtype=float) / nb
    kinds = np.array([s["kind"] for s in syms])
    i = np.arange(len(syms), dtype=float)
    cols = {"1": np.ones_like(i), "i": i, "i2": i ** 2,
            "c": c, "ic": i * c, "c2": c ** 2, "i2c": i ** 2 * c,
            "syn": (kinds == "sync").astype(float),
            "hdr": (kinds == "hdr").astype(float),
            "pay": (kinds == "pay").astype(float)}
    B = np.stack([cols[t] for t in terms], 1)
    return th, B


def main():
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    variants = {
        "old(M2c+k)": ["1", "i", "i2", "c", "syn", "hdr", "pay"],
        "+ic": ["1", "i", "i2", "c", "ic", "syn", "hdr", "pay"],
        "+ic+c2": ["1", "i", "i2", "c", "ic", "c2", "syn", "hdr", "pay"],
        "+ic+i2c": ["1", "i", "i2", "c", "ic", "i2c", "syn", "hdr", "pay"],
        "full": ["1", "i", "i2", "c", "ic", "c2", "i2c", "syn", "hdr", "pay"],
    }
    print("== 基扩展扫描（全帧 σ_res rad, n=148）:")
    store = {}
    for lab, terms in variants.items():
        sigs, coefs = [], []
        for fr in frames:
            nb = 1 << fr["sf"]
            th, B = build(fr["syms"], nb, terms)
            x, r = gn_fit(th, B)
            sigs.append(np.std(r))
            coefs.append(x)
        store[lab] = np.array(coefs)
        sigs = np.array(sigs)
        print("   %-10s σ med=%.4f p25=%.4f p75=%.4f p10=%.4f" % (
            lab, np.median(sigs), np.percentile(sigs, 25),
            np.percentile(sigs, 75), np.percentile(sigs, 10)))

    # ic 系数 vs 2π·εν 一致性（εν 用 nu 滑动独立估计：对 (i, ν̂−c) 线性回归）
    print("\n== 一致性检验: coef[ic] vs 2π·εν (environ slip rate from ν̂):")
    ratio, cic, enus = [], [], []
    for fr, xr in zip(frames, store["+ic"]):
        nb = 1 << fr["sf"]
        syms = fr["syms"]
        ii = np.arange(len(syms), dtype=float)
        cc = np.array([s["c"] for s in syms], dtype=float)
        nu = np.array([s["kappa"] + 0 for s in syms])   # κ̂ 相对 c（抛物线）
        dnu = np.array([((s["b"] + s["kappa"] - s["c"] + nb / 2) % nb - nb / 2)
                        for s in syms])
        # mod-1 圆统计滑率（= track_tone 口径）
        ph = np.exp(2j * np.pi * dnu / 1.0)  # dnu 已是 bin 单位 → exp(2πi dnu) 周期 1
        g = np.linspace(-0.03, 0.03, 241)
        sc = np.abs(ph @ np.exp(-2j * np.pi * np.outer(ii, g)))
        enu = g[int(np.argmax(sc))]
        c_ic = xr[variants["+ic"].index("ic")]
        cic.append(c_ic); enus.append(enu)
        if abs(enu) > 1e-4:
            ratio.append(c_ic / (2 * np.pi * enu))
    cic, enus = np.array(cic), np.array(enu and enus)
    print("   coef[ic] med=%+.4f p25=%+.4f p75=%+.4f" % (
        np.median(cic), np.percentile(cic, 25), np.percentile(cic, 75)))
    print("   εν(独立) med=%+.5f bin/符  → 2πεν med=%+.4f" % (
        np.median(enus), 2 * np.pi * np.median(enus)))
    if ratio:
        print("   coef[ic]/(2πεν) med=%.2f p25=%.2f p75=%.2f (n=%d, |εν|>1e-4)"
              % (np.median(ratio), np.percentile(ratio, 25),
                 np.percentile(ratio, 75), len(ratio)))

    # +ic 后分 kind σ 与同 c 碰撞对
    same, diff = [], []
    per_kind = {}
    for fr in frames:
        nb = 1 << fr["sf"]
        th, B = build(fr["syms"], nb, variants["+ic"])
        x, r = gn_fit(th, B)
        syms = fr["syms"]
        kinds = np.array([s["kind"] for s in syms])
        cs = np.array([s["c"] for s in syms])
        for kk in ("pre", "sync", "hdr", "pay"):
            per_kind.setdefault(kk, []).extend(r[kinds == kk])
        m = (kinds == "pay") | (kinds == "hdr")
        rr, ccr = r[m], cs[m]
        n = len(rr)
        for a in range(n):
            for b in range(a + 1, min(a + 40, n)):
                d = abs(wrap(rr[a] - rr[b]))
                (same if ccr[a] == ccr[b] else diff).append(d)
    print("\n== +ic 后: 同c med=%.3f (n=%d) vs 异c med=%.3f" % (
        np.median(same), len(same), np.median(diff)))
    for kk, v in per_kind.items():
        v = np.array(v)
        print("   %-5s σ=%.4f" % (kk, v.std()))


if __name__ == "__main__":
    main()

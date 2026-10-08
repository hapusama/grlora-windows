# -*- coding: utf-8 -*-
"""E1 探针5：chirp 相位二次律 π·v²/N 检验。

推导：u_v[n]=u_0[n]·e^{j(π/N)(v²+2vn)}（相位连续 FMCW 卷绕约定）→ 解斜音
起始相位含 (π/N)(v²+v)（+ CFO/STO 的线性 v 项）。v² mod 2N → 对未知值
伪随机、对假设值确定。基列 [πc²/N]（注意不是 c²/N²），期望系数 ≈ ±1。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_common import ROOT, wrap

IN = os.path.join(ROOT, "e1_native.jsonl")


def gn_fit(th, B, iters=10):
    x = np.zeros(B.shape[1])
    for _ in range(iters):
        r = wrap(th - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x += dx
    return x, wrap(th - B @ x)


def main():
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    variants = {
        "old":      ["1", "i", "c"],
        "+c2_over_N":  ["1", "i", "c", "q"],
        "+c2+kinds":   ["1", "i", "c", "q", "syn", "hdr", "pay"],
        "quad_only":   ["1", "i", "q"],
    }
    print("== 二次律检验（σ_res rad, n=148 帧）:")
    store = {}
    for lab, terms in variants.items():
        sigs, coefs, r2s = [], [], []
        for fr in frames:
            nb = 1 << fr["sf"]
            syms = fr["syms"]
            th = np.array([s["theta"] for s in syms])
            c = np.array([s["c"] for s in syms], dtype=float)
            kinds = np.array([s["kind"] for s in syms])
            i = np.arange(len(syms), dtype=float)
            cols = {"1": np.ones_like(i), "i": i, "i2": i ** 2,
                    "c": c / nb, "q": np.pi * c ** 2 / nb,
                    "syn": (kinds == "sync").astype(float),
                    "hdr": (kinds == "hdr").astype(float),
                    "pay": (kinds == "pay").astype(float)}
            B = np.stack([cols[t] for t in terms], 1)
            x, r = gn_fit(th, B)
            sigs.append(np.std(r)); coefs.append(x); r2s.append(r)
        store[lab] = np.array(coefs)
        sigs = np.array(sigs)
        print("   %-12s σ med=%.4f p25=%.4f p75=%.4f p10=%.4f" % (
            lab, np.median(sigs), np.percentile(sigs, 25),
            np.percentile(sigs, 75), np.percentile(sigs, 10)))

    # 二次项系数分布（+c2+kinds 变体, q 列索引）
    terms = variants["+c2+kinds"]
    iq_ = terms.index("q")
    q = store["+c2+kinds"][:, iq_]
    print("\n   coef[q=πc²/N]: med=%+.4f p25=%+.4f p75=%+.4f  （期望 ±1）"
          % (np.median(q), np.percentile(q, 25), np.percentile(q, 75)))
    ic_ = terms.index("c")
    cc = store["+c2+kinds"][:, ic_]
    print("   coef[c/N]: med=%+.4f p25=%+.4f p75=%+.4f" % (
        np.median(cc), np.percentile(cc, 25), np.percentile(cc, 75)))

    # 分 kind 残差（+c2+kinds）
    from collections import defaultdict
    per_kind = defaultdict(list)
    lags = []
    for fr in frames:
        nb = 1 << fr["sf"]
        syms = fr["syms"]
        th = np.array([s["theta"] for s in syms])
        c = np.array([s["c"] for s in syms], dtype=float)
        kinds = np.array([s["kind"] for s in syms])
        i = np.arange(len(syms), dtype=float)
        cols = {"1": np.ones_like(i), "i": i, "c": c / nb,
                "q": np.pi * c ** 2 / nb,
                "syn": (kinds == "sync").astype(float),
                "hdr": (kinds == "hdr").astype(float),
                "pay": (kinds == "pay").astype(float)}
        B = np.stack([cols[t] for t in terms], 1)
        x, r = gn_fit(th, B)
        for kk in ("pre", "sync", "hdr", "pay"):
            per_kind[kk].extend(r[kinds == kk])
        rp = r[kinds == "pay"]
        if len(rp) > 8:
            lags.append(np.corrcoef(rp[:-1], rp[1:])[0, 1])
    print("\n== +c2+kinds 后分 kind σ:")
    for kk in ("pre", "sync", "hdr", "pay"):
        print("   %-5s σ=%.4f (n=%d)" % (kk, np.std(per_kind[kk]), len(per_kind[kk])))
    print("   pay 残差 lag-1: med=%+.3f p25=%+.3f p75=%+.3f" % (
        np.median(lags), np.percentile(lags, 25), np.percentile(lags, 75)))

    # 同 c 碰撞对（应进一步塌缩）
    same, diff = [], []
    for fr in frames:
        nb = 1 << fr["sf"]
        syms = fr["syms"]
        th = np.array([s["theta"] for s in syms])
        c = np.array([s["c"] for s in syms], dtype=float)
        kinds = np.array([s["kind"] for s in syms])
        i = np.arange(len(syms), dtype=float)
        cols = {"1": np.ones_like(i), "i": i, "c": c / nb,
                "q": np.pi * c ** 2 / nb,
                "syn": (kinds == "sync").astype(float),
                "hdr": (kinds == "hdr").astype(float),
                "pay": (kinds == "pay").astype(float)}
        B = np.stack([cols[t] for t in terms], 1)
        x, r = gn_fit(th, B)
        m = (kinds == "pay") | (kinds == "hdr")
        rr, ccr = r[m], c[m].astype(int)
        for a in range(len(rr)):
            for b in range(a + 1, min(a + 40, len(rr))):
                d = abs(wrap(rr[a] - rr[b]))
                (same if ccr[a] == ccr[b] else diff).append(d)
    print("   同c |Δr| med=%.3f (n=%d) vs 异c med=%.3f" % (
        np.median(same), len(same), np.median(diff)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""E1 探针6：经验 φ̂(c) 剖面提取 + SF10 三捕获(同 payload 不同 P)跨捕获判别。

  - 半拆交叉验证：capture 内帧分两半，A 半学 φ̂(c)，B 半评测塌缩量（真平稳性）
  - 跨捕获迁移：14_8 学的 φ̂(c) 用到 14_16 / 14_32（同 payload，位置整体错开）
  - 剖面形状：对 c 的圆均值随 c 的结构（画文本相关量：与 c 的相关、相邻 c 相关）
"""
import sys
import os
import json
import numpy as np
from collections import defaultdict

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


def frame_resid(fr, theta_key="theta"):
    nb = 1 << fr["sf"]
    syms = fr["syms"]
    th = np.array([s[theta_key] for s in syms])
    c = np.array([s["c"] for s in syms], dtype=float)
    kinds = np.array([s["kind"] for s in syms])
    i = np.arange(len(syms), dtype=float)
    B = np.stack([np.ones_like(i), i, i ** 2, c / nb,
                  (kinds == "sync").astype(float),
                  (kinds == "hdr").astype(float),
                  (kinds == "pay").astype(float)], 1)
    x, r = gn_fit(th, B)
    # 再去帧圆均值
    r = wrap(r - np.angle(np.mean(np.exp(1j * r))))
    return r, (c % nb).astype(int), kinds, i


def learn_phi(frames_list, min_n=3):
    ph = defaultdict(list)
    for fr in frames_list:
        r, c, kinds, i = frame_resid(fr)
        for rr, cc, kk in zip(r, c, kinds):
            if kk in ("pay", "hdr"):
                ph[cc].append(rr)
    return {cc: np.angle(np.mean(np.exp(1j * np.array(v))))
            for cc, v in ph.items() if len(v) >= min_n}, \
        {cc: len(v) for cc, v in ph.items() if len(v) >= min_n}


def eval_phi(frames_list, ph):
    errs, cold = [], []
    for fr in frames_list:
        r, c, kinds, i = frame_resid(fr)
        m = np.isin(kinds, ("pay", "hdr")) & np.isin(c, list(ph.keys()))
        if m.sum() < 8:
            continue
        cal = wrap(r[m] - np.array([ph[int(cc)] for cc in c[m]]))
        errs.append(np.std(cal))
        cold.append(np.std(r[m]))
    return np.array(errs), np.array(cold)


def profile_stats(ph, nb):
    cs = np.array(sorted(ph.keys()), dtype=float)
    vs = np.array([ph[int(cc)] for cc in cs])
    # 相邻 c（间距 ≤8）的相位相关
    pairs = [(vs[j], vs[j + 1]) for j in range(len(cs) - 1) if cs[j + 1] - cs[j] <= 8]
    if len(pairs) > 8:
        P = np.array(pairs)
        adj_rho = np.corrcoef(P[:, 0], P[:, 1])[0, 1]
    else:
        adj_rho = float("nan")
    rho_c = np.corrcoef(cs, vs)[0, 1] if len(cs) > 8 else float("nan")
    return dict(n=len(cs), std=float(vs.std()), adj_rho=float(adj_rho), rho_c=float(rho_c))


def main():
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    by_cap = defaultdict(list)
    for fr in frames:
        by_cap[fr["cap"]].append(fr)

    print("== 半拆交叉验证（capture 内平稳性）:")
    for cap in ("0_0_0_10_14_8", "0_0_0_10_14_16", "0_0_0_10_14_32",
                "1_0_8_11_2_16", "1_0_12_11_2_16", "1_1_0_11_2_16"):
        fl = by_cap.get(cap, [])
        if len(fl) < 4:
            continue
        half = len(fl) // 2
        phA, _ = learn_phi(fl[:half])
        phB, _ = learn_phi(fl[half:])
        eB, cB = eval_phi(fl[half:], phA)
        eA, cA = eval_phi(fl[:half], phB)
        print("   %-16s nF=%2d  σcold=%.3f → σcal(A→B)=%.3f / (B→A)=%.3f  (n_phi=%d)"
              % (cap, len(fl), np.median(cB), np.median(eB), np.median(eA), len(phA)))

    print("\n== SF10 跨捕获迁移（同 payload,不同 P,位置错开）:")
    c8, c16, c32 = (by_cap.get("0_0_0_10_14_%d" % p, []) for p in (8, 16, 32))
    ph8, _ = learn_phi(c8)
    ph16, _ = learn_phi(c16)
    ph32, _ = learn_phi(c32)
    e16, cc16 = eval_phi(c16, ph8)
    e32, cc32 = eval_phi(c32, ph8)
    e8, cc8 = eval_phi(c8, ph16)
    print("   σ(14_16 cold)=%.3f → 用14_8的φ̂: %.3f (n_phi=%d, 帧数=%d)"
          % (np.median(cc16), np.median(e16), len(ph8), len(c16)))
    print("   σ(14_32 cold)=%.3f → 用14_8的φ̂: %.3f" % (np.median(cc32), np.median(e32)))
    print("   σ(14_8  cold)=%.3f → 用14_16的φ̂: %.3f" % (np.median(cc8), np.median(e8)))

    # 剖面形状
    print("\n== φ̂(c) 剖面形状:")
    for cap, ph in (("14_8", ph8), ("14_16", ph16), ("14_32", ph32)):
        s = profile_stats(ph, 1024)
        print("   %-6s n_c=%3d std=%.3f rad  相邻c相关=%+.3f  与c相关=%+.3f"
              % (cap, s["n"], s["std"], s["adj_rho"], s["rho_c"]))
    # SF11 一个 capture
    ph11, _ = learn_phi(by_cap["1_0_8_11_2_16"])
    s = profile_stats(ph11, 2048)
    print("   %-6s n_c=%3d std=%.3f rad  相邻c相关=%+.3f  与c相关=%+.3f"
          % ("1_0_8", s["n"], s["std"], s["adj_rho"], s["rho_c"]))

    # 相同 payload 验证：三 SF10 捕获 gt 是否一致
    if c8 and c16:
        print("\n== gt payload 一致性: 14_8 vs 14_16 = %s, vs 14_32 = %s"
              % (c8[0]["gt"] == c16[0]["gt"], c8[0]["gt"] == c32[0]["gt"]))

    # φ̂(c) 剖面与 sin/cos(πc/N·k) 的相关扫描（周期结构检验）
    cs = np.array(sorted(ph8.keys()), dtype=float)
    vs = np.array([ph8[int(cc)] for cc in cs])
    best = []
    for k in range(1, 65):
        x = 2 * np.pi * k * cs / 1024.0
        r2 = max(abs(np.corrcoef(vs, np.sin(x))[0, 1]),
                 abs(np.corrcoef(vs, np.cos(x))[0, 1]))
        best.append((r2, k))
    best.sort(reverse=True)
    print("   φ̂ 与 sin/cos(2πkc/N) 最大相关: %.3f @k=%d, 次大 %.3f @k=%d"
          % (best[0][0], best[0][1], best[1][0], best[1][1]))


if __name__ == "__main__":
    main()

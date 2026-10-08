# -*- coding: utf-8 -*-
"""E1 探针2：φ_H(c) 判别——帧内同 c 碰撞 / LOO 校准底 / 体制分裂 / 白性。"""
import sys
import os
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_common import ROOT, wrap

IN = os.path.join(ROOT, "e1_native.jsonl")


def fit_resid(syms, use_kinds=("pre", "sync", "hdr", "pay")):
    """M2c+kind 偏移主项 → 残差 / c / kind / 帧内序。"""
    sel = [s for s in syms if s["kind"] in use_kinds]
    th = np.array([s["theta"] for s in sel])
    c = np.array([s["c"] for s in sel], dtype=float)
    kinds = np.array([s["kind"] for s in sel])
    i = np.arange(len(sel), dtype=float)
    cols = [np.ones_like(i), i, i ** 2, c / 1024.0]
    for kk in ("sync", "hdr", "pay"):
        cols.append((kinds == kk).astype(float))
    B = np.stack(cols, 1)
    x = np.zeros(B.shape[1])
    for _ in range(6):
        r = wrap(th - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x += dx
    return wrap(th - B @ x), c, kinds, i


def main():
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]

    # T1: 帧内同 c 碰撞（pay+hdr）：同 c 对 vs 异 c 对 的残差差分布
    same, diff = [], []
    for fr in frames:
        r, c, kinds, i = fit_resid(fr["syms"])
        m = (kinds == "pay") | (kinds == "hdr")
        r, c = r[m], c[m]
        n = len(r)
        for a in range(n):
            for b in range(a + 1, n):
                d = abs(wrap(r[a] - r[b]))
                (same if c[a] == c[b] else diff).append(d)
    same, diff = np.array(same), np.array(diff)
    print("== T1. 帧内配对 |Δr| (rad): 同c n=%d med=%.3f p75=%.3f | 异c n=%d med=%.3f p75=%.3f"
          % (len(same), np.median(same), np.percentile(same, 75),
             len(diff), np.median(diff), np.percentile(diff, 75)))

    # T2: LOO per-capture φ_H 校准底（pay+hdr 残差，用其他帧学 φ_H）
    by_cap = defaultdict(list)
    for fr in frames:
        r, c, kinds, i = fit_resid(fr["syms"])
        for rr, cc, kk in zip(r, c, kinds):
            by_cap[fr["cap"]].append((int(cc) % (1 << fr["sf"]), rr, kk))
    for sel_k in (("pay", "hdr"), ("pay",)):
        loo_sig, cold_sig = [], []
        for fr in frames:
            if fr["psym"] < 30:
                continue
            r, c, kinds, i = fit_resid(fr["syms"])
            m = np.isin(kinds, sel_k)
            rj, cj = r[m], (c[m] % (1 << fr["sf"])).astype(int)
            other = defaultdict(list)
            for cc, rr, kk in by_cap[fr["cap"]]:
                if kk in sel_k:
                    other[int(cc)].append(rr)
            ph = {cc: np.angle(np.mean(np.exp(1j * np.array(v))))
                  for cc, v in other.items() if len(v) >= 3}
            cal = [rr - ph.get(int(cc), 0.0) for rr, cc in zip(rj, cj) if int(cc) in ph]
            if len(cal) >= 8:
                loo_sig.append(np.std(wrap(np.array(cal))))
                cold_sig.append(np.std(rj))
        print("== T2. LOO φ_H 校准（kinds=%s, n=%d 帧）: σ_cal med=%.3f p25=%.3f vs σ_cold med=%.3f"
              % (sel_k, len(loo_sig), np.median(loo_sig), np.percentile(loo_sig, 25),
                 np.median(cold_sig)))

    # T3: pre 独立 (ψ,ω) vs 全局 ω 的体制分裂
    dgap = []
    for fr in frames:
        syms = fr["syms"]
        th = np.array([s["theta"] for s in syms])
        npre = sum(1 for s in syms if s["kind"] == "pre")
        if npre < 6:
            continue
        ip = np.arange(npre, dtype=float)
        Bp = np.stack([np.ones_like(ip), ip], 1)
        xp = np.zeros(2)
        for _ in range(6):
            rp = wrap(th[:npre] - Bp @ xp)
            dxp, *_ = np.linalg.lstsq(Bp, rp, rcond=None)
            xp += dxp
        r, c, kinds, i = fit_resid(syms)
        # 全局 ω 在 hdr/pay 段的等效斜率（i 上取中点差分）
        dgap.append((xp[1], fr.get("cap")))
    w = np.array([d[0] for d in dgap])
    print("== T3. 前导独立斜率 ω_pre: med=%+.4f rad/符 (p25=%+.4f p75=%+.4f)"
          % (np.median(w), np.percentile(w, 25), np.percentile(w, 75)))

    # T4: LOO 校准后逐符号增量的白性（pay 段 lag-1）
    lags = []
    for fr in frames:
        if fr["psym"] < 30:
            continue
        r, c, kinds, i = fit_resid(fr["syms"])
        m = kinds == "pay"
        rp = r[m]
        if len(rp) > 8:
            lags.append(np.corrcoef(rp[:-1], rp[1:])[0, 1])
    print("== T4. pay 段 M2c+kind 残差 lag-1: med=%+.3f p25=%+.3f p75=%+.3f (n=%d)"
          % (np.median(lags), np.percentile(lags, 25), np.percentile(lags, 75), len(lags)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""M3 战表汇总（2026-10-04）：PER/有效包率/10%PER 门限/配对 w/l/失效分解。"""
import json
import os
from collections import defaultdict

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP, "m3_battle_checkpoint.jsonl")

CHAINS = [("u", "OURS-E2E"), ("cu", "OURS-cert"), ("dera", "DeRa"),
          ("a", "  A列"), ("b", "  B列"), ("ca", "cert-A"), ("cb", "cert-B")]
LEVELS = [-18, -20, -22, -24, -26]


def per_key(c):
    return c + "_per"


def load():
    rows = []
    for line in open(CKPT, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def thr10(curve):
    """curve: {level: per}。线性插值 10% PER 门限（负 dB，越大越浅）。"""
    lv = sorted([l for l in curve if curve[l] is not None], reverse=True)
    if not lv:
        return None
    for l in lv:
        if curve[l] >= 0.10:
            shallow = l
        else:
            break
    # 找最深 <0.1 与最浅 ≥0.1 的相邻对
    deep_side = [l for l in lv if curve[l] < 0.10]
    shal_side = [l for l in lv if curve[l] >= 0.10]
    if not shal_side:
        return "<%d（最浅档已超）" % max(lv)
    if not deep_side:
        return ">%d（最深档未到）" % min(lv)
    l1 = max(deep_side)
    l2 = min([l for l in shal_side if l < l1], default=None)
    if l2 is None:
        return ">%d" % min(lv)
    p1, p2 = curve[l1], curve[l2]
    if p2 == p1:
        return "%.1f" % l2
    return "%.1f" % (l1 + (0.10 - p1) / (p2 - p1) * (l2 - l1))


def main():
    rows = load()
    print("载入 %d 单元" % len(rows))
    # ---- native ----
    nat = [r for r in rows if r["level"] is None]
    if nat:
        print("\n== native（n=%d/δ）==" % (len(nat) // 3))
        for di, tag in ((0, "0"), (1, "0.02"), (2, "0.082")):
            rr = [r for r in nat if r["di"] == di]
            if not rr:
                continue
            s = " δ=%-5s" % tag
            for c, name in CHAINS:
                k = per_key(c)
                if any(k in r for r in rr):
                    v = np.mean([r.get(k, 1) for r in rr])
                    s += " %s=%.3f" % (name, v)
            print(s)
    # ---- 战表 ----
    res = {}
    print("\n== PER 战表（n/档）==")
    for di, tag in ((0, "0"), (1, "0.02"), (2, "0.082")):
        for lv in LEVELS + [None]:
            rr = [r for r in rows if r["di"] == di and r["level"] == lv
                  and r["seed"] is not None] if lv is not None else \
                 [r for r in rows if r["di"] == di and r["level"] is None]
            if not rr:
                continue
            key = "native" if lv is None else lv
            s = "[δ=%-5s %6s n=%3d]" % (tag, key, len(rr))
            for c, name in CHAINS:
                k = per_key(c)
                vals = [r[k] for r in rr if k in r]
                if vals:
                    s += " %s=%.3f" % (name, np.mean(vals))
            if lv is not None:
                res[("per", di, key)] = {c: float(np.mean(
                    [r[per_key(c)] for r in rr if per_key(c) in r]))
                    for c, _ in CHAINS if any(per_key(c) in r for r in rr)}
                res[("n", di, key)] = len(rr)
            print(s)
    # ---- 10% PER 门限 ----
    print("\n== 10%% PER 门限（dB）==")
    thr = {}
    for di, tag in ((0, "0"), (1, "0.02"), (2, "0.082")):
        for c, name in CHAINS:
            curve = {}
            ok = True
            for lv in LEVELS:
                k = per_key(c)
                vals = [r[k] for r in rows
                        if r["di"] == di and r["level"] == lv and k in r]
                if not vals:
                    ok = False
                    break
                curve[lv] = float(np.mean(vals))
            if not ok:
                continue
            t = thr10(curve)
            thr[(di, c)] = t
            print(" δ=%-5s %-10s %s" % (tag, name, t))
    # ---- 配对 w/l ----
    print("\n== 配对 w/l（OURS-E2E vs DeRa，同噪单元）==")
    for di, tag in ((0, "0"), (1, "0.02"), (2, "0.082")):
        for lv in LEVELS:
            rr = [r for r in rows if r["di"] == di and r["level"] == lv
                  and "u_per" in r and "dera_per" in r]
            if not rr:
                continue
            w = sum(1 for r in rr if r["u_per"] == 0 and r["dera_per"] == 1)
            l = sum(1 for r in rr if r["u_per"] == 1 and r["dera_per"] == 0)
            both = sum(1 for r in rr if r["u_per"] == r["dera_per"])
            print(" δ=%-5s %+4d: %dW / %dL / %d平 (n=%d)" % (tag, lv, w, l,
                                                             both, len(rr)))
    # ---- 失效分解 ----
    print("\n== 失效分解（检出失败单元的成因）==")
    for di, tag in ((0, "0"), (1, "0.02"), (2, "0.082")):
        rr = [r for r in rows if r["di"] == di and r["level"] is not None]
        fu = defaultdict(int)
        fd = defaultdict(int)
        for r in rr:
            if r.get("u_per", 0) == 1:
                fu[r.get("fail_u", "?")] += 1
            if r.get("dera_per", 0) == 1:
                fd[r.get("fail_d", "?")] += 1
        nu = sum(1 for r in rr if "u_per" in r)
        nd = sum(1 for r in rr if "dera_per" in r)
        print(" δ=%-5s OURS(n=%d): %s | DeRa(n=%d): %s"
              % (tag, nu, dict(fu), nd, dict(fd)))
    # ---- 检出率/锚质量 ----
    print("\n== 检出率与锚质量（OURs dep2）==")
    for di, tag in ((0, "0"), (1, "0.02"), (2, "0.082")):
        for lv in LEVELS:
            rr = [r for r in rows if r["di"] == di and r["level"] == lv]
            if not rr:
                continue
            det = np.mean([r.get("dep_det", False) for r in rr])
            dera = np.mean([r.get("dera_det", False) for r in rr])
            cert = np.mean([r.get("cert_det", False) for r in rr])
            ae = [r["anchor_err"] for r in rr if "anchor_err" in r]
            dh = [abs(r["dhat"] - {"0": 0, "0.02": 0.02,
                                   "0.082": 0.082}[tag])
                  for r in rr if "dhat" in r]
            s = " δ=%-5s %+4d: dep2=%.2f cert=%.2f dera=%.2f" % (tag, lv,
                                                                 det, cert,
                                                                 dera)
            if ae:
                s += " |anchor_err| p50=%.2f p90=%.2f" % (
                    np.percentile(ae, 50), np.percentile(ae, 90))
            if dh:
                s += " |dδ̂| p50=%.3f" % np.percentile(dh, 50)
            print(s)
    out = dict(thr={"%s|%s" % k: v for k, v in thr.items()},
               per={"%s|%s|%s" % k: v for k, v in res.items()
                    if k[0] == "per"})
    json.dump(out, open(os.path.join(EXP, "m3_sum_results.json"), "w"),
              indent=1)
    print("\nsaved -> m3_sum_results.json")


if __name__ == "__main__":
    main()

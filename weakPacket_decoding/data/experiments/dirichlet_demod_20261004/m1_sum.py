# -*- coding: utf-8 -*-
"""m1_sum.py — checkpoint 去重 + 终版汇总（全帧表/κ 桶/10% 门限/判定）。"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "m1_checkpoint.jsonl")
CHAINS = ["TREL-5", "DERA", "OURS-Dir", "DERA-DT"]
LEVELS = list(range(-20, -27, -1))

rows = {}
for line in open(CKPT, encoding="utf-8"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    rows[(r["level"], r["seed"], r["frame"])] = r     # 后写覆盖（去重）
units = list(rows.values())
print("唯一单元 %d（期望 3948）" % len(units))

agg = defaultdict(lambda: [0, 0, 0])
lv_n = defaultdict(int)
bk = defaultdict(lambda: [0, 0])
native = defaultdict(lambda: [0, 0, 0])
for r in units:
    if r["level"] is None:
        for c in CHAINS:
            a = native[c]
            a[0] += r["chains"][c]["sym_err"]
            a[1] += r["sym_tot"]
            a[2] += r["chains"][c]["crc_fail"]
        continue
    k = "%+d" % r["level"]
    lv_n[k] += 1
    for c in CHAINS:
        a = agg[(k, c)]
        a[0] += r["chains"][c]["sym_err"]
        a[1] += r["sym_tot"]
        a[2] += r["chains"][c]["crc_fail"]
    if r.get("bucket"):
        for c in CHAINS:
            b = bk[(r["bucket"], c)]
            b[0] += r["chains"][c]["sym_err"]
            b[1] += r["sym_tot"]

print("\n==== 战表（SER/PER，n=包数/档）====")
print("[native n=%d] %s" % (
    sum(1 for r in units if r["level"] is None),
    " | ".join("%s %.4f/%.4f" % (c, native[c][0] / max(native[c][1], 1),
                                 native[c][2] / 28) for c in CHAINS)))
for lv in LEVELS:
    k = "%+d" % lv
    print("[%s n=%d] %s" % (k, lv_n[k], " | ".join(
        "%s %.4f/%.4f" % (c, agg[(k, c)][0] / max(agg[(k, c)][1], 1),
                          agg[(k, c)][2] / max(lv_n[k], 1)) for c in CHAINS)))


def thr10(c):
    xs, ys = [], []
    for lv in LEVELS:
        a = agg.get(("%+d" % lv, c))
        if a and a[1] > 0:
            xs.append(float(lv))
            ys.append(a[0] / a[1])
    for i in range(len(xs) - 1):
        if ys[i] < 0.10 <= ys[i + 1]:
            return xs[i] + (xs[i + 1] - xs[i]) * (0.10 - ys[i]) / max(
                ys[i + 1] - ys[i], 1e-9)
        if ys[i] >= 0.10 > ys[i + 1]:
            return xs[i]
    return None


print("\n10%% SER 门限：")
t = {}
for c in CHAINS:
    t[c] = thr10(c)
    print("  %-8s %s" % (c, t[c]))
d32 = None if (t["OURS-Dir"] is None or t["DERA"] is None) else t["OURS-Dir"] - t["DERA"]
d42 = None if (t["DERA-DT"] is None or t["DERA"] is None) else t["DERA-DT"] - t["DERA"]
d41 = None if (t["DERA-DT"] is None or t["TREL-5"] is None) else t["DERA-DT"] - t["TREL-5"]
print("预注册判定量：③−② = %s dB；④−② = %s dB；④−① = %s dB" % (d32, d42, d41))

print("\nκ 分桶 SER（−20..−26 合并）：")
for b in ("k<0.2", "0.2-0.3", "0.3-0.5"):
    print("  %-7s %s" % (b, {c: round(bk[(b, c)][0] / max(bk[(b, c)][1], 1), 4)
                             for c in CHAINS}))

rep = {"n_units": len(units), "thr10": t, "d_ours_minus_dera": d32,
       "d_dt_minus_dera": d42, "d_dt_minus_trel": d41,
       "native": {c: native[c] for c in CHAINS},
       "table": {"%s|%s" % k: v for k, v in agg.items()},
       "buckets": {"%s|%s" % k: v for k, v in bk.items()}}
with open(os.path.join(HERE, "m1_battle_results.json"), "w",
          encoding="utf-8") as f:
    json.dump(rep, f, indent=1, default=float)
print("\nsaved m1_battle_results.json")

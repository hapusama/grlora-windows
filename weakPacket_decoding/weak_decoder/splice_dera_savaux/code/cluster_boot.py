# -*- coding: utf-8 -*-
"""capture 聚类 bootstrap CI（红队二轮：伪重复的正确统计）。

数据：v4-b（seeds 0-2）+ v5b（seeds 3-9）合并，档 {−20,−22,−24}，
每档 n=280 单元（28 帧 × 10 种子）。聚类 = capture（帧 0-9=cap8，
10-20=cap16，21-27=cap32）。cluster bootstrap：有放回抽 3 个 capture，
按合并单元池计算 PER 与 ΔPER（SAVT2−DERA，配对同单元），
B=20000 次重采 → 95% 百分位 CI。附每 capture 分解表。
"""
import json
import os
import numpy as np

LVLS = (-20, -22, -24)
CAP_OF = [0] * 10 + [1] * 11 + [2] * 7      # 28 帧的 capture 归属
B = 20000

rec = {}
for path, seedset in (
    (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments"
     r"\dera_savt2_20261005\checkpoint.jsonl", set(range(3))),
    (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments"
     r"\dera_savt2_stats_20261007\checkpoint.jsonl", set(range(3, 10))),
):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["level"] in LVLS and r["seed"] in seedset:
            rec[(r["level"], r["seed"], r["frame"])] = (
                r["chains"]["DERA×SAVT2"]["crc_fail"],
                r["chains"]["DERA×DERA"]["crc_fail"])

rng = np.random.default_rng(20261008)
print("== 每 capture 分解（SAVT2/DERA 失败数，n 单元）==")
for lv in LVLS:
    for cap in range(3):
        ks = [k for k in rec if k[0] == lv and CAP_OF[k[2]] == cap]
        s2 = sum(rec[k][0] for k in ks)
        de = sum(rec[k][1] for k in ks)
        print("  [%+d] cap%d: n=%3d  SAVT2=%3d  DERA=%3d"
              % (lv, cap, len(ks), s2, de))

print("\n== cluster bootstrap（B=%d，抽 capture，95%% CI）==" % B)
for lv in LVLS:
    ks_all = [k for k in rec if k[0] == lv]
    by_cap = [[k for k in ks_all if CAP_OF[k[2]] == c] for c in range(3)]
    d_boots = []
    s2_boots = []
    de_boots = []
    for _ in range(B):
        picks = [by_cap[i] for i in rng.integers(0, 3, 3)]
        units = [u for cl in picks for u in cl]
        s2 = np.mean([rec[u][0] for u in units])
        de = np.mean([rec[u][1] for u in units])
        d_boots.append(s2 - de)
        s2_boots.append(s2)
        de_boots.append(de)
    d_boots = np.array(d_boots)
    s2_boots = np.array(s2_boots)
    de_boots = np.array(de_boots)
    n = len(ks_all)
    s2 = np.mean([rec[u][0] for u in ks_all])
    de = np.mean([rec[u][1] for u in ks_all])
    lo_d, hi_d = np.percentile(d_boots, [2.5, 97.5])
    lo_s, hi_s = np.percentile(s2_boots, [2.5, 97.5])
    lo_e, hi_e = np.percentile(de_boots, [2.5, 97.5])
    p_neg = float(np.mean(d_boots >= 0))
    print("[%+d] n=%d" % (lv, n))
    print("   SAVT2 PER %.3f [%.3f,%.3f] | DERA %.3f [%.3f,%.3f]"
          % (s2, lo_s, hi_s, de, lo_e, hi_e))
    print("   ΔPER(SAVT2−DERA) = %+.3f [%.3f, %+.3f]  P(Δ≥0)=%.4f"
          % (s2 - de, lo_d, hi_d, p_neg))

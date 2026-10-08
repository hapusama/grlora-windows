# -*- coding: utf-8 -*-
"""SF11 capture 聚类 bootstrap（14 capture 聚类重采，B=20000，纯数据处理）。"""
import json
import numpy as np

MAP_JSON = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
            r"\experiments\sf11_splice_20261008\frame_capture_map.json")
CKPT = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
        r"\experiments\sf11_splice_20261008\checkpoint.jsonl")

with open(MAP_JSON, encoding="utf-8") as fh:
    meta = json.load(fh)
cap_of = meta["cap_of"]
n_cap = len(meta["caps"])

rec = {}
with open(CKPT, encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if r["level"] is None:
            continue
        rec[(r["level"], r["seed"], r["frame"])] = (
            r["chains"]["DERA×SAVT2"]["crc_fail"],
            r["chains"]["DERA×DERA"]["crc_fail"])

rng = np.random.default_rng(20261009)
print("capture 聚类 bootstrap（B=20000，n_cluster=%d）\n" % n_cap)
for lv in (-17, -20, -22, -24, -26):
    ks = [k for k in rec if k[0] == lv]
    groups = [[] for _ in range(n_cap)]
    for k in ks:
        groups[cap_of[k[2]]].append(k)
    d_b = []
    for _ in range(20000):
        pick = [groups[i] for i in rng.integers(0, n_cap, n_cap)]
        units = [u for g in pick for u in g]
        if not units:
            continue
        s2 = np.mean([rec[u][0] for u in units])
        de = np.mean([rec[u][1] for u in units])
        d_b.append(s2 - de)
    d_b = np.array(d_b)
    s2 = np.mean([rec[k][0] for k in ks])
    de = np.mean([rec[k][1] for k in ks])
    lo, hi = np.percentile(d_b, [2.5, 97.5])
    per_c = [np.mean([rec[u][0] for u in g]) - np.mean([rec[u][1] for u in g])
             for g in groups if g]
    n_fav = sum(1 for x in per_c if x < 0)
    print("[%+d] n=%d SAVT2 %.3f vs DERA %.3f | Δ=%+.3f [%.3f,%+.3f] "
          "P(Δ≥0)=%.4f | capture 方向 %d/%d 利 SAVT2"
          % (lv, len(ks), s2, de, s2 - de, lo, hi,
             float(np.mean(d_b >= 0)), n_fav, len(per_c)))

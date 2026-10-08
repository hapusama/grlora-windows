# -*- coding: utf-8 -*-
"""实验A系统战汇总：战表 + exp2 同种子配对完整性 + 关键读数。"""
import json
import os

HERE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
        r"\data\experiments\dera_front_battle_20260930")
E2 = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
      r"\data\experiments\full_chain_20260929")
CHAINS = ["OURS×TRIMMER", "OURS×TREL-5", "DERA×TRIMMER", "DERA×DERA",
          "DERA×TREL-5"]
LEVELS = ["native"] + ["%+d" % (-v) for v in (10, 14, 17, 20, 22, 24, 26)]


def load(path):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        recs[(r["level"], r["seed"], r["frame"])] = r
    return recs


def main():
    new = load(os.path.join(HERE, "checkpoint.jsonl"))
    old = load(os.path.join(E2, "checkpoint.jsonl"))

    # ---- 同种子配对完整性：OURS 前端两链应逐位复现 exp2 ----
    mm = {"sync": 0, "TRIMMER": 0, "TREL-5": 0}
    n_shared = 0
    for k, r in new.items():
        if k not in old:
            continue
        n_shared += 1
        o = old[k]
        if bool(r["sync_ours"]) != bool(o["sync_ok"]):
            mm["sync"] += 1
        if r["sync_ours"]:
            if r["chains"]["OURS×TRIMMER"]["sym_err"] != o["chains"]["TRIMMER"]["sym_err"] \
                    or r["chains"]["OURS×TRIMMER"]["crc_fail"] != o["chains"]["TRIMMER"]["crc_fail"]:
                mm["TRIMMER"] += 1
            if r["chains"]["OURS×TREL-5"]["sym_err"] != o["chains"]["TREL-5"]["sym_err"] \
                    or r["chains"]["OURS×TREL-5"]["crc_fail"] != o["chains"]["TREL-5"]["crc_fail"]:
                mm["TREL-5"] += 1
    print("配对完整性：共享单元 %d，不一致 sync=%d TRIMMER=%d TREL-5=%d（应全 0）"
          % (n_shared, mm["sync"], mm["TRIMMER"], mm["TREL-5"]))

    # ---- 战表 ----
    agg, counts, sync = {}, {}, {}
    for r in new.values():
        key = "native" if r["level"] is None else "%+d" % r["level"]
        counts[key] = counts.get(key, 0) + 1
        sync.setdefault(key, {"OURS": [0, 0], "DERA": [0, 0]})
        sync[key]["OURS"][0] += int(r["sync_ours"])
        sync[key]["OURS"][1] += 1
        sync[key]["DERA"][0] += int(r["sync_dera"])
        sync[key]["DERA"][1] += 1
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\n实验A系统战表（SER/PER；native n=28，其余 n=84）")
    for key in LEVELS:
        if key not in agg:
            continue
        n_pkt = counts[key]
        so = sync[key]["OURS"][0] / max(sync[key]["OURS"][1], 1)
        sd_ = sync[key]["DERA"][0] / max(sync[key]["DERA"][1], 1)
        parts = " | ".join("%s %.3f/%.3f"
                           % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n_pkt) for c in CHAINS)
        print("[%7s] sync OURS=%.2f DERA=%.2f | %s (n=%d)"
              % (key, so, sd_, parts, n_pkt))


if __name__ == "__main__":
    main()

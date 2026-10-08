# -*- coding: utf-8 -*-
"""DT-FUSE 联合战汇总：战表 + 10% SER 门限 + 同种子配对完整性 + 包级 CRC 并集诊断。"""
import json
import os

HERE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
        r"\data\experiments\dera_fusion_battle_20260930")
OLD = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
       r"\data\experiments\dera_battle_20260929")
CHAINS = ["PLAIN", "OLD-A", "TRIMMER", "DERA", "DT-FUSE", "SAVAUX",
          "NEW-0", "TREL-5", "BCJR-5"]
SHARED = [c for c in CHAINS if c != "DT-FUSE"]
LEVELS = ["native"] + ["%+d" % v for v in range(-16, -27, -1)]


def load(path):
    recs = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        recs[(r["level"], r["seed"], r["frame"])] = r
    return recs


def agg(recs):
    a = {}
    for r in recs.values():
        key = "native" if r["level"] is None else "%+d" % r["level"]
        d = a.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            d[c][0] += r["chains"][c]["sym_err"]
            d[c][1] += r["sym_tot"]
            d[c][2] += r["chains"][c]["crc_fail"]
    return a


def ser_threshold(a, c, target=0.10):
    """线性内插 10% SER 门限（dB）；未交叉返回 None。"""
    pts = [("%+d" % v, a["%+d" % v][c][0] / max(a["%+d" % v][c][1], 1))
           for v in range(-16, -27, -1)]
    for (k0, s0), (k1, s1) in zip(pts, pts[1:]):
        if s0 < target <= s1:
            v0, v1 = int(k0), int(k1)
            return v0 + (target - s0) * (v1 - v0) / (s1 - s0)
    return None


def main():
    new = load(os.path.join(HERE, "checkpoint.jsonl"))
    old = load(os.path.join(OLD, "checkpoint.jsonl"))
    a = agg(new)

    # ---- 同种子配对完整性：共享 8 链逐位复现旧 v2 战表 ----
    mismatch = 0
    for k, r in new.items():
        if k not in old:
            continue
        for c in SHARED:
            if r["chains"][c] != old[k]["chains"][c]:
                mismatch += 1
    print("配对完整性：共享单元 %d/%d，逐位不一致 %d（应为 0）"
          % (sum(1 for k in new if k in old), len(new), mismatch))

    # ---- 战表 ----
    print("\n战表（SER / PER；native n=28，其余 n=84）")
    for key in LEVELS:
        if key not in a:
            continue
        n_pkt = sum(1 for r in new.values()
                    if ("native" if r["level"] is None else "%+d" % r["level"]) == key)
        parts = " | ".join("%s %.3f/%.3f" % (c, a[key][c][0] / max(a[key][c][1], 1),
                                             a[key][c][2] / n_pkt) for c in CHAINS)
        print("[%8s] %s (n=%d)" % (key, parts, n_pkt))

    # ---- 10% SER 门限 ----
    print("\n10%% SER 门限（dB，线性内插）")
    for c in CHAINS:
        print("  %-8s %s" % (c, "%.2f" % ser_threshold(a, c)
                             if ser_threshold(a, c) is not None else "未交叉"))

    # ---- 包级 CRC 并集诊断（DERA∪TRIMMER = "联合接收机两判据择一"上界） ----
    print("\n包级 CRC 并集诊断（PER：并集=任一链 CRC 通过即通过）")
    for key in LEVELS:
        if key == "native":
            continue
        recs = [r for r in new.values()
                if ("native" if r["level"] is None else "%+d" % r["level"]) == key]
        n = len(recs)
        u_dt = sum(1 for r in recs if r["chains"]["DERA"]["crc_fail"]
                   and r["chains"]["TRIMMER"]["crc_fail"]) / max(n, 1)
        u_all = sum(1 for r in recs if r["chains"]["DERA"]["crc_fail"]
                    and r["chains"]["TRIMMER"]["crc_fail"]
                    and r["chains"]["DT-FUSE"]["crc_fail"]) / max(n, 1)
        per_d = sum(r["chains"]["DERA"]["crc_fail"] for r in recs) / max(n, 1)
        per_t = sum(r["chains"]["TRIMMER"]["crc_fail"] for r in recs) / max(n, 1)
        per_5 = sum(r["chains"]["TREL-5"]["crc_fail"] for r in recs) / max(n, 1)
        print("[%8s] DERA %.3f | TRIM %.3f | D∪T %.3f | D∪T∪FUSE %.3f | TREL-5 %.3f"
              % (key, per_d, per_t, u_dt, u_all, per_5))


if __name__ == "__main__":
    main()

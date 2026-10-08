# -*- coding: utf-8 -*-
"""机制确认：DeRa 对齐段 + 时延扫描 → Savaux raw SER 曲线与 diff 分布。"""
import importlib.util as ilu
import numpy as np

spec = ilu.spec_from_file_location(
    "sr", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments"
          r"\dera_savaux_splice_20261004\splice_runner.py")
sr = ilu.module_from_spec(spec)
spec.loader.exec_module(sr)

sr.init_worker()
G = sr.G
NF, N, OS = sr.NF, sr.N, sr.OS


def ser_diff(rows, gt):
    psym = len(gt)
    d = np.array([(int(np.argmax(rows[k])) - (gt[k] + 1)) % N
                  for k in range(psym)])
    return float(np.mean(d != 0)), dict(zip(*np.unique(d, return_counts=True)))


def run(fi, extra_delay=None, label=""):
    f = G["frames"][fi]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
    seg_a = seg_as[0]
    if extra_delay is not None:
        seg_a = sr.frac_delay(seg_a, extra_delay)
    rows = sr.sav_rows(seg_a, pay0s[0], f["psym"])
    ser, dist = ser_diff(rows, f["gt"])
    print("f%02d %-28s sto=%+.3f SER=%.3f %s"
          % (fi, label, f["sto_frac"], ser,
             dist if ser > 0 else ""))
    return ser


print("== A: 坏帧 cand0 裸 vs +PRIOR 同款时延(-sto_frac·OS) ==")
for fi in (0, 1, 19, 13):
    run(fi, None, "cand0 裸")
    f = G["frames"][fi]
    run(fi, -f["sto_frac"] * OS, "cand0 +delay(-sto_frac·OS)")

print("== B: 干净帧加同样时延是否变坏（对抗测试）==")
for fi in (14, 17, 5):
    run(fi, None, "cand0 裸")
    f = G["frames"][fi]
    run(fi, -f["sto_frac"] * OS, "cand0 +delay(-sto_frac·OS)")

print("== C: f00 cand0 时延细扫（无 CFO 偏置）==")
f = G["frames"][0]
lead = f["pre"] + 6
seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                        f["hs"] + (8 + f["psym"] + 2) * NF],
                 dtype=np.complex128)
cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
best = None
for d in np.arange(-1.5, 1.51, 0.125):
    rows = sr.sav_rows(sr.frac_delay(seg_as[0], float(d)), pay0s[0],
                       f["psym"])
    ser, dist = ser_diff(rows, f["gt"])
    tag = ""
    if best is None or ser < best[1]:
        best = (float(d), ser)
        tag = "  <-- best"
    print("  d=%+.3f SER=%.3f %s%s" % (d, ser, dist if ser > 0 else "", tag))
print("f00 细扫 argmin: d=%+.3f SER=%.3f" % best)

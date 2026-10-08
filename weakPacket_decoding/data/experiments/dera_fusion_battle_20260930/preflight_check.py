# -*- coding: utf-8 -*-
"""DT-FUSE 联合战 preflight：等价性自检 + native 冒烟门禁（协议 §6）。

1) 我方三链 kappa_trellis 模块 vs dera_battle 内联实现逐位等价（协议
   "不得内联拷贝"的合法性自校验）；
2) DERA 非相干行 vs LoRaTrimmer 度量行数值对照（联合链第二条的同一性）；
3) native 门禁：TREL-5 SER=0、其余链 SER≤0.06，不过禁止开跑全量。
用法：D:\mysoft2\miniconda3\envs\gr-lora\python.exe preflight_check.py
"""
import importlib.util
import sys
import time

import numpy as np

HERE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
        r"\data\experiments\dera_fusion_battle_20260930")
OLD = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
       r"\data\experiments\dera_battle_20260929")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fr = _load("fusion_runner", HERE + r"\fusion_runner.py")
br = _load("battle_runner", OLD + r"\battle_runner.py")

print("building frames ...", flush=True)
t0 = time.time()
frames = fr.build_frames()
print("%d frames built (%.0fs)" % (len(frames), time.time() - t0), flush=True)
N = fr.N

# ---------------- 1) 模块 vs 内联 等价性（抽 3 帧，含三种 preamble） ----------------
ok = True
for fi in (0, 13, 27):
    f = frames[fi]
    r_new = fr.chain_rows(f["seg"], f["psym"])
    r_old = br.chain_rows(f["seg"], f["psym"])
    for c in ("NEW-0", "TREL-5", "BCJR-5", "TRIMMER", "SAVAUX", "DERA"):
        same = np.array_equal(r_new[c], r_old[c])
        ok &= bool(same)
        print("  frame %2d %-8s array_equal=%s (maxdiff %.3e)"
              % (fi, c, same,
                 float(np.max(np.abs(r_new[c] - r_old[c])))))
assert ok, "模块/内联等价性失败"

# ---------------- 2) DERA 非相干行 vs TRIMMER 度量 ----------------
rel = []
for fi in (0, 13, 27):
    f = frames[fi]
    _s1, _coh, _nc = fr.DERA.demod_payload_all(f["seg"], 16, f["psym"])
    tri = fr.chain_rows(f["seg"], f["psym"])["TRIMMER"]
    a, b = _nc.astype(np.float64), tri.astype(np.float64)
    rel.append(float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))))
print("DERA-noncoh vs TRIMMER 度量 最大相对差: %s" % ["%.2e" % x for x in rel])

# ---------------- 3) native 门禁 ----------------
gate = {}
for f in frames:
    rows = fr.chain_rows(f["seg"], f["psym"])
    d = int(f["delta"])
    for c in fr.CHAINS:
        rc = rows["OLD-A"] if c == "PLAIN" else rows[c]
        hard = [(int(np.argmax(rc[k])) - d) % N for k in range(f["psym"])]
        gate.setdefault(c, [0, 0])
        gate[c][0] += sum(int(h != g) for h, g in zip(hard, f["gt"]))
        gate[c][1] += f["psym"]
print("\nNATIVE SER（门禁：TREL-5=0，其余≤0.06）")
for c in fr.CHAINS:
    e, t = gate[c]
    print("  %-8s %d/%d = %.4f" % (c, e, t, e / max(t, 1)))

trel = gate["TREL-5"][0]
others = max(gate[c][0] / max(gate[c][1], 1) for c in fr.CHAINS if c != "TREL-5")
if trel != 0 or others > 0.06:
    print("\nNATIVE GATE FAIL (TREL=%d, worst=%.4f) —— 禁止开跑全量" % (trel, others))
    sys.exit(1)
print("\nNATIVE GATE PASS —— 可开跑 fusion_runner.py")

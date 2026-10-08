# -*- coding: utf-8 -*-
"""DeRa port v2（逐候选越界切分+相干合并）验证脚本。

门槛（不坑 baseline 红线）：native（干净 OTA 帧 ×28）SER 必须 ≈0 才可入表。
附带 −20dB×seed0 抽查，预览 DERA 相干行 vs TRIMMER 非相干行的相对位置。

用法：conda envs/gr-lora 的 python 直接跑本文件（复用 battle_runner 的
build_frames/judge_crc_fast/chain_rows，物理口径与战表逐位一致）。
"""
import importlib.util
import sys
import time

import numpy as np

RUNNER = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
          r"\data\experiments\dera_battle_20260929\battle_runner.py")
spec = importlib.util.spec_from_file_location("battle_runner", RUNNER)
br = importlib.util.module_from_spec(spec)
sys.modules["battle_runner"] = br
spec.loader.exec_module(br)

print("building frames ...", flush=True)
t0 = time.time()
frames = br.build_frames()
print("%d frames built (%.0fs)" % (len(frames), time.time() - t0), flush=True)

DERA = br.DERA
N = br.N

# ---------------- native 红线门槛 ----------------
tot = err_coh = err_s1 = err_tri = 0
t0 = time.time()
for f in frames:
    s1, coh = DERA.demod_payload(f["seg"], 16, f["psym"])
    d = int(f["delta"])
    hard = [(int(np.argmax(coh[k])) - d) % N for k in range(f["psym"])]
    err_coh += sum(int(h != g) for h, g in zip(hard, f["gt"]))
    hard1 = [(int(np.argmax(s1[k])) - d) % N for k in range(f["psym"])]
    err_s1 += sum(int(h != g) for h, g in zip(hard1, f["gt"]))
    tri = br.chain_rows(f["seg"], f["psym"])["TRIMMER"]
    hardt = [(int(np.argmax(tri[k])) - d) % N for k in range(f["psym"])]
    err_tri += sum(int(h != g) for h, g in zip(hardt, f["gt"]))
    tot += f["psym"]
print("NATIVE gate: DERA-coh %d/%d | DERA-stage1 %d/%d | TRIMMER %d/%d  (%.0fs)"
      % (err_coh, tot, err_s1, tot, err_tri, tot, time.time() - t0), flush=True)

# ---------------- -20dB x seed0 抽查预览 ----------------
LV, SD = -20, 0
acc = {"DERA": [0, 0, 0], "TRIMMER": [0, 0, 0]}  # sym_err, sym_tot, crc_fail
t0 = time.time()
for fi, f in enumerate(frames):
    rng = np.random.default_rng((20260929 * 7919 + (LV + 100) * 131
                                 + SD * 17 + fi) % (2 ** 31))
    p_add = max(f["S"] / 10 ** (LV / 10.0) - f["N0"], 1e-30)
    seg = f["seg"] + (rng.standard_normal(len(f["seg"]))
                      + 1j * rng.standard_normal(len(f["seg"]))) * np.sqrt(p_add / 2.0)
    rows = br.chain_rows(seg, f["psym"])
    _s1, coh = DERA.demod_payload(seg, 16, f["psym"])
    rows["DERA"] = coh
    for c in ("DERA", "TRIMMER"):
        rc = rows[c]
        hard = [(int(np.argmax(rc[k])) - int(f["delta"])) % N
                for k in range(f["psym"])]
        e = sum(int(h != g) for h, g in zip(hard, f["gt"]))
        crc = br.judge_crc_fast(rc, int(f["delta"]), f["gt_hdr"],
                                f["plen"], f["cr"])
        acc[c][0] += e
        acc[c][1] += f["psym"]
        acc[c][2] += int(not crc)
print("SNR=%d seed=%d: %s" % (LV, SD, "  ".join(
    "%s SER %.3f PER %.3f" % (c, a[0] / max(a[1], 1), a[2] / len(frames))
    for c, a in acc.items())), "(%.0fs)" % (time.time() - t0), flush=True)

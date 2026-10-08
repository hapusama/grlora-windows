# -*- coding: utf-8 -*-
"""复杂度基准 + 步罚敏感性（红队自首项）。

B1: 逐组件 wall-time（native 单帧）：dera_sync / DERA 解码 / SAVT 整包
    网格 / SAVT2 双层网格 / Savaux 裸解调。输出每包相对倍数。
B2: 步罚敏感性：PEN_F ∈ {0.1,0.2,0.4,0.8} × −24 档 seed0 28 帧，
    DERA×SAVT2 PER——参数不敏感则 garden-of-forking-paths 攻击卸力。
"""
import importlib.util as ilu
import json
import time
import numpy as np

sr_spec = ilu.spec_from_file_location(
    "sr", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_runner.py")
sr = ilu.module_from_spec(sr_spec)
sr_spec.loader.exec_module(sr)
sr.init_worker()

v4_spec = ilu.spec_from_file_location(
    "v4", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_v4_runner.py")
v4 = ilu.module_from_spec(v4_spec)
v4_spec.loader.exec_module(v4)

v4b_spec = ilu.spec_from_file_location(
    "v4b", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
           r"\splice_dera_savaux\code\splice_v4b_runner.py")
v4b = ilu.module_from_spec(v4b_spec)
v4b_spec.loader.exec_module(v4b)

G = sr.G
NF, N, OS, SF_ = sr.NF, sr.N, sr.OS, sr.SF

print("== B1: 复杂度基准（native，3 帧 × {pre8,16,32}，wall-time 秒）==")
for fi in (0, 9, 20):
    f = G["frames"][fi]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    T = {}
    t0 = time.perf_counter(); sr.dera_sync(seg, f["pre"])
    T["dera_sync(检测)"] = time.perf_counter() - t0
    cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
    seg_a, pay0 = seg_as[0], pay0s[0]
    t0 = time.perf_counter(); sr.DERA_DEC.demod_payload(seg_a, pay0, f["psym"])
    T["DERA解码"] = time.perf_counter() - t0
    t0 = time.perf_counter(); sr.sav_rows(seg_a, pay0, f["psym"])
    T["SAVAUX裸解调"] = time.perf_counter() - t0
    t0 = time.perf_counter(); v4.trellis_rows(seg_a, pay0, f["psym"])
    T["SAVT整包网格(17)"] = time.perf_counter() - t0
    t0 = time.perf_counter(); v4b.savt2_outputs(seg_a, pay0, f["psym"])
    T["SAVT2双层网格(35)"] = time.perf_counter() - t0
    print("f%02d psym=%d | %s" % (fi, f["psym"],
          " | ".join("%s %.3fs" % (k, v) for k, v in T.items())))

print("\n== B2: 步罚敏感性（−24 档 seed0，28 帧，DERA×SAVT2）==")
lv, sd = -24, 0
crcs = {0: 0}
for pen in (0.1, 0.2, 0.4, 0.8):
    v4b.PEN_F = pen
    ok_n = 0
    tot_err = 0
    tot_den = 0
    for fi in range(28):
        f = G["frames"][fi]
        lead = f["pre"] + 6
        seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                                f["hs"] + (8 + f["psym"] + 2) * NF],
                         dtype=np.complex128)
        S, N0 = G["snr"][fi]
        rng = np.random.default_rng((20260930 * 7919 + (lv + 100) * 131
                                     + sd * 17 + fi * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (lv / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
        cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
        if not cands:
            continue
        outs = v4b.savt2_outputs(seg_as[0], pay0s[0], f["psym"])
        _d, ok, rows = sr.decode_chain(None, f, top5=outs)
        ok_n += int(ok)
        tot_err += sum((int(np.argmax(rows[k])) - _d) % N != g
                       for k, g in enumerate(f["gt"]))
        tot_den += f["psym"]
    print("PEN=%.1f  PER=%.3f  SER=%.4f" % (pen, 1 - ok_n / 28.0,
                                            tot_err / max(tot_den, 1)))
v4b.PEN_F = 0.2
print("(v4-b 战役用 PEN=0.2)")

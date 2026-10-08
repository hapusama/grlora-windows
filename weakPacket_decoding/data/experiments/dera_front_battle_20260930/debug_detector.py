# -*- coding: utf-8 -*-
"""检测 port 数值调试：frame13(pre=16) 平均符号谱峰位 + pre=8 帧扫描诊断。"""
import importlib.util
import sys

import numpy as np

E2 = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
      r"\data\experiments\full_chain_20260929")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2 = _load("exp2_runner", E2 + r"\exp2_runner.py")
from weak_decoder.baselines.dera.paper_dera_detector import DeRaDetector
from weak_decoder.chirp import build_upchirp

SF, N, OS, NF = e2.SF, e2.N, e2.OS, e2.NF
DET = DeRaDetector(SF, OS)
frames = e2.build_frames()

# ---- frame 13 (pre=16)：以 CSV 先验对齐后，直接看真前导的符号谱 ----
f = frames[13]
pre = f["pre"]
lead = pre + 6
seg = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + (8 + f["psym"] + 2) * NF],
                 dtype=np.complex128)
# 真前导首符号（段坐标）：hs 在段内位置 = lead·NF
pre_start = lead * NF
rows = np.stack([DET.signed_spectrum(seg, pre_start + i * NF, DET.down_ref)
                 for i in range(pre)])
mag = np.sqrt(np.mean(np.abs(rows) ** 2, axis=0))
top = np.argsort(mag)[::-1][:6]
print("frame13 csv cfo=%+.3f → 期望 signed k* = %.1f" % (f["cfo"], N + 2 * f["cfo"]))
print("平均谱 top6 k: %s  (bin 域 %s)  mag %s"
      % (top, np.round((top - N) / 2.0, 1), np.round(mag[top] / mag.max(), 3)))
# 列向 FFT 在真 k 上
k_true = int(round(N + 2 * f["cfo"]))
for k in (k_true - 1, k_true, k_true + 1, int(top[0])):
    F = np.abs(np.fft.fft(rows[:, k], DET.n_fine))
    q = int(np.argmax(F))
    print("  k=%d: score=%.1f dB (峰均比), q̂=%d → κ̂=%+.4f"
          % (k, 10 * np.log10(F[q] ** 2 / np.mean(F ** 2)), q,
             ((q / DET.n_fine + 0.5) % 1.0) - 0.5))

# ---- pre=8 帧（frame 0）：扫描事件 + 相干得分分布 ----
f = frames[0]
pre = f["pre"]
lead = pre + 6
seg = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + (8 + f["psym"] + 2) * NF],
                 dtype=np.complex128)
ev = DET.scan(seg)
print("\nframe0(pre=8) scan 事件数 %d（前2个）" % len(ev))
pre_start = lead * NF
rows8 = np.stack([DET.signed_spectrum(seg, pre_start + i * NF, DET.down_ref)
                  for i in range(8)])
mag8 = np.sqrt(np.mean(np.abs(rows8) ** 2, axis=0))
k8 = int(np.argmax(mag8))
print("  真前导平均谱峰 k=%d (bin %+d)，期望 %.1f"
      % (k8, (k8 - N) // 2, N + 2 * f["cfo"]))
for k in (k8 - 1, k8, k8 + 1):
    F = np.abs(np.fft.fft(rows8[:, k], DET.n_fine))
    q = int(np.argmax(F))
    print("  k=%d: score=%.1f dB, κ̂=%+.4f"
          % (k, 10 * np.log10(F[q] ** 2 / np.mean(F ** 2)),
             ((q / DET.n_fine + 0.5) % 1.0) - 0.5))
for e in ev[:2]:
    print("  event: start=%d bin=%d power=%.1f" % (e["start"], e["ref_bin"], e["power"]))
    cs = DET.coherent_stage(seg, 8, e)
    for c in cs[:4]:
        print("    cand start=%d score=%.1f f̂=%+.2f κ̂=%+.3f"
              % (c["start"], c["score_db"], c["f_bins"], c["kappa"]))

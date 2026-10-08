# -*- coding: utf-8 -*-
"""探针2：真前导上 (a) 偏移-不变性隔离（无填 vs 填充谱），(b) 列向 FFT 得分剖析。"""
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

SF, N, OS, NF = e2.SF, e2.N, e2.OS, e2.NF
DET = DeRaDetector(SF, OS)
frames = e2.build_frames()

f = frames[0]
pre = f["pre"]
lead = pre + 6
seg = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + (8 + f["psym"] + 2) * NF],
                 dtype=np.complex128)
# 真前导起点（段坐标）= (lead − pre − 4.25)·NF = 1.75·NF（pre=8）
ps = int((lead - pre - 4.25) * NF)
print("frame0 pre=%d 真前导段坐标 ps=%d (=1.75·NF=%d)  csv cfo=%+.4f"
      % (pre, ps, int(1.75 * NF), f["cfo"]))

print("\n(a) 偏移不变性：窗口 ps+off，比较无填 4096-FFT 峰 bin 与填充 8192 signed 峰")
for off in (-512, -256, 0, 256, 512):
    w = np.asarray(seg[ps + off: ps + off + NF], dtype=np.complex64)
    X = np.fft.fft(w * DET.down_ref)                    # 无填
    half = N // 2
    row = np.concatenate((np.abs(X[NF - half:]) ** 2, np.abs(X[:half]) ** 2))
    k1 = int(np.argmax(row))                            # 1-bin 格，signed
    sp = DET.signed_spectrum(seg, ps + off, DET.down_ref)
    k2 = int(np.argmax(np.abs(sp)))                     # 0.5-bin 格，signed
    print("  off=%+5d: 无填峰 k=%d (bin %+d) | 填充峰 k=%d (bin %+.1f)"
          % (off, k1, k1 - half, k2, (k2 - N) / 2.0))

print("\n(b) 列向 FFT 剖析（s=ps）：逐窗 k* 与列 |F|")
rows = np.stack([DET.signed_spectrum(seg, ps + i * NF, DET.down_ref)
                 for i in range(pre)])
mag = np.sqrt(np.mean(np.abs(rows) ** 2, axis=0))
k_star = int(np.argmax(mag))
print("  平均谱峰 k*=%d (bin %+.1f)，期望 bin %+.1f"
      % (k_star, (k_star - N) / 2.0, f["cfo"]))
per_win = [int(np.argmax(np.abs(rows[i]))) for i in range(pre)]
print("  逐窗峰 k: %s" % per_win)
for k in (k_star - 1, k_star, k_star + 1):
    col = rows[:, k]
    F = np.abs(np.fft.fft(col, DET.n_fine))
    q = int(np.argmax(F))
    top_q = np.argsort(F)[::-1][:5]
    print("  k=%d: |Y_i|=%s" % (k, np.round(np.abs(col) / np.abs(col).max(), 2)))
    print("        |F| top5 q=%s  归一 %s；score=%.1f dB；κ̂=%+.4f"
          % (top_q, np.round(F[top_q] / F.max(), 2),
             10 * np.log10(F[q] ** 2 / np.mean(F ** 2)),
             ((q / DET.n_fine + 0.5) % 1.0) - 0.5))

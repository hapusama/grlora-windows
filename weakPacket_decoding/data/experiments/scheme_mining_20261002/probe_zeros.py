# -*- coding: utf-8 -*-
"""A3 探针：前导零区？量化真伪？rms 包络 vs 时间。"""
import numpy as np

for path, tag in (
    (r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_32.bin", "SF10_32"),
    (r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin", "SF10_8"),
    (r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\lab1_sf11_TP2\1_0_8_11_2_16.bin", "SF11_1_0_8"),
):
    iq = np.memmap(path, dtype=np.complex64, mode="r")
    n = len(iq)
    print("== %s  %d samples" % (tag, n))
    z = np.asarray(iq[:2_000_000])
    print("  前 2M: 精确零占比 = %.4f  (I==0 & Q==0)"
          % float(np.mean((z.real == 0) & (z.imag == 0))))
    w = 100_000
    env = []
    for o in range(0, min(n, 12_000_000), w):
        x = np.asarray(iq[o:o + w])
        env.append(float(np.sqrt(np.mean(np.abs(x) ** 2))))
    env = np.array(env)
    print("  rms 包络(前12M, 100k窗): min=%.2e max=%.2e" % (env.min(), env.max()))
    print("   ", " ".join("%.1e" % e for e in env[:40]))
    # 中段量化检查（必含噪声）
    mid = n // 2
    q = np.asarray(iq[mid:mid + 1_000_000]).real
    u = np.unique(np.round(q * 1e9))
    print("  中段 1M I 路 unique(1e9) = %d  (float32 连续应≈海量)" % len(u))
    del iq

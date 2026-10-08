# -*- coding: utf-8 -*-
"""A3 探针2：'空隙'里那 8e-4 rms 的东西到底是什么（3 个 unique 值的真相）。"""
import numpy as np

path = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_32.bin"
iq = np.memmap(path, dtype=np.complex64, mode="r")

# 中段 1M 逐值统计
mid = len(iq) // 2
z = np.asarray(iq[mid:mid + 1_000_000])
I, Q = z.real, z.imag
print("I unique:", np.unique(I))
print("Q unique:", np.unique(Q))
print("非零样本占比: I %.4f  Q %.4f" % (np.mean(I != 0), np.mean(Q != 0)))
nz = np.where((I != 0) | (Q != 0))[0]
print("非零样本数 %d, 首 20 个 (I,Q):" % len(nz))
print(list(zip(I[nz[:20]].tolist(), Q[nz[:20]].tolist())))
# 非零样本的位置模式
if len(nz) > 100:
    d = np.diff(nz)
    print("非零间隔: min=%d 中位=%.0f max=%d, 众数=%d"
          % (d.min(), np.median(d), d.max(), np.bincount(d).argmax()))

# SF10_8 一段"完全零"空隙再确认
iq8 = np.memmap(r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin",
                dtype=np.complex64, mode="r")
z8 = np.asarray(iq8[1_000_000:2_000_000])
print("\nSF10_8 1M-2M: 精确零占比=%.6f, 非零 unique I=%s"
      % (float(np.mean((z8.real == 0) & (z8.imag == 0))), np.unique(z8.real)[:8]))

# SF11 空隙微结构
iq11 = np.memmap(r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\lab1_sf11_TP2\1_0_8_11_2_16.bin",
                 dtype=np.complex64, mode="r")
z11 = np.asarray(iq11[2_600_000:3_000_000])
print("\nSF11 2.6M-3.0M: 非零占比=%.4f, |x| 中位=%.2e, I unique(1e6r)=%d"
      % (float(np.mean((z11.real != 0) | (z11.imag != 0))),
         float(np.median(np.abs(z11))), len(np.unique(np.round(z11.real * 1e6)))))
a = np.abs(z11)
print("  |x| 分位: P50=%.2e P90=%.2e P99=%.2e max=%.2e"
      % tuple(np.percentile(a, [50, 90, 99, 100])))

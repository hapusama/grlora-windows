# -*- coding: utf-8 -*-
"""探针3：瞬时频率直接测量——合成前导 / OTA前导 / OTA header 与参考的乘积斜率。"""
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
from weak_decoder.chirp import build_upchirp

SF, N, OS, NF = e2.SF, e2.N, e2.OS, e2.NF
ref = np.conj(build_upchirp(SF, symbol_id=0, os_factor=OS)).astype(np.complex64)

def inst_freq_slope(y):
    ph = np.unwrap(np.angle(y))
    f = np.diff(ph) / (2 * np.pi)          # cycles/sample
    return float(np.mean(f)), float(np.polyfit(np.arange(len(f)), f, 1)[0])

# (1) 合成标准前导（我们自己的啁啾当 TX）
tx = build_upchirp(SF, symbol_id=0, os_factor=OS).astype(np.complex64)
pre8 = np.tile(tx, 8)
m, sl = inst_freq_slope(pre8[:NF] * ref)
print("合成前导×ref:  平均频率 %+.4f cyc/smp, 斜率 %+.3e cyc/smp²" % (m, sl))

# (2) OTA frame0 前导窗（ps=7168）与 (3) OTA header 窗
frames = e2.build_frames()
f = frames[0]
pre = f["pre"]
lead = pre + 6
seg = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + (8 + f["psym"] + 2) * NF],
                 dtype=np.complex128)
ps = int((lead - pre - 4.25) * NF)
m, sl = inst_freq_slope(seg[ps:ps + NF].astype(np.complex64) * ref)
print("OTA 前导窗×ref:  平均 %+.4f cyc/smp, 斜率 %+.3e" % (m, sl))
hdr = int(lead * NF)   # hs 段坐标
for j in (0, 4):
    m, sl = inst_freq_slope(seg[hdr + j * NF: hdr + (j + 1) * NF]
                            .astype(np.complex64) * ref)
    print("OTA header 窗 %d×ref:  平均 %+.4f cyc/smp, 斜率 %+.3e" % (j, m, sl))

# (4) OTA 前导窗与【2 倍斜率参考】对照（诊断残差率是否恰为 +μ）
n = np.arange(NF)
ref2 = np.conj(np.exp(2j * np.pi * (n * n / (2.0 * N) / (OS * OS) * 2.0
                                     - 0.5 * n / OS)).astype(np.complex64))
m, sl = inst_freq_slope(seg[ps:ps + NF].astype(np.complex64) * ref2)
print("OTA 前导窗×2μ参考: 平均 %+.4f, 斜率 %+.3e" % (m, sl))

# (5) 信号自身去斜前后带宽检查：直接看 OTA 前导窗的 FFT 峰随窗内位置
X = np.fft.fft(seg[ps:ps + NF].astype(np.complex64) * ref)
top = np.argsort(np.abs(X))[::-1][:5]
print("OTA 前导窗 4096-FFT top5 bin: %s  |X|² 归一 %s"
      % (top, np.round(np.abs(X[top]) ** 2 / np.abs(X[top[0]]) ** 2, 3)))

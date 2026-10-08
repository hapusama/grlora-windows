# -*- coding: utf-8 -*-
"""探针4：OTA 前导乘积的逐块局部频率（256 样本块，前 3 个符号）+ 相邻窗 FFT 峰直接对比。"""
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

frames = e2.build_frames()
f = frames[0]
pre = f["pre"]
lead = pre + 6
seg = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + (8 + f["psym"] + 2) * NF],
                 dtype=np.complex128)
ps = int((lead - pre - 4.25) * NF)

y = (seg[ps: ps + 3 * NF].astype(np.complex64)) * np.tile(ref, 3)
print("逐 256 块局部平均频率（cycles/sample，×4096=FFT bin）：")
for b in range(y.size // 256):
    blk = y[b * 256:(b + 1) * 256]
    ph = np.unwrap(np.angle(blk))
    fq = np.mean(np.diff(ph)) / (2 * np.pi)
    print("  blk %2d [smp %6d]: %+8.4f cyc/smp = %+7.1f bins"
          % (b, b * 256, fq, fq * NF))

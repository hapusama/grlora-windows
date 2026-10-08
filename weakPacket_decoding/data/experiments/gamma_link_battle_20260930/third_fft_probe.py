# -*- coding: utf-8 -*-
"""第三次 FFT 演示：符号轴啁啾变换（δ 银行列向 FFT）在真实 OTA 上的发现。

物理：第二次 FFT（DeRa）= 符号轴线性相位匹配滤波，假设旋转率恒定；
确定性 SFO 使相位二次化（πδi(i−1)）→ 相干积累散焦。第三次变换 =
先乘 e^{−jπδ̂i(i−1)} 再列向 FFT，δ 扫一个银行 —— 等价于符号轴的
chirp-z/FrFT（"带扭转的第三次 FFT"）。预言：P=32 帧（真实 δ≈0.0036）
得分峰应出现在 δ̂≈0.0036 处，且比 δ=0 高 ~3dB（闭式 −3.45dB）；
P=8 帧（闭式 −0.01dB）应无增益 = 天然负对照。
"""
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


def column_score(seg, pre_start, pre, delta, n_fine=256):
    """给定 δ 补偿的列向 FFT 相干 SNR（dB）。"""
    rows = np.stack([DET.signed_spectrum(seg, pre_start + i * NF,
                                         DET.down_ref)
                     for i in range(pre)])
    mag = np.sqrt(np.mean(np.abs(rows) ** 2, axis=0))
    k = int(np.argmax(mag))
    far = np.ones(rows.shape[1], bool)
    far[max(0, k - 16):k + 17] = False
    sig2 = float(np.median(np.abs(rows[:, far]) ** 2))
    i_ax = np.arange(pre, dtype=float)
    col = rows[:, k] * np.exp(-1j * np.pi * delta * i_ax * (i_ax - 1))
    F = np.abs(np.fft.fft(col, n_fine))
    return 10.0 * np.log10(float(F.max() ** 2) / (pre * sig2))


print("第三次 FFT（δ 银行）在真实 OTA 前导上的发现：")
print("%4s %3s %2s | %9s %9s %8s | %s" %
      ("idx", "pre", "n", "δ=0 得分", "δ̂* 得分", "增益dB", "δ̂* (bins/符)"))
by_pre = {}
for idx, f in enumerate(frames):
    pre = f["pre"]
    if pre not in by_pre:
        by_pre[pre] = []
    if len(by_pre[pre]) >= 3:
        continue
    by_pre[pre].append(idx)
    lead = pre + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    pre_start = int((lead - pre - 4.25) * NF)      # 真前导网格
    bank = np.linspace(-0.012, 0.020, 65)
    scores = np.array([column_score(seg, pre_start, pre, d) for d in bank])
    j = int(np.argmax(scores))
    print("%4d %3d %2d | %9.2f %9.2f %+8.2f | %+.5f"
          % (idx, pre, 1, scores[np.argmin(np.abs(bank))], scores[j],
             scores[j] - scores[np.argmin(np.abs(bank))], bank[j]))

# 深端一枪：P=32 帧 @−24dB（exp2 种子）δ 补偿前后
print("\nP=32 @−24dB（真实噪声，exp2 种子）δ=0 vs δ̂*：")
for idx, f in enumerate(frames):
    if f["pre"] != 32:
        continue
    pre, lead = f["pre"], f["pre"] + 6
    seg0 = np.asarray(f["iq"][f["hs"] - lead * NF:
                              f["hs"] + (8 + f["psym"] + 2) * NF],
                      dtype=np.complex128)
    S, N0 = e2.snr_parts(np.asarray(
        f["iq"][f["hs"] - lead * NF: f["hs"] + 8 * NF], dtype=np.complex128))
    rng = np.random.default_rng((20260930 * 7919 + (-24 + 100) * 131
                                 + 0 * 17 + idx * 7919) % (2 ** 31))
    p_add = max(S / 10 ** (-2.4) - N0, 1e-30)
    seg = seg0 + (rng.standard_normal(len(seg0))
                  + 1j * rng.standard_normal(len(seg0))) * np.sqrt(p_add / 2)
    pre_start = int((lead - pre - 4.25) * NF)
    bank = np.linspace(-0.012, 0.020, 65)
    scores = np.array([column_score(seg, pre_start, pre, d) for d in bank])
    j = int(np.argmax(scores))
    print("  frame %2d: δ=0 → %.1f dB | δ̂*=%+.5f → %.1f dB（增益 %+.2f）"
          % (idx, scores[np.argmin(np.abs(bank))], bank[j], scores[j],
             scores[j] - scores[np.argmin(np.abs(bank))]))

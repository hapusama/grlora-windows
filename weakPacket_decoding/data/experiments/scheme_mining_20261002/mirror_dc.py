# -*- coding: utf-8 -*-
"""A3 收尾：I/Q 镜像比（iq imbalance）+ DC 分量对解调的实际影响 + 带外肩部定量。"""
import numpy as np
import csv
import sys

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.chirp import build_upchirp

bp = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\lab1_sf11_TP2\1_0_8_11_2_16.bin"
cp = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
      r"\header_first\1_0_8_11_2_16_header_first_frames.csv")
SF, N, NF = 11, 2048, 8192
ref = np.conj(build_upchirp(SF, 0, os_factor=1))
iq = np.memmap(bp, dtype=np.complex64, mode="r")
rows = [r for r in csv.DictReader(open(cp, encoding="utf-8"))
        if r.get("header_valid") == "1"]
P = 16
print("镜频/直流检验（SF11 最强帧集, SNR 数万倍）:")
for fi, r0 in enumerate(rows[:6]):
    hs = int(r0["header_start_sample"])
    cfo = int(r0["source_grlora_cfo_int"]) + float(r0["source_grlora_cfo_frac"])
    starts = [hs - int((4.25 + P - k) * NF) + NF // 4 for k in range(P)]
    if starts[0] < 0:
        continue
    S = []
    for st in starts:
        seg = np.asarray(iq[st:st + NF], dtype=np.complex128)
        seg = seg * np.exp(-2j * np.pi * cfo * np.arange(NF) / NF)
        S.append(np.fft.fft(seg[::4] * ref))
    S = np.stack(S)
    pk = int(np.bincount(np.argmax(np.abs(S), axis=1)).argmax())
    al = S / (S[:, pk] / np.abs(S[:, pk]))[:, None]
    B = np.abs(al.mean(axis=0))
    mir = (N - pk) % N
    # 镜像位置取 ±2 bin 最大（ν 半 bin 时镜像也劈叉）
    mir_amp = B[(mir - 2) % N:(mir + 3) % N].max()
    ratio_db = 20 * np.log10(mir_amp / B[pk] + 1e-30)
    # 直流：整段均值 / 帧信号 rms
    frame = np.asarray(iq[hs - 20 * NF:hs + 40 * NF])
    dc = abs(frame.mean())
    rms = np.sqrt(np.mean(np.abs(frame) ** 2))
    print("  f%d: 峰bin=%d 镜像 %+.1f dBc ; DC/rms=%.1f%%"
          % (fi, pk, ratio_db, 100 * dc / rms))

# -*- coding: utf-8 -*-
"""A3 核查A：量化格点 + 量化噪声/热噪声功率分配（帧内 vs 空隙）。

对 SF10 三文件 + SF11 若干文件：
1. 帧内样本是否落在 1/128 格点上（量化证据）；
2. 空隙"噪声"功率 vs 量化噪声理论下限 Δ²/12 → 热噪声占比；
3. 若信号峰值 ~2 LSB，量化噪声占接收噪声的比例（可挽回 dB 上限）。
"""
import numpy as np
import csv
import glob
import os

HF = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
      r"\header_first")
USRP = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ"

NF10, NF11 = 4096, 8192
jobs = [("SF10_8", os.path.join(USRP, "0_0_0_10_14_8.bin"),
         os.path.join(HF, "0_0_0_10_14_8_header_first_frames.csv"), NF10),
        ("SF10_16", os.path.join(USRP, "0_0_0_10_14_16.bin"),
         os.path.join(HF, "0_0_0_10_14_16_header_first_frames.csv"), NF10),
        ("SF10_32", os.path.join(USRP, "0_0_0_10_14_32.bin"),
         os.path.join(HF, "0_0_0_10_14_32_header_first_frames.csv"), NF10),
        ("SF11_1_0_8", os.path.join(USRP, "lab1_sf11_TP2", "1_0_8_11_2_16.bin"),
         os.path.join(HF, "1_0_8_11_2_16_header_first_frames.csv"), NF11),
        ("SF11_1_1_3", os.path.join(USRP, "lab1_sf11_TP2", "1_1_3_11_2_16.bin"),
         os.path.join(HF, "1_1_3_11_2_16_header_first_frames.csv"), NF11)]

for tag, bp, cp, NF in jobs:
    iq = np.memmap(bp, dtype=np.complex64, mode="r")
    rows = [r for r in csv.DictReader(open(cp, encoding="utf-8"))
            if r.get("header_valid") == "1"]
    if not rows:
        print("%s: 无有效帧" % tag); continue
    # --- 帧内格点核查（最强帧：header 起 43 符号） ---
    hs = int(rows[0]["header_start_sample"])
    seg = np.asarray(iq[hs:hs + 43 * NF])
    lat = seg * 128.0
    resid = np.abs(lat - np.round(lat))
    on_lattice = float(np.mean(resid < 1e-3))
    peak_lsb = float(np.max(np.abs(lat)))
    rms_lsb = float(np.sqrt(np.mean(np.abs(lat) ** 2)))
    # --- 空隙：取文件前 5% 内、第一帧之前的纯空隙 ---
    first = min(int(r["header_start_sample"]) for r in rows)
    pre = np.asarray(iq[: max(first - NF, 0)])
    if len(pre) < 1000:
        # 用帧间空隙
        srt = sorted(int(r["header_start_sample"]) for r in rows)
        for a, b in zip(srt, srt[1:]):
            if b - a > 60 * NF:
                pre = np.asarray(iq[a + 50 * NF:b - 2 * NF]); break
    idle_rms_lsb = float(np.sqrt(np.mean(np.abs(pre * 128.0) ** 2))) if len(pre) else float("nan")
    D2_12 = 1.0 / 12.0                       # Δ=1 LSB 单位
    q_rms = np.sqrt(D2_12)
    therm_sq = max(idle_rms_lsb ** 2 - D2_12, 0.0)
    # 帧内：信号+噪声 vs 量化噪声占比
    noise_tot = idle_rms_lsb ** 2
    q_share = D2_12 / noise_tot if noise_tot > 0 else float("nan")
    print("%s: 帧内格点命中率=%.4f  峰值=%.2f LSB  rms=%.2f LSB" % (tag, on_lattice, peak_lsb, rms_lsb))
    print("   空隙噪声 rms=%.3f LSB; 量化噪声下限 Δ/√12=%.3f LSB; 量化噪声占比=%.0f%%"
          % (idle_rms_lsb, q_rms, 100 * q_share if q_share == q_share else -1))
    if q_share == q_share and 0 < q_share <= 1:
        claw = 10 * np.log10(1.0 / (1.0 - q_share))     # 完美去除量化噪声的 dB 上限
        print("   → 热噪声本底占比 %.0f%%, 理论可挽回上限 %.1f dB（需完美知道量化噪声）"
              % (100 * (1 - q_share), claw))
    del iq

# --- 对照：SF12 savaux 捕获（fs=2M, sc16?）---
p12 = glob.glob(r"D:\Desktop\proj\lora-rfsr-savaux\data\raw\ota\*.cfile")[0]
iq12 = np.memmap(p12, dtype=np.complex64, mode="r")
z = np.asarray(iq12[50_000_000:50_400_000])
for scale, name in ((128.0, "1/128"), (32768.0, "1/32768"), (8192.0, "1/8192")):
    lat = z * scale
    resid = np.abs(lat - np.round(lat))
    print("SF12 savaux: 格点 %s 命中率=%.4f 峰值=%.1f LSB" % (name, np.mean(resid < 1e-3), np.max(np.abs(lat))))
print("SF12 |x| max=%.4f rms=%.5f" % (np.max(np.abs(z)), np.sqrt(np.mean(np.abs(z) ** 2))))

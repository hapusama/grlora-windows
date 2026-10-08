# -*- coding: utf-8 -*-
"""A3 补充：量化深度核查 + 帧间空隙噪声底平坦度 + 边带尖峰定量。"""
import numpy as np
import csv

SF10 = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_32.bin"
CSV10 = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
         r"\header_first\0_0_0_10_14_32_header_first_frames.csv")

iq = np.memmap(SF10, dtype=np.complex64, mode="r")

# ---------- 1. 量化深度 ----------
q = np.asarray(iq[: 1 << 22]).real
qz = q[np.abs(q) < 0.004]
print("量化深度: |I|<0.004 样本数 %d, unique=%.0f" % (len(qz), len(np.unique(qz))))
u = np.unique(np.round(q[np.abs(q) < 0.0005] * 1e9))
print("  近零 unique(round(1e9*I)) 前 12 个:", u[:12])
step = np.diff(np.unique(np.round(q[np.abs(q) < 0.004] * 1e9)))
if len(step):
    print("  最小相邻差(1e9 单位) = %d  → 量化步长 ≈ %.2e  (%.1f bit @满量程0.0175)"
          % (step.min(), step.min() * 1e-9, np.log2(0.0175 / (step.min() * 1e-9))))
# 全幅 unique 计数
print("  1<<22 样本 I 路总 unique(1e6 舍入) = %d" % len(np.unique(np.round(q * 1e6))))

# ---------- 2. 帧间空隙噪声底 ----------
rows = [r for r in csv.DictReader(open(CSV10, encoding="utf-8"))
        if r.get("header_valid") == "1"]
hs = sorted(int(r["header_start_sample"]) for r in rows)
NF = 4096
print("\n%d 个有效帧, header_start 首末: %d .. %d" % (len(hs), hs[0], hs[-1]))
# 取两帧之间 > 3s 的空隙中心做纯噪声 PSD
gaps = []
for a, b in zip(hs, hs[1:]):
    g = b - (a + 50 * NF)
    if g > 1_500_000:
        gaps.append(a + 50 * NF + g // 2)
print("空隙中心样本:", gaps[:6], "..." if len(gaps) > 6 else "")

seg = 1 << 17
win = np.hanning(seg)
P = np.zeros(seg)
cnt = 0
for o in gaps:
    if o + seg < len(iq):
        x = np.asarray(iq[o:o + seg], dtype=np.complex128)
        P += np.abs(np.fft.fftshift(np.fft.fft(x * win))) ** 2
        cnt += 1
P /= cnt * (win ** 2).sum()
f = np.fft.fftshift(np.fft.fftfreq(seg)) * 500000.0
ib = np.abs(f) < 60e3
print("空隙噪声底 (%d 段): 带内中位 %.3e" % (cnt, np.median(P[ib])))
kf = np.array_split(np.where(ib)[0], 20)
flat = np.array([np.median(P[k]) for k in kf])
fc = np.array([np.median(f[k]) for k in kf])
print("  带内 3kHz 粒度平坦度: min/max=%.2f dB, 相对 std=%.1f%%"
      % (10 * np.log10(flat.min() / flat.max()), 100 * flat.std() / flat.mean()))
# 带内 vs 带外
for lo, hi, t in ((60e3, 70e3, "60-70k"), (70e3, 120e3, "70-120k"),
                  (120e3, 240e3, "120-240k")):
    s = (np.abs(f) > lo) & (np.abs(f) < hi)
    print("  %s 中位 vs 带内: %+.2f dB" % (t, 10 * np.log10(np.median(P[s]) / np.median(P[ib]))))
# 空隙里的窄峰（真实环境非白证据）
med = np.median(P)
sm = np.convolve(P, np.ones(201) / 201, "same")
resid = P / (sm + 1e-30)
cand = np.where((resid > 8) & (np.abs(f) < 240e3))[0]
groups = []
for i in cand:
    if groups and i - groups[-1][-1] <= 4:
        groups[-1].append(i)
    else:
        groups.append([i])
print("  空隙谱 9dB+ 窄峰 %d 个; 最强 10 个:" % len(groups))
top = sorted(groups, key=lambda g: -P[g[np.argmax(P[g])]])[:10]
for g in top:
    j = g[np.argmax(P[g])]
    print("    f=%+8.0f Hz  %+.1f dB vs 带内中位 (峰/底=%.0f)"
          % (f[j], 10 * np.log10(P[j] / np.median(P[ib])), resid[j]))

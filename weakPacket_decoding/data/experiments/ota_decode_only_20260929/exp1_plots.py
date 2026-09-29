# -*- coding: utf-8 -*-
"""2026-09-29 实验一结果可视化：SER/PER 曲线（数据源=exp1_run.log，与 JSON 同源）。"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHAINS = ["PLAIN", "OLD-A", "SAVAUX", "TRIMMER", "UNICHIRP", "NEW-0", "BCJR"]
COLOR = {"PLAIN": "#444444", "OLD-A": "#7b52ab", "SAVAUX": "#1f6fd6",
         "TRIMMER": "#2a9d4e", "UNICHIRP": "#9a9a9a", "NEW-0": "#e08214",
         "BCJR": "#d62728"}
MARK = {"PLAIN": "v", "OLD-A": "D", "SAVAUX": "s", "TRIMMER": "P",
        "UNICHIRP": "x", "NEW-0": "^", "BCJR": "o"}
LABEL = {"PLAIN": "PLAIN 标准硬链", "OLD-A": "OLD-A 部署裸抽取",
         "SAVAUX": "Savaux (TIM'22)", "TRIMMER": "LoRaTrimmer (MobiCom'24)",
         "UNICHIRP": "UniChirp", "NEW-0": "NEW-0 (BCJR消融)",
         "BCJR": "BCJR (我们)"}

LOG = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_decode_only_20260929\exp1_run.log"
OUT = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_decode_only_20260929"

# ---- 解析 log：每行 [native] / [SNR +xdB] 提取实测SNR与各链 SER/PER ----
series = {c: ([], [], []) for c in CHAINS}  # snr, ser, per
pat_head = re.compile(r"^\[(native|SNR\s*([+\-]?\d+)dB)\]")
pat_snr = re.compile(r"([+\-]?\d+\.\d)\s*dB\s*\|")
pat_pair = re.compile(r"([\w\-]+)\s*SER\s*([\d.]+)\s*PER\s*([\d.]+)")
for line in open(LOG, encoding="utf-8"):
    m = pat_head.match(line)
    if not m:
        continue
    if m.group(1) == "native":
        snr = float(re.search(r"中位\s*([+\-]?\d+\.\d)", line).group(1))
    else:
        snr = float(pat_snr.search(line).group(1))
    for name, ser, per in pat_pair.findall(line.split("|", 1)[1]):
        series[name][0].append(snr)
        series[name][1].append(float(ser))
        series[name][2].append(float(per))

for c in CHAINS:
    order = sorted(range(len(series[c][0])), key=lambda i: -series[c][0][i])
    series[c] = tuple(np.array([series[c][k][i] for i in order]) for k in range(3))
    assert len(series[c][0]) >= 28, (c, len(series[c][0]))


def style_axes(ax, title, ylabel):
    ax.axvspan(-26.5, -22.5, color="#1f6fd6", alpha=0.07)
    ax.axvline(-22.5, color="#1f6fd6", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("加噪后整包 SNR（dB，全500kHz带实测）")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.invert_xaxis()


# ---- 图1：SER 全链 ----
fig, ax = plt.subplots(figsize=(9.5, 5.8))
for c in CHAINS:
    xs, ys, _ = series[c]
    ax.semilogy(xs, np.maximum(ys, 1e-3), color=COLOR[c], marker=MARK[c], ms=4,
                lw=2.2 if c == "BCJR" else 1.3,
                ls="-" if c in ("BCJR", "SAVAUX", "NEW-0") else "--",
                label=LABEL[c], alpha=1.0 if c in ("BCJR", "SAVAUX") else 0.75)
ax.set_ylim(0.02, 1.1)
style_axes(ax, "实验一（同步给定，只比 demod+bin selection）：SER vs 整包SNR\n"
               "真实OTA 28帧×3种子 | 纯AWGN | 蓝区=Savaux反超段(≤−22.5dB)", "SER（对GT符号）")
ax.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.9)
ax.annotate("BCJR 全程最低（native~−22dB）", xy=(-20, 0.038), xytext=(-13.5, 0.055),
            fontsize=9, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
fig.tight_layout()
fig.savefig(OUT + r"\ser_curves.png", dpi=150)
plt.close(fig)

# ---- 图2：PER 全链 ----
fig, ax = plt.subplots(figsize=(9.5, 5.8))
for c in CHAINS:
    xs, _, ys = series[c]
    ax.semilogy(xs, np.minimum(np.maximum(ys, 0.05), 1.0), color=COLOR[c], marker=MARK[c],
                ms=4, lw=2.2 if c == "BCJR" else 1.3,
                ls="-" if c in ("BCJR", "SAVAUX", "NEW-0") else "--",
                label=LABEL[c], alpha=1.0 if c in ("BCJR", "SAVAUX") else 0.75)
ax.set_ylim(0.05, 1.15)
ax.set_yticks([0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
ax.set_yticklabels(["5%", "10%", "20%", "40%", "70%", "100%"])
style_axes(ax, "实验一：PER vs 整包SNR（整包CRC16未过率；无MIC，固定测试图案非LoRaWAN帧）\n"
               "真实OTA 28帧×3种子 | 纯AWGN", "PER")
ax.legend(loc="lower right", fontsize=8.5, ncol=2, framealpha=0.9)
ax.annotate("PLAIN/OLD-A 于 −19dB PER 打满", xy=(-19, 1.0), xytext=(-15, 0.5),
            fontsize=9, color="#444444", arrowprops=dict(arrowstyle="->", color="#444444"))
fig.tight_layout()
fig.savefig(OUT + r"\per_curves.png", dpi=150)
plt.close(fig)

# ---- 图3：headline——BCJR vs Savaux vs NEW-0（SER） ----
fig, ax = plt.subplots(figsize=(9.0, 5.4))
for c in ["BCJR", "SAVAUX", "NEW-0"]:
    xs, ys, _ = series[c]
    ax.semilogy(xs, np.maximum(ys, 1e-3), color=COLOR[c], marker=MARK[c], ms=5, lw=2.4,
                label=LABEL[c])
ax.set_ylim(0.02, 1.0)
style_axes(ax, "Headline：κ格边际化的价值与相干MRC的深端反超（SER）\n"
               "NEW-0→BCJR=κ格后验混合增益；≤−22.5dB Savaux 相干合并兑现", "SER（对GT符号）")
ax.annotate("κ格增益（消融差）", xy=(-21.6, 0.082), xytext=(-17.5, 0.16),
            fontsize=9.5, color="#e08214", arrowprops=dict(arrowstyle="->", color="#e08214"))
ax.annotate("交叉 −22~−23dB：\nSavaux 深端反超\n（相干MRC ~6dB 兑现）",
            xy=(-23.2, 0.115), xytext=(-16.8, 0.42),
            fontsize=9.5, color="#1f6fd6", arrowprops=dict(arrowstyle="->", color="#1f6fd6"))
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT + r"\headline_ser.png", dpi=150)
plt.close(fig)
print("3 figures saved")

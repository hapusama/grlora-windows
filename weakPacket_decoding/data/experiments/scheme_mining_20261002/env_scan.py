# -*- coding: utf-8 -*-
"""A3 环境扫描（只读 IQ）：PSD 平坦度 / DC 与本振泄漏 / 镜频 / 邻道 / 量化纹理 / 削顶。

数据：SF10 默认帧集 0_0_0_10_14_32.bin（fs=500k, BW125, OS4）
     + SF11 lab1 走廊集 1_0_8_11_2_16.bin（fs=500k, BW125, OS4）。
全部为 np.memmap 只读，不落任何 IQ 文件。
"""
import numpy as np

SF10 = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_32.bin"
SF11 = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\lab1_sf11_TP2\1_0_8_11_2_16.bin"


def psd_scan(path, tag, fs=500000.0, nseg=24, seg=1 << 19):
    iq = np.memmap(path, dtype=np.complex64, mode="r")
    m = len(iq)
    print("== %s  %d samples (%.1f s @ %.0f kHz)" % (tag, m, m / fs, fs / 1e3))
    # --- 幅度统计 / 量化纹理 / 削顶 ---
    a = np.abs(iq[: 40 << 20])
    print("  |x|: max=%.4f  P99.99=%.4f  rms=%.4f  crest=%.1f dB"
          % (a.max(), np.percentile(a, 99.99), np.sqrt(np.mean(a ** 2)),
             20 * np.log10(a.max() / (np.sqrt(np.mean(a ** 2)) + 1e-12))))
    q = np.asarray(iq.real[: 1 << 22])
    lv = np.unique(np.round(q[np.abs(q) < 0.02] * 1e6))
    print("  量化纹理: |I|<0.02 区间 unique(1e6*I) 个数 = %d (float 连续≈海量)" % len(lv))
    # --- Welch PSD ---
    win = np.hanning(seg)
    P = np.zeros(seg)
    rng = np.random.default_rng(7)
    for _ in range(nseg):
        o = rng.integers(0, m - seg)
        x = np.asarray(iq[o:o + seg], dtype=np.complex128)
        P += np.abs(np.fft.fftshift(np.fft.fft(x * win))) ** 2
    P /= nseg * (win ** 2).sum()
    f = np.fft.fftshift(np.fft.fftfreq(seg)) * fs
    ib = np.abs(f) < 62.5e3           # LoRa 带内
    ob = (np.abs(f) > 70e3) & (np.abs(f) < 240e3)  # 带外（他们的 N0 口径）
    n_ib = float(np.median(P[ib]))
    n_ob = float(np.median(P[ob]))
    # 带内平坦度（1kHz 粗粒度）
    kf = np.array_split(np.where(ib)[0], 25)
    flat = np.array([np.median(P[k]) for k in kf])
    print("  噪声底: 带内中位 %.3e, 带外(70-240k)中位 %.3e, 比值 %.2f dB"
          % (n_ib, n_ob, 10 * np.log10(n_ib / n_ob)))
    print("  带内平坦度: min/max = %.2f dB, std = %.2f dB"
          % (10 * np.log10(flat.min() / flat.max()),
             10 * np.log10(flat.std() / flat.mean() + 1e-30)))
    # 尖峰扫描（超过局部中位 12dB 的窄峰）
    med = np.median(P)
    sm = np.convolve(P, np.ones(301) / 301, "same")
    pk = P > med * 16
    idx = np.where(pk)[0]
    idx = idx[(f[idx] > -240e3) & (f[idx] < 240e3)]
    # 合并相邻
    groups = []
    for i in idx:
        if groups and i - groups[-1][-1] <= 3:
            groups[-1].append(i)
        else:
            groups.append([i])
    print("  谱尖峰(>本地+12dB) 个数=%d:" % len(groups))
    for g in groups[:12]:
        j = g[np.argmax(P[g])]
        print("    f=%+8.0f Hz  %+.1f dB vs 带内中位" % (f[j], 10 * np.log10(P[j] / n_ib)))
    # DC（本振泄漏）
    j0 = np.argmin(np.abs(f))
    print("  DC(0Hz) 泄漏: %+.1f dB vs 带内中位" % (10 * np.log10(P[j0] / n_ib)))
    # 邻道
    for tagf in (-250e3, -125e3, 125e3, 250e3):
        sel = np.where((np.abs(f - tagf) < 10e3) & ob)[0]
        if len(sel):
            print("  %+.0fkHz 邻道: %+.1f dB vs 带内中位"
                  % (tagf / 1e3, 10 * np.log10(np.median(P[sel]) / n_ib)))
    del iq
    return


psd_scan(SF10, "SF10 0_0_0_10_14_32")
psd_scan(SF11, "SF11 lab1 1_0_8_11_2_16")

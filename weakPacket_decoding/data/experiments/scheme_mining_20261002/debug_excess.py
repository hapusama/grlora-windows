# -*- coding: utf-8 -*-
"""A3 调试：前导相干平均谱形 excess=47% 的来源定位。

1) 合成自测：本项目 upchirp 公式 + AWGN → 同管道 → excess 应≈0；
2) 真实帧：打印峰±20 bin 的相干平均幅度 / Dirichlet 拟合值 / 单 upchirp 谱；
3) 检查 excess 是否随 P（8/16/32）变化、随 SNR 变化。
"""
import numpy as np
import csv
import sys

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.chirp import build_upchirp

N = 1024
NF = 4096


def dq(nu, N):
    num = np.sin(np.pi * nu)
    den = np.sin(np.pi * nu / N)
    return np.abs(np.where(np.abs(den) < 1e-14, 1e-14, num /
                  np.where(np.abs(den) < 1e-14, 1e-14, den)))


def fit_excess(B, pkbin, N, halfw=16, verbose=False):
    d = (np.arange(N) - pkbin + N // 2) % N - N // 2
    sel = np.abs(d) <= halfw
    dd = d[sel].astype(float)
    best = None
    for nu in np.linspace(pkbin - 0.5, pkbin + 0.5, 101):
        g = dq(nu - dd, N) if abs(nu) > 8 else dq(nu - dd, N)
        g /= np.linalg.norm(g)
        c = float(np.dot(B[sel], g))
        res = float(np.sum((B[sel] - c * g) ** 2))
        if best is None or res < best[0]:
            best = (res, nu, c)
    _, nu, c = best
    g1 = c * dq(nu - dd, N) / np.linalg.norm(dq(nu - dd, N))
    if verbose:
        print("  off  meas   fit  ratio")
        for i, m in enumerate(dd.astype(int)):
            print("  %+4d %.3f %.3f %.2f" % (m, B[sel][i], g1[i], B[sel][i] / (g1[i] + 1e-12)))
    return float(np.sum(np.maximum(B[sel] - g1, 0)) / np.sum(g1)), nu


# ---------- 1) 合成自测 ----------
rng = np.random.default_rng(3)
up4 = build_upchirp(10, 0, os_factor=4)
S = []
nu_true = 300.3
for k in range(8):
    ph = rng.uniform(0, 2 * np.pi)
    n = np.arange(NF)
    x = np.exp(2j * np.pi * 0.5 * (n ** 2) / NF / 4 + 1j * ph)  # 基带连续 chirp（频偏0）
    # 直接用项目 upchirp OS4 作信号（symbol 0 上行 chirp）+ CFO tone 偏移 nu
    sig = up4 * np.exp(2j * np.pi * (nu_true - 512) * n / NF + 1j * ph)
    sig = sig + (rng.standard_normal(NF) + 1j * rng.standard_normal(NF)) * 0.05
    S.append(np.fft.fft(sig[::4] * np.conj(build_upchirp(10, 0, os_factor=1))))
S = np.stack(S)
pk = int(np.bincount(np.argmax(np.abs(S), axis=1)).argmax())
al = S / (S[:, pk] / np.abs(S[:, pk]))[:, None]
B = np.abs(al.mean(axis=0))
ex, nu = fit_excess(B, pk, N)
print("合成自测: 峰=%d 拟合ν=%.2f(真%.2f) excess=%.4f  ← 管道自洽性" % (pk, nu, nu_true, ex))

# ---------- 2) 真实帧 ----------
bp = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin"
cp = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
      r"\header_first\0_0_0_10_14_16_header_first_frames.csv")
iq = np.memmap(bp, dtype=np.complex64, mode="r")
ref1 = np.conj(build_upchirp(10, 0, os_factor=1))
rows = [r for r in csv.DictReader(open(cp, encoding="utf-8"))
        if r.get("header_valid") == "1"]
r0 = rows[1]
hs = int(r0["header_start_sample"])
cfo = int(r0["source_grlora_cfo_int"]) + float(r0["source_grlora_cfo_frac"])
P = 16
starts = [hs - int((4.25 + P - k) * NF) + NF // 4 for k in range(P)]


def spec(st):
    seg = np.asarray(iq[st:st + NF], dtype=np.complex128)
    seg = seg * np.exp(-2j * np.pi * cfo * np.arange(NF) / NF)
    return np.fft.fft(seg[::4] * ref1)


S = np.stack([spec(st) for st in starts])
pks = np.argmax(np.abs(S), axis=1)
pk = int(np.bincount(pks).argmax())
print("\n真实帧: 各 upchirp 峰 bin =", pks, " 主峰=", pk)
al = S / (S[:, pk] / np.abs(S[:, pk]))[:, None]
B = np.abs(al.mean(axis=0))
ex, nu = fit_excess(B, pk, N, verbose=True)
print("相干平均 excess=%.3f ν=%.2f" % (ex, nu))
# 单 upchirp（不相干）
ex1, nu1 = fit_excess(np.abs(S[5]), int(pks[5]), N)
print("单 upchirp excess=%.3f  (噪底占位)" % ex1)
# 量化自测：把合成信号按 1/128 格点量化再看 excess
for lsb in (2.2, 1.4, 1.0):
    rng2 = np.random.default_rng(3)
    Sq = []
    for k in range(8):
        ph = rng2.uniform(0, 2 * np.pi)
        n = np.arange(NF)
        sig = up4 * np.exp(2j * np.pi * (nu_true - 512) * n / NF + 1j * ph)
        sig = sig / np.abs(sig).max() * lsb
        sig = sig + (rng2.standard_normal(NF) + 1j * rng2.standard_normal(NF)) * 0.3
        sigq = np.round(sig * 128) / 128          # 1/128 格点量化
        Sq.append(np.fft.fft(sigq[::4] * ref1))
    Sq = np.stack(Sq)
    pkq = int(np.bincount(np.argmax(np.abs(Sq), axis=1)).argmax())
    alq = Sq / (Sq[:, pkq] / np.abs(Sq[:, pkq]))[:, None]
    exq, nuq = fit_excess(np.abs(alq.mean(axis=0)), pkq, N)
    print("量化合成(LSB峰值=%.1f, SNR≈真实): excess=%.3f" % (lsb, exq))

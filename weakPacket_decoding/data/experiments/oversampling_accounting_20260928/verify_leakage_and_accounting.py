# -*- coding: utf-8 -*-
r"""
2026-09-28 验证脚本：分数 bin 泄漏结构 + 过采样副本能量记账

Part A（无噪声，结构验证）
  A1  分数 bin 单频音 -> 相邻双峰幅值比与相位差 vs 理论
      理论: |X[m+1]|/|X[m]| = sin(pi*k/N)/sin(pi*(1-k)//N),  dphi = -pi*(N-1)/N
  A2  preamble 整数 chip 偏移 -> 单峰搬至 N-d，其余 bin 数学上恰为 0
  A3  数据符号骑缝（前后符号不同）-> 双峰在 -(d+s) 与 -(d+s')，
      幅值比 (N-d):d，相位差 pi*(s-s')*(2d+s+s')/N

Part B（AWGN，SF10，os=4，SER 仿真）
  B1  per-sample SNR 扫描，6 种接收机：
      R1  oracle 1x      : 对齐到 os 采样网格（残余 <= 0.125 chip）
      R2  chip-grid 1x   : 对齐到 chip 网格（残余 kappa ~ U(-0.5,0.5]，量化税）
      R3  4-phase avg    : 4 个过采样相位的功率谱平均（"过采样副本叠加"）
      R3b 4-phase best   : 4 相位中按自身峰能量取最大
      R4  triple-bin sum : chip 网格 + 相邻三 bin 能量和
      R5  known-kappa    : chip 网格 + 已知 kappa 的分数 DFT（相干，oracle）
      R6  4 indep copies : 4 份独立噪声副本功率平均（对齐，正对照，应 +6.02 dB）
  B2  固定 SNR，扫 kappa：R2/R4/R5 的 SER 对 kappa 的敏感曲线

口径: per-sample SNR 按芯片率采样点计；Es/N0 = per-sample SNR + 10log10(N)。
只依赖 numpy。
"""
import os
import json
import numpy as np

N = 1024            # SF10
OS = 4              # 过采样倍数
M = N * OS
EXT = 8             # 两侧保护样本数（os 域）
RNG = np.random.default_rng(20260928)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = {}


def sym_wave(u, s):
    r"""canonical LoRa 符号：基带 chirp + 符号值 s 的频移，u 单位 chip（可为任意实数）。

    x(t) = exp(j2*pi*(u^2/(2N) - s*u/N))，无 mod、无预混叠：
    频率回绕在抽取时自然产生（物理正确）。符号边界处相位连续（u=0 与 u=N 处值相等）。
    """
    return np.exp(1j * 2 * np.pi * (u ** 2 / (2 * N) - s * u / N))


DOWN_CHIP = np.exp(-1j * np.pi * np.arange(N) ** 2 / N)   # chip 率去斜参考
T_EXT = (np.arange(-EXT, M + EXT)) / OS                   # 扩展窗口的 chip 时刻


def make_symbol_os(s, t0):
    """OS 率生成一个符号（含两侧保护），符号值 s，真实起点 t0（chips，含分数）。"""
    return sym_wave(T_EXT - t0, s)


def decimate_fft(x_ext, m0):
    """从 OS 流的 m0 样本起按 OS 抽取 N 点，去斜 + FFT。"""
    idx = m0 + EXT + np.arange(N) * OS
    return np.fft.fft(x_ext[idx] * DOWN_CHIP)


def decimate_fft_batch(x_ext, m0_arr):
    B = len(m0_arr)
    idx = m0_arr[:, None] + EXT + np.arange(N)[None, :] * OS
    y = x_ext[np.arange(B)[:, None], idx] * DOWN_CHIP[None, :]
    return np.fft.fft(y, axis=1)


# ---------------- Part A ----------------
def part_a():
    print("=" * 72)
    print("Part A: 泄漏结构（无噪声）")
    print("=" * 72)

    # A1 纯分数 bin 单频音
    print("\n[A1] 分数 bin 单频音：相邻双峰比与相位差")
    print("  kappa | |X[m+1]|/|X[m]| 实测/理论 | 相位差 实测/理论(deg)")
    a1 = []
    for kap in (0.1, 0.25, 0.3, 0.5):
        n = np.arange(N)
        x = np.exp(1j * 2 * np.pi * (300 + kap) * n / N)
        X = np.fft.fft(x)
        r_meas = abs(X[301]) / abs(X[300])
        r_theo = np.sin(np.pi * kap / N) / np.sin(np.pi * (1 - kap) / N)
        dphi = np.angle(X[301] / X[300])
        dtheo = -np.pi * (N - 1) / N
        row = dict(kappa=kap, ratio=float(r_meas), ratio_theory=float(r_theo),
                   dphi_deg=float(np.degrees(dphi)), dphi_theory_deg=float(np.degrees(dtheo)))
        a1.append(row)
        print("  {:5.2f} | {:8.5f} / {:8.5f} | {:8.3f} / {:8.3f}".format(
            kap, r_meas, r_theo, np.degrees(dphi), np.degrees(dtheo)))

    # A2 preamble 整数 chip 偏移
    print("\n[A2] preamble 整数 chip 偏移 d=137：单峰搬至 N-d，其余 bin 恒等于 0")
    d = 137
    x = make_symbol_os(0, float(d))
    X = decimate_fft(x, 0)
    mag = np.abs(X)
    peak = int(np.argmax(mag))
    others = mag.copy()
    others[peak] = 0
    print("  峰位 = {} (期望 {})，峰幅 = {:.3f}*N".format(peak, (N - d) % N, mag[peak] / N))
    print("  其余 bin 最大幅度/N = {:.3e}（理论上恰为 0）".format(others.max() / N))
    a2 = dict(d=d, peak=peak, expected_peak=(N - d) % N,
              peak_amp_over_N=float(mag[peak] / N),
              max_other_over_N=float(others.max() / N))

    # A3 数据符号骑缝
    print("\n[A3] 骑缝：前符号 s'=37，当前 s=100，偏移 d=200")
    s_cur, s_prev, d3 = 100, 37, 200
    t0 = float(d3)
    seg_prev = sym_wave(T_EXT - t0 + N, s_prev)   # 前符号的本地时间 u = t - (t0-N)
    seg_cur = sym_wave(T_EXT - t0, s_cur)
    x = np.where(T_EXT < t0, seg_prev, seg_cur)
    X = decimate_fft(x, 0)
    k1 = (-(d3 + s_cur)) % N
    k2 = (-(d3 + s_prev)) % N
    ratio_meas = abs(X[k1]) / abs(X[k2])
    ratio_theo = (N - d3) / d3
    dphi_meas = np.angle(X[k1] / X[k2])
    # canonical 相位公式: 相位常数 = 2pi*(phi^2/(2N) - s*phi/N), phi = -(d+s)
    ph = lambda dd, ss: 2 * np.pi * (dd ** 2 / 2 + 2 * dd * ss + 1.5 * ss ** 2) / N
    dphi_theo = ph(d3, s_cur) - ph(d3, s_prev)
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    print("  峰1 bin = {} (期望 {})，峰2 bin = {} (期望 {})".format(
        int(np.argmax(np.abs(X))), k1, k2, k2))
    print("  幅值比 实测 {:.4f} / 理论 {:.4f}".format(ratio_meas, ratio_theo))
    print("  相位差 实测 {:7.3f} / 理论 {:7.3f} (deg)".format(
        np.degrees(dphi_meas), np.degrees(wrap(dphi_theo))))
    # 峰1 是否是全局最大
    mag = np.abs(X)
    top = np.argsort(mag)[::-1][:4]
    print("  幅度 Top-4 bins: {} (值/N: {})".format(
        [int(b) for b in top], ["{:.3f}".format(mag[b] / (N - d3)) for b in top]))
    a3 = dict(s_cur=s_cur, s_prev=s_prev, d=d3, k1=k1, k2=k2,
              ratio=float(ratio_meas), ratio_theory=float(ratio_theo),
              dphi_deg=float(np.degrees(dphi_meas)),
              dphi_theory_deg=float(np.degrees(wrap(dphi_theo))),
              top_bins=[int(b) for b in top])

    RESULTS["A1"] = a1
    RESULTS["A2"] = a2
    RESULTS["A3"] = a3


# ---------------- Part B ----------------
def run_chunk(s_vals, t0_vals, sigma2, receivers):
    """生成一批 OS 符号 + 噪声，返回各接收机的判决 bin。"""
    B = len(s_vals)
    base = upchirv_batch(s_vals, t0_vals)
    noise = (RNG.standard_normal((B, len(T_EXT))) +
             1j * RNG.standard_normal((B, len(T_EXT)))) * np.sqrt(sigma2 / 2)
    x = base + noise
    exp_bin = (-s_vals) % N

    out = {}
    m_oracle = np.round(t0_vals * OS).astype(int)
    m_grid = np.round(t0_vals).astype(int) * OS

    if "R1" in receivers:
        Y = decimate_fft_batch(x, m_oracle)
        out["R1"] = np.argmax(np.abs(Y), axis=1)
    if "R2" in receivers:
        Y = decimate_fft_batch(x, m_grid)
        out["R2"] = np.argmax(np.abs(Y), axis=1)
        out["_Y2"] = Y
    if "R3" in receivers:
        T = np.zeros((B, N))
        cand_e, cand_s = [], []
        m0_floor = np.floor(t0_vals * OS).astype(int)
        for p in range(OS):
            Yp = decimate_fft_batch(x, m0_floor + p)
            T += np.abs(Yp) ** 2
            cand_e.append(np.max(np.abs(Yp) ** 2, axis=1))
            cand_s.append(np.argmax(np.abs(Yp), axis=1))
        if "R3b" in receivers:
            cand_e = np.array(cand_e)
            cand_s = np.array(cand_s)
            sel = np.argmax(cand_e, axis=0)
            out["R3b"] = cand_s[sel, np.arange(B)]
        out["R3"] = np.argmax(T, axis=1)
    if "R4" in receivers:
        Y = out.get("_Y2", decimate_fft_batch(x, m_grid))
        P = np.abs(Y) ** 2
        T = P + np.roll(P, 1, axis=1) + np.roll(P, -1, axis=1)
        out["R4"] = np.argmax(T, axis=1)
    if "R5" in receivers:
        idx = m_grid[:, None] + EXT + np.arange(N)[None, :] * OS
        y = x[np.arange(B)[:, None], idx] * DOWN_CHIP[None, :]
        k = np.arange(N)[None, :]
        kap = (t0_vals - np.round(t0_vals))[:, None]
        y = y * np.exp(1j * 2 * np.pi * kap * k / N)  # 旋转 +kappa，把音搬回整数 bin
        Y = np.fft.fft(y, axis=1)
        out["R5"] = np.argmax(np.abs(Y), axis=1)
    if "R6" in receivers:
        T = np.zeros(B, dtype=np.float64)
        pass
    # R6 需要独立噪声副本，单独处理
    return out, exp_bin, x


def upchirv_batch(s_vals, t0_vals):
    return sym_wave(T_EXT[None, :] - t0_vals[:, None], s_vals[:, None])


def run_r6(s_vals, t0_vals, sigma2, n_copies=4):
    B = len(s_vals)
    base = upchirv_batch(s_vals, t0_vals)
    m_oracle = np.round(t0_vals * OS).astype(int)
    T = np.zeros((B, N))
    for _ in range(n_copies):
        noise = (RNG.standard_normal((B, len(T_EXT))) +
                 1j * RNG.standard_normal((B, len(T_EXT)))) * np.sqrt(sigma2 / 2)
        Y = decimate_fft_batch(base + noise, m_oracle)
        T += np.abs(Y) ** 2
    return np.argmax(T, axis=1), (-s_vals) % N


def ser_of(dec, exp):
    return float(np.mean(dec != exp))


def part_b1():
    print("\n" + "=" * 72)
    print("Part B1: SER vs per-sample SNR（SF10, os=4, kappa~U(-0.5,0.5)）")
    print("Es/N0 = per-sample + 30.1 dB")
    print("=" * 72)
    n_sym = 1500
    chunk = 250
    snrs = [-24, -22, -20, -18, -16, -14]
    recs = ["R1", "R2", "R3", "R3b", "R4", "R5"]
    table = {}
    for snr_db in snrs:
        sigma2 = 10 ** (-snr_db / 10.0)
        err = {r: 0 for r in recs}
        err["R6"] = 0
        done = 0
        while done < n_sym:
            b = min(chunk, n_sym - done)
            s_vals = RNG.integers(0, N, b)
            t0_vals = RNG.uniform(-0.5, 0.5, b)  # 整数部分已对齐，只剩分数
            out, exp, _ = run_chunk(s_vals, t0_vals, sigma2, recs)
            for r in recs:
                err[r] += int(np.sum(out[r] != exp))
            dec6, exp6 = run_r6(s_vals, t0_vals, sigma2)
            err["R6"] += int(np.sum(dec6 != exp6))
            done += b
        row = {r: err[r] / n_sym for r in recs}
        row["R6"] = err["R6"] / n_sym
        table[snr_db] = row
        line = "SNR {:>4} dB | ".format(snr_db) + " ".join(
            "{} {:6.4f}".format(r, row[r]) for r in ["R1", "R2", "R3", "R3b", "R4", "R5", "R6"])
        print(line)
    RESULTS["B1"] = table


def part_b2():
    print("\n" + "=" * 72)
    print("Part B2: 固定 per-sample SNR = -20 dB，扫 kappa（SER 敏感曲线）")
    print("=" * 72)
    n_sym = 1500
    chunk = 250
    snr_db = -20.0
    sigma2 = 10 ** (-snr_db / 10.0)
    kaps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
    recs = ["R2", "R3", "R4", "R5"]
    table = {}
    print("  kappa | " + " ".join("{:>8}".format(r) for r in recs))
    for kap in kaps:
        err = {r: 0 for r in recs}
        done = 0
        while done < n_sym:
            b = min(chunk, n_sym - done)
            s_vals = RNG.integers(0, N, b)
            t0_vals = np.full(b, kap)
            out, exp, _ = run_chunk(s_vals, t0_vals, sigma2, recs)
            for r in recs:
                err[r] += int(np.sum(out[r] != exp))
            done += b
        row = {r: err[r] / n_sym for r in recs}
        table[kap] = row
        print("  {:5.2f} | ".format(kap) + " ".join("{:8.4f}".format(row[r]) for r in recs))
    RESULTS["B2"] = table


def main():
    part_a()
    part_b1()
    part_b2()
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=1, default=float)
    print("\nsaved -> results.json")


if __name__ == "__main__":
    main()

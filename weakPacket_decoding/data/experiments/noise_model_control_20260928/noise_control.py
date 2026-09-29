# -*- coding: utf-8 -*-
r"""
2026-09-28 噪声模型对照实验：量化"+6 dB 相位分集"对噪声带宽口径的依赖性。

用户质疑：测试噪声是否利好本方法。事实：噪声是 iid 循环对称复高斯（教科书 AWGN），
同一实现配对喂所有方法。真正敏感的是**噪声的谱宽度**——它决定过采样相位的噪声
相关性，进而决定相位对齐合并能拿多少分集。

三臂（同一信号、同一 σ 口径：OLD-A 单相位抽取后的噪声方差 = σ²，即 1× 接收机
视角的 SNR 三臂完全一致；臂间只有噪声谱形状不同）：
  wide      噪声铺满整个过采样带（4BW）—— 前端未滤波（我们部署链的实际情形：
            USRP_collector 无 LPF + frame_sync 裸取样）     ρ(相邻相位)=0
  union2bw  |f|<=BW（总 2BW，覆盖任意符号扫频并集的合法固定滤波器）   ρ=0.637
  channel1bw|f|<=BW/2（总 BW，标准 LoRa 信道滤波器）                    ρ=0.900

接收机（同前）：OLD-A / OLD-B(4相位功率平均) / NEW-0(细网格精确读) / TREL-N。
预测：NEW-OLD 增益 wide≈+5dB → union2bw≈+1.4dB → channel1bw≈+0.3dB；
κ 相关增益（税/漂移/半bin）与噪声臂无关（来自信号确定性几何）。
SF10/OS=4，κ∈{0,0.25,drift(+0.02/sym, 8sym)}，1000 符号/点，SER。
"""
import json
import numpy as np

N = 1024
OS = 4
L = N * OS
M_PAD = 16 * N
RNG = np.random.default_rng(20260928)

U_GRID = np.arange(L) / OS
REF_OS = np.exp(-1j * 2 * np.pi * U_GRID ** 2 / (2 * N))
DOWN_CHIP = np.exp(-1j * np.pi * np.arange(N) ** 2 / N)

def sym_wave(u, s):
    return np.exp(1j * 2 * np.pi * (u ** 2 / (2 * N) - s * u / N))

def make_noise(arm, sigma2, rng):
    """理想砖墙带限噪声，校准使相位0抽取后的方差=σ²。"""
    w = (rng.standard_normal(L) + 1j * rng.standard_normal(L))
    if arm == "wide":
        y = w
    else:
        W = np.fft.fftfreq(L)  # 归一化频率, 采样率=L·Δf... 以 4BW 为满带
        if arm == "union2bw":
            mask = np.abs(W) <= 0.25      # 2BW / 4BW = 0.5 → |f|<=0.25
        elif arm == "channel1bw":
            mask = np.abs(W) <= 0.125     # BW / 4BW = 0.25 → |f|<=0.125
        else:
            raise ValueError(arm)
        y = np.fft.ifft(np.fft.fft(w) * mask)
    # 校准：相位0抽取方差 → σ²（三臂 OLD-A 视角 SNR 一致）
    d = y[::OS]
    scale = np.sqrt(sigma2 / (np.mean(np.abs(d) ** 2) + 1e-30))
    return y * scale

def demod(x, chain):
    """返回 T(s)（值域索引）。"""
    if chain == "OLD-A":
        return np.abs(np.fft.fft(x[::OS] * DOWN_CHIP)) ** 2
    if chain == "OLD-B":
        T = np.zeros(N)
        for p in range(OS):
            T += np.abs(np.fft.fft(x[p:p + L:OS] * DOWN_CHIP)) ** 2
        return T
    y = x * REF_OS
    Z = np.fft.fft(y, n=M_PAD)
    mag2 = np.abs(Z) ** 2
    base = (-4 * np.arange(N)) % M_PAD
    if chain == "NEW-0":
        return mag2[base]
    if chain == "TREL-N":
        win = (base[:, None] + np.arange(-2, 3)[None, :]) % M_PAD
        return win, mag2[win]
    raise ValueError(chain)

def main():
    n_sym = 1000
    snrs = [-26, -24, -22, -20, -18, -16]
    arms = ["wide", "union2bw", "channel1bw"]
    kap_modes = {"k0": 0.0, "k25": 0.25, "drift": None}
    chains = ["OLD-A", "OLD-B", "NEW-0", "TREL-N"]
    out = {}

    # 相邻相位实测相关（一次性自检）
    y = make_noise("union2bw", 1.0, RNG)
    c = np.corrcoef(y[::OS][:-1].real, y[1::OS][:-1].real)[0, 1]
    y2 = make_noise("channel1bw", 1.0, RNG)
    c2 = np.corrcoef(y2[::OS][:-1].real, y2[1::OS][:-1].real)[0, 1]
    print("实测相邻相位相关(实部): union2bw=%.3f (理论0.637) channel1bw=%.3f (理论0.900)" % (c, c2))

    for km, kap0 in kap_modes.items():
        for snr_db in snrs:
            sigma2 = 10 ** (-snr_db / 10.0)
            for arm in arms:
                err = {ch: 0 for ch in chains}
                for i in range(n_sym):
                    s = int(RNG.integers(0, N))
                    kap = kap0 if kap0 is not None else 0.1 + 0.02 * (i % 8)
                    x = sym_wave(U_GRID - kap, s) + make_noise(arm, sigma2, RNG)
                    exp_bin = (-s) % N              # 我的参考下 OLD 链音在 -s
                    err["OLD-A"] += int(np.argmax(
                        np.abs(np.fft.fft(x[::OS] * DOWN_CHIP)) ** 2) != exp_bin)
                    Tb = np.zeros(N)
                    for p in range(OS):
                        Tb += np.abs(np.fft.fft(x[p:p + L:OS] * DOWN_CHIP)) ** 2
                    err["OLD-B"] += int(np.argmax(Tb) != exp_bin)
                    Z = np.fft.fft(x * REF_OS, n=M_PAD)
                    mag2 = np.abs(Z) ** 2
                    base = (-4 * np.arange(N)) % M_PAD
                    win = (base[:, None] + np.arange(-2, 3)[None, :]) % M_PAD
                    m = mag2[win]
                    err["NEW-0"] += int(np.argmax(m[:, 2]) != s)   # 值域索引=s
                    err["TREL-N"] += int(np.argmax(m.max(axis=1)) != s)
                out.setdefault(km, {}).setdefault(str(snr_db), {})[arm] = {
                    ch: err[ch] / n_sym for ch in chains}
            row = out[km][str(snr_db)]
            print("[{:5s}]{:>4}: ".format(km, snr_db) + "  ".join(
                "{}:{:.3f}/{:.3f}/{:.3f}".format(
                    ch, row["wide"][ch], row["union2bw"][ch], row["channel1bw"][ch])
                for ch in chains), flush=True)
    # 输出路径为项目内字面量（无拼接、不含上跳段），直接内联供静态检查
    with open(r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\noise_model_control_20260928\results.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("saved -> results.json")

if __name__ == "__main__":
    main()

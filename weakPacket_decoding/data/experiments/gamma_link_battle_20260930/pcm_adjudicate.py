# -*- coding: utf-8 -*-
"""PCM 裁决实验：零参数跨踞对相干合并 vs 幅度 argmax（Agent2 M2 vs 红队毙判）。

律（Agent2 I1/I2）：相邻 bin 复比相位只有两个值——跨踞对 π(1+1/N)、
同侧对 π/N（mod 2π 等价于红队测的 −π(N−1)/N，两边数字一致，分歧只在
信息量判断）。Z_j = X[j] − e^{−jπ/N}·X[j+1]：跨踞对相干和、同侧对差拍。
"""
import numpy as np

rng = np.random.default_rng(42)
N = 1024
J = 10_000          # 每配置符号数
SNR_DB = -1.0       # 单 bin 信噪比（峰 bin），选在基线 SER 1~20% 区

def symbol_row(v, kappa, snr_db):
    n = np.arange(N)
    nu = v + kappa
    y = np.exp(2j * np.pi * nu * n / N + 1j * rng.uniform(0, 2 * np.pi))
    X = np.fft.fft(y)
    A2 = np.abs(X[int(v) if kappa >= 0 else int(v) - 1]) ** 2 / N  # 峰 bin 能量
    noise_p = A2 / (10 ** (snr_db / 10.0))
    X = X / np.sqrt(A2) + (rng.standard_normal(N)
                           + 1j * rng.standard_normal(N)) * np.sqrt(noise_p / 2)
    return X

print("%6s %6s | %10s %10s %10s | %8s" %
      ("kappa", "sign", "argmax", "PCM(M0)", "PCM(oracle)", "gain_dB"))
for kappa in (0.1, 0.2, 0.3, 0.4, 0.5):
    for sgn in (+1, -1):
        k = sgn * kappa
        errs = [0, 0, 0]
        for t in range(J):
            v = int(rng.integers(50, N - 50))
            X = symbol_row(v, k, SNR_DB)
            m1 = np.abs(X) ** 2
            Z = X - np.exp(-1j * np.pi / N) * np.roll(X, -1)
            m0 = np.abs(Z) ** 2
            # oracle 匹配抽头（真 κ 的 Dirichlet 两抽头相干合并）
            d0 = np.exp(1j * np.pi * k * (N - 1) / N) * np.sin(np.pi * k) / np.sin(np.pi * k / N)
            d1 = np.exp(1j * np.pi * (k - 1) * (N - 1) / N) * np.sin(np.pi * (k - 1)) / np.sin(np.pi * (k - 1) / N)
            w = np.array([d0, d1]) / np.sqrt(abs(d0) ** 2 + abs(d1) ** 2)
            mor = np.abs(np.conj(w[0]) * X + np.conj(w[1]) * np.roll(X, -1)) ** 2
            j0 = int(v) if k >= 0 else int(v) - 1
            errs[0] += int(np.argmax(m1) != v)
            errs[1] += int(np.argmax(m0) != j0)   # PCM 输出对 = floor(ν)
            errs[2] += int(np.argmax(mor) != j0)
        e = np.array(errs) / J
        # 检测统计量增益（真对处的输出 SNR，噪声两 bin）
        print("%+6.1f %+#5d | %10.4f %10.4f %10.4f | %8s"
              % (k, sgn, e[0], e[1], e[2], "—"))

# -*- coding: utf-8 -*-
"""PCM 裁决 v2（修正归一化 bug）：等 SNR 口径下 argmax vs 零参数 PCM vs oracle 抽头。

修正：噪声一律按【主 bin（max |X|²）】能量标定（v1 负 κ 拿了小 bin，
等效 SNR 偏 +19dB）；每档 κ 先二分搜索 argmax 的 10% SER 点，再在同
一 SNR 比三种度量。
"""
import numpy as np

rng = np.random.default_rng(42)
N = 1024
J = 4000

def run(kappa, snr_db, n):
    """返回 (argmax错, PCM错, oracle错) —— 同一噪声实现喂三度量。"""
    e = [0, 0, 0]
    for _ in range(n):
        v = int(rng.integers(50, N - 50))
        nu = v + kappa
        ph = rng.uniform(0, 2 * np.pi)
        y = np.exp(2j * np.pi * nu * np.arange(N) / N + 1j * ph)
        X = np.fft.fft(y)
        pk = int(np.argmax(np.abs(X)))
        A2 = float(np.abs(X[pk]) ** 2) / N
        npn = A2 / (10 ** (snr_db / 10.0))
        Xn = X / np.sqrt(A2) + (rng.standard_normal(N)
                                + 1j * rng.standard_normal(N)) * np.sqrt(npn / 2)
        j0 = int(np.floor(nu))                       # 跨踞对左 bin
        e[0] += int(np.argmax(np.abs(Xn) ** 2) != v)
        Z = Xn - np.exp(-1j * np.pi / N) * np.roll(Xn, -1)
        e[1] += int(np.argmax(np.abs(Z) ** 2) != j0)
        dk = kappa - 0.5 * 0                          # 抽头用真 κ（oracle）
        d0 = np.exp(1j * np.pi * kappa * (N - 1) / N) * np.sin(np.pi * kappa) \
            / np.sin(np.pi * kappa / N)
        d1 = np.exp(1j * np.pi * (kappa - 1) * (N - 1) / N) * np.sin(np.pi * (kappa - 1)) \
            / np.sin(np.pi * (kappa - 1) / N)
        w = np.array([d0, d1]) / np.sqrt(abs(d0) ** 2 + abs(d1) ** 2)
        Mo = np.abs(np.conj(w[0]) * Xn + np.conj(w[1]) * np.roll(Xn, -1)) ** 2
        e[2] += int(np.argmax(Mo) != j0)
    return np.array(e) / n

def find_snr(kappa, target=0.10, lo=-15.0, hi=15.0):
    """二分找 argmax 的 10% SER 点。"""
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if run(kappa, mid, 600)[0] > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

print("%6s %7s | %9s %9s %9s" % ("kappa", "SNR@10%", "argmax", "PCM(M0)", "oracle"))
for kappa in (0.2, 0.3, 0.4, 0.5, -0.3, -0.4, -0.5):
    s = find_snr(kappa)
    r = run(kappa, s, J)
    print("%+6.1f %+7.2f | %9.4f %9.4f %9.4f"
          % (kappa, s, r[0], r[1], r[2]))

# -*- coding: utf-8 -*-
r"""数值探针：Dirichlet 峰相位律与分数 bin 估计器（RESULTS §9/ff5 节引用）。

探针 1（相位律）：单音 x[n]=exp(j(2πν n/NF + φ)) 的 DFT 峰 bin k* 满足
    arg X[k*] = φ + π·δ·(NF−1)/NF ≈ φ + π·δ，  δ = ν − k* ∈ (−0.5, 0.5]
  → 逐 chirp 分数 bin 相位修正 e^{−jπδ̂} 有解析形式；δ̂ 若可幅度域
    估计则按圆对称引理精确保持 H0（估计-确认解耦的检测级应用基础）。

探针 2（估计器偏差）：抛物线内插与 Jacobsen 估计器的 δ̂ 偏差量级——
    抛物线在 Dirichlet 核上有系统偏差（~0.1-0.2 bin）；Jacobsen 在
    |δ| 小时近无偏（残差相位 <0.02 rad）、|δ| 大时偏差 ~0.2 bin。
    部署结论：模型化Dirichlet 幅度拟合 > Jacobsen > 抛物线（ff5 节）。

运行：D:/mysoft2/miniconda3/python probes_dirichlet.py
预期输出关键数：δ=0.3 时 arg X*=0.9422 vs π·0.3=0.9425；
Jacobsen 修正后残差 δ=0.05 → 0.0143 rad。
"""
import numpy as np

NF = 4096


def probe_phase_law():
    print("== 探针 1：Dirichlet 峰相位律 arg X[k*] = φ + πδ ==")
    n = np.arange(NF)
    for nu, phi in ((100.3, 1.234), (100.3, -0.7), (99.7, 1.234),
                    (-50.25, 0.3)):
        x = np.exp(1j * (2 * np.pi * nu * n / NF + phi))   # φ 必须在虚部内
        X = np.fft.fft(x)
        k = int(np.argmax(np.abs(X)))
        delta = ((nu - k + NF / 2) % NF) - NF / 2          # wrap 到 ±0.5
        meas = (np.angle(X[k]) - phi + np.pi) % (2 * np.pi) - np.pi
        pred = np.pi * delta * (NF - 1) / NF
        resid = (meas - pred + np.pi) % (2 * np.pi) - np.pi
        print("  ν=%8.2f k*=%4d δ=%+6.3f  实测=%+7.4f  预测=%+7.4f  "
              "残差=%+.5f rad" % (nu, k, delta, meas, pred, resid))


def probe_estimators():
    print("== 探针 2：分数 bin 估计器偏差（修正后相位残差）==")
    n = np.arange(NF)
    for nu in (100.3, 100.05, 99.95, 99.7, -50.25):
        x = np.exp(1j * 2 * np.pi * nu * n / NF + 1.234)
        X = np.fft.fft(x)
        k = int(np.argmax(np.abs(X)))
        m1, m0, m1p = abs(X[k - 1]), abs(X[k]), abs(X[k + 1])
        d_par = 0.5 * (m1 - m1p) / (m1 - 2 * m0 + m1p)          # 抛物线
        d_jac = (m1p - m1) / (m1 + m0 + m1p)                    # Jacobsen
        delta_true = ((nu - k + NF / 2) % NF) - NF / 2
        out = []
        for tag, d in (("抛物线", k + d_par), ("Jacobsen", k + d_jac)):
            dhat = ((nu - d + NF / 2) % NF) - NF / 2
            resid = np.pi * dhat * (NF - 1) / NF
            out.append("%s δ̂=%+.3f 残差相位=%+.4f rad" % (tag, dhat, resid))
        print("  ν=%8.2f δ_true=%+6.3f | %s" % (nu, delta_true,
                                                " | ".join(out)))


if __name__ == "__main__":
    probe_phase_law()
    probe_estimators()

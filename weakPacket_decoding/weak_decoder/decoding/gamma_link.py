# -*- coding: utf-8 -*-
"""γ-链解调器（Frame-as-One / 统一度量连续工作点，v3 最小实现）。

依据 doc/神招辩论_20260930.md（候选"神招"，实验裁决前不进 COGNITION）：

  - 恒等式（probe1 实测吻合，辩论 §1 有精确证明）：
        γ_i = ∠(V₂V₁*) = π·κ_i   ——与符号值 c 无关（L_f+L_t=N 消去 c）。
    wrap 两段投影的差分相位 = 分数偏移 κ 的**候选盲、逐符号免费测量**。
  - 帧级模型：κ_i = κ₀ + δ·i（SFO 漂移下帧相位二次，一阶 κ 线性漂移）。
    |z_i|=|V₂V₁*| 加权最小二乘（错判符号的 z 是泄漏×泄漏，自然降权）。
  - 统一度量（精确后验均值，非近似；一次谐波以高斯特征函数进入）：
        Λ_i(c) = |V₁|²+|V₂|²+2·Re{V₁*V₂e^{-jπm̂_i}}·e^{-π²s_i²/2}
    s→0 = DeRa 全相干；s→∞ = LoRaTrimmer 非相干。工作点由数据自标定
    （s 来自 WLS 后验方差），不搜、不调参。
  - 与既有组件的关系：V₁/V₂ 投影 = DeRa port v2 的 front/tail 矩阵
    （= LoRaTrimmer 几何）；本模块只加 O(N)/符号 的差分读出 + 一次 2×2
    加权 LS + 衰减因子（辩论 §4-5 复杂度题）。

v3 范围（最小可裁实现）：硬读出（逐符号 argmax Λ 行），与 DERA/TRIM
链同口径；κ 格 trellis 融合与软 LLR 接口留待 v4（Q4-Q5）。
"""
from __future__ import annotations

import numpy as np

from ..baselines.dera.paper_dera_demod import _wrap_split_matrices


class GammaLinkDemodulator:
    """γ-链解调器（decode-only，接口与 DeRaDemodulator.demod_payload 对齐）。"""

    def __init__(self, sf: int, os_factor: int):
        self.sf = int(sf)
        self.os = int(os_factor)
        self.n = 1 << self.sf
        self.nf = self.n * self.os
        self._front, self._tail, self._split = _wrap_split_matrices(
            self.sf, self.os)
        # 诊断量（最后一次 demod_payload 的拟合结果）
        self.last_fit: dict | None = None

    def demod_payload(self, samples: np.ndarray, start_symbol: int,
                      psym: int, pilots: list[tuple[int, float]] | None = None,
                      ) -> np.ndarray:
        """返回 (psym, N) 统一度量行 Λ_i(c)（能量域，argmax 判决）。

        pilots: [(start_sample, x_index), ...] —— 内容已知符号（前导/
        sync）的窗口位置与其在 κ(x)=κ₀+δx 轨迹上的符号时刻（x 相对
        payload 符号 0，可为分数——SFD 4.25 结构使前导网格带分数偏移）。
        已知符号的 γ 测量一并入 WLS（"前导当导频"）：深端逐符 γ 噪声
        大，已知符号的额外测量磨尖 (κ₀,δ) 轨迹；接收机合法知识（同
        UniChirp 带噪前导重训的先例），所有链同一段信号。
        """
        idx = np.arange(int(psym))
        f_mat = np.empty((psym, self.n), dtype=np.complex128)
        t_mat = np.empty((psym, self.n), dtype=np.complex128)
        for k in idx:
            s = (int(start_symbol) + k) * self.nf
            w = np.asarray(samples[s:s + self.nf], dtype=np.complex64)
            if w.size != self.nf:
                raise ValueError(f"symbol {k} window exceeds input")
            f_mat[k] = self._front @ w
            t_mat[k] = self._tail @ w
        noncoh = np.abs(f_mat) ** 2 + np.abs(t_mat) ** 2
        chat = np.argmax(noncoh, axis=1)          # κ 鲁棒非相干初始化
        # 逐符 γ 测量（候选盲：在各符号自己的 argmax 上读）
        z = t_mat[idx, chat] * np.conj(f_mat[idx, chat])
        gamma = np.angle(z)                        # = π·κ mod 2π（真候选处）
        # ⚠️ γ 有本质 π 周期模糊：κ 与 κ+1 同 γ 但相干项反号（e^{-jπκ}
        # 周期 2）。因此 y=γ/π ∈ (−1,1] **不做任何预卷绕**——κ 穿越 ±0.5
        # 时 y 连续（0.49→0.51 无跳变），符号分支由拟合轨迹自洽决定；
        # ĉ=c±1 造成的 ±1 整数外点由下方"绕拟合残差的模 1 卷绕"处理。
        y = gamma / np.pi                          # ∈ (−1,1]
        w_i = np.abs(z).astype(np.float64)

        # 已知符号（前导/sync 导频）的 γ 测量并入
        y_all, w_all = list(y), list(w_i)
        x_all = [float(k) for k in idx]
        for p_start, p_x in (pilots or []):
            s = int(p_start)
            w_p = np.asarray(samples[s:s + self.nf], dtype=np.complex64)
            if w_p.size != self.nf:
                continue
            f_p = self._front @ w_p
            t_p = self._tail @ w_p
            c_p = int(np.argmax(np.abs(f_p) ** 2 + np.abs(t_p) ** 2))
            z_p = t_p[c_p] * np.conj(f_p[c_p])
            y_all.append(float(np.angle(z_p)) / np.pi)
            w_all.append(float(np.abs(z_p)))
            x_all.append(float(p_x))
        n_all = len(y_all)
        w_arr = np.asarray(w_all, dtype=np.float64)
        w_arr /= max(float(w_arr.sum()), 1e-30)    # 归一权重

        # |z| 加权 LS：y(x) ≈ κ₀ + δ·x。初始化 = 加权合角（DeRa ML φ̂0 同款
        # 池化，mod-2 角度直接定符号分支——κ 与 κ+1 在 γ 上同值但相干项
        # 反号，raw-LS 在混合分支帧上会落进镜像分支 κ+1/−δ）；随后两轮
        # 绕拟合残差的模 1 卷绕精化（消 ĉ=c±1 的整数外点）。
        x_arr = np.asarray(x_all, dtype=np.float64)
        y_arr = np.asarray(y_all, dtype=np.float64)
        H = np.stack([np.ones(n_all), x_arr], axis=1)
        Wd = np.diag(w_arr)
        HW = H.T @ Wd

        def _wls(yv):
            try:
                return np.linalg.solve(HW @ H, HW @ yv)
            except np.linalg.LinAlgError:
                return np.array([float(np.average(yv, weights=w_arr)), 0.0])

        beta = np.array([float(np.angle(np.sum(w_arr * np.exp(1j * np.pi
                                                            * y_arr)))
                               / np.pi), 0.0])
        for _ in range(3):
            r = y_arr - H @ beta
            r = ((r + 0.5) % 1.0) - 0.5
            beta = _wls(H @ beta + r)
        m_hat = beta[0] + beta[1] * idx.astype(np.float64)  # payload 逐符 κ̂
        resid = y_arr - H @ beta
        # σ̂²：相对权重口径，自由度按测量数计（归一化权重的 Σw=1，
        # 不能做自由度——v3 曾因此把 σ² 放大到天文数字、ρ 塌到 0）
        dof = max(float(n_all) - 2.0, 1.0)
        sigma2 = float(np.sum(w_arr * resid ** 2) / dof)
        # 逐符后验方差 s_i²（payload 行的杠杆率），加 σ² 本底
        try:
            cov = np.linalg.inv(HW @ H)
            lev_all = np.einsum("ij,jk,ik->i", H, cov, H)
        except np.linalg.LinAlgError:
            lev_all = np.full(n_all, 1.0 / max(n_all, 1))
        lev = lev_all[:psym]
        s2 = sigma2 * (lev + 1.0 / max(n_all, 1))
        rho = np.exp(-np.pi ** 2 * s2 / 2.0)       # 特征函数衰减因子

        # 统一度量行：Λ = |V1|²+|V2|²+2ρ·Re(V1*·V2·e^{−jπm̂})
        rows = noncoh.copy()
        for k in idx:
            theta = np.pi * m_hat[k]
            cross = 2.0 * rho[k] * np.real(
                np.conj(f_mat[k]) * t_mat[k] * np.exp(-1j * theta))
            rows[k] = rows[k] + cross
        self.last_fit = dict(kappa0=float(beta[0]), drift=float(beta[1]),
                             sigma2=sigma2, rho_min=float(rho.min()),
                             rho_mean=float(rho.mean()), n_meas=int(n_all))
        return rows

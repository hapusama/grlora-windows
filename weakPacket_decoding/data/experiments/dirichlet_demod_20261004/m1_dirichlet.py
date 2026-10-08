# -*- coding: utf-8 -*-
"""m1_dirichlet.py — γ 条件化 Dirichlet 相干重组解调器（OURS-Dir）核心模块。

方法依据（doc/物理律挖掘_20261001.md 定律2/4 + doc/相位机制会战_20261002.md §1-2）：

  物理结构（OS=4 解调窗，去斜参考 = conj(upchirp(0))）：
    值 v、分数偏移 κ 的音在 OS=4 去斜后是【分段音】——
      segA（n < 4(N−ν)）：频率 ν/NF；segB：频率 (ν−N)/NF，ν = v+δ+κ。
    抽取到 4 个相位支路（branch p = 样本 n≡p mod 4）后，两段都混叠到
    N 点 bin ν，但 segB 带支路相位翻转 e^{−j2πp/OS}（fold 几何）。
    DeRa 的 V1/V2 整数中心读出 = Dirichlet 部分和亏损（定律4：
    κ=0.3→−1.01dB、κ=0.5→−2.78dB）；本模块在 (v+κ̂) 分数位置用
    【分段长度自适应的 Dirichlet 抽头】做相干重组，恢复满能量。

  核心步（对候选 bin b，帧池化 κ̂ 条件化；fold=TX 整数格 4(N−b)）：
    L_A = N−b（segA OS1 长度），L_B = b，m* = N−b（segB 段起点）
    d_A[m] = D_{L_A}(κ̂−m)                     （segA 模板，段起点 0）
    d_B[m] = e^{j2π(κ̂−m)m*/N}·D_{L_B}(κ̂−m)     （segB 模板，段起点 m*）
    D_L(x) = e^{jπx(L−1)/N}·sin(πxL/N)/sin(πx/N)（闭式 Dirichlet 核）
    C_p^{A/B}(b) = Σ_m conj(d_{A/B}[m])·X_p[(b+m) mod N]
    S(b) = Σ_p e^{−j2π(b+κ̂)p/NF}·(C_p^A + e^{+j2πp/OS}·C_p^B)
    行 = Re(e^{−jφ̂}·S(b))，φ̂ = 帧公共相位 ML（payload 临时符号池化，
    DeRa Stage-2 同款机制；genie 可选作诊断）。

  退化性质：κ̂→0 时 A 核主抽头在 m=0（单抽头 argmax 近似）；|κ̂|→0.5 时
  两 bin 相干和自动出现（PCM 的全参数推广）。

复杂度：每符号 4×N-FFT + 8·N·(2T+1) 复 MAC（T=48 抽头半宽）≈ 0.79M cMAC，
对比 DeRa port 2×N×NF ≈ 8.4M cMAC（~25×）。

接口与 kappa_trellis / paper_dera_demod 对齐：demod_payload 返回 (psym, N)
bin 域行（argmax − δ = 值），judge 侧走 judge_crc_fast 同一判据。
"""
from __future__ import annotations

import numpy as np

from weak_decoder.chirp import build_upchirp


class DirichletDemodulator:
    """γ 条件化 Dirichlet 相干重组解调器（OURS-Dir，decode-only）。"""

    def __init__(self, sf: int, os_factor: int = 4, half_taps: int = 48,
                 mask_aware: bool = True):
        self.sf = int(sf)
        self._mask_aware = bool(mask_aware)
        self.os = int(os_factor)
        self.n = 1 << self.sf
        self.nf = self.n * self.os
        self.T = int(half_taps)
        # 各抽取相位支路的去斜参考（conj(upchirp(0)) 的 p::OS 抽取）
        ref_os = np.conj(build_upchirp(self.sf, symbol_id=0, os_factor=self.os))
        self._ref_p = [ref_os[p::self.os].astype(np.complex128)
                       for p in range(self.os)]
        self._m = np.arange(-self.T, self.T + 1)
        self._b = np.arange(self.n)
        self._idx = (self._b[:, None] + self._m[None, :]) % self.n
        # 带限掩模（与产线 wm() 同款：±0.125+40/NF）——去斜前压制带外
        # 噪声，使 /4 抽取近似临界采样（无掩模时 ±0.5 全带噪声混叠入
        # N 域，每 bin 噪声 ×4，深端歧视性实测差 ~6dB）。
        self._mask = (np.abs(np.fft.fftfreq(self.nf))
                      <= 0.125 + 40.0 / self.nf)
        self._up_os = np.conj(
            build_upchirp(self.sf, symbol_id=0, os_factor=self.os)
        ).astype(np.complex128)
        # 诊断量
        self.last_phi: float = 0.0
        self.last_S: np.ndarray | None = None

    # ---------------- 支路复谱 ----------------
    def branch_spectra(self, samples: np.ndarray, start_symbol: int,
                       psym: int) -> np.ndarray:
        """返回 (psym, os, N) 复谱 X[p_sym, p_branch, bin]。

        每符号：窗 → NF-FFT → 带限掩模 → IFFT → 去斜 → 4 相位抽取 →
        N-FFT（与产线 wm() 同一信号路径，仅复谱不取模）。
        """
        out = np.empty((int(psym), self.os, self.n), dtype=np.complex128)
        for k in range(int(psym)):
            s = (int(start_symbol) + k) * self.nf
            w = np.asarray(samples[s:s + self.nf], dtype=np.complex128)
            if w.size != self.nf:
                raise ValueError(f"symbol {k} window exceeds input")
            z = np.fft.ifft(np.fft.fft(w) * self._mask)
            for p in range(self.os):
                out[k, p] = np.fft.fft(z[p::self.os] * self._ref_p[p])
        return out

    # ---------------- 单符号核统计 ----------------
    def symbol_S(self, Xp: np.ndarray, kappa: float,
                 fold_off: float = 0.0) -> np.ndarray:
        """Xp: (os, N) 单符号支路谱；kappa: 该符号的 κ̂。返回 (N,) 复统计 S(b)。

        fold 几何：OTA 发射机 DDS 在带边卷绕，fold 位置由 TX 整数值决定
        = 4(N−b)（候选 b 的整数格），κ 只平移音频率（ν=b+κ̂）——故
        L_A = N−b（整数）、L_B = b、段 B 起点 m* = N−b，与 DeRa port 的
        split_samples[k]=(N−k)·os 同一几何。fold_off：分数 STO 残差引起的
        fold 帧公共偏移（OS1 样本；标定/诊断用，正常=0）。
        """
        n, os_ = self.n, self.os
        x = float(kappa) - self._m                       # (M,) 核偏移
        nu = self._b + float(kappa)                      # (N,) 假设音位
        LA = np.clip(n - self._b + float(fold_off), 0.0, float(n))
        LB = n - LA
        ms = LA                                          # fold 位置（OS1）
        xa = np.abs(x)
        den = np.where(xa < 1e-9, 1.0, np.sin(np.pi * x / n))

        def _dirich(L: np.ndarray) -> np.ndarray:
            """(N, M) 段 Dirichlet 核 D_L(κ̂−m)。"""
            LL = L[:, None]
            ratio = np.where(xa[None, :] < 1e-9, LL,
                             np.sin(np.pi * x[None, :] * LL / n) / den[None, :])
            ker = np.exp(1j * np.pi * x[None, :] * np.maximum(LL - 1.0, 0.0) / n) * ratio
            ker[L <= 0.0, :] = 0.0
            return ker

        dA = _dirich(LA)
        dB = np.exp(1j * 2.0 * np.pi * np.outer(ms, x) / n) * _dirich(LB)
        # 带限掩模感知：segA 频率 ν/NF（正侧）与 segB (ν−N)/NF（负侧）
        # 超出 ±0.125+40/NF（≈±552 bin）的段已被掩模抹掉——对应核置零，
        # 否则被掩段建模噪声、±1 整数判别塌掉（f27 的 b∈(552,770) 符号
        # 实测 argmax 随机翻 ±1）。全带核（无掩模链）不受影响（全 1）。
        if self._mask_aware:
            bmax = 0.125 + 40.0 / self.nf          # 归一化带边
            keepA = np.abs(nu / self.nf) <= bmax + 1.5 / self.nf
            keepB = np.abs((nu - n) / self.nf) <= bmax + 1.5 / self.nf
            dA = dA * keepA[:, None]
            dB = dB * keepB[:, None]
        conj_dA, conj_dB = np.conj(dA), np.conj(dB)
        S = np.zeros(n, dtype=np.complex128)
        for p in range(os_):
            G = Xp[p][self._idx]                         # (N, M) 环形抽头
            CA = np.sum(conj_dA * G, axis=1)
            CB = np.sum(conj_dB * G, axis=1)
            theta = 2.0 * np.pi * nu * p / self.nf       # 支路相位
            S += np.exp(-1j * theta) * (CA + np.exp(1j * 2.0 * np.pi * p / os_) * CB)
        return S

    # ---------------- 帧级解调 ----------------
    def demod_payload(self, samples: np.ndarray, start_symbol: int, psym: int,
                      kappa_hat=None, readout: str = "abs2",
                      ) -> np.ndarray:
        """返回 (psym, N) bin 域行。

        kappa_hat: (psym,) 逐符号 κ̂（帧池化，所有臂同源同喂）；
                   None 时取 0（单抽头退化）。
        readout: "abs2" = |S|²（默认，OTA 实测逐符号音相位随符号值散布
                 ±π（相位会战 §1-1：帧公共 φ̂ 不存在），幅度读出与 DeRa
                 Eq.23 同类——重组内部全相干，仅最终投影幅度）；
                 "re" = Re(e^{−jφ̂}S)（帧公共相位 ML 池化，合成/诊断用）。
        """
        psym = int(psym)
        if kappa_hat is None:
            kap = np.zeros(psym)
        else:
            kap = np.asarray(kappa_hat, dtype=np.float64)
            if kap.shape != (psym,):
                raise ValueError("kappa_hat must be (psym,)")
        X = self.branch_spectra(samples, start_symbol, psym)
        Smat = np.stack([self.symbol_S(X[k], kap[k]) for k in range(psym)])
        self.last_S = Smat
        if readout == "abs2":
            return np.abs(Smat) ** 2
        init = np.argmax(np.abs(Smat) ** 2, axis=1)
        z = np.sum(Smat[np.arange(psym), init])
        self.last_phi = float(np.angle(z)) if abs(z) > 0 else 0.0
        return np.real(np.exp(-1j * self.last_phi) * Smat)

    # ---------------- 标量候选核统计（κ̂ 估计器用） ----------------
    def S_at(self, Xp: np.ndarray, b: int, kappa: float,
             fold_off: float = 0.0) -> complex:
        """单候选 b、单 κ 假设的核统计（~8·(2T+1) cMAC）。"""
        n, os_ = self.n, self.os
        b = int(b) % n
        x = float(kappa) - self._m
        la = min(max(n - b + float(fold_off), 0.0), float(n))
        lb = n - la
        xa = np.abs(x)
        den = np.where(xa < 1e-9, 1.0, np.sin(np.pi * x / n))

        def _d(L: float) -> np.ndarray:
            if L <= 0.0:
                return np.zeros_like(x, dtype=np.complex128)
            ratio = np.where(xa < 1e-9, L, np.sin(np.pi * x * L / n) / den)
            return np.exp(1j * np.pi * x * max(L - 1.0, 0.0) / n) * ratio

        dA, dB = _d(la), np.exp(1j * 2.0 * np.pi * la * x / n) * _d(lb)
        nu = b + float(kappa)
        s = 0.0 + 0.0j
        for p in range(os_):
            G = Xp[p][(b + self._m) % n]
            CA = np.sum(np.conj(dA) * G)
            CB = np.sum(np.conj(dB) * G)
            s += np.exp(-1j * 2.0 * np.pi * nu * p / self.nf) * (
                CA + np.exp(1j * 2.0 * np.pi * p / os_) * CB)
        return s

    # ---------------- κ 格 + trellis 读出（主力） ----------------
    def _wm_columns(self, samples: np.ndarray, start_symbol: int, psym: int
                    ) -> list[np.ndarray]:
        """产线 wm() 同款 5 列谱（掩模+去斜+4 相位+ROT5，功率平均）。

        列 d 的频移 δ_d=(2−d)/4 把 κ≈δ_d 的音转上整数格——trellis 走格
        的整数判决由它给（实测 tie-break 稳定：native 28/28）；
        Dirichlet 行在同 κ 做满能量重组（本模块的增量件）。
        """
        rot5 = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0)
                                * np.arange(self.n) / self.n)
                         for d in range(5)])
        out = []
        for k in range(int(psym)):
            s = (int(start_symbol) + k) * self.nf
            w = np.asarray(samples[s:s + self.nf], dtype=np.complex128)
            if w.size != self.nf:
                raise ValueError(f"symbol {k} window exceeds input")
            z = np.fft.ifft(np.fft.fft(w) * self._mask)
            y = z * self._up_os
            P = np.empty((5, self.os, self.n))
            for p in range(self.os):
                chip = y[p::self.os]
                inp = (chip[None, :] * rot5[:, :]).reshape(5, self.n)
                P[:, p, :] = np.abs(np.fft.fft(inp, axis=1)) ** 2
            out.append(P.mean(axis=1).T)                # (N, 5)
        return out

    def kappa_pooled(self, samples: np.ndarray, start_symbol: int,
                     psym: int) -> float:
        """帧池化 κ̄：逐符号窄抽头（T=6）argmax κ（3 候选 × 全圆 0.05 格）
        的 |S|² 加权圆均值——常数，无轨迹/解开需求，对 mod-1 卷绕与
        ±1 候选错免疫。"""
        psym = int(psym)
        X = self.branch_spectra(samples, start_symbol, psym)
        kgrid = np.arange(0.0, 1.0, 0.05)
        T0 = self.T
        self.T = 6                       # 窄核估计（旁瓣抑制）
        try:
            ys = np.empty(psym)
            ws = np.empty(psym)
            for k in range(psym):
                p0 = np.sum(np.abs(X[k]) ** 2, axis=0)
                bpk = int(np.argmax(p0))
                best = (-1.0, 0.0)
                for b in ((bpk - 1) % self.n, bpk, (bpk + 1) % self.n):
                    for kg in kgrid:
                        v = abs(self.S_at(X[k], b, float(kg))) ** 2
                        if v > best[0]:
                            best = (v, float(kg))
                ys[k] = best[1] if best[1] <= 0.5 else best[1] - 1.0
                ws[k] = best[0]
        finally:
            self.T = T0
        ws = ws / max(ws.sum(), 1e-30)
        z = np.sum(ws * np.exp(2j * np.pi * ys))
        return float(np.angle(z) / (2 * np.pi)) if abs(z) > 0 else 0.0

    def lattice_grids(self, samples: np.ndarray, start_symbol: int, psym: int,
                      kappa_center: float) -> list[np.ndarray]:
        """(psym) 个 (N, 5) 细格谱：列 j 的 κ = κ̄+(j−2)/4，
        每格 = |S(b, κ_j)|² Dirichlet 相干重组行（TREL 列的升级件）。"""
        psym = int(psym)
        X = self.branch_spectra(samples, start_symbol, psym)
        cols = [float(kappa_center) + (j - 2) / 4.0 for j in range(5)]
        out = []
        for k in range(psym):
            rows = np.empty((self.n, 5))
            for j, kj in enumerate(cols):
                rows[:, j] = np.abs(self.symbol_S(X[k], kj)) ** 2
            out.append(rows)
        return out

    def demod_payload_lattice(self, samples: np.ndarray, start_symbol: int,
                              psym: int, kappa_center: float | None = None,
                              trellis: bool = True,
                              dirichlet_readout: bool = True
                              ) -> np.ndarray:
        """κ 格解调（与 KappaTrellisDemodulator 同型接口）。

        走格 = 产线 wm() 5 列 + emission_prominence + viterbi(radius=1)
        （整数判决/lift/漂移全部由这条已被 native 28/28 验证的路径给）；
        dirichlet_readout=True 时最终行 = 选中列 κ 上的 Dirichlet 相干
        重组 |S|²（满能量，TREL 列的单抽头泄漏读出的升级件），
        False 时行 = TREL 列本身（= TREL-5 复刻，消融对照）。
        kappa_center/trellis 保留给消融（lattice 中心平移/恒中心列）。
        """
        from weak_decoder.decoding.kappa_trellis import trellis as _tr
        psym = int(psym)
        if not dirichlet_readout:
            ms = self._wm_columns(samples, start_symbol, psym)
            if not trellis:
                return np.stack([m[:, 2] for m in ms])
            lam = _tr.emission_prominence(ms)
            path = _tr.viterbi(lam, 1)
            return np.stack([ms[k][:, path[k]] for k in range(psym)])
        ms = self._wm_columns(samples, start_symbol, psym)
        lam = _tr.emission_prominence(ms)
        if trellis:
            path = _tr.viterbi(lam, 1)
        else:
            path = np.full(psym, 2)
        # 选中列 d 的频移 δ_d=(2−d)/4 把 κ_t≈−δ_d 的音转上整数格
        # ⇒ Dirichlet 核心居 b+κ_sel（κ_sel=−δ_d=(d−2)/4）后 argmax
        # 落回同一整数。
        X = self.branch_spectra(samples, start_symbol, psym)
        rows = np.empty((psym, self.n))
        for k in range(psym):
            ksel = (int(path[k]) - 2) / 4.0
            rows[k] = np.abs(self.symbol_S(X[k], ksel)) ** 2
        return rows

    def kappa_mini_track(self, samples: np.ndarray, start_symbol: int,
                         psym: int) -> np.ndarray:
        """帧池化 κ̂ 轨迹（接收机行为，所有臂同源同喂）。

        逐符号：非相干峰 ±1 候选 × κ 细格（步长 0.05）标量核扫描 →
        (b̂_k, κ̂_k, w_k=|S|²)；帧级：|S|² 加权线性 WLS（κ₀+δ·i，
        残差模 1 卷绕 2 轮去 ±1 外点）。

        标定依据（m1_probe）：γ-链 WLS 的 κ̂ 在 OTA 上测的是 CFO+STO
        残差混合（κ_CFO+ε_t/4，帧间 −0.10~−0.52），而抽头核需要的音位
        κ*（fold=0 吸收延迟）实测 ≈ −0.05~−0.10——两者历元不一致
        （物理律 §2-13 的 δ/2 历元差的实测形态），故 κ̂ 用本估计器，
        γ-κ̂ 保留为消融列。
        """
        psym = int(psym)
        X = self.branch_spectra(samples, start_symbol, psym)
        # 逐符号音位：4 支路非相干功率谱峰 + 3 点对数抛物线分数。
        # （核扫描版在 κ=±0.5 精确落界处三峰简并 + 宽抽头旁瓣伪峰——
        # f27 实测音位恒 +0.500，DTFT 直核；抛物线分数在该处稳定。）
        y = np.empty(psym)
        w = np.empty(psym)
        for k in range(psym):
            p0 = np.sum(np.abs(X[k]) ** 2, axis=0)
            bpk = int(np.argmax(p0))
            l0, l1, l2 = (np.log(np.maximum(p0[(bpk + d) % self.n], 1e-30))
                          for d in (-1, 0, 1))
            den = l0 - 2.0 * l1 + l2
            frac = 0.0 if abs(den) < 1e-12 else 0.5 * (l0 - l2) / den
            frac = float(np.clip(frac, -0.5, 0.5))
            y[k] = frac
            w[k] = float(p0[bpk])
        w = w / max(w.sum(), 1e-30)
        # 滑窗圆均值（半宽 6）→ 相邻差分卷绕解开成未卷绕轨迹（可越
        # ±0.5/±1——f08 实测 0.95→1.15→0.4；TREL 的 κ 列格点隐式吃掉
        # 这个，我们的对应物 = 未卷绕核中心）。
        hw = 6
        m = np.empty(psym)
        for k in range(psym):
            lo, hi = max(0, k - hw), min(psym, k + hw + 1)
            zz = np.sum(w[lo:hi] * np.exp(2j * np.pi * y[lo:hi]))
            m[k] = np.angle(zz) / (2 * np.pi) if abs(zz) > 0 else 0.0
        out = np.empty(psym)
        out[0] = m[0]
        for k in range(1, psym):
            d = ((m[k] - m[k - 1] + 0.5) % 1.0) - 0.5
            out[k] = out[k - 1] + d
        return out


class DeraDirTapsDemodulator:
    """臂④：DeRa port 结构原样（逐候选越界切分 + 非相干初始化 + Stage-2
    ML 公共相位 + |·|² 幅度判决），仅把 V1/V2 读出频率从整数 c 挪到
    (c+κ̂)——实现 = 窗乘 e^{−j2πκ̂n/NF} 斜坡后走原投影矩阵。
    证明增量来自抽头结构（分数读出）而非链路其他件。"""

    def __init__(self, sf: int, os_factor: int):
        from weak_decoder.baselines.dera.paper_dera_demod import (
            _wrap_split_matrices)
        self.sf, self.os = int(sf), int(os_factor)
        self.n = 1 << self.sf
        self.nf = self.n * self.os
        self._front, self._tail, self._split = _wrap_split_matrices(
            self.sf, self.os)
        self._nn = np.arange(self.nf)
        self.ml_phase: float = 0.0

    def demod_payload(self, samples: np.ndarray, start_symbol: int, psym: int,
                      kappa_hat=None) -> np.ndarray:
        psym = int(psym)
        kap = (np.zeros(psym) if kappa_hat is None
               else np.asarray(kappa_hat, dtype=np.float64))
        f_all, t_all = [], []
        for k in range(psym):
            s = (int(start_symbol) + k) * self.nf
            w = np.asarray(samples[s:s + self.nf], dtype=np.complex64)
            if w.size != self.nf:
                raise ValueError(f"payload symbol {k} window exceeds input")
            ramp = np.exp(-2j * np.pi * kap[k] * self._nn / self.nf
                          ).astype(np.complex64)
            wr = (w * ramp).astype(np.complex64)
            f_all.append(self._front @ wr)
            t_all.append(self._tail @ wr)
        f_mat = np.stack(f_all)
        t_mat = np.stack(t_all)
        noncoh = np.abs(f_mat) ** 2 + np.abs(t_mat) ** 2
        init = np.argmax(noncoh, axis=1)
        idx = np.arange(f_mat.shape[0])
        z = np.sum(t_mat[idx, init] * np.conj(f_mat[idx, init]))
        self.ml_phase = float(np.angle(z)) if abs(z) > 0 else 0.0
        return np.abs(f_mat + np.exp(-1j * self.ml_phase) * t_mat) ** 2


def gamma_kappa_track(samples: np.ndarray, start_symbol: int, psym: int,
                      sf: int, os_factor: int) -> np.ndarray:
    """γ-链帧池化 κ̂ 轨迹（payload 锚定，所有臂同源同喂）。

    复刻 GammaLinkDemodulator 的 WLS（y=γ/π ∈ (−1,1]，|z| 加权，
    合角初始化 + 3 轮模 1 卷绕精化），返回逐符号 m_hat = κ₀+δ·i。
    不修改仓库模块（γ-链 v3 的 last_fit 只给 κ₀/δ，这里给出全轨迹）。
    """
    from weak_decoder.baselines.dera.paper_dera_demod import (
        _wrap_split_matrices)
    n = 1 << int(sf)
    nf = n * int(os_factor)
    front, tail, _ = _wrap_split_matrices(int(sf), int(os_factor))
    idx = np.arange(int(psym))
    f_mat = np.empty((psym, n), dtype=np.complex128)
    t_mat = np.empty((psym, n), dtype=np.complex128)
    for k in idx:
        s = (int(start_symbol) + k) * nf
        w = np.asarray(samples[s:s + nf], dtype=np.complex64)
        if w.size != nf:
            raise ValueError(f"symbol {k} window exceeds input")
        f_mat[k] = front @ w
        t_mat[k] = tail @ w
    noncoh = np.abs(f_mat) ** 2 + np.abs(t_mat) ** 2
    chat = np.argmax(noncoh, axis=1)
    z = t_mat[idx, chat] * np.conj(f_mat[idx, chat])
    y = np.angle(z) / np.pi
    w_i = np.abs(z).astype(np.float64)
    w_arr = w_i / max(float(w_i.sum()), 1e-30)
    x_arr = idx.astype(np.float64)
    H = np.stack([np.ones(psym), x_arr], axis=1)
    HW = H.T @ np.diag(w_arr)

    def _wls(yv):
        try:
            return np.linalg.solve(HW @ H, HW @ yv)
        except np.linalg.LinAlgError:
            return np.array([float(np.average(yv, weights=w_arr)), 0.0])

    beta = np.array([float(np.angle(np.sum(w_arr * np.exp(1j * np.pi * y))) / np.pi),
                     0.0])
    for _ in range(3):
        r = ((y - H @ beta) + 0.5) % 1.0 - 0.5
        beta = _wls(H @ beta + r)
    return beta[0] + beta[1] * x_arr

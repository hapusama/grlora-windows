# -*- coding: utf-8 -*-
"""D2F 两段 sub 信号相位补齐（phase completion，2026-10-06）。

用户指令：D2F 目前主要靠偏移参数网格（ε 梯/τ̂/Δ0）在工作，要像
DeRa 一样把**符号两段 sub 信号的相位补齐**加进来，进一步榨取相干
能量叠加；调通后对 DeRa 打 §5A 全链战 + 模块消融。

结构定位（对 DeRa Eq.20-22）：
  port（A 列）已有 V1/V2 两段投影 + **单常数** ML φ̂₀ 合并（Algorithm 1
  line 36；decode-only 下 fcfo=0 ⇒ 漂移项塌缩为常数）。
  本文件把合并相位升级为**逐符号斜坡** φ_k = φ̂₀ + s·(k−k_c)——
  DeRa 的 φ̂_i = φ̂₀ + 2πfcfo·i·T' 结构（fcfo 由检测级提供 ⇒ 我方
  斜率源=δ̂−δ_c 模型）+ 我方扩展（强符号加权盲拟合，斜坡下联合 ML φ̂₀）。

依据（诊断 2026-10-06，3 帧 GT-bin 实测）：
  - 精确修复候选 (0.02)：γ_k 斜率 ~+10 mrad/符，常数 φ̂₀ 尾损 ≤0.03dB
    （合并已近无损，相位补齐无增益——消融锚点）；
  - 半修复赢家 (0.01, τ̂_pre)：γ_k 线性走动（拟合残差 0.03-0.07 rad），
    常数 φ̂₀ 尾符号损失 **p90 +6~11dB**（相位误差近 π ⇒ V1/V2 对消）；
  - 解析模型 2π·δ_res 符号反/幅值差 3×（实测 ≈ −⅓·2π·δ_res，且
    (0.02) 档仍有 +10 mrad 残留）⇒ **盲拟合为主、模型斜率只作对照臂**。
  - F3 否定定律兼容：斜坡是 2 参数模型驱动拟合（非逐符号自由估计），
    拟合失败护栏（残差 >0.35 rad）自动退回常数 φ̂₀ = port 行为。

验收/消融锚点：rows_const 与 port 逐位一致（同窗切片 complex64、同
_project、同全符号 ML φ̂₀）——P−A 差 = 纯相位补齐贡献。
"""
import os
import sys

import numpy as np

_WD = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
for _p in (
    _WD,
    os.path.join(_WD, "data", "experiments", "keystone_battle_20261003"),
    os.path.join(_WD, "data", "experiments", "dep_anchor_20261004"),
    os.path.join(_WD, "data", "experiments", "e2e_final_20261004"),
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    os.path.dirname(os.path.abspath(__file__)),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m3_core as M3                       # noqa: E402  demap_judge
import fast_tmpl_core as FT                # noqa: E402  repair/fast 基座

NF = 4096
N = 1024
OS = 4
SLOPE_SCALE = -1.0 / 3.0                   # 模型斜率实测系数（诊断 3 帧）
FIT_MAX_RES = 0.35                         # 盲拟合护栏（rad）：超过退回常数


_DN_CACHE = None


def _downchirp_ref():
    """标准去斜参考 conj(build_upchirp(0))，与 port 同源（缓存）。"""
    global _DN_CACHE
    if _DN_CACHE is None:
        from weak_decoder.chirp import build_upchirp
        _DN_CACHE = np.conjugate(
            build_upchirp(sf=10, symbol_id=0, os_factor=4)
        ).astype(np.complex64)
    return _DN_CACHE


def fft_rows(base, start_sym, psym):
    """单次全窗去斜 FFT 行（**无两段参考结构** = 传统 demod 形态，
    DeRa 消融臂；os=4 正确形态 = 混叠 bin 非相干求和）。

    ⚠️ 混叠定律（本轮实验钉死，F12）：信号 fold 处频率跳变 =
    1/os 周期/样本（build_upchirp linear_after−before = −n/os）。
    os=1：跳变 = 整周期/样本 → 混叠为零，平参考 FFT 天然两段免疫
    （仓内 GT 链 chirp.dechirp_fft 即此形态）。os=4：跳变 = 1/4
    周期/样本 → 平参考 FFT 把能量劈进**相距 NF/os = N 个 bin 的两个
    混叠音**（前段/后段各一，fold 位置决定哪个大——实测干净符号峰位
    精确二值 {v+2, v+2−N} mod NF，无散布）；中位 fold 最长段只有半窗
    → 单 bin 最多亏 ~6dB + argmax 跳段。**两段参考滚动（port）的
    本质 = 让参考的 fold 对齐候选的 fold，两段落回同一 bin**——
    这才是 DeRa two-section 结构的必需性所在。
    本臂（公平的无两段形态）：一次 FFT/符号 + 逐候选混叠 bin 对
    非相干求和 rows[k] = spec[(k+2)%NF] + spec[(k+2−N)%NF]
    （+2 = 码值 +1 约定 + 实测 κ 偏置；常数由 Δ0 吸收）。"""
    dn = _downchirp_ref()
    rows = []
    kk = np.arange(N)
    for k in range(int(psym)):
        w = np.asarray(base[(int(start_sym) + k) * NF:
                            (int(start_sym) + k + 1) * NF],
                       dtype=np.complex64)
        spec = np.abs(np.fft.fft(w * dn)) ** 2
        rows.append(spec[(kk + 2) % NF] + spec[(kk + 2 - N) % NF])
    return np.stack(rows)


def stage1_rows(f_mat, t_mat):
    """Stage-1 行 |F+T|²（φ=0 合并，无 ML φ̂₀）——从投影直出，
    隔离 φ̂₀ 相位自适应价值。"""
    return np.abs(f_mat + t_mat).astype(np.float64) ** 2


def project_payload(base, start_sym, psym, dd):
    """逐符号 V1/V2 投影（复用 DD._project；窗切片 complex64 与 port
    逐位一致）。返回 (f_mat, t_mat, noncoh) 各 (psym, N)。"""
    fs, ts, ncs = [], [], []
    for k in range(int(psym)):
        w = np.asarray(base[(int(start_sym) + k) * NF:
                            (int(start_sym) + k + 1) * NF],
                       dtype=np.complex64)
        fp, tp, _s1, nc = dd._project(w)
        fs.append(fp)
        ts.append(tp)
        ncs.append(nc)
    return np.stack(fs), np.stack(ts), np.stack(ncs)


def merge_rows(f_mat, t_mat, phi_k):
    """两段相干合并行 |F + e^{−jφ_k}T|²（phi_k (psym,) 逐符号相位）。"""
    return np.abs(f_mat + np.exp(-1j * phi_k)[:, None] * t_mat
                  ).astype(np.float64) ** 2


def phi0_ml_all(z):
    """port line-36 口径 φ̂₀ = ∠(Σ z)（全 temp 符号）。"""
    return float(np.angle(np.sum(z))) if np.abs(np.sum(z)) > 0 else 0.0


def phi0_ml_ramp(z, slope, ks, kc):
    """斜坡下联合 ML φ̂₀ = ∠(Σ z_s·e^{+j·slope·(s−kc)})，强符号集。"""
    mag = np.abs(z)
    s = np.argsort(mag)[-max(4, len(z) // 2):]
    zr = z[s] * np.exp(1j * slope * (ks[s] - kc))
    return float(np.angle(np.sum(zr))) if np.abs(np.sum(zr)) > 0 else 0.0


def fit_slope(z, ks, strong_frac=0.5, max_res=FIT_MAX_RES):
    """强符号加权线性 γ(k) 拟合（盲斜率）。z=temp 符号处 T·conj(F)。

    抗 antiphase 零点（port 历史失败教训=unwrap 链被零点污染）：取 |z|
    前 (1−strong_frac) 强符号、|z| 加权一次拟合；加权残差 > max_res →
    拒绝返回 None（调用方退回常数 φ̂₀）。"""
    psym = len(z)
    n_strong = max(4, int(psym * (1.0 - strong_frac)))
    idx = np.argsort(np.abs(z))[-n_strong:]
    ph = np.unwrap(np.angle(z[idx]))
    w = np.abs(z[idx])
    kc = float(np.mean(ks[idx]))
    A = np.stack([np.ones_like(ks[idx]), ks[idx] - kc], axis=1)
    coef, *_ = np.linalg.lstsq(A * w[:, None], ph * w, rcond=None)
    resid = float(np.sqrt(np.average((ph - A @ coef) ** 2, weights=w)))
    if not np.isfinite(resid) or resid > max_res:
        return None, kc, resid
    return float(coef[1]), kc, resid


def demod_phased(base, start_sym, psym, dd, slope_mode="blind", d_res=0.0):
    """P 列解调：一次投影 → 三组行 + 诊断量。

    rows_const：常数 φ̂₀（全符号 ML，与 port 逐位一致=消融锚点）；
    rows_p：斜坡 φ_k（slope_mode=blind 强符号加权拟合 / model
    2π·d_res·SLOPE_SCALE；失败护栏退回 rows_const）。
    返回 dict(rows_const, rows_p, slope, resid, noncoh)。"""
    f_mat, t_mat, noncoh = project_payload(base, start_sym, psym, dd)
    idx = np.arange(psym)
    tmp = np.argmax(noncoh, axis=1)
    z = t_mat[idx, tmp] * np.conj(f_mat[idx, tmp])
    phi0 = phi0_ml_all(z)
    rows_const = merge_rows(f_mat, t_mat, np.full(psym, phi0))
    if slope_mode == "model":
        slope = 2.0 * np.pi * float(d_res) * SLOPE_SCALE
        kc = (psym - 1) / 2.0
        ph0 = phi0_ml_ramp(z, slope, idx.astype(float), kc)
        rows_p = merge_rows(f_mat, t_mat,
                            ph0 + slope * (idx - kc))
        return dict(rows_const=rows_const, rows_p=rows_p,
                    slope=slope, resid=0.0, noncoh=noncoh)
    slope, kc, resid = fit_slope(z, idx.astype(float))
    if slope is None:
        return dict(rows_const=rows_const, rows_p=rows_const,
                    slope=None, resid=resid, noncoh=noncoh)
    ph0 = phi0_ml_ramp(z, slope, idx.astype(float), kc)
    rows_p = merge_rows(f_mat, t_mat, ph0 + slope * (idx - kc))
    return dict(rows_const=rows_const, rows_p=rows_p, slope=slope,
                resid=resid, noncoh=noncoh)


def fast_arm_sets(seg, f, hs, pay0, nu0, dhat, origin, dd, kt=None,
                  ladder=None, sets=(("a", "p", "b"), ("a",), ("n",))):
    """一次候选循环 × 多套列组合各自独立 CRC 早退（链战消融专用，
    三臂共用 τ̂/修复/投影，省 ~60% 算力）。

    sets = 列组合元组们（默认：完整链 a→p→b / A-only / N-only）。
    每套独立状态机：候选内按列序 CRC，过了即定格；全候选失败记首败。
    返回 dict(组合索引 → arm result dict)（字段同 fast_arm_p）。"""
    pre, psym = f["pre"], f["psym"]
    if ladder is None:
        ladder = FT.ladder_order(dhat)
    res = [None for _ in sets]
    for dc in ladder:
        tau = FT.blind_tau0_pre(seg, hs, pre, nu0, dc, origin)
        base = FT._repaired_base(seg, pay0, dc, tau, nu0, origin, pre)
        ph = demod_phased(base, pre + 5, psym, dd)
        rows = {"a": ph["rows_const"], "p": ph["rows_p"],
                "n": ph["noncoh"]}
        need_b = any("b" in cols for i, cols in enumerate(sets)
                     if res[i] is None)
        if need_b and kt is not None:
            rows["b"] = kt.demod_payload(base, pre + 5, psym,
                                         readout="viterbi")
        for i, cols in enumerate(sets):
            if res[i] is not None:
                continue
            for tag in cols:
                if tag not in rows:
                    continue
                ok, ser, d0 = M3.demap_judge(rows[tag],
                                            np.zeros(psym, dtype=int), f)
                if ok:
                    res[i] = dict(ok=True, ser=ser, col=tag, delta_c=dc,
                                  tau=tau,
                                  p_slope=(float(ph["slope"])
                                           if tag == "p"
                                           and ph["slope"] is not None
                                           else None))
                    break
    return {i: (r if r is not None else dict(ok=False, ser=-1, col=None,
                                             delta_c=None, tau=None))
            for i, r in enumerate(res)}


def fast_arm_p(seg, f, hs, pay0, nu0, dhat, origin, dd, kt=None,
               ladder=None, tau_map=None, p_mode="blind",
               cols=("a", "p", "b")):
    """fast 臂 + 相位补齐列：候选 ε 梯 × cols 顺序 × Δ0 CRC 早退。

    列族：a = port 常数 φ̂₀（与 DD.demod_payload 逐位一致，消融锚点）；
    p = 斜坡合并（blind 逐候选盲拟合 / model 2π(δ̂−δ_c)/3，护栏退回 a）；
    b = trellis；n = **非相干合并 |F|²+|T|²（= LoRaTrimmer 度量，消融
    臂——量化两段相干叠加本身的价值）**。a/p/n 共享一次投影。
    返回 dict(ok/ser/col/delta_c/tau/p_slope/p_resid)。"""
    pre, psym = f["pre"], f["psym"]
    if ladder is None:
        ladder = FT.ladder_order(dhat)
    best_fail = None
    for dc in ladder:
        tau = tau_map[dc] if tau_map is not None else FT.blind_tau0_pre(
            seg, hs, pre, nu0, dc, origin)
        base = FT._repaired_base(seg, pay0, dc, tau, nu0, origin, pre)
        d_res = float(dhat) - float(dc)
        cols_done = {}
        if any(c in cols for c in ("a", "p", "n")):
            ph = demod_phased(base, pre + 5, psym, dd,
                              slope_mode=("blind" if p_mode == "blind"
                                          else "model"), d_res=d_res)
            cols_done["a"] = ph["rows_const"]
            cols_done["p"] = ph["rows_p"]
            cols_done["n"] = ph["noncoh"]
        if "b" in cols and kt is not None:
            cols_done["b"] = kt.demod_payload(base, pre + 5, psym,
                                              readout="viterbi")
        for tag in cols:
            if tag not in cols_done:
                continue
            ok, ser, d0 = M3.demap_judge(cols_done[tag],
                                         np.zeros(psym, dtype=int), f)
            if ok:
                return dict(ok=True, ser=ser, col=tag, delta_c=dc,
                            tau=tau,
                            p_slope=(float(ph["slope"])
                                     if tag == "p" and ph["slope"] is not None
                                     else None),
                            p_resid=(float(ph["resid"]) if tag == "p"
                                     else None))
            if best_fail is None:
                best_fail = dict(ok=False, ser=ser, col=tag, delta_c=dc,
                                 tau=tau)
    if best_fail is None:
        best_fail = dict(ok=False, ser=-1, col=None, delta_c=None, tau=None)
    return best_fail

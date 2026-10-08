# -*- coding: utf-8 -*-
"""D2F 快实现精确模板接收机（fast_tmpl，2026-10-05）。

用户下一代接收机设计（05_docs §4 首版范围）的快实现路径（HANDOFF §4
主线）：**每候选 ε 分数重采样 → 常规 dechirp-FFT（port 结构不动）**。
依据链：FINDINGS F1-F3——SFO 伤害主泄漏 = 段间微分相位；相位补偿必须
由 (ν₀, τ₀, ε) 解析预测（盲逐符 φ 被否定）；兑现上限 = 行2 实测
+1.1~+4.8dB（vs 现行链最强形态）。

机制：注入 y[m]=x(origin+(m−origin)(1+ε)) 的**精确逆** =
eps_undo=−ε/(1+ε) 重采样 + 分数时延 τ̂₀ → 时间基整数化 → fold 点回到
port 假设的 (N−k)·os 整数位 → V1/V2 切分精确成立 → ML φ̂₀ 合并逼近
完整模板相关（验收①：同参数下统计量差 ≤0.1dB）。

三件套来源（可部署口径）：
  ν̂₀ = dep4 连续锚（约定与 demod 端逐位校准，0.146bin 生死线 F5②；
        sfo_sto A_full 已验证"修复后直接用 ν̂₀"）
  ε   = δ̂ 梯 {0, ±0.01, ±0.02}（HANDOFF 指定；按 |δ_c−δ̂| 排序早退）
  τ̂₀ = 前导盲估（本文件新仪器）：code0 上啁啾全前导窗精确模板
       （复用 RT.z_full，codes=1）一维格搜索 ±2os 采样——GT 码值不进
       估计（前导码值公知）；Fresnel 旁瓣风险（F5①）由验收③量化。

链集成：ours_chain_decode3 = m3p2 链（斜率梯 A/B）全败后追加 fast 梯
救援——单调改进、δ=0 零回归（u ok 即跳过 fast）；候选预算
56+5×2×7=126 判次 vs DeRa 25，CRC 撞门 ~2⁻¹⁶/候选 可忽略（§D5 同款）。

验收硬标准（HANDOFF §4）：①统计量 ≤0.1dB；②ν̂₀ 约定 native 全解；
③盲 τ̂₀ vs oracle 分布 + native；④终判 §5A 全链 CRC。
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
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # 包根
    os.path.dirname(os.path.abspath(__file__)),                    # 01_core
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import d2_core as D2                       # noqa: E402  resample_sfo / eps_of_delta
import m3_core as M3                       # noqa: E402  demap_judge
import ref_template as RT                  # noqa: E402  z_full / est_tau0
import m3p2_core as M3P2                   # noqa: E402  ours_chain_decode2

NF = 4096
N = 1024
OS = 4
# ε 候选梯（δ bin/符；HANDOFF 指定 {0, ±0.01, ±0.02}）
FAST_LADDER = (0.0, 0.01, -0.01, 0.02, -0.02)
# δ̂ 门（与 m3_core.nu_rot_of 同源）：|δ̂|<0.04 视无漂移 → 候选 0 排最前
DHAT_GATE = 0.04


def eps_undo_of(delta_c):
    """δ 候选（bin/符）→ 注入的精确逆重采样率 ε_u = −ε/(1+ε)。

    注入 resample_sfo(eps) 为 y[m]=x(origin+(m−origin)(1+ε))；其精确逆
    是先验知识（接收机假设 δ 候选格与注入同族），sfo_sto A_full 同式。"""
    e = D2.eps_of_delta(delta_c)
    return 0.0 if e == 0.0 else -e / (1.0 + e)


def frac_delay(seg, tau):
    """全段分数时延（带限，频域相位坡；τ 采样，与 sfo_sto 逐字一致）。"""
    if abs(tau) < 1e-9:
        return seg
    f = np.fft.fftfreq(len(seg))
    return np.fft.ifft(np.fft.fft(seg) * np.exp(-2j * np.pi * f * tau))


def repair_timebase(seg, origin, delta_c, tau0):
    """候选时间基修复：ε 精确逆重采样（origin 锚定）+ 分数时延。

    τ₀ 符号约定（z_full 对拍实测钉死）：z_full 模型 u=n+s_k−τ₀ ⇒ 信号
    符号起点在窗格 Mk+τ₀ ⇒ 对齐 = **提前 τ₀** = 延迟 −τ₀。另证：tone
    分数位不随整体延迟移动（LoRa 去斜 STO 免疫——延迟只挪 fold 瞬态），
    只有整数 bin 记账在 |d|≈0.75 采样处翻转 ±1（链上 Δ0 合法吸收）；
    tone SNR 平顶 |d|∈[1.5,3] 两侧同高（native），低 SNR 下几何正确
    符号（−τ̂₀）预期占优（V1/V2 残余相对相位 2π·(τ₀+d)/os 最小）。"""
    seg_c = D2.resample_sfo(seg, origin, eps_undo_of(delta_c))
    return frac_delay(seg_c, -float(tau0))


def _repaired_base(seg, pay0, delta_c, tau0, nu0, origin, pre):
    """修复 + pad + ν̂₀ 直旋 → 解调基带（修复后走动已消，无需 nu_rot
    居中/斜率梯；ν̂₀ 约定 = dep4 锚直旋，sfo_sto A_full 已验证）。"""
    seg_c = repair_timebase(seg, origin, delta_c, tau0)
    pad = (pre + 5) * NF
    seg_p = np.concatenate((np.zeros(pad, dtype=np.complex128),
                            seg_c[pay0:]))
    return seg_p * np.exp(-2j * np.pi * nu0 * np.arange(len(seg_p)) / NF)


def fast_rows(seg, pay0, delta_c, tau0, nu0, origin, pre, psym, dd):
    """修复 + 常规 dechirp-FFT（port 结构不动）。返回 port 相干行。"""
    base = _repaired_base(seg, pay0, delta_c, tau0, nu0, origin, pre)
    return dd.demod_payload(base, pre + 5, psym)[1]


def blind_tau0_pre(seg, hs, pre, nu0, delta_c, origin,
                   step=0.05, span_os=2.0):
    """前导盲 τ̂₀：全前导上啁啾窗（code 值 0 → 模板码值 1）精确模板
    一维格搜索，目标 max_τ Σ_k |Z_k(τ)|²/M_sup（非相干跨窗）。

    前导窗起点 = 场首 = hs − (pre+4.25)·NF（0.25·NF=1024 恒整数）；
    z_full 的 (ν₀, ε_c, origin, τ) 模板与本臂修复参数同族自洽。
    码值公知不进 GT；Fresnel 多瓣风险（F5①，GT 辅助搜索曾落旁瓣），
    命中率由验收③对 oracle（GT 码值 payload est_tau0）量化。"""
    p_start = int(hs - (pre + 4.25) * NF)
    codes1 = np.ones(pre, dtype=int)
    eps = D2.eps_of_delta(delta_c)
    grid = np.arange(-span_os * OS, span_os * OS, step)
    best, btau = None, 0.0
    for t in grid:
        Z, Msup = RT.z_full(seg, p_start, codes1, nu0, eps, origin,
                            tau0=float(t))
        v = float(np.sum(np.abs(Z) ** 2 / Msup))
        if best is None or v > best:
            best, btau = v, float(t)
    return btau


def ladder_order(dhat):
    """ε 候选执行序：|δ̂|<DHAT_GATE → 0 最先（δ=0 零回归）；否则按
    |δ_c−δ̂| 升序（早退预算最小化），0 兜底在尾。"""
    cand = list(FAST_LADDER)
    if abs(dhat) < DHAT_GATE:
        return cand
    return sorted(cand, key=lambda c: (abs(c - dhat), abs(c)))


def fast_arm(seg, f, hs, pay0, nu0, dhat, origin, dd, kt=None,
             ladder=None, tau_mode="blind", tau0_oracle=None, tau_map=None):
    """E_fast 机制/链共用的 fast 臂：候选 ε 梯 × {A 列 [+B 列]} × Δ0 CRC
    早退（demap_judge 零走动反映射，修复后无整数走动）。

    τ̂ 来源三型：tau_map=dict(δ_c→τ̂) 冻结表（机制战，实验B 哲学）；
    blind=逐候选前导盲估（链战，可部署）；oracle=tau0_oracle 冻结
    （机制上限）。ladder=None → ladder_order(dhat)。返回 dict：
    ok/ser/delta_c/col/tau（col=a/b；全败时 col=None, delta_c=None）。"""
    pre, psym = f["pre"], f["psym"]
    best_fail = None
    for dc in (ladder if ladder is not None else ladder_order(dhat)):
        if tau_map is not None:
            tau = tau_map[dc]
        elif tau_mode == "blind":
            tau = blind_tau0_pre(seg, hs, pre, nu0, dc, origin)
        else:
            tau = tau0_oracle
        rows_a = fast_rows(seg, pay0, dc, tau, nu0, origin, pre, psym, dd)
        cols = [("a", rows_a)]
        if kt is not None:
            base = _repaired_base(seg, pay0, dc, tau, nu0, origin, pre)
            cols.append(("b", kt.demod_payload(base, pre + 5, psym,
                                               readout="viterbi")))
        for tag, rows in cols:
            ok, ser, d0 = M3.demap_judge(rows, np.zeros(psym, dtype=int), f)
            if ok:
                return dict(ok=True, ser=ser, col=tag, delta_c=dc, tau=tau)
            if best_fail is None:
                best_fail = dict(ok=False, ser=ser, col=tag, delta_c=dc,
                                 tau=tau)
    if best_fail is None:
        best_fail = dict(ok=False, ser=-1, col=None, delta_c=None, tau=None)
    return best_fail


def stat_gap_oracle(seg_inj, f, m0, nu0, delta, origin, tau0, dd, n0):
    """验收①仪器：oracle 参数（δ=GT、τ̂₀ 冻结、ν̂₀ 锚）下 fast 统计量
    vs ref_template 完整模板，逐符号输出 SNR 差（dB，fast−ref）。

    读 bin = mode(argmax−(gt+1))（帧级常数整数偏移——去斜 STO 免疫下
    整数记账随修复翻转，链上由 Δ0 合法吸收；机制对拍须读真实音位）。
    判据方向：gap ≥ −0.1dB = 无结构损失/无重复补偿；正方向超出 =
    port ML φ̂₀ 自适应红利（行1 定律：native 下 port 胜开环模板
    0.8~3.2dB），单列不改判。"""
    psym = f["psym"]
    gt1 = (np.asarray(f["gt"]) + 1) % N
    Z, Msup = RT.z_full(seg_inj, m0, gt1, nu0, D2.eps_of_delta(delta),
                        origin, tau0=tau0)
    snr_ref = np.abs(Z) ** 2 / (n0 * Msup)
    rows = fast_rows(seg_inj, m0, delta, tau0, nu0, origin,
                     f["pre"], psym, dd)
    am = np.argmax(rows, axis=1)
    mode_off = int(np.bincount((am - gt1) % N).argmax())
    snr_fast = rows[np.arange(psym), (gt1 + mode_off) % N] / (n0 * NF)
    gap = (10 * np.log10(np.maximum(snr_fast, 1e-30))
           - 10 * np.log10(np.maximum(snr_ref, 1e-30)))
    return dict(gap=gap, snr_fast=snr_fast, snr_ref=snr_ref,
                mode_off=mode_off)


def ours_chain_decode3(seg, det, f, kt, dd):
    """链集成（§5A 终判）：m3p2 链全败后 fast 梯救援。

    单调改进：u ok 即跳过 fast（δ=0 零回归 + 预算）；fast 臂全盲
    （ν̂₀/δ̂ 用 det 自产量，τ̂₀ 前导盲估，ε 梯 ladder_order(δ̂)）。
    返回 m3p2 的 out 附加：fast（fast 臂结果）、uf（总判决 = u ∪ fast）。"""
    out = M3P2.ours_chain_decode2(seg, det, f, kt, dd)
    if out["u"]["ok"]:
        out["fast"] = dict(ok=True, ser=out["u"]["ser"], col="skip",
                           delta_c=None, tau=None)
        out["uf"] = dict(out["u"])
        return out
    origin = det["hs"] - int((f["pre"] + 4.25) * NF)
    fr = fast_arm(seg, f, det["hs"], det["pay0"], det["nu0h"],
                  det["dhat"], origin, dd, kt=kt, tau_mode="blind")
    out["fast"] = fr
    out["uf"] = (dict(ok=True, ser=fr["ser"], col="fast_" + str(fr["col"]),
                      delta_c=fr["delta_c"]) if fr["ok"]
                 else dict(ok=False, ser=out["u"]["ser"], col=out["u"]["col"]))
    return out

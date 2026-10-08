# -*- coding: utf-8 -*-
"""SFO 损伤通道分解（2026-10-05）：时间轴斜坡 vs 段间微分相位。

用户命题："SFO 从时间角度可用 STO 取代（横轴影响）；真正不妥的是
相位——dechirp 后残余带斜线，两段 sub 积累的相位不一样。"

检验设计（单一变量，同 DeRa port 判决，ν̂₀=dep4 约定）：
  B       最强单 STO（常数时延，中心化）——port 标准单 ML φ̂0
  B_phi   同 B 的时延 + **逐符号理想段间相位**（GT 码上量
          φ_k=∠(T_k·F_k*) 注入 demod_symbol(phi0=φ_k)，再全候选重组合）
  A_full  ε 反转重采样 + τ̂₀（时间轴精确修复，oracle）
  A_phi   A + 逐符号 φ_k（对照：A 残余只剩常数 τ₀ 相位，增益应小）

判定：B_phi 追回 A 的大半 ⇒ 用户归因成立（伤害主通道=段间微分相位）；
B_phi 仍落后 ⇒ 时间轴走动另有独立伤害（音位游走/切分点漂移）。
δ=0.02，28 帧 × SNR{−20,−22,−24} × 2 种子 → stdout。
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\weak_decoder\D2F_20261005")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\weak_decoder\D2F_20261005\01_core")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\e2e_final_20261004")

import d2_core as D2
import m3_core as M3
import m3p_core as M3P
import front_runner as FR
import ref_template as RT
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator

NF, N = 4096, 1024
DD = DeRaDemodulator(10, 4)
SEED_CONST = 20261005
DELTA = 0.02
LEVELS = (-20, -22, -24)
SEEDS = (0, 1)


def frac_delay(seg, tau):
    if abs(tau) < 1e-9:
        return seg
    fr = np.fft.fftfreq(len(seg))
    return np.fft.ifft(np.fft.fft(seg) * np.exp(-2j * np.pi * fr * tau))


def phi_rows(seg_base, pre, psym, gt1):
    """逐符号段间相位：φ_k 在**跟踪 bin**（非相干 argmax）上量——
    在假设 bin 上量会踩 bin 边界错位（时延~4采样=1 bin 平移时全读
    噪声相位，φ 散布 5.5 rad 假象，本轮诊断实锤），跟踪 bin 与真实
    接收机的相位估计行为一致。返回全候选行 |F+e^{-jφk}T|²。"""
    rows = np.empty((psym, N))
    phis = np.empty(psym)
    for k in range(psym):
        w = np.asarray(seg_base[(pre + 5 + k) * NF:
                                (pre + 6 + k) * NF], dtype=np.complex64)
        f_proj, t_proj, _s1, _nc = DD._project(w)
        c = int(np.argmax(np.abs(f_proj) ** 2 + np.abs(t_proj) ** 2))
        phis[k] = float(np.angle(t_proj[c] * np.conj(f_proj[c])))
        rows[k] = np.abs(f_proj + np.exp(-1j * phis[k]) * t_proj) ** 2
    return rows, phis


def main():
    frames = FR.build_frames()
    agg = collections.defaultdict(lambda: collections.defaultdict(
        lambda: dict(n=0, per=0)))
    phi_rec = []
    for fi, f in enumerate(frames):
        pre, psym = f["pre"], f["psym"]
        lead = pre + 6
        seg0 = np.asarray(f["iq"][f["hs"] - lead * NF:
                                 f["hs"] + (8 + psym + 2) * NF + 64],
                          dtype=np.complex128)
        origin = (lead - (pre + 4.25)) * NF
        m0 = lead * NF + 8 * NF
        eps = D2.eps_of_delta(DELTA)
        seg_inj = D2.resample_sfo(seg0, origin, eps)
        nu0 = float(M3P.dep4_blind_detect(seg_inj, pre)["nu0h"])
        gt1 = np.asarray(f["gt"]) + 1
        tau_nat = RT.est_tau0(seg0, m0, gt1, nu0, 0.0, 0.0, None)
        r_mean = float(np.mean([m0 + k * NF - origin
                                for k in range(psym)])) * eps
        tau_best = tau_nat + r_mean
        S, n0 = FR.snr_parts(seg_inj[:lead * NF + 8 * NF])
        seg_B = frac_delay(seg_inj, tau_best)
        seg_A = frac_delay(D2.resample_sfo(seg_inj, origin,
                                           -eps / (1 + eps)), tau_nat)
        if fi < 3:
            _, phB = phi_rows(_rot(seg_B, nu0, pre, psym, m0), pre, psym, gt1)
            _, phA = phi_rows(_rot(seg_A, nu0, pre, psym, m0), pre, psym, gt1)
            phi_rec.append(dict(frame=fi,
                                phiB=(phB - np.mean(phB)).round(3).tolist(),
                                phiA=(phA - np.mean(phA)).round(3).tolist()))
        for lv in LEVELS:
            for sd in SEEDS:
                rng = np.random.default_rng(
                    (SEED_CONST * 7919 + (int(lv) + 100) * 131
                     + sd * 101 + fi * 7919 + 3 * 104729) % (2 ** 31))
                p_add = max(S / 10 ** (lv / 10.0) - n0, 1e-30)
                noise = ((rng.standard_normal(len(seg_inj))
                          + 1j * rng.standard_normal(len(seg_inj)))
                         * np.sqrt(p_add / 2.0))
                rec = dict(frame=fi, level=lv, seed=sd)
                for tag, seg_c in (("B", seg_B), ("A", seg_A)):
                    base = _rot(seg_c + noise, nu0, pre, psym, m0)
                    rows_std = DD.demod_payload(base, pre + 5, psym)[1]
                    ok, ser, d0 = M3.demap_judge(rows_std,
                                                 np.zeros(psym, dtype=int), f)
                    agg[lv][tag]["n"] += 1
                    agg[lv][tag]["per"] += int(not ok)
                    rows_phi, _ = phi_rows(base, pre, psym, gt1)
                    okp, serp, d0p = M3.demap_judge(
                        rows_phi, np.zeros(psym, dtype=int), f)
                    agg[lv][tag + "_phi"]["n"] += 1
                    agg[lv][tag + "_phi"]["per"] += int(not okp)
    print("δ=0.02 相位通道分解（PER，n=56/档）")
    print("SNR  | B(STO)  B_phi  | A(SFO)  A_phi")
    for lv in LEVELS:
        r = [agg[lv][t]["per"] / agg[lv][t]["n"]
             for t in ("B", "B_phi", "A", "A_phi")]
        print("%4d | %.3f   %.3f  | %.3f   %.3f" % (lv, *r))
    print("\n段间相位 φ_k 去均值后（前3帧，B vs A，rad）：")
    for p in phi_rec:
        print(" f%d B:%s" % (p["frame"], p["phiB"][:12]))
        print("    A:%s" % (p["phiA"][:12]))


def _rot(seg_c, nu0, pre, psym, m0):
    pad = (pre + 5) * NF
    seg_p = np.concatenate((np.zeros(pad, dtype=np.complex128),
                            seg_c[m0:]))
    return seg_p * np.exp(-2j * np.pi * nu0
                          * np.arange(len(seg_p)) / NF)


if __name__ == "__main__":
    main()

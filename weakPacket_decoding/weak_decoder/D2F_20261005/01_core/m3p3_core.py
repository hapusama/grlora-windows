# -*- coding: utf-8 -*-
"""D2F 第三仲裁列（m3p3，2026-10-05）：turbo 相位跨符号相干重读出。

动机（用户令"把 DeRa 式精细相位对齐设计到系统里"后的源码审计）：
  ① 式(22) 的段内 V1/V2 相干合并 A 列已有（port v2 f_proj+t_proj +
     ML φ̂0，native SER .002）；每符号漂移相位项由 m3p2 斜率梯预旋
     塌缩（梯内残差 ≤0.005 → 相位漂移 ≤0.27 rad → ≤0.1dB）——段内
     精细对齐已闭环；
  ② 真正的漏勺 = **跨符号相干**：A 列逐符号独立取幅度峰、B 列非相干、
     DeRa 解码器同样没有——turbo 相位列（phase_track e3 已验：
     −24dB 残差帧回收 10.6%、零反伤）是超出 DeRa 解码器的那一级。

本文件：把 e3 turbo 装进 D2F 链——
  build_mt_chain：链上 seg_base（ν_rot 旋后、未预旋）→ payload-only
    测量 mt（2 行占位 + psym 实行，P=0 约定使 e3 的 [P+2:] 切片对齐）；
  d_column：soft κ-line（自拟合走动）→ τ 三支 track_phase（LOO φ̂）→
    coherent_rows（von Mises 卷积读出，坏 φ̂ 优雅回退）→ **链上判决**
    （M3.demap_judge：bin 轴与 A 列同轴，Δ0 宽搜索吸收约定差；
    CRC 过即选支，否则 loo_res 最优支）；
  ours_chain_decode3：m3p2 全流程 + A→B 全败后 D 列兜底（构造性零
    回归：D 只在失败帧触发；错误 CRC 撞过 ~2⁻¹⁶/支 可忽略）。
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
    os.path.join(_WD, "data", "experiments", "phase_track_20261002"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m3_core as M3                      # noqa: E402
import m3p2_core as M3P2                  # noqa: E402
import e1_common as C1                    # noqa: E402
import e2_arm as E2                       # noqa: E402
import e3_common as D3P                   # noqa: E402

NF = 4096
N = 1024


def build_mt_chain(seg_base, fr, ds):
    """链上 payload-only 测量（e2 unit_measure 的链上变体）。

    seg_base = ν_rot 旋后、未斜率预旋的段（(pre+5)·NF 零垫 + payload）。
    返回 mt = dict(P=0, psym, Ys, amp)：2 行占位 + psym 实行，
    e3 全部 [P+2:] 切片恰好落在实行上；κ-line 的 sis 起点常数被
    kap0/φ̂ 吸收。extract_raw 失败（窗越界）返回 None。"""
    pre, psym = fr["pre"], fr["psym"]
    k_rot = E2.ST["ROT"].shape[0]
    ys = [np.zeros((k_rot, N), np.complex64) for _ in range(2)]
    am = [0.3, 0.3]
    for k in range(psym):
        m = C1.extract_raw(seg_base, pre + 5 + k, ds, ds.ref_dn, 0)
        if m is None:
            return None
        dr = m["dr"].astype(np.complex64)
        X = np.fft.fft(dr)
        p2 = np.abs(X) ** 2
        keep = np.ones(N, bool)
        top = np.argsort(p2)[::-1][:6]
        for t in top:
            keep[t] = keep[(t + 1) % N] = keep[(t - 1) % N] = False
        s = float(np.sqrt(max(np.mean(p2[keep]), 1e-30)))
        yk = (np.fft.fft(dr[None, :] * E2.ST["ROT"], axis=1) / s
              ).astype(np.complex64)
        ys.append(yk)
        am.append(float(np.clip(
            np.sqrt(max((np.abs(yk) ** 2).max() - E2.NOISE_MAX, 0.3)),
            0.3, 30.0)))
    return dict(P=0, psym=psym, Ys=np.stack(ys), amp=np.array(am),
                n=N, ntot=2 + psym)


def d_column(mt, rows_a_log, f):
    """turbo 相位列：Pass1 = A 列行（log 域）→ κ-line/τ 支/RTS LOO φ̂ →
    von Mises 相干重读出 → 链上 demap_judge（Δ0 宽搜索 + CRC）。

    返回 dict(ok, ser, col='d', tau)——ok=True 为 CRC 通过支；全败时
    loo_res 最优支的判决（ok=False）。"""
    q = D3P.softmax_rows(rows_a_log)
    kap0, enu, kap_line, jstar = D3P.soft_kappa_line(mt, q)
    best, best_ok = None, False
    for tau in D3P.TAU_BRANCH:
        c = D3P.track_phase_at(mt, q, kap_line, jstar, tau)
        rows = D3P.coherent_rows(mt, c["phi"], c["var"], c["tau"],
                                 kap_line, jstar)
        # demap_judge→fast_evidence 期望功率域行（内部 10log10）；
        # D 行为 log-I₀ 域 ⇒ 先 exp 回功率域（argmax 不变，避免 log(log)）
        rows_p = np.exp(np.clip(
            rows - rows.max(axis=1, keepdims=True), -50.0, 0.0))
        ok, ser, d0 = M3.demap_judge(rows_p, np.zeros(mt["psym"], dtype=int),
                                     f)
        if ok:
            return dict(ok=True, ser=ser, col="d", tau=float(tau))
        if best is None or c["loo_res"] < best[0]:
            best = (float(c["loo_res"]), dict(ok=False, ser=ser, col="d",
                                              tau=float(tau)))
    return best[1]


def ours_chain_decode3(seg, det, f, kt, dd, ds):
    """m3p2 斜率梯两列仲裁 + D 列兜底。返回结构与 m3p2 一致（多 d 字段）。"""
    pre, psym = f["pre"], f["psym"]
    pad = (pre + 5) * NF
    seg_p0 = np.concatenate((np.zeros(pad, dtype=np.complex128),
                             seg[det["pay0"]:]))
    n_rot = np.arange(len(seg_p0))
    base = seg_p0 * np.exp(-2j * np.pi * det["nu_rot"] * n_rot / NF)
    n_win = np.arange(NF)
    out = {}
    best_fail = None
    rows_a0_log = None
    for sl in M3P2.SLOPE_LADDER:
        if sl == 0.0:
            seg_p = base
        else:
            seg_p = base.copy()
            for s in range(psym):
                w0 = (pre + 5 + s) * NF
                seg_p[w0:w0 + NF] *= np.exp(
                    -2j * np.pi * sl * (s - (psym - 1) / 2.0) * n_win / NF)
        rows_a = dd.demod_payload(seg_p, pre + 5, psym)[1]
        if rows_a0_log is None:
            rows_a0_log = np.log(rows_a + 1e-30)
            rows_a0_log -= rows_a0_log.max(axis=1, keepdims=True)
        rows_b = kt.demod_payload(seg_p, pre + 5, psym, readout="viterbi")
        for tag, rows in (("a", rows_a), ("b", rows_b)):
            ok, ser, d0 = M3.demap_judge(rows, np.zeros(psym, dtype=int), f)
            if tag == "a" and "a" not in out:
                out["a"] = dict(ok=ok, ser=ser, d0=d0, slope=sl)
            if ok:
                out.setdefault("u", dict(ok=True, ser=ser, col=tag,
                                         slope=sl))
                return out
            if best_fail is None:
                best_fail = dict(ok=False, ser=ser, col=tag)
    if "a" not in out:
        out["a"] = best_fail
    out.setdefault("u", best_fail)
    # ---- D 列兜底（A→B 全败才触发；失败帧上只可能改善）----
    try:
        mt = build_mt_chain(base, f, ds)
        if mt is not None:
            dres = d_column(mt, rows_a0_log, f)
            out["d"] = dres
            if dres["ok"]:
                out["u"] = dict(ok=True, ser=dres["ser"], col="d",
                                slope=0.0, tau=dres["tau"])
    except Exception:
        out["d"] = dict(ok=False, ser=-1, col="d", err=1)
    return out

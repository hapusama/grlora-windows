# -*- coding: utf-8 -*-
"""M3+2 核心扩展（2026-10-04 深夜）：走动斜率梯仲裁列。

诊断（m3p 终战在途数据 + 快测）：
  ① m3_core 的 nu_rot 补偿门 |δ̂|≥0.04 是为 dep2 噪声底设的——真实
     δ=0.02 低于门 ⇒ **链从未施加任何走动补偿**（dc=0.000 实测）；
  ② 绝对斜率梯快测（10 个 crc_rej 单元）：slope∈{0.01,0.02} 救回
     2/10（救点都在 dhat 附近）——de-walk 有效但只覆盖两成；
  ③ δ=0.02 下 B 列(TREL 走格) 稳定强于 A 列(port 定长切分)：−18 档
     .615 vs .904 ⇒ 剩余缺口=port 两段切分点随 SFO 漂移（下一层修法，
     本文件不做）。

本扩展：ours_chain_decode2 = 原两列仲裁 + **斜率梯 {0, .01, .02, .03}
× 两列** 的 CRC 逐级仲裁（绝对梯，接收机合法、无 GT；早退保平均成本；
δ=0 帧首候选 slope=0 即过 ⇒ 零回归）。候选预算：4×2×Δ0(7)=56 判次
vs DeRa 25——错误 CRC 撞过 ~2⁻¹⁶/候选，可忽略（§D5 同款声明）。

⚠️ 本文件为 method_m3p_20261005 档案副本；权威原件在
data/experiments/e2e_final_20261004/m3p2_core.py。
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
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m3_core as M3                      # noqa: E402

NF = 4096
SLOPE_LADDER = (0.0, 0.01, 0.02, 0.03)


def ours_chain_decode2(seg, det, f, kt, dd):
    """m3_core.ours_chain_decode 的斜率梯版：per-symbol de-walk 预旋
    （绝对斜率 sl，中心 (psym−1)/2）后跑两列，CRC 逐级仲裁早退。"""
    pre, psym = f["pre"], f["psym"]
    pad = (pre + 5) * NF
    seg_p0 = np.concatenate((np.zeros(pad, dtype=np.complex128),
                             seg[det["pay0"]:]))
    n_rot = np.arange(len(seg_p0))
    base = seg_p0 * np.exp(-2j * np.pi * det["nu_rot"] * n_rot / NF)
    n_win = np.arange(NF)
    out = {}
    best_fail = None
    for sl in SLOPE_LADDER:
        if sl == 0.0:
            seg_p = base
        else:
            seg_p = base.copy()
            for s in range(psym):
                w0 = (pre + 5 + s) * NF
                seg_p[w0:w0 + NF] *= np.exp(
                    -2j * np.pi * sl * (s - (psym - 1) / 2.0) * n_win / NF)
        rows_a = dd.demod_payload(seg_p, pre + 5, psym)[1]
        rows_b = kt.demod_payload(seg_p, pre + 5, psym, readout="viterbi")
        for tag, rows in (("a", rows_a), ("b", rows_b)):
            ok, ser, d0 = M3.demap_judge(rows, np.zeros(psym, dtype=int), f)
            if tag == "a":
                out.setdefault("a", dict(ok=ok, ser=ser, d0=d0, slope=sl))
            if ok:
                out.setdefault("u", dict(ok=True, ser=ser, col=tag,
                                         slope=sl))
                return out
            if best_fail is None:
                best_fail = dict(ok=False, ser=ser, col=tag)
    if "a" not in out:
        out["a"] = best_fail
    out.setdefault("u", best_fail)
    return out

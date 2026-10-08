# -*- coding: utf-8 -*-
"""M3+ 核心库（2026-10-04 深夜）：终判链 = dep4k16 检测 + 连续锚 + A→B 仲裁。

相对 m3_core（dep2 版）的两处替换（其余逐字复用 M3）：
  ① 检测：盲对齐胜点上的确认从 dep2 confirm → **dep4k16 两级验证**
     （m2c_core.score_dep4，非相干提名 K16 + 全场两群相干验证）。
     门限 = m2c H0 池化 13.74dB@1e-3（跨 δ/P 池化单门限，接收机不知 δ；
     H0 跨 δ 中位漂移 0.06dB 已验；来源 m2c_roc_results.json 13.733/
     13.735/13.752 三档均值）。
  ② 锚：胜点提名的连续 ν̂（m2d_core.nominate_refine，池化幅度谱抛物线
     精化，native P8 误差 ≤0.011）替换 0.5 格 ν_c——只作 demod 频旋锚，
     不进检测统计量（门限校准不受影响）。δ̂ 沿用 dep4 验证格 argmax
     （0.01 步；nu_rot 门 0.04 下步长足够）。
铁律不变：链内无 GT；demap_judge/nu_rot_of/ours_chain_decode 复用 m3_core。
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
import m2c_core as MC2                    # noqa: E402
import m2d_core as MD5                    # noqa: E402

NF = 4096
DEP4_THR = 10 ** (13.74 / 10.0)           # 池化单门限（声明见上）


def dep4_blind_detect(seg, pre, thr=None):
    """OURS-M3+ 检测：门扫 + dep2-lite 筛查（复用 M3._scan_scores）→
    胜点 dep4k16 全统计量过门 → 连续锚。返回 det dict / None。"""
    thr = DEP4_THR if thr is None else thr
    best = None
    for ev in M3.gate_scan(seg, pre)[:3]:
        hs0 = ev["start"] + int((pre + 4.25) * NF)
        if hs0 + 10 * NF > len(seg) or hs0 < int((pre + 5.25) * NF):
            continue
        sc, _bd, _bk, _dg = M3._scan_scores(seg, hs0, pre,
                                            M3.SCAN_SCREEN, lite=True)
        i = int(np.argmax(sc))
        if best is None or sc[i] > best[1]:
            best = (int(hs0 + M3.SCAN_SCREEN[i]), float(sc[i]))
    if best is None:
        return None
    hs_b = best[0]
    o = MC2.score_dep4(seg[None, :], hs_b, pre, k_list=(16,),
                       ret_diag=True)
    sc16 = float(o["score_k16"][0])
    if not np.isfinite(sc16) or sc16 < thr:
        return None
    r = int(np.argmax(o["best_per_nom"][0]))
    nu_ref, _cols = MD5.nominate_refine(seg[None, :], hs_b, pre, k_max=16)
    return dict(score=sc16, hs=int(hs_b),
                pay0=int(hs_b) + 8 * NF,
                nu0h=float(nu_ref[0, r]),
                dhat=float(o["dhat_k16"][0]),
                khat=0.0, c_e=(pre - 1) / 2.0)

# -*- coding: utf-8 -*-
"""D2：CFAR 2D (q, δ) 搜索膨胀标定——cert-grid 统计量的 H0 MC + 闭式对照。

cert-grid = cert 冻结模板（θ/位置二次模板）+ δ 网格搜索（41 格）+ κ-FFT
（256 点）——可部署闭式 CFAR 的搜索膨胀对象。H0 下 σ² 已知时单格 ~Exp(1)；
1D（仅 q）上穿闭式 γ−½lnγ=ln(c/FAR)（d1 U4 验证 1.12%@1e-3）；2D 膨胀 =
MC 精确分位（20k 样本域试验，复用 dep 的缓存 TW 路径）+ σ̂² 估计涨落
（远子格 2.5k 格 ⇒ χ² 涨落 <2.5%）。
→ d2_cfar2d.json
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C
import d1_battle as A
import d2_core as D

HERE = os.path.dirname(os.path.abspath(__file__))
FARS = (1e-2, 1e-3)
N_MC = 20000


def main():
    t0 = time.time()
    frames = A.build_frames()
    f = [fr for fr in frames if fr["pre"] == 16][0]
    pre = 16
    inj = D.resample_sfo(f["seg"], f["lead"] - (pre + 4.25) * D.NF,
                         D.eps_of_delta(0.02))
    tm = D.clean_template_q(inj, f["lead"], pre)
    wins = C.field_windows(f["lead"], pre)
    idx, b = C.frame_slots(pre)
    c_pre, c_sfd = D._centroids(pre)
    o_known = np.zeros(len(b))
    o_known[pre] = D.O_SYNC[0]
    o_known[pre + 1] = D.O_SYNC[1]
    n_arr = np.arange(D.NF)
    TWc = D._tw_walk(pre)
    th = tm["theta"]
    conj = tm["conj_dn"]
    sqb_th = np.sqrt(b) * np.exp(-1j * th)
    if conj:
        sqb_th[pre + 2:] = np.conj(sqb_th[pre + 2:])
    # 模板锚（cert：冻结 ν0/mirror）
    nu0 = tm["nu0"] + c_pre * tm["delta"]        # 轨迹在 c_pre 的位置
    mh = tm["pos"][pre + 2] + (tm["delta"]) * (idx[pre + 2] - c_sfd)
    B = 8
    l_arr = np.array([w[2] for w in wins], dtype=float)
    norm = float(np.sum(b * l_arr))          # var(T) = Σ b_j·l_j·σ_w²
    rng = np.random.default_rng(20261003)
    maxv = np.empty(N_MC)
    for t in range(N_MC // B):
        segs = (rng.standard_normal((B, len(f["seg"])))
                + 1j * rng.standard_normal((B, len(f["seg"])))) \
            * np.sqrt(0.5)
        Xall = np.empty((B, len(b), len(D.DGRID)), dtype=np.complex128)
        for j, (s, r, l) in enumerate(wins):
            base = (nu0 + o_known[j]) if j < pre + 2 else mh
            bp = np.exp(-2j * np.pi * base * n_arr[None, :l] / D.NF)
            Wj = (segs[:, s:s + l] * r[None, :l] * bp).astype(np.complex64)
            Xall[:, j] = Wj @ TWc[j]
        x = Xall * sqb_th[None, :, None]
        Fm = np.abs(np.fft.fft(x, C.N_FINE, axis=1))
        maxv[t * B:(t + 1) * B] = Fm.reshape(B, -1).max(axis=1) ** 2 / norm
        if (t + 1) % 1000 == 0:
            print("  %d/%d (%.0fs)" % (t + 1, N_MC, time.time() - t0),
                  flush=True)

    idxb, bb = C.frame_slots(8)
    cup = C.upcrossing_c(idxb, bb)
    res = dict(n_mc=N_MC, upcross_c=float(cup),
               max_mean=float(maxv.mean()), max_var=float(maxv.var()))
    for far in FARS:
        emp = float(np.quantile(maxv, 1 - far))
        clo1d = C.cfar_threshold(far, cup)
        res["FAR=%g" % far] = dict(
            emp_2d=emp, closed_1d=clo1d,
            inflation_db=10 * np.log10(emp / clo1d),
            far_at_1d_closed=float(np.mean(maxv > clo1d)),
            far_at_2d_emp_minus=float(np.mean(maxv > emp)))
    json.dump(res, open(os.path.join(HERE, "d2_cfar2d.json"), "w"),
              indent=1)
    print(json.dumps(res, indent=1, default=str))
    print("→ d2_cfar2d.json (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()

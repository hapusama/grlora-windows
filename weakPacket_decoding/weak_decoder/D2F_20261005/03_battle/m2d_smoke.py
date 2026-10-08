# -*- coding: utf-8 -*-
"""M2d/dep5 冒烟（2026-10-04 夜）。五项（设计文档 §5）：
  S0 native：28 帧全过门限 + δ̂ 对齐干净模板 + 锚 |ν̂−ν0| 分布；
  S1 离格聚焦平坦性：δ∈{0, 0.015, 0.087} 无加噪聚焦增益，档间差 ≤0.1dB
     （细轴离格免疫的量化验证，杀判据 0.3dB）；
  S2 H0 双报：合成 AWGN 经验 1e-2 分位 vs 解析参考门限（bias 报告）。
     注：score 对绝对功率不变（σ̂² 自归一），H0 只跑一档；
  S3 深端 H1：δ∈{0.015, 0.087} @−32dB，2 种子，过门率与中位裕量；
  S4 K 召回：@−34dB、δ=0.015，K∈{2,4,8,16} 的 |ν̂−ν0|≤0.35 召回率
     （主臂 K 档预注册依据）。
结果写当前目录 m2d_smoke_results.json（相对路径，从本目录运行）。

⚠️ method_m3p_20261005 档案副本；运行请用权威原件
（data/experiments/dep_anchor_20261004/m2d_smoke.py，从该目录跑，
结果落在原目录与既有记录同处）。裁定见 05_docs/dep5 设计文档 §6。
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_battle as A
import d2_core as D
import m2d_core as W

SEED_CONST = 20261003


def inj_of(fi, delta, frames):
    f = frames[fi]
    pre = f["pre"]
    eps = D.eps_of_delta(delta)
    return D.resample_sfo(f["seg"], f["lead"] - (pre + 4.25) * 4096, eps)


def noise_add(inj, S, N0, level, seed, salt):
    rng = np.random.default_rng((SEED_CONST * 7919
                                 + (int(level) + 100) * 131
                                 + seed * 17 + salt * 7919) % (2 ** 31))
    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
    return inj + ((rng.standard_normal(len(inj))
                   + 1j * rng.standard_normal(len(inj)))
                  * np.sqrt(p_add / 2.0))


def main():
    t0 = time.time()
    frames = A.build_frames()
    by_pre = {}
    for i, f in enumerate(frames):
        by_pre.setdefault(f["pre"], []).append(i)
    res = {"frames": len(frames), "pres": sorted(by_pre)}

    def batch(fi_list, delta=None, level=None, seed=0):
        segs = []
        for fi in fi_list:
            f = frames[fi]
            seg = f["seg"] if delta is None else inj_of(fi, delta, frames)
            if level is not None:
                S, N0 = A.snr_parts(seg)
                seg = noise_add(seg, S, N0, level, seed, f["hs"] % 4099)
            segs.append(seg)
        return (np.stack(segs), frames[fi_list[0]]["lead"],
                frames[fi_list[0]]["pre"])

    # ---------------- S0 native ----------------
    s0 = {"pass": True, "low": [], "dhat": [], "nu_err": [], "score_min": 1e9}
    for pre, lst in by_pre.items():
        segs, lead, _ = batch(lst)
        o = W.score_dep5(segs, lead, pre, ret_diag=True)
        for i, fi in enumerate(lst):
            sc = float(o["score_k8"][i])
            s0["dhat"].append(float(o["dhat_k8"][i]))
            s0["nu_err"].append(min(abs(o["nu_ref"][i, :4]
                                        - frames[fi]["tm"]["nu0"])))
            s0["score_min"] = min(s0["score_min"], sc)
            if sc < 15.0:
                s0["pass"] = False
                s0["low"].append(dict(frame=fi,
                                      score_db=10 * np.log10(sc)))
    s0.update(n=len(s0["dhat"]),
              dhat_med=float(np.median(s0["dhat"])),
              dhat_absmax=float(np.max(np.abs(s0["dhat"]))),
              nu_err_p95=float(np.quantile(s0["nu_err"], 0.95)),
              score_min_db=float(10 * np.log10(s0["score_min"])))
    res["S0_native"] = s0

    # ---------------- S1 离格聚焦平坦性 ----------------
    s1 = {"pass": True, "gain_db": {}}
    for dval in (0.0, 0.015, 0.087):
        gains = []
        for pre, lst in by_pre.items():
            segs, lead, _ = batch(lst, delta=dval)
            o = W.score_dep5(segs, lead, pre)
            gains.extend((10 * np.log10(np.maximum(
                o["score_k8"], 1e-30))).tolist())
        s1["gain_db"]["%g" % dval] = float(np.median(gains))
    g = s1["gain_db"]
    s1["spread_db"] = max(g.values()) - min(g.values())
    if s1["spread_db"] > 0.1 or min(g.values()) < 10.0:
        s1["pass"] = False
    res["S1_offgrid"] = s1

    # ---------------- S2 H0 双报（score 尺度不变 ⇒ 单档） ----------------
    s2 = {}
    sc_all = {k: [] for k in (2, 8, 16)}
    for pre, lst in by_pre.items():
        f0 = frames[lst[0]]
        L = len(f0["seg"])
        rng = np.random.default_rng(SEED_CONST * 31 + 7)
        for _ in range(6):
            segs = ((rng.standard_normal((len(lst), L))
                     + 1j * rng.standard_normal((len(lst), L)))
                    * np.sqrt(0.5))
            o = W.score_dep5(segs, f0["lead"], pre)
            for k in sc_all:
                sc_all[k].extend((10 * np.log10(np.maximum(
                    o["score_k%d" % k], 1e-30))).tolist())
    for k, v in sc_all.items():
        v = np.array(v)
        _, info = W.thr_analytic_dep5(8, 1e-2, k_nom=k)
        s2["k%d" % k] = dict(n=len(v),
                             emp_thr_db=float(np.quantile(v, 0.99)),
                             ana_db=info["thr_union_db"])
        s2["k%d" % k]["bias_db"] = (s2["k%d" % k]["emp_thr_db"]
                                    - info["thr_union_db"])
    res["S2_h0"] = s2

    # ---------------- S3 深端 H1 @−32 ----------------
    s3 = {}
    _, info8 = W.thr_analytic_dep5(8, 1e-2, k_nom=8)
    thr_ref = info8["thr_union_db"]
    for dval in (0.015, 0.087):
        sc = []
        for seed in (0, 1):
            for pre, lst in by_pre.items():
                segs, lead, _ = batch(lst, delta=dval, level=-32, seed=seed)
                o = W.score_dep5(segs, lead, pre)
                sc.extend((10 * np.log10(np.maximum(
                    o["score_k8"], 1e-30))).tolist())
        sc = np.array(sc)
        s3["d%g" % dval] = dict(n=len(sc), med_db=float(np.median(sc)),
                                pass_frac=float(np.mean(sc > thr_ref)),
                                thr_ref_db=thr_ref)
    res["S3_deep_h1"] = s3

    # ---------------- S4 K 召回 @−34 ----------------
    s4 = {}
    for pre, lst in by_pre.items():
        f0 = frames[lst[0]]
        for k in (2, 4, 8, 16):
            hit = tot = 0
            for seed in (0, 1):
                segs = []
                for fi in lst:
                    f = frames[fi]
                    inj = inj_of(fi, 0.015, frames)
                    S, N0 = A.snr_parts(inj)
                    segs.append(noise_add(inj, S, N0, -34, seed,
                                          f["hs"] % 4099))
                segs = np.stack(segs)
                nu_ref, _ = W.nominate_refine(segs, f0["lead"], pre,
                                              k_max=16)
                for i, fi in enumerate(lst):
                    nu0 = frames[fi]["tm"]["nu0"]
                    hit += int(np.min(np.abs(nu_ref[i, :k] - nu0)) <= 0.35)
                    tot += 1
            s4["P%d_k%d" % (pre, k)] = dict(recall=hit / tot, n=tot)
    res["S4_k_recall"] = s4

    res["runtime_s"] = time.time() - t0
    json.dump(res, open("m2d_smoke_results.json", "w"),
              indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:4000])
    print("→ m2d_smoke_results.json")


if __name__ == "__main__":
    main()

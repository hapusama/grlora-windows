# -*- coding: utf-8 -*-
"""D1 单元测试（击杀开关）：keystone=δ网格一致性、聚焦增益、CFAR 闭式 vs MC
（1e5 H0，<5%）、合成检出率 100%。全部结果落 d1_unit_results.json。
用法：python d1_unit.py [U1|U2|U3|U4|U5|ALL]"""
import json
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # 小矩阵多线程 BLAS 在本机抖动（37x 0.1s vs 17s）

import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C

RES = {}
TL = lambda: time.time()


def u1_template_roundtrip():
    """U1：干净合成帧 → clean_template 回收 ν0/o/δ/θ/conj。"""
    t0 = TL()
    out = {}
    for d in (0.0, 0.03):
        seg, hs = C.synth_frame(pre=8, delta=d, noise=False)
        tm = C.clean_template(seg, hs, 8)
        out["d=%g" % d] = dict(
            nu0_err=abs(tm["nu0"] - C.SYN["nu0"]),
            o_sync_err=max(abs(tm["o"][8] - 24), abs(tm["o"][9] - 32)),
            o_sfd=round(float(tm["o"][10]), 3),
            delta_err=abs(tm["delta"] - d),
            conj_dn=bool(tm["conj_dn"]),
            theta_sfd_dev=float(np.max(np.abs(
                np.angle(np.exp(1j * (tm["theta"][10:]
                                       - np.mean(C.SYN["theta_sfd"]))))))))
    ok = all(v["nu0_err"] < 0.03 and v["o_sync_err"] < 0.05
             and v["delta_err"] < 0.003 and v["conj_dn"]
             and abs(v["theta_sfd_dev"]) < 0.15
             for v in out.values())
    print("U1 模板回收", "PASS" if ok else "FAIL", out, "%.1fs" % (TL() - t0))
    return ok


def _focus_gain(delta, pre=8):
    seg, hs = C.synth_frame(pre=pre, delta=delta, noise=False)
    tm = C.clean_template(seg, hs, pre)
    wins = C.field_windows(hs, pre)
    idx, b = C.frame_slots(pre)
    p = tm["nu0"] + tm["o"] + tm["sgn"] * idx * tm["delta"]
    x = np.array([C.row_dtft(seg, wins[j][0], wins[j][1], wins[j][2],
                             p[j])[0] for j in range(len(b))])
    if tm["conj_dn"]:
        x = x.copy()
        x[pre + 2:] = np.conj(x[pre + 2:])
    x = np.sqrt(b) * x * np.exp(-1j * tm["theta"])
    F = np.abs(np.fft.fft(x, C.N_FINE))
    coh = float(F.max() ** 2) / float(b.sum())
    single = float(np.median(np.abs(x) ** 2 / b))
    return 10 * np.log10(coh / single)


def u2_focus_gain():
    """U2：无噪全场聚焦增益 ≥ 10log10(12.25)−0.3dB（δ 全网格匹配）。"""
    t0 = TL()
    gains = {d: round(_focus_gain(d), 3)
             for d in (0.0, 0.01, 0.03, 0.082)}
    thr = 10 * np.log10(12.25) - 0.3
    ok = all(g >= thr for g in gains.values())
    print("U2 聚焦增益 %.2f dB 门限 |" % thr, gains, "PASS" if ok else "FAIL",
          "%.1fs" % (TL() - t0))
    return ok


def u3_keystone_vs_grid():
    """U3：keystone（κ-bank×剪切重标度+单 FFT）≈ δ 网格精确和。"""
    t0 = TL()
    rows = []
    for d in (0.0, 0.03, 0.082):
        for kap in (0.0, 0.21):
            seg, hs = C.synth_frame(pre=8, delta=d, kappa=kap, snr_db=-8.0,
                                    seed=7)
            # 模板在干净同参帧上冻结（battle 同哲学）
            cseg, chs = C.synth_frame(pre=8, delta=d, kappa=kap, noise=False)
            tm = C.clean_template(cseg, chs, 8)
            sg = C.score_grid(seg, hs, 8, tm)
            sk = C.score_keystone(seg, hs, 8, tm)
            rows.append(dict(delta=d, kappa=kap,
                             grid=round(sg["score_db"], 2),
                             ks=round(sk["score_db"], 2),
                             ddB=round(sk["score_db"] - sg["score_db"], 2),
                             d_grid=round(sg["delta"], 4),
                             d_ks=round(sk["delta"], 4),
                             k_ks=round(sk["kappa"], 3),
                             k_grid=round(sg["kappa"], 3)))
            print("  δ=%.3f κ=%.2f grid=%.2f ks=%.2f (Δ%.2f) δ̂g=%.3f "
                  "δ̂k=%.3f" % (d, kap, sg["score_db"], sk["score_db"],
                               sk["score_db"] - sg["score_db"],
                               sg["delta"], sk["delta"]), flush=True)
    ddB = [abs(r["ddB"]) for r in rows]
    dd = [abs(r["d_ks"] - r["d_grid"]) for r in rows]
    ok = max(ddB) <= 0.5 and max(dd) <= 0.012
    print("U3 keystone=网格", "PASS" if ok else "FAIL",
          "max|ΔdB|=%.2f max|Δδ|=%.4f (%.1fs)" % (max(ddB), max(dd),
                                                   TL() - t0))
    return ok


def u4_cfar_mc(n_mc=100000, chunk=500):
    """U4：CFAR 闭式 vs 经验分位（1e5 H0，偏差<5% 才过）。"""
    t0 = TL()
    cseg, chs = C.synth_frame(pre=8, delta=0.0, noise=False)
    tm = C.clean_template(cseg, chs, 8)
    # 快路径自校验（protocol §6：与逐单元版等价）
    rng = np.random.default_rng(123)
    chk = rng.standard_normal((8, (8 + 6) * C.NF)) \
        + 1j * rng.standard_normal((8, (8 + 6) * C.NF))
    chk *= 0.37
    s_fast, _, _ = C.score_cert_batch(chk, chs, 8, tm)
    s_slow = [C.score_ours(chk[i], chs, 8, tm, "cert")["score_db"]
              for i in range(8)]
    dv = float(np.max(np.abs(10 * np.log10(s_fast)
                             - np.array(s_slow))))
    print("  自校验 fast-vs-slow max Δ = %.2e dB" % dv, flush=True)
    assert dv < 1e-6
    scores = np.empty(n_mc)
    n0 = 0
    while n0 < n_mc:
        b = min(chunk, n_mc - n0)
        segs = (rng.standard_normal((b, (8 + 6) * C.NF))
                + 1j * rng.standard_normal((b, (8 + 6) * C.NF)))
        segs *= np.sqrt(0.55)  # 任意 σ_t
        s, _, _ = C.score_cert_batch(segs, chs, 8, tm)
        scores[n0:n0 + b] = s
        n0 += b
        if n0 % 10000 == 0:
            print("  MC %d/%d (%.0fs)" % (n0, n_mc, TL() - t0), flush=True)
    idx, bb = C.frame_slots(8)
    cup = C.upcrossing_c(idx, bb)
    rows = {}
    for far in (1e-2, 1e-3):
        q_emp = float(np.quantile(scores, 1.0 - far))
        gamma = C.cfar_threshold(far, cup)
        cell = C.cell_threshold(far)
        dev = abs(q_emp - gamma) / q_emp
        rows["far=%g" % far] = dict(
            emp_th=round(q_emp, 3), closed=round(gamma, 3),
            cell_exact=round(cell, 3),
            dev_pct=round(100 * dev, 2),
            emp_far_check=round(float(np.mean(scores > gamma)), 6))
        print("  FAR=%g：经验门限 %.3f 闭式(上穿 c=%.2f) %.3f 偏差 %.2f%% "
              "| 单格精确 %.3f | 实测FAR %.2e"
              % (far, q_emp, cup, gamma, 100 * dev, cell,
                 rows["far=%g" % far]["emp_far_check"]), flush=True)
    ok = all(0 <= rows[k]["dev_pct"] < 5.0 for k in rows)
    print("U4 CFAR 闭式 vs MC", "PASS" if ok else "FAIL",
          "%.1fs" % (TL() - t0))
    return ok, rows, float(cup)


def u5_detection(n_real=50):
    """U5：检出率 100%（闭式门限 @FAR=1e-3），δ×SNR 全组合。"""
    t0 = TL()
    cseg, chs = C.synth_frame(pre=8, delta=0.0, noise=False)
    tm = C.clean_template(cseg, chs, 8)
    idx, bb = C.frame_slots(8)
    thr = C.cfar_threshold(1e-3, C.upcrossing_c(idx, bb))
    rows = {}
    for d in (0.0, 0.01, 0.03, 0.082):
        for snr in (-18.0, -22.0):
            ct, ht = C.synth_frame(pre=8, delta=d, noise=False)
            tmd = C.clean_template(ct, ht, 8)
            hits = 0
            for r in range(n_real):
                seg, hs = C.synth_frame(pre=8, delta=d, snr_db=snr,
                                        seed=1000 + r)
                s, _, _ = C.score_cert_batch(seg[None, :], hs, 8, tmd)
                hits += int(s[0] > thr)
            rows["d=%.3f,snr=%g" % (d, snr)] = hits
            print("  δ=%.3f SNR=%g：检出 %d/%d" % (d, snr, hits, n_real),
                  flush=True)
    ok = all(v == n_real for v in rows.values())
    print("U5 检出率", "PASS" if ok else "FAIL", rows, "%.1fs" % (TL() - t0))
    return ok


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    res = {}
    if which in ("U1", "ALL"):
        res["U1"] = u1_template_roundtrip()
    if which in ("U2", "ALL"):
        res["U2"] = u2_focus_gain()
    if which in ("U3", "ALL"):
        res["U3"] = u3_keystone_vs_grid()
    if which in ("U4", "ALL"):
        ok, rows, neff = u4_cfar_mc()
        res["U4"] = ok
        res["U4_rows"] = rows
        res["U4_upcross_c"] = float(neff)
    if which in ("U5", "ALL"):
        res["U5"] = u5_detection()
    json.dump(res, open("d1_unit_results.json", "w"), indent=1,
              default=str)
    print("\n== 汇总 ==")
    for k, v in res.items():
        if k.startswith("U") and isinstance(v, bool):
            print(k, "PASS" if v else "FAIL")
    print("→ d1_unit_results.json")


if __name__ == "__main__":
    main()

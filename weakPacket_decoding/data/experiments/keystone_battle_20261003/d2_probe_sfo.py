# -*- coding: utf-8 -*-
"""D2 probe：induced-SFO 注入器自校验（诚实损伤检查）。

  P1 插值器精度：纯音解析对照（|err| dB，各频点）；
  P2 ε=0 恒等（控制组：注入路径零损伤的路径级验证）；
  P3 δ 读回：28 帧 × δ 档，clean_template_q 的 δ̂ vs 目标（≤0.005）；
  P4 二次相位律：匹配轨迹上 DTFT 相位 vs idx 的二次拟合系数 → q2 实测；
  P5 cert-q 无噪聚焦增益 vs δ（机制级平坦性，U2 的物理注入版）；
  P6 注入对 dep/dera 的零损伤旁证：δ=0.005（最小档）无噪分数 vs 原始。
→ d2_probe_sfo.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C
import d1_battle as A
import d2_core as D

HERE = os.path.dirname(os.path.abspath(__file__))
DELTAS = (0.005, 0.01, 0.02, 0.04, 0.082)
res = {}


def field_origin(lead, pre):
    return lead - (pre + 4.25) * D.NF


def main():
    t0 = time.time()
    # ---------------- P1 插值器精度 ----------------
    rng = np.random.default_rng(7)
    errs = {}
    for fc in (0.02, 0.1, 0.2, 0.24):
        n = np.arange(60000, dtype=np.float64)
        x = np.exp(2j * np.pi * fc * n)
        for eps in (8.2e-5, 2e-5):
            origin = 20000.0
            y = D.resample_sfo(x, origin, eps)
            t = origin + (n - origin) * (1 + eps)
            ref = np.exp(2j * np.pi * fc * t)
            mid = slice(1000, 59000)
            errs["f=%.2f eps=%.0e" % (fc, eps)] = float(
                10 * np.log10(np.mean(np.abs(y[mid] - ref[mid]) ** 2)))
    res["interp_err_db"] = errs
    print("P1 插值误差(dB)：", {k: round(v, 1) for k, v in errs.items()},
          flush=True)

    # ---------------- 帧集 ----------------
    print("冻结帧集…", flush=True)
    frames = A.build_frames()
    print("帧数 %d (%.0fs)" % (len(frames), time.time() - t0), flush=True)

    # ---------------- P2 ε=0 恒等 ----------------
    f = frames[0]
    seg0 = D.resample_sfo(f["seg"], field_origin(f["lead"], f["pre"]), 0.0)
    ident = float(np.max(np.abs(seg0 - f["seg"])))
    res["eps0_identity_maxabs"] = ident
    print("P2 ε=0 恒等 max|Δ| =", ident, flush=True)

    # ---------------- P3 δ 读回 + P4 q2 + P5/P6/P7 ----------------
    readback = {str(d): [] for d in DELTAS}
    q2_up = {str(d): [] for d in DELTAS}
    q2_sfd = {str(d): [] for d in DELTAS}
    gain_native = {str(d): [] for d in DELTAS}
    sig2_native = {str(d): [] for d in DELTAS}
    dera_gain = {str(d): [] for d in DELTAS}      # P7: DeRa 无噪分数
    dep_gain = {str(d): [] for d in DELTAS}       # P7: dep 无噪分数
    c2_native = []                                 # P4b: 未注入二次系数
    pres = []
    for fi, f in enumerate(frames):
        pre = f["pre"]
        pres.append(pre)
        origin = field_origin(f["lead"], pre)
        tm_nat = D.clean_template_q(f["seg"], f["lead"], pre)
        c2_native.append(tm_nat["c2"])
        for d_t in DELTAS:
            eps = D.eps_of_delta(d_t)
            inj = D.resample_sfo(f["seg"], origin, eps)
            tm = D.clean_template_q(inj, f["lead"], pre)
            readback[str(d_t)].append(tm["delta"])
            # P4：匹配轨迹相位（上族 / conj SFD 族）二次系数
            wins = C.field_windows(f["lead"], pre)
            idx, b = C.frame_slots(pre)
            ph_up, ph_sfd = [], []
            for j, (s, r, l) in enumerate(wins):
                pj = tm["nu0"] + tm["o"][j] + tm["sgn"][j] * idx[j] * tm["delta"]
                v = C.row_dtft(inj, s, r, l, pj)[0]
                if j < pre + 2:
                    ph_up.append(np.angle(v))
                else:
                    ph_sfd.append(np.angle(np.conj(v)))
            ph_up = np.unwrap(np.array(ph_up))
            ph_sfd = np.unwrap(np.array(ph_sfd))
            c2u = np.polyfit(idx[:pre + 2], ph_up, 2)[0]
            c2s = np.polyfit(idx[pre + 2:], ph_sfd, 2)[0]
            q2_up[str(d_t)].append(c2u)
            q2_sfd[str(d_t)].append(c2s)
            # P5：cert-q 无噪分数（本底 σ² 用远列——无噪段带外底）
            sc, _, s2 = C.score_cert_batch(inj[None, :], f["lead"], pre, tm)
            gain_native[str(d_t)].append(float(10 * np.log10(sc[0])))
            sig2_native[str(d_t)].append(float(s2[0]))
            # P7：DeRa / dep 无噪分数（物理律#5 实测：走动散焦）
            sd, _, _ = C.score_dera_batch(inj[None, :], f["lead"], pre, tm)
            dera_gain[str(d_t)].append(float(10 * np.log10(sd[0])))
            sp, _, _, _ = D.score_dep_batch(inj[None, :], f["lead"], pre, tm)
            dep_gain[str(d_t)].append(float(10 * np.log10(sp[0])))
        if (fi + 1) % 7 == 0:
            print("  %d/%d 帧 (%.0fs)" % (fi + 1, len(frames),
                                          time.time() - t0), flush=True)

    rb_err = {}
    for k in readback:
        arr = np.array(readback[k])
        arr0 = np.array([tm["delta"] for tm in
                         [D.clean_template_q(f["seg"], f["lead"], f["pre"])
                          for f in frames]]) if False else None
        rb_err[k] = dict(target=float(k), mean=float(arr.mean()),
                         std=float(arr.std()),
                         max_abs_err=float(np.max(np.abs(arr - float(k)))))
    res["delta_readback"] = rb_err
    print("P3 δ 读回：", {k: (round(v["mean"], 4), round(v["max_abs_err"], 4))
                         for k, v in rb_err.items()}, flush=True)

    # P3b：逐帧 (δ̂_inj − δ̂_native) vs 目标（注入净映射误差）
    nat_delta = []
    for f in frames:
        tm_nat = D.clean_template_q(f["seg"], f["lead"], f["pre"])
        nat_delta.append(tm_nat["delta"])
    nat_delta = np.array(nat_delta)
    rb_net = {}
    for k in readback:
        net = np.array(readback[k]) - nat_delta
        rb_net[k] = dict(mean=float(net.mean()), std=float(net.std()),
                         max_abs_err=float(np.max(np.abs(net - float(k)))))
    res["delta_readback_net"] = rb_net
    print("P3b 净读回(扣 native)：",
          {k: (round(v["mean"], 4), round(v["max_abs_err"], 4))
           for k, v in rb_net.items()}, flush=True)

    q2s = {}
    for k in q2_up:
        qu = np.array(q2_up[k])
        qs = np.array(q2_sfd[k])
        q2s[k] = dict(up_mean=float(qu.mean()), up_std=float(qu.std()),
                      sfd_mean=float(qs.mean()), sfd_std=float(qs.std()),
                      up_maxabs=float(np.max(np.abs(qu))),
                      sfd_maxabs=float(np.max(np.abs(qs))))
    res["q2_c2_raw"] = q2s
    res["c2_native"] = dict(mean=float(np.mean(c2_native)),
                            maxabs=float(np.max(np.abs(c2_native))))
    res["frame_pre"] = pres
    print("P4 c2 均值（up/sfd, rad/idx²）：",
          {k: (round(v["up_mean"], 4), round(v["sfd_mean"], 4))
           for k, v in q2s.items()},
          "| native c2 mean %.4f" % np.mean(c2_native), flush=True)

    gn = {k: dict(mean=float(np.mean(v)), min=float(np.min(v)),
                  max=float(np.max(v))) for k, v in gain_native.items()}
    res["cert_native_gain_db"] = gn
    res["cert_native_sig2"] = {k: float(np.mean(v))
                               for k, v in sig2_native.items()}
    print("P5 cert-q 无噪分数(dB)：",
          {k: round(v["mean"], 1) for k, v in gn.items()}, flush=True)

    # P7：DeRa / dep 无噪分数，按 P 分组（物理律#5 实测裁决）
    pres_a = np.array(pres)
    dg_by_p = {}
    pg_by_p = {}
    cg_by_p = {}
    for k in dera_gain:
        dg = np.array(dera_gain[k])
        pg = np.array(dep_gain[k])
        cg = np.array(gain_native[k])
        d0 = np.array(dera_gain["0.005"])
        for p in (8, 16, 32):
            m = pres_a == p
            dg_by_p.setdefault(str(p), {})[k] = float(np.mean(dg[m]))
            pg_by_p.setdefault(str(p), {})[k] = float(np.mean(pg[m]))
            cg_by_p.setdefault(str(p), {})[k] = float(np.mean(cg[m]))
    res["dera_native_db_byP"] = dg_by_p
    res["dep_native_db_byP"] = pg_by_p
    res["cert_native_db_byP"] = cg_by_p
    for p in ("8", "16", "32"):
        print("P7 P=%s dera(dB)：" % p,
              {k: round(v, 2) for k, v in dg_by_p[p].items()},
              "| dep:", {k: round(v, 2) for k, v in pg_by_p[p].items()},
              flush=True)
    law5 = {str(p): {str(d): 10 * np.log10(D.dera_defocus_loss(d, int(p)))
                     for d in DELTAS} for p in (8, 16, 32)}
    res["law5_closed_db"] = law5
    print("定律5 闭式(dB)：", {p: {k: round(v, 2) for k, v in law5[p].items()}
                             for p in law5}, flush=True)

    json.dump(res, open(os.path.join(HERE, "d2_probe_sfo.json"), "w"),
              indent=1)
    print("完成 %.0fs → d2_probe_sfo.json" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

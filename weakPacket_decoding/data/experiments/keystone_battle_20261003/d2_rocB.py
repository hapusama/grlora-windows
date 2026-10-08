# -*- coding: utf-8 -*-
"""D2 Battle B ROC 分析：三臂 × δ 的 Pd=0.9 门限曲线、配对 w/l、
定律5 对照（DeRa 实测散焦 vs Dirichlet/二次律双闭式）、per-P 分解。
→ d2_rocB_results.json / d2_battleB_table.md
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C
import d2_core as D

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "d2_battleB_checkpoint.jsonl")
FARS = (1e-3,)
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.082)
LEVELS = [-26, -28, -30, -32, -34, -36, -38]
H0_LEVELS = [-28, -34, -38]
ARMS = ("cert", "dep", "dera")


def load():
    h1 = {di: {a: {lv: [] for lv in LEVELS} for a in ARMS} for di in
          range(len(DELTAS))}
    h0 = {di: {a: {lv: [] for lv in H0_LEVELS} for a in ARMS} for di in
          range(len(DELTAS))}
    pairs = []
    for line in open(CKPT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") == "h1":
            di = r["di"]
            rec = (r["cert"], r["dep"], r["dera"], r["frame"], r["seed"],
                   r["pre"], r.get("dhat"))
            pairs.append((di, r["level"]) + rec)
            for a in ARMS:
                h1[di][a][r["level"]].append(r[a])
        elif r.get("kind") == "h0":
            for a in ARMS:
                h0[r["di"]][a][r["level"]].append(r[a])
    return h1, h0, pairs


def snr_at_pd09(pds, lvs=LEVELS):
    lvs = np.array(lvs, dtype=float)
    p = np.array(pds)
    for i in range(len(p) - 1):
        if p[i] >= 0.9 > p[i + 1]:
            return float(lvs[i] + (p[i] - 0.9) / (p[i] - p[i + 1])
                         * (lvs[i + 1] - lvs[i]))
    if p[0] < 0.9:
        return float(lvs[0])
    return float(lvs[-1])


def main():
    h1, h0, pairs = load()
    n_h1 = {di: len(h1[di]["cert"][LEVELS[0]]) for di in range(len(DELTAS))}
    n_h0 = {di: sum(len(h0[di]["cert"][lv]) for lv in H0_LEVELS)
            for di in range(len(DELTAS))}
    print("H1 每档 %s | H0 每 δ %s" % (n_h1, n_h0), flush=True)
    res = dict(n_h1=n_h1, n_h0=n_h0)

    # ---- H0 跨档/跨 δ 不变性检查（CFAR 性）----
    inv = {}
    for a in ARMS:
        meds = {di: {lv: float(np.median(h0[di][a][lv])) for lv in H0_LEVELS}
                for di in range(len(DELTAS))}
        lv_drift = max(max(m.values()) - min(m.values())
                       for m in meds.values())
        allv = [v for m in meds.values() for v in m.values()]
        inv[a] = dict(by_di=meds, level_drift=lv_drift,
                      di_drift=max(allv) - min(allv))
    res["h0_invariance"] = inv
    print("H0 中位漂移（档/δ）：%s"
          % {a: (round(inv[a]["level_drift"], 3), round(inv[a]["di_drift"], 3))
             for a in ARMS}, flush=True)

    # ---- 门限：主口径 = 逐 δ 池化（3 档）精确分位；cert 附闭式 ----
    idx, bb = C.frame_slots(8)
    cup = C.upcrossing_c(idx, bb)
    thr = {}
    for f in FARS:
        for a in ARMS:
            for di in range(len(DELTAS)):
                pool = np.concatenate([h0[di][a][lv] for lv in H0_LEVELS])
                thr[("t", a, di, f)] = float(np.quantile(pool, 1 - f))
        thr[("closed1d", f)] = float(C.cfar_threshold(f, cup))
    res["cert_closed_1d"] = {str(f): C.cfar_threshold(f, cup) for f in FARS}

    # ---- Pd 曲线 + Pd=0.9 门限 ----
    pd_tab = {}
    snr09 = {}
    for f in FARS:
        for a in ARMS:
            for di in range(len(DELTAS)):
                curve = [float(np.mean(np.array(h1[di][a][lv])
                                       > thr[("t", a, di, f)]))
                         for lv in LEVELS]
                pd_tab["%s@%g/d%s" % (a, f, di)] = curve
                snr09[("%s@%g" % (a, f), di)] = snr_at_pd09(curve)
    res["pd_curves"] = pd_tab

    # ---- 门限 vs δ 表 + 配对 w/l（同噪同窗，逐 δ）----
    tbl = {}
    wl = {}
    for f in FARS:
        for di in range(len(DELTAS)):
            tc = thr[("t", "cert", di, f)]
            tp = thr[("t", "dep", di, f)]
            td = thr[("t", "dera", di, f)]
            row = dict(cert=snr09[("cert@%g" % f, di)],
                       dep=snr09[("dep@%g" % f, di)],
                       dera=snr09[("dera@%g" % f, di)],
                       cert_minus_dera=snr09[("cert@%g" % f, di)]
                       - snr09[("dera@%g" % f, di)],
                       dep_minus_dera=snr09[("dep@%g" % f, di)]
                       - snr09[("dera@%g" % f, di)],
                       thr_cert_db=10 * np.log10(tc),
                       thr_dep_db=10 * np.log10(tp),
                       thr_dera_db=10 * np.log10(td))
            tbl["FAR=%g δ=%g" % (f, DELTAS[di])] = row
            w = l_ = b_ = 0
            wdp = ldp = bdp = 0
            for p in pairs:
                if p[0] != di or p[1] is None:
                    continue
                _di, _lv, c, dp_, dr = p[0], p[1], p[2], p[3], p[4]
                if c > tc and dr <= td:
                    w += 1
                elif c <= tc and dr > td:
                    l_ += 1
                elif c > tc and dr > td:
                    b_ += 1
                if dp_ > tp and dr <= td:
                    wdp += 1
                elif dp_ <= tp and dr > td:
                    ldp += 1
                elif dp_ > tp and dr > td:
                    bdp += 1
            wl["FAR=%g δ=%g" % (f, DELTAS[di])] = dict(
                cert_vs_dera=dict(win=w, loss=l_, both=b_),
                dep_vs_dera=dict(win=wdp, loss=ldp, both=bdp))
    res["pd09_table"] = tbl
    res["paired_wl"] = wl

    # ---- per-P 分解（P=8/16/32 各臂 Pd=0.9 与 DeRa 散焦实测 vs 双闭式）----
    perP = {}
    for f in FARS:
        for p_sel in (8, 16, 32):
            for a in ARMS:
                curves = []
                for di in range(len(DELTAS)):
                    vals = []
                    for p in pairs:
                        if p[0] == di and p[5] == p_sel:
                            pass
                    # 从 pairs 重建（level, arm 分组）
                # 简化：直接重新读 h1 by pre —— pairs 里有 pre
                sub = {}
                for p in pairs:
                    if p[5] != p_sel:
                        continue
                    sub.setdefault((p[0], p[1]), []).append(
                        {"cert": p[2], "dep": p[3], "dera": p[4]})
                curve = []
                for di in range(len(DELTAS)):
                    pd_l = []
                    for lv in LEVELS:
                        sc = [u[a] for (ddi, llv), us in sub.items()
                              if ddi == di and llv == lv for u in us]
                        t = thr[("t", a, di, f)]
                        pd_l.append(float(np.mean(np.array(sc) > t))
                                    if sc else np.nan)
                    curve.append(snr_at_pd09(pd_l))
                perP["P=%d %s@%g" % (p_sel, a, f)] = curve
    res["pd09_byP"] = perP

    # ---- DeRa 散焦实测（Pd=0.9 移位 + native 分数）vs 双闭式 ----
    probe = json.load(open(os.path.join(HERE, "d2_probe_sfo.json")))
    law5 = {}
    for p_sel in (8, 16, 32):
        base = perP["P=%d dera@%g" % (p_sel, FARS[0])][0]
        meas = [perP["P=%d dera@%g" % (p_sel, FARS[0])][di] - base
                for di in range(len(DELTAS))]
        law5["P=%d" % p_sel] = dict(
            meas_pd09_shift=meas,
            native_shift=[probe["dera_native_db_byP"][str(p_sel)][k]
                          - probe["dera_native_db_byP"][str(p_sel)]["0.005"]
                          for k in probe["dera_native_db_byP"][str(p_sel)]],
            dirichlet_db=[10 * np.log10(D.dera_defocus_dirichlet(d, p_sel))
                          for d in DELTAS],
            quad_db=[10 * np.log10(D.dera_defocus_loss(d, p_sel))
                     for d in DELTAS])
    res["law5_adjudication"] = law5

    json.dump(res, open(os.path.join(HERE, "d2_rocB_results.json"), "w"),
              indent=1, default=str)
    print(json.dumps(res["pd09_table"], indent=1, default=str)[:2500])
    print("→ d2_rocB_results.json")


if __name__ == "__main__":
    main()

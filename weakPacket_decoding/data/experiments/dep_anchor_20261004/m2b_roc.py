# -*- coding: utf-8 -*-
"""M2b ROC：dep3a/dep3b（m2b_battle）× cert/dep/dera（d2_battleB）× dep2
（m2_battle）同噪 join → Pd09 战表、门限、配对 w/l、真 GT 锚质量、δ̂ 伪峰。
→ m2b_roc_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "..", "keystone_battle_20261003",
                   "d2_battleB_checkpoint.jsonl")
NEW = os.path.join(HERE, "m2b_battle_checkpoint.jsonl")
M2CK = os.path.join(HERE, "m2_battle_checkpoint.jsonl")
FARS = (1e-3,)
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.082)
LEVELS = [-26, -28, -30, -32, -34, -36, -38]
H0_LEVELS = [-28, -34, -38]
ARMS = ("cert", "dep", "dera", "dep2", "dep3a", "dep3b")


def load():
    h1, h0 = {}, {}
    for path, tag in ((OLD, "old"), (M2CK, "dep2"), (NEW, "dep3")):
        win_ct = {}
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            kind = r.get("kind")
            if kind == "h1":
                k = (r["di"], r["level"], r["seed"], r["frame"])
                d = h1.setdefault(k, {})
                if tag == "old":
                    d.update(cert=r["cert"], dep=r["dep"], dera=r["dera"],
                             pre=r["pre"])
                elif tag == "dep2":
                    d.update(dep2=r["dep2"], dhat2=r["dhat"],
                             nu_err_old=r["nu_err"])
                else:
                    d.update(dep3a=r["dep3a"], dep3b=r["dep3b"],
                             dhat_a=r["dhat_a"], dhat_b=r["dhat_b"],
                             nu_err=r["nu_err"], acq=r["acq"])
            elif kind == "h0":
                k0 = (r["di"], r["level"], r["seed"], r["frame"])
                i = win_ct.get(k0, 0)
                win_ct[k0] = i + 1
                k = k0 + (i,)
                d = h0.setdefault(k, {})
                if tag == "old":
                    d.update(cert=r["cert"], dep=r["dep"], dera=r["dera"],
                             pre=r["pre"])
                elif tag == "dep2":
                    d.update(dep2=r["dep2"])
                else:
                    d.update(dep3a=r["dep3a"], dep3b=r["dep3b"])
    return h1, h0


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
    h1, h0 = load()
    n_h1 = sum(1 for v in h1.values() if "dep3a" in v and "cert" in v)
    n_h0 = sum(1 for v in h0.values() if "dep3a" in v and "cert" in v)
    print("join: H1 %d / H0 %d 单元" % (n_h1, n_h0), flush=True)
    res = dict(n_h1_join=n_h1, n_h0_join=n_h0)

    # ---- H0 不变性 ----
    inv = {}
    for a in ARMS:
        meds = {}
        for k, v in h0.items():
            if a in v:
                meds.setdefault((k[0], k[1]), []).append(v[a])
        allv = [np.median(x) for x in meds.values()]
        if allv:
            inv[a] = dict(di_drift=10 * np.log10(max(allv) / min(allv)),
                          n=len(allv) * 4)
    res["h0_invariance_db"] = inv

    # ---- 门限：主口径 per-(arm,δ,P) 池化精确分位；副口径跨 δ 池化 ----
    thr = {}
    for f in FARS:
        for a in ARMS:
            for di in range(len(DELTAS)):
                poolD = [v[a] for k, v in h0.items()
                         if k[0] == di and a in v]
                thr[("t", a, di, f)] = float(np.quantile(poolD, 1 - f)) \
                    if poolD else float("nan")
                for P_sel in (8, 16, 32):
                    poolP = [v[a] for k, v in h0.items()
                             if k[0] == di and a in v and v.get("pre") == P_sel]
                    thr[("tP", a, di, P_sel, f)] = \
                        float(np.quantile(poolP, 1 - f)) \
                        if len(poolP) > 300 else float("nan")
                # 跨 δ 池化 per P
                for P_sel in (8, 16, 32):
                    poolX = [v[a] for k, v in h0.items()
                             if a in v and v.get("pre") == P_sel]
                    thr[("tX", a, P_sel, f)] = \
                        float(np.quantile(poolX, 1 - f)) \
                        if len(poolX) > 1000 else float("nan")

    # ---- Pd09 + 配对 w/l ----
    pd_tab = {}
    wl = {}
    for f in FARS:
        for di in range(len(DELTAS)):
            row = {}
            for a in ARMS:
                curve = []
                for lv in LEVELS:
                    sc = [(v[a], v.get("pre")) for k, v in h1.items()
                          if k[0] == di and k[1] == lv and a in v]
                    if sc:
                        curve.append(float(np.mean([
                            val > thr[("tP", a, di, pre_, f)]
                            if not np.isnan(thr[("tP", a, di, pre_, f)])
                            else False for val, pre_ in sc])))
                row[a] = snr_at_pd09(curve) if curve else float("nan")
                # 跨 δ 池化门限口径
                curveX = []
                for lv in LEVELS:
                    sc = [(v[a], v.get("pre")) for k, v in h1.items()
                          if k[0] == di and k[1] == lv and a in v]
                    if sc:
                        curveX.append(float(np.mean([
                            val > thr[("tX", a, pre_, f)]
                            if not np.isnan(thr[("tX", a, pre_, f)])
                            else False for val, pre_ in sc])))
                row[a + "_Xthr"] = snr_at_pd09(curveX) if curveX \
                    else float("nan")
            for a in ("dep3a", "dep3b", "dep2"):
                row[a + "_minus_dera"] = row[a] - row["dera"]
                row[a + "_minus_dera_X"] = row[a + "_Xthr"] \
                    - row["dera_Xthr"]
            row["cert_minus_dera"] = row["cert"] - row["dera"]
            for a in ARMS:
                row["thr_%s_db" % a] = 10 * np.log10(thr[("t", a, di, f)])
            pd_tab["FAR=%g δ=%g" % (f, DELTAS[di])] = row
            # 配对 w/l（dep3a/dep3b/dep2 vs dera）
            for a in ("dep3a", "dep3b", "dep2"):
                w = l_ = b_ = 0
                for k, v in h1.items():
                    if k[0] != di or a not in v:
                        continue
                    t2 = thr[("tP", a, di, v.get("pre"), f)]
                    td = thr[("tP", "dera", di, v.get("pre"), f)]
                    if np.isnan(t2) or np.isnan(td):
                        continue
                    if v[a] > t2 and v["dera"] <= td:
                        w += 1
                    elif v[a] <= t2 and v["dera"] > td:
                        l_ += 1
                    elif v[a] > t2 and v["dera"] > td:
                        b_ += 1
                wl.setdefault("FAR=%g δ=%g" % (f, DELTAS[di]), {})[a] = \
                    dict(win=w, loss=l_, both=b_,
                         loss_rate=l_ / max(w + l_, 1))
    res["pd09_table"] = pd_tab
    res["paired_wl"] = wl

    # ---- per-P（δ=0）----
    perP = {}
    for P_sel in (8, 16, 32):
        for a in ARMS:
            t_a = thr[("tP", a, 0, P_sel, FARS[0])]
            if np.isnan(t_a):
                perP["P=%d %s" % (P_sel, a)] = float("nan")
                continue
            curve = []
            for lv in LEVELS:
                sc = [v[a] for k, v in h1.items()
                      if k[0] == 0 and k[1] == lv and a in v
                      and v.get("pre") == P_sel]
                curve.append(float(np.mean(np.array(sc) > t_a))
                             if sc else np.nan)
            perP["P=%d %s" % (P_sel, a)] = snr_at_pd09(curve)
    res["pd09_byP_delta0"] = perP

    # ---- 真 GT 锚质量 / 采集成功率 / δ̂ 伪峰 ----
    anch = {}
    for di in (0, 1, 3, 5):
        for lv in (-30, -34, -36):
            sc = [v["nu_err"] for k, v in h1.items()
                  if k[0] == di and k[1] == lv and "nu_err" in v]
            if not sc:
                continue
            sc = np.abs(np.array(sc))
            anch["δ=%g lv=%d" % (DELTAS[di], lv)] = dict(
                n=len(sc), acq_ok=float(np.mean(sc < 0.5)),
                nu_p50=float(np.quantile(sc, 0.5)),
                nu_p90=float(np.quantile(sc, 0.9)))
    res["anchor_true_gt"] = anch
    dhat = {}
    for a, key in (("dep3a", "dhat_a"), ("dep3b", "dhat_b"),
                   ("dep2", "dhat2")):
        for di in range(len(DELTAS)):
            for lv in (LEVELS[0], -30, -34):
                e = np.array([v[key] - DELTAS[di] for k, v in h1.items()
                              if k[0] == di and k[1] == lv and key in v])
                if not len(e):
                    continue
                dhat["%s δ=%g lv=%d" % (a, DELTAS[di], lv)] = dict(
                    n=len(e), p50=float(np.quantile(e, 0.5)),
                    frac_gt004=float(np.mean(np.abs(e) > 0.04)),
                    frac_gt008=float(np.mean(np.abs(e) > 0.08)))
    res["dhat_dist"] = dhat

    # ---- 确定性抽查：m2_core.dep2 重算 vs m2 checkpoint（噪声恒等）----
    det = []
    import d1_core as C
    import d1_battle as A
    import d2_core as D
    import m2_core as M2
    frames = A.build_frames()
    SEED_CONST = 20261003
    for (di, level, seed) in ((0, -30, 0), (3, -34, 1), (5, -26, 2)):
        old = {}
        for line in open(M2CK, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == "h1" and (r["di"], r["level"], r["seed"]) \
                    == (di, level, seed):
                old[r["frame"]] = r["dep2"]
        for fi, f in enumerate(frames):
            if fi not in old:
                continue
            pre = f["pre"]
            dl = DELTAS[di]
            inj = D.resample_sfo(f["seg"],
                                 f["lead"] - (pre + 4.25) * 4096,
                                 D.eps_of_delta(dl)) if dl else f["seg"]
            S, N0 = A.snr_parts(inj)
            rng = np.random.default_rng((SEED_CONST * 7919
                                         + (int(level) + 100) * 131
                                         + seed * 17 + (f["hs"] % 4099) * 7919
                                         + di * 104729) % (2 ** 31))
            p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
            seg = (inj + ((rng.standard_normal(len(inj))
                           + 1j * rng.standard_normal(len(inj)))
                          * np.sqrt(p_add / 2.0)))[None, :]
            sc, _, _, _ = M2.score_dep2(seg, f["lead"], pre)[:4]
            det.append(abs(float(sc[0]) / old[fi] - 1.0))
    res["determinism_max_rel_dev"] = float(max(det)) if det else None
    res["determinism_n"] = len(det)

    json.dump(res, open(os.path.join(HERE, "m2b_roc_results.json"), "w"),
              indent=1, default=str)
    print("== Pd09 战表（FAR=1e-3，主口径 per-(arm,δ,P) 门限）==")
    for k, v in pd_tab.items():
        print("%s | dera %.2f | dep2 %+.2f | dep3a %+.2f | dep3b %+.2f"
              " | cert %+.2f | thr: dep3a %.2f dep3b %.2f"
              % (k, v["dera"], v["dep2_minus_dera"],
                 v["dep3a_minus_dera"], v["dep3b_minus_dera"],
                 v["cert_minus_dera"], v["thr_dep3a_db"],
                 v["thr_dep3b_db"]))
    print("== 配对 w/l（vs dera，主口径）==")
    for k, v in wl.items():
        print("%s | dep3a %dW/%dL | dep3b %dW/%dL | dep2 %dW/%dL"
              % (k, v["dep3a"]["win"], v["dep3a"]["loss"],
                 v["dep3b"]["win"], v["dep3b"]["loss"],
                 v["dep2"]["win"], v["dep2"]["loss"]))
    print("determinism:", res["determinism_max_rel_dev"],
          "n=", res["determinism_n"])
    print("→ m2b_roc_results.json")


if __name__ == "__main__":
    main()

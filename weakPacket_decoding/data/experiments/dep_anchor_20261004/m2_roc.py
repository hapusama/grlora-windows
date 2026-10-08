# -*- coding: utf-8 -*-
"""M2 ROC：dep2（新 checkpoint）× cert/dep/dera（d2_battleB 老 checkpoint，
同噪配对 join）→ Pd09 战表、门限、配对 w/l、锚质量/δ̂ 分布、H0 不变性。
→ m2_roc_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_core as C
import m2_core as M

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "..", "keystone_battle_20261003",
                   "d2_battleB_checkpoint.jsonl")
NEW = os.path.join(HERE, "m2_battle_checkpoint.jsonl")
NEWC = os.path.join(HERE, "m2_battleC_checkpoint.jsonl")
FARS = (1e-3,)
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.082)
LEVELS = [-26, -28, -30, -32, -34, -36, -38]
H0_LEVELS = [-28, -34, -38]
ARMS = ("cert", "dep", "dera", "dep2", "dep2c")


def load():
    """返回 h1[(di,level,seed,frame)] = {cert,dep,dera,dep2,dhat2,nu_err,...}
    与 h0[(di,level,seed,frame,win)] = {...}。老/新 checkpoint 按键 join。"""
    h1, h0 = {}, {}
    for path, is_new in ((OLD, False), (NEW, "dep2"), (NEWC, "dep2c")):
        win_ct = {}          # 窗口计数器按文件独立（join 键对齐）
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            kind = r.get("kind")
            if kind == "h1":
                k = (r["di"], r["level"], r["seed"], r["frame"])
                d = h1.setdefault(k, {})
                if is_new == "dep2":
                    d.update(dep2=r["dep2"], dhat2=r["dhat"], nu_err=r["nu_err"],
                             acq=r["acq"], khat2=r["khat"], nu0=r["nu0"])
                elif is_new == "dep2c":
                    d.update(dep2c=r["dep2c"], dhatc=r["dhat"],
                             khatc=r["khat"], nu0c=r["nu0"])
                else:
                    d.update(cert=r["cert"], dep=r["dep"], dera=r["dera"],
                             dhat_old=r.get("dhat"), pre=r["pre"])
            elif kind == "h0":
                k0 = (r["di"], r["level"], r["seed"], r["frame"])
                i = win_ct.get(k0, 0)
                win_ct[k0] = i + 1
                k = k0 + (i,)
                d = h0.setdefault(k, {})
                if is_new == "dep2":
                    d.update(dep2=r["dep2"])
                elif is_new == "dep2c":
                    d.update(dep2c=r["dep2c"])
                else:
                    d.update(cert=r["cert"], dep=r["dep"], dera=r["dera"],
                             pre=r["pre"])
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
    n_h1 = sum(1 for v in h1.values() if "dep2" in v and "cert" in v)
    n_h0 = sum(1 for v in h0.values() if "dep2" in v and "cert" in v)
    print("join: H1 %d / H0 %d 单元（cert+dep+dera+dep2 四臂齐）" % (n_h1, n_h0),
          flush=True)
    res = dict(n_h1_join=n_h1, n_h0_join=n_h0)

    # ---- 确定性抽查（老 checkpoint 的 cert/dera 与本轮噪声一致性：靠 join
    # 噪声公式逐字一致 + 老 checkpoint 自身；另在 probe 里独立复核）----
    # ---- H0 不变性（CFAR 性）----
    inv = {}
    for a in ARMS:
        by = {}
        for k, v in h0.items():
            if a in v:
                by.setdefault((k[0], k[1]), []).append(v[a])
        meds = {}
        for (di, lv), vals in by.items():
            meds[(di, lv)] = float(np.median(vals))
        allv = list(meds.values())
        if allv:
            inv[a] = dict(di_drift=max(allv) - min(allv), n=len(allv) * 4)
        else:
            inv[a] = dict(di_drift=None, n=0)
    res["h0_invariance"] = inv

    # ---- 门限：逐 δ 池化（3 档）精确分位 ----
    thr = {}
    for f in FARS:
        for a in ARMS:
            for di in range(len(DELTAS)):
                pool = [v[a] for k, v in h0.items()
                        if k[0] == di and a in v]
                if len(pool) < 500:
                    pool = [v[a] for k, v in h0.items() if a in v]   # 跨 δ 池化
                if not pool:
                    thr[("t", a, di, f)] = float("nan")
                    continue
                thr[("t", a, di, f)] = float(np.quantile(pool, 1 - f))
                for P_sel in (8, 16, 32):
                    poolP = [v[a] for k, v in h0.items() if a in v
                             and v.get("pre") == P_sel]
                    thr[("tP", a, di, P_sel, f)] = float(np.quantile(
                        poolP, 1 - f)) if len(poolP) > 500 else float("nan")
    # dep2 解析（composition）与合成域参照
    ana = {}
    for pre in (8, 16, 32):
        for f in FARS:
            g, fac = M.thr_analytic(pre, f)
            ana["P%d@%g" % (pre, f)] = dict(db=10 * np.log10(g), fac=fac)
    res["dep2_analytic"] = ana

    # ---- Pd 曲线 + Pd09 + 配对 w/l ----
    pd_tab = {}
    wl = {}
    for f in FARS:
        for di in range(len(DELTAS)):
            row = {}
            for a in ARMS:
                curve = []
                miss = True
                for lv in LEVELS:
                    sc = [(v[a], v.get("pre")) for k, v in h1.items()
                          if k[0] == di and k[1] == lv and a in v]
                    if sc:
                        miss = False
                        fires = [val > thr[("tP", a, di, pre_, f)]
                                 if not np.isnan(thr[("tP", a, di, pre_, f)])
                                 else False for val, pre_ in sc]
                        curve.append(float(np.mean(fires)))
                row[a] = snr_at_pd09(curve) if not miss else float("nan")
            row["dep2_minus_dera"] = row["dep2"] - row["dera"]
            row["cert_minus_dera"] = row["cert"] - row["dera"]
            row["dep_minus_dera"] = row["dep"] - row["dera"]
            row["dep2_minus_cert"] = row["dep2"] - row["cert"]
            row["dep2c_minus_dera"] = row["dep2c"] - row["dera"]
            row["dep2c_minus_cert"] = row["dep2c"] - row["cert"]
            for a in ARMS:
                row["thr_%s_db" % a] = 10 * np.log10(thr[("t", a, di, f)])
            pd_tab["FAR=%g δ=%g" % (f, DELTAS[di])] = row
            # 配对 w/l（dep2 vs dera；同噪同窗）
            w = l_ = b_ = 0
            for k, v in h1.items():
                if k[0] != di or "dep2" not in v:
                    continue
                t2 = thr[("tP", "dep2", di, v.get("pre"), f)]
                td = thr[("tP", "dera", di, v.get("pre"), f)]
                if np.isnan(t2) or np.isnan(td):
                    continue
                if v["dep2"] > t2 and v["dera"] <= td:
                    w += 1
                elif v["dep2"] <= t2 and v["dera"] > td:
                    l_ += 1
                elif v["dep2"] > t2 and v["dera"] > td:
                    b_ += 1
            wc = lc = bc = 0
            for k, v in h1.items():
                if k[0] != di or "dep2c" not in v:
                    continue
                t2 = thr[("tP", "dep2c", di, v.get("pre"), f)]
                td = thr[("tP", "dera", di, v.get("pre"), f)]
                if np.isnan(t2) or np.isnan(td):
                    continue
                if v["dep2c"] > t2 and v["dera"] <= td:
                    wc += 1
                elif v["dep2c"] <= t2 and v["dera"] > td:
                    lc += 1
                elif v["dep2c"] > t2 and v["dera"] > td:
                    bc += 1
            wl["FAR=%g δ=%g" % (f, DELTAS[di])] = dict(
                win=w, loss=l_, both=b_, dep2c_win=wc, dep2c_loss=lc,
                dep2c_both=bc)
    res["thr_perP"] = {"%s P%d" % (a, P_sel):
                       10 * np.log10(thr[("tP", a, 0, P_sel, FARS[0])])
                       for a in ARMS for P_sel in (8, 16, 32)}
    res["pd09_table"] = pd_tab
    res["paired_wl"] = wl

    # ---- per-P 分解（P=8/16/32 各臂 Pd09）----
    perP = {}
    for f in FARS:
        for P_sel in (8, 16, 32):
            for a in ARMS:
                t_a = thr[("tP", a, 0, P_sel, f)]
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

    # ---- 锚质量 / δ̂ 分布（from battle H1 记录）----
    anch = {}
    for di in (0, 3, 5):                       # δ=0, 0.02, 0.082
        for lv in (-30, -34, -36):
            sc = [v["nu_err"] for k, v in h1.items()
                  if k[0] == di and k[1] == lv and "nu_err" in v]
            if not sc:
                continue
            sc = np.abs(np.array(sc))
            acq_ok = np.mean(sc < 0.5)
            anch["δ=%g lv=%d" % (DELTAS[di], lv)] = dict(
                n=len(sc), acq_ok=float(acq_ok),
                nu_p50=float(np.quantile(sc, 0.5)),
                nu_p90=float(np.quantile(sc, 0.9)))
    res["anchor"] = anch
    dhat = {}
    for di in range(len(DELTAS)):
        for lv in (LEVELS[0], -34):
            e = np.array([v["dhat2"] - DELTAS[di] for k, v in h1.items()
                          if k[0] == di and k[1] == lv and "dhat2" in v])
            if not len(e):
                continue
            dhat["δ=%g lv=%d" % (DELTAS[di], lv)] = dict(
                n=len(e), p50=float(np.quantile(e, 0.5)),
                p90=float(np.quantile(np.abs(e), 0.9)),
                frac_gt004=float(np.mean(np.abs(e) > 0.04)),
                frac_gt008=float(np.mean(np.abs(e) > 0.08)))
    res["dhat_dist"] = dhat

    json.dump(res, open(os.path.join(HERE, "m2_roc_results.json"), "w"),
              indent=1, default=str)
    print(json.dumps(pd_tab, indent=1, default=str))
    print("wl:", json.dumps(wl))
    print("→ m2_roc_results.json")


if __name__ == "__main__":
    main()

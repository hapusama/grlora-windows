# -*- coding: utf-8 -*-
"""D2 Battle C 汇总：PER/检出率/SER 战表 + 10% PER 门限 + 配对 + 预注册判定。
→ d2_sumC_results.json / stdout 战表
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "d2_battleC_checkpoint.jsonl")
DELTAS_C = (0.0, 0.02, 0.082)
LEVELS_C = [-20, -22, -24, -26, -28]
CHAIN_PER = ("dep_per", "cert_per", "dera_per")


def snr_at(per, lvs=LEVELS_C):
    """10% PER 门限（PER 随 SNR 下降曲线的线性内插）。"""
    lvs = np.array(lvs, dtype=float)
    p = np.asarray(per, dtype=float)
    for i in range(len(p) - 1):
        if p[i] <= 0.10 < p[i + 1]:
            return float(lvs[i] + (0.10 - p[i]) / (p[i + 1] - p[i] + 1e-12)
                         * (lvs[i + 1] - lvs[i]))
    if p[0] > 0.10:
        return float(lvs[0])
    return float(lvs[-1])


def main():
    rows = []
    for line in open(CKPT, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    res = dict(n_units=len(rows))
    native = [r for r in rows if r["level"] is None]
    nat_tab = {}
    for k in ("dep_det", "cert_det", "dera_det") + CHAIN_PER:
        vals = [r.get(k) for r in native]
        vals = [v for v in vals if v is not None]
        nat_tab[k] = dict(n=len(vals), mean=float(np.mean(vals)))
    res["native"] = nat_tab
    print("native (n=%d)：" % len(native),
          {k: v["mean"] for k, v in nat_tab.items()})

    tab = {}
    curves = {}
    for dl in DELTAS_C:
        for lv in LEVELS_C:
            sel = [r for r in rows if r["delta"] == dl and r["level"] == lv]
            if not sel:
                continue
            ent = dict(n=len(sel))
            for k in ("dep_det", "cert_det", "dera_det"):
                v = [r.get(k) for r in sel if r.get(k) is not None]
                ent[k] = float(np.mean(v)) if v else None
            for k in CHAIN_PER:
                v = [r.get(k) for r in sel if r.get(k) is not None]
                ent[k] = float(np.mean(v)) if v else None
            for k in ("dep_ser", "dera_ser"):
                v = [r.get(k) for r in sel
                     if r.get(k) is not None and r.get(k) >= 0]
                ent[k] = float(np.mean(v)) if v else None
            tab["δ=%g@%d" % (dl, lv)] = ent
        curves[dl] = {}
        for k in CHAIN_PER:
            curves[dl][k] = [tab["δ=%g@%d" % (dl, lv)][k]
                             for lv in LEVELS_C
                             if "δ=%g@%d" % (dl, lv) in tab]
    res["table"] = tab
    res["curves"] = {str(k): v for k, v in curves.items()}

    # 10% PER 门限 + 判定
    thr10 = {}
    for dl in DELTAS_C:
        thr10[dl] = {k: snr_at(curves[dl][k]) for k in curves[dl]}
    res["thr10"] = {str(k): v for k, v in thr10.items()}
    verdict = {}
    for dl in (0.02, 0.082):
        ours = thr10[dl].get("dep_per")
        dr = thr10[dl].get("dera_per")
        margin = (dr - ours) if (ours is not None and dr is not None) else None
        verdict["δ=%g" % dl] = dict(
            ours_dep=ours, dera=dr, margin_db=margin,
            ours_cert=thr10[dl].get("cert_per"),
            ours_trel=thr10[dl].get("dep_trel_per"),
            win=bool(margin is not None and margin >= 1.5))
    res["verdict"] = verdict

    # 配对（同单元同噪）：dep 链 vs dera 链 PER 胜负
    pair = {}
    for dl in DELTAS_C:
        w = l_ = t_ = 0
        for r in rows:
            if r["delta"] != dl or r["level"] is None:
                continue
            dp = r.get("dep_per")
            dr = r.get("dera_per")
            if dp is None or dr is None:
                continue
            w += int(dp < dr)
            l_ += int(dp > dr)
            t_ += int(dp == dr)
        pair["δ=%g" % dl] = dict(win=w, loss=l_, tie=t_)
    res["paired_per"] = pair

    # γ-fit 漂移 vs 注入 δ（检测→解码链的 δ̂ 健康）
    fitq = {}
    for dl in DELTAS_C:
        v = [r.get("gfit_d") for r in rows
             if r["delta"] == dl and r.get("gfit_d") is not None
             and r["level"] is not None]
        fitq["δ=%g" % dl] = dict(mean=float(np.mean(v)),
                                 median=float(np.median(v)),
                                 p10=float(np.quantile(v, 0.1)),
                                 p90=float(np.quantile(v, 0.9))) if v else None
    res["gfit_drift"] = fitq

    json.dump(res, open(os.path.join(HERE, "d2_sumC_results.json"), "w"),
              indent=1, default=str)
    for dl in DELTAS_C:
        print("δ=%g PER 曲线（dep / dep×TREL / cert / dera）：" % dl)
        for k in CHAIN_PER:
            print("  %-12s" % k, [None if c is None else round(c, 3)
                                  for c in curves[dl][k]])
    print("10%% PER 门限：", json.dumps(res["thr10"], default=str))
    print("判定：", json.dumps(verdict))
    print("配对：", json.dumps(pair))
    print("→ d2_sumC_results.json")


if __name__ == "__main__":
    main()

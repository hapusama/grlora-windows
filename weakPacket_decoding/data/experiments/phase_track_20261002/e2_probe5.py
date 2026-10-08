# -*- coding: utf-8 -*-
"""E2 探针5：宽域 τ 扫描定 STO 滑动项系数（d = 2πτ·Δv/N + walk）。

探针4 已示 |Δv| 分桶的卷绕线性结构（过渡带 |Δv|≈300 → α=π/300≈0.0105 rad/bin
⇒ τ=αN/2π≈1.7 N 样本）。此处逐帧 + 池化扫描 τ∈[−4,4]，看修正后增量 σ
是否塌缩到 Wiener 水平（~0.12 rad）。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
from e2_probe4 import collect, CACHE

OUT = os.path.join(C.ROOT, "e2_probe5_results.json")


def scan_tau(d, dv, n, taus):
    conc = np.array([abs(np.mean(np.exp(1j * (d - 2 * np.pi * t * dv / n))))
                     for t in taus])
    j = int(np.argmax(conc))
    # 抛物线细化
    if 0 < j < len(taus) - 1:
        y0, y1, y2 = conc[j - 1], conc[j], conc[j + 1]
        dmax = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
        t = taus[j] + np.clip(dmax, -0.5, 0.5) * (taus[1] - taus[0])
    else:
        t = taus[j]
    return float(t), float(conc[j])


def main():
    data = collect()
    gids = sorted(data)
    n = 1024
    taus = np.linspace(-4, 4, 1601)
    per_frame = []
    Dp, DVp = [], []
    for g in gids:
        th = data[g]["th"]
        d = C.wrap(np.diff(th))
        dv = np.diff(data[g]["v"])
        Dp.append(d)
        DVp.append(dv)
        t, cc = scan_tau(d, dv, n, taus)
        dcor = C.wrap(d - 2 * np.pi * t * dv / n)
        per_frame.append(dict(gid=g, tau=t, conc=cc,
                              inc_sigma_cor=float(np.std(dcor)),
                              inc_sigma_raw=float(np.std(d))))
    D = np.concatenate(Dp); DV = np.concatenate(DVp)
    t_pool, cc_pool = scan_tau(D, DV, n, taus)
    # 池化 τ 逐帧修正
    for r, g in zip(per_frame, gids):
        d = C.wrap(np.diff(data[g]["th"]))
        dv = np.diff(data[g]["v"])
        r["inc_sigma_pooltau"] = float(np.std(C.wrap(d - 2 * np.pi * t_pool * dv / n)))
    for r in per_frame:
        r["gid"] = int(r["gid"])
        print("gid%3d τ̂=%+.3f conc=%.3f inc: raw=%.3f → cor=%.3f (poolτ=%.3f)" % (
            r["gid"], r["tau"], r["conc"], r["inc_sigma_raw"], r["inc_sigma_cor"],
            r["inc_sigma_pooltau"]), flush=True)

    def stat(k):
        v = np.array([r[k] for r in per_frame])
        return dict(med=float(np.median(v)), p25=float(np.percentile(v, 25)),
                    p75=float(np.percentile(v, 75)))

    taus_hat = np.array([r["tau"] for r in per_frame])
    summary = dict(tau_pool=t_pool, conc_pool=cc_pool,
                   tau_hat=stat("tau"), inc_raw=stat("inc_sigma_raw"),
                   inc_cor=stat("inc_sigma_cor"), inc_pooltau=stat("inc_sigma_pooltau"),
                   frac_below_0p2=float(np.mean([r["inc_sigma_cor"] < 0.2
                                                 for r in per_frame])))
    print(json.dumps(summary, indent=1))
    json.dump(dict(summary=summary, per_frame=per_frame),
              open(OUT, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

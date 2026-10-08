# -*- coding: utf-8 -*-
"""E1 探针7：φ(c) 可校准性的 5 折交叉验证（判定性）。

每帧 payload 唯一 → 按 c 池化无位置混叠。SF11 全池 120 帧（≈13 样本/bin）、
SF10 全池 28 帧，5 折 CV：4 折学 φ̂(c)，1 折评测。同时报组数/组样本数。
若 σcal 显著低于 σcold → φ(c) capture-级平稳可校准（机制立项的校准路径）。
"""
import sys
import os
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_common import ROOT, wrap

IN = os.path.join(ROOT, "e1_native.jsonl")


def gn_fit(th, B, iters=10):
    x = np.zeros(B.shape[1])
    for _ in range(iters):
        r = wrap(th - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x += dx
    return x, wrap(th - B @ x)


def frame_resid(fr):
    nb = 1 << fr["sf"]
    syms = fr["syms"]
    th = np.array([s["theta"] for s in syms])
    c = np.array([s["c"] for s in syms], dtype=float)
    kinds = np.array([s["kind"] for s in syms])
    i = np.arange(len(syms), dtype=float)
    B = np.stack([np.ones_like(i), i, i ** 2, c / nb,
                  (kinds == "sync").astype(float),
                  (kinds == "hdr").astype(float),
                  (kinds == "pay").astype(float)], 1)
    x, r = gn_fit(th, B)
    r = wrap(r - np.angle(np.mean(np.exp(1j * r))))
    m = np.isin(kinds, ("pay", "hdr"))
    return r[m], (c[m] % nb).astype(int)


def main():
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    rng = np.random.default_rng(20261002)
    for sf in (10, 11):
        fl = [fr for fr in frames if fr["sf"] == sf]
        idx = rng.permutation(len(fl))
        folds = np.array_split(idx, 5)
        cal_all, cold_all = [], []
        for f in range(5):
            tr = np.concatenate([folds[g] for g in range(5) if g != f])
            te = folds[f]
            ph = defaultdict(list)
            for j in tr:
                r, c = frame_resid(fl[j])
                for rr, cc in zip(r, c):
                    ph[cc].append(rr)
            phv = {cc: (len(v), np.angle(np.mean(np.exp(1j * np.array(v)))))
                   for cc, v in ph.items()}
            for j in te:
                r, c = frame_resid(fl[j])
                hit = np.array([cc in phv for cc in c])
                if hit.sum() < 8:
                    continue
                cal = wrap(r[hit] - np.array([phv[int(cc)][1] for cc in c[hit]]))
                cal_all.extend(cal)
                cold_all.extend(r[hit])
        cnts = np.array([v[0] for v in phv.values()])
        print("SF%d: n_eval=%d  σcold=%.3f → σcal=%.3f   φ̂ 组数=%d, 组样本 med=%.1f p10=%.1f"
              % (sf, len(cold_all), np.std(cold_all), np.std(cal_all),
                 len(phv), np.median(cnts), np.percentile(cnts, 10)))
        # 置换对照：φ̂ 随机重排 bin
        keys = np.array(list(phv.keys()))
        vals = np.array([phv[k][1] for k in keys])
        perm = rng.permutation(len(vals))
        phr = dict(zip(keys, vals[perm]))
        calp = []
        for j in np.concatenate(folds):
            r, c = frame_resid(fl[j])
            hit = np.array([cc in phr for cc in c])
            if hit.sum() >= 8:
                calp.extend(wrap(r[hit] - np.array([phr[int(cc)] for cc in c[hit]])))
        print("      置换对照 σcal_perm=%.3f（应≈σcold）" % np.std(calp))


if __name__ == "__main__":
    main()

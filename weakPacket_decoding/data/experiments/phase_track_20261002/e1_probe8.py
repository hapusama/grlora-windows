# -*- coding: utf-8 -*-
"""E1 探针8：payload 残差是否被低 SNR 帧驱动。"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_common import ROOT, wrap

frames = [json.loads(l) for l in open(os.path.join(ROOT, "e1_native.jsonl"), encoding="utf-8")]


def gn_fit(th, B, iters=10):
    x = np.zeros(B.shape[1])
    for _ in range(iters):
        r = wrap(th - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x += dx
    return x, wrap(th - B @ x)


rows = []
for fr in frames:
    nb = 1 << fr["sf"]
    syms = fr["syms"]
    th = np.array([s["theta"] for s in syms])
    c = np.array([s["c"] for s in syms], float)
    kinds = np.array([s["kind"] for s in syms])
    i = np.arange(len(syms), dtype=float)
    B = np.stack([np.ones_like(i), i, i ** 2, c / nb,
                  (kinds == "sync").astype(float),
                  (kinds == "hdr").astype(float),
                  (kinds == "pay").astype(float)], 1)
    x, r = gn_fit(th, B)
    mp = kinds == "pay"
    amp = np.array([s["amp"] for s in syms])
    rows.append((fr["snr_native"], r[mp].std(), float(np.mean(amp[mp])), fr["sf"]))
a = np.array(rows)
for lo, hi in ((-10, 5), (5, 10), (10, 20), (20, 40)):
    m = (a[:, 0] >= lo) & (a[:, 0] < hi)
    if m.sum():
        print("nativeSNR [%2d,%2d): n=%3d  sigma_pay med=%.3f  amp med=%.2f"
              % (lo, hi, m.sum(), np.median(a[m, 1]), np.median(a[m, 2])))
print("corr(snr, sigma_pay) = %+.3f" % np.corrcoef(a[:, 0], a[:, 1])[0, 1])
print("corr(amp, sigma_pay) = %+.3f" % np.corrcoef(a[:, 2], a[:, 1])[0, 1])

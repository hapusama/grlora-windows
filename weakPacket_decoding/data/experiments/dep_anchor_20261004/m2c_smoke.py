# -*- coding: utf-8 -*-
"""M2c native 冒烟（28/28）：dep4 两级链在无加噪 OTA 帧上的健全性。

检查：提名列 rank（GT=clean template 的 ν0+c_e·δ）、score_dB ≫ 门限、
K8/K16 一致性（嵌套）。→ m2c_smoke_results.json
"""
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_battle as A
import m2c_core as W

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    frames = A.build_frames()
    n_ok = 0
    recs = []
    t0 = time.time()
    for fi, f in enumerate(frames):
        pre = f["pre"]
        seg = f["seg"][None, :]
        nu_gt = f["tm"]["nu0"] + (pre - 1) / 2.0 * f["tm"]["delta"]
        out = W.score_dep4(seg, f["lead"], pre, ret_diag=True)
        nu_c = out["nu_c"][0]
        err = nu_c - nu_gt
        err = (err + 512) % 1024 - 512          # wrap 到 ±512 bin
        rank = int(np.argmin(np.abs(err)))
        sc8 = float(out["score_k8"][0])
        sc16 = float(out["score_k16"][0])
        ok = rank == 0 and sc8 > 1e3
        n_ok += ok
        recs.append(dict(frame=fi, pre=pre, rank=rank,
                         nu_err=float(err[0]),
                         score_k8_db=10 * np.log10(sc8),
                         score_k16_db=10 * np.log10(sc16),
                         dhat=float(out["dhat_k16"][0]), ok=bool(ok)))
        print("f%02d P%2d rank %2d nu_err %+0.3f | k8 %6.1f dB k16 %6.1f dB"
              " dhat %+.3f" % (fi, pre, rank, err[0],
                               recs[-1]["score_k8_db"],
                               recs[-1]["score_k16_db"], recs[-1]["dhat"]),
              flush=True)
    res = dict(n_ok=n_ok, n=len(frames), elapsed=round(time.time() - t0, 1),
               frames=recs)
    json.dump(res, open(os.path.join(HERE, "m2c_smoke_results.json"), "w"),
              indent=1)
    print("native %d/%d（%.0fs）" % (n_ok, len(frames), time.time() - t0))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""M2c probe：① 合成 H0 小 MC（纯 CN 噪声全链，门限 sanity vs 解析 union）
② 提名召回（真列 rank vs level/δ，注入 GT）③ 运行时/单窗成本。
→ m2c_probe.json（分片可续：--part h0/recall 各自独立跑）
"""
import argparse
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
import d2_core as D
import m2c_core as W

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_CONST = 20261003


def synth_h0(pre, n_units, seed, chunk=100):
    """纯合成 AWGN 单元（协议 H0；分布同 battle rng0 路径，独立种子域）。"""
    rng = np.random.default_rng(seed)
    L = int((pre + 8) * W.NF + 8 * W.NF)
    out = []
    for done in range(0, n_units, chunk):
        m = min(chunk, n_units - done)
        segs = (rng.standard_normal((m, L)) + 1j * rng.standard_normal((m, L))) \
            * np.sqrt(0.5)
        hs = (pre + 6) * W.NF
        o = W.score_dep4(segs, hs, pre)
        out.append(np.stack([o["score_k8"], o["score_k16"]], axis=1))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="h0")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    res_path = os.path.join(HERE, "m2c_probe.json")
    res = json.load(open(res_path, encoding="utf-8")) \
        if os.path.exists(res_path) else {}

    if args.part == "h0":
        h0 = {}
        for pre in (8, 16, 32):
            t0 = time.time()
            sc = synth_h0(pre, args.n, 20261004 + pre)
            el = time.time() - t0
            q = {"%g_%s" % (f, ("k8", "k16")[i]):
                 float(np.quantile(sc[:, i], 1 - f))
                 for f in (1e-2, 1e-3) for i in (0, 1)}
            med = [float(np.median(sc[:, 0])), float(np.median(sc[:, 1]))]
            ana = {}
            for k in (8, 16):
                g, diag = W.thr_analytic_dep4(pre, 1e-3, k_nom=k)
                ana["k%d" % k] = dict(union_db=diag["thr_union_db"],
                                      single_db=diag["thr_single_db"],
                                      shift=diag["union_shift_db"],
                                      c_total=diag["c_total"])
            h0["P%d" % pre] = dict(n=args.n, elapsed=round(el, 1),
                                   sec_per_unit=round(el / args.n, 3),
                                   quantiles=q, median=med, analytic=ana)
            res["h0_mc"] = h0
            json.dump(res, open(res_path, "w"), indent=1)
            print("P%d n=%d %.0fs (%.3fs/u) | MC 1e-3: k8 %.2f k16 %.2f dB"
                  " | 1e-2: k8 %.2f k16 %.2f | ana union k8 %.2f k16 %.2f"
                  % (pre, args.n, el, el / args.n,
                     10 * np.log10(q["0.001_k8"]),
                     10 * np.log10(q["0.001_k16"]),
                     10 * np.log10(q["0.01_k8"]),
                     10 * np.log10(q["0.01_k16"]),
                     ana["k8"]["union_db"], ana["k16"]["union_db"]),
                  flush=True)
        res["h0_mc"] = h0

    elif args.part == "recall":
        frames = A.build_frames()
        by_pre = {}
        for i, f in enumerate(frames):
            by_pre.setdefault(f["pre"], []).append(i)
        rec = {}
        DELTAS = (0.0, 0.02, 0.082)
        LEVELS = (-30, -32, -34, -36)
        for pre, lst in by_pre.items():
            t0 = time.time()
            for di, dl in enumerate(DELTAS):
                for lv in LEVELS:
                    segs = np.empty((len(lst), len(frames[lst[0]]["seg"])),
                                    dtype=np.complex128)
                    for i, fi in enumerate(lst):
                        f = frames[fi]
                        inj = D.resample_sfo(
                            f["seg"], f["lead"] - (pre + 4.25) * W.NF,
                            D.eps_of_delta(dl)) if dl else f["seg"]
                        S, N0 = A.snr_parts(inj)
                        rng = np.random.default_rng(
                            (SEED_CONST * 7919 + (int(lv) + 100) * 131
                             + 17 + (f["hs"] % 4099) * 7919
                             + di * 104729) % (2 ** 31))
                        p_add = max(S / 10 ** (lv / 10.0) - N0, 1e-30)
                        segs[i] = inj + (rng.standard_normal(len(inj))
                                         + 1j * rng.standard_normal(len(inj))
                                         ) * np.sqrt(p_add / 2.0)
                    lead = frames[lst[0]]["lead"]
                    nu_c, cols = W.nominate(segs, lead, pre)
                    o = W.verify(segs, lead, pre, nu_c, ret_diag=True)
                    ranks = []
                    for i, fi in enumerate(lst):
                        f = frames[fi]
                        inj = D.resample_sfo(
                            f["seg"], f["lead"] - (pre + 4.25) * W.NF,
                            D.eps_of_delta(dl)) if dl else f["seg"]
                        tmq = D.clean_template_q(inj, f["lead"], pre)
                        nu_gt = tmq["nu0"] + (pre - 1) / 2.0 * tmq["delta"]
                        e = (nu_c[i] - nu_gt + 512) % 1024 - 512
                        ranks.append(int(np.argmin(np.abs(e))))
                    ranks = np.array(ranks)
                    key = "P%d δ%g lv%d" % (pre, dl, lv)
                    rec[key] = dict(n=len(ranks),
                                    r0=float(np.mean(ranks == 0)),
                                    r_lt8=float(np.mean(ranks < 8)),
                                    r_lt16=float(np.mean(ranks < 16)),
                                    med_rank=float(np.median(ranks)))
                    print("%s rank0 %.2f <8 %.2f <16 %.2f"
                          % (key, rec[key]["r0"], rec[key]["r_lt8"],
                             rec[key]["r_lt16"]), flush=True)
            rec["P%d_elapsed" % pre] = round(time.time() - t0, 1)
        res["recall"] = rec
    json.dump(res, open(res_path, "w"), indent=1)
    print("→ m2c_probe.json")


def inj_cache(f, dl):
    return D.resample_sfo(f["seg"], f["lead"] - (f["pre"] + 4.25) * W.NF,
                          D.eps_of_delta(dl))


if __name__ == "__main__":
    main()

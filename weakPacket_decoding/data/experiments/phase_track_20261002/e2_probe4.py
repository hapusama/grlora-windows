# -*- coding: utf-8 -*-
"""E2 探针4：增量结构定位——payload 相位增量 d_i 与 (Δv, v, v mod m) 的关系。

内容随机（28 帧仅 1 帧重复）→ 若 d=g(Δv) 确定性数据依赖，跨帧按 Δv 分桶
圆均值幅度应显著；若 d 与内容无关（纯共享过程），各桶幅度≈0。
另做同 capture 帧对逐位比较（内容不同，共享过程假设下的差分 σ）。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
from e2_probe import zdtft, kappa_from_pre, extract_dr

OUT = os.path.join(C.ROOT, "e2_probe4_results.json")
CACHE = os.path.join(C.ROOT, "e2_theta_cache.npz")


def collect():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {g: dict(th=z["th"][i], nu=z["nu"][i], kap=float(z["kap"][i]),
                        v=z["v"][i], cap=str(z["cap"][i]))
                for i, g in enumerate(z["gid"])}
    frames = [json.loads(l) for l in open(os.path.join(C.ROOT, "e1_native.jsonl"),
                                          encoding="utf-8")]
    frames = [f for f in frames if f["sf"] == 10]
    out = {}
    for f in frames:
        src = next(s for s in C.SF10_SOURCES if s[0] == f["cap"])
        ds = C.DS(f["sf"])
        iq = np.memmap(src[1], dtype=np.complex64, mode="r")
        r = dict(header_start_sample=f["hs"], source_grlora_cfo_int=f["cfo_int"],
                 source_grlora_cfo_frac=str(f["cfo_frac"]),
                 source_grlora_payload_sto_frac=str(f["sto_frac"]))
        seg, _i0, _bo = C.align_seg(iq, r, ds, f["P"], f["psym"])
        del iq
        n = ds.n
        pre_drs = [extract_dr(seg, 1.0 + j - 0.25, ds, "pre") for j in range(f["P"])]
        pay_drs = [extract_dr(seg, f["P"] + 13.0 + k, ds, "pay")
                   for k in range(f["psym"])]
        kap = kappa_from_pre(pre_drs, n)
        th = np.array([np.angle(zdtft(dr, v + kap, n)) for dr, v in zip(pay_drs, f["gt"])])
        out[f["gid"]] = dict(th=th, nu=(np.array(f["gt"], float) + kap) % n,
                             kap=kap, v=np.array(f["gt"], float), cap=f["cap"])
    np.savez(CACHE, gid=np.array(sorted(out)),
             th=np.array([out[g]["th"] for g in sorted(out)]),
             nu=np.array([out[g]["nu"] for g in sorted(out)]),
             kap=np.array([out[g]["kap"] for g in sorted(out)]),
             v=np.array([out[g]["v"] for g in sorted(out)]),
             cap=np.array([out[g]["cap"] for g in sorted(out)]))
    return out


def main():
    data = collect()
    gids = sorted(data)
    # ---- 1. d vs Δv / v mod m 分桶 ----
    D_all, DV_all, V_all = [], [], []
    for g in gids:
        th = data[g]["th"]
        v = data[g]["v"]
        D_all.append(C.wrap(np.diff(th)))
        DV_all.append(np.diff(v))
        V_all.append(v[:-1])
    D = np.concatenate(D_all); DV = np.concatenate(DV_all); V = np.concatenate(V_all)
    buckets = {}
    for m, key in ((None, "dv_mod_none"),):
        pass
    def bucket_report(x, name, edges):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            msk = (x >= lo) & (x < hi)
            if msk.sum() >= 8:
                z = np.mean(np.exp(1j * D[msk]))
                rows.append(dict(lo=float(lo), hi=float(hi), n=int(msk.sum()),
                                 mag=float(abs(z)), ang=float(np.angle(z))))
        print("== %s 分桶（n, |mean e^{jd}|, angle):" % name)
        for r in rows:
            print("   [%7.1f,%7.1f) n=%4d |z|=%.3f ang=%+.3f" %
                  (r["lo"], r["hi"], r["n"], r["mag"], r["ang"]))
        return rows
    dvb = bucket_report(np.abs(DV), "|Δv|", np.arange(0, 545, 64))
    vmb = bucket_report(V % 8, "v mod 8", np.arange(-0.5, 8.5, 1))
    vb = bucket_report(V, "v", np.arange(0, 1088, 128))
    # Δv 奇偶
    par = {}
    for p in (0, 1):
        msk = (np.mod(DV, 2) == p)
        z = np.mean(np.exp(1j * D[msk]))
        par[p] = dict(n=int(msk.sum()), mag=float(abs(z)), ang=float(np.angle(z)))
        print("Δv parity %d: n=%d |z|=%.3f ang=%+.3f" % (p, *par[p].values()))
    # ---- 2. 同 capture 帧对逐位差 ----
    pair_sig = []
    for ga in gids:
        for gb in gids:
            if gb <= ga or data[ga]["cap"] != data[gb]["cap"]:
                continue
            d = C.wrap(data[ga]["th"] - data[gb]["th"])
            d = d - np.angle(np.mean(np.exp(1j * d)))
            pair_sig.append(float(np.std(d)))
    print("\n== 同 capture 帧对逐位差 σ: med=%.3f p25=%.3f p75=%.3f (n=%d)" % (
        np.median(pair_sig), np.percentile(pair_sig, 25),
        np.percentile(pair_sig, 75), len(pair_sig)))
    # 每帧 inc 与 κ 的关系
    inc = {g: float(np.std(C.wrap(np.diff(data[g]["th"])))) for g in gids}
    kaps = np.array([data[g]["kap"] for g in gids])
    incs = np.array([inc[g] for g in gids])
    print("== inc vs κ0 相关: %.3f；κ0 分布:" % np.corrcoef(kaps, incs)[0, 1],
          np.round(np.percentile(kaps, [10, 25, 50, 75, 90]), 3))
    # 干净帧（inc<0.4）列表
    clean = [g for g in gids if inc[g] < 0.4]
    print("== 干净帧 (inc<0.4):", clean, "κ:", [round(data[g]['kap'], 3) for g in clean])
    json.dump(dict(dv_buckets=dvb, vmod8_buckets=vmb, v_buckets=vb, parity=par,
                   pair_sigma=dict(med=float(np.median(pair_sig)),
                                   p25=float(np.percentile(pair_sig, 25)),
                                   p75=float(np.percentile(pair_sig, 75)),
                                   n=len(pair_sig)),
                   clean_frames={str(g): inc[g] for g in clean},
                   corr_inc_kappa=float(np.corrcoef(kaps, incs)[0, 1])),
              open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

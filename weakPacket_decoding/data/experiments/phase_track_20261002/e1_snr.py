# -*- coding: utf-8 -*-
"""E1 SNR 可行性扫描 + γ 噪声地板（2026-10-02 资金数字）。

协议遵守：整包 AWGN（§2 口径，S/N0 取 native 段标定），种子 (实验日,档位,种子,帧) 派生；
native 档必含。GT = native 干净测量 θ_true（协议 §1：GT 只来自干净信号）。

单位 = (gid, snr, seed)。每单位测：
  L0 测量地板   σ(wrap(θ̂ − θ_true))，分 pre / pay 段
  GENIE 后验    全帧 M2c+kind 圆域拟合 → 对 θ_true 的 RMSE（分段）
  GENIE pre     前导段 M1 拟合 RMSE
  CAUSAL        滚动一步预测（前缀 M1 拟合，warm-up k∈{4,8,16}）对 θ_true 的 RMSE（分段）
  γ 地板        逐符号 Δγ（PCM）与 Δy（V1V2, mod-2 卷绕）σ + 帧池化 WLS κ̂ 误差
输出：e1_snr_units.jsonl（断点续跑）→ e1_snr_results.json（汇总）。
"""
import sys
import os
import json
import time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C

ROOT = C.ROOT
IN = os.path.join(ROOT, "e1_native.jsonl")
OUTU = os.path.join(ROOT, "e1_snr_units.jsonl")
OUTR = os.path.join(ROOT, "e1_snr_results.json")
KS = (4, 8, 16)
SEGS = ("pre", "pay")            # 汇总分段（sync/hdr 计入 pay 侧结构同类）
SEEDS = list(range(C.N_SEEDS))

_W = {}


def gn_fit(th, B, iters=10):
    x = np.zeros(B.shape[1])
    for _ in range(iters):
        r = C.wrap(th - B @ x)
        dx, *_ = np.linalg.lstsq(B, r, rcond=None)
        x += dx
    return x


def seg_mask(kinds, seg):
    if seg == "pre":
        return kinds == "pre"
    return (kinds == "pay") | (kinds == "hdr") | (kinds == "sync")


def unit_metrics(syms_n, syms_t, sf):
    """syms_n: 带噪测量; syms_t: native 测量（对齐顺序）。"""
    nb = 1 << sf
    th_n = np.array([s["theta"] for s in syms_n])
    th_t = np.array([s["theta"] for s in syms_t])
    kinds = np.array([s["kind"] for s in syms_n])
    c = np.array([s["c"] for s in syms_n], dtype=float)
    i = np.arange(len(th_n), dtype=float)
    out = {}
    # L0 测量地板
    e0 = C.wrap(th_n - th_t)
    for sg in SEGS:
        m = seg_mask(kinds, sg)
        out["l0_%s" % sg] = float(np.sqrt(np.mean(e0[m] ** 2)))
    # GENIE 全帧 M2c+kind
    B = np.stack([np.ones_like(i), i, i ** 2, c / nb,
                  (kinds == "sync").astype(float),
                  (kinds == "hdr").astype(float),
                  (kinds == "pay").astype(float)], 1)
    x = gn_fit(th_n, B)
    eg = C.wrap(th_t - B @ x)
    for sg in SEGS:
        m = seg_mask(kinds, sg)
        out["genie_%s" % sg] = float(np.sqrt(np.mean(eg[m] ** 2)))
    # GENIE pre-only M1
    mp = kinds == "pre"
    if mp.sum() >= 4:
        Bp = np.stack([np.ones(mp.sum()), i[mp]], 1)
        xp = gn_fit(th_n[mp], Bp)
        out["genie_pre_m1"] = float(np.sqrt(np.mean(C.wrap(th_t[mp] - Bp @ xp) ** 2)))
    # CAUSAL 滚动 M1（前缀拟合预测下一符号）
    for k in KS:
        errs = {sg: [] for sg in SEGS}
        for j in range(k, len(th_n)):
            Bp = np.stack([np.ones(j), i[:j]], 1)
            xp = gn_fit(th_n[:j], Bp)
            pred = xp[0] + xp[1] * i[j]
            for sg in SEGS:
                if seg_mask(kinds, sg)[j]:
                    errs[sg].append(C.wrap(pred - th_t[j]))
        for sg in SEGS:
            out["causal%d_%s" % (k, sg)] = (
                float(np.sqrt(np.mean(np.array(errs[sg]) ** 2))) if errs[sg] else None)
    # γ 地板
    g_n = np.array([s["gamma_pcm"] for s in syms_n])
    g_t = np.array([s["gamma_pcm"] for s in syms_t])
    zg_n = np.array([s["z_pcm"] for s in syms_n])
    zg_t = np.array([s["z_pcm"] for s in syms_t])
    dg = C.wrap(g_n - g_t)
    out["gamma_pcm_sigma"] = float(np.sqrt(np.mean(dg ** 2)))
    # PCM γ 圆域池化 κ̂（γ=πκ mod π → e^{2iγ} 不变量；权重 z²）
    if len(g_n) >= 8:
        w = zg_n ** 2
        kap = float(np.angle(np.sum(w * np.exp(2j * g_n))) / (2 * np.pi))
        w_t = zg_t ** 2
        kap_t = float(np.angle(np.sum(w_t * np.exp(2j * g_t))) / (2 * np.pi))
        out["kap_pcm_err"] = float((kap - kap_t + 0.5) % 1.0 - 0.5)
        out["kap_pcm_gap"] = float(abs(np.angle(np.sum(w * np.exp(1j * (g_n - g_t))))))
    y_n = np.array([s["y_v1v2"] for s in syms_n if s.get("y_v1v2") is not None])
    y_t = np.array([s["y_v1v2"] for s in syms_t if s.get("y_v1v2") is not None])
    z_n = np.array([s["z_v1v2"] for s in syms_n if s.get("z_v1v2") is not None])
    if len(y_n) >= 8:
        dy = (y_n - y_t + 1.0) % 2.0 - 1.0          # y 周期 2（angle/π）
        out["y_sigma"] = float(np.sqrt(np.mean(dy ** 2)))
        w = z_n ** 2
        kap = float(np.angle(np.sum(w * np.exp(2j * np.pi * y_n))) / (2 * np.pi))
        w_t = np.array([s["z_v1v2"] for s in syms_t if s.get("z_v1v2") is not None]) ** 2
        kap_t = float(np.angle(np.sum(w_t * np.exp(2j * np.pi * y_t))) / (2 * np.pi))
        out["kap_wls_err"] = float((kap - kap_t + 0.5) % 1.0 - 0.5)
    return out


def run_unit(u):
    gid, snr, seed, cap = u["gid"], u["snr"], u["seed"], u["cap"]
    fr, seg, ds = _W["frames"][gid], _W["segs"][gid], _W["ds"][gid]
    noisy = C.inject_noise(seg, snr, seed, gid, fr["S"], fr["N0"]) if snr is not None else seg
    syms_n, _tr, _cl = C.measure_frame(
        noisy, fr["P"], fr["psym"], fr["ldro"], fr["gt_hdr"], fr["gt"], ds, fine=False)
    if len(syms_n) != len(fr["syms"]):
        return dict(u, skip="len_mismatch")
    m = unit_metrics(syms_n, fr["syms"], fr["sf"])
    return dict(u, **m)


def init_worker(gids):
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    keep = [fr for fr in frames if fr["gid"] in gids]
    caps = sorted(set(fr["cap"] for fr in keep))
    _W["frames"] = {fr["gid"]: fr for fr in keep}
    _W["segs"] = {}
    _W["ds"] = {}
    for cap in caps:
        src = next(s for s in C.SF10_SOURCES + C.SF11_SOURCES if s[0] == cap)
        ds = C.DS(src[3])
        iq = np.memmap(src[1], dtype=np.complex64, mode="r")
        for fr in (f for f in keep if f["cap"] == cap):
            r = dict(header_start_sample=fr["hs"],
                     source_grlora_cfo_int=fr["cfo_int"],
                     source_grlora_cfo_frac=str(fr["cfo_frac"]),
                     source_grlora_payload_sto_frac=str(fr["sto_frac"]))
            seg, _i0, _bo = C.align_seg(iq, r, ds, fr["P"], fr["psym"])
            _W["segs"][fr["gid"]] = seg
            _W["ds"][fr["gid"]] = ds
        del iq


def main():
    t0 = time.time()
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    _W["raw"] = frames
    by_cap = {}
    for fr in frames:
        by_cap.setdefault(fr["cap"], []).append(fr["gid"])
    units = []
    for fr in frames:
        units.append(dict(gid=fr["gid"], snr=None, seed=None, cap=fr["cap"]))
        for snr in C.SNR_LEVELS:
            for sd in SEEDS:
                units.append(dict(gid=fr["gid"], snr=snr, seed=sd, cap=fr["cap"]))
    done = set()
    if os.path.exists(OUTU):
        for l in open(OUTU, encoding="utf-8"):
            rec = json.loads(l)
            if "skip" not in rec:
                done.add((rec["gid"], rec["snr"], rec["seed"]))
        print("断点续跑: 已完成 %d 单元" % len(done), flush=True)
    todo = [u for u in units if (u["gid"], u["snr"], u["seed"]) not in done]
    print("总单元 %d, 待跑 %d" % (len(units), len(todo)), flush=True)
    n = 0
    fout = open(OUTU, "a", encoding="utf-8")
    # 按 capture 分块派发（池随 capture 重建，worker 内缓存 seg）
    for cap, gids in sorted(by_cap.items()):
        cu = [u for u in todo if u["cap"] == cap]
        if not cu:
            continue
        with Pool(processes=6, initializer=init_worker, initargs=(set(gids),)) as pool:
            for res in pool.imap_unordered(run_unit, cu, chunksize=8):
                fout.write(json.dumps(res) + "\n")
                n += 1
                if n % 500 == 0:
                    fout.flush()
                    print("  %d/%d (%.0fs)" % (n, len(todo), time.time() - t0), flush=True)
    fout.close()
    print("测量完成: %d 单元 (%.0fs), 汇总..." % (n, time.time() - t0), flush=True)

    # ---- 汇总 ----
    recs = [json.loads(l) for l in open(OUTU, encoding="utf-8") if "skip" not in l]
    keys = ["l0_pre", "l0_pay", "genie_pre", "genie_pay", "genie_pre_m1",
            "gamma_pcm_sigma", "y_sigma", "kap_pcm_err"] + \
           ["causal%d_%s" % (k, sg) for k in KS for sg in SEGS]
    nat = {fr["gid"]: fr["sf"] for fr in frames}
    snrs = [None] + C.SNR_LEVELS
    summary = {}
    for snr in snrs:
        sub = [r for r in recs if r["snr"] == snr]
        if not sub:
            continue
        lab = "native" if snr is None else str(snr)
        summary[lab] = {"n_units": len(sub)}
        for k in keys:
            v = np.array([r[k] for r in sub if r.get(k) is not None and r[k] == r[k]])
            if len(v):
                summary[lab][k] = dict(n=len(v), med=float(np.median(v)),
                                       p25=float(np.percentile(v, 25)),
                                       p75=float(np.percentile(v, 75)))
        for kname, key in (("kap_pcm_rmse", "kap_pcm_err"), ("kap_wls_rmse", "kap_wls_err")):
            kv = np.array([r[key] for r in sub if r.get(key) is not None])
            if len(kv):
                summary[lab][kname] = float(np.sqrt(np.mean(kv ** 2)))
        # 分 SF 中位
        for sf in (10, 11):
            ss = [r for r in sub if nat.get(r["gid"]) == sf]
            if not ss:
                continue
            d = {}
            for k in ("l0_pre", "genie_pre_m1", "causal8_pre", "gamma_pcm_sigma"):
                v = np.array([r[k] for r in ss if r.get(k) is not None and r[k] == r[k]])
                if len(v):
                    d[k] = float(np.median(v))
            summary[lab]["sf%d" % sf] = d
    json.dump(summary, open(OUTR, "w", encoding="utf-8"), indent=1)
    hdr = ["native"] + [str(s) for s in C.SNR_LEVELS]
    for k in ("l0_pre", "l0_pay", "genie_pre", "genie_pre_m1", "genie_pay",
              "causal4_pre", "causal8_pre", "causal16_pre",
              "causal4_pay", "causal8_pay", "causal16_pay",
              "gamma_pcm_sigma", "y_sigma"):
        row = "%-16s" % k
        for lab in hdr:
            d = summary.get(lab, {}).get(k)
            row += " %7s" % ("—" if not d else "%.3f" % d["med"])
        print(row)
    print("kap_wls_rmse: " + " ".join(
        "%s:%.4f" % (lab, summary[lab].get("kap_wls_rmse", float("nan"))) for lab in hdr))


if __name__ == "__main__":
    main()

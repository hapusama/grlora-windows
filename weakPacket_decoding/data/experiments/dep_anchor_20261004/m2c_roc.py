# -*- coding: utf-8 -*-
"""M2c ROC：dep4（两级）× 老臂 join（cert/dera/dep2/dep3a/dep3b）→
Pd09 战表（FAR=1e-3，per-(arm,δ,P) 池化精确分位，m2b 同口径）、
配对 w/l、K8/K16 消融、H0 不变性、δ̂ 伪峰、确定性抽查。
→ m2c_roc_results.json
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "..", "keystone_battle_20261003",
                   "d2_battleB_checkpoint.jsonl")
M2CK = os.path.join(HERE, "m2_battle_checkpoint.jsonl")
M2BCK = os.path.join(HERE, "m2b_battle_checkpoint.jsonl")
NEW = os.path.join(HERE, "m2c_battle_checkpoint.jsonl")
FAR = 1e-3
DI_LIST = (0, 3, 5)
DELTAS = {0: 0.0, 3: 0.02, 5: 0.082}
LEVELS_DEP4 = {0: [-28, -30, -32, -34, -36],
               3: [-28, -30, -32, -34, -36],
               5: [-26, -28, -30, -32, -34]}
LEVELS_OLD = [-26, -28, -30, -32, -34, -36, -38]
ARMS_OLD = ("cert", "dera", "dep2", "dep3a", "dep3b")
ARMS_NEW = ("dep4k8", "dep4k16")
LEVELS = {**{a: LEVELS_OLD for a in ARMS_OLD},
          **{a: LEVELS_DEP4 for a in ARMS_NEW}}
SEEDS_H1 = 12


def load():
    h1, h0 = {}, {}
    for path, tag in ((OLD, "old"), (M2CK, "dep2"), (M2BCK, "dep3"),
                      (NEW, "dep4")):
        wc = {}
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
                    d.update(cert=r["cert"], dera=r["dera"], pre=r["pre"])
                elif tag == "dep2":
                    d.update(dep2=r["dep2"])
                elif tag == "dep3":
                    d.update(dep3a=r["dep3a"], dep3b=r["dep3b"])
                else:
                    d.update(dep4k8=r["dep4k8"], dep4k16=r["dep4k16"],
                             dhat16=r.get("dhat16"))
            elif kind == "h0":
                k0 = (r["di"], r["level"], r["seed"], r["frame"])
                i = wc.get(k0, 0)
                wc[k0] = i + 1
                k = k0 + (i,)
                d = h0.setdefault(k, {})
                if tag == "old":
                    d.update(cert=r["cert"], dera=r["dera"], pre=r["pre"])
                elif tag == "dep2":
                    d.update(dep2=r["dep2"])
                elif tag == "dep3":
                    d.update(dep3a=r["dep3a"], dep3b=r["dep3b"])
                else:
                    if "dep4k16" in d:      # 重复作业去重（逐位相同）
                        continue
                    d.update(dep4k8=r["dep4k8"], dep4k16=r["dep4k16"])
    return h1, h0


def snr_at_pd09(pds, lvs):
    lvs = np.array(lvs, dtype=float)
    p = np.array(pds, dtype=float)
    for i in range(len(p) - 1):
        if p[i] >= 0.9 > p[i + 1]:
            return float(lvs[i] + (p[i] - 0.9) / (p[i] - p[i + 1])
                         * (lvs[i + 1] - lvs[i]))
    if p[0] < 0.9:
        return float(lvs[0])
    return float(lvs[-1])


def main():
    h1, h0 = load()
    n_join = sum(1 for v in h1.values() if "dep4k16" in v and "cert" in v)
    n_h0 = sum(1 for v in h0.values() if "dep4k16" in v and "cert" in v)
    print("join: H1 %d / H0 %d 单元" % (n_join, n_h0), flush=True)
    res = dict(n_h1_join=n_join, n_h0_join=n_h0, far=FAR)

    # ---- H0 不变性（跨 δ 中位数漂移）----
    inv = {}
    for a in ARMS_NEW:
        meds = {}
        for k, v in h0.items():
            if a in v:
                meds.setdefault((k[0], k[1]), []).append(v[a])
        allv = [np.median(x) for x in meds.values()]
        if allv:
            inv[a] = dict(di_drift=10 * np.log10(max(allv) / min(allv)),
                          n=len(allv) * 4)
    res["h0_invariance_db"] = inv

    # ---- 门限：per-(arm,δ,P) 池化精确分位（主口径）----
    thr = {}
    for a in ARMS_OLD + ARMS_NEW:
        for di in DI_LIST:
            for P in (8, 16, 32):
                pool = [v[a] for k, v in h0.items()
                        if k[0] == di and a in v and v.get("pre") == P]
                thr[("tP", a, di, P)] = float(np.quantile(pool, 1 - FAR)) \
                    if len(pool) > 300 else float("nan")

    # ---- Pd09 战表（每臂自己的电平格）----
    pd_tab = {}
    for di in DI_LIST:
        row = {}
        for a in ARMS_OLD + ARMS_NEW:
            lvs = LEVELS[a][di] if isinstance(LEVELS[a], dict) else LEVELS[a]
            curve = []
            for lv in lvs:
                sc = [(v[a], v.get("pre")) for k, v in h1.items()
                      if k[0] == di and k[1] == lv and k[2] < SEEDS_H1
                      and a in v]
                curve.append(float(np.mean([
                    val > thr[("tP", a, di, pre_)]
                    if not np.isnan(thr[("tP", a, di, pre_)]) else False
                    for val, pre_ in sc])) if sc else float("nan"))
            row[a] = snr_at_pd09(curve, lvs)
        for a in ARMS_NEW:
            row[a + "_minus_dera"] = row[a] - row["dera"]
        row["cert_minus_dera"] = row["cert"] - row["dera"]
        row["dep3b_minus_dera"] = row["dep3b"] - row["dera"]
        for a in ARMS_OLD + ARMS_NEW:
            row["thr_%s_db" % a] = 10 * np.log10(
                np.mean([thr[("tP", a, di, P)] for P in (8, 16, 32)]))
        pd_tab["δ=%g" % DELTAS[di]] = row
    res["pd09_table"] = pd_tab

    # ---- per-P（δ=0）----
    perP = {}
    for P in (8, 16, 32):
        for a in ARMS_OLD + ARMS_NEW:
            t_a = thr[("tP", a, 0, P)]
            if np.isnan(t_a):
                perP["P=%d %s" % (P, a)] = float("nan")
                continue
            lvs = LEVELS[a][0] if isinstance(LEVELS[a], dict) else LEVELS[a]
            curve = []
            for lv in lvs:
                sc = [v[a] for k, v in h1.items()
                      if k[0] == 0 and k[1] == lv and k[2] < SEEDS_H1
                      and a in v and v.get("pre") == P]
                curve.append(float(np.mean(np.array(sc) > t_a))
                             if sc else np.nan)
            perP["P=%d %s" % (P, a)] = snr_at_pd09(curve, lvs)
    res["pd09_byP_delta0"] = perP

    # ---- 配对 w/l（共享偶数电平，per-P 门限）----
    wl = {}
    shared = {0: [-28, -30, -32, -34], 3: [-28, -30, -32, -34],
              5: [-26, -28, -30, -32, -34]}
    for di in DI_LIST:
        for a in ("dep4k16", "dep4k8", "dep3b"):
            w = l_ = b_ = 0
            for k, v in h1.items():
                if k[0] != di or k[1] not in shared[di] or k[2] >= SEEDS_H1 \
                        or a not in v:
                    continue
                t2 = thr[("tP", a, di, v.get("pre"))]
                td = thr[("tP", "dera", di, v.get("pre"))]
                if np.isnan(t2) or np.isnan(td):
                    continue
                if v[a] > t2 and v["dera"] <= td:
                    w += 1
                elif v[a] <= t2 and v["dera"] > td:
                    l_ += 1
                elif v[a] > t2 and v["dera"] > td:
                    b_ += 1
            wl.setdefault("δ=%g" % DELTAS[di], {})[a] = \
                dict(win=w, loss=l_, both=b_,
                     loss_rate=l_ / max(w + l_, 1))
    res["paired_wl"] = wl

    # ---- δ̂ 伪峰（M3 交接）----
    dhat = {}
    for di in DI_LIST:
        for lv in (-28, -30, -32, -34):
            e = np.array([v["dhat16"] - DELTAS[di] for k, v in h1.items()
                          if k[0] == di and k[1] == lv and k[2] < SEEDS_H1
                          and "dhat16" in v])
            if not len(e):
                continue
            dhat["δ=%g lv=%d" % (DELTAS[di], lv)] = dict(
                n=len(e), p50=float(np.quantile(e, 0.5)),
                frac_gt004=float(np.mean(np.abs(e) > 0.04)),
                frac_gt008=float(np.mean(np.abs(e) > 0.08)))
    res["dhat_dist"] = dhat

    # ---- 确定性抽查（在途重算 2 个 H1 作业 + 1 个 H0 作业）----
    det = []
    import d1_battle as A
    import d2_core as D
    import m2c_core as W
    frames = A.build_frames()
    SEED_CONST = 20261003
    DELTAS_ALL = (0.0, 0.005, 0.01, 0.02, 0.04, 0.082)
    for (di, level, seed, kind) in ((0, -30, 3, "h1"), (5, -28, 1, "h1"),
                                    (3, -34, 2, "h0")):
        old = {}
        for line in open(NEW, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") == kind and (r["di"], r["level"], r["seed"]) \
                    == (di, level, seed):
                old.setdefault(r["frame"], []).append(r["dep4k16"])
        by_pre = {}
        for i, f in enumerate(frames):
            by_pre.setdefault(f["pre"], []).append(i)
        for pre, lst in by_pre.items():
            if kind == "h1":
                segs = np.empty((len(lst), len(frames[lst[0]]["seg"])),
                                dtype=np.complex128)
                for i, fi in enumerate(lst):
                    f = frames[fi]
                    eps = D.eps_of_delta(DELTAS_ALL[di])
                    inj = D.resample_sfo(
                        f["seg"],
                        f["lead"] - (pre + 4.25) * 4096, eps)
                    S, N0 = A.snr_parts(inj)
                    rng = np.random.default_rng(
                        (SEED_CONST * 7919 + (int(level) + 100) * 131
                         + seed * 17 + (f["hs"] % 4099) * 7919
                         + di * 104729) % (2 ** 31))
                    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
                    segs[i] = inj + ((rng.standard_normal(len(inj))
                                      + 1j * rng.standard_normal(len(inj)))
                                     * np.sqrt(p_add / 2.0))
                o = W.score_dep4(segs, frames[lst[0]]["lead"], pre)
                for i, fi in enumerate(lst):
                    det.append(abs(float(o["score_k16"][i])
                                   / old[fi][0] - 1.0))
            else:
                for fi in lst[:2]:
                    f = frames[fi]
                    S, N0 = A.snr_parts(f["seg"])
                    L = len(f["seg"])
                    rng0 = np.random.default_rng(
                        (SEED_CONST * 131 + (int(level) + 100) * 17
                         + seed * 31 + f["hs"] % 7919
                         + di * 104729) % (2 ** 31))
                    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
                    segs = np.stack([
                        (rng0.standard_normal(L)
                         + 1j * rng0.standard_normal(L))
                        * np.sqrt((N0 + p_add) / 2.0)
                        for _ in range(4)])
                    o = W.score_dep4(segs, f["lead"], f["pre"])
                    for j in range(4):
                        det.append(abs(float(o["score_k16"][j])
                                       / old[fi][j] - 1.0))
    res["determinism_max_rel_dev"] = float(max(det)) if det else None
    res["determinism_n"] = len(det)

    json.dump(res, open(os.path.join(HERE, "m2c_roc_results.json"), "w"),
              indent=1, default=str)
    print("== Pd09（FAR=1e-3）==")
    for k, v in pd_tab.items():
        print("%s | dera %.2f | dep4k16 %+.2f | dep4k8 %+.2f | dep3b %+.2f"
              " | cert %+.2f | thr k16 %.2f dera %.2f"
              % (k, v["dera"], v["dep4k16_minus_dera"],
                 v["dep4k8_minus_dera"], v["dep3b_minus_dera"],
                 v["cert_minus_dera"], v["thr_dep4k16_db"],
                 v["thr_dera_db"]))
    print("== byP δ=0 ==")
    for P in (8, 16, 32):
        print("P%d dera %.2f dep4k16 %.2f dep4k8 %.2f dep3b %.2f cert %.2f"
              % (P, perP["P=%d dera" % P], perP["P=%d dep4k16" % P],
                 perP["P=%d dep4k8" % P], perP["P=%d dep3b" % P],
                 perP["P=%d cert" % P]))
    print("== w/l ==")
    for k, v in wl.items():
        print(k, {a: (d["win"], d["loss"], round(d["loss_rate"], 3))
                  for a, d in v.items()})
    print("determinism:", res["determinism_max_rel_dev"],
          "n=", res["determinism_n"])
    print("→ m2c_roc_results.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""M2c 战斗（dep4 两级 vs 老臂 join）：δ∈{0,0.02,0.082}（di=0/3/5 与
m2b 索引一致），窄扫电平，种子 12（H1）/10（H0）。

噪声派生公式与 m2b_battle 逐字一致（SEED_CONST/di/salt 全同）⇒ 老臂
（cert/dera/dep2/dep3a/dep3b）数字直接 join。H0 = 纯合成 AWGN。
分片驱动：--budget 秒（默认 165）；checkpoint 断点续跑。
→ m2c_battle_checkpoint.jsonl
"""
import argparse
import json
import multiprocessing as mp
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import sys

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_battle as A
import d2_core as D
import m2c_core as W

NF = 4096
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "m2c_battle_checkpoint.jsonl")
SEED_CONST = 20261003
DELTAS_ALL = (0.0, 0.005, 0.01, 0.02, 0.04, 0.082)
DI_LIST = (0, 3, 5)
LEVELS_H1 = {0: (-28, -30, -32, -34, -36),
             3: (-28, -30, -32, -34, -36),
             5: (-26, -28, -30, -32, -34, -36)}
LEVELS_H0 = (-28, -34, -38)
N_SEEDS_H1 = 12
N_SEEDS_H0 = 10
N_H0_CENTERS = 4

G = {}


def field_origin(lead, pre):
    return lead - (pre + 4.25) * NF


def inj_cache(fi, di):
    frames = G["frames"]
    key = (fi, di)
    cch = G.setdefault("cache", {})
    if key not in cch:
        f = frames[fi]
        pre = f["pre"]
        eps = D.eps_of_delta(DELTAS_ALL[di])
        inj = D.resample_sfo(f["seg"], field_origin(f["lead"], pre), eps)
        cch[key] = (inj, A.snr_parts(inj))
        if len(cch) > 30:
            for k in list(cch)[:len(cch) - 30]:
                del cch[k]
    return cch[key]


def add_noise_inj(inj, S, N0, level, seed, salt, di):
    rng = np.random.default_rng((SEED_CONST * 7919
                                 + (int(level) + 100) * 131
                                 + seed * 17 + salt * 7919
                                 + di * 104729) % (2 ** 31))
    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
    return inj + ((rng.standard_normal(len(inj))
                   + 1j * rng.standard_normal(len(inj)))
                  * np.sqrt(p_add / 2.0))


def run_h1_unit(u):
    di, level, seed = u
    frames = G["frames"]
    by_pre = G["by_pre"]
    out = []
    for pre, lst in by_pre.items():
        segs = np.empty((len(lst), len(frames[lst[0]]["seg"])),
                        dtype=np.complex128)
        for i, fi in enumerate(lst):
            f = frames[fi]
            inj, (S, N0) = inj_cache(fi, di)
            segs[i] = add_noise_inj(inj, S, N0, level, seed,
                                    f["hs"] % 4099, di)
        lead = frames[lst[0]]["lead"]
        o = W.score_dep4(segs, lead, pre, ret_diag=True)
        for i, fi in enumerate(lst):
            out.append(dict(kind="h1", di=di, delta=DELTAS_ALL[di],
                            level=level, seed=seed, frame=fi, pre=pre,
                            dep4k8=float(o["score_k8"][i]),
                            dep4k16=float(o["score_k16"][i]),
                            dhat16=float(o["dhat_k16"][i])))
    return out


def run_h0_unit(u):
    di, level, seed = u
    frames = G["frames"]
    by_pre = G["by_pre"]
    out = []
    for pre, lst in by_pre.items():
        segs = []
        owners = []
        for fi in lst:
            f = frames[fi]
            _, (S, N0) = inj_cache(fi, di)
            L = len(f["seg"])
            rng0 = np.random.default_rng((SEED_CONST * 131
                                          + (int(level) + 100) * 17
                                          + seed * 31 + f["hs"] % 7919
                                          + di * 104729) % (2 ** 31))
            p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
            sig2_tot = N0 + p_add
            for _ in range(N_H0_CENTERS):
                segs.append((rng0.standard_normal(L)
                             + 1j * rng0.standard_normal(L))
                            * np.sqrt(sig2_tot / 2.0))
                owners.append(fi)
        segs = np.stack(segs)
        lead = frames[lst[0]]["lead"]
        o = W.score_dep4(segs, lead, pre)
        for j, fi in enumerate(owners):
            out.append(dict(kind="h0", di=di, delta=DELTAS_ALL[di],
                            level=level, seed=seed, frame=fi, pre=pre,
                            dep4k8=float(o["score_k8"][j]),
                            dep4k16=float(o["score_k16"][j])))
    return out


def init_worker():
    G["frames"] = A.build_frames()
    by_pre = {}
    for i, f in enumerate(G["frames"]):
        by_pre.setdefault(f["pre"], []).append(i)
    G["by_pre"] = by_pre


def run_job(tag_job):
    tag, di, lv, sd = tag_job
    return (run_h1_unit if tag.startswith("h1") else run_h0_unit)(
        (di, lv, sd))


def run_job_tagged(tag_job):
    return tag_job, run_job(tag_job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=165.0)
    ap.add_argument("--slice", type=int, default=42,
                    help="本轮作业数上限（预算切片）")
    args = ap.parse_args()
    t0 = time.time()
    cnt = {}
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") in ("h1", "h0"):
                k = (r["kind"] + "done", r["di"], r["level"], r["seed"])
                cnt[k] = cnt.get(k, 0) + 1
        print("断点恢复：%d 作业有单元" % len(cnt), flush=True)
    h1_jobs = [(di, lv, sd) for di in DI_LIST
               for lv in LEVELS_H1[di] for sd in range(N_SEEDS_H1)]
    h0_jobs = [(di, lv, sd) for di in DI_LIST
               for lv in LEVELS_H0 for sd in range(N_SEEDS_H0)]
    done_jobs = {k for k, v in cnt.items()
                 if v >= (28 if k[0] == "h1done" else 112)}
    todo = [("h1done",) + u for u in h1_jobs if ("h1done",) + u not in done_jobs]
    todo += [("h0done",) + u for u in h0_jobs
             if ("h0done",) + u not in done_jobs]
    print("待跑 %d 作业（H1 %d / H0 %d）"
          % (len(todo), sum(1 for t in todo if t[0] == "h1"),
             sum(1 for t in todo if t[0] == "h0")), flush=True)
    if not todo:
        print("全部完成", flush=True)
        return
    todo = todo[:args.slice]

    ctx = mp.get_context("spawn")
    n_done = 0
    with open(CKPT, "a", encoding="utf-8") as fh:
        try:
            with ctx.Pool(processes=3, initializer=init_worker) as pool:
                for tag_job, recs in pool.imap_unordered(
                        run_job_tagged, todo, chunksize=1):
                    tag = tag_job[0]
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")
                    fh.write(json.dumps(dict(kind=tag + "done",
                                             di=tag_job[1],
                                             level=tag_job[2],
                                             seed=tag_job[3])) + "\n")
                    fh.flush()
                    n_done += 1
                    if n_done % 10 == 0:
                        print("%s %d/%d (%.0fs)"
                              % (tag, n_done, len(todo), time.time() - t0),
                              flush=True)
        except Exception as e:
            print("异常：%r（checkpoint 已保留）" % e, flush=True)
    print("本轮结束：完成 %d / 待跑余 %d，用时 %.0fs"
          % (n_done, len(todo) - n_done, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

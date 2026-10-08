# -*- coding: utf-8 -*-
"""M2 战·C 臂：dep2c（耦合全前导相干锚 + 全场确认，对照臂——H0 门限与
DeRa 同等待遇 = battle H0 精确分位，无闭式声明）。同帧集/同噪配对
（种子公式与 d2_battleB 逐字一致）。→ m2_battleC_checkpoint.jsonl
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_battle as A
import d2_core as D
import m2_core as M

NF = 4096
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "m2_battleC_checkpoint.jsonl")
SEED_CONST = 20261003
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.082)
LEVELS = [-26, -28, -30, -32, -34, -36, -38]
H0_LEVELS = [-28, -34, -38]
N_SEEDS = 20
N_H0_CENTERS = 4

G = {}


def field_origin(lead, pre):
    return lead - (pre + 4.25) * NF


def inj_cache(fi):
    frames = G["frames"]
    key = (fi, G["di"])
    cch = G.setdefault("cache", {})
    if key not in cch:
        f = frames[fi]
        pre = f["pre"]
        eps = D.eps_of_delta(DELTAS[G["di"]])
        inj = D.resample_sfo(f["seg"], field_origin(f["lead"], pre), eps)
        cch[key] = (inj, A.snr_parts(inj))
        if len(cch) > 40:
            for k in list(cch)[:len(cch) - 40]:
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
    G["di"] = di
    out = []
    for fi, f in enumerate(G["frames"]):
        inj, (S, N0) = inj_cache(fi)
        seg = add_noise_inj(inj, S, N0, level, seed, f["hs"] % 4099, di)
        r = M.score_dep2c(seg[None, :], f["lead"], f["pre"])
        out.append(dict(kind="h1", di=di, delta=DELTAS[di], level=level,
                        seed=seed, frame=fi, pre=f["pre"],
                        dep2c=float(r["score"][0]), dhat=float(r["dhat"][0]),
                        khat=float(r["khat"][0]), nu0=float(r["nu0h"][0])))
    return out


def run_h0_unit(u):
    di, level, seed = u
    G["di"] = di
    out = []
    for fi, f in enumerate(G["frames"]):
        _, (S, N0) = inj_cache(fi)
        L = len(f["seg"])
        rng0 = np.random.default_rng((SEED_CONST * 131
                                      + (int(level) + 100) * 17 + seed * 31
                                      + f["hs"] % 7919
                                      + di * 104729) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        sig2_tot = N0 + p_add
        ws = []
        for k in range(N_H0_CENTERS):
            ws.append((rng0.standard_normal(L) + 1j * rng0.standard_normal(L))
                      * np.sqrt(sig2_tot / 2.0))
        sub = np.stack(ws)
        r = M.score_dep2c(sub, f["lead"], f["pre"])
        for j in range(sub.shape[0]):
            out.append(dict(kind="h0", di=di, delta=DELTAS[di], level=level,
                            seed=seed, frame=fi, pre=f["pre"],
                            dep2c=float(r["score"][j])))
        del ws, sub
    return out


def init_worker():
    G["frames"] = A.build_frames()


def main():
    t0 = time.time()
    print("M2 battleC：dep2c 臂（耦合对照）δ %s｜seeds %d"
          % (list(DELTAS), N_SEEDS), flush=True)
    done_jobs = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("kind") in ("h1done", "h0done"):
                    done_jobs.add((r["kind"], r["di"], r["level"], r["seed"]))
            except Exception:
                pass
        print("断点恢复：%d" % len(done_jobs), flush=True)
    import random
    h1_jobs = [(di, lv, sd) for di in range(len(DELTAS))
               for lv in LEVELS for sd in range(N_SEEDS)]
    h0_jobs = [(di, lv, sd) for di in range(len(DELTAS))
               for lv in H0_LEVELS for sd in range(N_SEEDS)]
    random.Random(20261004).shuffle(h1_jobs)   # 均匀 δ 覆盖（部分完成时无偏）
    random.Random(20261005).shuffle(h0_jobs)
    todo_h1 = [u for u in h1_jobs if ("h1done",) + u not in done_jobs]
    todo_h0 = [u for u in h0_jobs if ("h0done",) + u not in done_jobs]
    print("待跑 H1 %d / H0 %d" % (len(todo_h1), len(todo_h0)), flush=True)

    ctx = mp.get_context("spawn")
    n = 0
    with open(CKPT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=4, initializer=init_worker) as pool:
            for tag, jobs, fn in (("h1", todo_h1, run_h1_unit),
                                  ("h0", todo_h0, run_h0_unit)):
                for i, recs in enumerate(pool.imap_unordered(fn, jobs,
                                                             chunksize=1)):
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")
                    fh.write(json.dumps(dict(kind=tag + "done", di=jobs[i][0],
                                             level=jobs[i][1],
                                             seed=jobs[i][2])) + "\n")
                    fh.flush()
                    n += 1
                    if n % 40 == 0:
                        print("%s %d/%d (%.0fs)" % (tag, i + 1, len(jobs),
                                                    time.time() - t0),
                              flush=True)
    print("完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""M2b 复验战：dep3a / dep3b vs DeRa/cert/dep2（同噪配对 join）。

噪声派生公式与 d2_battleB / m2_battle 逐字一致 ⇒ cert/dera/dep2 老数字
直接 join。帧按 pre 分组批量（约 9× 提速）；双模式共享采集。
H0 = 纯合成 AWGN（keystone 发现2）。真锚 GT = 注入帧模板（修 M2 的
未注入模板度量偏置）。→ m2b_battle_checkpoint.jsonl
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
import d1_core as C
import d1_battle as A
import d2_core as D
import m2b_core as W

NF = 4096
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "m2b_battle_checkpoint.jsonl")
SEED_CONST = 20261003
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.082)
LEVELS = [-26, -28, -30, -32, -34, -36, -38]
H0_LEVELS = [-28, -34, -38]
N_SEEDS = 20
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
        eps = D.eps_of_delta(DELTAS[di])
        inj = D.resample_sfo(f["seg"], field_origin(f["lead"], pre), eps)
        cch[key] = (inj, A.snr_parts(inj))
        if len(cch) > 30:
            for k in list(cch)[:len(cch) - 30]:
                del cch[k]
    return cch[key]


def tmq_cache(fi, di):
    key = (fi, di)
    cch = G.setdefault("tmq", {})
    if key not in cch:
        f = G["frames"][fi]
        inj, _ = inj_cache(fi, di)
        cch[key] = D.clean_template_q(inj, f["lead"], f["pre"])
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
        acq = W.acquire(segs, lead, pre)
        sa, da, ka, ksa = W.confirm(segs, lead, pre, acq["nu0h"],
                                    mode="dep3a")
        sb, db, kb, ksb = W.confirm(segs, lead, pre, acq["nu0h"],
                                    mode="dep3b")
        for i, fi in enumerate(lst):
            f = frames[fi]
            tmq = tmq_cache(fi, di)
            nu_err = acq["nu0h"][i] - (tmq["nu0"] + acq["c_e"]
                                       * tmq["delta"])
            out.append(dict(kind="h1", di=di, delta=DELTAS[di], level=level,
                            seed=seed, frame=fi, pre=pre,
                            dep3a=float(sa[i]), dep3b=float(sb[i]),
                            dhat_a=float(da[i]), dhat_b=float(db[i]),
                            nu0=float(acq["nu0h"][i]),
                            nu_err=float(nu_err),
                            acq=float(acq["acq_score"][i])))
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
        acq = W.acquire(segs, lead, pre)
        sa, da, ka, ksa = W.confirm(segs, lead, pre, acq["nu0h"],
                                    mode="dep3a")
        sb, db, kb, ksb = W.confirm(segs, lead, pre, acq["nu0h"],
                                    mode="dep3b")
        for j, fi in enumerate(owners):
            out.append(dict(kind="h0", di=di, delta=DELTAS[di], level=level,
                            seed=seed, frame=fi, pre=pre,
                            dep3a=float(sa[j]), dep3b=float(sb[j]),
                            acq=float(acq["acq_score"][j])))
    return out


def init_worker():
    G["frames"] = A.build_frames()
    by_pre = {}
    for i, f in enumerate(G["frames"]):
        by_pre.setdefault(f["pre"], []).append(i)
    G["by_pre"] = by_pre


def main():
    t0 = time.time()
    print("M2b battle：δ %s｜levels %s｜seeds %d｜dep3a+dep3b"
          % (list(DELTAS), LEVELS, N_SEEDS), flush=True)
    done_jobs = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("kind") in ("h1done", "h0done"):
                    done_jobs.add((r["kind"], r["di"], r["level"],
                                   r["seed"]))
            except Exception:
                pass
        print("断点恢复：%d 完成作业" % len(done_jobs), flush=True)
    h1_jobs = [(di, lv, sd) for di in range(len(DELTAS))
               for lv in LEVELS for sd in range(N_SEEDS)]
    h0_jobs = [(di, lv, sd) for di in range(len(DELTAS))
               for lv in H0_LEVELS for sd in range(N_SEEDS)]
    todo_h1 = [u for u in h1_jobs if ("h1done",) + u not in done_jobs]
    todo_h0 = [u for u in h0_jobs if ("h0done",) + u not in done_jobs]
    print("待跑 H1 %d / H0 %d" % (len(todo_h1), len(todo_h0)), flush=True)

    ctx = mp.get_context("spawn")
    n_written = 0
    with open(CKPT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=8, initializer=init_worker) as pool:
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
                    n_written += 1
                    if n_written % 20 == 0:
                        print("%s %d/%d (%.0fs)"
                              % (tag, i + 1, len(jobs), time.time() - t0),
                              flush=True)
    print("完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

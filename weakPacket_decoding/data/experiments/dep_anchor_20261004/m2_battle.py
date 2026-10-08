# -*- coding: utf-8 -*-
"""M2 复验战：dep2（拆分认证可部署）vs DeRa/cert/dep —— 与 d2_battleB
同帧集/同噪配对（噪声派生公式逐字一致 ⇒ cert/dera/dep 老臂数字可直接
join 复用，另有确定性抽查复核）。

臂：dep2（K_a=6 采集 + confirm 固定格 bank）。δ 六档 × level −26..−38 ×
20 种子 × 28 帧；H0 {−28,−34,−38}×4 窗纯合成 AWGN（keystone 发现2）。
JSONL 断点续跑。→ m2_battle_checkpoint.jsonl
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
import m2_core as M

NF = 4096
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "m2_battle_checkpoint.jsonl")
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
    frames = G["frames"]
    out = []
    for fi, f in enumerate(frames):
        inj, (S, N0) = inj_cache(fi)
        seg = add_noise_inj(inj, S, N0, level, seed, f["hs"] % 4099, di)
        sc, dh, kh, ph, acq = M.score_dep2(seg[None, :], f["lead"], f["pre"])
        c_e = acq["c_e"]
        nu_err = acq["nu0h"][0] - (f["tm"]["nu0"] + c_e * f["tm"]["delta"])
        out.append(dict(kind="h1", di=di, delta=DELTAS[di], level=level,
                        seed=seed, frame=fi, pre=f["pre"],
                        dep2=float(sc[0]), dhat=float(dh[0]),
                        khat=float(kh[0]), nu0=float(acq["nu0h"][0]),
                        nu_err=float(nu_err),
                        acq=float(acq["acq_score"][0])))
    return out


def run_h0_unit(u):
    di, level, seed = u
    G["di"] = di
    frames = G["frames"]
    out = []
    for fi, f in enumerate(frames):
        _, (S, N0) = inj_cache(fi)
        L = len(G["frames"][fi]["seg"])
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
        sc, dh, kh, ph, acq = M.score_dep2(sub, f["lead"], f["pre"])
        for j in range(sub.shape[0]):
            out.append(dict(kind="h0", di=di, delta=DELTAS[di], level=level,
                            seed=seed, frame=fi, pre=f["pre"],
                            dep2=float(sc[j]), acq=float(acq["acq_score"][j])))
        del ws, sub
    return out


def init_worker():
    G["frames"] = A.build_frames()


def main():
    t0 = time.time()
    print("M2 battle：δ %s｜levels %s｜seeds %d｜dep2(K_a=6)"
          % (list(DELTAS), LEVELS, N_SEEDS), flush=True)
    done_jobs = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("kind") in ("h1done", "h0done"):
                    done_jobs.add((r["kind"], r["di"], r["level"], r["seed"]))
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
        with ctx.Pool(processes=6, initializer=init_worker) as pool:
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
                    if n_written % 40 == 0:
                        print("%s %d/%d (%.0fs)" % (tag, i + 1, len(jobs),
                                                    time.time() - t0),
                              flush=True)
    print("完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

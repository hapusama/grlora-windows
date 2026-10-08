# -*- coding: utf-8 -*-
"""dep2c 的 H0 快速通道（H1 完成后单独跑 H0，8 workers）。
复用 m2_battleC 的 run_h0_unit（逐字）与 checkpoint。"""
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
import m2_battle_c as BC

CKPT = BC.CKPT
DELTAS = BC.DELTAS
H0_LEVELS = BC.H0_LEVELS
N_SEEDS = BC.N_SEEDS


def main():
    t0 = time.time()
    done_jobs = set()
    for line in open(CKPT, encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("kind") == "h0done":
                done_jobs.add((r["di"], r["level"], r["seed"]))
        except Exception:
            pass
    h0_jobs = [(di, lv, sd) for di in range(len(DELTAS))
               for lv in H0_LEVELS for sd in range(N_SEEDS)]
    todo = [u for u in h0_jobs if u not in done_jobs]
    print("H0 快速通道：%d 待跑（已完成 %d）" % (len(todo), len(done_jobs)),
          flush=True)
    ctx = mp.get_context("spawn")
    n = 0
    with open(CKPT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=8, initializer=BC.init_worker) as pool:
            for i, recs in enumerate(pool.imap_unordered(BC.run_h0_unit, todo,
                                                         chunksize=1)):
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
                fh.write(json.dumps(dict(kind="h0done", di=todo[i][0],
                                         level=todo[i][1],
                                         seed=todo[i][2])) + "\n")
                fh.flush()
                n += 1
                if n % 20 == 0:
                    print("h0 %d/%d (%.0fs)" % (i + 1, len(todo),
                                                time.time() - t0), flush=True)
    print("完成 %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

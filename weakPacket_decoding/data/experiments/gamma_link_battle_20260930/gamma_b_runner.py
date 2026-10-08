# -*- coding: utf-8 -*-
"""实验B 加列：GAMMA（γ-链 v3）同种子入 dera_battle_20260929 战表。

复用 battle_runner 的物理/种子（派生常数 20260929）→ 共享链应逐位复现
已发布表；GAMMA 为新列。链：TRIMMER/DERA/TREL-5/GAMMA（我方三链之
TREL 走 kappa_trellis 模块）。
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
        r"\data\experiments\gamma_link_battle_20260930")
BATTLE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
          r"\data\experiments\dera_battle_20260929")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


br = _load("battle_runner", BATTLE + r"\battle_runner.py")

from weak_decoder.decoding.gamma_link import GammaLinkDemodulator
from weak_decoder.decoding.kappa_trellis import KappaTrellisDemodulator

GAM = GammaLinkDemodulator(br.SF, br.OS)
KT = KappaTrellisDemodulator(br.SF, br.OS)
CHAINS = ["TRIMMER", "DERA", "TREL-5", "GAMMA"]
EXP_DIR = HERE
CKPT = os.path.join(EXP_DIR, "checkpoint_B.jsonl")
N = br.N


def chain_rows(seg, psym):
    tri = np.stack([br.trim_demod(samples=seg, start_sample=(16 + k) * br.NF,
                                  sf=br.SF, os_factor=br.OS,
                                  cfo_int=0).metric for k in range(psym)])
    _s1, coh = br.DERA.demod_payload(seg, 16, psym)
    trel = KT.demod_payload(seg, 16, psym, readout="viterbi")
    gamma = GAM.demod_payload(seg, 16, psym)
    return {"TRIMMER": tri, "DERA": coh, "TREL-5": trel, "GAMMA": gamma}


def run_unit(u):
    level, seed, fi = u
    f = br.G["frames"][fi]
    seg = f["seg"]
    if level is not None:
        rng = np.random.default_rng((20260929 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi) % (2 ** 31))
        p_add = max(f["S"] / 10 ** (level / 10.0) - f["N0"], 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    rows_ = chain_rows(seg, f["psym"])
    out = {}
    for c in CHAINS:
        cc = f["delta"]
        rc = rows_[c]
        hard = [(int(np.argmax(rc[k])) - cc) % N for k in range(f["psym"])]
        sym_err = int(sum(int(h != g) for h, g in zip(hard, f["gt"])))
        crc_ok = br.judge_crc_fast(rc, cc, f["gt_hdr"], f["plen"], f["cr"])
        out[c] = {"sym_err": sym_err, "crc_fail": int(not crc_ok)}
    return {"level": level, "seed": seed, "frame": fi,
            "sym_tot": f["psym"], "chains": out}


def init_worker():
    br.G["frames"] = br.build_frames()


def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["level"], r["seed"], r["frame"]))
            except Exception:
                pass
        print("断点恢复：%d" % len(done), flush=True)
    units = [(None, 0, fi) for fi in range(28)]
    for lv in range(-16, -27, -1):
        for sd in range(3):
            for fi in range(28):
                units.append((lv, sd, fi))
    units = [u for u in units if (u[0], u[1], u[2]) not in done]
    print("待跑 %d 单元" % len(units), flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, __import__("os").cpu_count() - 1),
                  initializer=init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units,
                                                        chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 112 == 0:
                    print("  %d/%d (%.0fs)" % (i + 1, len(units),
                                               time.time() - t0), flush=True)

    agg = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["sym_tot"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\n实验B 加列表（SER/PER）")
    for key in ["native"] + ["%+d" % v for v in range(-16, -27, -1)]:
        if key not in agg:
            continue
        n_pkt = sum(1 for line in open(CKPT, encoding="utf-8")
                    if ("native" if json.loads(line)["level"] is None
                        else "%+d" % json.loads(line)["level"]) == key)
        parts = " | ".join("%s %.3f/%.3f"
                           % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n_pkt) for c in CHAINS)
        print("[%7s] %s (n=%d)" % (key, parts, n_pkt), flush=True)
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

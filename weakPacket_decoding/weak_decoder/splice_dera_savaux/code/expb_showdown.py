# -*- coding: utf-8 -*-
"""完美同步对碰（实验B式）：SAVT2 vs DeRa 解码器 vs Savaux 裸链。

同一份干净 CSV 全先验对齐段、同一噪声（种子 20261030 与 v 系同源）、
同一 δ CRC 仲裁判据——隔离"解调器本体"维度（无前端残差时网格无事可做，
DeRa 解码器应当最强；此对碰给出我方链的优势边界：增益仅存在于真实
前端残差下）。
"""
import sys
import os
import json
import time
import numpy as np

import importlib.util as _ilu
_sr_spec = _ilu.spec_from_file_location(
    "sr", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_runner.py")
sr = _ilu.module_from_spec(_sr_spec)
_sr_spec.loader.exec_module(sr)

_v4b_spec = _ilu.spec_from_file_location(
    "v4b", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
           r"\splice_dera_savaux\code\splice_v4b_runner.py")
v4b = _ilu.module_from_spec(_v4b_spec)
_v4b_spec.loader.exec_module(v4b)

SF, N, OS, NF = sr.SF, sr.N, sr.OS, sr.NF
DERA_DEC = sr.DERA_DEC

CHAINS = ["PRIOR×SAVT2", "PRIOR×DERA", "PRIOR×SAVAUX"]
EXP_DIR = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
           r"\experiments\expb_showdown_20261008")
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")
LEVELS = [None, -20, -22, -24, -26]
N_SEEDS = 3


def run_unit(u):
    level, seed, fi = u
    f = sr.G["frames"][fi]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = sr.G["snr"][fi]
    if level is not None:
        rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    n_rel = np.arange(len(seg))
    seg_p = seg * np.exp(-2j * np.pi * f["cfo"] * n_rel / NF)
    seg_p = sr.frac_delay(seg_p, -f["sto_frac"] * OS)

    out = {c: {"sym_err": 0, "crc_fail": 1, "den": 0} for c in CHAINS}
    try:
        outs = v4b.savt2_outputs(seg_p, lead + 8, f["psym"])
        _d, ok, rows = sr.decode_chain(None, f, top5=outs)
        hard = [(int(np.argmax(rows[k])) - _d) % N for k in range(f["psym"])]
        out["PRIOR×SAVT2"] = {
            "sym_err": int(sum(int(h != g) for h, g in zip(hard, f["gt"]))),
            "crc_fail": int(not ok), "den": f["psym"]}
        _s1, rows = DERA_DEC.demod_payload(seg_p, lead + 8, f["psym"])
        _d, ok = sr.decode_chain(rows, f)
        hard = [(int(np.argmax(rows[k])) - _d) % N for k in range(f["psym"])]
        out["PRIOR×DERA"] = {
            "sym_err": int(sum(int(h != g) for h, g in zip(hard, f["gt"]))),
            "crc_fail": int(not ok), "den": f["psym"]}
        rows = sr.sav_rows(seg_p, lead + 8, f["psym"])
        _d, ok = sr.decode_chain(rows, f)
        hard = [(int(np.argmax(rows[k])) - _d) % N for k in range(f["psym"])]
        out["PRIOR×SAVAUX"] = {
            "sym_err": int(sum(int(h != g) for h, g in zip(hard, f["gt"]))),
            "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass
    return {"level": level, "seed": seed, "frame": fi, "chains": out}


def init_worker():
    sr.init_worker()


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
    units = [(lv, sd, fi) for lv in LEVELS
             for sd in range(N_SEEDS if lv is not None else 1)
             for fi in range(28)]
    units = [u for u in units if (u[0], u[1], u[2]) not in done]
    print("待跑 %d 单元" % len(units), flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=6, initializer=init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units,
                                                        chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
    agg, counts = {}, {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        counts[key] = counts.get(key, 0) + 1
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\n完美同步对碰（SER/PER，同判据同种子）")
    for key in ["native"] + ["%+d" % v for v in LEVELS[1:]]:
        if key not in agg:
            continue
        n = counts[key]
        print("[%7s] %s (n=%d)" % (key, " | ".join(
            "%s %.3f/%.3f" % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n) for c in CHAINS), n))
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

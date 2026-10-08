# -*- coding: utf-8 -*-
"""实验A 加格子：DERA×GAMMA（γ-链挂 DeRa 前端），与 dera_front_battle
同种子（20260930 派生）可直接并入其战表。档：native/−20/−22/−24。
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
        r"\data\experiments\gamma_link_battle_20260930")
FRONT = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
         r"\data\experiments\dera_front_battle_20260930")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fr = _load("front_runner", FRONT + r"\front_runner.py")

from weak_decoder.decoding.gamma_link import GammaLinkDemodulator

GAM = GammaLinkDemodulator(fr.SF, fr.OS)
CKPT = os.path.join(HERE, "checkpoint_A_gamma.jsonl")
N = fr.N


def run_unit(u):
    level, seed, fi = u
    f = fr.G["frames"][fi]
    pre = f["pre"]
    lead = pre + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * fr.NF:
                            f["hs"] + (8 + f["psym"] + 2) * fr.NF],
                     dtype=np.complex128)
    S, N0 = fr.G["snr"][fi]
    if level is not None:
        rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    out = {"sym_err": 0, "crc_fail": 1, "den": 0}
    sync_ok = False
    try:
        cands, seg_as, pay0s = fr.dera_sync(seg, pre)
        if cands:
            sync_ok = True
            top5 = []
            for seg_a, pay0 in zip(seg_as, pay0s):
                if pay0 + f["psym"] + 1 > len(seg_a) // fr.NF:
                    continue
                top5.append(GAM.demod_payload(seg_a, pay0, f["psym"]))
            if top5:
                _d, ok, rows = fr.decode_chain(None, f, top5=top5)
                hard = [(int(np.argmax(rows[k])) - _d) % N
                        for k in range(f["psym"])]
                out = {"sym_err": int(sum(int(h != g)
                                          for h, g in zip(hard, f["gt"]))),
                       "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        sync_ok = sync_ok and False
    return {"level": level, "seed": seed, "frame": fi, "sync_dera": sync_ok,
            "chains": {"DERA×GAMMA": out}}


def init_worker():
    fr.init_worker()


def main():
    t0 = time.time()
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
    for lv in (-17, -20, -22, -24):
        for sd in range(3):
            for fi in range(28):
                units.append((lv, sd, fi))
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
                if (i + 1) % 28 == 0:
                    print("  %d/%d (%.0fs)" % (i + 1, len(units),
                                               time.time() - t0), flush=True)

    agg = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        a = agg.setdefault(key, [0, 0, 0, 0, 0])   # sym, den, crc, sync, n
        a[0] += r["chains"]["DERA×GAMMA"]["sym_err"]
        a[1] += r["chains"]["DERA×GAMMA"]["den"]
        a[2] += r["chains"]["DERA×GAMMA"]["crc_fail"]
        a[3] += int(r["sync_dera"])
        a[4] += 1
    print("\nDERA×GAMMA（SER/PER；sync 与 dera_front_battle 同种子一致）")
    for key in ["native", "-17", "-20", "-22", "-24"]:
        if key not in agg:
            continue
        a = agg[key]
        print("[%7s] sync=%.2f | SER %.3f | PER %.3f (n=%d)"
              % (key, a[3] / max(a[4], 1), a[0] / max(a[1], 1), a[2] / a[4],
                 a[4]), flush=True)
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

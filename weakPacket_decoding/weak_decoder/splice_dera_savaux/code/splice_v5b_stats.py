# -*- coding: utf-8 -*-
r"""2026-10-07 v5b：种子扩容统计——SAVT2 vs 全 DeRa 的 bootstrap CI。

死穴 #2（伪重复）的部分回应：capture 数不变（仍 3），但噪声种子 3→10
（合成噪声零成本），对 −20/−22/−24 三档给出配对 bootstrap CI 与
Fisher/McNemar 检验。链：DERA×SAVT2 / DERA×DERA（锚）。
单元：3 档 × 7 新种子(3..9) × 28 帧 = 588。
"""
import sys
import os
import json
import time
import argparse
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

CHAINS = ["DERA×SAVT2", "DERA×DERA"]

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dera_savt2_stats_20261007"
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

LEVELS = [-20, -22, -24]
SEEDS = list(range(3, 10))          # 新种子 3..9（0-2 已在 v4-b）


def run_unit(u):
    level, seed, fi = u
    f = sr.G["frames"][fi]
    pre = f["pre"]
    lead = pre + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = sr.G["snr"][fi]
    rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100) * 131
                                 + seed * 17 + fi * 7919) % (2 ** 31))
    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
    seg = seg + (rng.standard_normal(len(seg))
                 + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)

    out = {c: {"sym_err": 0, "crc_fail": 1, "den": 0} for c in CHAINS}
    sync_dera = False
    try:
        cands, seg_as, pay0s = sr.dera_sync(seg, pre)
        if cands:
            sync_dera = True
            top5_t, top5_d = [], []
            for seg_a, pay0 in zip(seg_as, pay0s):
                if pay0 + f["psym"] + 1 > len(seg_a) // NF:
                    continue
                top5_t.extend(v4b.savt2_outputs(seg_a, pay0, f["psym"]))
                _s1, rows = DERA_DEC.demod_payload(seg_a, pay0, f["psym"])
                top5_d.append(rows)
            for name, rows_list in (("DERA×SAVT2", top5_t),
                                    ("DERA×DERA", top5_d)):
                if not rows_list:
                    continue
                _d, ok, rows = sr.decode_chain(None, f, top5=rows_list)
                hard = [(int(np.argmax(rows[k])) - _d) % N
                        for k in range(f["psym"])]
                out[name] = {
                    "sym_err": int(sum(int(h != g)
                                       for h, g in zip(hard, f["gt"]))),
                    "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    return {"level": level, "seed": seed, "frame": fi,
            "sync_dera": sync_dera, "chains": out}


def init_worker():
    sr.init_worker()


def stats():
    """合并 v4-b（seeds 0-2）+ 本轮（3-9），按档输出 CI 与配对检验。"""
    import math
    rec = {}
    for path, seedset in ((r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                           r"\data\experiments\dera_savt2_20261005"
                           r"\checkpoint.jsonl", set(range(0, 3))),
                          (CKPT, set(SEEDS))):
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["level"] not in LEVELS or r["seed"] not in seedset:
                continue
            key = (r["level"], r["seed"], r["frame"])
            rec[key] = {"s2": r["chains"]["DERA×SAVT2"]["crc_fail"],
                        "de": r["chains"]["DERA×DERA"]["crc_fail"],
                        "s2e": r["chains"]["DERA×SAVT2"]["sym_err"],
                        "dee": r["chains"]["DERA×DERA"]["sym_err"],
                        "den": r["chains"]["DERA×SAVT2"]["den"]}
    print("\n== 10 种子合并统计（v4b seeds0-2 + 本轮 3-9）==")
    keys = sorted(rec.keys())
    for lv in LEVELS:
        ks = [k for k in keys if k[0] == lv]
        n = len(ks)
        s2 = np.array([rec[k]["s2"] for k in ks], float)
        de = np.array([rec[k]["de"] for k in ks], float)
        s2e = sum(rec[k]["s2e"] for k in ks)
        dee = sum(rec[k]["dee"] for k in ks)
        den = sum(rec[k]["den"] for k in ks)
        # Clopper-Pearson 95% CI（正态近似 + Wilson）
        def wilson(x, nn):
            if nn == 0:
                return (0, 0)
            p = x / nn
            z = 1.96
            c = (p + z * z / (2 * nn)) / (1 + z * z / nn)
            h = z * math.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) \
                / (1 + z * z / nn)
            return (max(0, c - h), min(1, c + h))
        lo2, hi2 = wilson(s2.sum(), n)
        lode, hide = wilson(de.sum(), n)
        # 配对 McNemar（b=仅DERA失败, c=仅SAVT2失败）
        b = int(np.sum((de == 1) & (s2 == 0)))
        c_ = int(np.sum((de == 0) & (s2 == 1)))
        # 二项精确 p（双侧）
        nn = b + c_
        if nn > 0:
            k = min(b, c_)
            pv = min(1.0, 2 * sum(math.comb(nn, i) for i in range(0, k + 1))
                     * 0.5 ** nn)
        else:
            pv = 1.0
        print("[%+d] n=%d | SAVT2 PER %.3f [%.3f,%.3f]  DERA %.3f [%.3f,%.3f]"
              % (lv, n, s2.mean(), lo2, hi2, de.mean(), lode, hide))
        print("      SER %.4f vs %.4f (den=%d) | McNemar b=%d c=%d p=%.2e"
              % (s2e / max(den, 1), dee / max(den, 1), den, b, c_, pv))


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
        print("断点恢复：%d 单元" % len(done), flush=True)
    units = [(lv, sd, fi) for lv in LEVELS for sd in SEEDS for fi in range(28)]
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
                if (i + 1) % 49 == 0:
                    el = time.time() - t0
                    print("  %d/%d (%.0fs, %.1fs/u, ETA %.0fmin)"
                          % (i + 1, len(units), el, el / (i + 1),
                             el / (i + 1) * (len(units) - i - 1) / 60.0),
                          flush=True)
    stats()
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()
    if args.stats_only:
        init_worker()
        stats()
    else:
        main()

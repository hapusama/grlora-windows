# -*- coding: utf-8 -*-
r"""2026-10-07 v5：去混血战——网格对齐是否与前端无关（红队死穴 #1 直接回应）。

问题：SAVT2 的两层时延网格长在 DeRa 前端上；换成我方产线前端（OURS，
对齐残差 ±0.2-0.3 bin、Savaux 原样时 −20 档 .168/.619）网格能否同样
救活？若 OURS×SAVT2 ≫ OURS×SAVAUX 且逼近 DERA×SAVT2，则"网格=前端
无关的可插拔对齐层"成立，混血寄生攻击被拆除大半。

臂（5）：
  OURS×SAVAUX   —— v1 基线（.168/.619 @−20，配对锚）
  OURS×SAVT2    —— 主打：我方前端 + 两层网格
  PRIOR×SAVT2   —— 干净全先验天花板（网格形态）
  DERA×SAVT2    —— v4-b 冠军锚（同种子逐位复现）
  DERA×DERA     —— 全 DeRa 参照
档 native + {−17..−26}×3 种子 = 448 单元；种子 20260930 同源。
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

CHAINS = ["OURS×SAVAUX", "OURS×SAVT2", "PRIOR×SAVT2", "DERA×SAVT2",
          "DERA×DERA"]

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dera_savt2_dehybrid_20261007"
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

LEVELS = [None, -17, -20, -22, -24, -26]
N_SEEDS = 3


def run_unit(u):
    level, seed, fi = u
    f = sr.G["frames"][fi]
    pre = f["pre"]
    lead = pre + 6
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

    out = {c: {"sym_err": 0, "crc_fail": 1, "den": 0} for c in CHAINS}
    sync = {"OURS": False, "DERA": False}

    # ---- PRIOR×SAVT2（oracle 天花板，网格形态）----
    try:
        n_rel = np.arange(len(seg))
        seg_p = seg * np.exp(-2j * np.pi * f["cfo"] * n_rel / NF)
        seg_p = sr.frac_delay(seg_p, -f["sto_frac"] * OS)
        outs = v4b.savt2_outputs(seg_p, lead + 8, f["psym"])
        _d, ok, rows = sr.decode_chain(None, f, top5=outs)
        hard = [(int(np.argmax(rows[k])) - _d) % N
                for k in range(f["psym"])]
        out["PRIOR×SAVT2"] = {
            "sym_err": int(sum(int(h != g) for h, g in zip(hard, f["gt"]))),
            "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    # ---- 我方前端 ----
    try:
        sa = sr.sync_and_align(seg, pre)
        if sa is not None:
            sync["OURS"] = True
            seg_a, pay0 = sa["seg_a"], sa["pay0"]
            top5 = v4b.savt2_outputs(seg_a, pay0, f["psym"])
            _d, ok, rows = sr.decode_chain(None, f, top5=top5)
            hard = [(int(np.argmax(rows[k])) - _d) % N
                    for k in range(f["psym"])]
            out["OURS×SAVT2"] = {
                "sym_err": int(sum(int(h != g)
                                   for h, g in zip(hard, f["gt"]))),
                "crc_fail": int(not ok), "den": f["psym"]}
            rows_v1 = sr.sav_rows(seg_a, pay0, f["psym"])
            d_v1, ok_v1 = sr.decode_chain(rows_v1, f)
            hard_v1 = [(int(np.argmax(rows_v1[k])) - d_v1) % N
                       for k in range(f["psym"])]
            out["OURS×SAVAUX"] = {
                "sym_err": int(sum(int(h != g)
                                   for h, g in zip(hard_v1, f["gt"]))),
                "crc_fail": int(not ok_v1), "den": f["psym"]}
    except Exception:
        pass

    # ---- DeRa 前端（锚）----
    try:
        cands, seg_as, pay0s = sr.dera_sync(seg, pre)
        if cands:
            sync["DERA"] = True
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
            "sync_ours": sync["OURS"], "sync_dera": sync["DERA"],
            "chains": out}


def init_worker():
    sr.init_worker()


def aggregate():
    agg, counts, sync = {}, {}, {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        counts[key] = counts.get(key, 0) + 1
        sync.setdefault(key, [0, 0, 0])
        sync[key][0] += int(r["sync_ours"])
        sync[key][1] += int(r["sync_dera"])
        sync[key][2] += 1
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\nv5 去混血战表（SER=同步成功单元内 / PER=含同步失败）")
    for key in ["native"] + ["%+d" % v for v in LEVELS[1:]]:
        if key not in agg:
            continue
        n_pkt = counts[key]
        so = sync[key][0] / max(sync[key][2], 1)
        sd_ = sync[key][1] / max(sync[key][2], 1)
        parts = " | ".join("%s %.3f/%.3f"
                           % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n_pkt) for c in CHAINS)
        print("[%7s] sync OURS=%.2f DERA=%.2f | %s (n=%d)"
              % (key, so, sd_, parts, n_pkt), flush=True)


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
    units = [(lv, sd, fi)
             for lv in LEVELS
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
                if (i + 1) % 28 == 0:
                    el = time.time() - t0
                    print("  %d/%d (%.0fs, %.1fs/u, ETA %.0fmin)"
                          % (i + 1, len(units), el, el / (i + 1),
                             el / (i + 1) * (len(units) - i - 1) / 60.0),
                          flush=True)
    aggregate()
    print("%.0fs elapsed" % (time.time() - t0))


def smoke():
    init_worker()
    print("== native 门禁（28 帧，串行）==")
    tot = {c: [0, 0, 0] for c in CHAINS}
    n_ours = n_dera = 0
    for fi in range(28):
        r = run_unit((None, 0, fi))
        n_ours += int(r["sync_ours"])
        n_dera += int(r["sync_dera"])
        for c in CHAINS:
            d = r["chains"][c]
            tot[c][0] += d["sym_err"]
            tot[c][1] += d["den"]
            tot[c][2] += d["crc_fail"]
    print("sync OURS=%d/28 DERA=%d/28" % (n_ours, n_dera))
    for c in CHAINS:
        print("%s SER=%.4f PER=%.3f" % (c, tot[c][0] / max(tot[c][1], 1),
                                        tot[c][2] / 28.0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        main()

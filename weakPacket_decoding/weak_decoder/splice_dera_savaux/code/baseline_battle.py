# -*- coding: utf-8 -*-
r"""2026-10-08 基线对比战：拼接框架 × 全部发表基线（评审点名项）。

把近战役缺席的基线补进同种子对比（回应 R1-W4/红队：缺 LoRaTrimmer/
UniChirp/传统硬解链基线）。臂（5）：
  DERA×SAVT2   —— 我方冠军（锚，与 dera_savt2_20261005 配对）
  DERA×DERA    —— 全 DeRa（锚）
  DERA×TRIMMER —— LoRaTrimmer（MobiCom'24 port；front battle 复现锚）
  DERA×UNICHIRP—— UniChirp（SECON'26 port；每单元用 4 个前导符号训练
                  相位模型——接收机行为，协议 §4 同款，如实报告其
                  OTA 段内训练天然受限）
  DERA×OLDA    —— 传统 offset-0 抽取硬解链（PLAIN 同款谱行）
物理/种子（20260930→与 v1-v4 同源）与判据完全一致。
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

from weak_decoder.baselines.unichirp.paper_unichirp_demod import (
    UniChirpTrainingSymbol, UniChirpDemodConfig, build_unichirp_phase_model,
    demod_unichirp_symbol as uni_demod, unichirp_full_spectrum)
from weak_decoder.chirp import build_upchirp

SF, N, OS, NF = sr.SF, sr.N, sr.OS, sr.NF
DERA_DEC = sr.DERA_DEC
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")
_REF_NYQ = np.conjugate(build_upchirp(sf=SF, symbol_id=0, os_factor=1)
                        ).astype(np.complex64)

CHAINS = ["DERA×SAVT2", "DERA×DERA", "DERA×TRIMMER", "DERA×UNICHIRP",
          "DERA×OLDA"]

EXP_DIR = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
           r"\experiments\baseline_battle_20261008")
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

LEVELS = [None, -17, -20, -22, -24, -26]
N_SEEDS = 3


def olda_rows(seg_a, pay0, psym):
    rows = []
    for k in range(psym):
        st = (pay0 + k) * NF
        s = seg_a[st:st + NF][::OS]
        rows.append(np.abs(np.fft.fft(s * _REF_NYQ)) ** 2)
    return np.stack(rows)


def unichirp_rows(seg_a, pay0, psym):
    train = []
    for k in range(4):
        full = unichirp_full_spectrum(samples=seg_a, start_sample=k * NF,
                                      sf=SF, os_factor=OS, cfo_int=0,
                                      cfo_frac=0.0, config=UNI_CFG)
        train.append(UniChirpTrainingSymbol(start_sample=k * NF,
                                            raw_fft_bin=int(
                                                np.argmax(full[:N])),
                                            abs_symbol_index=float(k)))
    model, _o = build_unichirp_phase_model(
        samples=seg_a, training_symbols=tuple(train), sf=SF, os_factor=OS,
        config=UNI_CFG)
    rows = []
    for k in range(psym):
        res = uni_demod(samples=seg_a, start_sample=(pay0 + k) * NF, sf=SF,
                        os_factor=OS, phase_rad=model.predict(pay0 + k),
                        config=UNI_CFG)
        rows.append(res.metric)
    return np.stack(rows)


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
    sync_dera = False
    try:
        cands, seg_as, pay0s = sr.dera_sync(seg, pre)
        if cands:
            sync_dera = True
            top5 = {c: [] for c in CHAINS}
            for seg_a, pay0 in zip(seg_as, pay0s):
                if pay0 + f["psym"] + 1 > len(seg_a) // NF:
                    continue
                top5["DERA×SAVT2"].extend(
                    sr_v4b_savt2(seg_a, pay0, f["psym"]))
                _s1, rows = DERA_DEC.demod_payload(seg_a, pay0, f["psym"])
                top5["DERA×DERA"].append(rows)
                top5["DERA×TRIMMER"].append(np.stack([
                    sr.trim_demod(samples=seg_a,
                                  start_sample=(pay0 + k) * NF, sf=SF,
                                  os_factor=OS, cfo_int=0).metric
                    for k in range(f["psym"])]))
                top5["DERA×UNICHIRP"].append(
                    unichirp_rows(seg_a, pay0, f["psym"]))
                top5["DERA×OLDA"].append(olda_rows(seg_a, pay0, f["psym"]))
            for name, rows_list in top5.items():
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


def sr_v4b_savt2(seg_a, pay0, psym):
    """v4-b 的 savt2_outputs（经模块加载，避免本文件复制实现漂移）。"""
    global _V4B
    try:
        return _V4B.savt2_outputs(seg_a, pay0, psym)
    except NameError:
        spec = _ilu.spec_from_file_location(
            "v4b", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                   r"\weak_decoder\splice_dera_savaux\code"
                   r"\splice_v4b_runner.py")
        _V4B = _ilu.module_from_spec(spec)
        spec.loader.exec_module(_V4B)
        return _V4B.savt2_outputs(seg_a, pay0, psym)


def init_worker():
    sr.init_worker()


def aggregate():
    agg, counts, sync = {}, {}, {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        counts[key] = counts.get(key, 0) + 1
        sync[key] = sync.get(key, 0) + int(r["sync_dera"])
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\n基线对比战表（SER=同步成功单元内 / PER=含同步失败）")
    for key in ["native"] + ["%+d" % v for v in LEVELS[1:]]:
        if key not in agg:
            continue
        n_pkt = counts[key]
        parts = " | ".join("%s %.3f/%.3f"
                           % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n_pkt) for c in CHAINS)
        print("[%7s] sync DERA=%.2f | %s (n=%d)"
              % (key, sync[key] / n_pkt, parts, n_pkt), flush=True)


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
    tot = {c: [0, 0, 0] for c in CHAINS}
    n_sync = 0
    for fi in range(28):
        r = run_unit((None, 0, fi))
        n_sync += int(r["sync_dera"])
        for c in CHAINS:
            d = r["chains"][c]
            tot[c][0] += d["sym_err"]
            tot[c][1] += d["den"]
            tot[c][2] += d["crc_fail"]
        print("  f%02d %s" % (fi, " ".join(
            "%s %d/%d" % (c.split("×")[-1], r["chains"][c]["sym_err"],
                          r["chains"][c]["crc_fail"]) for c in CHAINS)),
            flush=True)
    print("sync DERA=%d/28" % n_sync)
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

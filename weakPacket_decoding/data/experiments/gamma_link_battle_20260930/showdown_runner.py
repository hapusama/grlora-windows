# -*- coding: utf-8 -*-
"""决战 runner：同一条 DeRa 前端下 DERA×{TRIMMER, DERA, GAMMA, GAMMA-P}。

用户指令（2026-10-01）：要打赢 DeRa；或同用 DeRa 同步结果与同条件的
LoRaTrimmer 比较。GAMMA-P = γ-链 + **前导/sync 当导频**（已知符号窗口的
γ 测量并入 WLS，磨尖深端 (κ₀,δ) 轨迹；接收机合法知识，同 UniChirp 带噪
前导重训先例；所有链同一段信号）。档 native + −17/−20/−22/−24/−25，
种子派生 20260930（与 dera_front_battle 同种子 → DERA×TRIMMER/
DERA×DERA 旧档位应逐位复现，−25 为新增档）。
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
CHAINS = ["DERA×TRIMMER", "DERA×DERA", "DERA×GAMMA", "DERA×GAMMA-P"]
CKPT = os.path.join(HERE, "checkpoint_A_showdown.jsonl")
N = fr.N


def pilot_windows(cand, pre, pay0):
    """前导+sync 已知符号窗口及其 κ(x) 轨迹 x 索引（相对 payload 符 0）。"""
    out = []
    base = cand.start_sample / float(fr.NF)
    for i in range(pre + 2):                      # preamble P + sync 2
        out.append((cand.start_sample + i * fr.NF, base + i - pay0))
    return out


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
    out = {c: {"sym_err": 0, "crc_fail": 1, "den": 0} for c in CHAINS}
    sync_ok = False
    try:
        cands, seg_as, pay0s = fr.dera_sync(seg, pre)
        if cands:
            sync_ok = True
            for name, mode in (("DERA×TRIMMER", "T"), ("DERA×DERA", "D"),
                               ("DERA×GAMMA", "G"), ("DERA×GAMMA-P", "P")):
                top5 = []
                for c, seg_a, pay0 in zip(cands, seg_as, pay0s):
                    if pay0 + f["psym"] + 1 > len(seg_a) // fr.NF:
                        continue
                    if mode == "T":
                        rows = np.stack([
                            fr.trim_demod(
                                samples=seg_a, start_sample=(pay0 + k) * fr.NF,
                                sf=fr.SF, os_factor=fr.OS,
                                cfo_int=0).metric for k in range(f["psym"])])
                    elif mode == "D":
                        _s1, rows = fr.DERA_DEC.demod_payload(seg_a, pay0,
                                                              f["psym"])
                    elif mode == "G":
                        rows = GAM.demod_payload(seg_a, pay0, f["psym"])
                    else:
                        rows = GAM.demod_payload(
                            seg_a, pay0, f["psym"],
                            pilots=pilot_windows(c, pre, pay0))
                    top5.append(rows)
                if top5:
                    _d, ok, rows = fr.decode_chain(None, f, top5=top5)
                    hard = [(int(np.argmax(rows[k])) - _d) % N
                            for k in range(f["psym"])]
                    out[name] = {"sym_err": int(sum(int(h != g)
                                                    for h, g in zip(hard,
                                                                    f["gt"]))),
                                 "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass
    return {"level": level, "seed": seed, "frame": fi, "sync_dera": sync_ok,
            "chains": out}


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
    for lv in (-17, -20, -22, -24, -25):
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

    agg, sync = {}, {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        sync.setdefault(key, [0, 0])
        sync[key][0] += int(r["sync_dera"])
        sync[key][1] += 1
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\n决战表（同 DeRa 前端，SER/PER；native n=28，其余 n=84）")
    for key in ["native", "-17", "-20", "-22", "-24", "-25"]:
        if key not in agg:
            continue
        n_pkt = sync[key][1]
        parts = " | ".join("%s %.3f/%.3f"
                           % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n_pkt) for c in CHAINS)
        print("[%7s] sync=%.2f | %s (n=%d)"
              % (key, sync[key][0] / max(sync[key][1], 1), parts, n_pkt),
              flush=True)
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

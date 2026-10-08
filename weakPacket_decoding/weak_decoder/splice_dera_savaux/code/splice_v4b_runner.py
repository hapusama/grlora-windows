# -*- coding: utf-8 -*-
r"""2026-10-05 v4-b：偏移副本版时延网格（每副本独立细格 + 跨副本 CRC 仲裁）。

验证（validate_v4b.py）：native 零回归；SAVT 残余 29 失败单元再救 5 个
（跨副本仲裁净增量，raw SER 基本不变 .170→.166——增益来自 FEC 级证据
而非行质量）。

臂（3）：
  DERA×SAVT   —— v4 冠军配对锚（同种子复现 dera_savt_20261005）
  DERA×SAVT2  —— 主打：5 粗副本（中心 ±1.5/±0.75/0）× 各自 7 态细格
                 （±0.375 步 0.125，转移 ±1 细步 + 0.2 罚）→ 副本按路径
                 得分降序进 δ CRC 仲裁（Stage-3 原语同款）
  DERA×DERA   —— 全 DeRa 参照
档 native + {−17..−26}×3 = 448 单元；种子 20260930 同源。
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

_v4_spec = _ilu.spec_from_file_location(
    "v4", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_v4_runner.py")
v4 = _ilu.module_from_spec(_v4_spec)
_v4_spec.loader.exec_module(v4)

SF, N, OS, NF = sr.SF, sr.N, sr.OS, sr.NF
DERA_DEC = sr.DERA_DEC

CHAINS = ["DERA×SAVT", "DERA×SAVT2", "DERA×DERA"]

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dera_savt2_20261005"
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

LEVELS = [None, -17, -20, -22, -24, -26]
N_SEEDS = 3

CENTERS = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
FINE = np.arange(-0.375, 0.376, 0.125)
PEN_F = 0.2


def savt2_outputs(seg_a, pay0, psym):
    """5 副本各自细格 Viterbi → 按路径得分降序的行列表。"""
    abs_d = np.concatenate([CENTERS[k] + FINE for k in range(len(CENTERS))])
    n_d = len(abs_d)
    E = np.empty((psym, n_d))
    rows_all = np.empty((psym, n_d, N))
    for j, d in enumerate(abs_d):
        seg_d = seg_a if d == 0.0 else sr.frac_delay(seg_a, float(d))
        for i in range(psym):
            res = sr.sav_demod(samples=seg_d, start_sample=(pay0 + i) * NF,
                               sf=SF, os_factor=OS, cfo_int=0)
            p = np.abs(res.combined_spectrum.astype(np.complex128)) ** 2
            rows_all[i, j] = p
            E[i, j] = np.log(np.max(p) + 1e-30)
    outs = []
    n_f = len(FINE)
    for k in range(len(CENTERS)):
        sl = slice(k * n_f, (k + 1) * n_f)
        Ek = E[:, sl]
        score = Ek[0].copy()
        back = np.zeros((psym, n_f), dtype=int)
        for i in range(1, psym):
            ns = np.full(n_f, -1e30)
            bk = np.zeros(n_f, dtype=int)
            for s in range(n_f):
                best, arg = -1e30, s
                for s0 in (s - 1, s, s + 1):
                    if 0 <= s0 < n_f:
                        vv = score[s0] - PEN_F * abs(s - s0)
                        if vv > best:
                            best, arg = vv, s0
                ns[s], bk[s] = best, arg
            score = ns + Ek[i]
            back[i] = bk
        path = np.zeros(psym, dtype=int)
        path[-1] = int(np.argmax(score))
        for i in range(psym - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        rows_k = rows_all[:, sl][np.arange(psym), path]
        outs.append((float(score[path[-1]]), rows_k))
    outs.sort(key=lambda t: -t[0])
    return [o[1] for o in outs]


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
                top5["DERA×SAVT"].append(
                    v4.trellis_rows(seg_a, pay0, f["psym"]))
                top5["DERA×SAVT2"].extend(
                    savt2_outputs(seg_a, pay0, f["psym"]))
                _s1, rows = DERA_DEC.demod_payload(seg_a, pay0, f["psym"])
                top5["DERA×DERA"].append(rows)
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
    print("\nv4-b 战表（SER=同步成功单元内 / PER=含同步失败）")
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
    print("== native 门禁（28 帧，串行）==")
    tot = {c: [0, 0, 0] for c in CHAINS}
    n_sync = 0
    for fi in range(28):
        r = run_unit((None, 0, fi))
        n_sync += int(r["sync_dera"])
        line = []
        for c in CHAINS:
            d = r["chains"][c]
            tot[c][0] += d["sym_err"]
            tot[c][1] += d["den"]
            tot[c][2] += d["crc_fail"]
            line.append("%d/%d" % (d["sym_err"], d["crc_fail"]))
        print("  f%02d sync=%d %s" % (fi, r["sync_dera"], " | ".join(line)),
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

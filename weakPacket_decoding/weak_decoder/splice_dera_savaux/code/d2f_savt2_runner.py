# -*- coding: utf-8 -*-
r"""2026-10-08 v6 去混血决战：D2F 检测 × SAVT2 —— 第一条纯自研链。

用户指令："我们能不能用我们的 D2F 的检测"——正是评审会审 P0。拼图互补
逻辑：D2F 检测强项=dep4k16 两级验证（sync 全档≥DERA）+ 连续锚
（native ≤0.011 bin）；其弱点=对齐精度最后一公里（E3-lite PER 中段未
胜）——恰好是 SAVT2 双层时延网格的主场。

臂（4）：
  D2F×SAVT2     —— ★纯自研链：dep4k16 盲检测+连续锚 ν̂（m3p_core）
                   + 双层网格 + Savaux + δ CRC 仲裁
  DERA×SAVT2    —— 配对锚（与 dera_savt2_20261005 逐位复现）
  DERA×DERA     —— 全 DeRa 参照
  D2F×DERADEC   —— D2F 前端 × DeRa 解码器（D2F 线 A 列的近似形态）
物理/种子（20261030 同源）与 v1-v5 一致；对齐约定：
  seg_a = seg·exp(−2πj·nu_rot·n/NF)，nu_rot=nu0h（dhat<0.04 门），
  pay0 = det.pay0//NF（= hs//NF+8，与 DeRa 路径同）。
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

_m3p_spec = _ilu.spec_from_file_location(
    "m3p", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
           r"\D2F_20261005\01_core\m3p_core.py")
m3p = _ilu.module_from_spec(_m3p_spec)
_m3p_spec.loader.exec_module(m3p)

_v4b_spec = _ilu.spec_from_file_location(
    "v4b", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
           r"\splice_dera_savaux\code\splice_v4b_runner.py")
v4b = _ilu.module_from_spec(_v4b_spec)
_v4b_spec.loader.exec_module(v4b)

SF, N, OS, NF = sr.SF, sr.N, sr.OS, sr.NF
DERA_DEC = sr.DERA_DEC

CHAINS = ["D2F×SAVT2", "DERA×SAVT2", "DERA×DERA", "D2F×DERADEC"]

EXP_DIR = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
           r"\experiments\d2f_savt2_20261008")
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

LEVELS = [None, -17, -20, -22, -24, -26]
N_SEEDS = 3

KCOPIES = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
FINE_D = np.arange(-0.625, 0.626, 0.125)      # 11 态（覆盖 ±0.47 时延残差）
PEN_F = 0.2


def judge_hard(rows, d):
    """judge 一致值映射的硬判决（SF11 runner 同款；对 v1-v4 行与原公式
    逐位等价，对 κ 副本行仅此映射正确）。"""
    rc = np.roll(rows, -(int(d) - 1), axis=1)
    return [int(np.argmax(np.roll(rc[k], -1))) for k in range(rows.shape[0])]


def savt2f_outputs(seg_a, pay0, psym):
    """频率偏移副本版：5 个 κ 副本（整体旋 ±0.5 bin）× 各自 11 态细时延
    网格（±0.625 步 0.125，转移 ±1 步 + 0.2 罚）→ 副本按路径得分降序。"""
    outs = []
    n = np.arange(len(seg_a))
    n_f = len(FINE_D)
    for kc in KCOPIES:
        seg_k = seg_a * np.exp(-2j * np.pi * kc * n / NF) if kc != 0.0 \
            else seg_a
        E = np.empty((psym, n_f))
        rows_all = np.empty((psym, n_f, N))
        for j, d in enumerate(FINE_D):
            seg_d = seg_k if d == 0.0 else sr.frac_delay(seg_k, float(d))
            for i in range(psym):
                res = sr.sav_demod(samples=seg_d,
                                   start_sample=(pay0 + i) * NF, sf=SF,
                                   os_factor=OS, cfo_int=0)
                p = np.abs(res.combined_spectrum.astype(np.complex128)) ** 2
                rows_all[i, j] = p
                E[i, j] = np.log(np.max(p) + 1e-30)
        score = E[0].copy()
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
            score = ns + E[i]
            back[i] = bk
        path = np.zeros(psym, dtype=int)
        path[-1] = int(np.argmax(score))
        for i in range(psym - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        outs.append((float(score[path[-1]]),
                     rows_all[np.arange(psym), path]))
    outs.sort(key=lambda t: -t[0])
    return [o[1] for o in outs]


def d2f_sync(seg, pre, psym):
    """D2F 前端：dep4k16 盲检测 + 连续锚 → (seg_a, pay0_sym, det)。"""
    det = m3p.dep4_blind_detect(seg, pre)
    if det is None:
        return None
    nu_rot = m3p.M3.nu_rot_of(det["nu0h"], det["dhat"], det["c_e"],
                              pre, psym)
    n = np.arange(len(seg))
    seg_a = seg * np.exp(-2j * np.pi * nu_rot * n / NF)
    pay0 = det["pay0"] // NF
    return seg_a, pay0, det


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
    sync = {"D2F": False, "DERA": False}

    # ---- D2F 前端两链 ----
    try:
        r2 = d2f_sync(seg, pre, f["psym"])
        if r2 is not None:
            seg_a, pay0, det = r2
            if pay0 + f["psym"] + 1 <= len(seg_a) // NF:
                sync["D2F"] = True
                outs = savt2f_outputs(seg_a, pay0, f["psym"])
                _d, ok, rows = sr.decode_chain(None, f, top5=outs)
                hard = judge_hard(rows, _d)
                out["D2F×SAVT2"] = {
                    "sym_err": int(sum(int(h != g)
                                       for h, g in zip(hard, f["gt"]))),
                    "crc_fail": int(not ok), "den": f["psym"]}
                _s1, rows = DERA_DEC.demod_payload(seg_a, pay0, f["psym"])
                _d, ok = sr.decode_chain(rows, f)
                hard = judge_hard(rows, _d)
                out["D2F×DERADEC"] = {
                    "sym_err": int(sum(int(h != g)
                                       for h, g in zip(hard, f["gt"]))),
                    "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    # ---- DeRa 前端两链（锚）----
    try:
        cands, seg_as, pay0s = sr.dera_sync(seg, pre)
        if cands:
            sync["DERA"] = True
            top_t, top_d = [], []
            for seg_a, pay0 in zip(seg_as, pay0s):
                if pay0 + f["psym"] + 1 > len(seg_a) // NF:
                    continue
                top_t.extend(v4b.savt2_outputs(seg_a, pay0, f["psym"]))
                _s1, rows = DERA_DEC.demod_payload(seg_a, pay0, f["psym"])
                top_d.append(rows)
            for name, rows_list in (("DERA×SAVT2", top_t),
                                    ("DERA×DERA", top_d)):
                if not rows_list:
                    continue
                _d, ok, rows = sr.decode_chain(None, f, top5=rows_list)
                hard = judge_hard(rows, _d)
                out[name] = {
                    "sym_err": int(sum(int(h != g)
                                       for h, g in zip(hard, f["gt"]))),
                    "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    return {"level": level, "seed": seed, "frame": fi,
            "sync_d2f": sync["D2F"], "sync_dera": sync["DERA"],
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
        sync[key][0] += int(r["sync_d2f"])
        sync[key][1] += int(r["sync_dera"])
        sync[key][2] += 1
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\nv6 去混血战表（SER=同步成功单元内 / PER=含同步失败）")
    for key in ["native"] + ["%+d" % v for v in LEVELS[1:]]:
        if key not in agg:
            continue
        n_pkt = counts[key]
        s2 = sync[key][0] / n_pkt
        sd_ = sync[key][1] / n_pkt
        parts = " | ".join("%s %.3f/%.3f"
                           % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n_pkt) for c in CHAINS)
        print("[%7s] sync D2F=%.2f DERA=%.2f | %s (n=%d)"
              % (key, s2, sd_, parts, n_pkt), flush=True)


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
    s2 = sd = 0
    for fi in range(28):
        r = run_unit((None, 0, fi))
        s2 += int(r["sync_d2f"])
        sd += int(r["sync_dera"])
        for c in CHAINS:
            d = r["chains"][c]
            tot[c][0] += d["sym_err"]
            tot[c][1] += d["den"]
            tot[c][2] += d["crc_fail"]
        print("  f%02d d2f=%d dera=%d %s" % (
            fi, r["sync_d2f"], r["sync_dera"],
            " ".join("%s %d/%d" % (c.replace("DERA×", "D").replace(
                "D2F×", "F"), r["chains"][c]["sym_err"],
                r["chains"][c]["crc_fail"]) for c in CHAINS)), flush=True)
    print("sync D2F=%d/28 DERA=%d/28" % (s2, sd))
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

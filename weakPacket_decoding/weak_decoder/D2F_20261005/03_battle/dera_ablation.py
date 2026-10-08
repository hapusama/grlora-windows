# -*- coding: utf-8 -*-
"""DeRa 解码器消融（2026-10-06）：DeRa 的解码增益来自哪。

问题（用户指令）：DeRa 解码增益主要来自哪？针对 DeRa 做消融。

DeRa 解码器（= 我方 A 列 port）部件分解与消融臂（fast 路径、参数
干净冻结=与 phase_align_check 同源同噪声 → full/nc 两臂直接 join）：
  full  两段投影 + 非相干初始化 ML φ̂₀ 相干合并 + Δ0 CRC（完整 port）
  nc    非相干合并 |F|²+|T|²（= LoRaTrimmer 度量；join 昨日 N 臂）
  fft   单次全窗去斜 FFT（**无两段结构** = gr-lora 传统形态；FFT 圆周
        性处理回绕但段间相位不可修）
  s1    Stage-1 行 |F+T|²（φ=0 合并，无 φ̂₀）——隔离相位自适应价值
  noD0  full 行但 Δ0=0 单判（隔离 CRC 重试/仲裁价值；ints 众数反映射
        保留 = 与 demap_judge 同源）
解剖部分（ana，native+−24，单候选 dc∈{0.02 对齐, 0.01 半修复}）：
切分/相位结构的条件价值——fold 整数对齐时两段结构应 ≈ fft；fold
行走时（半修复赢家工况）plain FFT 的段间相位对消无 φ̂₀ 可修。

→ 04_results/dera_ablation.jsonl（断点 part+frame+level+seed）
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\weak_decoder\D2F_20261005")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\weak_decoder\D2F_20261005\01_core")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\dera_front_battle_20260930")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\e2e_final_20261004")

import d2_core as D2                             # noqa: E402
import m3_core as M3                             # noqa: E402
import m3p_core as M3P                           # noqa: E402
import front_runner as FR                        # noqa: E402
import fast_tmpl_core as FT                      # noqa: E402
import phase_align_core as PA                    # noqa: E402
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator

NF, N = 4096, 1024
DELTA = 0.02
SEED_CONST = 20261005
LEVELS = (None, -18, -20, -22, -24)
SEEDS = tuple(range(5))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "04_results", "dera_ablation.jsonl")
DD = None
G = {}


def _params(fi):
    if fi in G["par"]:
        return G["par"][fi]
    f = G["frames"][fi]
    pre, psym = f["pre"], f["psym"]
    lead = pre + 6
    tail = (8 + psym + 2) * NF + 64
    seg0 = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + tail],
                      dtype=np.complex128)
    origin = (lead - (pre + 4.25)) * NF
    seg_inj = D2.resample_sfo(seg0, origin, D2.eps_of_delta(DELTA))
    det = M3P.dep4_blind_detect(seg_inj, pre)
    S, n0 = FR.snr_parts(seg_inj[: lead * NF + 8 * NF])
    order = FT.ladder_order(det["dhat"])
    tau_map = {dc: FT.blind_tau0_pre(seg_inj, lead * NF, pre,
                                     float(det["nu0h"]), dc, origin)
               for dc in order}
    G["par"][fi] = dict(seg_inj=seg_inj, origin=origin,
                        m0=(lead + 8) * NF,
                        nu0=float(det["nu0h"]), dhat=float(det["dhat"]),
                        order=order, tau_map=tau_map, S=S, n0=n0)
    if len(G["par"]) > 8:
        for k in list(G["par"])[:len(G["par"]) - 8]:
            del G["par"][k]
    return G["par"][fi]


def judge_d0_only(rows, f):
    """Δ0=0 单判（消融臂；ints 众数反映射保留，与 demap_judge 同源）。"""
    am = np.argmax(rows, axis=1)
    psym = rows.shape[0]
    ints = np.zeros(psym, dtype=int)
    best = None
    for cand in (ints, ints + 1):
        c = (am - cand) % N
        mode = np.bincount(c, minlength=N).argmax()
        dev = np.minimum((c - mode) % N, (mode - c) % N)
        sc = float(np.sum(dev ** 2))
        if best is None or sc < best[0]:
            best = (sc, cand)
    cand = best[1]
    rows_dm = np.stack([np.roll(rows[k], -(cand[k]))
                        for k in range(psym)])
    return bool(FR.judge_crc(rows_dm, 1, f["gt_hdr"], f["plen"], f["cr"]))


def _ladder_arms(seg, f, p):
    """梯内逐候选一次投影 → full/s1/nc/fft 行 + noD0，独立 CRC 状态机。"""
    pre, psym = f["pre"], f["psym"]
    arms = {t: None for t in ("full", "s1", "nc", "fft", "noD0")}
    for dc in p["order"]:
        base = FT._repaired_base(seg, p["m0"], dc, p["tau_map"][dc],
                                 p["nu0"], p["origin"], pre)
        ph = PA.demod_phased(base, pre + 5, psym, DD)
        fm, tm, nc_rows = PA.project_payload(base, pre + 5, psym, DD)
        rows = {"full": ph["rows_const"], "nc": nc_rows,
                "s1": PA.stage1_rows(fm, tm),
                "fft": PA.fft_rows(base, pre + 5, psym)}
        rows["noD0"] = rows["full"]
        for tag in arms:
            if arms[tag] is None:
                if tag == "noD0":
                    ok = judge_d0_only(rows["full"], f)
                    arms[tag] = dict(ok=bool(ok), dc=dc)
                else:
                    ok, ser, d0 = M3.demap_judge(rows[tag],
                                                 np.zeros(psym, dtype=int),
                                                 f)
                    if ok:
                        arms[tag] = dict(ok=True, ser=ser, dc=dc)
        if all(v is not None for v in arms.values()):
            break
    return {t: (v if v is not None else dict(ok=False, dc=None))
            for t, v in arms.items()}


def _anatomy(seg, f, p):
    """单候选 dc∈{0.02,0.01}：full/nc/fft/s1 四变体判决（条件价值）。"""
    pre, psym = f["pre"], f["psym"]
    out = {}
    for dc in (0.02, 0.01):
        base = FT._repaired_base(seg, p["m0"], dc, p["tau_map"][dc],
                                 p["nu0"], p["origin"], pre)
        ph = PA.demod_phased(base, pre + 5, psym, DD)
        fm, tm, nc_rows = PA.project_payload(base, pre + 5, psym, DD)
        var = {"full": ph["rows_const"], "nc": nc_rows,
               "s1": PA.stage1_rows(fm, tm),
               "fft": PA.fft_rows(base, pre + 5, psym)}
        out["dc%.2f" % dc] = {
            t: M3.demap_judge(r, np.zeros(psym, dtype=int), f)[0]
            for t, r in var.items()}
    return out


def run_unit(u):
    fi, lv, sd = u
    f = G["frames"][fi]
    p = _params(fi)
    if lv is None:
        seg = p["seg_inj"]
    else:
        rng = np.random.default_rng(
            (SEED_CONST * 7919 + (int(lv) + 100) * 131
             + sd * 101 + fi * 7919 + 3 * 104729) % (2 ** 31))
        p_add = max(p["S"] / 10 ** (lv / 10.0) - p["n0"], 1e-30)
        seg = p["seg_inj"] + (
            (rng.standard_normal(len(p["seg_inj"]))
             + 1j * rng.standard_normal(len(p["seg_inj"])))
            * np.sqrt(p_add / 2.0))
    rec = dict(frame=fi, part="ab", level=lv, seed=sd, delta=DELTA,
               **_ladder_arms(seg, f, p))
    if lv in (None, -24):
        rec["ana"] = _anatomy(seg, f, p)
    return rec


def init_worker():
    global DD
    G["frames"] = FR.build_frames()
    G["par"] = {}
    DD = DeRaDemodulator(10, 4)


def main():
    t0 = time.time()
    budget_min = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["frame"], r["level"], r["seed"]))
            except Exception:
                pass
        print("断点恢复：%d 单元" % len(done), flush=True)
    units = [(fi, lv, sd) for fi in range(28) for lv in LEVELS
             for sd in SEEDS]
    units = [u for u in units if u not in done]
    ctx = mp.get_context("spawn")
    n = 0
    with open(OUT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=7, initializer=init_worker) as pool:
            for r in pool.imap_unordered(run_unit, units, chunksize=1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n += 1
                if n % 70 == 0:
                    el = (time.time() - t0) / 60.0
                    print("  %d/%d (%.1fmin)" % (n, len(units), el),
                          flush=True)
                    if el > budget_min:
                        print("预算到，优雅退出", flush=True)
                        break
    print("完成 %d 单元，%.1fmin" % (n, (time.time() - t0) / 60.0),
          flush=True)

    # ---- 汇总 ----
    import collections
    agg = collections.defaultdict(
        lambda: collections.defaultdict(lambda: dict(n=0, per=0)))
    ana = collections.defaultdict(
        lambda: collections.defaultdict(lambda: dict(n=0, per=0)))
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("part") != "ab":
            continue
        for arm in ("full", "nc", "fft", "s1", "noD0"):
            a = agg[r["level"]][arm]
            a["n"] += 1
            a["per"] += int(not r[arm]["ok"])
        if "ana" in r:
            for dc, vard in r["ana"].items():
                for t, ok in vard.items():
                    b = ana[(r["level"], dc)][t]
                    b["n"] += 1
                    b["per"] += int(not ok)
    print("\nDeRa 解码器消融（δ=0.02 fast 路径冻结参数，n=140/档）")
    print("SNR     | full(port) | nc(Trimmer) | fft(无两段) | s1(无φ̂₀) | noD0(无重试)")
    for lv in sorted(agg, key=lambda x: (x is not None, x)):
        row = ["PER %.3f" % (agg[lv][a]["per"] / max(agg[lv][a]["n"], 1))
               for a in ("full", "nc", "fft", "s1", "noD0")]
        print("%-7s | %s | %s | %s | %s | %s" % (lv, *row))
    print("\n解剖（单候选，native 与 −24）")
    for lv in (None, -24):
        for dc in ("dc0.02", "dc0.01"):
            d = ana[(lv, dc)]
            if d["full"]["n"]:
                print("%-6s %s n=%d | full %.3f | nc %.3f | fft %.3f | s1 %.3f" % (
                    lv, dc, d["full"]["n"],
                    d["full"]["per"] / d["full"]["n"],
                    d["nc"]["per"] / d["nc"]["n"],
                    d["fft"]["per"] / d["fft"]["n"],
                    d["s1"]["per"] / d["s1"]["n"]))


if __name__ == "__main__":
    main()

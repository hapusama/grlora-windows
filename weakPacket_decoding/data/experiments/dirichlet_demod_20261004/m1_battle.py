# -*- coding: utf-8 -*-
"""m1_battle.py — 四臂先验给定对打（实验B 口径，预注册判定）。

臂：
  TREL-5   产线 kappa_trellis viterbi（现状基线）
  DERA     DeRa port v2（原样，V1/V2 整数中心 + Stage-2 ML φ̂0）
  OURS     γ 条件化 Dirichlet 相干重组（wm 5 列 trellis 走格给整数/κ，
           选中列 κ_sel=(d−2)/4 上的分段 Dirichlet 满能量行 |S|²，
           mask-aware；带限掩模与产线 wm 同款）
  DERA-DT  DeRa 结构 + 分数读出（V1/V2 抽头挪到 κ̂：窗乘 e^{−j2πκ̂n/NF}
           走原投影矩阵；κ̂ 与 OURS 同源 = 同一 trellis 路径列）

铁律：AWGN 唯一、同一实现喂所有链、全先验对齐（δ/STO/CFO 冻结）、
整包 SNR 口径、GT 只来自干净原生解、native 冒烟 + 判据负控。
种子派生 (20261004·7919+(lv+100)·131+seed·101+fi)——与 dera_battle 的
20260929 基不同（独立重复，exp1/exp1b 先例），且 seed·101+fi 无碰撞。
κ̂ 同源同喂：OURS 与 DERA-DT 的 κ̂ 来自同一 noisy 信号同一 trellis 路径
（接收机行为，无 GT）。

输出 m1_checkpoint.jsonl（逐单元）+ m1_battle_results.json（汇总）。
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
        r"\data\experiments\dirichlet_demod_20261004")
BATTLE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
          r"\data\experiments\dera_battle_20260929")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


br = _load("battle_runner_m1", BATTLE + r"\battle_runner.py")

from m1_dirichlet import DirichletDemodulator, DeraDirTapsDemodulator
from weak_decoder.decoding.kappa_trellis import KappaTrellisDemodulator
from weak_decoder.decoding.kappa_trellis import trellis as _tr

SF, N, OS, NF = br.SF, br.N, br.OS, br.NF
KT = KappaTrellisDemodulator(SF, OS)
DIR = DirichletDemodulator(SF, OS)
DDT = DeraDirTapsDemodulator(SF, OS)
CHAINS = ["TREL-5", "DERA", "OURS-Dir", "DERA-DT"]
EXP_DIR = HERE
CKPT = os.path.join(EXP_DIR, "m1_checkpoint.jsonl")
LEVELS = list(range(-20, -27, -1))
SEEDS = 20

G = {}


def chain_rows(seg, psym):
    rows = {}
    rows["TREL-5"] = KT.demod_payload(seg, 16, psym, readout="viterbi")
    _s1, coh = br.DERA.demod_payload(seg, 16, psym)
    rows["DERA"] = coh
    # OURS + 共享 κ̂ 路径（同一 noisy 信号的 trellis 走格）
    ms = DIR._wm_columns(seg, 16, psym)
    lam = _tr.emission_prominence(ms)
    path = _tr.viterbi(lam, 1)
    X = DIR.branch_spectra(seg, 16, psym)
    rows["OURS-Dir"] = np.stack([
        np.abs(DIR.symbol_S(X[k], (int(path[k]) - 2) / 4.0)) ** 2
        for k in range(psym)])
    ksel = (path.astype(np.float64) - 2.0) / 4.0
    rows["DERA-DT"] = DDT.demod_payload(seg, 16, psym, kappa_hat=ksel)
    return rows


def run_unit(u):
    level, seed, fi = u
    f = G["frames"][fi]
    seg = f["seg"]
    if level is not None:
        rng = np.random.default_rng((20261004 * 7919 + (int(level) + 100) * 131
                                     + seed * 101 + fi) % (2 ** 31))
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
            "sym_tot": f["psym"], "bucket": f["bucket"], "chains": out}


def init_worker():
    br.G["frames"] = br.build_frames()
    # 干净帧 κ 桶（分析用，冻结；不喂任何链）——用干净信号 trellis 路径的
    # 列统计：mean|κ_sel|（kappa_pooled 的窄核 argmax 在 ±0.5 落界帧翻
    # 转，实测 27/28 落 k<0.2 与 DTFT 真值矛盾，弃用）
    for f in br.G["frames"]:
        ms = DIR._wm_columns(f["seg"], 16, f["psym"])
        path = _tr.viterbi(_tr.emission_prominence(ms), 1)
        kb = float(np.mean(np.abs(path.astype(float) - 2.0) / 4.0))
        f["bucket"] = ("k<0.2" if kb < 0.2
                       else "0.2-0.3" if kb < 0.3 else "0.3-0.5")
    G["frames"] = br.G["frames"]
    # 判据负控
    f = G["frames"][0]
    rows = {c: np.random.exponential(1.0, (f["psym"], N)) for c in CHAINS}
    ok = br.judge_crc_fast(rows["TREL-5"], f["delta"], f["gt_hdr"],
                           f["plen"], f["cr"])
    assert not ok, "负控失败：随机行不应过 CRC"


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
    units = [(None, 0, fi) for fi in range(28)]
    for lv in LEVELS:
        for sd in range(SEEDS):
            for fi in range(28):
                units.append((lv, sd, fi))
    units = [u for u in units if (u[0], u[1], u[2]) not in done]
    print("待跑 %d 单元" % len(units), flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, os.cpu_count() - 2),
                  initializer=init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units,
                                                        chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 140 == 0:
                    print("  %d/%d (%.0fs)" % (i + 1, len(units),
                                               time.time() - t0), flush=True)

    # ---- 汇总 ----
    agg = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        for c in CHAINS:
            a = agg.setdefault((key, c), [0, 0, 0])
            a[0] += r["chains"][c]["sym_err"]
            a[1] += r["sym_tot"]
            a[2] += r["chains"][c]["crc_fail"]
        agg.setdefault((key, "n"), [0])
        agg[(key, "n")][0] += 1
        if r["level"] is None:
            continue
        bk = r.get("bucket")
        if bk:
            for c in CHAINS:
                a = agg.setdefault((key, bk, c), [0, 0])
                a[0] += r["chains"][c]["sym_err"]
                a[1] += r["sym_tot"]

    def thr10(c):
        xs, ys = [], []
        for lv in LEVELS:
            a = agg.get(("%+d" % lv, c))
            if a and a[1] > 0:
                xs.append(float(lv))
                ys.append(a[0] / a[1])
        for i in range(len(xs) - 1):
            if ys[i] >= 0.10 > ys[i + 1] or (ys[i] > 0.10 > ys[i + 1]):
                return xs[i] + (xs[i + 1] - xs[i]) * (ys[i] - 0.10) / max(
                    ys[i] - ys[i + 1], 1e-9)
        return None

    print("\n==== 战表（SER/PER）====", flush=True)
    for key in ["native"] + ["%+d" % v for v in LEVELS]:
        n = agg[(key, "n")][0]
        parts = " | ".join(
            "%s %.4f/%.4f" % (c, agg[(key, c)][0] / max(agg[(key, c)][1], 1),
                              agg[(key, c)][2] / max(n, 1)) for c in CHAINS)
        print("[%7s] %s (n=%d)" % (key, parts, n), flush=True)
    print("\n10%% SER 门限：", {c: thr10(c) for c in CHAINS}, flush=True)
    print("\nκ 分桶 SER：", flush=True)
    for bk in ("k<0.2", "0.2-0.3", "0.3-0.5"):
        row = {}
        for c in CHAINS:
            e = t = 0
            for lv in LEVELS:
                a = agg.get(("%+d" % lv, bk, c))
                if a:
                    e += a[0]
                    t += a[1]
            row[c] = round(e / max(t, 1), 4)
        print("  %-7s %s" % (bk, row), flush=True)
    rep = {"levels": LEVELS, "seeds": SEEDS, "elapsed_s": time.time() - t0,
           "thr10": {c: thr10(c) for c in CHAINS},
           "table": {"%s|%s" % k: v for k, v in agg.items()}}
    with open(os.path.join(EXP_DIR, "m1_battle_results.json"), "w",
              encoding="utf-8") as fjson:
        json.dump(rep, fjson, indent=1, default=float)
    print("saved m1_battle_results.json (%.0fs)" % rep["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()

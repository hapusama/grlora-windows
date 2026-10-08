# -*- coding: utf-8 -*-
r"""G1 相干版：候选集合 × DERA 式两段相干合并（主方案第二步 v0）。

问题（用户 2026-10-04 主方案）：少量共享同步假设下，充分相干处理能否
兑现收益？与 TREL（非相干软合并）同种子同噪声配对比较。

实现：G1 网格（ν×τ）逐点用 DERA port v2 的相干合并谱行（DeRaDemodulator
.demod_payload 的 cohd，即两段窗 Eqs.20-22 结构）判 SER/CRC。噪声/残差/
判据/种子公式与 g1_map_runner 完全一致（配对可比）。
输出：checkpoint_dera_<lv>.jsonl + A/B/C 三臂 + 与 TREL 配对差。
"""
import sys
import json
import os
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dts_residual_20261004")
import dts_runner as D

SF, N, OS, NF = D.SF, D.N, D.OS, D.NF

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\g1_rescue_map_20261004"
LEVEL = int(os.environ.get("G1_LEVEL", "-24"))
CKPT = os.path.join(EXP_DIR, os.environ.get("G1_CKPT", "checkpoint_dera_m24.jsonl"))

SEEDS = [0, 1, 2]
NUS = [round(-0.30 + 0.05 * i, 2) for i in range(13)]
TAUS = [-0.5, -0.25, 0.0, 0.25, 0.5]


def run_unit(u):
    seed, fi, nu, tau = u
    f = D.G["frames"][fi]
    seg = D.apply_residual(f["seg"], "nu", nu)
    if tau != 0.0:
        seg = D.apply_residual(seg, "tau", tau)
    rng = np.random.default_rng((20261004 * 7919 + (LEVEL + 100) * 131
                                 + seed * 17 + fi) % (2 ** 31))
    p_add = max(f["S"] / 10 ** (LEVEL / 10.0) - f["N0"], 1e-30)
    seg = seg + (rng.standard_normal(len(seg))
                 + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    _s1d, cohd = D.DERA.demod_payload(seg, 16, f["psym"])
    cc = f["delta"]
    hard = [(int(np.argmax(cohd[k])) - cc) % N for k in range(f["psym"])]
    err = [int(h != g) for h, g in zip(hard, f["gt"])]
    crc = D.judge_crc_fast(cohd, cc, f["gt_hdr"], f["plen"], f["cr"])
    return {"frame": fi, "seed": seed, "nu": nu, "tau": tau, "psym": f["psym"],
            "dera": {"ser": sum(err) / f["psym"], "crc": int(crc),
                     "err_mask": err}}


def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["seed"], r["frame"], r["nu"], r["tau"]))
            except Exception:
                pass
        print("断点恢复：%d 单元" % len(done), flush=True)
    units = [(sd, fi, nu, tau) for sd in SEEDS for fi in range(28)
             for nu in NUS for tau in TAUS
             if (sd, fi, nu, tau) not in done]
    print("待跑 %d 单元（%d workers）" % (len(units), os.cpu_count()), flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, os.cpu_count() - 1),
                  initializer=D.init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units, chunksize=4)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 1000 == 0:
                    print("  %d/%d (%.0fs)" % (i + 1, len(units), time.time() - t0),
                          flush=True)

    # 汇总：DERA 三臂 + 与 TREL 同种子配对
    dera = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        dera.setdefault((r["seed"], r["frame"]), {})[(r["nu"], r["tau"])] = r
    tck = {"-20": "checkpoint.jsonl", "-22": "checkpoint_m-22.jsonl",
           "-24": "checkpoint_m-24.jsonl"}["%d" % LEVEL]
    trel = {}
    for line in open(os.path.join(EXP_DIR, tck), encoding="utf-8"):
        r = json.loads(line)
        trel.setdefault((r["seed"], r["frame"]), {})[(r["nu"], r["tau"])] = r

    a_fail = b_res = c_pot = 0
    pair = []
    for k, g in dera.items():
        a = g.get((0.0, 0.0))
        if a is None:
            continue
        if not a["dera"]["crc"]:
            a_fail += 1
            if any(v["dera"]["crc"] for v in g.values()):
                b_res += 1
            ckey = min(g, key=lambda p: g[p]["dera"]["ser"])
            if g[ckey]["dera"]["ser"] < 0.5:
                c_pot += 1
        tg = trel.get(k, {})
        ta = tg.get((0.0, 0.0))
        if ta is not None:
            pair.append((int(a["dera"]["crc"]), int(ta["chains"]["trel"]["crc"])))
    print("\n[DERA 相干版三臂 @%ddB] 单元 %d" % (LEVEL, len(dera)))
    print("  A fail %d (%.1f%%) | B rescue %d (%.0f%%) | C pot %d" %
          (a_fail, 100 * a_fail / len(dera), b_res,
           100 * b_res / max(a_fail, 1), c_pot))
    if pair:
        p = np.array(pair)
        print("  配对（同种子同噪声）@前导点: DERA CRC 通过率 %.1f%% vs TREL %.1f%%"
              % (100 * p[:, 0].mean(), 100 * p[:, 1].mean()))
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

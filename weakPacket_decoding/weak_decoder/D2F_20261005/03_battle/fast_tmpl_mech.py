# -*- coding: utf-8 -*-
"""fast_tmpl 机制级噪声战（2026-10-05）：δ=0.02，时间基修复 vs 替代法。

sfo_sto 实验的扩展轮：同数据/同噪声派生（SEED_CONST=20261005、槽位 3），
seeds 0-4（sfo_sto 0-1 的 A_full/D_ladder 数字可作同噪声交叉核对）。
统一仲裁 = M3.demap_judge（Δ0∈{0,±1,±2,±3}；与 sfo_sto 的
FR.decode_chain d_try∈{0,±1,±2} 不同——本实验内全部臂同一判据，归因
防护；与 sfo_sto 发表数字的差异属仲裁差异，交叉核对时声明）。

臂（全部实验B哲学：参数干净信号上冻结）：
  E_fast_oracle  δ_c=GT(0.02) 单候选 + oracle τ̂₀（GT 码值 payload est）
  E_fast_blind   δ 梯 ladder_order(δ̂_clean) + 冻结盲 τ̂_pre（前导，code0）
  E_fast_blind_ab 同上 + B 列 trellis（修复后 κ≈常数，走格退化但可用）
  A_full         sfo_sto oracle 臂复制（resample+frac_delay(tau0_nat)，
                 原 +τ 符号逐字保留——join 一致性优先）
  D_ladder_A     sfo_sto 斜率梯 A 列臂复制（现行 D2F 机制参照）

→ 04_results/fast_tmpl_mech.jsonl（断点：frame+level+seed 键）。
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
import ref_template as RT                        # noqa: E402
import fast_tmpl_core as FT                      # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from decode_trellis import KappaTrellisDemodulator  # noqa: E402
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator

NF, N, OS = 4096, 1024, 4
DELTA = 0.02
SEED_CONST = 20261005
LEVELS = (None, -18, -20, -22, -24)
SEEDS = tuple(range(5))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "04_results", "fast_tmpl_mech.jsonl")
KT, DD = None, None
G = {}


def _params(fi):
    """干净注入信号上的冻结参数（逐 worker 缓存）：ν̂₀/δ̂（dep4 锚）、
    oracle τ̂₀（GT 码值）、盲 τ̂_pre 表（δ_c → τ̂，前导 code0）、
    native τ̂₀（sfo_sto A_full 口径）。"""
    if fi in G["par"]:
        return G["par"][fi]
    f = G["frames"][fi]
    pre, psym = f["pre"], f["psym"]
    lead = pre + 6
    tail = (8 + psym + 2) * NF + 64
    seg0 = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + tail],
                      dtype=np.complex128)
    origin = (lead - (pre + 4.25)) * NF
    hs_abs = lead * NF
    m0 = (lead + 8) * NF
    gt1 = np.asarray(f["gt"]) + 1
    eps = D2.eps_of_delta(DELTA)
    seg_inj = D2.resample_sfo(seg0, origin, eps)
    det = M3P.dep4_blind_detect(seg_inj, pre)
    nu0 = float(det["nu0h"])
    dhat = float(det["dhat"])
    tau_or = RT.est_tau0(seg_inj, m0, gt1, nu0, eps, origin, None)
    tau_nat = RT.est_tau0(seg0, m0, gt1, nu0, 0.0, 0.0, None)
    order = FT.ladder_order(dhat)
    tau_map = {dc: FT.blind_tau0_pre(seg_inj, hs_abs, pre, nu0, dc, origin)
               for dc in order}
    S, n0 = FR.snr_parts(seg_inj[: lead * NF + 8 * NF])
    G["par"][fi] = dict(seg0=seg0, seg_inj=seg_inj, origin=origin,
                        hs_abs=hs_abs, m0=m0, nu0=nu0, dhat=dhat,
                        tau_or=float(tau_or), tau_nat=float(tau_nat),
                        order=order, tau_map=tau_map, S=S, n0=n0,
                        det_ok=det is not None)
    if len(G["par"]) > 8:
        for k in list(G["par"])[:len(G["par"]) - 8]:
            del G["par"][k]
    return G["par"][fi]


def _err_of(rows, f):
    am = np.argmax(rows, axis=1)
    ints = np.zeros(f["psym"], dtype=int)
    # 与 demap_judge 相同的反映射（cand∈{ints,ints+1} 的众数吸收）
    best = None
    for cand in (ints, ints + 1):
        c = (am - cand) % N
        mode = np.bincount(c, minlength=N).argmax()
        dev = np.minimum((c - mode) % N, (mode - c) % N)
        sc = float(np.sum(dev ** 2))
        if best is None or sc < best[0]:
            best = (sc, cand)
    cand = best[1]
    c = (am - cand) % N
    d0 = int(np.bincount(c, minlength=N).argmax())
    err = ((am - cand - d0) % N) != np.asarray(f["gt"])
    return err.astype(int).tolist()


def run_unit(u):
    fi, lv, sd = u
    f = G["frames"][fi]
    pre, psym = f["pre"], f["psym"]
    p = _params(fi)
    if not p["det_ok"]:
        return dict(frame=fi, level=lv, seed=sd, delta=DELTA, det_miss=1)
    if lv is None:
        seg = p["seg_inj"]
    else:
        rng = np.random.default_rng(
            (SEED_CONST * 7919 + (int(lv) + 100) * 131 + sd * 101
             + fi * 7919 + 3 * 104729) % (2 ** 31))
        p_add = max(p["S"] / 10 ** (lv / 10.0) - p["n0"], 1e-30)
        seg = p["seg_inj"] + (
            (rng.standard_normal(len(p["seg_inj"]))
             + 1j * rng.standard_normal(len(p["seg_inj"])))
            * np.sqrt(p_add / 2.0))
    rec = dict(frame=fi, level=lv, seed=sd, delta=DELTA, pre=pre,
               nu0=p["nu0"], dhat=p["dhat"],
               tau_or=p["tau_or"], tau_nat=p["tau_nat"],
               tau_pre=p["tau_map"].get(DELTA))

    # ---- E_fast 三臂（统一 demap_judge 仲裁）----
    r = FT.fast_arm(seg, f, p["hs_abs"], p["m0"], p["nu0"], p["dhat"],
                    p["origin"], DD, ladder=(DELTA,), tau_mode="oracle",
                    tau0_oracle=p["tau_or"])
    rec["e_or"] = dict(ok=r["ok"], ser=r["ser"], tau=r["tau"])
    r = FT.fast_arm(seg, f, p["hs_abs"], p["m0"], p["nu0"], p["dhat"],
                    p["origin"], DD, ladder=p["order"], tau_map=p["tau_map"])
    rec["e_bl"] = dict(ok=r["ok"], ser=r["ser"], dc=r["delta_c"],
                       tau=r["tau"], col=r["col"])
    r = FT.fast_arm(seg, f, p["hs_abs"], p["m0"], p["nu0"], p["dhat"],
                    p["origin"], DD, kt=KT, ladder=p["order"],
                    tau_map=p["tau_map"])
    rec["e_ab"] = dict(ok=r["ok"], ser=r["ser"], dc=r["delta_c"],
                       col=r["col"])

    # ---- A_full（sfo_sto 逐字复制：eps_undo + frac_delay(+tau_nat)）----
    seg_c = D2.resample_sfo(seg, p["origin"], -D2.eps_of_delta(DELTA)
                            / (1.0 + D2.eps_of_delta(DELTA)))
    seg_c = FT.frac_delay(seg_c, p["tau_nat"])
    pad = (pre + 5) * NF
    seg_p = np.concatenate((np.zeros(pad, dtype=np.complex128),
                            seg_c[p["m0"]:]))
    base = seg_p * np.exp(-2j * np.pi * p["nu0"] * np.arange(len(seg_p)) / NF)
    rows = DD.demod_payload(base, pre + 5, psym)[1]
    ok, ser, d0 = M3.demap_judge(rows, np.zeros(psym, dtype=int), f)
    rec["a_full"] = dict(ok=bool(ok), ser=ser)
    if lv in (None, -22):
        rec["a_full_err"] = _err_of(rows, f)
        rec["e_bl_err"] = _err_of(FT.fast_rows(
            seg, p["m0"], rec["e_bl"]["dc"] if rec["e_bl"]["dc"] is not None
            else DELTA, rec["e_bl"]["tau"], p["nu0"], p["origin"], pre,
            psym, DD), f)

    # ---- D_ladder_A（sfo_sto 逐字复制：ν_rot 居中 + 斜率梯，A 列）----
    nu_arm = M3.nu_rot_of(p["nu0"], DELTA, (pre - 1) / 2.0, pre, psym)
    seg_p2 = np.concatenate((np.zeros(pad, dtype=np.complex128),
                             seg[p["m0"]:]))
    base2 = seg_p2 * np.exp(-2j * np.pi * nu_arm * np.arange(len(seg_p2))
                            / NF)
    n_win = np.arange(NF)
    last = None
    for sl in (0.0, 0.01, 0.02, 0.03):
        sp = base2
        if sl:
            sp = base2.copy()
            for s in range(psym):
                w0 = (pre + 5 + s) * NF
                sp[w0:w0 + NF] *= np.exp(
                    -2j * np.pi * sl * (s - (psym - 1) / 2.0) * n_win / NF)
        rows = DD.demod_payload(sp, pre + 5, psym)[1]
        ok, ser, d0 = M3.demap_judge(rows, np.zeros(psym, dtype=int), f)
        if ok:
            rec["d_lad"] = dict(ok=True, ser=ser, slope=sl)
            break
        last = (rows, ser)
    if "d_lad" not in rec:
        rec["d_lad"] = dict(ok=False, ser=last[1], slope=None)
    return rec


def init_worker():
    global KT, DD
    G["frames"] = FR.build_frames()
    G["par"] = {}
    KT = KappaTrellisDemodulator(10, 4)
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
             for sd in SEEDS]          # 帧主序：worker 冻结参数缓存命中
    units = [u for u in units if u not in done]
    print("待跑 %d 单元（总 %d）" % (len(units), 28 * len(LEVELS) * len(SEEDS)),
          flush=True)
    ctx = mp.get_context("spawn")
    n = 0
    with open(OUT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=7, initializer=init_worker) as pool:
            for r in pool.imap_unordered(run_unit, units, chunksize=1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n += 1
                if n % 35 == 0:
                    el = (time.time() - t0) / 60.0
                    print("  %d/%d (%.1fmin)" % (n, len(units), el),
                          flush=True)
                    if el > budget_min:
                        print("预算到，优雅退出", flush=True)
                        break
    print("本批完成 %d 单元，%.1fmin" % (n, (time.time() - t0) / 60.0),
          flush=True)

    # ---- 汇总 ----
    import collections
    agg = collections.defaultdict(
        lambda: collections.defaultdict(lambda: dict(n=0, per=0, ser=0)))
    pos = collections.defaultdict(lambda: collections.Counter())
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("det_miss"):
            for arm in ("e_or", "e_bl", "e_ab", "a_full", "d_lad"):
                agg[r["level"]][arm]["n"] += 1
                agg[r["level"]][arm]["per"] += 1
            continue
        for arm in ("e_or", "e_bl", "e_ab", "a_full", "d_lad"):
            a = agg[r["level"]][arm]
            a["n"] += 1
            a["per"] += int(not r[arm]["ok"])
            a["ser"] += r[arm]["ser"]
        for arm in ("e_bl", "a_full"):
            e = r.get(arm + "_err")
            if e:
                for k, v in enumerate(e):
                    pos[(r["level"], arm)][k] += v
    print("\nδ=0.02  PER（n=140/档，seeds 0-4，统一 demap_judge 仲裁）")
    print("SNR     | E_fast_oracle | E_fast_blind  | E_fast_blind_ab | "
          "A_full        | D_ladder_A")
    for lv in sorted(agg, key=lambda x: (x is not None, x)):
        row = []
        for arm in ("e_or", "e_bl", "e_ab", "a_full", "d_lad"):
            a = agg[lv][arm]
            row.append("PER %.3f        " % (a["per"] / max(a["n"], 1)))
        print("%-7s | %s | %s | %s | %s | %s" % (lv, *row))
    print("\n−22 档逐符号错误位置（bin=10 符号，前 40）：")
    for arm in ("e_bl", "a_full"):
        c = pos[(-22, arm)]
        if c:
            hist = [sum(c.get(k, 0) for k in range(b, b + 10))
                    for b in range(0, 40, 10)]
            print("  %-8s %s" % (arm, hist))


if __name__ == "__main__":
    main()

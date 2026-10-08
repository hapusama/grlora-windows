# -*- coding: utf-8 -*-
"""fast_tmpl 链战（2026-10-05，§5A 终判）：D2F-fast vs m3p2 链 vs DeRa。

同噪声 join（m3p 系派生：SEED_CONST=20261004、di∈{0:δ=0, 1:δ=0.02}、
levels {None,−18..−24}、seeds 0-19、28 帧）：
  - m3p_checkpoint.jsonl：dera_per/dera_ser/dep_det/dhat/anchor_err 直接
    join（同噪声同实现）；
  - m3p2_checkpoint.jsonl：u_per/u_ser/u_col/u_slope/oa_per/ob_per join；
  - 本战现算：**仅 u_per=1 的单元**跑 dep4 检测（确定性，与 m3p2 内
    det 逐位一致；ν̂₀ 未存 checkpoint 故重建）→ ours_chain_decode3 的
    fast 救援臂（全盲：δ 梯 ladder_order(δ̂) + 前导盲 τ̂_pre 在含噪
    段上逐候选估计 + A/B 两列 + Δ0 CRC）。u_per=0 单元 fast 跳过
    （单调构造：uf = u ∪ fast，零回归 by design）。

δ=0.082 不入本战（fast 梯上限 ±0.02；该档 m3p2 双灭不变，声明）。
−26 档不入（mech 未覆盖，本轮预算聚焦 −18..−24 五档，声明）。
候选预算：m3p2 56 + fast 70 = 126 判次 vs DeRa 25（错误 CRC 撞门
~2⁻¹⁶/候选，§D5 同款声明）。
→ 04_results/fast_tmpl_checkpoint.jsonl
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
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from decode_trellis import KappaTrellisDemodulator  # noqa: E402
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator

NF, N = 4096, 1024
SEED_CONST = 20261004
DELTAS = {0: 0.0, 1: 0.02}
LEVELS = [None, -18, -20, -22, -24]
N_SEEDS = 20
E2E = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\e2e_final_20261004"
CKPT_M3P = os.path.join(E2E, "m3p_checkpoint.jsonl")
CKPT_M3P2 = os.path.join(E2E, "m3p2_checkpoint.jsonl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "04_results", "fast_tmpl_checkpoint.jsonl")
KT, DD = None, None
G = {}


def _seg(fi, di, lv, sd):
    """m3p_battle.run_unit 逐字的段构造与噪声派生（join 一致性）。"""
    f = G["frames"][fi]
    pre, psym = f["pre"], f["psym"]
    lead = pre + 6
    tail = (8 + psym + 2) * NF + 64
    key = (fi, di)
    cch = G.setdefault("inj", {})
    if key not in cch:
        seg0 = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + tail],
                          dtype=np.complex128)
        origin = (lead - (pre + 4.25)) * NF
        inj = D2.resample_sfo(seg0, origin, D2.eps_of_delta(DELTAS[di]))
        snr = FR.snr_parts(inj[: lead * NF + 8 * NF])
        cch[key] = (inj, origin, snr)
        if len(cch) > 32:
            for k in list(cch)[:len(cch) - 32]:
                del cch[k]
    inj, origin, (S, N0) = cch[key]
    if lv is None:
        return inj, origin
    rng = np.random.default_rng((SEED_CONST * 7919
                                 + (int(lv) + 100) * 131
                                 + sd * 101 + fi * 7919
                                 + di * 104729) % (2 ** 31))
    p_add = max(S / 10 ** (lv / 10.0) - N0, 1e-30)
    seg = inj + ((rng.standard_normal(len(inj))
                  + 1j * rng.standard_normal(len(inj)))
                 * np.sqrt(p_add / 2.0))
    return seg, origin


def run_unit(u):
    di, lv, sd, fi = u
    f = G["frames"][fi]
    pre = f["pre"]
    old2 = G["m3p2"].get((di, lv, sd, fi), {})
    rec = dict(di=di, delta=DELTAS[di], level=lv, seed=sd, frame=fi, pre=pre)
    for k in ("u_per", "u_ser", "u_col", "u_slope", "oa_per", "ob_per",
              "dep_det", "dhat", "anchor_err", "pay_err", "fail_u"):
        if k in old2:
            rec[k] = old2[k]
    old1 = G["m3p"].get((di, lv, sd, fi), {})
    for k in ("dera_per", "dera_ser", "dera_det", "fail_d"):
        if k in old1:
            rec[k] = old1[k]
    if rec.get("u_per") != 1:
        if "u_per" not in rec:
            # join 缺口（m3p2 行缺 u_per，n≈534-559/560）：全链现算补齐，
            # 保证 uf 分母覆盖（声明：u 列仍以 m3p2 join 数字为准）
            try:
                seg, origin = _seg(fi, di, lv, sd)
                det = M3P.dep4_blind_detect(seg, pre)
                rec["dep_det"] = det is not None
                if det is None:
                    rec["u_per"] = 1
                    rec["fast"] = dict(ok=False, why="dep_miss_fresh")
                    rec["uf_per"] = 1
                    return rec
                det = dict(det)
                det["nu_rot"] = M3.nu_rot_of(det["nu0h"], det["dhat"],
                                             det["c_e"], pre, f["psym"])
                det["dc"] = det["dhat"] if abs(det["dhat"]) >= 0.04 else 0.0
                r3 = FT.ours_chain_decode3(seg, det, f, KT, DD)
                rec["u_per"] = int(not r3["u"]["ok"])
                rec["u_fresh"] = 1
                if r3["u"]["ok"]:
                    rec["fast"] = "skip_u_ok"
                    rec["uf_per"] = 0
                else:
                    rec["fast"] = dict(ok=r3["fast"]["ok"],
                                       ser=r3["fast"]["ser"],
                                       col=r3["fast"]["col"],
                                       dc=r3["fast"]["delta_c"],
                                       tau=r3["fast"]["tau"])
                    rec["uf_per"] = int(not r3["uf"]["ok"])
            except Exception as ex:
                rec["u_per"] = 1
                rec["fast"] = dict(ok=False, why="err:" + repr(ex)[:120])
                rec["uf_per"] = 1
            return rec
        rec["fast"] = "skip_u_ok"
        return rec

    # ---- m3p2 失败单元：重建 det（确定性同 m3p2）→ fast 救援（全盲）----
    try:
        seg, origin = _seg(fi, di, lv, sd)
        if rec.get("dep_det") is False:
            rec["fast"] = dict(ok=False, why="dep_miss")
            rec["uf_per"] = 1
            return rec
        det = M3P.dep4_blind_detect(seg, pre)
        if det is None:
            rec["fast"] = dict(ok=False, why="dep_miss_fresh")
            rec["uf_per"] = 1
            return rec
        det = dict(det)
        det["nu_rot"] = M3.nu_rot_of(det["nu0h"], det["dhat"], det["c_e"],
                                     pre, f["psym"])
        fr = FT.fast_arm(seg, f, det["hs"], det["pay0"], det["nu0h"],
                         det["dhat"], origin, DD, kt=KT, tau_mode="blind")
        rec["fast"] = dict(ok=fr["ok"], ser=fr["ser"], col=fr["col"],
                           dc=fr["delta_c"], tau=fr["tau"])
        rec["uf_per"] = int(not fr["ok"])
        rec["uf_ser"] = fr["ser"]
        rec["dhat_fresh"] = float(det["dhat"])
    except Exception as ex:
        rec["fast"] = dict(ok=False, why="err:" + repr(ex)[:120])
        rec["uf_per"] = 1
    return rec


def init_worker():
    global KT, DD
    G["frames"] = FR.build_frames()
    G["inj"] = {}
    KT = KappaTrellisDemodulator(10, 4)
    DD = DeRaDemodulator(10, 4)
    G["m3p2"], G["m3p"] = {}, {}
    for path, key in ((CKPT_M3P2, "m3p2"), (CKPT_M3P, "m3p")):
        src = path if os.path.exists(path) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "04_results", os.path.basename(path))
        if os.path.exists(src):
            for line in open(src, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                G[key][(r["di"], r["level"], r["seed"], r["frame"])] = r


def main():
    t0 = time.time()
    budget_min = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["di"], r["level"], r["seed"], r["frame"]))
            except Exception:
                pass
        print("断点恢复：%d 单元" % len(done), flush=True)
    units = ([(di, None, 0, fi) for di in (0, 1) for fi in range(28)]
             + [(di, lv, sd, fi) for sd in range(N_SEEDS) for di in (0, 1)
                for lv in LEVELS[1:] for fi in range(28)])
    units = [u for u in units if u not in done]
    print("待跑 %d 单元（总 %d；仅 m3p2 失败单元现算）"
          % (len(units), 2 * 28 * (N_SEEDS * (len(LEVELS) - 1) + 1)),
          flush=True)
    ctx = mp.get_context("spawn")
    n = 0
    with open(OUT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=5, initializer=init_worker) as pool:
            for r in pool.imap_unordered(run_unit, units, chunksize=1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n += 1
                if n % 56 == 0:
                    el = (time.time() - t0) / 60.0
                    print("  %d/%d (%.1fmin)" % (n, len(units), el),
                          flush=True)
                    if el > budget_min:
                        print("预算到，优雅退出", flush=True)
                        break
    print("本批完成 %d 单元，%.1fmin" % (n, (time.time() - t0) / 60.0),
          flush=True)

    # ---- 汇总：PER 表（uf = u ∪ fast；DeRa join m3p）----
    import collections
    agg = collections.defaultdict(
        lambda: collections.defaultdict(lambda: dict(n=0, per=0)))
    rescued = collections.Counter()
    fdc = collections.Counter()
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        di, lv = r["di"], r["level"]
        u = r.get("u_per")
        if u is None:
            continue
        for name, val in (("u", u), ("uf", r.get("uf_per", u)),
                          ("dera", r.get("dera_per"))):
            if val is not None:
                a = agg[(di, lv)][name]
                a["n"] += 1
                a["per"] += int(val)
        fr = r.get("fast")
        if isinstance(fr, dict) and u == 1:
            if fr.get("ok"):
                rescued[(di, lv)] += 1
                if fr.get("dc") is not None:
                    fdc[(di, lv, round(float(fr["dc"]), 3))] += 1
    print("\n=== 链战 PER（20 种子 × 28 帧/档；uf = m3p2 ∪ fast 救援）===")
    print("δ     | SNR   | n   | m3p2(u) | D2F-fast(uf) | DeRa(join)")
    for di in (0, 1):
        for lv in sorted({k[1] for k in agg if k[0] == di},
                         key=lambda x: (x is not None, x)):
            a = {nm: agg[(di, lv)][nm] for nm in ("u", "uf", "dera")}
            print("%.3f | %-5s | %3d | %.3f   | %.3f        | %s"
                  % (DELTAS[di], lv, a["u"]["n"], a["u"]["per"] / a["u"]["n"],
                     a["uf"]["per"] / a["uf"]["n"],
                     ("%.3f" % (a["dera"]["per"] / a["dera"]["n"]))
                     if a["dera"]["n"] else "—"))
    print("\nfast 救回数（u败→fast过）：%s" % dict(rescued))
    print("救回单元 δ_c 分布：%s" % dict(fdc))


if __name__ == "__main__":
    main()

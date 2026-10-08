# -*- coding: utf-8 -*-
"""phase_ab_battle：两段相位补齐链战 + 系统模块消融（2026-10-06，§5A）。

同噪声 join（m3p 派生 SEED_CONST=20261004，di∈{0:δ=0, 1:δ=0.02}，
levels {None,−18..−24}，seeds 0-19，28 帧）：
  join（免费）：
    oa/ob/u    = m3p/m3p2 checkpoint（系统阶梯 L0/L1/L2）
    uf         = fast_tmpl_checkpoint（L3 = +fast 救援 A→B）
    dera_*     = m3p checkpoint（DeRa 同噪声参照）
  现算（仅 u_per=1 单元，dep4 确定性重建 + 一次候选循环三套列组合）：
    fp = a→p→b（L4 = +P 相位补齐列，新 shipping 候选）
    fa = a     （消融：救援段去 B/P）
    fn = n     （消融：救援段非相干合并 = Trimmer 度量，量化两段
                相干叠加价值）

系统阶梯表（同噪声同判据逐级加模块）＝"模块设计是否合理"的直接回答。
→ 04_results/phase_ab_checkpoint.jsonl
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
import m3p_core as M3P                           # noqa: E402
import front_runner as FR                        # noqa: E402
import fast_tmpl_core as FT                      # noqa: E402
import phase_align_core as PA                    # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from decode_trellis import KappaTrellisDemodulator  # noqa: E402
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator

NF, N = 4096, 1024
SEED_CONST = 20261004
DELTAS = {0: 0.0, 1: 0.02}
LEVELS = [None, -18, -20, -22, -24]
N_SEEDS = 20
E2E = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
       r"\experiments\e2e_final_20261004")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "04_results", "phase_ab_checkpoint.jsonl")
KT, DD = None, None
G = {}


def _seg(fi, di, lv, sd):
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
    for k in ("u_per", "u_col", "oa_per", "ob_per", "dep_det", "dhat"):
        if k in old2:
            rec[k] = old2[k]
    old1 = G["m3p"].get((di, lv, sd, fi), {})
    for k in ("u1_per", "dera_per", "dera_ser"):
        if k == "u1_per" and "u_per" in old1:
            rec["u1_per"] = old1["u_per"]
        elif k in old1:
            rec[k] = old1[k]
    if rec.get("u_per") != 1:
        rec["arms"] = "skip_u_ok"
        return rec
    try:
        seg, origin = _seg(fi, di, lv, sd)
        det = M3P.dep4_blind_detect(seg, pre)
        if det is None:
            rec["arms"] = dict(fp=dict(ok=False, why="dep_miss"),
                               fa=dict(ok=False, why="dep_miss"),
                               fn=dict(ok=False, why="dep_miss"))
            rec.update(ufp_per=1, ufa_per=1, ufn_per=1)
            return rec
        sets = PA.fast_arm_sets(seg, f, det["hs"], det["pay0"],
                                float(det["nu0h"]), float(det["dhat"]),
                                origin, DD, kt=KT,
                                sets=(("a", "p", "b"), ("a",), ("n",)))
        rec["arms"] = dict(fp=sets[0], fa=sets[1], fn=sets[2])
        rec.update(ufp_per=int(not sets[0]["ok"]),
                   ufa_per=int(not sets[1]["ok"]),
                   ufn_per=int(not sets[2]["ok"]),
                   dhat_fresh=float(det["dhat"]))
    except Exception as ex:
        rec["arms"] = dict(err=repr(ex)[:160])
        rec.update(ufp_per=1, ufa_per=1, ufn_per=1)
    return rec


def init_worker():
    global KT, DD
    G["frames"] = FR.build_frames()
    G["inj"] = {}
    KT = KappaTrellisDemodulator(10, 4)
    DD = DeRaDemodulator(10, 4)
    G["m3p2"], G["m3p"], G["fastb"] = {}, {}, {}
    for path, key in ((os.path.join(E2E, "m3p2_checkpoint.jsonl"), "m3p2"),
                      (os.path.join(E2E, "m3p_checkpoint.jsonl"), "m3p")):
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
    fb = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "04_results", "fast_tmpl_checkpoint.jsonl")
    if os.path.exists(fb):
        for line in open(fb, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            G["fastb"][(r["di"], r["level"], r["seed"], r["frame"])] = r


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
    print("待跑 %d 单元（总 %d；u败单元现算三臂）"
          % (len(units), 2 * 28 * (N_SEEDS * (len(LEVELS) - 1) + 1)),
          flush=True)
    ctx = mp.get_context("spawn")
    n = 0
    with open(OUT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=7, initializer=init_worker) as pool:
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

    # ---- 汇总：系统阶梯 + 消融 ----
    import collections
    agg = collections.defaultdict(
        lambda: collections.defaultdict(lambda: dict(n=0, per=0)))
    fb = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "04_results", "fast_tmpl_checkpoint.jsonl")
    for line in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        key = (r["di"], r["level"])
        if "u_per" not in r:
            continue
        for nm, v in (("L0_oa", r.get("oa_per")), ("L0_ob", r.get("ob_per")),
                      ("L1_u", r.get("u1_per")), ("L2_u", r.get("u_per")),
                      ("L3_uf", None),
                      ("L4_ufp", r.get("ufp_per", r.get("u_per"))),
                      ("L4a_ufa", r.get("ufa_per", r.get("u_per"))),
                      ("L4n_ufn", r.get("ufn_per", r.get("u_per"))),
                      ("dera", r.get("dera_per"))):
            if v is not None:
                agg[key][nm]["n"] += 1
                agg[key][nm]["per"] += int(v)
    if os.path.exists(fb):
        for line in open(fb, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("u_per") is None:
                continue
            a = agg[(r["di"], r["level"])]["L3_uf"]
            a["n"] += 1
            a["per"] += r.get("uf_per", r["u_per"])
    print("\n=== 系统阶梯（模块逐级添加，同噪声同判据，PER）===")
    print("δ     | SNR   | L0 oa(A,无梯) | L1 u(+B,无梯) | L2 u(+梯) | "
          "L3 uf(+fast) | L4 ufp(+P) | ab ufa(A-only) | ab ufn(N-only) | DeRa")
    for di in (0, 1):
        for lv in sorted({k[1] for k in agg if k[0] == di},
                         key=lambda x: (x is not None, x)):
            def g(nm):
                a = agg[(di, lv)][nm]
                return ("%.3f(n%d)" % (a["per"] / a["n"], a["n"])) \
                    if a["n"] else "—"
            print("%.3f | %-5s | %s | %s | %s | %s | %s | %s | %s | %s" % (
                DELTAS[di], lv, g("L0_oa"), g("L1_u"), g("L2_u"), g("L3_uf"),
                g("L4_ufp"), g("L4a_ufa"), g("L4n_ufn"), g("dera")))
    print("\n（L0=m3p oa 单A列无梯；L1=m3p u(+B)；L2=m3p2 u(+梯)；"
          "L3=+fast(A→B)；L4=+P；现算臂只在 u败 单元，其余继承 join u=0；"
          "L3 全档数字与 RESULTS_FAST_TMPL §4 uf 列一致")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""M3+2 主战场（2026-10-04 深夜）：斜率梯仲裁列修复 δ=0.02。

M3+ 终战（RESULTS_M3P.md）δ=0 已判（50%PER +0.64dB、184:74）；本战
只跑 δ∈{0, 0.02}（0.082 预注册失效场景）：OURS-M3+2 = M3+ 链 +
m3p2_core.ours_chain_decode2（斜率梯 {0,.01,.02,.03}×两列 CRC 逐级
仲裁，δ=0 首候选即过=零回归构造）。DeRa 臂 100% join 自
m3p_checkpoint（全量 8484 单元同噪声）。
用法：python m3p2_battle.py [预算分钟]；断点续跑 m3p2_checkpoint.jsonl。

⚠️ 本文件为 method_m3p_20261005 档案副本；CKPT/OLD 解析到原目录
data/experiments/e2e_final_20261004/（join 一致性保持）。
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

import m3_core as M3                      # noqa: E402
import m3p_core as M3P                    # noqa: E402
import m3p2_core as M3P2                  # noqa: E402
import d2_core as D                       # noqa: E402
import front_runner as FR                 # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from decode_trellis import KappaTrellisDemodulator  # noqa: E402  D2F 自有副本（逐位一致验证）
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator  # noqa: E402

SF, N, OS, NF = M3.SF, M3.N, M3.OS, M3.NF
EXP_DIR = M3.EXP_DIR
CKPT = os.path.join(EXP_DIR, "m3p2_checkpoint.jsonl")
OLD = os.path.join(EXP_DIR, "m3p_checkpoint.jsonl")
SEED_CONST = 20261004
DELTAS = (0.0, 0.02)
LEVELS = [-18, -20, -22, -24, -26]
N_SEEDS = 20

KT = None
DD = None
G = {}


def _fail_class(det, ok, ser, anchor_err, pay_err):
    if det is None:
        return "miss"
    if anchor_err is not None and (anchor_err > 0.75 or pay_err > 2 * NF):
        return "wrong_anchor"
    if ok:
        return "ok"
    return "demod_err" if (ser is None or ser > 0) else "crc_rej"


def run_unit(u):
    di, level, seed, fi = u
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
        inj = D.resample_sfo(seg0, origin, D.eps_of_delta(DELTAS[di]))
        tm_d = D.clean_template_q(inj, lead * NF, pre)
        snr = FR.snr_parts(inj[: lead * NF + 8 * NF])
        cch[key] = (inj, tm_d, snr)
        if len(cch) > 32:
            for k in list(cch)[:len(cch) - 32]:
                del cch[k]
    inj, tm_d, (S, N0) = cch[key]
    if level is None:
        seg = inj
    else:
        rng = np.random.default_rng((SEED_CONST * 7919
                                     + (int(level) + 100) * 131
                                     + seed * 101 + fi * 7919
                                     + di * 104729) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = inj + ((rng.standard_normal(len(inj))
                      + 1j * rng.standard_normal(len(inj)))
                     * np.sqrt(p_add / 2.0))

    out = dict(di=di, delta=DELTAS[di], level=level, seed=seed, frame=fi,
               pre=pre)

    # ---- OURS-M3+2：dep4k16 检测 + 连续锚 + 斜率梯两列仲裁 ----
    try:
        det = M3P.dep4_blind_detect(seg, pre)
        out["dep_det"] = det is not None
        if det is not None:
            det = dict(det)
            det["nu_rot"] = M3.nu_rot_of(det["nu0h"], det["dhat"], det["c_e"],
                                         pre, psym)
            det["dc"] = det["dhat"] if abs(det["dhat"]) >= 0.04 else 0.0
            r = M3P2.ours_chain_decode2(seg, det, f, KT, DD)
            out.update(oa_per=int(not r["a"]["ok"]), oa_ser=r["a"]["ser"],
                       ob_per=int(not r.get("b", {"ok": False})["ok"]),
                       ob_ser=r.get("b", {"ser": -1})["ser"],
                       u_per=int(not r["u"]["ok"]), u_ser=r["u"]["ser"],
                       u_col=r["u"].get("col", "a"),
                       u_slope=r["u"].get("slope", 0.0))
            out["anchor_err"] = float(abs(
                det["nu_rot"] - M3.nu_rot_of(tm_d["nu0"], tm_d["delta"],
                                             (pre - 1) / 2.0, pre, psym)))
            out["pay_err"] = int(abs(det["pay0"] - (lead + 8) * NF))
            out["dhat"] = float(det["dhat"])
            out["det_score"] = float(det["score"])
            out["fail_u"] = _fail_class(True, r["u"]["ok"], r["u"]["ser"],
                                        out["anchor_err"], out["pay_err"])
        else:
            out["fail_u"] = "miss"
    except Exception as ex:
        out["dep_err"] = repr(ex)[:150]
        out["fail_u"] = "err"

    # ---- DeRa 臂：m3p 全量 join（同噪声），缺失才现算 ----
    old = G["old"].get((di, level, seed, fi))
    if old is not None and "dera_per" in old:
        for k in ("dera_per", "dera_ser", "dera_det", "dera_anchor_err",
                  "fail_d"):
            if k in old:
                out[k] = old[k]
        out["dera_join"] = 1
    else:
        try:
            cands, seg_as, pay0s = FR.dera_sync(seg, pre)
            out["dera_det"] = bool(cands)
            if cands:
                top5 = []
                for seg_a, pay0 in zip(seg_as, pay0s):
                    if pay0 + psym + 1 > len(seg_a) // NF:
                        continue
                    _s1, rows = FR.DERA_DEC.demod_payload(seg_a, pay0, psym)
                    top5.append(rows)
                if top5:
                    _d, ok, rows = FR.decode_chain(None, f, top5=top5)
                    hard = [(int(np.argmax(rows[k])) - _d) % N
                            for k in range(psym)]
                    ser = int(sum(int(h != g) for h, g in zip(hard, f["gt"])))
                    out["dera_per"] = int(not ok)
                    out["dera_ser"] = ser
                    aerr = float(min(abs(c.hs_est - lead * NF) for c in cands))
                    out["dera_anchor_err"] = aerr
                    out["fail_d"] = _fail_class(True, ok, ser,
                                                aerr / 4.0 if aerr <= 8
                                                else 99.0, aerr)
                else:
                    out["dera_per"] = 1
                    out["fail_d"] = "wrong_anchor"
            else:
                out["dera_per"] = 1
                out["fail_d"] = "miss"
        except Exception as ex:
            out["dera_err"] = str(ex)[:80]
    return out


def init_worker():
    global KT, DD
    G["frames"] = FR.build_frames()
    KT = KappaTrellisDemodulator(SF, OS)
    DD = DeRaDemodulator(SF, OS)
    G["old"] = {}
    if os.path.exists(OLD):
        for line in open(OLD, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            G["old"][(r["di"], r["level"], r["seed"], r["frame"])] = r


def main():
    t0 = time.time()
    budget_min = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["di"], r["level"], r["seed"], r["frame"]))
            except Exception:
                pass
        print("断点恢复：%d 单元" % len(done), flush=True)
    units = ([(di, None, 0, fi) for di in range(len(DELTAS))
              for fi in range(28)]
             + [(di, lv, sd, fi) for sd in range(N_SEEDS)
                for di in range(len(DELTAS))
                for lv in LEVELS for fi in range(28)])
    units = [u for u in units if u not in done]
    print("待跑 %d 单元（总 %d）" % (len(units),
                                     2 * 28 * (N_SEEDS * len(LEVELS) + 1)),
          flush=True)
    ctx = mp.get_context("spawn")
    n = 0
    with open(CKPT, "a", encoding="utf-8") as fh:
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


if __name__ == "__main__":
    main()

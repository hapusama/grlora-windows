# -*- coding: utf-8 -*-
"""M3 冒烟/探针（2026-10-04）：P1 门偏置 / P2 Δ 敏感性 / P3 纯噪系统 FAR /
P4 native 全链 + 计时 / P5 判据负控。"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import m3_core as M3                      # noqa: E402
import d1_core as C                       # noqa: E402
import d2_core as D                       # noqa: E402
import front_runner as FR                 # noqa: E402
from weak_decoder.decoding.kappa_trellis import KappaTrellisDemodulator  # noqa: E402
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator  # noqa: E402

SF, N, OS, NF = M3.SF, M3.N, M3.OS, M3.NF
OUT = os.path.join(M3.EXP_DIR, "m3_smoke_results.json")
KT = KappaTrellisDemodulator(SF, OS)
DD = DeRaDemodulator(SF, OS)
G = {}


DELTAS = (0.0, 0.02, 0.082)


def seg_of(f, di=0):
    lead = f["pre"] + 6
    tail = (8 + f["psym"] + 2) * NF + 64
    seg0 = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + tail],
                      dtype=np.complex128)
    inj = D.resample_sfo(seg0, (lead - (f["pre"] + 4.25)) * NF,
                         D.eps_of_delta(DELTAS[di]))
    return seg0, inj, lead


def p1_gate_bias():
    print("== P1 门事件偏置（native，δ=0）==", flush=True)
    res = {}
    for f in G["frames"]:
        seg, _, lead = seg_of(f)
        hs_t = lead * NF
        ev = M3.gate_scan(seg)
        e = [(e_["start"] + int((f["pre"] + 4.25) * NF) - hs_t)
             for e_ in ev[:2]]
        res.setdefault(f["pre"], []).append(e)
        print("  f%02d pre=%2d ev_bias=%s" % (G["frames"].index(f), f["pre"],
                                              e), flush=True)
    allb = [x for v in res.values() for row in v for x in row]
    print("  偏置范围 [%d, %d] 中位 %.0f" % (min(allb), max(allb),
                                             float(np.median(allb))),
          flush=True)
    return {str(k): v for k, v in res.items()}, dict(
        mn=int(min(allb)), mx=int(max(allb)), med=float(np.median(allb)))


def p2_delta_sensitivity():
    print("== P2 dep2 分数 vs Δ（native 2 帧/pre）==", flush=True)
    out = {}
    for pre in (8, 16, 32):
        fs = [f for f in G["frames"] if f["pre"] == pre][:2]
        for f in fs:
            seg, _, lead = seg_of(f)
            hs0 = lead * NF
            offs = np.arange(-384, 385, 24)
            sc, _ = M3._scan_scores(seg, hs0, pre, offs)
            peak = offs[int(np.argmax(sc))]
            print("  pre=%2d psym=%2d: peak@%+d  max=%.1f  min=%.1f  "
                  "score(0)=%.1f" % (pre, f["psym"], peak, sc.max(),
                                     sc.min(), sc[list(offs).index(0)]),
                  flush=True)
            out["p%d_f%02d" % (pre, G["frames"].index(f))] = dict(
                offs=[int(o) for o in offs], sc=[float(x) for x in sc])
    return out


def p3_noise_mc(n_per_pre=30):
    print("== P3 纯噪系统 FAR（%d 段/pre）==" % n_per_pre, flush=True)
    rng = np.random.default_rng(777)
    npass = 0
    for pre in (8, 16, 32):
        L = int((pre + 6 + 8 + 20 + 2) * NF + 64)
        for t in range(n_per_pre):
            seg = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) \
                * np.sqrt(0.5)
            det = M3.dep2_blind_detect(seg, pre)
            if det is not None:
                npass += 1
                print("  !! 过门 pre=%d t=%d score=%.1f" % (pre, t,
                                                             det["score"]),
                      flush=True)
        print("  pre=%d done" % pre, flush=True)
    return dict(n_total=3 * n_per_pre, n_pass=npass)


def p4_native(di_list=(0, 1, 2), n_frames=28):
    print("== P4 native 全链（δ∈%s）==" % (list(di_list),), flush=True)
    rows = []
    t0 = time.time()
    for di in di_list:
        for f in G["frames"][:n_frames]:
            r = run_one(f, di, None, 0)
            rows.append(r)
            print("  d%d f%02d dep2=%s cert=%s dera=%s "
                  "A=%.0f/%.0f B=%.0f/%.0f U=%s"
                  % (di, r["frame"], r.get("dep_det"), r.get("cert_det"),
                     r.get("dera_det"),
                     r.get("oa_per", 9) * 10, r.get("oa_ser", -1),
                     r.get("ob_per", 9) * 10, r.get("ob_ser", -1),
                     r.get("u_per", 9)), flush=True)
    dt = time.time() - t0
    print("  P4 计时 %.1fs / %d 单元" % (dt, len(rows)), flush=True)
    return rows, dict(sec_per_unit=dt / max(len(rows), 1))


def p5_negative_control():
    print("== P5 判据负控 ==", flush=True)
    f = G["frames"][0]
    seg, _, lead = seg_of(f)
    det = M3.dep2_blind_detect(seg, f["pre"])
    assert det is not None, "native 检出失败"
    pre, psym = f["pre"], f["psym"]
    det = dict(det)
    det["nu_rot"] = M3.nu_rot_of(det["nu0h"], det["dhat"], det["c_e"],
                                 pre, psym)
    pad = (pre + 5) * NF
    seg_p = np.concatenate((np.zeros(pad), seg[det["pay0"]:]))
    n_rot = np.arange(len(seg_p))
    seg_p = seg_p * np.exp(-2j * np.pi * det["nu_rot"] * n_rot / NF)
    rows_a, rows_b = M3._demod_two_columns(seg_p, pre, psym, KT, DD)
    ok_a, _, _ = M3.demap_judge(rows_a, np.zeros(psym, int), f)
    ok_b, _, _ = M3.demap_judge(rows_b, np.zeros(psym, int), f)
    # 负控 1：单符号损坏
    bad = rows_a.copy()
    bad[3] = np.roll(bad[3], 137)
    ok_bad, _, _ = M3.demap_judge(bad, np.zeros(psym, int), f)
    # 负控 2：整段随机行
    rng = np.random.default_rng(5)
    ok_rand, _, _ = M3.demap_judge(rng.permutation(rows_a), np.zeros(psym, int),
                                   f)
    print("  A=%s B=%s 单符损坏拒=%s 随机行拒=%s"
          % (ok_a, ok_b, not ok_bad, not ok_rand), flush=True)
    assert ok_a and ok_b, "native 判据失败"
    assert (not ok_bad) and (not ok_rand), "负控失败"
    return dict(a=ok_a, b=ok_b, bad_rejected=not ok_bad,
                rand_rejected=not ok_rand)


# ---------------------------------------------------------------- 单元
def run_one(f, di, level, seed):
    """单帧单档全链（冒烟用；战场版在 m3_battle.py 内并行化复用此逻辑）。"""
    pre, psym = f["pre"], f["psym"]
    lead = pre + 6
    tail = (8 + psym + 2) * NF + 64
    seg0 = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + tail],
                      dtype=np.complex128)
    origin = (lead - (pre + 4.25)) * NF
    inj = D.resample_sfo(seg0, origin, D.eps_of_delta((0.0, 0.02, 0.082)[di]))
    tm_d = D.clean_template_q(inj, lead * NF, pre)
    S, N0 = FR.snr_parts(inj[: lead * NF + 8 * NF])
    if level is not None:
        rng = np.random.default_rng((20261004 * 7919 + (int(level) + 100) * 131
                                     + seed * 101 + G["frames"].index(f) * 7919
                                     + di * 104729) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = inj + ((rng.standard_normal(len(inj))
                      + 1j * rng.standard_normal(len(inj)))
                     * np.sqrt(p_add / 2.0))
    else:
        seg = inj
    out = dict(di=di, level=level, seed=seed, frame=G["frames"].index(f),
               pre=pre, psym=psym)

    # ---- OURS-E2E（dep2 系统检测）----
    try:
        det = M3.dep2_blind_detect(seg, pre)
        out["dep_det"] = det is not None
        if det is not None:
            det = dict(det)
            det["nu_rot"] = M3.nu_rot_of(det["nu0h"], det["dhat"], det["c_e"],
                                        pre, psym)
            r = M3.ours_chain_decode(seg, det, f, KT, DD)
            out.update(oa_per=int(not r["a"]["ok"]), oa_ser=r["a"]["ser"],
                       ob_per=int(not r["b"]["ok"]), ob_ser=r["b"]["ser"],
                       u_per=int(not r["u"]["ok"]), u_ser=r["u"]["ser"],
                       u_col=r["u"]["col"])
            out["anchor_err"] = abs(
                det["nu_rot"] - M3.nu_rot_of(tm_d["nu0"], tm_d["delta"],
                                             (pre - 1) / 2.0, pre, psym))
            out["pay_err"] = abs(det["pay0"] - (lead + 8) * NF)
    except Exception as ex:
        out["dep_err"] = repr(ex)[:150]

    # ---- OURS-E2E-cert（bookkeeping 上界）----
    try:
        sc, _kap, _s2 = C.score_cert_batch(seg[None, :], lead * NF, pre, tm_d)
        out["cert_det"] = bool(sc[0] > M3.CERT_THR)
        if out["cert_det"]:
            det = dict(pay0=(lead + 8) * NF,
                       nu_rot=M3.nu_rot_of(tm_d["nu0"], tm_d["delta"],
                                           (pre - 1) / 2.0, pre, psym))
            r = M3.ours_chain_decode(seg, det, f, KT, DD)
            out.update(ca_per=int(not r["a"]["ok"]), ca_ser=r["a"]["ser"],
                       cb_per=int(not r["b"]["ok"]), cb_ser=r["b"]["ser"],
                       cu_per=int(not r["u"]["ok"]), cu_ser=r["u"]["ser"])
    except Exception as ex:
        out["cert_err"] = repr(ex)[:150]

    # ---- DeRa 全链 ----
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
                out["dera_per"] = int(not ok)
                out["dera_ser"] = int(sum(int(h != g)
                                          for h, g in zip(hard, f["gt"])))
    except Exception as ex:
        out["dera_err"] = str(ex)[:80]
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    G["frames"] = FR.build_frames()
    res = {}
    if mode in ("all", "p1"):
        res["p1_raw"], res["p1_sum"] = p1_gate_bias()
    if mode in ("all", "p2"):
        res["p2"] = p2_delta_sensitivity()
    if mode in ("all", "p5"):
        res["p5"] = p5_negative_control()
    if mode in ("all", "p4"):
        res["p4"], res["p4_time"] = p4_native()
    if mode in ("all", "p3"):
        res["p3"] = p3_noise_mc()
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print("saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""phase_align 机制实验（2026-10-06）：两段相位补齐的余量测量 + 消融臂。

臂（fast 路径，参数干净冻结 = ν̂₀(dep4 锚)/δ̂/τ̂_pre(前导盲估，逐候选
冻结表)，实验B 哲学；与 fast_tmpl_mech 同数据同噪声派生 → e_bl/A 列
可同表对拍）：
  A  port 常数 φ̂₀ 相干合并（消融锚点，与 port 逐位一致）
  P  斜坡合并（强符号盲拟合 + 护栏退回 A）
  N  非相干合并 |F|²+|T|²（= LoRaTrimmer 度量，量化两段相干叠加价值）

每候选**一次投影**三臂共用，臂间独立 CRC 早退状态机（各臂独立 PER）。
  §clean  28 帧 × dc∈{0.01,0.02}：P−A 增益@tone-bin（mode 修正）、
          盲斜率 vs tone-bin 真值斜率、coh−noncoh 逐符号 SNR 差
  §PER    levels {native,−18..−24} × seeds 0-4 × 28 帧
→ 04_results/phase_align_check.jsonl（断点 part+frame+level+seed）
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
                   "..", "04_results", "phase_align_check.jsonl")
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
                        hs_abs=lead * NF, m0=(lead + 8) * NF,
                        nu0=float(det["nu0h"]), dhat=float(det["dhat"]),
                        order=order, tau_map=tau_map, S=S, n0=n0)
    if len(G["par"]) > 8:
        for k in list(G["par"])[:len(G["par"]) - 8]:
            del G["par"][k]
    return G["par"][fi]


def _arm_cols(seg, f, p):
    """三臂一次投影共用，独立 CRC 状态机。"""
    pre, psym = f["pre"], f["psym"]
    done = {t: None for t in ("a", "p", "n")}
    for dc in p["order"]:
        base = FT._repaired_base(seg, p["m0"], dc, p["tau_map"][dc],
                                 p["nu0"], p["origin"], pre)
        ph = PA.demod_phased(base, pre + 5, psym, DD)
        rows = {"a": ph["rows_const"], "p": ph["rows_p"],
                "n": ph["noncoh"]}
        for tag in ("a", "p", "n"):
            if done[tag] is None:
                ok, ser, d0 = M3.demap_judge(rows[tag],
                                             np.zeros(psym, dtype=int), f)
                if ok:
                    done[tag] = dict(ok=True, ser=ser, dc=dc)
                    if tag == "p" and ph["slope"] is not None:
                        done[tag]["slope"] = float(ph["slope"])
        if all(v is not None for v in done.values()):
            break
    return {t: (v if v is not None else dict(ok=False, ser=-1, dc=None))
            for t, v in done.items()}


def run_clean(fi):
    f = G["frames"][fi]
    p = _params(fi)
    pre, psym = f["pre"], f["psym"]
    gt1 = (np.asarray(f["gt"]) + 1) % N
    rec = dict(frame=fi, part="clean", delta=DELTA, pre=pre, dhat=p["dhat"])
    for dc in (0.02, 0.01):
        base = FT._repaired_base(p["seg_inj"], p["m0"], dc,
                                 p["tau_map"][dc], p["nu0"], p["origin"],
                                 pre)
        ph = PA.demod_phased(base, pre + 5, psym, DD)
        fm, tm, nc = PA.project_payload(base, pre + 5, psym, DD)
        tmp = np.argmax(nc, axis=1)
        mo = int(np.bincount((tmp - gt1) % N).argmax())
        tb = (gt1 + mo) % N
        ia = np.arange(psym)
        z_t = tm[ia, tb] * np.conj(fm[ia, tb])
        sl_true = float(np.polyfit(ia - ia.mean(),
                                   np.unwrap(np.angle(z_t)), 1)[0])
        sc = ph["rows_const"][ia, tb] / (p["n0"] * NF)
        sp = ph["rows_p"][ia, tb] / (p["n0"] * NF)
        sn = nc[ia, tb] / (p["n0"] * NF)
        lsc = 10 * np.log10(np.maximum(sc, 1e-30))
        g = 10 * np.log10(np.maximum(sp, 1e-30)) - lsc
        coh_nc = lsc - 10 * np.log10(np.maximum(sn, 1e-30))
        rec["dc%.2f" % dc] = dict(
            mode_off=mo, tau=p["tau_map"][dc],
            slope_blind=(None if ph["slope"] is None
                         else float(ph["slope"])),
            slope_true=sl_true, resid=float(ph["resid"]),
            gain_med=float(np.median(g)),
            gain_tail8=float(np.median(g[-8:])),
            cohnc_med=float(np.median(coh_nc)),
            cohnc_p10=float(np.percentile(coh_nc, 10)))
    rec["native"] = _arm_cols(p["seg_inj"], f, p)
    return rec


def run_per(u):
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
    return dict(frame=fi, part="per", level=lv, seed=sd, delta=DELTA,
                **_arm_cols(seg, f, p))


def init_worker():
    global DD
    G["frames"] = FR.build_frames()
    G["par"] = {}
    DD = DeRaDemodulator(10, 4)


def main():
    t0 = time.time()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["part"], r["frame"], r.get("level"),
                          r.get("seed")))
            except Exception:
                pass
        print("断点恢复：%d 单元" % len(done), flush=True)
    units = ([("c", fi) for fi in range(28)]
             + [("p", (fi, lv, sd)) for fi in range(28) for lv in LEVELS
                for sd in SEEDS])
    ctx = mp.get_context("spawn")
    n = 0
    with open(OUT, "a", encoding="utf-8") as fh:
        with ctx.Pool(processes=7, initializer=init_worker) as pool:
            for kind, u in units:
                if kind == "c":
                    key = ("clean", u, None, None)
                    if key in done:
                        continue
                    r = pool.apply(run_clean, (u,))
                else:
                    key = ("per", u[0], u[1], u[2])
                    if key in done:
                        continue
                    r = pool.apply(run_per, (u,))
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                n += 1
                if n % 70 == 0:
                    print("  %d (%.1fmin)" % (n, (time.time() - t0) / 60),
                          flush=True)
    print("完成 %d 单元，%.1fmin" % (n, (time.time() - t0) / 60),
          flush=True)

    # ---- 汇总 ----
    import collections
    agg = collections.defaultdict(
        lambda: collections.defaultdict(lambda: dict(n=0, per=0)))
    cl = []
    for line in open(OUT, encoding="utf-8"):
        r = json.loads(line)
        if r.get("part") == "per":
            for arm in ("a", "p", "n"):
                agg[r["level"]][arm]["n"] += 1
                agg[r["level"]][arm]["per"] += int(not r[arm]["ok"])
        else:
            cl.append(r)
    print("\nδ=0.02 PER（fast 冻结参数，n/档见括号；A=e_bl A 列对拍）")
    print("SNR     | A(coh φ̂0) | P(ramp)   | N(noncoh)")
    for lv in sorted(agg, key=lambda x: (x is not None, x)):
        nn = agg[lv]["a"]["n"]
        row = ["PER %.3f" % (agg[lv][a]["per"] / max(nn, 1))
               for a in ("a", "p", "n")]
        print("%-7s | %s | %s | %s  (n=%d)" % (lv, *row, nn))
    for dc in ("dc0.02", "dc0.01"):
        gm = [r[dc]["gain_med"] for r in cl if dc in r]
        sb = [r[dc]["slope_blind"]
              for r in cl if dc in r]
        st = [r[dc]["slope_true"] for r in cl if dc in r]
        cn = [r[dc]["cohnc_med"] for r in cl if dc in r]
        err = [abs(b - t) * 1000 for b, t in zip(sb, st)
               if b is not None]
        nfit = sum(1 for b in sb if b is not None)
        print("\n%s: P−A gain_med med %+.3f dB | 盲斜率差 med %.2f mrad"
              "（≤3mrad %d/%d，fit_ok %d/28） | coh−noncoh med %.2f dB" % (
                  dc, np.median(gm), np.median(err),
                  sum(1 for e in err if e <= 3), nfit, nfit,
                  np.median(cn)))


if __name__ == "__main__":
    main()

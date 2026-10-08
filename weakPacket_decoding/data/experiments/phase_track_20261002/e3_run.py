# -*- coding: utf-8 -*-
"""E3 主跑：臂 D（turbo 相位）+ D-adaptive（σ_w 自估，诊断）+ DNP（D-noPhase）
+ E8/E16（块部分相干）+ A/B 复算（配对验证）。

数据/口径/种子 = E2 逐字节一致（e1_common + e2_arm 注噪与测量路径）。
断点续跑 e3_units.jsonl；逐 capture 并行。用法：
    python e3_run.py            # 全矩阵 2828 单元
    python e3_run.py 5 -24      # 只跑 5 个种子（pilot）
"""
import sys
import os
import json
import time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
import e2_arm as E2
import e3_common as D3

ROOT = C.ROOT
IN = os.path.join(ROOT, "e1_native.jsonl")
OUTU = os.path.join(ROOT, "e3_units.jsonl")
SEEDS = list(range(C.N_SEEDS))

_W = {}


def run_unit(u):
    gid, snr, seed = u["gid"], u["snr"], u["seed"]
    fr, seg0, ds = _W["frames"][gid], _W["segs"][gid], _W["ds"][gid]
    seg = (C.inject_noise(seg0, snr, seed, gid, fr["S"], fr["N0"])
           if snr is not None else seg0)
    mt = E2.unit_measure(seg, fr, ds)
    if mt is None:
        return dict(u, skip="extract_fail")
    gt = np.asarray(fr["gt"])
    rowsA, kA = E2.arm_A_twin(mt)
    rowsB, dB = E2.arm_B_genie(mt, gt)
    out = dict(gid=gid, snr=snr, seed=seed, sym_tot=mt["psym"],
               A=dict(sym_err=int((np.argmax(rowsA, 1) != gt).sum()),
                      crc_fail=int(not E2.judge(rowsA, fr["gt_hdr"], fr["plen"], fr["cr"]))),
               B=dict(sym_err=int((np.argmax(rowsB, 1) != gt).sum()),
                      crc_fail=int(not E2.judge(rowsB, fr["gt_hdr"], fr["plen"], fr["cr"])),
                      tau_hat=dB["tau_hat"], kap0=dB["kap_hat"], enu=dB["enu"]))

    def eval_rows(rows, tag):
        return dict(sym_err=int((np.argmax(rows, 1) != gt).sum()),
                    crc_fail=int(not E2.judge(rows, fr["gt_hdr"], fr["plen"], fr["cr"])))

    rowsD, diagD = D3.arm_D_turbo(mt, fr, dB, rowsA)          # 预注册 D（σ_w=0.08）
    out["D"] = eval_rows(rowsD, "D")
    out["D_diag"] = diagD
    # D-adaptive：同管线 σ_w 自估（诊断臂，只跑相干轮）
    q = D3.softmax_rows(rowsA)
    rd_a = []
    rows_da = rowsA
    for it in range(3):
        if it == 0:
            rd_a.append(dict(it=0, SER=float(np.mean(np.argmax(rowsA, 1) != gt))))
            if out["A"]["crc_fail"] == 0:
                break
            continue
        kap0, enu, kap_line, jstar = D3.soft_kappa_line(mt, q)
        cands = [D3.track_phase(mt, q, kap_line, jstar, tau_branches=(t,),
                                adaptive=True) for t in D3.TAU_BRANCH]
        cands.sort(key=lambda c: c["loo_res"])
        pick, rows_da = None, None
        for c in cands:
            rws = D3.coherent_rows(mt, c["phi"], c["var"], c["tau"], kap_line, jstar)
            if E2.judge(rws, fr["gt_hdr"], fr["plen"], fr["cr"]):
                pick, rows_da = c, rws
                break
        if pick is None:
            pick = cands[0]
            rows_da = D3.coherent_rows(mt, pick["phi"], pick["var"], pick["tau"],
                                       kap_line, jstar)
        rd_a.append(dict(it=it, SER=float(np.mean(np.argmax(rows_da, 1) != gt)),
                         sig_w=pick["sig_w"], tau=pick["tau"]))
        if E2.judge(rows_da, fr["gt_hdr"], fr["plen"], fr["cr"]) or it == 2:
            break
        q = D3.softmax_rows(rows_da)
    out["DA"] = eval_rows(rows_da, "DA")
    out["DA_diag"] = rd_a
    rowsN, roundsN = D3.arm_D_nophase(mt, fr, rowsA)
    out["DNP"] = eval_rows(rowsN, "DNP")
    out["DNP_rounds"] = roundsN
    r8, dE8 = D3.arm_E(mt, fr, rowsA, 8)
    out["E8"] = eval_rows(r8, "E8")
    out["E8_diag"] = dE8
    r16, dE16 = D3.arm_E(mt, fr, rowsA, 16)
    out["E16"] = eval_rows(r16, "E16")
    out["E16_diag"] = dE16
    return out


def init_worker(gids):
    E2.build_static()
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    keep = [fr for fr in frames if fr["gid"] in gids]
    caps = sorted(set(fr["cap"] for fr in keep))
    _W["frames"] = {fr["gid"]: fr for fr in keep}
    _W["segs"] = {}
    _W["ds"] = {}
    for cap in caps:
        src = next(s for s in C.SF10_SOURCES + C.SF11_SOURCES if s[0] == cap)
        ds = C.DS(src[3])
        iq = np.memmap(src[1], dtype=np.complex64, mode="r")
        for fr in (f for f in keep if f["cap"] == cap):
            r = dict(header_start_sample=fr["hs"],
                     source_grlora_cfo_int=fr["cfo_int"],
                     source_grlora_cfo_frac=str(fr["cfo_frac"]),
                     source_grlora_payload_sto_frac=str(fr["sto_frac"]))
            seg, _i0, _bo = C.align_seg(iq, r, ds, fr["P"], fr["psym"])
            _W["segs"][fr["gid"]] = seg
            _W["ds"][fr["gid"]] = ds
        del iq


def main():
    n_seed_cap = None
    snr_cap = None
    if len(sys.argv) >= 3:
        n_seed_cap = int(sys.argv[1])
        snr_cap = int(sys.argv[2])
    t0 = time.time()
    E2.build_static()
    E2.negative_control()
    frames = [json.loads(l) for l in open(IN, encoding="utf-8")]
    frames = [f for f in frames if f["sf"] == 10]
    by_cap = {}
    for fr in frames:
        by_cap.setdefault(fr["cap"], []).append(fr["gid"])
    units = []
    for fr in frames:
        units.append(dict(gid=fr["gid"], snr=None, seed=None))
        for snr in ([snr_cap] if snr_cap else C.SNR_LEVELS):
            for sd in (SEEDS[:n_seed_cap] if n_seed_cap else SEEDS):
                units.append(dict(gid=fr["gid"], snr=snr, seed=sd))
    done = set()
    if os.path.exists(OUTU):
        for l in open(OUTU, encoding="utf-8"):
            rec = json.loads(l)
            done.add((rec["gid"], rec["snr"], rec["seed"]))
        print("断点续跑: 已完成 %d 单元" % len(done), flush=True)
    todo = [u for u in units if (u["gid"], u["snr"], u["seed"]) not in done]
    print("总单元 %d, 待跑 %d" % (len(units), len(todo)), flush=True)
    n = 0
    fout = open(OUTU, "a", encoding="utf-8")
    for cap, gids in sorted(by_cap.items()):
        cu = [u for u in todo if u["gid"] in set(gids)]
        if not cu:
            continue
        with Pool(processes=6, initializer=init_worker, initargs=(set(gids),)) as pool:
            for res in pool.imap_unordered(run_unit, cu, chunksize=2):
                fout.write(json.dumps(res) + "\n")
                n += 1
                if n % 50 == 0:
                    fout.flush()
                    print("  %d/%d (%.0fs)" % (n, len(todo), time.time() - t0),
                          flush=True)
    fout.close()
    print("完成 %d 单元 (%.0fs)" % (n, time.time() - t0))


if __name__ == "__main__":
    main()

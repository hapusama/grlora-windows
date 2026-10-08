# -*- coding: utf-8 -*-
"""E3 配对验证：抽 3 个单元复算臂 A/B，与 e2_units.jsonl 逐位一致才放行；
顺带计时单个单元的全臂（A/B/D/DNP/E8/E16）成本。"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
import e2_arm as E2
import e3_common as D3

ROOT = C.ROOT
PICK = [(5, -22, 3), (11, -24, 7), (16, None, None)]     # 深端两档 + native


def load_frames():
    frames = [json.loads(l) for l in open(os.path.join(ROOT, "e1_native.jsonl"),
                                          encoding="utf-8")]
    return {fr["gid"]: fr for fr in frames if fr["sf"] == 10}


def prep(gid):
    fr = load_frames()[gid]
    src = next(s for s in C.SF10_SOURCES + C.SF11_SOURCES if s[0] == fr["cap"])
    ds = C.DS(src[3])
    iq = np.memmap(src[1], dtype=np.complex64, mode="r")
    r = dict(header_start_sample=fr["hs"],
             source_grlora_cfo_int=fr["cfo_int"],
             source_grlora_cfo_frac=str(fr["cfo_frac"]),
             source_grlora_payload_sto_frac=str(fr["sto_frac"]))
    seg, _i0, _bo = C.align_seg(iq, r, ds, fr["P"], fr["psym"])
    return fr, seg, ds


def main():
    E2.build_static()
    e2 = {}
    for l in open(os.path.join(ROOT, "e2_units.jsonl"), encoding="utf-8"):
        rec = json.loads(l)
        e2[(rec["gid"], rec["snr"], rec["seed"])] = rec
    ok_all = True
    for gid, snr, seed in PICK:
        fr, seg0, ds = prep(gid)
        seg = (C.inject_noise(seg0, snr, seed, gid, fr["S"], fr["N0"])
               if snr is not None else seg0)
        t0 = time.time()
        mt = E2.unit_measure(seg, fr, ds)
        t1 = time.time()
        rowsA, kA = E2.arm_A_twin(mt)
        gt = np.array(fr["gt"])
        serA = int((np.argmax(rowsA, 1) != gt).sum())
        crcA = int(not E2.judge(rowsA, fr["gt_hdr"], fr["plen"], fr["cr"]))
        rowsB, dB = E2.arm_B_genie(mt, gt)
        serB = int((np.argmax(rowsB, 1) != gt).sum())
        crcB = int(not E2.judge(rowsB, fr["gt_hdr"], fr["plen"], fr["cr"]))
        t2 = time.time()
        ref = e2[(gid, snr, seed)]
        ok = (serA == ref["A"]["sym_err"] and crcA == ref["A"]["crc_fail"]
              and serB == ref["B"]["sym_err"] and crcB == ref["B"]["crc_fail"])
        ok_all &= ok
        print("gid=%d snr=%s seed=%s: A %d/%d B %d/%d vs E2 %d/%d %d/%d -> %s"
              % (gid, snr, seed, serA, crcA, serB, crcB,
                 ref["A"]["sym_err"], ref["A"]["crc_fail"],
                 ref["B"]["sym_err"], ref["B"]["crc_fail"],
                 "一致" if ok else "!!不一致!!"), flush=True)
        # 全臂计时
        t3 = time.time()
        rowsD, diagD = D3.arm_D_turbo(mt, fr, dB, rowsA)
        t4 = time.time()
        rowsN, roundsN = D3.arm_D_nophase(mt, fr, rowsA)
        r8, dE8 = D3.arm_E(mt, fr, rowsA, 8)
        r16, dE16 = D3.arm_E(mt, fr, rowsA, 16)
        t5 = time.time()
        serD = int((np.argmax(rowsD, 1) != gt).sum())
        crcD = int(not E2.judge(rowsD, fr["gt_hdr"], fr["plen"], fr["cr"]))
        print("  D: ser=%d crc=%d rounds=%s" % (serD, crcD,
              [(r_["it"], round(r_["SER"], 3), r_["crc_ok"],
                None if r_["phi_rmse"] is None else round(r_["phi_rmse"], 3))
               for r_ in diagD["rounds"]]))
        print("  DNP: %s | E8 ser=%.3f crc=%s | E16 ser=%.3f crc=%s"
              % ([(r_["it"], round(r_["SER"], 3), r_["crc_ok"]) for r_ in roundsN],
                 dE8["SER"], dE8["crc_ok"], dE16["SER"], dE16["crc_ok"]))
        print("  计时: measure %.2fs A %.2fs B %.2fs D %.2fs DNP+E %.2fs"
              % (t1 - t0, t2 - t1, t2 - t1, t4 - t3, t5 - t4), flush=True)
    print("配对验证:", "全部一致，放行" if ok_all else "存在不一致，禁止开跑")
    return ok_all


if __name__ == "__main__":
    main()

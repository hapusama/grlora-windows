# -*- coding: utf-8 -*-
"""E3 调试：native 单元上解剖软 κ-line 拟合与相位跟踪（对照 genie）。"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
import e2_arm as E2
import e3_common as D3


def main():
    E2.build_static()
    gid = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    snr = (int(sys.argv[2]) if len(sys.argv) > 2 else -24)
    if snr <= -900:
        snr = None
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    frames = {fr["gid"]: fr for fr in
              (json.loads(l) for l in open(os.path.join(C.ROOT, "e1_native.jsonl"),
                                           encoding="utf-8")) if fr["sf"] == 10}
    fr = frames[gid]
    src = next(s for s in C.SF10_SOURCES if s[0] == fr["cap"])
    ds = C.DS(10)
    iq = np.memmap(src[1], dtype=np.complex64, mode="r")
    r = dict(header_start_sample=fr["hs"],
             source_grlora_cfo_int=fr["cfo_int"],
             source_grlora_cfo_frac=str(fr["cfo_frac"]),
             source_grlora_payload_sto_frac=str(fr["sto_frac"]))
    seg0, _i0, _bo = C.align_seg(iq, r, ds, fr["P"], fr["psym"])
    seg = (C.inject_noise(seg0, snr, seed, gid, fr["S"], fr["N0"])
           if snr is not None else seg0)
    mt = E2.unit_measure(seg, fr, ds)
    gt = np.array(fr["gt"])
    rowsA, _ = E2.arm_A_twin(mt)
    rowsB, dB = E2.arm_B_genie(mt, gt)
    print("genie: kap0=%.4f enu=%.4f tau=%.3f conc=%.3f serB=%d"
          % (dB["kap_hat"], dB["enu"], dB["tau_hat"], dB["conc"],
             int((np.argmax(rowsB, 1) != gt).sum())))
    q = D3.softmax_rows(rowsA)
    print("q top1 hit: %d/%d  qmax med=%.3f" % (
        int((np.argmax(q, 1) == gt).sum()), len(gt), float(np.median(q.max(1)))))
    kap0, enu, kap_line, jstar = D3.soft_kappa_line(mt, q)
    print("soft fit: kap0=%.4f enu=%.4f  (genie %.4f %.4f)"
          % (kap0, enu, dB["kap_hat"], dB["enu"]))
    # τ 三支解剖
    for t in D3.TAU_BRANCH:
        y, rr = D3.soft_meas(mt, q, kap_line, t, jstar)
        tr = D3.track_phase(mt, q, kap_line, jstar)
        # 重算该支路（track_phase 只回最优）——手动复算 SSE
        amp = mt["amp"][mt["P"] + 2:]
        lam = np.minimum((2.0 * amp * np.maximum(rr, 1e-3)) ** 2, 400.0)
        d = D3.wrap(np.diff(y))
        med = float(np.median(d))
        gate = np.zeros(len(y), bool)
        for i in np.where(np.abs(D3.wrap(d - med)) > 1.2)[0]:
            gate[i if lam[i] <= lam[i + 1] else i + 1] = True
        lam_g = lam * np.where(gate, 0.01, 1.0)
        xs, xv = D3._rts(y, lam_g, D3.SIGMA_W)
        res = D3.wrap(y - xs)
        sse = float(np.sum(lam_g * res ** 2))
        e = D3.wrap(xs - np.asarray(dB["phi_hat"]))
        print("  tau=%+.2f: r_med=%.3f gate=%.2f sse=%.1f  phiRMSE_vs_genie=%.3f"
              % (t, float(np.median(rr)), float(np.mean(gate)), sse,
                 float(np.sqrt(np.mean(e ** 2)))))
    tr = D3.track_phase(mt, q, kap_line, jstar)
    print("picked tau=%.3f (genie %.3f) gate=%.2f" % (
        tr["tau"], dB["tau_hat"], tr["gate_rate"]))
    rows = D3.coherent_rows(mt, tr["phi"], tr["var"], tr["tau"], kap_line, jstar)
    print("coherent ser=%d  |B ser=%d" % (
        int((np.argmax(rows, 1) != gt).sum()),
        int((np.argmax(rowsB, 1) != gt).sum())))
    rowsN = D3.noncoh_rows(mt, jstar)
    bad = np.where(np.argmax(rowsN, 1) != gt)[0]
    print("DNP ser=%d bad_idx=%s" % (len(bad), bad.tolist()))
    # genie 的 jstar 对照
    jB = (np.round(((dB["kap_hat"] + dB["enu"] * np.arange(mt["P"] + 2, mt["P"] + 2 + mt["psym"])) - D3.KAP_GRID[0]) % 1.0 * 16.0).astype(int)) % D3.NK
    print("jstar diff count:", int((jstar != jB).sum()), jstar[:8], jB[:8])


if __name__ == "__main__":
    main()

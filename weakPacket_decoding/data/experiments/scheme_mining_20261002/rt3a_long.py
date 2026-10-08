# -*- coding: utf-8 -*-
"""R3a 收尾（用户直接跑）：A′ 复活条件的最后一个角——长 payload。

rt3a_full.log 已测（SF8/N=32）：CR4/8 强门 B+ 门深 −15.05dB，仍比放松门限
检测 frontier（list −19.53 / base −18.21）浅 4.5dB → 复活失败。
但门深随帧能量 10log10(N) 走：N=128（62B payload）预期 B+ ≈ −21dB，
可能翻到 frontier 之下。本脚本只测这一个决定性配置 + H0 误过率。

配置 cr48L：SF8 M=256，CR4/8，NB=16 块 × 8 符号 = N=128，plen=62B+2B CRC。
臂：B（margF 唯一判决）、B+（Chase 列表 L=28）。A 弱门不再需要。
frontier 沿用 rt3a_revival_results.json（同检测器/同 W=1000/同 top-K=20）。

运行：py -3.12 rt3a_long.py   （约 5-8 分钟）
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\scheme_mining_20261002")

import rt3a_revival as R                                    # noqa: E402
from weak_decoder.decoding.payload_codec import (           # noqa: E402
    WHITENING_SEQ, decode_payload_symbols)

# 长 payload 需要完整白化表（rt3a 只内联了 24B）
R.WHIT = np.array(WHITENING_SEQ, dtype=np.int64)
CFGL = dict(cr=4, BLK=8, NB=16, N=128, plen=62)
R.CFGS["cr48L"] = CFGL

CH = 200                    # 帧分块（控内存：y complex128 200x136x256 ≈ 110MB）


def validate_long():
    rng = np.random.default_rng(2026)
    nib, payload = R.gen_payload(rng, 16, CFGL)
    bins = R.nib_stream_to_bins(nib, CFGL)
    vals = (bins - 1) & 255
    v2 = True
    for k in range(16):
        r = decode_payload_symbols([int(v) for v in vals[k]], sf=8, cr=4,
                                   ldro=False, payload_len=62, has_crc=True,
                                   crc_mode="grlora")
        v2 &= r.crc_valid and bytes(r.payload_bytes) == bytes(
            int(x) for x in payload[k])
    nibd = R.hard_hamming_nibbles(R.rx_vals_to_cwbits(R.FOLD1[vals], CFGL), CFGL)
    okm, plm = R.crc_check(nibd, CFGL)
    v3 = bool(okm.all()) and bool((plm == payload).all())
    print("validate cr48L: V2 ref-RX=%s  V3 my-RX=%s" % (v2, v3), flush=True)
    return v2 and v3


def gate_pd(rng, gamma_db, n_total, lam0, list_mode):
    cnt, done = 0, 0
    sig = 10.0 ** (-gamma_db / 20.0)
    while done < n_total:
        c = min(CH, n_total - done)
        y, _, _ = R.gen_frames(rng, c, float(gamma_db), CFGL)
        cnt += int(R.gate_B(y, sig, CFGL, lam0, list_mode).sum())
        done += c
    return cnt / done


def q0_gate(rng, n_total, lam0, list_mode):
    cnt, done = 0, 0
    while done < n_total:
        c = min(CH, n_total - done)
        w = ((rng.standard_normal((c, 8 + CFGL["N"], R.M))
              + 1j * rng.standard_normal((c, 8 + CFGL["N"], R.M)))
             / np.sqrt(2))
        cnt += int(R.gate_B(w, 1.0, CFGL, lam0, list_mode).sum())
        done += c
    return cnt, done


def main():
    t0 = time.time()
    assert validate_long(), "cr48L codec 对齐失败"
    rep = {"validated": True, "seed": 4711, "cfg": CFGL}
    rng = np.random.default_rng(4711)

    print("\n== cr48L 门深（P(CRC)=0.9，per-sample dB，n=800/gamma）==")
    gammas = [-23.0, -22.0, -21.0, -20.0, -19.0]
    curves = {"B": [], "Blist": []}
    for g in gammas:
        lam0 = R.M * 10 ** (g / 10.0)
        pb = gate_pd(rng, g, 800, lam0, False)
        pl = gate_pd(rng, g, 800, lam0, True)
        curves["B"].append(pb)
        curves["Blist"].append(pl)
        print(" g=%+.2f  B=%.3f  B+=%.3f" % (g, pb, pl), flush=True)
    rep["gate_depth"] = {"gammas": gammas, "curves": curves,
                         "pd09": {k: {"0.9": R.pd09(curves[k], gammas),
                                      "0.5": R.pd09(curves[k], gammas, 0.5)}
                                  for k in curves}}
    print(" Pd.9 门深: B=%s  B+=%s | Pd.5: B+=%s"
          % (rep["gate_depth"]["pd09"]["B"]["0.9"],
             rep["gate_depth"]["pd09"]["Blist"]["0.9"],
             rep["gate_depth"]["pd09"]["Blist"]["0.5"]), flush=True)

    print("\n== H0 q_gate（纯噪声，lam0 按 −21dB 假设，n=3000）==")
    lam0 = R.M * 10 ** (-21.0 / 10.0)
    q = {}
    for tag, lm in (("B", False), ("Blist", True)):
        cnt, n = q0_gate(np.random.default_rng(99 + int(lam0)), 3000, lam0, lm)
        q[tag] = cnt / n
        print(" q_%s = %d/%d = %.2e" % (tag, cnt, n, q[tag]), flush=True)
    rep["q_h0"] = {"assume_gamma_db": -21.0, "n": 3000, "q": q}

    # frontier 敏感性：若 q_B+ 偏离 rt3a 的 1.33e-3，重算 f_cell 与 frontier
    import rt_a_tail as A
    q1, q2 = 65.62, 84.84
    b_exp = (q2 - q1) / np.log(10.0)

    def thr_exp(far):
        return q2 + b_exp * np.log(1e-4 / far)

    def frontier_thr(thr, lo, hi, iters=7, n=1500, seed=555):
        for it in range(iters):
            mid = 0.5 * (lo + hi)
            pd = float(np.mean(A.t1_h1(mid, n, seed + 7919 * it) > thr))
            if pd >= 0.9:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    qb = max(q["Blist"], 28 * 2.0 ** -16)
    f_cell = 1e-6 / (1000.0 * qb)
    fl = frontier_thr(thr_exp(f_cell), -24, -17)
    print("\n q_B+=%.2e -> f_cell=%.2e -> 检测 frontier=%+.3f dB（rt3a N=32 版为 "
          "-19.53）" % (qb, f_cell, fl), flush=True)
    rep["frontier_recheck"] = {"q_used": qb, "f_cell": f_cell, "frontier": fl}

    gate = rep["gate_depth"]["pd09"]["Blist"]["0.9"]
    base = -18.21
    sysf = max(gate, fl) if (gate is not None) else None   # 系统被较浅者钳制
    rep["verdict"] = {"gate_Bplus": gate, "frontier_list": fl, "base_1e9": base,
                      "system_frontier": sysf,
                      "net_db": (base - sysf) if sysf is not None else None}
    print("\n== 裁决：门深 B+=%s  检测 frontier=%+.2f  基线=%+.2f" %
          (gate, fl, base))
    print("   系统 frontier（被较浅者钳）=%s  净 dB=%s"
          % (sysf, rep["verdict"]["net_db"]))

    rep["elapsed_s"] = time.time() - t0
    with open("rt3a_long_results.json", "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("saved rt3a_long_results.json (%.1f s)" % rep["elapsed_s"])


if __name__ == "__main__":
    main()

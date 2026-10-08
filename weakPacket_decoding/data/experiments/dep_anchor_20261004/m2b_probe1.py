# -*- coding: utf-8 -*-
"""M2b 探针 1：确认场相位残差结构（干净域）——dep3b θ 模型的可行性判定。

对 28 帧 × δ∈{0,0.082}：
  z_j = 确认行 DTFT @ 真位置（真锚+真 o+真 δ，GT 锚定——只用于看结构）；
  全场最优线性斜率移除（等价 κ-FFT）后，残相 r_j 的行结构；
  关键问题：r_sync−r_pre、r_sfd−r_pre 是否跨帧可预测（对 ν0_frac/o_sfd 回归）。
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_core as C
import d1_battle as A
import d2_core as D
import m2_core as M

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t0 = time.time()
    frames = A.build_frames()
    rows_out = []
    for fi, f in enumerate(frames):
        pre = f["pre"]
        for dl in (0.0, 0.082):
            eps = D.eps_of_delta(dl)
            inj = D.resample_sfo(f["seg"],
                                 f["lead"] - (pre + 4.25) * M.NF,
                                 eps) if dl else f["seg"]
            tmq = D.clean_template_q(inj, f["lead"], pre)
            wins = C.field_windows(f["lead"], pre)
            L = M._confirm_layout(pre)
            rows, b = L["rows"], L["b"]
            idx, c_e = L["idx"], L["c_e"]
            nu0, dl_t = tmq["nu0"], tmq["delta"]
            z = np.zeros(len(rows), dtype=np.complex128)
            for j, rw in enumerate(rows):
                sgn_j = tmq["sgn"][rw]
                pj = nu0 + tmq["o"][rw] + sgn_j * (idx[j] - c_e) * dl_t
                z[j] = C.row_dtft(inj, wins[rw][0], wins[rw][1],
                                  wins[rw][2], pj)[0]
                if tmq["conj_dn"] and sgn_j < 0:
                    z[j] = np.conj(z[j])
            # 全场最优线性斜率（细格 brute force）
            best, bsl = -1.0, 0.0
            for sl in np.linspace(-np.pi, np.pi, 8193):
                v = abs(np.sum(np.sqrt(b) * z
                               * np.exp(-1j * sl * (idx - c_e))))
                if v > best:
                    best, bsl = v, sl
            r = np.angle(z * np.exp(-1j * bsl * (idx - c_e)))
            pre_sel = [j for j, rw in enumerate(rows) if rw < pre]
            syn_sel = [j for j, rw in enumerate(rows)
                       if pre <= rw < pre + 2]
            sfd_sel = [j for j, rw in enumerate(rows) if rw >= pre + 2]
            rows_out.append(dict(
                frame=fi, pre=pre, delta=dl, nu0=nu0,
                nu0_frac=((nu0 + 0.5) % 1.0) - 0.5,
                o_sfd=[float(tmq["o"][rw]) for rw in rows
                       if rw >= pre + 2],
                conj=bool(tmq["conj_dn"]),
                r_pre=[float(np.angle(np.mean(np.exp(1j * r[pre_sel]))))],
                r_syn=[float(x) for x in r[syn_sel]],
                r_sfd=[float(x) for x in r[sfd_sel]],
                best_coh=best ** 2 / float(np.sum(np.sqrt(b)) ** 2)))
    json.dump(rows_out, open(os.path.join(HERE, "m2b_probe1.json"), "w"),
              indent=1)
    # ---- 汇总 ----
    for pre in (8, 16, 32):
        sel = [r for r in rows_out if r["pre"] == pre]
        print("P=%d（%d 帧×2δ）残相结构：" % (pre, len(sel) // 2))
        for tag, key in (("sync", "r_syn"), ("sfd", "r_sfd")):
            ds = np.array([np.angle(np.mean(np.exp(1j * (np.array(r[key])
                          - r["r_pre"][0])))) for r in sel])
            m = np.angle(np.mean(np.exp(1j * ds)))
            sd = np.std(np.angle(np.exp(1j * (ds - m))))
            print("  Δ%s−pre: %+6.1f° ± %5.1f°" % (tag, np.degrees(m),
                                                    np.degrees(sd)))
        # 与 ν0_frac 的回归（ψ(δ) 律可预测性）
        for tag, key in (("sync", "r_syn"), ("sfd", "r_sfd")):
            x = np.array([r["nu0_frac"] for r in sel])
            ds = np.array([np.angle(np.mean(np.exp(1j * (np.array(r[key])
                          - r["r_pre"][0])))) for r in sel])
            # 预测子：π·frac（ψ 律预测残相 = π·(o 相关 frac)）
            pred = np.pi * (x * 0 + 1)  # 占位，下面手动算几个候选
            for name, pp in (("π·νfrac", np.pi * x),
                             ("2π·νfrac", 2 * np.pi * x),
                             ("const", np.zeros_like(x))):
                e = np.angle(np.exp(1j * (ds - pp)))
                print("  Δ%s~%-9s 残 std=%5.1f°" % (
                    tag, name, np.degrees(np.std(e))))
        coh = [r["best_coh"] for r in sel]
        print("  全场最优线性相干度：med=%.3f (no-θ 上限)" % np.median(coh))
    print("%.0fs → m2b_probe1.json" % (time.time() - t0))


if __name__ == "__main__":
    main()

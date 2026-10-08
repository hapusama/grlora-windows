# -*- coding: utf-8 -*-
"""f00/f19 逐帧归因：DeRa 前端哪个残差伤了 Savaux（native，无噪）。"""
import sys
import importlib.util as ilu
import numpy as np

spec = ilu.spec_from_file_location(
    "sr", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments"
          r"\dera_savaux_splice_20261004\splice_runner.py")
sr = ilu.module_from_spec(spec)
spec.loader.exec_module(sr)

sr.init_worker()
G = sr.G
NF, N = sr.NF, sr.N


def row_diffs(rows, gt, psym):
    """argmax bin 与 GT 期望 bin (v+1)%N 的模差分布。"""
    d = []
    for k in range(psym):
        b = int(np.argmax(rows[k]))
        v = gt[k]
        d.append((b - (v + 1)) % N)
    return np.array(d)


def ser_of(rows, gt, psym):
    return float(np.mean([(int(np.argmax(rows[k])) - (gt[k] + 1)) % N != 0
                          for k in range(psym)]))


for fi in (0, 19, 5, 2):
    f = G["frames"][fi]
    pre, lead = f["pre"], f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    print("=" * 70)
    print("f%02d cap=%s pre=%d psym=%d 真值 cfo=%.4f sto_frac=%.4f"
          % (fi, f["cap"], pre, f["psym"], f["cfo"], f["sto_frac"]))
    # PRIOR 行
    n_rel = np.arange(len(seg))
    seg_p = seg * np.exp(-2j * np.pi * f["cfo"] * n_rel / NF)
    seg_p = sr.frac_delay(seg_p, -f["sto_frac"] * sr.OS)
    rows_p = sr.sav_rows(seg_p, lead + 8, f["psym"])
    dp = row_diffs(rows_p, f["gt"], f["psym"])
    print("PRIOR  SER=%.3f  diff分布: %s"
          % (ser_of(rows_p, f["gt"], f["psym"]),
             dict(zip(*np.unique(dp, return_counts=True)))))
    # DeRa 前端
    cands, seg_as, pay0s = sr.dera_sync(seg, pre)
    for ci, (c, seg_a, pay0) in enumerate(zip(cands, seg_as, pay0s)):
        rows = sr.sav_rows(seg_a, pay0, f["psym"])
        dr = row_diffs(rows, f["gt"], f["psym"])
        if ci < 3:
            print("cand%d f_bins=%+.4f(f_err=%+.4f) kappa=%.4f hs_err=%d "
                  "pay0=%d SER=%.3f diff: %s"
                  % (ci, c.f_bins, c.f_bins - f["cfo"], c.kappa,
                     c.hs_est - lead * NF, pay0,
                     ser_of(rows, f["gt"], f["psym"]),
                     dict(zip(*np.unique(dr, return_counts=True)))))
    # 分数 CFO 细扫（在 best 候选对齐段上，delay=0）
    c, seg_a, pay0 = cands[0], seg_as[0], pay0s[0]
    best = None
    for off in np.arange(-0.6, 0.61, 0.1):
        seg_o = seg_a * np.exp(-2j * np.pi * off * n_rel / NF)
        rows_o = sr.sav_rows(seg_o, pay0, f["psym"])
        s = ser_of(rows_o, f["gt"], f["psym"])
        if best is None or s < best[1]:
            best = (float(off), s)
    print("分数CFO细扫 best off=%+.2f SER=%.3f" % best)
    # 分数时延细扫（在 best CFO 段上）
    off = best[0]
    seg_o = seg_a * np.exp(-2j * np.pi * off * n_rel / NF)
    best2 = None
    for d in np.arange(-1.0, 1.01, 0.25):
        rows_o = sr.sav_rows(sr.frac_delay(seg_o, d), pay0, f["psym"])
        s = ser_of(rows_o, f["gt"], f["psym"])
        if best2 is None or s < best2[1]:
            best2 = (float(d), s)
    print("分数时延细扫 best d=%+.2f SER=%.3f" % best2)

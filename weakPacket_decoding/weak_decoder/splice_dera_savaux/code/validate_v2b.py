# -*- coding: utf-8 -*-
"""v2b：SAVK2 = 支路时域 κ 旋转（真分数 bin GLRT）+ Savaux 支路 DFT
（含 wrap 修正）+ Eq.37 相干合并。对照 TREL 的支路 κ 列（单支路、不相干）
与 v1 SAVAUX（整数 bin 合并）。

Q2'：坏帧 bare cand0（无 fd）× SAVK2 是否消灭混跳区。
Q3'：f00 时延剖面下 SAVK2 的 SER 曲线（对照 SAVK/SAVAUX）。
"""
import importlib.util as ilu
import numpy as np

spec = ilu.spec_from_file_location(
    "sr", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_runner.py")
sr = ilu.module_from_spec(spec)
spec.loader.exec_module(sr)
sr.init_worker()
G = sr.G
NF, N, OS, SF_ = sr.NF, sr.N, sr.OS, sr.SF

from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    _paper_branch_spectrum, _oversampled_downchirp)

KAPPA_GRID = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
_DOWN = _oversampled_downchirp(sf=SF_, os_factor=OS, cfo_int=0, cfo_frac=0.0)
_ROTS = {k: np.exp(-2j * np.pi * k * np.arange(N) / N).astype(np.complex64)
         for k in KAPPA_GRID}
_W = {int(k): np.exp(-2j * np.pi * np.arange(OS)[:, None]
                    * np.arange(N)[None, :] / (N * OS))
      for k in range(1)}  # 整数 k 的 Eq.37 权重（κ 已进支路时域）
_W0 = _W[0]


def savk2_rows(seg_a, pay0, psym):
    """SAVK2 行：row[k] = max_κ |Σ_q w_q(k)·Y_q^κ[k]|²。"""
    rows = np.empty((psym, N), dtype=np.float64)
    for i in range(psym):
        st = (pay0 + i) * NF
        symbol = np.asarray(seg_a[st:st + NF], dtype=np.complex64)
        dech = symbol * _DOWN
        branches = [dech[q::OS] for q in range(OS)]
        best = None
        for kap in KAPPA_GRID:
            rot = _ROTS[kap]
            comb = np.zeros(N, dtype=np.complex128)
            for q, br in enumerate(branches):
                Yq = _paper_branch_spectrum(
                    dechirped_branch=br * rot, sf=SF_, os_factor=OS,
                    branch_index=q)
                comb += _W0[q] * Yq.astype(np.complex128)
            p = np.abs(comb) ** 2
            best = p if best is None else np.maximum(best, p)
        rows[i] = best
    return rows


def raw_ser(rows, gt):
    d = [(int(np.argmax(rows[k])) - g) % N for k, g in enumerate(gt)]
    mode = int(np.bincount(d).argmax())
    return float(np.mean(np.array(d) != mode)), mode


print("Q2' 坏帧/净帧 bare cand0（无 fd）：SAVAUX -> SAVK(整数重权) -> SAVK2(时域旋转)")
for fi in (0, 1, 13, 19, 8, 14, 5, 2):
    f = G["frames"][fi]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
    r_old = sr.sav_rows(seg_as[0], pay0s[0], f["psym"])
    r_new = savk2_rows(seg_as[0], pay0s[0], f["psym"])
    s_old, _ = raw_ser(r_old, f["gt"])
    s_new, _ = raw_ser(r_new, f["gt"])
    print("  f%02d sto=%+.3f  SAVAUX SER=%.3f  ->  SAVK2 SER=%.3f"
          % (fi, f["sto_frac"], s_old, s_new))

print("\nQ3' f00 cand0 时延剖面（SAVK2）：")
f = G["frames"][0]
lead = f["pre"] + 6
seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                        f["hs"] + (8 + f["psym"] + 2) * NF],
                 dtype=np.complex128)
cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
for d in np.arange(-1.0, 1.01, 0.25):
    seg_d = sr.frac_delay(seg_as[0], float(d))
    r = savk2_rows(seg_d, pay0s[0], f["psym"])
    s, _ = raw_ser(r, f["gt"])
    print("  d=%+.2f  SAVK2 SER=%.3f" % (d, s))

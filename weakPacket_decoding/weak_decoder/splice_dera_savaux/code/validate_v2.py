# -*- coding: utf-8 -*-
"""v2 前置验证：SAVK（分支相干 κ 格 GLRT）+ 相干度 fd 判据。

三问：
  Q1 κ=0 时 SAVK 是否严格退化为 Eq.37 合并谱（|combined_spectrum|²）；
  Q2 坏帧上 bare cand0（不做 fd）× SAVK 是否直接消灭混跳区——若是，
     κ 格本身就是修复，fd 变冗余 → 最强链简化为 DeRa 前端 + SAVK；
  Q3 相干度判据 c(d)=|Σ_q w_q(ĉ)Y_q[ĉ]|²/Σ_q|Y_q[ĉ]|²（≤OS=4）在
     时延扫描中是否恰在混跳区凹陷（对比平坦的能量判据）。
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
NF, N, OS = sr.NF, sr.N, sr.OS
KAPPA_GRID = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])

from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)

_Q = np.arange(OS)[:, None]
_KK = np.arange(N)[None, :]
_W = {k: np.exp(-2j * np.pi * _Q * (_KK + k) / (N * OS)) for k in KAPPA_GRID}


def savk_rows(seg_a, pay0, psym):
    """分支相干 κ 格 GLRT 行：row[k] = max_κ |Σ_q w_q(k+κ) Y_q[k]|²。"""
    rows = np.empty((psym, N), dtype=np.float64)
    for i in range(psym):
        res = sav_demod(samples=seg_a, start_sample=(pay0 + i) * NF,
                        sf=SF_, os_factor=OS, cfo_int=0)
        Y = np.stack(res.branch_spectra).astype(np.complex128)
        best = None
        for kap in KAPPA_GRID:
            comb = np.abs((_W[kap] * Y).sum(axis=0)) ** 2
            best = comb if best is None else np.maximum(best, comb)
        rows[i] = best
    return rows


def raw_ser(rows, gt):
    d = [(int(np.argmax(rows[k])) - g) % N for k, g in enumerate(gt)]
    mode = int(np.bincount(d).argmax())
    return float(np.mean(np.array(d) != mode)), mode


SF_ = sr.SF

# ---- Q1: κ=0 一致性 ----
f = G["frames"][2]
lead = f["pre"] + 6
seg = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + (8 + f["psym"] + 2) * NF],
                 dtype=np.complex128)
res = sav_demod(samples=seg, start_sample=(lead + 8) * NF, sf=SF_, os_factor=OS)
Y = np.stack(res.branch_spectra).astype(np.complex128)
w0 = np.exp(-2j * np.pi * _Q * _KK / (N * OS))
mine = np.abs((w0 * Y).sum(axis=0)) ** 2
ref = np.abs(res.combined_spectrum.astype(np.complex128)) ** 2
rel = np.max(np.abs(mine - ref)) / np.max(ref)
print("Q1 κ=0 一致性: max rel err = %.3e  %s"
      % (rel, "PASS" if rel < 1e-5 else "FAIL"))

# ---- Q2: 坏帧 bare cand0 × SAVK ----
print("\nQ2 坏帧/净帧 bare cand0（无 fd）：")
for fi in (0, 1, 13, 19, 8, 14, 17, 5, 2):
    f = G["frames"][fi]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
    rows_old = sr.sav_rows(seg_as[0], pay0s[0], f["psym"])
    rows_new = savk_rows(seg_as[0], pay0s[0], f["psym"])
    s_old, m_old = raw_ser(rows_old, f["gt"])
    s_new, m_new = raw_ser(rows_new, f["gt"])
    print("  f%02d sto=%+.3f  SAVAUX(raw,冻结δ) SER=%.3f  ->  SAVK SER=%.3f"
          % (fi, f["sto_frac"], s_old, s_new))

# ---- Q3: 相干度判据 vs 能量判据 的时延剖面（f00 cand0）----
print("\nQ3 f00 cand0 时延剖面（能量 vs 相干度）：")
f = G["frames"][0]
lead = f["pre"] + 6
seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                        f["hs"] + (8 + f["psym"] + 2) * NF],
                 dtype=np.complex128)
cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
for d in np.arange(-1.5, 1.51, 0.25):
    seg_d = sr.frac_delay(seg_as[0], float(d))
    e_sum = 0.0
    c_sum = 0.0
    for k in range(4):
        res = sav_demod(samples=seg_d, start_sample=(lead - 4 + k) * NF,
                        sf=SF_, os_factor=OS, cfo_int=0)
        Y = np.stack(res.branch_spectra).astype(np.complex128)
        comb = (w0 * Y).sum(axis=0)
        e = float(np.max(np.abs(comb) ** 2))
        c_hat = int(np.argmax(np.abs(comb)))
        coh = float(np.abs(comb[c_hat]) ** 2
                    / np.sum(np.abs(Y[:, c_hat]) ** 2))
        e_sum += e
        c_sum += coh
    rows_new = savk_rows(seg_d, pay0s[0], f["psym"])
    s_new, _ = raw_ser(rows_new, f["gt"])
    print("  d=%+.2f  energy=%.3e  coh=%+.3f  SAVK SER=%.3f"
          % (d, e_sum, c_sum / 4.0, s_new))

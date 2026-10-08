# -*- coding: utf-8 -*-
"""消融 (a)：同格去 Viterbi——逐符号独立 argmax（35 态网格，无转移约束）。

隔离"序列级决策"的净贡献（TMC 审稿人点名的缺失消融）。对照：
SAVT2（Viterbi）已有数字；本脚本补 GRID-IND。
档 −22/−24 × seeds {0,1,2} × 28 帧；另附 native 28 帧（零回归门）。
"""
import importlib.util as ilu
import numpy as np

sr_spec = ilu.spec_from_file_location(
    "sr", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_runner.py")
sr = ilu.module_from_spec(sr_spec)
sr_spec.loader.exec_module(sr)
sr.init_worker()
G = sr.G
NF, N, OS, SF_ = sr.NF, sr.N, sr.OS, sr.SF

ABS_D = np.concatenate([c + np.arange(-0.375, 0.376, 0.125)
                        for c in (-1.5, -0.75, 0.0, 0.75, 1.5)])


def grid_ind_rows(seg_a, pay0, psym):
    """逐符号独立 argmax：row[i] = 发射最大的那个态的合并谱行。"""
    n_d = len(ABS_D)
    E = np.empty((psym, n_d))
    rows_all = np.empty((psym, n_d, N))
    for j, d in enumerate(ABS_D):
        seg_d = seg_a if d == 0.0 else sr.frac_delay(seg_a, float(d))
        for i in range(psym):
            res = sr.sav_demod(samples=seg_d, start_sample=(pay0 + i) * NF,
                               sf=SF_, os_factor=OS, cfo_int=0)
            p = np.abs(res.combined_spectrum.astype(np.complex128)) ** 2
            rows_all[i, j] = p
            E[i, j] = np.max(p)
    idx = np.argmax(E, axis=1)
    return rows_all[np.arange(psym), idx]


def eval_unit(level, seed, fi):
    f = G["frames"][fi]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = G["snr"][fi]
    if level is not None:
        rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
    if not cands:
        return None
    rows = grid_ind_rows(seg_as[0], pay0s[0], f["psym"])
    d = [(int(np.argmax(rows[k])) - g) % N for k, g in enumerate(f["gt"])]
    mode = int(np.bincount(d).argmax())
    ser = float(np.mean(np.array(d) != mode))
    _dd, ok = sr.decode_chain(rows, f)
    return ser, bool(ok)


print("== native 28 帧（零回归门：GRID-IND 应全对或接近）==")
bad = 0
for fi in range(28):
    r = eval_unit(None, 0, fi)
    if r is None:
        continue
    bad += r[0] > 0
print("native GRID-IND 坏帧: %d/28" % bad)

print("\n== −22/−24 × 3 种子（对照 SAVT2 v4-b 同单元数字）==")
for lv in (-22, -24):
    crc = 0
    n = 0
    sers = []
    for sd in (0, 1, 2):
        for fi in range(28):
            r = eval_unit(lv, sd, fi)
            if r is None:
                continue
            n += 1
            crc += r[1]
            sers.append(r[0])
    a = np.array(sers)
    print("[%+d] n=%d GRID-IND PER=%.3f SER=%.4f med=%.3f"
          % (lv, n, 1 - crc / n, a.mean(), np.median(a)))
print("\n对照（v4-b 战役同口径）：−22 SAVT2 .006/.048；−24 SAVT2 .044/.262")

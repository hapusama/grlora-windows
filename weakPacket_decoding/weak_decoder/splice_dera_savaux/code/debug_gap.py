# -*- coding: utf-8 -*-
"""A/B 归因：同数据集同先验下，实验B SAVAUX@−26=.421 vs 拼接 PRIOR@−26=.511。

隔离三个候选：
  (1) δ 记账：CRC 仲裁失败回退 δ=0  vs  battle 式逐帧干净冻结 δ；
  (2) SNR 标定窗：拼接[前导+header] vs battle[整帧对齐段]；
  (3) 噪声实现：种子常数/段起点（20260930/hs−(pre+6)NF vs 20260929/hs−8NF）。

输出：四变体 SER + 逐帧错误贡献 + S/N0 标定比。
结论（gap2.log）：A .5112 复现 checkpoint；B 冻结δ .4248 ≈ 实验B .421；
缺口全部来自 6 个 δ_clean=1 帧在深端 CRC 全挂时被回退 δ=0 整帧记错。
"""
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


def align_prior(seg, f):
    n_rel = np.arange(len(seg))
    seg_p = seg * np.exp(-2j * np.pi * f["cfo"] * n_rel / NF)
    return sr.frac_delay(seg_p, -f["sto_frac"] * sr.OS)


def rows_at(seg_p, pay0, psym):
    return sr.sav_rows(seg_p, pay0, psym)


def delta_mode(rows, gt):
    d = [(int(np.argmax(rows[k])) - g) % N for k, g in enumerate(gt)]
    return int(np.bincount(d).argmax())


def err_count(rows, gt, delta):
    return sum((int(np.argmax(rows[k])) - delta) % N != g
               for k, g in enumerate(gt))


frames = G["frames"]
print("== 0) 干净基线：冻结 δ_clean 与 clean 错误 ==")
deltas = {}
clean_err = {}
for fi, f in enumerate(frames):
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    rows = rows_at(align_prior(seg, f), lead + 8, f["psym"])
    d0 = delta_mode(rows, f["gt"])
    deltas[fi] = d0
    clean_err[fi] = err_count(rows, f["gt"], d0)
print("δ_clean 分布:", dict(zip(*np.unique(list(deltas.values()),
                                           return_counts=True))))
print("clean 总错误(冻结δ): %d / %d = %.4f"
      % (sum(clean_err.values()), sum(f["psym"] for f in frames),
         sum(clean_err.values()) / sum(f["psym"] for f in frames)))

print("\n== 1) S/N0 标定窗比（拼接窗 vs battle 窗）==")
sratios = []
for fi, f in enumerate(frames):
    s_mine = np.asarray(f["iq"][f["hs"] - (f["pre"] + 6) * NF:
                               f["hs"] + 8 * NF], dtype=np.complex128)
    s_bat = np.asarray(f["iq"][f["hs"] - 8 * NF:
                               f["hs"] + (8 + f["psym"] + 1) * NF],
                       dtype=np.complex128)
    Sm, Nm = sr.snr_parts(s_mine)
    Sb, Nb = sr.snr_parts(s_bat)
    sratios.append(10 * np.log10((Sm / (Sm + Nm) + 1e-30)
                                 / (Sb / (Sb + Nb) + 1e-30)))
print("SNR窗差 dB: med=%+.3f  mean=%+.3f  min=%+.3f max=%+.3f"
      % (np.median(sratios), np.mean(sratios),
         min(sratios), max(sratios)))

print("\n== 2) 四变体 SER @ −25/−26（3 种子）==")
levels = [-25, -26]
seeds = [0, 1, 2]
variants = ["A_我的段+CRC仲裁", "B_我的段+冻结δ", "C_battle段+冻结δ",
            "D_battle段+种子20260929+冻结δ"]
tot = {v: {lv: 0 for lv in levels} for v in variants}
pf_err = {"A": {lv: {} for lv in levels}, "B": {lv: {} for lv in levels}}
pf_delta_a = {lv: {} for lv in levels}
for fi, f in enumerate(frames):
    lead = f["pre"] + 6
    seg_m = np.asarray(f["iq"][f["hs"] - lead * NF:
                               f["hs"] + (8 + f["psym"] + 2) * NF],
                       dtype=np.complex128)
    Sm, Nm = G["snr"][fi]
    seg_b = np.asarray(f["iq"][f["hs"] - 8 * NF:
                               f["hs"] + (8 + f["psym"] + 1) * NF],
                       dtype=np.complex128)
    Sb, Nb = sr.snr_parts(seg_b)
    for lv in levels:
        for sd in seeds:
            rng_m = np.random.default_rng(
                (20260930 * 7919 + (lv + 100) * 131 + sd * 17 + fi * 7919)
                % (2 ** 31))
            p_m = max(Sm / 10 ** (lv / 10.0) - Nm, 1e-30)
            noisy_m = seg_m + (rng_m.standard_normal(len(seg_m))
                               + 1j * rng_m.standard_normal(len(seg_m))) \
                * np.sqrt(p_m / 2.0)
            rows_m = rows_at(align_prior(noisy_m, f), lead + 8, f["psym"])
            # A: CRC 仲裁（拼接 PRIOR 臂原样）
            _d, ok = sr.decode_chain(rows_m, f)
            e_a = err_count(rows_m, f["gt"], _d)
            tot["A_我的段+CRC仲裁"][lv] += e_a
            pf_err["A"][lv][fi] = pf_err["A"][lv].get(fi, 0) + e_a
            pf_delta_a[lv][fi] = _d
            # B: 冻结 δ_clean
            e_b = err_count(rows_m, f["gt"], deltas[fi])
            tot["B_我的段+冻结δ"][lv] += e_b
            pf_err["B"][lv][fi] = pf_err["B"][lv].get(fi, 0) + e_b
            # C: battle 段+窗+同噪声公式（种子仍 20260930 结构、段起点 hs−8NF）
            rng_c = np.random.default_rng(
                (20260930 * 7919 + (lv + 100) * 131 + sd * 17 + fi * 7919)
                % (2 ** 31))
            p_c = max(Sb / 10 ** (lv / 10.0) - Nb, 1e-30)
            noisy_c = seg_b + (rng_c.standard_normal(len(seg_b))
                               + 1j * rng_c.standard_normal(len(seg_b))) \
                * np.sqrt(p_c / 2.0)
            rows_c = rows_at(align_prior(noisy_c, f), 16, f["psym"])
            tot["C_battle段+冻结δ"][lv] += err_count(rows_c, f["gt"],
                                                     deltas[fi])
            # D: battle 种子常数 20260929
            rng_d = np.random.default_rng(
                (20260929 * 7919 + (lv + 100) * 131 + sd * 17 + fi * 7919)
                % (2 ** 31))
            noisy_d = seg_b + (rng_d.standard_normal(len(seg_b))
                               + 1j * rng_d.standard_normal(len(seg_b))) \
                * np.sqrt(p_c / 2.0)
            rows_d = rows_at(align_prior(noisy_d, f), 16, f["psym"])
            tot["D_battle段+种子20260929+冻结δ"][lv] += err_count(
                rows_d, f["gt"], deltas[fi])

n_sym = sum(f["psym"] for f in frames) * len(seeds)
print("n_sym/变体/档 = %d" % n_sym)
for lv in levels:
    print("[%d] " % lv + " | ".join("%s %.4f" % (v, tot[v][lv] / n_sym)
                                    for v in variants))

print("\n== 3) 逐帧错误贡献 @−26（A=CRC仲裁 vs B=冻结δ，各 3 种子合计）==")
order = sorted(pf_err["A"][-26], key=lambda i: -(pf_err["A"][-26][i]
                                                 - pf_err["B"][-26][i]))
for fi in order[:10]:
    print("f%02d δ_clean=%d δ_仲裁(mode)=%d  errA=%3d errB=%3d  差=%+d"
          % (fi, deltas[fi],
             int(np.bincount([pf_delta_a[-26][fi]]).argmax()),
             pf_err["A"][-26][fi], pf_err["B"][-26][fi],
             pf_err["A"][-26][fi] - pf_err["B"][-26][fi]))
n3 = sum(pf_err["A"][-26][i] - pf_err["B"][-26][i] for i in order)
print("A−B 总差 @−26: %+d / %d 符号" % (n3, n_sym))

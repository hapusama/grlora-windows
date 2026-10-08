# -*- coding: utf-8 -*-
r"""ff7：鲁棒性与对比实验（agent 审计并行运行的补充验证轮）。

相对 ff6 的增量：
  1. NC8 经典非相干基线（Σ|z_i|²，H0=gamma 闭式）——第三家族参照；
  2. 决策带密集档位（−18/−20/−22/−24/−26/−28/−30）+ 5 种子；
  3. 留一种子门限（阈值不接触被评估种子的 H0）→ 过拟合检查；
  4. 分 P 组（8/16/32）战表；
  5. 内建 GOF（归一化分数 vs 精确指数）+ 逐 capture 虚警均匀性 +
     A_P 斜率合理性检查。
变体：A_P（DeRa 式 GLRT-max）/ NC8 / FF_cert8（ff6 同款）/ FF_tmpl。
"""
import sys
import os
import json
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
                 r"\experiments\fullfield_cert_detect_20261003")
import ff_runner as R
import ff3_runner as F3

M = R.N_FINE
CKPT = os.path.join(os.path.dirname(F3.__file__), "checkpoint_ff7.jsonl")
LEVELS = [None] + [-v for v in (18, 20, 22, 24, 26, 28, 30)]
N_SEEDS = 5


def unit_scores7(rows, tmpl, anc, sigma2):
    pre = len(tmpl["k_j"]) - 4
    K = len(tmpl["k_j"])
    z_all = np.array([rows[j, k] for j, k in enumerate(tmpl["k_j"])],
                     dtype=np.complex128)
    z_pre = z_all[:pre]
    # A_P
    s_A = float(np.max(np.abs(np.fft.fft(z_pre, M))) ** 2) / (pre * sigma2)
    # NC8：非相干（经典 CAD 族）
    s_NC = float(np.sum(np.abs(z_pre) ** 2)) / (pre * sigma2)
    # cert8（ff6 同款：κ̂+bank9+确认集逐 chirp 锚）
    kap_hat = F3.kappa_est(z_all[:4])
    i_c = np.arange(4, K)
    ph_i = np.angle(tmpl["c_j"][4:]) - 2 * np.pi * anc["kap_true"] * i_c
    best = 0.0
    for db in range(-4, 5):
        kap = kap_hat + db / M
        S = np.sum(z_all[i_c] * np.exp(-1j * (2 * np.pi * kap * i_c
                                              + ph_i)))
        v = float(np.abs(S) ** 2)
        if v > best:
            best = v
    s_cert = best / (len(i_c) * sigma2)
    # oracle
    ph = tmpl["c_j"] / np.abs(tmpl["c_j"])
    zt = z_all.copy()
    if tmpl["conj_dn"]:
        zt[-2:] = np.conj(zt[-2:])
        ph = ph.copy()
        ph[-2:] = np.conj(ph[-2:])
    s_tmpl = float(np.abs(np.sum(zt * np.conj(ph))) ** 2) / (K * sigma2)
    return dict(A_P=s_A, NC8=s_NC, FF_cert8=s_cert, FF_tmpl=s_tmpl)


def eval_unit(f, tmpl, anc, level, seed):
    pre = f["pre"]
    start = R.field_start(f["hs"], pre)
    span = (pre + 4) * R.NF
    m = R.far_mask(2048, sorted(set(tmpl["k_j"])))
    recs = []
    seg = R.make_noisy(f, level, seed, f["hs"] % 4099,
                       start - R.NF, start + span + R.NF)
    rows = R.spectra(seg, R.chirp_windows(R.NF, pre))
    sig2 = float(np.median(np.abs(rows[:, m]) ** 2))
    if sig2 > 0:
        recs.append(dict(kind="h1", level=level, seed=seed, cap=f["cap"],
                         pre=pre,
                         scores=unit_scores7(rows, tmpl, anc, sig2)))
    for c in R.h0_centers(f, R.N_H0_PER_FRAME, level, seed):
        seg0 = R.make_noisy(f, level, seed, (f["hs"] + c) % 4099,
                            c - R.NF, c + span + R.NF)
        rows0 = R.spectra(seg0, R.chirp_windows(R.NF, pre))
        sig20 = float(np.median(np.abs(rows0[:, m]) ** 2))
        if sig20 <= 0:
            continue
        recs.append(dict(kind="h0", level=level, seed=seed, cap=f["cap"],
                         pre=pre,
                         scores=unit_scores7(rows0, tmpl, anc, sig20)))
    return recs


def summarize(ckpt_path):
    rows = [json.loads(l) for l in open(ckpt_path, encoding="utf-8")]
    h1 = [r for r in rows if r["kind"] == "h1"]
    h0 = [r for r in rows if r["kind"] == "h0"]
    noisy0 = [r for r in h0 if r["level"] is not None]
    V = ["A_P", "NC8", "FF_cert8", "FF_tmpl"]
    PFA = 1e-2
    lvs = sorted(set(r["level"] for r in noisy0), key=lambda x: -x)
    thr = {}
    for v in V:
        for lv in lvs:
            x = np.array([r["scores"][v] for r in noisy0
                          if r["level"] == lv])
            thr[(v, lv)] = float(np.quantile(x, 1 - PFA))
    # 分 P 组战表
    for pre_g in (8, 16, 32):
        sub = [r for r in h1 if r["pre"] == pre_g]
        if not sub:
            continue
        print("\n== P=%d（n=%d）裕度/Pd + 净胜 ==" % (pre_g, len(sub)))
        print("  [%6s]" % "level"
              + "".join("%15s" % v for v in V) + "   净胜(cert8)")
        for lv in lvs:
            s = [r for r in sub if r["level"] == lv]
            if not s:
                continue
            mgs, pds = {}, {}
            for v in V:
                mg = np.median([10 * np.log10(r["scores"][v])
                                - 10 * np.log10(thr[(v, lv)]) for r in s])
                pd = np.mean([r["scores"][v] > thr[(v, lv)] for r in s])
                mgs[v], pds[v] = mg, pd
            cells = "".join("%5.1f/%4.2f" % (mgs[v], pds[v]) for v in V)
            print("  [%6d] %s   %+5.2f"
                  % (lv, cells, mgs["FF_cert8"] - mgs["A_P"]))
    # 留一种子
    print("\n留一种子门限（净胜 cert8 vs A_P，每档 5 个留一值极差）：")
    seeds = sorted(set(r["seed"] for r in noisy0))
    for lv in lvs:
        los = []
        for s_hold in seeds:
            sub0 = [r for r in noisy0 if r["level"] == lv
                    and r["seed"] != s_hold]
            sub1 = [r for r in h1 if r["level"] == lv
                    and r["seed"] == s_hold]
            t = {}
            for v in ("A_P", "FF_cert8"):
                t[v] = float(np.quantile([r["scores"][v] for r in sub0],
                                         1 - PFA))
            mgA = np.median([10 * np.log10(r["scores"]["A_P"]) - 10 *
                             np.log10(t["A_P"]) for r in sub1])
            mgC = np.median([10 * np.log10(r["scores"]["FF_cert8"]) - 10 *
                             np.log10(t["FF_cert8"]) for r in sub1])
            los.append(mgC - mgA)
        print("  [%6d] 净胜 %s | 极差 %.2f dB"
              % (lv, " ".join("%+5.2f" % x for x in los),
                 max(los) - min(los)))
    # GOF：归一化分数 vs 精确指数（P(X>4.605x)=e^{-x}）
    print("\nGOF（归一化分数 s/q99，理论 Exp，生存 e^{-4.605x}）：")
    for v in V:
        x = np.array([r["scores"][v] / thr[(v, r["level"])]
                      for r in noisy0])
        qs = [np.quantile(x, q) for q in (0.5, 0.9, 0.99)]
        th = [-np.log(1 - q) / 4.605 for q in (0.5, 0.9, 0.99)]
        # 分箱卡方
        edges = np.quantile(x, np.linspace(0, 1, 11))
        obs, _ = np.histogram(x, edges)
        exp_p = np.exp(-4.605 * edges[:-1]) - np.exp(-4.605 * edges[1:])
        exp_c = exp_p * len(x)
        chi2 = float(np.sum((obs - exp_c) ** 2 / np.maximum(exp_c, 1e-9)))
        print("  %-9s q50/q90/q99 实测 %s vs 理论 %s | χ²=%5.1f (df≈9)"
              % (v, " ".join("%.3f" % q for q in qs),
                 " ".join("%.3f" % t for t in th), chi2))
    # 逐 cap 虚警均匀性
    print("\n逐 capture 虚警均匀性（各档门限下 h0 超阈值率，期望 1e-2）：")
    for cap in sorted(set(r["cap"] for r in noisy0)):
        rates = []
        for v in V:
            ex = [r["scores"][v] > thr[(v, r["level"])] for r in noisy0
                  if r["cap"] == cap]
            rates.append(np.mean(ex))
        print("  %s %s" % (cap, " ".join("%.4f" % x for x in rates)))
    # 斜率合理性
    print("\nA_P 斜率检查（中位 margin 对 level 线性回归）：")
    for cap in sorted(set(r["cap"] for r in h1)):
        sub = [r for r in h1 if r["cap"] == cap and r["level"] is not None]
        xs = np.array([r["level"] for r in sub], dtype=float)
        ys = np.array([10 * np.log10(r["scores"]["A_P"])
                       - 10 * np.log10(thr[("A_P", r["level"])])
                       for r in sub])
        b, a = np.polyfit(xs, ys, 1)
        r2 = 1 - np.sum((ys - a - b * xs) ** 2) / np.sum((ys - np.mean(ys))
                                                         ** 2)
        print("  %s P=%d: 斜率 %+.3f dB/dB, R²=%.4f"
              % (cap, sub[0]["pre"], b, r2))


def main():
    t0 = time.time()
    R.FRAMES = R.build_frames()
    TMPL = [R.frame_template(f) for f in R.FRAMES]
    ANC = [F3.clean_anchors(f, t) for f, t in zip(R.FRAMES, TMPL)]
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["kind"], r["level"], r["seed"], r["frame"]))
            except Exception:
                pass
        print("断点恢复：跳过 %d" % len(done))
    units = []
    for lv in LEVELS:
        for sd in range(N_SEEDS if lv is not None else 1):
            for fi in range(len(R.FRAMES)):
                units.append((lv, sd, fi))
    n = 0
    with open(CKPT, "a", encoding="utf-8") as fh:
        for lv, sd, fi in units:
            if ("h1", lv, sd, fi) in done:
                continue
            for r in eval_unit(R.FRAMES[fi], TMPL[fi], ANC[fi], lv, sd):
                r["frame"] = fi
                if (r["kind"], r["level"], r["seed"], fi) in done:
                    continue
                fh.write(json.dumps(r) + "\n")
                n += 1
            fh.flush()
    print("ff7 完成：%d 条，%.0fs" % (n, time.time() - t0))
    summarize(CKPT)


if __name__ == "__main__":
    main()

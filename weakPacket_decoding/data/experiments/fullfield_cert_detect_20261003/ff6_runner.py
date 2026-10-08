# -*- coding: utf-8 -*-
r"""ff6：认证确认检测器（split κ̂ + bank9 + 确认集逐 chirp 相位锚）vs A_P。

溯源补交（独立审计 A0 指出本轮原以 heredoc 内联运行未入库）。
逻辑与 ff3_runner/ff7_runner 的 cert8 完全一致：
  A_P      DeRa 式 max-q 列向 FFT（现役基线）；
  FF_cert8 κ̂ 从前导半场（4 chirp，与确认样本不相交）估计 → 确认集
           {chirp 4..K−1}（含后半前导+sync+SFD，8 chirp @P=8）逐 chirp
           相位锚（干净域冻结，GT 锚定声明）+ bank9（κ̂±4 bin）取 max；
  FF_tmpl  per-chirp 干净模板（oracle）。
checkpoint_ff6.jsonl 记录 schema 同 ff3（kind/level/seed/cap/pre/
scores/frame）；scores 含 dkap（κ̂ 误差，bin）。
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
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "checkpoint_ff6.jsonl")


def unit_scores6(rows, tmpl, anc, sigma2):
    pre = len(tmpl["k_j"]) - 4
    K = len(tmpl["k_j"])
    z_all = np.array([rows[j, k] for j, k in enumerate(tmpl["k_j"])],
                     dtype=np.complex128)
    z_pre = z_all[:pre]
    s_A = float(np.max(np.abs(np.fft.fft(z_pre, M))) ** 2) / (pre * sigma2)
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
    ph = tmpl["c_j"] / np.abs(tmpl["c_j"])
    zt = z_all.copy()
    if tmpl["conj_dn"]:
        zt[-2:] = np.conj(zt[-2:])
        ph = ph.copy()
        ph[-2:] = np.conj(ph[-2:])
    s_tmpl = float(np.abs(np.sum(zt * np.conj(ph))) ** 2) / (K * sigma2)
    return dict(A_P=s_A, FF_cert8=s_cert, FF_tmpl=s_tmpl,
                dkap=F3.wrap_dk(kap_hat, anc["kap_true"]) * M)


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
                         scores=unit_scores6(rows, tmpl, anc, sig2)))
    for c in R.h0_centers(f, R.N_H0_PER_FRAME, level, seed):
        seg0 = R.make_noisy(f, level, seed, (f["hs"] + c) % 4099,
                            c - R.NF, c + span + R.NF)
        rows0 = R.spectra(seg0, R.chirp_windows(R.NF, pre))
        sig20 = float(np.median(np.abs(rows0[:, m]) ** 2))
        if sig20 <= 0:
            continue
        recs.append(dict(kind="h0", level=level, seed=seed, cap=f["cap"],
                         pre=pre,
                         scores=unit_scores6(rows0, tmpl, anc, sig20)))
    return recs


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
    for lv in R.LEVELS:
        for sd in range(R.N_SEEDS if lv is not None else 1):
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
    print("ff6 完成：%d 新记录，%.0fs" % (n, time.time() - t0))


if __name__ == "__main__":
    main()

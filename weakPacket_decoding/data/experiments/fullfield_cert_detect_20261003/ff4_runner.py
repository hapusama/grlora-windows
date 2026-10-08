# -*- coding: utf-8 -*-
r"""ff4：认证局部 GLRT——定宽 κ-bank 的拆分确认检测器。

ff3 定律：拆分（估计/确认样本解耦）→ H0 精确指数（分位比 0.506 vs
0.500）+ 门限 −3.3dB；但 K_e=4 的 κ̂ 在 −22 以下崩塌（误差从亚 bin →
几十 bin），确认和失相干，深端输 A_P（其 256 维 max = κ 的全 GLRT，
永不失相干但付满 H0 税）。

ff4 补上中间态：**确认统计量在 κ̂ 中心的固定宽 bank 上取 max**——
  H0：bank 中心依赖估计半场（与确认样本不相交）→ 条件高斯保持，
      H0 = b 个相关指数的 max 族（解析；b=1 退化为纯指数）；
  H1：覆盖 |Δκ̂| ≤ (b−1)/2 bin 的中心误差——宽度 b 对抗估计噪声；
  设计规则：门限代价 ~10log10(ln b 增量)，κ̂ 误差分布（实测）→
  目标 Pd 所需最小 b。
变体：A_P / FF_point(b=1) / FF_bank9(±4) / FF_bank25(±12) / FF_tmpl，
K_e=4，其余协议同 ff3。
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
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "checkpoint_ff4.jsonl")
BANKS = (1, 9, 25)


def unit_scores4(rows, tmpl, anc, sigma2):
    pre = len(tmpl["k_j"]) - 4
    K = len(tmpl["k_j"])
    z_all = np.array([rows[j, k] for j, k in enumerate(tmpl["k_j"])],
                     dtype=np.complex128)
    z_pre = z_all[:pre]
    s_A = float(np.max(np.abs(np.fft.fft(z_pre, M))) ** 2) / (pre * sigma2)
    kap_hat = F3.kappa_est(z_all[:4])
    i_up = np.arange(4, pre + 2)
    i_dn = np.arange(pre + 2, K)
    zd = np.conj(z_all[pre + 2:]) if tmpl["conj_dn"] else z_all[pre + 2:]
    ph_up = np.where(i_up >= pre, np.exp(-1j * anc["r_syn"]), 1.0)
    K_c = len(i_up) + len(i_dn)
    vals = {b: 0.0 for b in BANKS}
    half = {b: (b - 1) // 2 for b in BANKS}
    for db in range(-max(half.values()), max(half.values()) + 1):
        kap = kap_hat + db / M
        Su = np.sum(z_all[i_up] * np.exp(-1j * 2 * np.pi * kap * i_up)
                    * ph_up)
        Sd = np.sum(zd * np.exp(-1j * (2 * np.pi * kap * i_dn
                                       + anc["r_dn"])))
        v = float(np.abs(Su) ** 2 + np.abs(Sd) ** 2)
        for b in BANKS:
            if abs(db) <= half[b] and v > vals[b]:
                vals[b] = v
    out = dict(A_P=s_A, FF_tmpl=0.0,
               dkap_bins=F3.wrap_dk(kap_hat, anc["kap_true"]) * M)
    for b in BANKS:
        out["FF_bank%d" % b] = vals[b] / (K_c * sigma2)
    ph = tmpl["c_j"] / np.abs(tmpl["c_j"])
    zt = z_all.copy()
    if tmpl["conj_dn"]:
        zt[-2:] = np.conj(zt[-2:])
        ph = ph.copy()
        ph[-2:] = np.conj(ph[-2:])
    out["FF_tmpl"] = float(np.abs(np.sum(zt * np.conj(ph))) ** 2) \
        / (K * sigma2)
    return out


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
                         pre=pre, scores=unit_scores4(rows, tmpl, anc,
                                                      sig2)))
    for c in R.h0_centers(f, R.N_H0_PER_FRAME, level, seed):
        seg0 = R.make_noisy(f, level, seed, (f["hs"] + c) % 4099,
                            c - R.NF, c + span + R.NF)
        rows0 = R.spectra(seg0, R.chirp_windows(R.NF, pre))
        sig20 = float(np.median(np.abs(rows0[:, m]) ** 2))
        if sig20 <= 0:
            continue
        recs.append(dict(kind="h0", level=level, seed=seed, cap=f["cap"],
                         pre=pre,
                         scores=unit_scores4(rows0, tmpl, anc, sig20)))
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
            if n and n % 1000 == 0:
                print("  %d 条 (%.0fs)" % (n, time.time() - t0), flush=True)
    print("完成：%d 新记录，%.0fs" % (n, time.time() - t0))


if __name__ == "__main__":
    main()

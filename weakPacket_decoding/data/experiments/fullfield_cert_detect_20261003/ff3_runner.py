# -*- coding: utf-8 -*-
r"""ff3：拆分式认证检测器（estimate/confirm 样本解耦）——超越 DeRa 式统计量的机制级验证。

ff2 定律（RESULTS §8）：est-then-sum 的确认统计量若与 κ̂ 估计共用样本，
H0 被 argmax 选择偏置抬 +4.9dB（耦合税）；只有 κ 独立已知（FF_se_ok）
才能收获 +0.8~1.5dB 且门限更低。ff3 的正解 = **帧内数据拆分**：

    κ̂ 从前 K_e 个前导 chirp 估计（H0 下与确认样本独立）；
    确认集 = 其余前导 + sync word(2 upchirp) + SFD(2 downchirp)，
    用帧相位恒等式（DeRa Eq.6 = 2πκ）的线性斜率 + 段级类型偏移做相干和。

理论预测（预注册）：
  - H0：确认和 |Σ z w|² 给定权重 = 精确指数分布（条件高斯），分档门限
    应 ≈ FF_tmpl 的 8.56dB（vs A_P 11.26dB）→ +2.7dB 证书红利；
  - H1：能量 = (P−K_e+4)/P × A_P（P=8/16/32 全能量恢复）+ FF_se_ok 级
    收获 − κ̂ 估计损失；
  - 深端 Pd 上限 = P(κ̂ 正确)（est 仅 K_e chirp）——诚实报告权衡。

变体：
    A_P            DeRa 式 max-q 列向 FFT（现役基线，同 ff_runner）；
    FF_split_pre   纯认证锚：确认 = 其余前导 upchirp（无 sync/down、
                   无模型、无 max）→ 精确指数 H0；
    FF_split_syn   + sync（无类型偏移模型 → 含 135° 跳变失配，消融）；
    FF_split_model + sync/down 段级偏移（干净域冻结：粗模型 2 标量/帧）
                   + down 共轭约定（干净域冻结）→ 主变体；
    FF_split_e8    同 model 但 K_e=8（κ̂ 更稳 vs 能量让度，权衡曲线）；
    FF_tmpl        per-chirp 干净模板（oracle 上界）。

协议：同 ff_runner/ff2（真实 .bin、整包加噪、种子常数 20261003、
GT 冻结、死区过滤、σ̂² 共享）。类型偏移/共轭从干净信号冻结（GT 锚定
口径，H0 下与噪声独立 → 条件高斯论证保持）。
"""
import sys
import os
import json
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
                 r"\experiments\fullfield_cert_detect_20261003")
import ff_runner as R

M = R.N_FINE
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "checkpoint_ff3.jsonl")


def kappa_est(z_est):
    """κ̂ = 零填 FFT 峰（亚网格抛物线内插），wrap 到 (−0.5, 0.5]。"""
    F = np.abs(np.fft.fft(z_est, M))
    q = int(np.argmax(F))
    qf = R._interp(F, q)
    kap = qf / M
    if kap > 0.5:
        kap -= 1.0
    return float(kap)


def wrap_dk(kap, kap_true):
    d = (kap - kap_true + 0.5) % 1.0 - 0.5
    return float(d)


def clean_anchors(f, tmpl):
    """干净域冻结：κ_true（亚网格）、段级类型偏移 r_syn/r_dn。"""
    start = R.field_start(f["hs"], f["pre"])
    wins = R.chirp_windows(start, f["pre"])
    seg = np.asarray(f["iq"][wins[0][0]:wins[-1][0] + R.NF],
                     dtype=np.complex64)
    rows = R.spectra(seg, [(j * R.NF, ref)
                           for j, (_, ref) in enumerate(wins)])
    z = np.array([rows[j, k] for j, k in enumerate(tmpl["k_j"])],
                 dtype=np.complex128)
    pre = f["pre"]
    F = np.abs(np.fft.fft(z[:pre], M))
    kap_true = kappa_est(z[:pre])
    i_all = np.arange(len(z), dtype=float)
    ramp = np.exp(-1j * 2 * np.pi * kap_true * i_all)
    zr = z * ramp
    r_syn = float(np.angle(np.sum(zr[pre:pre + 2])))
    zd = np.conj(zr[pre + 2:]) if tmpl["conj_dn"] else zr[pre + 2:]
    r_dn = float(np.angle(np.sum(zd)))
    return dict(kap_true=kap_true, r_syn=r_syn, r_dn=r_dn)


def unit_scores3(rows, tmpl, anc, sigma2, K_e=4):
    pre = len(tmpl["k_j"]) - 4
    K = len(tmpl["k_j"])
    z_all = np.array([rows[j, k] for j, k in enumerate(tmpl["k_j"])],
                     dtype=np.complex128)
    z_pre = z_all[:pre]
    # A_P（现役基线）
    s_A = float(np.max(np.abs(np.fft.fft(z_pre, M))) ** 2) / (pre * sigma2)
    # stage-1：κ̂（前 K_e 前导，与确认样本不相交）
    kap_hat = kappa_est(z_all[:K_e])
    # FF_split_pre：确认 = 其余前导（纯指数锚）
    i1 = np.arange(K_e, pre)
    S1 = np.sum(z_all[i1] * np.exp(-1j * 2 * np.pi * kap_hat * i1))
    s_pre = float(np.abs(S1) ** 2) / (len(i1) * sigma2)
    # FF_split_syn：+ sync（无模型，消融）
    i2 = np.arange(K_e, pre + 2)
    S2 = np.sum(z_all[i2] * np.exp(-1j * 2 * np.pi * kap_hat * i2))
    s_syn = float(np.abs(S2) ** 2) / (len(i2) * sigma2)
    # FF_split_model：+ 段级偏移 + down（主变体）
    w_up = np.exp(-1j * 2 * np.pi * kap_hat * i2)
    ph_up = np.where(i2 >= pre, np.exp(-1j * anc["r_syn"]), 1.0)
    Su = np.sum(z_all[i2] * w_up * ph_up)
    i3 = np.arange(pre + 2, K)
    zd = np.conj(z_all[pre + 2:]) if tmpl["conj_dn"] else z_all[pre + 2:]
    Sd = np.sum(zd * np.exp(-1j * (2 * np.pi * kap_hat * i3
                                   + anc["r_dn"])))
    s_mod = float(np.abs(Su) ** 2 + np.abs(Sd) ** 2) \
        / ((len(i2) + len(i3)) * sigma2)
    # FF_tmpl（per-chirp oracle，v1 逐字）
    ph = tmpl["c_j"] / np.abs(tmpl["c_j"])
    zt = z_all.copy()
    if tmpl["conj_dn"]:
        zt[-2:] = np.conj(zt[-2:])
        ph = ph.copy()
        ph[-2:] = np.conj(ph[-2:])
    s_tmpl = float(np.abs(np.sum(zt * np.conj(ph))) ** 2) / (K * sigma2)
    dkap = wrap_dk(kap_hat, anc["kap_true"])
    return dict(A_P=s_A, FF_split_pre=s_pre, FF_split_syn=s_syn,
                FF_split_model=s_mod, FF_tmpl=s_tmpl,
                kap_hat=kap_hat, dkap=dkap)


def eval_unit(f, tmpl, anc, level, seed, K_e=4):
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
        sc = unit_scores3(rows, tmpl, anc, sig2, K_e)
        recs.append(dict(kind="h1", level=level, seed=seed, cap=f["cap"],
                         pre=pre, scores=sc, ke=K_e))
    for c in R.h0_centers(f, R.N_H0_PER_FRAME, level, seed):
        seg0 = R.make_noisy(f, level, seed, (f["hs"] + c) % 4099,
                            c - R.NF, c + span + R.NF)
        rows0 = R.spectra(seg0, R.chirp_windows(R.NF, pre))
        sig20 = float(np.median(np.abs(rows0[:, m]) ** 2))
        if sig20 <= 0:
            continue
        sc0 = unit_scores3(rows0, tmpl, anc, sig20, K_e)
        recs.append(dict(kind="h0", level=level, seed=seed, cap=f["cap"],
                         pre=pre, scores=sc0, ke=K_e))
    return recs


def main():
    t0 = time.time()
    R.FRAMES = R.build_frames()
    TMPL = [R.frame_template(f) for f in R.FRAMES]
    ANC = [clean_anchors(f, t) for f, t in zip(R.FRAMES, TMPL)]
    print("干净锚冻结完成（κ_true/r_syn/r_dn）", flush=True)

    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["kind"], r["level"], r["seed"], r["frame"],
                          r.get("ke", 4)))
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
            for ke in (4, 8):
                if ke == 8 and R.FRAMES[fi]["pre"] < 16:
                    continue
                if all((k, lv, sd, fi, ke) in done for k in ("h1", "h0")):
                    continue
                for r in eval_unit(R.FRAMES[fi], TMPL[fi], ANC[fi], lv,
                                   sd, ke):
                    r["frame"] = fi
                    if (r["kind"], r["level"], r["seed"], fi, ke) in done:
                        continue
                    fh.write(json.dumps(r) + "\n")
                    n += 1
            fh.flush()
            if n and n % 1000 == 0:
                print("  %d 条 (%.0fs)" % (n, time.time() - t0), flush=True)
    print("完成：%d 新记录，%.0fs" % (n, time.time() - t0))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""ff5：全场 GLRT-max（DeRa 结构 + 全场能量 + 类型锚点）——机制级超越的主变体。

ff3/ff4 定律链：拆分族证书完美（指数 H0 0.506）但深端受 κ̂ 孔径钳制；
A_P 的 256 维 max = 未知 κ 的 GLRT，深端永不失相干，但只用 8 前导
（DeRa 检测级不吃 sync/SFD）且付满 H0 税。

ff5 主变体 **FF_GLBLRT**：全场 12 已知 chirp 的 GLRT-max——
    S(q) = Σ_pre z_i e^{-j2πqi} + e^{-j·r_syn} Σ_syn z_i e^{-j2πqi}
         + e^{-j·r_dn} Σ_dn zd_i e^{-j2πqi}，score = max_q |S(q)|²/(Kσ²)。
  - 权重与数据无关（锚点干净域冻结）→ H0 = 固定权 max-of-256 相关
    指数族（与 A_P 同证书类别，可解析计算）；
  - H1 = +10log10(12/8)=+1.76dB 全场能量，q-max 深端鲁棒（无 κ̂）；
  - 类型锚点 = DeRa Eqs.20-21 解码级相位模型的检测级搬运（其检测级
    未使用）——冻结版为上界，扰动消融测可部署性。
锚点扰动消融：±30°（模型误差小）/ ±90°（粗模型）逐 chirp 随机扰动，
测 Pd-锚点误差敏感曲线。
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
CKPT = os.path.join(EXP_DIR, "checkpoint_ff5.jsonl")
PERT = (30.0, 90.0)


def glrt_scores(z_all, tmpl, anc, sigma2, rng=None):
    pre = len(tmpl["k_j"]) - 4
    K = len(tmpl["k_j"])
    i_all = np.arange(K)
    zd = np.conj(z_all[pre + 2:]) if tmpl["conj_dn"] else z_all[pre + 2:]
    z_eff = np.concatenate([z_all, zd])          # down 段按约定并入
    i_eff = np.concatenate([i_all, i_all[pre + 2:]])
    w_type = np.concatenate([
        np.where(i_all >= pre, np.exp(-1j * anc["r_syn"]), 1.0),
        np.where(i_all[pre + 2:] >= pre, np.exp(-1j * anc["r_dn"]), 1.0)])
    idx = np.arange(256)
    q_grid = (idx / M)[1:]                        # q=0 与 DC 列退化，弃
    Fq = np.abs(np.fft.fft(z_eff * w_type
                           * np.exp(-1j * 2 * np.pi * 0
                                    * i_eff), 1)) # placeholder
    # 直接按定义：S(q) = Σ_i z_eff,i w_i e^{-j2πq i}
    S = np.exp(-1j * 2 * np.pi * np.outer(q_grid, i_eff)) @ (z_eff * w_type)
    s_glrt = float(np.max(np.abs(S) ** 2)) / (len(z_eff) * sigma2)
    out = dict(FF_GLBLRT=s_glrt)
    if rng is not None:
        for p in PERT:
            pert = np.exp(1j * np.deg2rad(p) *
                          rng.uniform(-1, 1, size=len(w_type)))
            S_p = np.exp(-1j * 2 * np.pi
                         * np.outer(q_grid, i_eff)) @ (z_eff * w_type
                                                       * pert)
            out["FF_pert%d" % int(p)] = float(
                np.max(np.abs(S_p) ** 2)) / (len(z_eff) * sigma2)
    return out


def unit_scores(rows, tmpl, anc, sigma2, rng=None):
    pre = len(tmpl["k_j"]) - 4
    K = len(tmpl["k_j"])
    z_all = np.array([rows[j, k] for j, k in enumerate(tmpl["k_j"])],
                     dtype=np.complex128)
    s_A = float(np.max(np.abs(np.fft.fft(z_all[:pre], M))) ** 2) \
        / (pre * sigma2)
    out = dict(A_P=s_A)
    out.update(glrt_scores(z_all, tmpl, anc, sigma2, rng))
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
        rng = np.random.default_rng((20261003 * 13 + seed * 7
                                     + (f["hs"] % 8191) * 31) % (2 ** 31))
        recs.append(dict(kind="h1", level=level, seed=seed, cap=f["cap"],
                         pre=pre,
                         scores=unit_scores(rows, tmpl, anc, sig2, rng)))
    for c in R.h0_centers(f, R.N_H0_PER_FRAME, level, seed):
        seg0 = R.make_noisy(f, level, seed, (f["hs"] + c) % 4099,
                            c - R.NF, c + span + R.NF)
        rows0 = R.spectra(seg0, R.chirp_windows(R.NF, pre))
        sig20 = float(np.median(np.abs(rows0[:, m]) ** 2))
        if sig20 <= 0:
            continue
        rng0 = np.random.default_rng((20261003 * 17 + seed * 11
                                      + (c % 8191) * 29) % (2 ** 31))
        recs.append(dict(kind="h0", level=level, seed=seed, cap=f["cap"],
                         pre=pre,
                         scores=unit_scores(rows0, tmpl, anc, sig20,
                                            rng0)))
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

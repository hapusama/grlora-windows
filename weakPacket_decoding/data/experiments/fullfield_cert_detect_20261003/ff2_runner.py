# -*- coding: utf-8 -*-
r"""ff2：可部署全场收获器（全局类型相位偏移模型 + 两段式确认统计量）。

ff_runner（同目录）证明：全场 oracle 界 FF_tmpl 深端裕度 +4.2~+5.6dB
over A_P，缺口 = 类型切换相位跳变（~135°/230°）+ 搜索族 H0 膨胀。
本轮验证可部署收获路径：

  诊断 D：干净域逐帧测 (κ_true, φ0, r_sync, r_dn)，检验类型偏移的
  跨帧全局稳定性（理论：r_sync 由协议常数 bin 偏移决定=常数；r_dn 由
  (k_dn−k_pre) 列位差线性决定——几何确定量，非自由参数）。

  变体（同一份带噪场谱/σ̂²/期望列，全部真实 .bin 数据）：
    A_P        DeRa 式前导基线（同 ff_runner，同单元重算）；
    FF_cal     单段式：全场相干 + 全局类型偏移（校准集冻结），
               列向 FFT 256 格线性搜索；
    FF_conf    两段式：前导 FFT 出 (κ̂,φ̂0) → 全场相干确认统计量，
               **不再搜索**（吃门限红利的主形态）；
    FF_conf_ok 同上但 κ 用干净前导（消融：估计损失）；
    FF_tmpl    oracle 界（参照，同单元重算）。

  校准/测试划分：frame 偶数=校准集（全局 r_sync、(base_dn,c_dn)、down
  共轭约定），奇数=测试集；测试集上报告（全帧版作界一并给出）。
  H0 单元跑完整可部署管线（含 κ̂ 从噪声估计）——门限诚实。

协议：同 ff_runner（整包加噪、种子常数 20261003、GT 冻结、死区过滤）。
"""
import sys
import os
import json
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
                 r"\experiments\fullfield_cert_detect_20261003")
import ff_runner as R

N_FINE = R.N_FINE
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "checkpoint_ff2.jsonl")


# ---------------- 诊断 D + 校准 ----------------
def clean_field_z(f, tmpl):
    """干净场逐 chirp 复数值（列=期望列）+ 真实 κ/φ0（前导 ramp 拟合）。"""
    start = R.field_start(f["hs"], f["pre"])
    wins = R.chirp_windows(start, f["pre"])
    seg = np.asarray(f["iq"][wins[0][0]:wins[-1][0] + R.NF],
                     dtype=np.complex64)
    rows = R.spectra(seg, [(j * R.NF, ref)
                           for j, (_, ref) in enumerate(wins)])
    z = np.array([rows[j, k] for j, k in enumerate(tmpl["k_j"])],
                 dtype=np.complex128)
    pre = f["pre"]
    ph = np.unwrap(np.angle(z[:pre]))
    b, a = np.polyfit(np.arange(pre, dtype=float), ph, 1)
    kappa = b / (2 * np.pi)
    # φ0 取 j=0 相位（FFT 在 j=0 无旋转）
    return z, float(kappa), float(a % (2 * np.pi))


def clean_kappa(z_clean_pre):
    """干净前导的 κ（FFT 峰，鲁棒；替代 polyfit/unwrap 的 nan 源）。"""
    F = np.abs(np.fft.fft(z_clean_pre, N_FINE))
    q = int(np.argmax(F))
    kap = q / N_FINE
    return kap - 1.0 if kap > 0.5 else kap


def calibrate(frames, tmpls):
    """诊断 D：类型偏移跨帧稳定性（结论：全局常数模型失败 → 自校准）。"""
    stats = {True: [], False: []}
    r_syn_all = []
    for i, (f, t) in enumerate(zip(frames, tmpls)):
        if i % 2 == 1:
            continue
        z, kap, phi0 = clean_field_z(f, t)
        pre = f["pre"]
        ramp = np.exp(-1j * (2 * np.pi * kap * np.arange(len(z))
                             + phi0))
        zr = z * ramp
        r_syn = np.angle(np.mean(zr[pre:pre + 2]))
        d = (t["k_j"][-1] - t["k_pre"])
        for conj in (True, False):
            zd = np.conj(zr[-2:]) if conj else zr[-2:]
            stats[conj].append((float(d), float(np.angle(np.mean(zd)))))
        r_syn_all.append(r_syn)
    r_sync = float(np.angle(np.mean(np.exp(1j * np.array(r_syn_all)))))
    circ_std_syn = float(np.std(np.angle(np.exp(1j *
                         (np.array(r_syn_all) - r_sync)))))
    best_conj, best_var = None, None
    for conj in (True, False):
        arr = np.array([r for _, r in stats[conj]])
        var = float(np.var(np.angle(np.exp(1j * (arr - np.angle(
            np.mean(np.exp(1j * arr))))))))
        if best_var is None or var < best_var:
            best_conj, best_var = conj, var
    print("诊断 D（偶数帧 n=%d）：r_sync=%+.1f° (圆std %.1f°) | "
          "down_conj=%s r_dn 圆方差 %.3f rad² → 全局常数模型%s"
          % (len(r_syn_all), np.degrees(r_sync), np.degrees(circ_std_syn),
             best_conj, best_var,
             "成立" if circ_std_syn < 0.17 and best_var < 0.1 else "失败"))
    return dict(conj_dn=bool(best_conj), std_syn=circ_std_syn,
                var_dn=best_var)


def r_offsets(tmpl, cal):
    """逐帧类型偏移预测：r_sync 全局常数；r_dn = base + c·(k_dn−k_pre)。"""
    d = tmpl["k_j"][-1] - tmpl["k_pre"]
    return (0.0, cal["r_sync"], cal["base_dn"] + cal["c_dn"] * d)


# ---------------- 变体统计量 ----------------
def unit_scores2(rows, tmpl, cal, sigma2, kappa_clean):
    """A_P 基线 + 两段式自校准收获器 + oracle 界（同份带噪谱/σ̂²/列）。

    FF_se   两段式自校准：前导 FFT 峰得 κ̂ → pre/sync/down 三段在 κ̂ 处
            各自内部相干、幅度相加（未知类型偏移 ML；down 段两种共轭
            约定取大），**无任何 max 搜索**；
    FF_se_ok 同上但 κ 用干净前导（消融：κ̂ 估计损失）。
    """
    pre = len(tmpl["k_j"]) - 4
    z_all = np.array([rows[j, k] for j, k in enumerate(tmpl["k_j"])],
                     dtype=np.complex128)
    z_pre = z_all[:pre]
    K = len(z_all)
    s_A = float(np.max(np.abs(np.fft.fft(z_pre, N_FINE))) ** 2) \
        / (pre * sigma2)

    def _se(kap):
        w_pre = np.exp(-1j * 2 * np.pi * kap * np.arange(pre))
        w_syn = np.exp(-1j * 2 * np.pi * kap * np.arange(pre, pre + 2))
        w_dn = np.exp(-1j * 2 * np.pi * kap * np.arange(pre + 2, K))
        zd = z_all[pre + 2:]
        seg_dn = max(float(np.abs(np.sum(zd * w_dn))),
                     float(np.abs(np.sum(np.conj(zd) * w_dn))))
        seg = (np.abs(np.sum(z_pre * w_pre))
               + np.abs(np.sum(z_all[pre:pre + 2] * w_syn))
               + seg_dn)
        return float(seg ** 2) / (K * sigma2)

    Fp = np.abs(np.fft.fft(z_pre, N_FINE))
    kap_hat = int(np.argmax(Fp)) / N_FINE
    if kap_hat > 0.5:
        kap_hat -= 1.0
    s_se = _se(kap_hat)
    s_se_ok = _se(kappa_clean)
    # FF_tmpl：oracle 界（v1 逐字：帧级约定同时作用 z 与 ph）
    ph = tmpl["c_j"] / np.abs(tmpl["c_j"])
    zt = z_all.copy()
    if tmpl["conj_dn"]:
        zt[-2:] = np.conj(zt[-2:])
        ph = ph.copy()
        ph[-2:] = np.conj(ph[-2:])
    s_tmpl = float(np.abs(np.sum(zt * np.conj(ph))) ** 2) / (K * sigma2)
    return dict(A_P=s_A, FF_se=s_se, FF_se_ok=s_se_ok, FF_tmpl=s_tmpl,
                kappa_clean=kappa_clean, kappa_hat=kap_hat)


def eval_unit2(f, tmpl, cal, level, seed, kappa_clean):
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
        sc = unit_scores2(rows, tmpl, cal, sig2, kappa_clean)
        recs.append(dict(kind="h1", level=level, seed=seed, cap=f["cap"],
                         pre=pre, scores=sc))
    for c in R.h0_centers(f, R.N_H0_PER_FRAME, level, seed):
        seg0 = R.make_noisy(f, level, seed, (f["hs"] + c) % 4099,
                            c - R.NF, c + span + R.NF)
        rows0 = R.spectra(seg0, R.chirp_windows(R.NF, pre))
        sig20 = float(np.median(np.abs(rows0[:, m]) ** 2))
        if sig20 <= 0:
            continue
        sc0 = unit_scores2(rows0, tmpl, cal, sig20, kappa_clean)
        recs.append(dict(kind="h0", level=level, seed=seed, cap=f["cap"],
                         pre=pre, scores=sc0))
    return recs


def main():
    t0 = time.time()
    R.FRAMES = R.build_frames()
    TMPL = [R.frame_template(f) for f in R.FRAMES]
    cal = calibrate(R.FRAMES, TMPL)
    kaps = []
    for f, t in zip(R.FRAMES, TMPL):
        z, _, _ = clean_field_z(f, t)
        kaps.append(clean_kappa(z[:f["pre"]]))

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
            if all((k, lv, sd, fi) in done for k in ("h1", "h0")):
                continue
            for r in eval_unit2(R.FRAMES[fi], TMPL[fi], cal, lv, sd,
                                kaps[fi]):
                r["frame"] = fi
                if (r["kind"], r["level"], r["seed"], fi) in done:
                    continue
                fh.write(json.dumps(r) + "\n")
                n += 1
            fh.flush()
            if n and n % 500 == 0:
                print("  %d 条 (%.0fs)" % (n, time.time() - t0), flush=True)
    print("完成：%d 新记录，%.0fs" % (n, time.time() - t0))


if __name__ == "__main__":
    main()

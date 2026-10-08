# -*- coding: utf-8 -*-
r"""2026-10-03 全场相干×证书实验（机制级检测统计量，实验B-检测型）。

验证对象（doc/去斜与采样挖掘_20261003.md §3.3 复合检测统计量的三支柱）：
  ① 全场相干积分：DeRa 式只用 P 个前导 upchirp，全场 = P + 2 sync-word
     upchirp + 2 SFD downchirp（0.25 部分符号弃用，声明）。理论
     10log10((P+4.25)/P)：P=8 → +1.85dB。
  ② 漂移校正余量：FF-lin（列向 FFT，线性相位搜索，可部署）vs
     FF-tmpl（干净模板相位相干和，含二次漂移/分数 bin 相位的 oracle 上界）
     ——差值 = 真实数据上漂移校正的剩余空间（keystone 的对象，E2 前的
     真实界）。
  ③ 证书：统计量自归一（÷K·σ̂²）→ H0 分布应与加噪档位无关（CFAR 性）；
     实证校准门限@Pfa=1e-2 + 高斯稳定性检查。

协议遵守（EXPERIMENT_PROTOCOL v1.0）：
  - 纯 AWGN 同一实现喂所有变体；整包 SNR 口径（带外底标定，逐包 P_add）；
  - GT 只来自干净原生解（帧集与 dera_front_battle_20260930 完全一致，
    build 带原生 CRC assert）；本实验为机制级消融：窗口位置 GT 锚定
    （实验B 哲学），不做扫描/locate——系统级集成属后续 E3 型实验；
  - 期望列/相位参考在干净信号上冻结（全部变体同一份，公平）；
  - 种子常数 20261003（派生公式与 front_runner 同构）。

偏差声明（写 RESULTS 时复述）：
  1. 机制级（GT 锚定），非全链检测——变体间可比，不与 port 的
     4.5dB 系统门限直接比；
  2. σ̂² 用带噪谱远列中位数（全部变体共享，Eq.47 形态）；
  3. SFD 0.25 部分符号未计入（K_tot = P+4）；
  4. downchirp 相位约定（是否取共轭）在干净信号上按相干性判定并冻结
     （可部署版在带噪 max 下同样判定，阈值处等价）。
"""
import sys
import os
import csv
import json
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
from weak_decoder.baselines.dera.paper_dera_detector import DeRaDetector

SF, N, OS, NF = 10, 1024, 4, 4096
DET = DeRaDetector(SF, OS)          # 只复用 signed_spectrum / 参考啁啾
N_FINE = 256                        # 列向 FFT 长度（port 同款）
GUARD = 16                          # 噪声列与期望列的最小距离（padded bin）
# FF_q：固定二次相位剪切 bank（cyc/符²，网格先验固定 → H0 可推导，
# 与 DeRa 数据驱动候选表的可证性差异即证书不对称性命题的对象）
Q_GRID = np.round(np.arange(-0.006, 0.0061, 0.0005), 6)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")
SEED_CONST = 20261003
LEVELS = [None] + [-v for v in (10, 14, 17, 20, 22, 24, 26, 28, 30, 32)]
N_SEEDS = 3
N_H0_PER_FRAME = 5                  # 每帧每档每种子的噪声窗数

SOURCES = [
    ("0_0_0_10_14_8",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
     r"\header_first\0_0_0_10_14_8_header_first_frames.csv", 8),
    ("0_0_0_10_14_16",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
     r"\header_first\0_0_0_10_14_16_header_first_frames.csv", 16),
    ("0_0_0_10_14_32",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_32.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
     r"\header_first\0_0_0_10_14_32_header_first_frames.csv", 32),
]


def fv(x):
    s = (x or "0").strip()
    return float(s.split("|")[0]) if s else 0.0


def snr_parts(seg):
    X = np.fft.fftshift(np.fft.fft(seg))
    p = np.abs(X) ** 2 / len(seg)
    f = np.fft.fftshift(np.fft.fftfreq(len(seg))) * 500000.0
    ob = (np.abs(f) > 70000) & (np.abs(f) < 240000)
    n0 = float(np.mean(p[ob]))
    total = float(np.mean(np.abs(seg) ** 2))
    return max(total - n0, 1e-30), n0


FRAMES = []          # [dict(cap, iq, hs, pre, snr(S,N0), noise_ok_ranges)]


def build_frames():
    frames = []
    for cap, bin_path, csv_path, pre in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1"
                and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            psym = int(r["payload_symbol_count"])
            res = demod_symbol_sequence(
                samples=np.asarray(iq, dtype=np.complex64),
                header_start_sample=int(r["header_start_sample"]), sf=SF,
                os_factor=OS,
                cfo_int=int(r["source_grlora_cfo_int"]),
                cfo_frac=fv(r.get("source_grlora_cfo_frac")),
                sfo_hat=fv(r.get("source_grlora_sfo_hat")),
                sfo_cum_initial=fv(r.get("source_grlora_branch_sfo_cum_initial")),
                header_count=8, payload_count=psym, payload_ldro=False)
            gt_hdr = [x.symbol_value for x in res[:8]]
            gt = [x.symbol_value for x in res[8:]]
            dec = decode_explicit_frame_symbols(gt_hdr, gt, sf=SF,
                                                bw=125000.0, ldro_mode=2)
            assert dec.header.header_valid and dec.payload.crc_valid
            hs = int(r["header_start_sample"])
            cal = np.asarray(iq[hs - (pre + 6) * NF: hs + 8 * NF],
                             dtype=np.complex128)
            # 活区功率参考（捕获文件存在大段零填充死区，H0 选窗须过滤）
            live = np.asarray(iq[hs - 4 * NF:hs], dtype=np.complex64)
            frames.append(dict(cap=cap, iq=iq, hs=hs, pre=pre,
                               snr=snr_parts(cal),
                               live_pw=float(np.mean(np.abs(live) ** 2)),
                               frame_list=None))
    # 每个 capture 的全部 hs（H0 噪声窗避让用）
    by_cap = {}
    for f in frames:
        by_cap.setdefault(f["cap"], []).append(f["hs"])
    for f in frames:
        f["all_hs"] = by_cap[f["cap"]]
    return frames


def field_start(hs: int, pre: int) -> int:
    """已知场首符号窗（前导自身符号网格，port 结构常量）。"""
    hs_sym = int(round(hs / NF))
    return (hs_sym - pre - 4) * NF - NF // 4


def chirp_windows(start: int, pre: int):
    """[(窗起点, 参考啁啾)]：j=0..pre-1 前导/sync 之前；j=pre,pre+1 sync
    upchirp；j=pre+2,pre+3 SFD downchirp。0.25 部分符号弃用。"""
    out = []
    for j in range(pre + 4):
        ref = DET.up_ref if j >= pre + 2 else DET.down_ref
        out.append((start + j * NF, ref))
    return out


def spectra(seg: np.ndarray, wins):
    return np.stack([DET.signed_spectrum(seg, s, ref) for s, ref in wins])


def far_mask(n_cols: int, k_list):
    m = np.ones(n_cols, dtype=bool)
    for k in k_list:
        m[max(0, k - GUARD):k + GUARD + 1] = False
    return m


def _interp(mag_row, idx):
    if idx <= 0 or idx >= len(mag_row) - 1:
        return float(idx)
    a, b, c = float(mag_row[idx - 1]), float(mag_row[idx]), \
        float(mag_row[idx + 1])
    d = a - 2.0 * b + c
    return idx + 0.5 * (a - c) / d if d != 0 else float(idx)


def frame_template(f):
    """干净信号上冻结：逐 chirp 期望列 k_j、干净复数值 c_j、downchirp
    共轭约定、真实走动斜率（诊断）。"""
    start = field_start(f["hs"], f["pre"])
    wins = chirp_windows(start, f["pre"])
    seg = np.asarray(f["iq"][wins[0][0]:wins[-1][0] + NF],
                     dtype=np.complex64)
    rows = spectra(seg, [(j * NF, ref) for j, (_, ref) in enumerate(wins)])
    k_j = [int(np.argmax(np.abs(r))) for r in rows]
    c_j = np.array([rows[j, k] for j, k in enumerate(k_j)],
                   dtype=np.complex128)
    # downchirp 相位约定：在干净全场上比较共轭与否的列向 FFT 相干性
    def _best_conv(conj_dn):
        z = c_j.copy()
        if conj_dn:
            z[-2:] = np.conj(z[-2:])
        Fq = np.abs(np.fft.fft(z, N_FINE))
        return float(np.max(Fq))
    conj_dn = _best_conv(True) > _best_conv(False)
    # 走动诊断：前导 upchirp 分数峰位斜率（bin/符，0.5 格 → bin 域 /2）
    pre_rows = rows[:f["pre"]]
    pos = np.array([_interp(np.abs(r), int(np.argmax(np.abs(r))))
                    for r in pre_rows]) / 2.0
    slope = float(np.polyfit(np.arange(f["pre"]), pos, 1)[0]) \
        if f["pre"] >= 4 else 0.0
    return dict(k_j=k_j, c_j=c_j, conj_dn=conj_dn, walk=slope,
                k_pre=int(np.bincount(k_j[:f["pre"]]).argmax()),
                phi_j=np.angle(c_j).tolist())


def unit_scores(seg_field, tmpl, sigma2):
    """五个变体的检测统计量（同一份带噪场谱、同一 σ̂²）。

    A_P    DeRa 式：仅前导 P chirp，单列，列向 FFT 线性相位搜索；
    FF_up  全场 upchirp（P 前导 + 2 sync），线性搜索；
    FF_lin 全场（含 2 SFD downchirp），线性搜索；
    FF_q   全场，固定二次相位 bank + 线性搜索（可部署剪切修复）；
    FF_tmpl 干净模板相位相干和（oracle：含二次漂移/分数 bin/约定项）。
    """
    rows = seg_field
    k_j, c_j = tmpl["k_j"], tmpl["c_j"]
    pre = len(k_j) - 4
    z_all = np.array([rows[j, k] for j, k in enumerate(k_j)],
                     dtype=np.complex128)
    if tmpl["conj_dn"]:
        z_all[-2:] = np.conj(z_all[-2:])
    z_up, z_pre = z_all[:pre + 2], z_all[:pre]
    j_idx = np.arange(len(z_all), dtype=np.float64)
    # A_P
    Fa = np.abs(np.fft.fft(z_pre, N_FINE))
    s_A = float(np.max(Fa) ** 2) / (pre * sigma2)
    # FF_up
    Fu = np.abs(np.fft.fft(z_up, N_FINE))
    s_up = float(np.max(Fu) ** 2) / (len(z_up) * sigma2)
    # FF_lin
    Ff = np.abs(np.fft.fft(z_all, N_FINE))
    s_FF = float(np.max(Ff) ** 2) / (len(z_all) * sigma2)
    # FF_q：固定剪切 bank（二次项先扣再线性搜索）
    s_q = 0.0
    for c in Q_GRID:
        zq = z_all * np.exp(-1j * c * j_idx ** 2)
        Fq = np.abs(np.fft.fft(zq, N_FINE))
        v = float(np.max(Fq) ** 2)
        if v > s_q:
            s_q = v
    s_q /= (len(z_all) * sigma2)
    # FF_ml：二次 bank + 公共线性斜率 q + 类型偏移边缘化（pre/sync/down
    # 三段幅度相加 = 未知确定性偏移的 ML；变换族固定 → H0 可推导）
    s_ml = 0.0
    for c in Q_GRID:
        zq = z_all * np.exp(-1j * c * j_idx ** 2)
        Fpre = np.abs(np.fft.fft(zq[:pre], N_FINE))
        Fsyn = np.abs(np.fft.fft(zq[pre:pre + 2], N_FINE))
        Fdn = np.abs(np.fft.fft(zq[pre + 2:], N_FINE))
        v = float(np.max(Fpre + Fsyn + Fdn))
        if v > s_ml:
            s_ml = v
    s_ml = s_ml ** 2 / (len(z_all) * sigma2)
    # FF_tmpl
    ph = c_j / np.abs(c_j)
    if tmpl["conj_dn"]:
        ph = ph.copy()
        ph[-2:] = np.conj(ph[-2:])
    s_tmpl = float(np.abs(np.sum(z_all * np.conj(ph))) ** 2) \
        / (len(z_all) * sigma2)
    return dict(A_P=s_A, FF_up=s_up, FF_lin=s_FF, FF_q=s_q, FF_ml=s_ml,
                FF_tmpl=s_tmpl)


def make_noisy(f, level, seed, rng_salt, lo, hi):
    seg = np.asarray(f["iq"][lo:hi], dtype=np.complex128)
    if level is not None:
        S, N0 = f["snr"]
        rng = np.random.default_rng((SEED_CONST * 7919
                                     + (int(level) + 100) * 131
                                     + seed * 17 + rng_salt * 7919)
                                    % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) \
            * np.sqrt(p_add / 2.0)
    return seg


def h0_centers(f, n, level, seed):
    """噪声窗中心（避开所有帧 ±2(P+6)NF）。"""
    rng = np.random.default_rng((SEED_CONST * 31 + (int(level or 0) + 100) * 7
                                 + seed * 13 + f["hs"] % 9973) % (2 ** 31))
    guard = 2 * (f["pre"] + 6) * NF
    lo_b, hi_b = (f["pre"] + 8) * NF, len(f["iq"]) - (f["pre"] + 8) * NF
    out, tries = [], 0
    while len(out) < n and tries < 400:
        tries += 1
        c = int(rng.integers(lo_b, hi_b))
        if all(abs(c - h) > guard for h in f["all_hs"]):
            blk = np.asarray(f["iq"][c - NF:c + NF], dtype=np.complex64)
            if float(np.mean(np.abs(blk) ** 2)) < 0.05 * f["live_pw"]:
                continue          # 零填充死区
            out.append(c)
    return out


def eval_unit(f, tmpl, level, seed):
    """一个 (帧, 档, 种子)：H1（帧位置）+ N_H0 个噪声窗。"""
    pre = f["pre"]
    start = field_start(f["hs"], pre)
    span = (pre + 4) * NF
    recs = []
    # ---- H1 ----
    seg = make_noisy(f, level, seed, f["hs"] % 4099,
                     start - NF, start + span + NF)
    rows = spectra(seg, chirp_windows(NF, pre))   # 相对窗起点=NF
    m = far_mask(rows.shape[1], sorted(set(tmpl["k_j"])))
    sig2 = float(np.median(np.abs(rows[:, m]) ** 2))
    sc = unit_scores(rows, tmpl, sig2)
    recs.append(dict(kind="h1", level=level, seed=seed, cap=f["cap"],
                     pre=pre, scores=sc, sigma2=sig2,
                     walk=tmpl["walk"], conj_dn=tmpl["conj_dn"],
                     phi_j=tmpl["phi_j"], k_j=tmpl["k_j"]))
    # ---- H0 ----
    for ci, c in enumerate(h0_centers(f, N_H0_PER_FRAME, level, seed)):
        # 噪声窗用与 H1 相同的期望列/相位模板（结构已知的二择检测）
        seg0 = make_noisy(f, level, seed, (f["hs"] + c) % 4099,
                          c - NF, c + span + NF)
        rows0 = spectra(seg0, chirp_windows(NF, pre))
        sig20 = float(np.median(np.abs(rows0[:, m]) ** 2))
        if sig20 <= 0:
            continue
        sc0 = unit_scores(rows0, tmpl, sig20)
        recs.append(dict(kind="h0", level=level, seed=seed, cap=f["cap"],
                         pre=pre, scores=sc0, sigma2=sig20,
                         walk=tmpl["walk"], conj_dn=tmpl["conj_dn"]))
    return recs


def main():
    t0 = time.time()
    global FRAMES
    print("冻结帧集与 GT（原生解 + CRC assert）…", flush=True)
    FRAMES = build_frames()
    print("帧集：%d" % len(FRAMES), flush=True)
    TMPL = [frame_template(f) for f in FRAMES]
    for i, t in enumerate(TMPL):
        print("  frame%02d %s P=%d walk=%+.4f bin/sym conj_dn=%d"
              % (i, FRAMES[i]["cap"], FRAMES[i]["pre"], t["walk"],
                 t["conj_dn"]), flush=True)

    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["kind"], r["level"], r["seed"],
                          r["cap"], r["frame"]))
            except Exception:
                pass
        print("断点恢复：跳过 %d 记录" % len(done), flush=True)

    units = []
    for lv in LEVELS:
        for sd in range(N_SEEDS if lv is not None else 1):
            for fi in range(len(FRAMES)):
                units.append((lv, sd, fi))
    n_written = 0
    with open(CKPT, "a", encoding="utf-8") as fh:
        for lv, sd, fi in units:
            key0 = ("h1", lv, sd, FRAMES[fi]["cap"], fi)
            if key0 in done and ("h0", lv, sd, FRAMES[fi]["cap"], fi) in done:
                continue
            for r in eval_unit(FRAMES[fi], TMPL[fi], lv, sd):
                r["frame"] = fi
                if (r["kind"], r["level"], r["seed"], r["cap"], fi) in done:
                    continue
                fh.write(json.dumps(r) + "\n")
                n_written += 1
            n_written and fh.flush()
            if n_written and n_written % 500 == 0:
                print("  %d 条 (%.0fs)" % (n_written, time.time() - t0),
                      flush=True)
    print("完成：%d 新记录，总耗时 %.0fs" % (n_written, time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()

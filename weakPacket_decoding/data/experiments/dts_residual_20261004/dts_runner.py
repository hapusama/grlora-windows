# -*- coding: utf-8 -*-
r"""DTS-1 残差容忍度扫描 runner（2026-10-04，Decoding-Tolerant Sync 第一刀）。

问题：sync 没做好（残差 ν_frac/τ/φ/κ）对 decoding 的伤害有多大？
方法：实验C 注入型（EXPERIMENT_PROTOCOL v1.1 §5A.4）——在实验B 冻结先验
对齐段上叠加受控残差 ε，所有链吃同一段带残差信号（公平输入铁律保持），
噪声仍整包 AWGN。产出 SER(ε) 容忍度曲线 → dB 等价损失传递函数。

残差轴（ε 注入 = "同步没做好"的受控参数化）：
  nu   : 分数 CFO 残差，ε bin（段乘 exp(2jπ·ε·n/NF)）
  tau  : 亚 chip STO 残差，ε chip（frac_delay(ε·OS)）
  phi  : 全段固定相位残差，ε 度（只打击相干合并链，如实报告）
  kappa: 逐符号线性 bin 走动残差，ε bin/符（第 k 符号频偏 ε·k bin；
         bin-walk 参数化，非完整 SFO 物理；对齐 A3 实测 0.002~0.24e-1 界）

链：PLAIN / OLD-A / TRIMMER(est-compensate) / DERA(est-compensate) / TREL-5
     (marginalize-soft)。容忍度不对称命题的观测对象 = 后两族。
δ 冻结于干净信号（实验B 规则）；残差注入后不重估，逐符号翻转如实计入。
模板：dera_battle_20260929/battle_runner.py（帧构建/谱行/判据逐行复用，
native ε=0 臂 = 与实验B 战表的一致性检查）。
"""
import sys
import csv
import json
import os
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, SymFECSymbolEvidence, build_symfec_symbol_evidence,
    decode_symfec_payload_from_evidences)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator
from weak_decoder.chirp import build_upchirp

SF, N, OS, NF = 10, 1024, 4, 4096
FEC_CFG = SymFECConfig()
ROT5 = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)])
MASK = np.abs(np.fft.fftfreq(NF)) <= 0.125 + 40.0 / NF
REF_OS = np.conj(build_upchirp(SF, symbol_id=0, os_factor=OS))
DERA = DeRaDemodulator(SF, OS)

CHAINS = ["PLAIN", "OLD-A", "TRIMMER", "DERA", "TREL-5"]

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dts_residual_20261004"
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

# 残差轴 × 档位（冒烟 = nu 轴；其余轴脚本已实现，逐轴扩展）
AXES = {
    "nu":    [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, -0.05, -0.1, -0.2, -0.3, -0.5],
    "tau":   [0.0, 0.125, 0.25, 0.375, 0.5],
    "phi":   [0.0, 15.0, 30.0, 45.0],
    "kappa": [0.0, 0.005, 0.01, 0.02],
}
LEVELS = [None, -20]   # native + 工作区档
SEEDS = [0]            # 冒烟 1 种子；扩展 3 种子

SOURCES = [
    ("0_0_0_10_14_8",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_8_header_first_frames.csv"),
    ("0_0_0_10_14_16",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_frames.csv"),
    ("0_0_0_10_14_32",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_32.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_32_header_first_frames.csv"),
]

G = {}


def frac_delay(x, samples):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x))
    return np.fft.ifft(X * np.exp(-2j * np.pi * f * samples))


def seg_aligned(iq, r, psym):
    hs = int(r["header_start_sample"])
    seg = np.asarray(iq[hs - 8 * NF: hs + (8 + psym + 1) * NF], dtype=np.complex128)
    n = np.arange(len(seg))
    f = int(r["source_grlora_cfo_int"]) + float(r["source_grlora_cfo_frac"])
    seg = seg * np.exp(-2j * np.pi * f * n / NF)
    return frac_delay(seg, -float(r["source_grlora_payload_sto_frac"]) * OS)


def apply_residual(seg, axis, eps):
    """在冻结先验对齐段上叠加受控同步残差（所有链同一段，公平输入）。"""
    if eps == 0.0:
        return seg
    n = np.arange(len(seg))
    if axis == "nu":
        return seg * np.exp(2j * np.pi * eps * n / NF)
    if axis == "tau":
        return frac_delay(seg, eps * OS)
    if axis == "phi":
        return seg * np.exp(1j * np.deg2rad(eps))
    if axis == "kappa":
        out = seg.copy()
        for k in range(len(seg) // NF):
            s = k * NF
            nl = np.arange(NF)
            out[s:s + NF] = seg[s:s + NF] * np.exp(2j * np.pi * (eps * k) * nl / NF)
        return out
    raise ValueError(axis)


def snr_parts(seg):
    X = np.fft.fftshift(np.fft.fft(seg))
    p = np.abs(X) ** 2 / len(seg)
    f = np.fft.fftshift(np.fft.fftfreq(len(seg))) * 500000.0
    ob = (np.abs(f) > 70000) & (np.abs(f) < 240000)
    n0 = float(np.mean(p[ob]))
    total = float(np.mean(np.abs(seg) ** 2))
    return max(total - n0, 1e-30), n0


def wm(x, k):
    seg = np.fft.ifft(np.fft.fft(x[k * NF:(k + 1) * NF]) * MASK)
    y = seg * REF_OS
    chips = np.stack([y[p:p + NF:OS] for p in range(OS)])
    inp = (chips[None, :, :] * ROT5[:, None, :]).reshape(20, N)
    P = np.abs(np.fft.fft(inp, axis=1)) ** 2
    return P.reshape(5, 4, N).mean(axis=1).T


def olda_rows(seg, k):
    s = seg[k * NF:(k + 1) * NF].copy()
    return np.abs(np.fft.fft(s[::OS] * REF_OS[::OS])) ** 2


def _lse(x):
    m = x.max()
    return m + np.log(np.sum(np.exp(x - m)))


def viterbi_path(lam, radius=1):
    K, D = lam.shape
    acc = lam[0].copy()
    back = np.zeros((K, D), dtype=int)
    for k in range(1, K):
        cand = np.stack([acc[np.clip(np.arange(D) + s, 0, D - 1)]
                         for s in range(-radius, radius + 1)])
        best = np.argmax(cand, axis=0)
        back[k] = best - radius
        acc = cand[best, np.arange(D)] + lam[k]
    path = np.zeros(K, dtype=int)
    path[-1] = int(np.argmax(acc))
    for k in range(K - 1, 0, -1):
        path[k - 1] = np.clip(path[k] + back[k, path[k]], 0, D - 1)
    return path


def fast_evidence(power_row, symbol_index, floor_db=30.0, top_count=8):
    db = 10.0 * np.log10(np.maximum(power_row, 1e-30))
    rel = np.maximum(db - float(np.max(db)), -abs(floor_db))
    score_by_value = np.roll(rel, -1)
    best_raw = np.roll(np.arange(N), -1)
    argmax_bin = int(np.argmax(rel))
    order = np.argsort(rel)[::-1][:max(1, int(top_count))]
    return SymFECSymbolEvidence(
        symbol_index=int(symbol_index), raw_scores=rel,
        score_by_value=score_by_value, best_raw_bin_by_value=best_raw,
        argmax_raw_bin=argmax_bin,
        argmax_symbol_value=(argmax_bin - 1) % N,
        peak_margin_db=float(rel[order[0]] - rel[order[1]]) if N > 1 else 0.0,
        top_raw_bins=tuple(int(b) for b in order),
        top_symbol_values=tuple((int(b) - 1) % N for b in order))


def judge_crc_fast(rows_bin, delta, gt_hdr, plen, cr_i):
    rc = np.roll(rows_bin, -(int(delta) - 1), axis=1)
    evs = [fast_evidence(rc[k], k) for k in range(rc.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=cr_i, ldro=False, config=FEC_CFG,
        header_symbol_values=list(gt_hdr), payload_len=plen,
        has_crc=True, crc_mode="grlora")
    if res.payload_decode is None:
        return False
    return bool(res.payload_decode.crc_valid)


def chain_rows(seg, psym):
    rows = {}
    rows["OLD-A"] = np.stack([olda_rows(seg, 16 + k) for k in range(psym)])
    tri = np.zeros((psym, N))
    for k in range(psym):
        st = (16 + k) * NF
        tri[k] = trim_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                            cfo_int=0).metric
    rows["TRIMMER"] = tri
    _s1d, _cohd = DERA.demod_payload(seg, 16, psym)
    rows["DERA"] = _cohd
    ms = [wm(seg, 16 + k) for k in range(psym)]
    lam = np.array([np.log(np.maximum(m.max(axis=0) - np.median(m, axis=0), 1e-30))
                    for m in ms])
    path = viterbi_path(lam)
    rows["TREL-5"] = np.stack([ms[k][:, path[k]] for k in range(psym)])
    return rows


def build_frames():
    frames = []
    for cap, bin_path, csv_path in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            psym = int(r["payload_symbol_count"])
            res = demod_symbol_sequence(
                samples=np.asarray(iq, dtype=np.complex64),
                header_start_sample=int(r["header_start_sample"]), sf=SF, os_factor=OS,
                cfo_int=int(r["source_grlora_cfo_int"]),
                cfo_frac=float(r["source_grlora_cfo_frac"]),
                sfo_hat=float(r["source_grlora_sfo_hat"]),
                sfo_cum_initial=float(r.get("source_grlora_branch_sfo_cum_initial") or 0.0),
                header_count=8, payload_count=psym, payload_ldro=False)
            gt_hdr = [x.symbol_value for x in res[:8]]
            gt = [x.symbol_value for x in res[8:]]
            dec = decode_explicit_frame_symbols(gt_hdr, gt, sf=SF, bw=125000.0, ldro_mode=2)
            assert dec.header.header_valid and dec.payload.crc_valid
            seg = seg_aligned(iq, r, psym)
            S, N0 = snr_parts(seg)
            ms = [wm(seg, 16 + k) for k in range(psym)]
            rows_now = np.stack([m[:, 2] for m in ms])
            d = [(int(np.argmax(rows_now[k])) - g) % N for k, g in enumerate(gt)]
            delta = int(np.bincount(d).argmax())
            frames.append(dict(gt_hdr=gt_hdr, gt=gt, seg=seg, S=S, N0=N0,
                               psym=psym, plen=int(r["payload_len"]),
                               cr=int(r["cr"]), delta=delta))
    return frames


def init_worker():
    global G
    G["frames"] = build_frames()
    rng = np.random.default_rng(0)
    row = rng.exponential(1.0, N)
    a = fast_evidence(row, 0)
    b = build_symfec_symbol_evidence(row, sf=SF, symbol_index=0, ldro=False)
    assert np.allclose(a.score_by_value, b.score_by_value), "快证据不一致"
    assert np.allclose(a.best_raw_bin_by_value % N, b.best_raw_bin_by_value % N)


def run_unit(u):
    axis, eps, level, seed, fi = u
    f = G["frames"][fi]
    # 残差先施加于干净段（信号属性），噪声后加（AWGN 铁律不变）
    seg = apply_residual(f["seg"], axis, eps)
    if level is not None:
        rng = np.random.default_rng((20261004 * 7919 + (int(level) + 100) * 131
                                     + int(round(eps * 1000)) * 37
                                     + seed * 17 + fi) % (2 ** 31))
        p_add = max(f["S"] / 10 ** (level / 10.0) - f["N0"], 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    rows_ = chain_rows(seg, f["psym"])
    out = {}
    for c in CHAINS:
        cc = f["delta"]
        rc = rows_["OLD-A"] if c == "PLAIN" else rows_[c]
        hard = [(int(np.argmax(rc[k])) - cc) % N for k in range(f["psym"])]
        sym_err = int(sum(int(h != g) for h, g in zip(hard, f["gt"])))
        if c == "PLAIN":
            try:
                dec = decode_explicit_frame_symbols(f["gt_hdr"], hard, sf=SF,
                                                    bw=125000.0, ldro_mode=2)
                crc_ok = bool(dec.header.header_valid and dec.payload.crc_valid)
            except Exception:
                crc_ok = False
        else:
            crc_ok = judge_crc_fast(rc, cc, f["gt_hdr"], f["plen"], f["cr"])
        out[c] = {"sym_err": sym_err, "crc_fail": int(not crc_ok)}
    return {"axis": axis, "eps": eps, "level": level, "seed": seed, "frame": fi,
            "sym_tot": f["psym"], "chains": out}


def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["axis"], r["eps"], r["level"], r["seed"], r["frame"]))
            except Exception:
                pass
        print("断点恢复：已有 %d 个完成单元" % len(done), flush=True)

    axes = AXES if "--all-axes" in sys.argv else {"nu": AXES["nu"]}
    units = []
    for ax, eps_list in axes.items():
        for eps in eps_list:
            for lv in LEVELS:
                for sd in SEEDS:
                    for fi in range(28):
                        u = (ax, eps, lv, sd, fi)
                        if u not in done:
                            units.append(u)
    print("待跑 %d 单元（%d workers）" % (len(units), os.cpu_count()), flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, os.cpu_count() - 1),
                  initializer=init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units, chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 100 == 0:
                    print("  %d/%d 单元完成 (%.0fs)" % (i + 1, len(units),
                                                        time.time() - t0), flush=True)

    # 汇总：SER/PER per (axis, eps, level) per chain
    agg = {}
    pk = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        lk = "native" if r["level"] is None else "%+d" % r["level"]
        key = (r["axis"], r["eps"], lk)
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["sym_tot"]
            a[c][2] += r["chains"][c]["crc_fail"]
        pk[key] = pk.get(key, 0) + 1
    print("\nDTS-1 容忍度表（SER；n=包数）")
    for (ax, eps, lk) in sorted(agg, key=lambda k: (k[0], abs(k[1]), k[2])):
        a = agg[(ax, eps, lk)]
        parts = " | ".join("%s %.3f" % (c, a[c][0] / max(a[c][1], 1)) for c in CHAINS)
        print("[%5s ε=%+.3f %-6s] %s (n=%d)" % (ax, eps, lk, parts, pk[(ax, eps, lk)]),
              flush=True)
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

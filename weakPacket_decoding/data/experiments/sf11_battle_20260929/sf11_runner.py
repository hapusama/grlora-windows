# -*- coding: utf-8 -*-
r"""2026-09-29 SF11 围殴战（runner v2 参数化版，lab1_sf11_TP2 全量 16 文件）。

与 dera_battle 的 battle_runner.py 同构，仅参数与数据源不同：
SF11 / N=2048 / NF=8192 / preamble 16（payload 偏移 = 16+8 = 24）。
铁律不变：纯 AWGN、同一实现喂所有链、全先验对齐（int+frac CFO +
STO 亚 chip 时延）、整包 SNR 口径、δ 干净信号冻结、断点续跑。
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
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.unichirp.paper_unichirp_demod import (
    UniChirpTrainingSymbol, UniChirpDemodConfig, build_unichirp_phase_model,
    demod_unichirp_symbol as uni_demod, unichirp_full_spectrum)
from weak_decoder.chirp import build_upchirp

SF, N, OS, NF = 11, 2048, 4, 8192
PRE = 16                       # 段内前导区符号数（preamble 16 + sync/down 组合窗）
OFF_PAY = PRE + 8              # payload 起始符号 = 前导区 + 8 header
FEC_CFG = SymFECConfig()
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")
ROT5 = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)])
MASK = np.abs(np.fft.fftfreq(NF)) <= 0.125 + 40.0 / NF
REF_OS = np.conj(build_upchirp(SF, symbol_id=0, os_factor=OS))

CHAINS = ["PLAIN", "OLD-A", "TRIMMER", "SAVAUX", "NEW-0", "TREL-5", "BCJR-5"]

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\sf11_battle_20260929"
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

_IQ_DIR = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\lab1_sf11_TP2"
_HF = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first"
SOURCES = [("1_0_%d_11_2_16" % p,
            _IQ_DIR + r"\1_0_%d_11_2_16.bin" % p,
            _HF + r"\1_0_%d_11_2_16_header_first_frames.csv" % p)
           for p in range(8, 17)] + \
          [("1_1_%d_11_2_16" % p,
            _IQ_DIR + r"\1_1_%d_11_2_16.bin" % p,
            _HF + r"\1_1_%d_11_2_16_header_first_frames.csv" % p)
           for p in range(0, 8)]

G = {}


def fv(x):
    """CSV 数值字段安全解析：SF11 的 branch_* 列为 4 相位多值（首值=有效支路）。"""
    s = (x or "0").strip()
    if not s:
        return 0.0
    return float(s.split("|")[0])


def frac_delay(x, samples):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x))
    return np.fft.ifft(X * np.exp(-2j * np.pi * f * samples))


def seg_aligned(iq, r, psym):
    hs = int(r["header_start_sample"])
    seg = np.asarray(iq[hs - PRE * NF: hs + (8 + psym + 1) * NF], dtype=np.complex128)
    n = np.arange(len(seg))
    f = int(r["source_grlora_cfo_int"]) + fv(r.get("source_grlora_cfo_frac"))
    seg = seg * np.exp(-2j * np.pi * f * n / NF)
    return frac_delay(seg, -fv(r.get("source_grlora_payload_sto_frac")) * OS)


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


def fb_post(lam, radius=1):
    K, D = lam.shape
    alpha = np.zeros((K, D))
    alpha[0] = lam[0]
    beta = np.zeros((K, D))
    for k in range(1, K):
        for d in range(D):
            lo, hi = max(0, d - radius), min(D - 1, d + radius)
            alpha[k, d] = lam[k, d] + _lse(alpha[k - 1, lo:hi + 1])
            beta[k - 1, d] = _lse(lam[k, lo:hi + 1] + beta[k, lo:hi + 1])
    post = alpha + beta
    post = np.exp(post - post.max(axis=1, keepdims=True))
    return post / post.sum(axis=1, keepdims=True)


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
    """LDRO payload 证据：值 v 占 bins {4v+1..4v+4}（v=511 另含 bin0）。

    与逐 bin 版 build_symfec_symbol_evidence(ldro=True) 数值一致：
    score_by_value[v] = 该 4-bin 组内 rel 的最大值。
    """
    db = 10.0 * np.log10(np.maximum(power_row, 1e-30))
    rel = np.maximum(db - float(np.max(db)), -abs(floor_db))
    n_val = N // 4                     # LDRO 值域 = 512
    score = np.full(n_val, -np.inf)
    score[:n_val - 1] = rel[1: 4 * (n_val - 1) + 1].reshape(n_val - 1, 4).max(axis=1)
    score[n_val - 1] = max(float(rel[4 * (n_val - 1) + 1:].max()), float(rel[0]))
    best = np.full(n_val, -1, dtype=np.int64)
    best[:n_val - 1] = (np.arange(n_val - 1) * 4 + 1
                        + np.argmax(rel[1: 4 * (n_val - 1) + 1].reshape(n_val - 1, 4), axis=1))
    tail = rel[4 * (n_val - 1) + 1:]
    tail_bins = np.concatenate([np.arange(4 * (n_val - 1) + 1, N), [0]])
    best[n_val - 1] = int(tail_bins[np.argmax(np.concatenate([tail, rel[:1]]))])
    argmax_bin = int(np.argmax(rel))
    order = np.argsort(rel)[::-1][:max(1, int(top_count))]
    return SymFECSymbolEvidence(
        symbol_index=int(symbol_index), raw_scores=rel,
        score_by_value=score, best_raw_bin_by_value=best,
        argmax_raw_bin=argmax_bin,
        argmax_symbol_value=((argmax_bin - 1) % N) // 4,
        peak_margin_db=float(rel[order[0]] - rel[order[1]]) if N > 1 else 0.0,
        top_raw_bins=tuple(int(b) for b in order),
        top_symbol_values=tuple(((int(b) - 1) % N) // 4 for b in order))


def judge_crc_fast(rows_bin, delta, gt_hdr, plen, cr_i):
    """LDRO 判据：bin 级 roll(−(δ−1)) 使音落在值格 4v+2 附近，再 4-bin 组最大。"""
    rc = np.roll(rows_bin, -(int(delta) - 1), axis=1)
    evs = [fast_evidence(rc[k], k) for k in range(rc.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=cr_i, ldro=True, config=FEC_CFG,
        header_symbol_values=list(gt_hdr), payload_len=plen,
        has_crc=True, crc_mode="grlora")
    if res.payload_decode is None:
        return False
    return bool(res.payload_decode.crc_valid)


def uni_train(seg):
    train = []
    for k in range(4):
        full = unichirp_full_spectrum(samples=seg, start_sample=k * NF, sf=SF,
                                      os_factor=OS, cfo_int=0, cfo_frac=0.0,
                                      config=UNI_CFG)
        train.append(UniChirpTrainingSymbol(start_sample=k * NF,
                                            raw_fft_bin=int(np.argmax(full[:N])),
                                            abs_symbol_index=float(k)))
    model, _o = build_unichirp_phase_model(samples=seg, training_symbols=tuple(train),
                                           sf=SF, os_factor=OS, config=UNI_CFG)
    return model


def chain_rows(seg, psym):
    rows = {}
    rows["OLD-A"] = np.stack([olda_rows(seg, OFF_PAY + k) for k in range(psym)])
    tri = np.zeros((psym, N)); sav = np.zeros((psym, N))
    for k in range(psym):
        st = (OFF_PAY + k) * NF
        tri[k] = trim_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                            cfo_int=0).metric
        sav[k] = np.abs(sav_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                                  cfo_int=0).combined_spectrum) ** 2
    rows["TRIMMER"] = tri
    rows["SAVAUX"] = sav
    uni = np.zeros((psym, N))
    uni_m = uni_train(seg)
    for k in range(psym):
        uni[k] = uni_demod(samples=seg, start_sample=(OFF_PAY + k) * NF, sf=SF,
                           os_factor=OS, phase_rad=uni_m.predict(OFF_PAY + k),
                           config=UNI_CFG).metric
    rows["UNICHIRP_SMOKE"] = uni  # 不入 CHAINS，仅诊断
    ms = [wm(seg, OFF_PAY + k) for k in range(psym)]
    rows["NEW-0"] = np.stack([m[:, 2] for m in ms])
    lam = np.array([np.log(np.maximum(m.max(axis=0) - np.median(m, axis=0), 1e-30))
                    for m in ms])
    path = viterbi_path(lam)
    rows["TREL-5"] = np.stack([ms[k][:, path[k]] for k in range(psym)])
    post = fb_post(lam)
    rows["BCJR-5"] = np.stack([ms[k] @ post[k] for k in range(psym)])
    return rows


def build_frames():
    frames = []
    for cap, bin_path, csv_path in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            psym = int(r["payload_symbol_count"])
            try:
                res = demod_symbol_sequence(
                    samples=np.asarray(iq, dtype=np.complex64),
                    header_start_sample=int(r["header_start_sample"]), sf=SF, os_factor=OS,
                    cfo_int=int(r["source_grlora_cfo_int"]),
                    cfo_frac=fv(r.get("source_grlora_cfo_frac")),
                    sfo_hat=fv(r.get("source_grlora_sfo_hat")),
                    sfo_cum_initial=fv(r.get("source_grlora_branch_sfo_cum_initial")),
                    header_count=8, payload_count=psym, payload_ldro=True)
                gt_hdr = [x.symbol_value for x in res[:8]]
                gt = [x.symbol_value for x in res[8:]]
                dec = decode_explicit_frame_symbols(gt_hdr, gt, sf=SF, bw=125000.0,
                                                    ldro_mode=1)
                if not (dec.header.header_valid and dec.payload.crc_valid):
                    raise ValueError("GT CRC 未过")
            except Exception:
                # 弱功率捕获：原生解不出的帧无 GT，跳过（计数见 symbols 侧日志）
                continue
            seg = seg_aligned(iq, r, psym)
            S, N0 = snr_parts(seg)
            ms = [wm(seg, OFF_PAY + k) for k in range(psym)]
            rows_now = np.stack([m[:, 2] for m in ms])
            # δ 冻结（bin 级）：LDRO 值 v 的音在 bin 4v+1，偏移 = argmax−(4v+1)
            d = [(int(np.argmax(rows_now[k])) - (4 * g + 1)) % N
                 for k, g in enumerate(gt)]
            delta = int(np.bincount(d).argmax())
            frames.append(dict(gt_hdr=gt_hdr, gt=gt,
                               seg=seg.astype(np.complex64), S=S, N0=N0,
                               psym=psym, plen=int(r["payload_len"]),
                               cr=int(r["cr"]), delta=delta))
    return frames


def count_frames():
    n = 0
    for _cap, _b, csv_path in SOURCES:
        if not os.path.exists(csv_path):
            continue
        n += sum(1 for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                 if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0)
    return n


def init_worker():
    global G
    G["frames"] = build_frames()
    rng = np.random.default_rng(0)
    row = rng.exponential(1.0, N)
    a = fast_evidence(row, 0)
    b = build_symfec_symbol_evidence(row, sf=SF, symbol_index=0, ldro=True)
    assert np.allclose(a.score_by_value, b.score_by_value), "快证据(LDRO)不一致"
    assert np.allclose(a.best_raw_bin_by_value % N, b.best_raw_bin_by_value % N)


def run_unit(u):
    level, seed, fi = u
    f = G["frames"][fi]
    seg = f["seg"].astype(np.complex128)
    if level is not None:
        rng = np.random.default_rng((20260929 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919) % (2 ** 31))
        p_add = max(f["S"] / 10 ** (level / 10.0) - f["N0"], 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    rows_ = chain_rows(seg, f["psym"])
    out = {}
    for c in CHAINS:
        cc = f["delta"]
        rc = rows_["OLD-A"] if c == "PLAIN" else rows_[c]
        # bin 级 argmax -> LDRO 值：((b - δ - 1) mod N) // 4
        hard = [(((int(np.argmax(rc[k])) - cc - 1) % N) // 4) for k in range(f["psym"])]
        sym_err = int(sum(int(h != g) for h, g in zip(hard, f["gt"])))
        if c == "PLAIN":
            try:
                dec = decode_explicit_frame_symbols(f["gt_hdr"], hard, sf=SF,
                                                    bw=125000.0, ldro_mode=1)
                crc_ok = bool(dec.header.header_valid and dec.payload.crc_valid)
            except Exception:
                crc_ok = False
        else:
            crc_ok = judge_crc_fast(rc, cc, f["gt_hdr"], f["plen"], f["cr"])
        out[c] = {"sym_err": sym_err, "crc_fail": int(not crc_ok)}
    return {"level": level, "seed": seed, "frame": fi,
            "sym_tot": f["psym"], "chains": out}


def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)
    # 主进程先 build 一遍拿确定性帧数（与 worker 的 build_frames 完全一致）
    n_frames = len(build_frames())
    print("SF11 有效帧数（原生 GT 可解 + CRC 通过）: %d" % n_frames, flush=True)
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["level"], r["seed"], r["frame"]))
            except Exception:
                pass
        print("断点恢复：已有 %d 个完成单元" % len(done), flush=True)
    units = [(None, 0, fi) for fi in range(n_frames)]
    for lv in range(-16, -27, -1):
        for sd in range(3):
            for fi in range(n_frames):
                units.append((lv, sd, fi))
    units = [u for u in units if (u[0], u[1], u[2]) not in done]
    print("待跑 %d 单元（%d workers）" % (len(units), os.cpu_count()), flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=6, initializer=init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units, chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 168 == 0:
                    print("  %d/%d 单元完成 (%.0fs)" % (i + 1, len(units),
                                                        time.time() - t0), flush=True)

    agg = {}
    counts = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        counts[key] = counts.get(key, 0) + 1
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["sym_tot"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\nSF11 战表（SER / PER）")
    for key in ["native"] + ["%+d" % v for v in range(-16, -27, -1)]:
        if key not in agg:
            continue
        a = agg[key]
        n_pkt = counts[key]
        parts = " | ".join("%s %.3f/%.3f" % (c, a[c][0] / max(a[c][1], 1),
                                             a[c][2] / n_pkt) for c in CHAINS)
        print("[%8s] %s  (n=%d)" % (key, parts, n_pkt), flush=True)
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""2026-09-29 实验一：同步给定，只比 demod + bin selection（自包含、stdout 版）。

铁律（用户 2026-09-29 三令）：噪声有且仅有 AWGN（复高斯白，直接加在真实
基带信号上，同一实现喂所有链）；无任何利好我方的窗口错位/色噪声/注入。

设计：
  - 每帧先验（来自 framesync CSV）全量统一施加：整数+分数 CFO 频偏 +
    STO 亚 chip 分数时延 delay=-sto*OS —— 全链同一段对齐后信号。
  - 每链每帧整 bin 映射 δ 在干净信号上冻结（同步先验的一部分；噪声档
    不再调整；逐符号与 δ 不一致的翻转如实计入 SER）。
  - 噪声按目标【整包加噪后 SNR】逐包标定（S=段功率-带外噪声底，
    N_post=N0+P_add，全 500kHz 分析带宽）。native + +1 到 -26 dB、
    1dB 一档、3 种子。
  - SER = bin 选择(argmax)对 GT 的错误率（demod 级）；PER = 整包
    CRC16 未过（已修复判据）。结果打印到 stdout（RESULT_JSON_BEGIN
    与 RESULT_JSON_END 之间为完整 JSON；本次运行已另存
    results_exp1.json，重跑可复核）。
依赖的解调原语（wm/olda/bcjr/roll_evidence）自 ota_v6_fixed.py 内联，
约定与该提交版逐字一致。
"""
import sys
import csv
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, build_symfec_symbol_evidence, decode_symfec_payload_from_evidences)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.unichirp.paper_unichirp_demod import (
    UniChirpTrainingSymbol, UniChirpDemodConfig, build_unichirp_phase_model,
    demod_unichirp_symbol as uni_demod, unichirp_full_spectrum)
from weak_decoder.chirp import build_upchirp

SF, N, OS, NF = 10, 1024, 4, 4096
FEC_CFG = SymFECConfig()
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")
ROT = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)])
MASK = np.abs(np.fft.fftfreq(NF)) <= 0.125 + 40.0 / NF
REF_OS = np.conj(build_upchirp(SF, symbol_id=0, os_factor=OS))

CHAINS = ["PLAIN", "OLD-A", "SAVAUX", "TRIMMER", "UNICHIRP", "NEW-0", "BCJR"]
EV_CHAINS = ["OLD-A", "SAVAUX", "TRIMMER", "UNICHIRP", "NEW-0", "BCJR"]

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


def wm(x, k):
    """(N,5) 细网格窗口能量；列 d ↔ κ=(2-d)/4。与 ota_v6_fixed 逐字一致。"""
    seg = np.fft.ifft(np.fft.fft(x[k * NF:(k + 1) * NF]) * MASK)
    y = seg * REF_OS
    chips = np.stack([y[p:p + NF:OS] for p in range(OS)])
    inp = (chips[None, :, :] * ROT[:, None, :]).reshape(20, N)
    P = np.abs(np.fft.fft(inp, axis=1)) ** 2
    return P.reshape(5, 4, N).mean(axis=1).T


def olda_rows(seg, k, prefiltered=False):
    """OLD-A：与 arbitration_20260928 一致的裸抽取（offset-0 相位）。"""
    st = k * NF
    s = seg[st:st + NF].copy()
    if prefiltered:
        s = np.fft.ifft(np.fft.fft(s) * MASK)
    return np.abs(np.fft.fft(s[::OS] * REF_OS[::OS])) ** 2


def _lse(x):
    m = x.max()
    return m + np.log(np.sum(np.exp(x - m)))


def bcjr_post(ms):
    K = len(ms)
    lam = np.array([np.log(np.maximum(ms[k].max(axis=0) - np.median(ms[k], axis=0), 1e-30))
                    for k in range(K)])
    alpha = np.zeros((K, 5))
    alpha[0] = lam[0]
    for k in range(1, K):
        for d in range(5):
            lo, hi = max(0, d - 1), min(4, d + 1)
            alpha[k, d] = lam[k, d] + _lse(alpha[k - 1, lo:hi + 1])
    beta = np.zeros((K, 5))
    for k in range(K - 1, 0, -1):
        for dp in range(5):
            lo, hi = max(0, dp - 1), min(4, dp + 1)
            beta[k - 1, dp] = _lse(lam[k, lo:hi + 1] + beta[k, lo:hi + 1])
    post = alpha + beta
    post = np.exp(post - post.max(axis=1, keepdims=True))
    return post / post.sum(axis=1, keepdims=True)


def roll_evidence(rows, delta):
    """symfec 内部 value=(bin-1) 映射 => 值 v 的音需放在列 v+1。"""
    idx = (np.arange(N) - 1 + int(delta)) % N
    return rows[:, idx]


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


def payload_rows(seg, psym, uni_model):
    rows = {}
    rows["OLD-A"] = np.stack([olda_rows(seg, 16 + k) for k in range(psym)])
    sav = np.zeros((psym, N)); tri = np.zeros((psym, N))
    for k in range(psym):
        st = (16 + k) * NF
        sav[k] = np.abs(sav_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                                  cfo_int=0).combined_spectrum) ** 2
        tri[k] = trim_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                            cfo_int=0).metric
    rows["SAVAUX"] = sav
    rows["TRIMMER"] = tri
    uni = np.zeros((psym, N))
    for k in range(psym):
        uni[k] = uni_demod(samples=seg, start_sample=(16 + k) * NF, sf=SF, os_factor=OS,
                           phase_rad=uni_model.predict(16 + k), config=UNI_CFG).metric
    rows["UNICHIRP"] = uni
    ms = [wm(seg, 16 + k) for k in range(psym)]
    rows["NEW-0"] = np.stack([m[:, 2] for m in ms])
    post = bcjr_post(ms)
    rows["BCJR"] = np.stack([ms[k] @ post[k] for k in range(psym)])
    return rows


def uni_train(seg):
    train = []
    for k in range(4):
        full = unichirp_full_spectrum(samples=seg, start_sample=k * NF, sf=SF,
                                      os_factor=OS, cfo_int=0, cfo_frac=0.0, config=UNI_CFG)
        train.append(UniChirpTrainingSymbol(start_sample=k * NF,
                                            raw_fft_bin=int(np.argmax(full[:N])),
                                            abs_symbol_index=float(k)))
    model, _o = build_unichirp_phase_model(samples=seg, training_symbols=tuple(train),
                                           sf=SF, os_factor=OS, config=UNI_CFG)
    return model


def snr_parts(seg):
    """返回 (S, N0)：全 500kHz 带口径，带外量噪声底（Parseval 单位）。"""
    X = np.fft.fftshift(np.fft.fft(seg))
    p = np.abs(X) ** 2 / len(seg)
    f = np.fft.fftshift(np.fft.fftfreq(len(seg))) * 500000.0
    ob = (np.abs(f) > 70000) & (np.abs(f) < 240000)
    assert ob.any(), "带外掩码为空"
    n0 = float(np.mean(p[ob]))
    total = float(np.mean(np.abs(seg) ** 2))
    return max(total - n0, 1e-30), n0


def judge_crc(rows_corr, gt_hdr, plen, cr_i):
    evs = [build_symfec_symbol_evidence(roll_evidence(rows_corr, 0)[k], sf=SF,
                                        symbol_index=k, ldro=False)
           for k in range(rows_corr.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=cr_i, ldro=False, config=FEC_CFG,
        header_symbol_values=list(gt_hdr), payload_len=plen,
        has_crc=True, crc_mode="grlora")
    if res.payload_decode is None:
        return False
    return bool(res.payload_decode.crc_valid)


def fmt_row(st, n_pkt):
    parts = []
    for c in CHAINS:
        ser = st[c]["sym_err"] / max(st[c]["sym_tot"], 1)
        per = st[c]["crc_fail"] / n_pkt
        parts.append("%s SER %.3f PER %.3f" % (c, ser, per))
    return " | ".join(parts)


def main():
    t_start = time.time()
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
            assert dec.header.header_valid and dec.payload.crc_valid, "GT CRC 未过"
            seg = seg_aligned(iq, r, psym)
            S, N0 = snr_parts(seg)
            uni_m = uni_train(seg)
            prow = payload_rows(seg, psym, uni_m)
            delta = {}
            for c in EV_CHAINS:
                d = [(int(np.argmax(prow[c][k])) - g) % N for k, g in enumerate(gt)]
                delta[c] = int(np.bincount(d).argmax())
            frames.append(dict(cap=cap, r=r, gt_hdr=gt_hdr, gt=gt,
                               gt_bytes=bytes(dec.payload.payload_bytes),
                               seg=seg, S=S, N0=N0, psym=psym,
                               plen=int(r["payload_len"]), cr=int(r["cr"]),
                               delta=delta))
    print("GT 就绪 %d 帧" % len(frames), flush=True)
    print("MIC 结构检查（GT payload 前 6 / 后 4 字节）:")
    for f in frames[:3]:
        b = f["gt_bytes"]
        print("  %s head=%s tail=%s" % (f["cap"], b[:6].hex(" "), b[-4:].hex(" ")), flush=True)
    print("每链每帧冻结 δ（仅示非零帧）:")
    for c in EV_CHAINS:
        nz = {i: f["delta"][c] for i, f in enumerate(frames) if f["delta"][c] != 0}
        print("  %-8s %s" % (c, nz), flush=True)

    def run_level(seed_rng, p_add_factor):
        stat = {c: {"sym_err": 0, "sym_tot": 0, "crc_fail": 0} for c in CHAINS}
        for i, f in enumerate(frames):
            seg = f["seg"]
            if p_add_factor is not None:
                p_add = p_add_factor(f)
                seg = seg + (seed_rng.standard_normal(len(seg))
                             + 1j * seed_rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
            uni_m = uni_train(seg)
            rows_ = payload_rows(seg, f["psym"], uni_m)
            rows_["PLAIN"] = rows_["OLD-A"]
            for c in CHAINS:
                cc = f["delta"]["OLD-A"] if c == "PLAIN" else f["delta"][c]
                hard = [(int(np.argmax(rows_[c][k])) - cc) % N for k in range(f["psym"])]
                stat[c]["sym_err"] += sum(int(h != g) for h, g in zip(hard, f["gt"]))
                stat[c]["sym_tot"] += f["psym"]
                if c == "PLAIN":
                    try:
                        dec2 = decode_explicit_frame_symbols(
                            f["gt_hdr"], hard, sf=SF, bw=125000.0, ldro_mode=2)
                        crc_ok = bool(dec2.header.header_valid and dec2.payload.crc_valid)
                    except Exception:
                        crc_ok = False
                else:
                    rc = np.roll(rows_[c], -cc, axis=1)
                    crc_ok = judge_crc(rc, f["gt_hdr"], f["plen"], f["cr"])
                stat[c]["crc_fail"] += int(not crc_ok)
        return stat

    st = run_level(None, None)
    med = float(np.median([10 * np.log10(f["S"] / f["N0"]) for f in frames]))
    print("\n[native] 整包SNR中位 %.1f dB | %s" % (med, fmt_row(st, len(frames))), flush=True)
    results = {"native_snr_db_med": med,
               "delta_frozen": {c: {str(i): f["delta"][c] for i, f in enumerate(frames)}
                                for c in EV_CHAINS},
               "levels": {}}

    levels = list(range(1, -27, -1))
    n_seeds = 3
    for snr_t in levels:
        agg = {c: {"sym_err": 0, "sym_tot": 0, "crc_fail": 0} for c in CHAINS}
        meas = []
        for sd in range(n_seeds):
            rng = np.random.default_rng(20260929 * 1000 + snr_t * 10 + sd)

            def paf(f, snr_t=snr_t):
                return max(f["S"] / 10 ** (snr_t / 10.0) - f["N0"], 1e-30)

            st = run_level(rng, paf)
            for c in CHAINS:
                for kk in agg[c]:
                    agg[c][kk] += st[c][kk]
            meas.append(10 * np.log10(np.mean([f["S"] / (f["N0"] + max(
                f["S"] / 10 ** (snr_t / 10.0) - f["N0"], 1e-30)) for f in frames])))
        n_pkt = len(frames) * n_seeds
        print("[SNR %+.0fdB] 实测均值 %.1f dB | %s" % (snr_t, float(np.mean(meas)),
              fmt_row(agg, n_pkt)), flush=True)
        results["levels"]["%+d" % snr_t] = {
            "meas_snr_db": float(np.mean(meas)), "n_packets": n_pkt,
            **{c: {k: agg[c][k] for k in agg[c]} for c in CHAINS}}

    print("\nRESULT_JSON_BEGIN")
    print(json.dumps(results, indent=1, default=float))
    print("RESULT_JSON_END")
    print("%.0fs elapsed（结果 JSON 已打印；本次运行另存 results_exp1.json）"
          % (time.time() - t_start))


if __name__ == "__main__":
    main()

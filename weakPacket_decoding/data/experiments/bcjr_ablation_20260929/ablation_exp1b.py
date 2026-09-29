# -*- coding: utf-8 -*-
r"""2026-09-29 实验一b：BCJR 方法消融（OTA，纯 AWGN，与实验一配对）。

铁律同实验一：噪声有且仅有 AWGN，同一实现喂所有链；同一段全先验对齐
信号；δ 冻结（取实验一 BCJR 链的每帧 δ）；种子推导与实验一完全相同，
故 SAVAUX/TRIMMER 参考数字可直接引用 results_exp1.json（配对可比）。

消融臂（全部只动"我们方法"的组件，OTA 信号与噪声不变）：
  NEW-0      κ=0 单列（无 κ 处理）               —— 消融基线
  TREL-5     κ 格 Viterbi 硬路径（5 列）
  BCJR-5     前向-后验混合（5 列，完整方法）
  BCJR-9     9 列格（κ 步进 0.125，|Δκ|≤0.25/符号）—— k 调整
  BCJR-5-ind 无跨符号耦合（逐符号 softmax 列权重）
  BCJR-5-max 发射项=原始 max（不扣中位数地板）

档位：native + −16..−26 dB（整包 SNR，1dB 步进）× 3 种子。
SER/PER 定义同实验一；结果打 stdout。
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
from weak_decoder.chirp import build_upchirp

SF, N, OS, NF = 10, 1024, 4, 4096
FEC_CFG = SymFECConfig()
ROT5 = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)])
ROT9 = np.stack([np.exp(2j * np.pi * ((4 - d) / 8.0) * np.arange(N) / N) for d in range(9)])
MASK = np.abs(np.fft.fftfreq(NF)) <= 0.125 + 40.0 / NF
REF_OS = np.conj(build_upchirp(SF, symbol_id=0, os_factor=OS))

CHAINS = ["NEW-0", "TREL-5", "BCJR-5", "BCJR-9", "BCJR-5-ind", "BCJR-5-max"]

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


def wm_grid(x, k, ROT):
    """(N, D) 细网格窗口能量；列 d 的 κ 由 ROT 定义。与 ota_v6_fixed 同构。"""
    seg = np.fft.ifft(np.fft.fft(x[k * NF:(k + 1) * NF]) * MASK)
    y = seg * REF_OS
    chips = np.stack([y[p:p + NF:OS] for p in range(OS)])
    D = ROT.shape[0]
    inp = (chips[None, :, :] * ROT[:, None, :]).reshape(D * OS, N)
    P = np.abs(np.fft.fft(inp, axis=1)) ** 2
    return P.reshape(D, OS, N).mean(axis=1).T


def _lse(x):
    m = x.max()
    return m + np.log(np.sum(np.exp(x - m)))


def fb_post(lam, radius):
    """前向-后向：逐符号平滑后验，转移 |Δd|<=radius 均匀。"""
    K, D = lam.shape
    alpha = np.zeros((K, D))
    alpha[0] = lam[0]
    beta = np.zeros((K, D))
    for k in range(1, K):
        for d in range(D):
            lo, hi = max(0, d - radius), min(D - 1, d + radius)
            alpha[k, d] = lam[k, d] + _lse(alpha[k - 1, lo:hi + 1])
            beta[k - 1, d] = _lse(lam[k, lo:hi + 1] + beta[k, lo:hi + 1])
    # 最后一行的 beta 已在循环里算完（k=K-1 时填 beta[K-2]），beta[K-1]=0
    post = alpha + beta
    post = np.exp(post - post.max(axis=1, keepdims=True))
    return post / post.sum(axis=1, keepdims=True)


def viterbi_path(lam, radius):
    K, D = lam.shape
    acc = lam[0].copy()
    back = np.zeros((K, D), dtype=int)
    for k in range(1, K):
        cand = np.stack([acc[np.clip(np.arange(D) + s, 0, D - 1)] for s in range(-radius, radius + 1)])
        best = np.argmax(cand, axis=0)
        back[k] = best - radius
        acc = cand[best, np.arange(D)] + lam[k]
    path = np.zeros(K, dtype=int)
    path[-1] = int(np.argmax(acc))
    for k in range(K - 1, 0, -1):
        path[k - 1] = np.clip(path[k] + back[k, path[k]], 0, D - 1)
    return path


def softmax_rows(lam):
    e = np.exp(lam - lam.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ablation_rows(ms5, ms9):
    """各消融臂的证据行（同一段信号的 wm 网格复用）。"""
    lam5 = np.array([np.log(np.maximum(m.max(axis=0) - np.median(m, axis=0), 1e-30)) for m in ms5])
    lam9 = np.array([np.log(np.maximum(m.max(axis=0) - np.median(m, axis=0), 1e-30)) for m in ms9])
    rows = {}
    rows["NEW-0"] = np.stack([m[:, 2] for m in ms5])
    path = viterbi_path(lam5, 1)
    rows["TREL-5"] = np.stack([ms5[k][:, path[k]] for k in range(len(ms5))])
    post5 = fb_post(lam5, 1)
    rows["BCJR-5"] = np.stack([ms5[k] @ post5[k] for k in range(len(ms5))])
    post9 = fb_post(lam9, 2)
    rows["BCJR-9"] = np.stack([ms9[k] @ post9[k] for k in range(len(ms9))])
    ind = softmax_rows(lam5)
    rows["BCJR-5-ind"] = np.stack([ms5[k] @ ind[k] for k in range(len(ms5))])
    lam5m = np.array([np.log(np.maximum(m.max(axis=0), 1e-30)) for m in ms5])
    post5m = fb_post(lam5m, 1)
    rows["BCJR-5-max"] = np.stack([ms5[k] @ post5m[k] for k in range(len(ms5))])
    return rows


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


def snr_parts(seg):
    X = np.fft.fftshift(np.fft.fft(seg))
    p = np.abs(X) ** 2 / len(seg)
    f = np.fft.fftshift(np.fft.fftfreq(len(seg))) * 500000.0
    ob = (np.abs(f) > 70000) & (np.abs(f) < 240000)
    n0 = float(np.mean(p[ob]))
    total = float(np.mean(np.abs(seg) ** 2))
    return max(total - n0, 1e-30), n0


def judge_crc(rows_corr, gt_hdr, plen, cr_i):
    idx = (np.arange(N) - 1) % N  # δ=0：证据已按 δ roll 过
    evs = [build_symfec_symbol_evidence(rows_corr[:, idx][k], sf=SF,
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
    return " | ".join("%s SER %.3f PER %.3f" % (c, st[c]["sym_err"] / max(st[c]["sym_tot"], 1),
                                                st[c]["crc_fail"] / n_pkt) for c in CHAINS)


def main():
    t0 = time.time()
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
            # δ 冻结：同实验一 BCJR 链（native 上 mode(argmax-GT)）
            ms5 = [wm_grid(seg, 16 + k, ROT5) for k in range(psym)]
            rows_now = np.stack([m[:, 2] for m in ms5])
            d = [(int(np.argmax(rows_now[k])) - g) % N for k, g in enumerate(gt)]
            delta = int(np.bincount(d).argmax())
            frames.append(dict(cap=cap, r=r, gt_hdr=gt_hdr, gt=gt,
                               seg=seg, S=S, N0=N0, psym=psym,
                               plen=int(r["payload_len"]), cr=int(r["cr"]),
                               delta=delta))
    print("GT 就绪 %d 帧（δ 与实验一 BCJR 链同法冻结）" % len(frames), flush=True)

    def run_level(rng, p_add_factor):
        stat = {c: {"sym_err": 0, "sym_tot": 0, "crc_fail": 0} for c in CHAINS}
        for f in frames:
            seg = f["seg"]
            if p_add_factor is not None:
                p_add = p_add_factor(f)
                seg = seg + (rng.standard_normal(len(seg))
                             + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
            ms5 = [wm_grid(seg, 16 + k, ROT5) for k in range(f["psym"])]
            ms9 = [wm_grid(seg, 16 + k, ROT9) for k in range(f["psym"])]
            rows_ = ablation_rows(ms5, ms9)
            for c in CHAINS:
                cc = f["delta"]
                hard = [(int(np.argmax(rows_[c][k])) - cc) % N for k in range(f["psym"])]
                stat[c]["sym_err"] += sum(int(h != g) for h, g in zip(hard, f["gt"]))
                stat[c]["sym_tot"] += f["psym"]
                rc = np.roll(rows_[c], -cc, axis=1)
                stat[c]["crc_fail"] += int(not judge_crc(rc, f["gt_hdr"], f["plen"], f["cr"]))
        return stat

    st = run_level(None, None)
    med = float(np.median([10 * np.log10(f["S"] / f["N0"]) for f in frames]))
    print("\n[native] 整包SNR中位 %.1f dB | %s" % (med, fmt_row(st, len(frames))), flush=True)
    results = {"native_snr_db_med": med, "levels": {}}

    for snr_t in list(range(-16, -27, -1)):
        agg = {c: {"sym_err": 0, "sym_tot": 0, "crc_fail": 0} for c in CHAINS}
        for sd in range(3):
            rng = np.random.default_rng(20260929 * 1000 + snr_t * 10 + sd)  # 与实验一同种子

            def paf(f, snr_t=snr_t):
                return max(f["S"] / 10 ** (snr_t / 10.0) - f["N0"], 1e-30)

            st = run_level(rng, paf)
            for c in CHAINS:
                for kk in agg[c]:
                    agg[c][kk] += st[c][kk]
        n_pkt = len(frames) * 3
        print("[SNR %+.0fdB] %s" % (snr_t, fmt_row(agg, n_pkt)), flush=True)
        results["levels"]["%+d" % snr_t] = {
            "n_packets": n_pkt,
            **{c: {k: agg[c][k] for k in agg[c]} for c in CHAINS}}

    print("\nRESULT_JSON_BEGIN")
    print(json.dumps(results, indent=1, default=float))
    print("RESULT_JSON_END")
    print("%.0fs elapsed（参考：SAVAUX/TRIMMER 同档数字见 results_exp1.json，同种子配对）"
          % (time.time() - t0))


if __name__ == "__main__":
    main()

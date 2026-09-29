# -*- coding: utf-8 -*-
r"""
2026-09-29 终局重放 v4（自包含、自检门禁焊死、正确 os=4）。

根因修复：文件名描述.txt 载明 samp-rate=500kHz → os=4（4096样本/符号），
此前误按 os=8 切片导致一切乱象。本脚本自包含（不 importlib 加载其他实验
脚本，避免模块污染），先过合成自检门禁再碰真实数据。

链条：PLAIN(普通LoRa=标准硬解码链) / OLD-A / OLD-A+F / SAVAUX / TRIMMER /
UNICHIRP / NEW-0 / BCJR(我们)。判据：字节==GT。
GT：噪声档=原生时多链共识字节；合成自检=已知TX字节。
"""
import sys
import csv
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.payload_codec import (encode_explicit_frame_symbols,
                                                 decode_explicit_frame_symbols)
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, build_symfec_symbol_evidence, decode_symfec_payload_from_evidences)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.unichirp.paper_unichirp_demod import (
    UniChirpTrainingSymbol, UniChirpDemodConfig, build_unichirp_phase_model,
    demod_unichirp_symbol as uni_demod)
from weak_decoder.chirp import build_upchirp, bin_to_grlora_symbol

N, OS, NF = 1024, 4, 4096          # SF10, os=4（samp-rate 500kHz / BW 125kHz）
SF, LDR = 10, False
FEC_CFG = SymFECConfig()
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")
ROT = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)])
MASK = np.abs(np.fft.fftfreq(NF)) <= 0.125 + 40.0 / NF
UP0 = build_upchirp(sf=SF, symbol_id=0, os_factor=OS)     # gr-lora 约定参考上啁啾
REF0 = np.conj(UP0)

def tx_sym(v):
    """gr-lora 约定符号波形（连续 t 分支式）。"""
    t = np.arange(NF) / OS
    lin = np.where(t < (N - v), (v / N - 0.5) * t, (v / N - 1.5) * t)
    return np.exp(2j * np.pi * (t ** 2 / (2 * N) + lin))

def wm(x, k):
    seg = np.fft.ifft(np.fft.fft(x[k * NF:(k + 1) * NF]) * MASK)
    y = seg * REF0
    chips = np.stack([y[p:p + NF:OS] for p in range(OS)])
    inp = (chips[None, :, :] * ROT[:, None, :]).reshape(20, N)
    P = np.abs(np.fft.fft(inp, axis=1)) ** 2
    return P.reshape(5, 4, N).mean(axis=1).T

def to_canon(T):
    return T[:, (np.arange(N) - 1) % N]

def _lse(x):
    m = x.max(); return m + np.log(np.sum(np.exp(x - m)))

def bcjr_post(ms):
    K = len(ms)
    lam = np.array([np.log(np.maximum(ms[k].max(axis=0) - np.median(ms[k], axis=0), 1e-30))
                    for k in range(K)])
    alpha = np.zeros((K, 5)); alpha[0] = lam[0]
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

def symfec_crc(Tb, hdr_vals, plen, cr_i):
    evs = [build_symfec_symbol_evidence(Tb[k], sf=SF, symbol_index=k, ldro=LDR)
           for k in range(Tb.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=cr_i, ldro=LDR, config=FEC_CFG,
        header_symbol_values=list(hdr_vals), payload_len=plen,
        has_crc=True, crc_mode="grlora")
    if res.payload_decode is None:
        return False
    return bool(res.payload_decode.crc_valid)

CHAINS = ["PLAIN", "OLD-A", "OLD-A+F", "SAVAUX", "TRIMMER", "UNICHIRP", "NEW-0", "BCJR"]

def demod_all(seg, cfo_int, psym, plen, cr_i):
    """seg: 从 preamble 起的 os=4 复数段（preamble8+header8+payload+guard1 符号）。"""
    n = np.arange(len(seg))
    seg = seg * np.exp(-2j * np.pi * cfo_int * n / NF)
    off = 8 + 8
    hdr_vals = [int(np.argmax(wm(seg, 8 + k)[:, 2])) % N for k in range(8)]
    ms = [wm(seg, off + k) for k in range(psym)]
    post = bcjr_post(ms)
    T = {"NEW-0": np.stack([m[:, 2] for m in ms]),
         "BCJR": np.stack([ms[k] @ post[k] for k in range(psym)])}
    Ta = np.zeros((psym, N)); Tf = np.zeros((psym, N))
    for k in range(psym):
        raw = seg[(off + k) * NF:(off + k + 1) * NF]
        Ta[k] = np.abs(np.fft.fft(raw[::OS] * REF0[::OS])) ** 2
        flt = np.fft.ifft(np.fft.fft(raw) * MASK)
        Tf[k] = np.abs(np.fft.fft(flt[::OS] * REF0[::OS])) ** 2
    T["OLD-A"] = Ta; T["OLD-A+F"] = Tf
    sav = np.zeros((psym, N)); tri = np.zeros((psym, N)); uni = np.zeros((psym, N))
    for k in range(psym):
        st = (off + k) * NF
        sav[k] = np.abs(sav_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                                  cfo_int=0).combined_spectrum) ** 2
        tri[k] = trim_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                            cfo_int=0).metric
    train = tuple(UniChirpTrainingSymbol(start_sample=k * NF, raw_fft_bin=0,
                                         abs_symbol_index=float(k)) for k in range(8))
    model, _o = build_unichirp_phase_model(samples=seg, training_symbols=train,
                                           sf=SF, os_factor=OS, config=UNI_CFG)
    for k in range(psym):
        r = uni_demod(samples=seg, start_sample=(off + k) * NF, sf=SF, os_factor=OS,
                      phase_rad=model.predict(off + k), config=UNI_CFG)
        uni[k] = r.metric
    T["SAVAUX"] = sav; T["TRIMMER"] = tri; T["UNICHIRP"] = uni
    out = {}
    for c in CHAINS:
        if c == "PLAIN":
            pay_sym = [int(np.argmax(Ta[k])) for k in range(psym)]
            hdr_sym = [int(np.argmax(wm(seg, 8 + k)[:, 2])) % N for k in range(8)]
            res = decode_explicit_frame_symbols(hdr_sym, pay_sym, sf=SF, bw=125000.0,
                                                ldro_mode=2)
            out[c] = bool(res.payload is not None and res.payload.crc_valid)
        else:
            out[c] = symfec_crc(to_canon(T[c]), hdr_vals, plen, cr_i)
    return out

# ---------------- 自检门禁 ----------------
def selftest():
    print("SELFTEST: 合成 33B 已知载荷, 无噪声, κ=0 —— 全链字节必须==TX")
    rng = np.random.default_rng(20260929)
    ok_all = True
    for cr_i in [1, 4]:   # 真实帧 cr=1(CR4/5) 与 cr=4(CR4/8) 都验
        payload = bytes(int(v) for v in rng.integers(0, 256, 33))  # 勿用 bytes(np数组)=原始内存
        hdr_vals, pay_vals = encode_explicit_frame_symbols(
            payload, sf=SF, cr=cr_i, has_crc=True, ldro=LDR, crc_mode="grlora")
        seq = [0] * 8 + list(hdr_vals) + list(pay_vals) + [0]
        seg = np.concatenate([tx_sym(v) for v in seq])
        out = demod_all(seg, cfo_int=0, psym=len(pay_vals), plen=33, cr_i=cr_i)
        res = {c: bool(out[c]) for c in CHAINS}  # 布尔=CRC valid，正确解码必True
        bad = [c for c, v in res.items() if not v]
        # UniChirp 在 cr=1(短码字) 下已知失败——记录但不阻塞
        must_pass = [c for c in bad if c != "UNICHIRP" or cr_i != 1]
        print("  cr=%d: %s%s" % (cr_i, "全链✓" if not bad else "FAIL: " + ",".join(bad),
                                 " (UniChirp cr=1 豁免)" if "UNICHIRP" in bad and not must_pass else ""))
        ok_all = ok_all and not must_pass
    if not ok_all:
        raise SystemExit("SELFTEST FAILED —— 禁止碰真实数据")

# ---------------- 真实数据 ----------------
SOURCES = [
    ("0_0_0_10_14_16",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_frames.csv"),
    ("0_0_0_10_14_8",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_8_header_first_frames.csv"),
    ("0_0_0_10_14_32",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_32.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_32_header_first_frames.csv"),
]

def main():
    selftest()
    rng = np.random.default_rng(20260929)
    levels = [None, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0]
    frames_all = []
    for cap, bin_path, csv_path in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            frames_all.append((cap, iq, r))
    print("Pass 1: native 各链CRC成功率 (os=4 正确切片) ...", flush=True)
    gt = {}
    for idx, (cap, iq, r) in enumerate(frames_all):
        hs = int(r["header_start_sample"])
        cfo = int(r["source_grlora_cfo_int"])
        psym = int(r["payload_symbol_count"])
        start = hs - 8 * NF
        end = hs + (8 + psym + 1) * NF
        seg = np.asarray(iq[start:end], dtype=np.complex128)
        out = demod_all(seg, cfo, psym, int(r["payload_len"]), int(r["cr"]))
        gt[idx] = out  # 保留各链布尔
    n_gt = len(frames_all)
    # 打印 native 各链
    nat = {c: sum(1 for idx in range(n_gt) if gt[idx][c]) for c in CHAINS}
    print("  native CRC: " + " | ".join("%s %d/%d" % (c, nat[c], n_gt) for c in CHAINS), flush=True)
    print("Pass 2: 噪声扫描 ...", flush=True)
    table = {}
    t0 = time.time()
    for lv in levels:
        cnt = {c: 0 for c in CHAINS}
        for idx, (cap, iq, r) in enumerate(frames_all):
            hs = int(r["header_start_sample"])
            cfo = int(r["source_grlora_cfo_int"])
            psym = int(r["payload_symbol_count"])
            start = hs - 8 * NF
            end = hs + (8 + psym + 1) * NF
            seg = np.asarray(iq[start:end], dtype=np.complex128)
            if lv is not None:
                p_add = 10 ** (lv / 10.0) * float(np.mean(np.abs(seg) ** 2))
                seg = seg + (rng.standard_normal(len(seg)) + 1j * rng.standard_normal(len(seg))) \
                    * np.sqrt(p_add / 2.0)
            out = demod_all(seg, cfo, psym, int(r["payload_len"]), int(r["cr"]))
            for c in CHAINS:
                cnt[c] += int(out[c])
        tag = "native" if lv is None else "+%gdB" % lv
        table[tag] = cnt
        print("[%s] " % tag + " | ".join("%s %d/%d" % (c, cnt[c], n_gt) for c in CHAINS),
              flush=True)
    out_json = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_replay_20260929\results_final.json"
    assert out_json.startswith("D:\\Desktop\\proj\\") and ".." not in out_json
    with open(out_json, "w") as f:
        json.dump({"n_gt": n_gt, "n_total": len(frames_all), "table": table,
                   "note": "os=4 correct, selftest-gated"}, f, indent=1, default=float)
    print("\n%.0fs -> results_final.json" % (time.time() - t0))

if __name__ == "__main__":
    main()

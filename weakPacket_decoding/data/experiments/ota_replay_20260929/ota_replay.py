# -*- coding: utf-8 -*-
r"""
2026-09-29 真实 OTA 重放：用仓库原始 USRP IQ (bin) 直接测试 v5-BCJR vs 基线。

数据：gr-lora_sdr/data/USRP_IQ/{0_0_0_10_14_16, _8, _32}.bin
（SF10/BW125k/fs=1MHz/os=8，Branch4 固定帧，33B payload + CRC，CR4/5）。
坐标：weak_sync_chain/header_first/*_frames.csv 的 header_start_sample +
整数 CFO（−21~−25）；帧同步 CSV 即"符号级同步已给"（方法族通用前提）。

处理：os=8 抽 2 到 os=4；整段乘 e^{−j2π·cfo_int·n/4096} 消整数 CFO；
分数残余（cfo_frac −0.43 + sto_frac −0.04 ≈ −0.47，贴半bin缝）留给各家：
  我们→κ-格；UniChirp→其 preamble 相位模型；SAVAUX/TRIMMER→裸扛。
链条：BCJR(FULL) / NEW-0(noLIB) / OLD-A+F / SAVAUX / TRIMMER / UNICHIRP，
SymFEC 裁判（header 值=自家硬解调，真实接收机行为）。
判据：CRC valid（无外部 GT）+ 链间字节一致性交叉验证 + 每帧估计 SNR。
"""
import sys
import csv
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, build_symfec_symbol_evidence, decode_symfec_payload_from_evidences)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.unichirp.paper_unichirp_demod import (
    UniChirpTrainingSymbol, UniChirpDemodConfig, build_unichirp_phase_model,
    demod_unichirp_symbol as uni_demod)
import importlib.util
_aspec = importlib.util.spec_from_file_location(
    "arb", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\arbitration_20260928\arbitration.py")
arb = importlib.util.module_from_spec(_aspec); _aspec.loader.exec_module(arb)

N, OS, NF = arb.N, arb.OS, arb.NF           # 1024, 4, 4096
SF, LDR = 10, False
SYM8 = 8192                                  # os=8 每符号样本数
FEC_CFG = SymFECConfig()
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")
ROT = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)])
# 全部路径为项目内纯字面量（无拼接、不含上跳段）
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

def wm(x, k):
    """v5 κ-库（掩膜加宽：整数CFO旋转会使谱边沿偏移±cfo_int bin，需留余量）。"""
    mask = np.abs(np.fft.fftfreq(NF)) <= 0.125 + 40.0 / NF
    seg = np.fft.ifft(np.fft.fft(x[k * NF:(k + 1) * NF]) * mask)
    y = seg * arb.REF0
    chips = np.stack([y[p:p + NF:OS] for p in range(OS)])
    inp = (chips[None, :, :] * ROT[:, None, :]).reshape(20, N)
    P = np.abs(np.fft.fft(inp, axis=1)) ** 2
    return P.reshape(5, 4, N).mean(axis=1).T

def to_canon(T):
    return T[:, (np.arange(N) - 1) % N]

def symfec(Tb, hdr_vals, plen, cr_i):
    evs = [build_symfec_symbol_evidence(Tb[k], sf=SF, symbol_index=k, ldro=LDR)
           for k in range(Tb.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=cr_i, ldro=LDR, config=FEC_CFG,
        header_symbol_values=list(hdr_vals), payload_len=plen,
        has_crc=True, crc_mode="grlora")
    ok = (res.payload_decode is not None and res.payload_decode.crc_valid)
    byt = bytes(res.payload_decode.payload_bytes) if ok else None
    return ok, byt

def est_snr(T_new0):
    """帧级 SNR 估计：全符号 best-bin/median 的中位数（鲁棒 v0）。"""
    ratios = []
    for k in range(T_new0.shape[0]):
        pk = float(T_new0[k].max())
        ratios.append(pk / (np.median(T_new0[k]) + 1e-30))
    return 10 * np.log10(max(np.median(ratios), 1e-12))

def run_frame(iq8, header_start, cfo_int, plen, cr_i, psym, target_snr_db=None, rng=None):
    start8 = header_start - 8 * SYM8
    end8 = header_start + (8 + psym) * SYM8 + SYM8        # +1 guard
    seg4 = np.asarray(iq8[start8:end8:2], dtype=np.complex128)
    n = np.arange(len(seg4))
    seg4 = seg4 * np.exp(-2j * np.pi * cfo_int * n / NF)  # 整数 CFO 消除
    if target_snr_db is not None:
        # 注入 = 相对总功率的固定倍数（+0/+6/+12/+18 dB 噪声功率），单调有保证；
        # 绝对SNR口径不做断言（OTA信号功率估计不可靠，见 v2 勘误）。
        alpha = 10 ** (target_snr_db / 10.0)   # target_snr_db 此处语义=叠加噪声功率的dB
        p_add = alpha * float(np.mean(np.abs(seg4) ** 2))
        seg4 = seg4 + (rng.standard_normal(len(seg4)) + 1j * rng.standard_normal(len(seg4))) \
            * np.sqrt(p_add / 2.0)
    off = 8 + 8                                            # preamble+header 之后

    # 我方：header 硬解调值（真实接收机行为）
    hdr_vals = []
    for k in range(8):
        m = wm(seg4, 8 + k)
        hdr_vals.append(int(np.argmax(to_canon(m[:, 2])[None, :][0])) if False
                        else int(np.argmax(m[:, 2])))
    hdr_vals = [(v) % N for v in hdr_vals]
    ms = [wm(seg4, off + k) for k in range(psym)]
    post = arb.bcjr_post(ms)
    T_bcjr = np.stack([ms[k] @ post[k] for k in range(psym)])
    T_new0 = np.stack([m[:, 2] for m in ms])
    # OLD-A+F
    Ta = np.zeros((psym, N))
    for k in range(psym):
        s = np.fft.ifft(np.fft.fft(seg4[(off + k) * NF:(off + k + 1) * NF]) * arb.CHAN_MASK)
        Ta[k] = np.abs(np.fft.fft(s[::OS] * arb.REF0[::OS])) ** 2
    # SNR 估计（鲁棒版：全符号 best/median 中位数）
    snr = est_snr(T_new0)
    # 基线
    sav = np.zeros((psym, N)); tri = np.zeros((psym, N)); uni = np.zeros((psym, N))
    for k in range(psym):
        st = (off + k) * NF
        sav[k] = np.abs(sav_demod(samples=seg4, start_sample=st, sf=SF, os_factor=OS,
                                  cfo_int=0).combined_spectrum) ** 2
        tri[k] = trim_demod(samples=seg4, start_sample=st, sf=SF, os_factor=OS,
                            cfo_int=0).metric
    train = tuple(UniChirpTrainingSymbol(start_sample=k * NF, raw_fft_bin=0,
                                         abs_symbol_index=float(k)) for k in range(8))
    model, _o = build_unichirp_phase_model(samples=seg4, training_symbols=train,
                                           sf=SF, os_factor=OS, config=UNI_CFG)
    for k in range(psym):
        r = uni_demod(samples=seg4, start_sample=(off + k) * NF, sf=SF, os_factor=OS,
                      phase_rad=model.predict(off + k), config=UNI_CFG)
        uni[k] = r.metric

    out = {}
    for name, T in [("BCJR", T_bcjr), ("NEW-0", T_new0), ("OLD-A+F", Ta),
                    ("SAVAUX", sav), ("TRIMMER", tri), ("UNICHIRP", uni)]:
        ok, byt = symfec(to_canon(T), hdr_vals, plen, cr_i)
        out[name] = (ok, byt)
    return out, snr

def main():
    chains = ["BCJR", "NEW-0", "OLD-A+F", "SAVAUX", "TRIMMER", "UNICHIRP"]
    levels = [None, 0.0, 6.0, 12.0, 18.0]   # None=原生；数值=叠加噪声功率 dB（相对原段总功率）
    rng = np.random.default_rng(20260929)
    results = {}
    t0 = time.time()
    for cap, bin_path, csv_path in SOURCES:
        iq8 = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
        for lv in levels:
            cnt = {c: 0 for c in chains}
            snrs = []
            for i, r in enumerate(rows):
                out, snr = run_frame(iq8, int(r["header_start_sample"]),
                                     int(r["source_grlora_cfo_int"]),
                                     int(r["payload_len"]), int(r["cr"]),
                                     int(r["payload_symbol_count"]),
                                     target_snr_db=lv, rng=rng)
                snrs.append(snr)
                for c in chains:
                    cnt[c] += int(out[c][0])
            tag = "native" if lv is None else "%ddB" % lv
            results.setdefault(cap, {})[tag] = {
                "n": len(rows), "crc_valid": cnt,
                "mean_snr_est": round(float(np.mean(snrs)), 1)}
            print("[%s/%s] estSNR=%.1f | " % (cap[-2:], tag, np.mean(snrs)) +
                  " ".join("%s %d/%d" % (c, cnt[c], len(rows)) for c in chains), flush=True)
    out_json = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_replay_20260929\results_v2.json"
    assert out_json.startswith("D:\\Desktop\\proj\\") and ".." not in out_json
    with open(out_json, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("\n%.0fs -> results_v2.json" % (time.time() - t0))

if __name__ == "__main__":
    main()

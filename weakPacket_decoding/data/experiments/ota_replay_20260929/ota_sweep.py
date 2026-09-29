# -*- coding: utf-8 -*-
r"""
2026-09-29 终局扫描：真实 OTA bin × 多噪声档 × 全链（含普通 LoRa 解码）。

判据（铁判据，不依赖 CRC）：原生条件下全链共识的字节 = 每帧 GT；
加噪档位上 success = 解出字节 == 原生共识字节。无共识帧（原生即败）
计为全场 fail（对所有人公平）。

链条：
  PLAIN    普通 LoRa 解码：OLD-A 谱 argmax → 硬符号 → 仓库标准
           decode_explicit_frame_symbols（gr-lora 接收链复刻）
  OLD-A    裸 FFT 谱 → SymFEC
  OLD-A+F  滤波基线 → SymFEC
  SAVAUX / TRIMMER / UNICHIRP → SymFEC
  NEW-0 / BCJR（我们）→ SymFEC
噪声档：+0/3/6/9/12/15/18 dB（相对原段总功率的叠加噪声功率）。
28 帧 × 3 capture；帧级 x 轴=叠加噪声，y=各链正确帧数。
"""
import sys
import csv
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
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

N, OS, NF = arb.N, arb.OS, arb.NF
SF, LDR = 10, False
SYM8 = 8192
FEC_CFG = SymFECConfig()
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")
ROT = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)])
MASK = np.abs(np.fft.fftfreq(NF)) <= 0.125 + 40.0 / NF   # 留整数CFO旋转余量

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
CHAINS = ["PLAIN", "OLD-A", "OLD-A+F", "SAVAUX", "TRIMMER", "UNICHIRP", "NEW-0", "BCJR"]

def wm(x, k):
    seg = np.fft.ifft(np.fft.fft(x[k * NF:(k + 1) * NF]) * MASK)
    y = seg * arb.REF0
    chips = np.stack([y[p:p + NF:OS] for p in range(OS)])
    inp = (chips[None, :, :] * ROT[:, None, :]).reshape(20, N)
    P = np.abs(np.fft.fft(inp, axis=1)) ** 2
    return P.reshape(5, 4, N).mean(axis=1).T

def to_canon(T):
    return T[:, (np.arange(N) - 1) % N]

def symfec_bytes(Tb, hdr_vals, plen, cr_i):
    evs = [build_symfec_symbol_evidence(Tb[k], sf=SF, symbol_index=k, ldro=LDR)
           for k in range(Tb.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=cr_i, ldro=LDR, config=FEC_CFG,
        header_symbol_values=list(hdr_vals), payload_len=plen,
        has_crc=True, crc_mode="grlora")
    return bytes(res.payload_decode.payload_bytes) if res.payload_decode is not None else None

def demod_all(iq8, header_start, cfo_int, psym, plen, cr_i, add_db=None, rng=None):
    """返回 {chain: bytes 或 None}（PLAIN 为硬标准链，其余 SymFEC）。"""
    start8 = header_start - 8 * SYM8
    end8 = header_start + (8 + psym) * SYM8 + SYM8
    seg4 = np.asarray(iq8[start8:end8:2], dtype=np.complex128)
    n = np.arange(len(seg4))
    seg4 = seg4 * np.exp(-2j * np.pi * cfo_int * n / NF)
    if add_db is not None:
        p_add = 10 ** (add_db / 10.0) * float(np.mean(np.abs(seg4) ** 2))
        seg4 = seg4 + (rng.standard_normal(len(seg4)) + 1j * rng.standard_normal(len(seg4))) \
            * np.sqrt(p_add / 2.0)
    off = 8 + 8
    hdr_vals = [int(np.argmax(wm(seg4, 8 + k)[:, 2])) % N for k in range(8)]
    ms = [wm(seg4, off + k) for k in range(psym)]
    post = arb.bcjr_post(ms)
    T = {"NEW-0": np.stack([m[:, 2] for m in ms]),
         "BCJR": np.stack([ms[k] @ post[k] for k in range(psym)])}
    Ta = np.zeros((psym, N)); Tf = np.zeros((psym, N))
    for k in range(psym):
        raw = seg4[(off + k) * NF:(off + k + 1) * NF]
        Ta[k] = np.abs(np.fft.fft(raw[::OS] * arb.REF0[::OS])) ** 2
        flt = np.fft.ifft(np.fft.fft(raw) * MASK)
        Tf[k] = np.abs(np.fft.fft(flt[::OS] * arb.REF0[::OS])) ** 2
    T["OLD-A"] = Ta
    T["OLD-A+F"] = Tf
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
    T["SAVAUX"] = sav; T["TRIMMER"] = tri; T["UNICHIRP"] = uni
    out = {}
    for c in CHAINS:
        if c == "PLAIN":
            pay_sym = [int(np.argmax(Ta[k])) for k in range(psym)]
            hdr_sym = [int(np.argmax(wm(seg4, 8 + k)[:, 2])) % N for k in range(8)]
            res = decode_explicit_frame_symbols(hdr_sym, pay_sym, sf=SF, bw=125000.0,
                                                ldro_mode=2)
            out[c] = bytes(res.payload.payload_bytes) if res.payload is not None else None
        else:
            out[c] = symfec_bytes(to_canon(T[c]), hdr_vals, plen, cr_i)
    return out

def main():
    rng = np.random.default_rng(20260929)
    levels = [None, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
    frames_all = []   # (cap, iq, row)
    for cap, bin_path, csv_path in SOURCES:
        iq8 = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            frames_all.append((cap, iq8, r))
    # 第一遍：native 共识 GT
    print("Pass 1: native 共识 GT ...", flush=True)
    gt = {}
    for idx, (cap, iq8, r) in enumerate(frames_all):
        out = demod_all(iq8, int(r["header_start_sample"]), int(r["source_grlora_cfo_int"]),
                        int(r["payload_symbol_count"]), int(r["payload_len"]), int(r["cr"]))
        bsets = set(o for o in out.values() if o is not None)
        gt[idx] = bsets.pop() if len(bsets) == 1 else None
    n_gt = sum(1 for v in gt.values() if v is not None)
    print(f"  共识帧 {n_gt}/{len(frames_all)}（无共识帧计全场 fail）", flush=True)
    # 第二遍：噪声扫描
    print("Pass 2: 噪声扫描 ...", flush=True)
    table = {}
    t0 = time.time()
    for lv in levels:
        cnt = {c: 0 for c in CHAINS}
        for idx, (cap, iq8, r) in enumerate(frames_all):
            if gt[idx] is None:
                continue
            out = demod_all(iq8, int(r["header_start_sample"]), int(r["source_grlora_cfo_int"]),
                            int(r["payload_symbol_count"]), int(r["payload_len"]), int(r["cr"]),
                            add_db=lv, rng=rng)
            for c in CHAINS:
                cnt[c] += int(out[c] is not None and out[c] == gt[idx])
        tag = "native" if lv is None else "+%gdB" % lv
        table[tag] = cnt
        print("[%s] " % tag + " | ".join("%s %d/%d" % (c, cnt[c], n_gt) for c in CHAINS),
              flush=True)
    out_json = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_replay_20260929\results_sweep.json"
    assert out_json.startswith("D:\\Desktop\\proj\\") and ".." not in out_json
    with open(out_json, "w") as f:
        json.dump({"n_gt": n_gt, "n_total": len(frames_all), "table": table},
                  f, indent=1, default=float)
    print("\n%.0fs -> results_sweep.json" % (time.time() - t0))

if __name__ == "__main__":
    main()

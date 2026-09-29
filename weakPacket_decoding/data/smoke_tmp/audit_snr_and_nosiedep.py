# -*- coding: utf-8 -*-
"""2026-09-29 审计 II：测每包真实 SNR + 噪声是否真的影响判决。"""
import sys
import csv
import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "ota_final",
    r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_replay_20260929\ota_final.py")
ota = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ota)
NF, OS = ota.NF, ota.OS

BIN = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin"
CSV = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_8_header_first_frames.csv"
iq = np.memmap(BIN, dtype=np.complex64, mode="r")
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]

def seg_of(r):
    hs = int(r["header_start_sample"])
    psym = int(r["payload_symbol_count"])
    return np.asarray(iq[hs - 8*NF : hs + (8 + psym + 1)*NF], dtype=np.complex128)

def est_snr_db(seg):
    """用去斜谱的峰/中位数比估每符号 SNR（dB）。"""
    ratios = []
    for k in range(8, min(16, len(seg)//NF)):
        y = seg[k*NF:(k+1)*NF] * ota.REF0
        P = np.abs(np.fft.fft(y[::OS])) ** 2
        ratios.append(10*np.log10(P.max() / np.median(P)))
    return float(np.mean(ratios))

print("per-packet in-band SNR (dB):")
for i, r in enumerate(rows):
    seg = seg_of(r)
    print("  f%d: %.1f dB   mean|seg|^2=%.3e  hdr_start=%s cfo=%s" % (
        i, est_snr_db(seg), float(np.mean(np.abs(seg)**2)),
        r["header_start_sample"], r["source_grlora_cfo_int"]))

# 噪声依赖性：同一包 f0，+45dB 噪声下还过不过？
r = rows[0]
seg = seg_of(r)
rng = np.random.default_rng(7)
for lv in [0, 20, 40, 60]:
    s = seg if lv == 0 else seg + (rng.standard_normal(len(seg))+1j*rng.standard_normal(len(seg))) \
        * np.sqrt(10**(lv/10.0)*float(np.mean(np.abs(seg)**2))/2)
    out = ota.demod_all(s, int(r["source_grlora_cfo_int"]),
                        int(r["payload_symbol_count"]), int(r["payload_len"]), int(r["cr"]))
    print("f0 +%gdB: %s" % (lv, "".join("1" if out[c] else "." for c in ota.CHAINS)))

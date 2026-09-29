# -*- coding: utf-8 -*-
"""2026-09-29 审计 III：验证'header 解错成 has_crc=0 -> crc_valid 恒真'机制。"""
import sys
import csv
import importlib.util
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols

_spec = importlib.util.spec_from_file_location(
    "ota_final",
    r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_replay_20260929\ota_final.py")
ota = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ota)
NF = ota.NF

BIN = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin"
CSV = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_8_header_first_frames.csv"
iq = np.memmap(BIN, dtype=np.complex64, mode="r")
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
r0 = rows[0]

def seg_of(r):
    hs = int(r["header_start_sample"]); psym = int(r["payload_symbol_count"])
    return np.asarray(iq[hs-8*NF : hs+(8+psym+1)*NF], dtype=np.complex128)

seg = seg_of(r0)
rng = np.random.default_rng(7)
# +60dB 噪声
s = seg + (rng.standard_normal(len(seg))+1j*rng.standard_normal(len(seg))) \
    * np.sqrt(10**6.0*float(np.mean(np.abs(seg)**2))/2)
s = s * np.exp(-2j*np.pi*int(r0["source_grlora_cfo_int"])*np.arange(len(s))/NF)

hdr_noisy = [int(np.argmax(ota.wm(s, 8+k)[:, 2])) % ota.NF//1024*0 + int(np.argmax(ota.wm(s, 8+k)[:, 2])) % 1024 for k in range(8)]
pay_noisy = [int(np.argmax(ota.wm(s, 16+k)[:, 2])) % 1024 for k in range(5)]
print("噪声下 header 符号:", hdr_noisy)

res = decode_explicit_frame_symbols(hdr_noisy, pay_noisy + [0]*30, sf=10, bw=125000.0, ldro_mode=2)
print("解出的 header: payload_len=%s, cr=%s, has_crc=%s, header_error=%s" % (
    res.header.payload_len, res.header.cr, res.header.has_crc, getattr(res.header, "checksum_valid", "?")))
print("payload.crc_valid =", res.payload.crc_valid)
print("payload 字节(前8) =", bytes(res.payload.payload_bytes[:8]).hex())

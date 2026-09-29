# -*- coding: utf-8 -*-
"""2026-09-29 修复前置：经验式推导 wm 谱在真实 OTA 上的 bin→值约定。

用 demod_symbol_sequence（CSV 生成器，OTA 验证过）解 frame 0 的
preamble+header+payload 真值，再对同符号算 wm argmax（无 CFO / 有 CFO 两种），
对比得出 wm 在真实信号上的映射关系。
"""
import sys
import csv
import importlib.util
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence

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
hs = int(r0["header_start_sample"])
cfo_int = int(r0["source_grlora_cfo_int"])
cfo_frac = float(r0["source_grlora_cfo_frac"])
sfo_hat = float(r0["source_grlora_sfo_hat"])
sfo_cum0 = float(r0.get("source_grlora_branch_sfo_cum_initial") or 0.0)

# 验证过的解调：header 前 4 个 preamble 符号 + 8 header + 前 8 payload
res = demod_symbol_sequence(
    samples=np.asarray(iq[:hs + 17 * NF], dtype=np.complex64),
    header_start_sample=hs,
    sf=10, os_factor=4,
    cfo_int=cfo_int, cfo_frac=cfo_frac, sfo_hat=sfo_hat, sfo_cum_initial=sfo_cum0,
    header_count=8, payload_count=8, payload_ldro=False)
print("CSV gray_symbols :", r0["gray_symbols"])
print("重放 header 值   :", [x.symbol_value for x in res[:8]])
print("payload 前8 值   :", [x.symbol_value for x in res[8:]])
print("payload raw bin  :", [x.raw_fft_bin for x in res[8:]])

# wm 在同一段上的 argmax
seg = np.asarray(iq[hs - 8 * NF:hs + 17 * NF], dtype=np.complex128)
n = np.arange(len(seg))
for tag, s in [("raw", seg), ("derot", seg * np.exp(-2j * np.pi * cfo_int * n / NF))]:
    wm_arg = [int(np.argmax(ota.wm(s, k)[:, 2])) % 1024 for k in range(8, 24)]
    print("wm %s hdr+pay argmax: %s" % (tag, wm_arg))

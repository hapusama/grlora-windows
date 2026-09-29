# -*- coding: utf-8 -*-
"""2026-09-29 审计：ota_final 深端非单调 + 全链同数的机制定位。

results_final.json 反常：
  1) 8 条链在所有噪声档成功率完全相同；
  2) 深端非单调：+30dB 9/28, +33dB 12/28, +36dB 18/28（噪声越大越好）。
本脚本在单个 capture 上复跑 +30/+33/+36 三档，逐包打印各链结果 + 每包
谱质量（正确符号 bin 的能量排名），定位"通过"到底由什么决定。
"""
import sys
import csv
import importlib.util
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "ota_final",
    r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_replay_20260929\ota_final.py")
ota = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ota)

NF = ota.NF
CHAINS = ota.CHAINS

BIN = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin"
CSV = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_8_header_first_frames.csv"

iq = np.memmap(BIN, dtype=np.complex64, mode="r")
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
        if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
print("frames: %d" % len(rows))

rng = np.random.default_rng(20260929)

def seg_of(r):
    hs = int(r["header_start_sample"])
    psym = int(r["payload_symbol_count"])
    start = hs - 8 * NF
    end = hs + (8 + psym + 1) * NF
    return np.asarray(iq[start:end], dtype=np.complex128)

# 与 ota_final 相同的 rng 消耗顺序：levels 外层、帧内层
for lv in [30.0, 33.0, 36.0]:
    print("\n===== +%gdB =====" % lv)
    for i, r in enumerate(rows):
        seg = seg_of(r)
        p_add = 10 ** (lv / 10.0) * float(np.mean(np.abs(seg) ** 2))
        seg = seg + (rng.standard_normal(len(seg)) + 1j * rng.standard_normal(len(seg))) \
            * np.sqrt(p_add / 2.0)
        out = ota.demod_all(seg, int(r["source_grlora_cfo_int"]),
                            int(r["payload_symbol_count"]), int(r["payload_len"]), int(r["cr"]))
        flags = "".join("1" if out[c] else "." for c in CHAINS)
        # 谱质量：header 第一个符号正确 bin 的 argmax 命中
        print("  f%d len=%s cr=%s psym=%s -> %s" % (i, r["payload_len"], r["cr"], r["payload_symbol_count"], flags))

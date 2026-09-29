# -*- coding: utf-8 -*-
"""2026-09-29 捉虫：Savaux 在 OTA native 崩溃是否因分数同步先验缺失。

对比同一段真实信号的 4 种预纠偏（对全链统一施加）：
  a) 仅整数 CFO（v6 现状）
  b) 整数 + cfo_frac
  c) 整数 + cfo_frac + sto_frac
  d) 整数 + cfo_frac - sto_frac
每帧取前 5 个 payload 符号，报告 SAVAUX/TRIMMER/OLD-A/wm(col2) 的
argmax 与 GT 符号的命中数。无任何合成：信号、先验、GT 全部来自真数据。
"""
import sys
import csv
import importlib.util
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence

_spec = importlib.util.spec_from_file_location(
    "v6", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_replay_20260929\ota_v6_fixed.py")
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)
NF, N, OS = v6.NF, v6.N, v6.OS

from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)

VARIANTS = ["int", "+frac", "+frac+sto", "+frac-sto"]

def derot(seg, r, variant):
    n = np.arange(len(seg))
    cfo_i = int(r["source_grlora_cfo_int"])
    cfo_f = float(r["source_grlora_cfo_frac"])
    sto = float(r["source_grlora_payload_sto_frac"])
    f = cfo_i + cfo_f + (0.0 if variant == "int" else
                         (sto if variant == "+frac+sto" else
                          (-sto if variant == "+frac-sto" else 0.0)))
    return seg * np.exp(-2j * np.pi * f * n / NF)

def main():
    tot = {v: {c: [0, 0] for c in ["SAVAUX", "TRIMMER", "OLD-A", "wm2"]} for v in VARIANTS}
    for cap, bin_path, csv_path in v6.SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            hs = int(r["header_start_sample"]); psym = int(r["payload_symbol_count"])
            gt5 = [x.symbol_value for x in demod_symbol_sequence(
                samples=np.asarray(iq, dtype=np.complex64), header_start_sample=hs,
                sf=10, os_factor=4, cfo_int=int(r["source_grlora_cfo_int"]),
                cfo_frac=float(r["source_grlora_cfo_frac"]),
                sfo_hat=float(r["source_grlora_sfo_hat"]),
                sfo_cum_initial=float(r.get("source_grlora_branch_sfo_cum_initial") or 0.0),
                header_count=8, payload_count=5, payload_ldro=False)[8:]]
            seg0 = np.asarray(iq[hs - 8*NF: hs + (8 + 5 + 1)*NF], dtype=np.complex128)
            for v in VARIANTS:
                seg = derot(seg0, r, v)
                for k in range(5):
                    st = 16 * NF + k * NF
                    sav = int(np.argmax(np.abs(sav_demod(
                        samples=seg, start_sample=st, sf=10, os_factor=4,
                        cfo_int=0).combined_spectrum) ** 2))
                    tri = int(np.argmax(trim_demod(
                        samples=seg, start_sample=st, sf=10, os_factor=4,
                        cfo_int=0).metric))
                    olda = int(np.argmax(np.abs(np.fft.fft(
                        seg[st:st+NF][::OS] * v6.REF_OS[::OS])) ** 2))
                    w2 = int(np.argmax(v6.wm(seg, 16 + k)[:, 2]))
                    for name, a in [("SAVAUX", sav), ("TRIMMER", tri),
                                    ("OLD-A", olda), ("wm2", w2)]:
                        tot[v][name][1] += 1
                        tot[v][name][0] += int(a == gt5[k])
    print("native 命中率（140 符号 = 28 帧 x 5）:")
    for v in VARIANTS:
        row = " | ".join("%s %d/%d" % (c, tot[v][c][0], tot[v][c][1])
                         for c in ["SAVAUX", "TRIMMER", "OLD-A", "wm2"])
        print("  %-10s %s" % (v, row))

if __name__ == "__main__":
    main()

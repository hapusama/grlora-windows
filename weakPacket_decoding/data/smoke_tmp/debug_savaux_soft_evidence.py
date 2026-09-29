# -*- coding: utf-8 -*-
"""2026-09-29 捉虫 II：Savaux 软证据质量 vs TRIMMER。

同一段真实信号（+frac-sto 纠偏）、同一 GT header、同一 SymFEC 裁判：
逐帧对比 argmax 命中 vs SymFEC 软选中符号命中，定位 Savaux 整包崩、
TRIMMER 整包好的机制。另打印一个 Savaux 谱行的副峰结构。
"""
import sys
import csv
import importlib.util
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, build_symfec_symbol_evidence, decode_symfec_payload_from_evidences)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)

_spec = importlib.util.spec_from_file_location(
    "v6", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ota_replay_20260929\ota_v6_fixed.py")
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)
NF, N, OS = v6.NF, v6.N, v6.OS
FEC = SymFECConfig()

def seg_fixed(iq, r, psym):
    hs = int(r["header_start_sample"])
    seg = np.asarray(iq[hs - 8*NF: hs + (8 + psym + 1)*NF], dtype=np.complex128)
    n = np.arange(len(seg))
    f = (int(r["source_grlora_cfo_int"]) + float(r["source_grlora_cfo_frac"])
         - float(r["source_grlora_payload_sto_frac"]))
    return seg * np.exp(-2j * np.pi * f * n / NF)

def chain_rows(seg, psym, which):
    out = np.zeros((psym, N))
    for k in range(psym):
        st = (16 + k) * NF
        if which == "SAVAUX":
            out[k] = np.abs(sav_demod(samples=seg, start_sample=st, sf=10,
                                      os_factor=4, cfo_int=0).combined_spectrum) ** 2
        else:
            out[k] = trim_demod(samples=seg, start_sample=st, sf=10,
                                os_factor=4, cfo_int=0).metric
    return out

def main():
    cap, bin_path, csv_path = v6.SOURCES[0]
    iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
            if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
    for i, r in enumerate(rows):
        psym = int(r["payload_symbol_count"])
        gt_hdr = [x.symbol_value for x in demod_symbol_sequence(
            samples=np.asarray(iq, dtype=np.complex64), header_start_sample=int(r["header_start_sample"]),
            sf=10, os_factor=4, cfo_int=int(r["source_grlora_cfo_int"]),
            cfo_frac=float(r["source_grlora_cfo_frac"]), sfo_hat=float(r["source_grlora_sfo_hat"]),
            sfo_cum_initial=float(r.get("source_grlora_branch_sfo_cum_initial") or 0.0),
            header_count=8, payload_count=psym, payload_ldro=False)[:8]]
        gt = [x.symbol_value for x in demod_symbol_sequence(
            samples=np.asarray(iq, dtype=np.complex64), header_start_sample=int(r["header_start_sample"]),
            sf=10, os_factor=4, cfo_int=int(r["source_grlora_cfo_int"]),
            cfo_frac=float(r["source_grlora_cfo_frac"]), sfo_hat=float(r["source_grlora_sfo_hat"]),
            sfo_cum_initial=float(r.get("source_grlora_branch_sfo_cum_initial") or 0.0),
            header_count=8, payload_count=psym, payload_ldro=False)[8:]]
        seg = seg_fixed(iq, r, psym)
        line = "f%-2d" % i
        for which in ["SAVAUX", "TRIMMER"]:
            rows_ = chain_rows(seg, psym, which)
            d = np.bincount([(int(np.argmax(rows_[k])) - gt[k]) % N for k in range(psym)],
                            minlength=N)
            delta = int(np.argmax(d))
            hard = int(d[delta])
            evs = [build_symfec_symbol_evidence(v6.roll_evidence(rows_, delta)[k], sf=10,
                                                symbol_index=k, ldro=False) for k in range(psym)]
            res = decode_symfec_payload_from_evidences(
                evidences=evs, sf=10, cr=int(r["cr"]), ldro=False, config=FEC,
                header_symbol_values=gt_hdr, payload_len=int(r["payload_len"]),
                has_crc=True, crc_mode="grlora")
            soft = sum(int(s == g) for s, g in zip(res.selected_symbol_values, gt)) if res else 0
            crc = bool(res.payload_decode and res.payload_decode.crc_valid) if res else False
            line += " | %s δ=%d hard %d/%d soft %d/%d crc %d" % (
                which[:3], delta, hard, psym, soft, psym, int(crc))
        print(line, flush=True)
        if i == 0:
            sav_row = chain_rows(seg, 1, "SAVAUX")[0]
            top = np.argsort(sav_row)[::-1][:6]
            print("   SAVAUX sym0 谱 top6 bins/val:", [(int(b), float(sav_row[b]/sav_row.max())) for b in top],
                  " GT=", gt[0])

if __name__ == "__main__":
    main()

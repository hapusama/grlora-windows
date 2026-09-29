# -*- coding: utf-8 -*-
"""2026-09-29 捉虫 III（定案）：亚 chip 时间对齐能否救回 Savaux。

在 +frac CFO 频偏纠正基础上，再做 ±sto_frac chip 的 FFT 分数时延，
彻底对齐符号网格；对比 SAVAUX/TRIMMER 的整包硬命中与 CRC。
若 Savaux 恢复 => v6 崩溃主因是 harness 没喂足先验；
若仍分裂 => 相干合并对先验残差的固有脆弱（真实边界）。
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

def frac_delay(x, samples):
    """FFT 分数时延（右移 samples 个采样点）。"""
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x))
    return np.fft.ifft(X * np.exp(-2j * np.pi * f * samples))

def seg_align(iq, r, psym, sign):
    hs = int(r["header_start_sample"])
    seg = np.asarray(iq[hs - 8*NF: hs + (8 + psym + 1)*NF], dtype=np.complex128)
    n = np.arange(len(seg))
    f = int(r["source_grlora_cfo_int"]) + float(r["source_grlora_cfo_frac"])
    seg = seg * np.exp(-2j * np.pi * f * n / NF)
    sto_chips = float(r["source_grlora_payload_sto_frac"])
    return frac_delay(seg, sign * sto_chips * OS)

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

def gt_of(iq, r, psym):
    res = demod_symbol_sequence(
        samples=np.asarray(iq, dtype=np.complex64), header_start_sample=int(r["header_start_sample"]),
        sf=10, os_factor=4, cfo_int=int(r["source_grlora_cfo_int"]),
        cfo_frac=float(r["source_grlora_cfo_frac"]), sfo_hat=float(r["source_grlora_sfo_hat"]),
        sfo_cum_initial=float(r.get("source_grlora_branch_sfo_cum_initial") or 0.0),
        header_count=8, payload_count=psym, payload_ldro=False)
    return [x.symbol_value for x in res[:8]], [x.symbol_value for x in res[8:]]

def main():
    cap, bin_path, csv_path = v6.SOURCES[0]
    iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
            if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
    for sign, tag in [(-1, "delay=+sto"), (+1, "delay=-sto")]:
        print("=== %s ===" % tag)
        ok = {"SAVAUX": 0, "TRIMMER": 0}
        for i, r in enumerate(rows):
            psym = int(r["payload_symbol_count"])
            gt_hdr, gt = gt_of(iq, r, psym)
            seg = seg_align(iq, r, psym, sign)
            line = "f%-2d" % i
            for which in ["SAVAUX", "TRIMMER"]:
                rows_ = chain_rows(seg, psym, which)
                d = np.bincount([(int(np.argmax(rows_[k])) - gt[k]) % N for k in range(psym)], minlength=N)
                delta = int(np.argmax(d))
                hard = int(d[delta])
                evs = [build_symfec_symbol_evidence(v6.roll_evidence(rows_, delta)[k], sf=10,
                                                    symbol_index=k, ldro=False) for k in range(psym)]
                res = decode_symfec_payload_from_evidences(
                    evidences=evs, sf=10, cr=int(r["cr"]), ldro=False, config=FEC,
                    header_symbol_values=gt_hdr, payload_len=int(r["payload_len"]),
                    has_crc=True, crc_mode="grlora")
                crc = bool(res.payload_decode and res.payload_decode.crc_valid)
                ok[which] += int(crc)
                line += " | %s δ=%d hard %2d/%d crc %d" % (which[:3], delta, hard, psym, int(crc))
            print(line, flush=True)
        print("  小计 crc: SAVAUX %d/10 TRIMMER %d/10" % (ok["SAVAUX"], ok["TRIMMER"]))

if __name__ == "__main__":
    main()

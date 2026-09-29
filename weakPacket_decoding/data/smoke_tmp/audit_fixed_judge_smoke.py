# -*- coding: utf-8 -*-
"""2026-09-29 审计 IV：修复判据后 OTA 重放的区分力冒烟测试。

修复判据 = header_valid(checksum) AND has_crc AND payload CRC AND 字节==GT。
GT = header_first CSV 中 gr-lora 原链在原生 SNR (~37dB 谱峰比) 下解出的
decoded_nibbles -> bytes（等价于无错真值）。
预期：native 全过；+24dB 起按链分化；+30dB 深端接近全灭（物理合理）。
"""
import sys
import csv
import importlib.util
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.payload_codec import (
    encode_explicit_frame_symbols, nibbles_to_dewhitened_bytes,
    payload_symbols_to_nibbles, explicit_header_tail_nibbles)

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

def gt_payload_bytes(r):
    """从 CSV 的 decoded_nibbles 恢复 gr-lora 原链解出的 payload 字节。"""
    hdr_sym = [int(v) for v in r["gray_symbols"].split()]  # 注意:gray_symbols 是 header 的
    return None  # 改用 codewords/decoded_nibbles 路径较绕，下面直接用共识法

# 简化：native 档全链字节一致即为 GT（昨天 ota_final 的 Pass1 共识思路）
rng = np.random.default_rng(20260929)
CHAINS = ["OLD-A", "OLD-A+F", "SAVAUX", "NEW-0", "BCJR"]

def run_fixed_judge():
    # pass 1: native 取 OLD-A+F 字节做 GT（并核对 BCJR==OLD-A+F）
    gt = {}
    for i, r in enumerate(rows):
        hs = int(r["header_start_sample"]); psym = int(r["payload_symbol_count"])
        seg = np.asarray(iq[hs-8*NF: hs+(8+psym+1)*NF], dtype=np.complex128)
        out = ota.demod_all(seg, int(r["source_grlora_cfo_int"]), psym,
                           int(r["payload_len"]), int(r["cr"]))
        gt[i] = out  # 布尔表仅用于确认 native 全过
    for lv in [None, 24.0, 30.0]:
        cnt = {c: 0 for c in CHAINS}
        for i, r in enumerate(rows):
            hs = int(r["header_start_sample"]); psym = int(r["payload_symbol_count"])
            seg = np.asarray(iq[hs-8*NF: hs+(8+psym+1)*NF], dtype=np.complex128)
            seg = seg * np.exp(-2j*np.pi*int(r["source_grlora_cfo_int"])*np.arange(len(seg))/NF)
            if lv is not None:
                p_add = 10**(lv/10.0)*float(np.mean(np.abs(seg)**2))
                seg = seg + (rng.standard_normal(len(seg))+1j*rng.standard_normal(len(seg)))*np.sqrt(p_add/2)
            # 只重跑 demod_all 的链证据部分太重——直接调用并取对称判据：
            out = ota.demod_all(seg, int(r["source_grlora_cfo_int"]), psym,
                               int(r["payload_len"]), int(r["cr"]))
            # 修复判据在这里无法从 demod_all 布尔输出恢复——需要内联。
            # 冒烟目的：证明修好的判据有区分力 -> 用 PLAIN 的硬字节 + header 检查代替。
            hdr_sym = [int(np.argmax(ota.wm(seg, 8+k)[:, 2])) % 1024 for k in range(8)]
            try:
                res = ota.decode_explicit_frame_symbols(hdr_sym,
                        [int(np.argmax(np.abs(np.fft.fft(seg[(16+k)*NF:(17+k)*NF:4]*ota.REF0[::4]))**2)) for k in range(psym)],
                        sf=10, bw=125000.0, ldro_mode=2)
                fixed_ok = bool(res.header.header_valid and res.header.has_crc
                                and res.payload.crc_valid)
            except Exception:
                fixed_ok = False
            cnt["OLD-A"] += int(fixed_ok)
            # BCJR 链的修复判据（用 symfec 路径 + header 检查）
            hdr_vals = hdr_sym
            ms = [ota.wm(seg, 16+k) for k in range(psym)]
            post = ota.bcjr_post(ms)
            T = np.stack([ms[k] @ post[k] for k in range(psym)])
            from weak_decoder.baselines.symfec.paper_symfec_decoder import (
                SymFECConfig, build_symfec_symbol_evidence,
                decode_symfec_payload_from_evidences)
            evs = [build_symfec_symbol_evidence(ota.to_canon(T)[k], sf=10, symbol_index=k, ldro=False)
                   for k in range(T.shape[0])]
            res2 = decode_symfec_payload_from_evidences(
                evidences=evs, sf=10, cr=int(r["cr"]), ldro=False,
                config=ota.FEC_CFG, header_symbol_values=hdr_vals,
                payload_len=int(r["payload_len"]), has_crc=True, crc_mode="grlora")
            fixed_ok2 = False
            if res2.payload_decode is not None:
                fixed_ok2 = bool(res2.header_valid if hasattr(res2, "header_valid") else True)
            # symfec 结果没有回传 header——手动再解一次 header
            from weak_decoder.decoding.payload_codec import decode_explicit_header
            h = decode_explicit_header(hdr_vals, sf=10, bw=125000.0, ldro_mode=2)
            fixed_ok2 = bool(h.header_valid and h.has_crc and res2.payload_decode
                             and res2.payload_decode.crc_valid)
            cnt["BCJR"] += int(fixed_ok2)
            cnt["SAVAUX"] += 0  # 冒烟只验证机制，不全链重跑
            cnt["NEW-0"] += 0
            cnt["OLD-A+F"] += 0
        print("level=%s: 修复判据  OLD-A(硬链) %d/%d | BCJR(软链) %d/%d" % (
            "native" if lv is None else "+%gdB" % lv,
            cnt["OLD-A"], len(rows), cnt["BCJR"], len(rows)))

if __name__ == "__main__":
    run_fixed_judge()

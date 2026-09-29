# -*- coding: utf-8 -*-
"""2026-09-29 审计：复现 ota_replay_20260929 的'CRC 空真'验伪实验。

问题：RESULTS.md 声称"纯随机 payload 谱 + 正确 header -> 3/3 全过 CRC"。
但 payload_codec 的 CRC 是标准独立校验（对解码字节重算 CRC、与解码出的
CRC 字节比对），随机字节通过率应为 2^-16。本脚本裁决谁对：
  A. 随机谱 + 正确 header 过 SymFEC -> CRC 通过率（多次重复，非 3 次）
  B. 对照：正确谱打乱 1 个符号 -> CRC 应翻转
"""
import sys
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.payload_codec import encode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, build_symfec_symbol_evidence, decode_symfec_payload_from_evidences,
    _canonical_raw_bin)

SF, CR, LDR = 10, 4, False
N = 1024
FEC_CFG = SymFECConfig()
rng = np.random.default_rng(20260929)

def evidences_from_spectra(T_rows):
    return [build_symfec_symbol_evidence(T_rows[k], sf=SF, symbol_index=k, ldro=LDR)
            for k in range(T_rows.shape[0])]

def crc_pass(T_rows, hdr_vals, plen):
    evs = evidences_from_spectra(T_rows)
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=CR, ldro=LDR, config=FEC_CFG,
        header_symbol_values=list(hdr_vals), payload_len=plen,
        has_crc=True, crc_mode="grlora")
    if res.payload_decode is None:
        return False, None
    return bool(res.payload_decode.crc_valid), bytes(res.payload_decode.payload_bytes)

def main():
    n_trial = 200
    # 正确 header + 正确 payload 谱（理想尖峰谱）
    payload = bytes(int(v) for v in rng.integers(0, 256, 33))
    hdr_vals, pay_vals = encode_explicit_frame_symbols(
        payload, sf=SF, cr=CR, has_crc=True, ldro=LDR, crc_mode="grlora")

    T_true = np.zeros((len(pay_vals), N))
    for k, v in enumerate(pay_vals):
        T_true[k, v % N] = 1e6
    ok, byt = crc_pass(T_true, hdr_vals, 33)
    print("A0 正确谱+正确header: crc=%s, 字节==TX: %s" % (ok, byt == payload))

    # 纯随机谱（行内 iid 指数分布）
    n_pass = 0
    n_dec = 0
    for t in range(n_trial):
        T_rand = rng.exponential(1.0, size=(len(pay_vals), N))
        ok, _ = crc_pass(T_rand, hdr_vals, 33)
        n_pass += int(ok)
        n_dec += 1
    print("A1 纯随机谱 x%d: CRC 通过 %d 次 (期望~%d if 2^-16)" % (n_trial, n_pass, n_trial / 65536))

    # 随机谱但只随机 payload 数据部分、CRC 符号位置保留真值？
    # （检查另一种可能的空真机制：CRC nibbles 是否来自 header tail）
    n_pass2 = 0
    for t in range(50):
        T_mix = rng.exponential(1.0, size=(len(pay_vals), N))
        ok, _ = crc_pass(T_mix, hdr_vals, 33)
        n_pass2 += int(ok)
    print("A2 再来 50 次: 通过 %d" % n_pass2)

    # B 对照：正确谱、把第一个 payload 符号行换成随机
    n_fail = 0
    for t in range(30):
        T_cor = T_true.copy()
        wrong = int(rng.integers(0, N))
        while wrong == pay_vals[0] % N:
            wrong = int(rng.integers(0, N))
        T_cor[0, :] = 0
        T_cor[0, wrong] = 1e6
        ok, _ = crc_pass(T_cor, hdr_vals, 33)
        n_fail += int(not ok)
    print("B  1符号损坏 x30: CRC 拒绝 %d 次（健康判据应接近 30）" % n_fail)

if __name__ == "__main__":
    main()

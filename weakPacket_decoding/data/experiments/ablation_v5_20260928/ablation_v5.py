# -*- coding: utf-8 -*-
r"""
2026-09-28 v5 流水线正式消融：每个模块砍一刀，其余不动。

链条（同一信号/噪声/真LoRa编码/SymFEC 裁判）：
  FULL    完整方法（前置滤波 + 5列κ-库 + BCJR后验 + 软混合 + 软FEC）
  noFILT  去掉前置带限滤波（宽带臂应大跌、信道臂应≈不变——自洽检验）
  noLIB   κ-库退化成单列(δ=0)（"5个假设"本身的价值）
  noBCJR  前向-后向→逐符号贪心选列（跨符号联合的价值）
  noMIX   后验混合→MAP 硬列（对κ求和 vs 求最大的价值）
  noSOFT  软谱→one-hot 硬证据（软证据喂FEC的价值）
demod 用批量化版（与逐次调用版逐位等价，已验证）。100 包/点，CI≈±0.10。
"""
import sys
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.payload_codec import encode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, build_symfec_symbol_evidence, decode_symfec_payload_from_evidences)
import importlib.util
_aspec = importlib.util.spec_from_file_location(
    "arb", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\arbitration_20260928\arbitration.py")
arb = importlib.util.module_from_spec(_aspec); _aspec.loader.exec_module(arb)

N, OS, NF = arb.N, arb.OS, arb.NF
SF, CR, LDR = 10, 4, False
PRE_N, GUARD, PRE_AMBLE_S = 8, 1, 512
RNG = np.random.default_rng(20260928)
FEC_CFG = SymFECConfig()
ROT = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)])

def tx_wave(u, v):
    return arb.tx_wave(u, v)

def wm(x, k, prefilter=True):
    """批量化 κ-库（与 arbitration 逐次版逐位等价）。"""
    seg = x[k * NF:(k + 1) * NF]
    if prefilter:
        seg = np.fft.ifft(np.fft.fft(seg) * arb.CHAN_MASK)
    y = seg * arb.REF0
    chips = np.stack([y[p:p + NF:OS] for p in range(OS)])            # (4,N)
    inp = (chips[None, :, :] * ROT[:, None, :]).reshape(20, N)       # (20,N)
    P = np.abs(np.fft.fft(inp, axis=1)) ** 2
    return P.reshape(5, 4, N).mean(axis=1).T                          # (N,5)

def prominence(m):
    return np.log(np.maximum(m.max(axis=0) - np.median(m, axis=0), 1e-30))

def bcjr(ms):
    return arb.bcjr_post(ms)

def symfec(Tb, hdr_vals, payload):
    Tb = Tb[:, (np.arange(N) - 1) % N]
    evs = [build_symfec_symbol_evidence(Tb[k], sf=SF, symbol_index=k, ldro=LDR)
           for k in range(Tb.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=CR, ldro=LDR, config=FEC_CFG,
        header_symbol_values=list(hdr_vals), payload_len=len(payload),
        has_crc=True, crc_mode="grlora")
    return (res.payload_decode is not None
            and bytes(res.payload_decode.payload_bytes) == bytes(payload))

def run_packet(arm, kap0, sigma2):
    payload = bytes(RNG.integers(0, 256, 1))
    hdr_vals, pay_vals = encode_explicit_frame_symbols(
        payload, sf=SF, cr=CR, has_crc=True, ldro=LDR, crc_mode="grlora")
    n_pay = len(pay_vals)
    total = PRE_N + len(hdr_vals) + n_pay + GUARD
    kaps = (np.full(total, kap0) if kap0 is not None
            else 0.1 + 0.004 * np.arange(total))
    seq = [0] * PRE_N + list(hdr_vals) + list(pay_vals) + [0]
    x = np.concatenate([tx_wave(arb.U_GRID - kaps[i], seq[i]) for i in range(total)])
    x = x + arb.make_noise(arm, sigma2, RNG, length=x.size)
    off = PRE_N + len(hdr_vals)

    ms = [wm(x, off + k) for k in range(n_pay)]
    post = bcjr(ms)
    T_full = np.stack([ms[k] @ post[k] for k in range(n_pay)])
    ms_nf = [wm(x, off + k, prefilter=False) for k in range(n_pay)]
    T_nofilt = np.stack([ms_nf[k] @ bcjr(ms_nf)[k] for k in range(n_pay)])
    T_nolib = np.stack([m[:, 2] for m in ms])
    T_nobcjr = np.stack([ms[k][:, int(np.argmax(prominence(ms[k])))] for k in range(n_pay)])
    T_nomix = np.stack([ms[k][:, int(np.argmax(post[k]))] for k in range(n_pay)])
    am = T_full.argmax(axis=1)
    T_nosoft = np.full_like(T_full, 1.0)
    T_nosoft[np.arange(n_pay), am] = 1e6

    out = {}
    for name, T in [("FULL", T_full), ("noFILT", T_nofilt), ("noLIB", T_nolib),
                    ("noBCJR", T_nobcjr), ("noMIX", T_nomix), ("noSOFT", T_nosoft)]:
        out[name] = symfec(T, hdr_vals, payload)
    return out

def selftest():
    print("SELFTEST: wide +10dB 全臂应 PDR=1")
    for kap in (0.0, 0.25):
        res = run_packet("wide", kap, 10 ** (-10 / 10.0))
        print("  kap=%.2f: %s" % (kap, res))
        if not all(res.values()):
            raise SystemExit("SELFTEST FAILED")

def main():
    selftest()
    n_pkt = 100
    snrs = [-28, -26]
    kap_modes = {"k25": 0.25, "k50": 0.5, "drift": None}
    arms = ["wide", "chan"]
    chains = ["FULL", "noFILT", "noLIB", "noBCJR", "noMIX", "noSOFT"]
    results = {}
    t0 = time.time()
    for km, kap0 in kap_modes.items():
        for arm in arms:
            for snr_db in snrs:
                sigma2 = 10 ** (-snr_db / 10.0)
                cnt = {c: 0 for c in chains}
                for _ in range(n_pkt):
                    for c, ok in run_packet(arm, kap0, sigma2).items():
                        cnt[c] += int(ok)
                results.setdefault(km, {}).setdefault(arm, {})[str(snr_db)] = {
                    c: cnt[c] / n_pkt for c in chains}
                row = results[km][arm][str(snr_db)]
                print("[{:5s}/{:4s}]{:>4}: ".format(km, arm, snr_db) + " | ".join(
                    "{} {:.2f}".format(c, row[c]) for c in chains), flush=True)
    # 输出路径为项目内字面量（无拼接、不含上跳段），直接内联供静态检查
    with open(r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ablation_v5_20260928\results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("\n%.0fs elapsed -> results.json" % (time.time() - t0))

if __name__ == "__main__":
    main()

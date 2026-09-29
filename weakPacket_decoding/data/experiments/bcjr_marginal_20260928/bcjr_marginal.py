# -*- coding: utf-8 -*-
r"""
2026-09-28 BCJR 边际化验证："估计死区里不估、直接边际化"主线的核心方法实验。

对比（同一信号/噪声/真LoRa编码/SymFEC 裁判）：
  NEW-0    细网格精确读（无 κ 处理；半bin处软平局证据的参照）
  TREL     κ-格 Viterbi 硬路径 + 镜像仲裁（当前最好）
  BCJR     κ-格前向-后向软输出边际化（新）：逐符号平滑 κ 后验，
           T(s) = Σ_d P(κ_d|全包)·m[s,d] —— 对滋扰求和而非求最大
  ORACLE   真 κ 相干（上界）

判据：
  ① κ=0.5 浅端回退是否消失（TREL 0.68@-22 vs NEW-0 0.98 的缺口）
  ② 深端增益是否保留（-24 处 TREL 0.625）
  ③ 常规模式（k25/drift）是否不劣于 TREL

包结构与 symfec_pk 相同：8 preamble(512) + 8 header(真值) + payload(1B→8符号) + guard。
发射约定/裁判/白化全部同 symfec_pk_20260928。80 包/点。
"""
import sys
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.payload_codec import encode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, build_symfec_symbol_evidence, decode_symfec_payload_from_evidences,
    _canonical_raw_bin)

N = 1024
OS = 4
NF = N * OS
M_PAD = 16 * N
PRE_N = 8
PRE_AMBLE_S = 512
GUARD = 1
SF, CR, LDR = 10, 4, False
BASE2VAL = (512 - np.arange(N)) % N
RNG = np.random.default_rng(20260928)
FEC_CFG = SymFECConfig()

U_GRID = np.arange(NF) / OS
REF_OS = np.exp(-1j * 2 * np.pi * U_GRID ** 2 / (2 * N))

def sym_wave(u, s):
    return np.exp(1j * 2 * np.pi * (u ** 2 / (2 * N) - s * u / N))

def val_to_my_s(values):
    out = []
    for v in values:
        b = _canonical_raw_bin(int(v), sf=SF, is_header=False, ldro=LDR)
        out.append((512 - b) % N)
    return np.array(out, dtype=int)

def make_packet(s_all, kaps, sigma2):
    x = np.concatenate([sym_wave(U_GRID - kaps[k], s_all[k]) for k in range(len(s_all))])
    return x + (RNG.standard_normal(x.size) + 1j * RNG.standard_normal(x.size)) * np.sqrt(sigma2 / 2)

def window_matrix(x, k):
    """返回 (N,5) 细网格窗口能量；列 d ↔ κ=(2-d)/4；行=假设符号值。"""
    y = x[k * NF:(k + 1) * NF] * REF_OS
    Z = np.fft.fft(y, n=M_PAD)
    mag2 = np.abs(Z) ** 2
    base = (-4 * np.arange(N)) % M_PAD
    win = (base[:, None] + np.arange(-2, 3)[None, :]) % M_PAD
    return mag2[win]

def viterbi_kappa(ms):
    nsym = len(ms)
    logr = np.array([np.log(ms[j].max(axis=0) + 1e-30) for j in range(nsym)])
    acc = logr[0].copy()
    back = np.zeros((nsym, 5), dtype=int)
    for j in range(1, nsym):
        cand = np.stack([acc[np.maximum(np.arange(5) - 1, 0)], acc, acc[np.minimum(np.arange(5) + 1, 4)]])
        back[j] = np.array([np.argmax(cand[:, d]) for d in range(5)]) - 1
        acc = cand[np.argmax(cand, axis=0), np.arange(5)] + logr[j]
    path = np.zeros(nsym, dtype=int)
    path[-1] = int(np.argmax(acc))
    for j in range(nsym - 1, 0, -1):
        path[j - 1] = np.clip(path[j] + back[j, path[j]], 0, 4)
    return path

def _lse(x):
    m = x.max()
    return m + np.log(np.sum(np.exp(x - m)))

def bcjr_kappa(ms):
    """前向-后向：返回 (K,5) 逐符号平滑 κ 后验。转移 |Δd|<=1 均匀。"""
    K = len(ms)
    lam = np.array([np.log(ms[k].max(axis=0) + 1e-30) for k in range(K)])  # 发射对数似然(∝)
    alpha = np.zeros((K, 5))
    alpha[0] = lam[0]
    for k in range(1, K):
        for d in range(5):
            lo, hi = max(0, d - 1), min(4, d + 1)
            alpha[k, d] = lam[k, d] + _lse(alpha[k - 1, lo:hi + 1])
    beta = np.zeros((K, 5))
    for k in range(K - 1, 0, -1):
        for dp in range(5):
            lo, hi = max(0, dp - 1), min(4, dp + 1)
            beta[k - 1, dp] = _lse(lam[k, lo:hi + 1] + beta[k, lo:hi + 1])
    post = alpha + beta
    post = np.exp(post - post.max(axis=1, keepdims=True))
    return post / post.sum(axis=1, keepdims=True)

def symfec_decode(Tv_bin, hdr_vals, tx_payload):
    evs = [build_symfec_symbol_evidence(Tv_bin[k], sf=SF, symbol_index=k, ldro=LDR)
           for k in range(Tv_bin.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=CR, ldro=LDR, config=FEC_CFG,
        header_symbol_values=list(hdr_vals), payload_len=len(tx_payload),
        has_crc=True, crc_mode="grlora")
    ok = (res.payload_decode is not None
          and bytes(res.payload_decode.payload_bytes) == bytes(tx_payload))
    hard = np.mean([int(e.argmax_symbol_value) for e in evs])
    return ok, evs

def run_packet(kap0, sigma2):
    payload = bytes(RNG.integers(0, 256, 1))
    hdr_vals, pay_vals = encode_explicit_frame_symbols(
        payload, sf=SF, cr=CR, has_crc=True, ldro=LDR, crc_mode="grlora")
    hdr_s = val_to_my_s(hdr_vals); pay_s = val_to_my_s(pay_vals)
    n_pay = len(pay_vals)
    total = PRE_N + len(hdr_s) + n_pay + GUARD
    kaps = (np.full(total, kap0) if kap0 is not None
            else 0.1 + 0.004 * np.arange(total))
    s_all = np.concatenate([np.full(PRE_N, PRE_AMBLE_S), hdr_s, pay_s,
                            np.full(GUARD, PRE_AMBLE_S)])
    x = make_packet(s_all, kaps, sigma2)
    off = PRE_N + len(hdr_s)

    ms = [window_matrix(x, off + k) for k in range(n_pay)]
    # NEW-0: 精确读 col2
    T_new0 = np.stack([m[:, 2] for m in ms])
    # TREL: viterbi 路径列 + 镜像仲裁
    path = viterbi_kappa(ms)
    T_trel = np.stack([ms[k][:, path[k]] for k in range(n_pay)])
    mirror = None
    if np.mean(path == 0) > 0.5:
        mirror = np.stack([ms[k][(np.arange(N) - 1) % N, 4] for k in range(n_pay)])
    # BCJR: 后验加权混合
    post = bcjr_kappa(ms)
    T_bcjr = np.stack([ms[k] @ post[k] for k in range(n_pay)])
    # ORACLE
    T_orc = []
    for k in range(n_pay):
        y = x[(off + k) * NF:(off + k + 1) * NF] * np.exp(
            -1j * 2 * np.pi * (U_GRID - kaps[off + k]) ** 2 / (2 * N))
        T_orc.append(np.abs(np.fft.fft(y, n=M_PAD)[(-4 * np.arange(N)) % M_PAD]) ** 2)
    T_orc = np.stack(T_orc)

    out = {}
    for name, T in [("NEW-0", T_new0), ("TREL", T_trel), ("BCJR", T_bcjr), ("ORACLE", T_orc)]:
        Tb = T[:, BASE2VAL]
        ok, _ = symfec_decode(Tb, hdr_vals, payload)
        if name == "TREL" and mirror is not None and not ok:
            ok2, _ = symfec_decode(mirror[:, BASE2VAL], hdr_vals, payload)
            ok = ok or ok2
        out[name] = ok
    return out

def selftest():
    print("SELFTEST: kappa=0/0.25, +10 dB，四链应全 PDR=1")
    for kap in (0.0, 0.25):
        res = run_packet(kap, 10 ** (-10 / 10.0))
        print("  kap=%.2f: %s" % (kap, res))
        if not all(res.values()):
            raise SystemExit("SELFTEST FAILED")

def main():
    selftest()
    n_pkt = 80
    snrs = [-26, -24, -22, -20]
    kap_modes = {"k25": 0.25, "k50": 0.5, "drift": None}
    chains = ["NEW-0", "TREL", "BCJR", "ORACLE"]
    results = {}
    t0 = time.time()
    for km, kap0 in kap_modes.items():
        for snr_db in snrs:
            sigma2 = 10 ** (-snr_db / 10.0)
            cnt = {c: 0 for c in chains}
            for _ in range(n_pkt):
                for c, ok in run_packet(kap0, sigma2).items():
                    cnt[c] += int(ok)
            results.setdefault(km, {})[str(snr_db)] = {
                c: cnt[c] / n_pkt for c in chains}
            row = results[km][str(snr_db)]
            print("[{:5s}]{:>4}: ".format(km, snr_db) + " | ".join(
                "{} {:.3f}".format(c, row[c]) for c in chains), flush=True)
    # 输出路径为项目内字面量（无拼接、不含上跳段），直接内联供静态检查
    with open(r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\bcjr_marginal_20260928\results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("\n%.0fs elapsed -> results.json" % (time.time() - t0))

if __name__ == "__main__":
    main()

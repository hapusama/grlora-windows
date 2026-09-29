# -*- coding: utf-8 -*-
r"""
2026-09-28 裁决实验：统一 gr-lora 波形约定 + 双噪声臂 + 滤波基线。
回答两个决定论文命运的问题：
  Q1 排名存活吗：标准信道滤波噪声（chan 臂）下，四基线 vs 我方 BCJR 的排名？
  Q2 增益归属：宽带臂（wide，部署链现实）内，OLD-A+F（接收侧加理想信道滤波）
     能追回多少？我方对最强基线的优势在 chan 臂还剩多少？

统一约定：TX 用 build_upchirp 的连续形式（符号恒在 ±BW/2，fold 在 N−v），
所有谱都在 gr-lora bin 约定，SymFEC evidence 原生消费（无 512 置换）。
噪声臂（OLD-A 抽取后方差 = σ² 口径不变）：
  wide  os 域 iid（前端未滤波 = 部署链现实，collector 无 LPF 已核实）
  chan  |f|<=BW/2 理想砖墙（标准 LoRa 信道滤波前端；本约定下信号恒在带内，臂合法）
链条：OLD-A / OLD-A+F / SAVAUX / TRIMMER / UNICHIRP / NEW-0 / BCJR / ORACLE。
自检门禁：κ=0/0.25 +10dB 全链 PDR=1 才开跑。60 包/点。
"""
import sys
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.payload_codec import encode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, build_symfec_symbol_evidence, decode_symfec_payload_from_evidences)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.unichirp.paper_unichirp_demod import (
    UniChirpTrainingSymbol, UniChirpDemodConfig, build_unichirp_phase_model,
    demod_unichirp_symbol as uni_demod)

N = 1024
OS = 4
NF = N * OS
M_PAD = 16 * N
PRE_N = 8
GUARD = 1
SF, CR, LDR = 10, 4, False
RNG = np.random.default_rng(20260928)
FEC_CFG = SymFECConfig()
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")
FREQS = np.fft.fftfreq(NF)
CHAN_MASK = np.abs(FREQS) <= 0.125 + 1e-12   # |f|<=BW/2 @OS=4

U_GRID = np.arange(NF) / OS
REF = np.conj(np.exp(2j * np.pi * _ph(U_GRID, 0))) if False else None  # 见下

def _ph(t, v):
    """gr-lora 约定 chirp 连续相位：斜率连续、线性项在 t=N−v 分支（无 mod、无相位跳变）。

    与 build_upchirp 在 os 网格逐点一致，且是其向分数时刻的正确连续延拓
    （mod 折叠式在分数时刻差一个连续相位项——曾致 κ≠0 全链失效）。
    """
    lin = np.where(t < (N - v), (v / N - 0.5) * t, (v / N - 1.5) * t)
    return t ** 2 / (2 * N) + lin

def tx_wave(u, v):
    return np.exp(2j * np.pi * _ph(u, v))

REF0 = np.conj(tx_wave(U_GRID, 0))            # 去斜参考（gr-lora 约定）

def make_noise(arm, sigma2, rng, length=None):
    length = length or NF * 40
    w = (rng.standard_normal(length) + 1j * rng.standard_normal(length))
    if arm == "chan":
        w = np.fft.ifft(np.fft.fft(w) * np.tile(CHAN_MASK, length // NF + 1)[:length])
    # 校准：OLD-A 相位0抽取方差 = σ²
    d = w[::OS]
    return w * np.sqrt(sigma2 / (np.mean(np.abs(d) ** 2) + 1e-30))

HANN = np.ones(NF)   # 矩形窗：旁瓣问题已由逐行模板能量归一化根治，Hann 白损1.3dB
_LAM_TABLE = None

def _build_lam_table():
    """加窗匹配滤波的 (N,5) 两抽头复模板 A1/A2 与能量表（一次性干净仿真）。

    T(v,d) = |A1*·Z[j1] + A2*·Z[j2]|² / (|A1|²+|A2|²)——直接存抽头、不做
    λ 除法提取（λ 形式在 |A1|→0 的行会爆炸，v=512 恰差 2 倍的教训）。
    """
    global _LAM_TABLE
    a1 = np.zeros((N, 5), dtype=complex)
    a2 = np.zeros((N, 5), dtype=complex)
    j1_all = np.zeros((N, 5), dtype=int)
    j2_all = np.zeros((N, 5), dtype=int)
    for d in range(5):
        delta = (2 - d) / 4.0
        vv = np.arange(N)
        j1 = (4 * vv - (2 - d)) % M_PAD
        j2 = (j1 - 4 * N) % M_PAD
        j1_all[:, d] = j1; j2_all[:, d] = j2
        for lo in range(0, N, 128):
            hi = min(N, lo + 128)
            sims = np.stack([tx_wave(U_GRID - delta, v) for v in range(lo, hi)])
            Z = np.fft.fft(sims * (REF0 * HANN)[None, :], n=M_PAD, axis=1)
            r = np.arange(hi - lo)
            a1[lo:hi, d] = Z[r, j1[lo:hi]]
            a2[lo:hi, d] = Z[r, j2[lo:hi]]
    _LAM_TABLE = (a1, a2, j1_all, j2_all)

ROT = [np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)]

def window_matrix(x, k):
    """(N,5) 旋转库 κ-检测（带限前置 + chip 域折叠混叠相干 + 4相位功率平均）。

    v4：两抽头补零域只承载完整匹配滤波 ~12% 能量（短窗折叠段主瓣极宽），
    白损 ~9dB。chip N-FFT 的整数采样混叠天然把折叠双段相干叠加（OLDAF 同
    水平统计量对分数 δ 的推广）；带限前置补齐基线分支结构等效拥有的 +6dB。
    行 = gr-lora bin（测量域），列 d ↔ κ=(2−d)/4。
    """
    seg = x[k * NF:(k + 1) * NF]
    seg = np.fft.ifft(np.fft.fft(seg) * CHAN_MASK)   # 必须先滤原始流：去斜是时变调制，后置掩膜会把带外噪声搬进带内（无效）
    y = seg * REF0
    m = np.zeros((N, 5))
    for p in range(OS):
        chip = y[p:p + NF:OS]
        for d in range(5):
            m[:, d] += np.abs(np.fft.fft(chip * ROT[d])) ** 2
    return m / OS

def _lse(x):
    m = x.max()
    return m + np.log(np.sum(np.exp(x - m)))

def bcjr_post(ms):
    K = len(ms)
    # 发射=峰突出度（max−median）：列间区分度远强于裸 max
    lam = np.array([np.log(np.maximum(ms[k].max(axis=0) - np.median(ms[k], axis=0), 1e-30))
                    for k in range(K)])
    alpha = np.zeros((K, 5)); alpha[0] = lam[0]
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

def symfec(Tb, hdr_vals, payload):
    # gr-lora 值约定：canonical bin = value+1（bin_to_grlora_symbol(b)=(b-1)）；
    # 各链实测谱的音在 bin=v → 统一右移一格进 canonical 域。
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
    x = np.concatenate([tx_wave(U_GRID - kaps[i], seq[i]) for i in range(total)])
    x = x + make_noise(arm, sigma2, RNG, length=x.size)
    off = PRE_N + len(hdr_vals)

    out = {}
    # 我方细网格族
    ms = [window_matrix(x, off + k) for k in range(n_pay)]
    out["NEW-0"] = symfec(np.stack([m[:, 2] for m in ms]), hdr_vals, payload)
    post = bcjr_post(ms)
    out["BCJR"] = symfec(np.stack([ms[k] @ post[k] for k in range(n_pay)]), hdr_vals, payload)
    # OLD-A / OLD-A+F
    def olda(k, prefilter):
        seg = x[(off + k) * NF:(off + k + 1) * NF].copy()
        if prefilter:
            seg = np.fft.ifft(np.fft.fft(seg) * CHAN_MASK)
        return np.abs(np.fft.fft(seg[::OS] * REF0[::OS])) ** 2
    out["OLD-A"] = symfec(np.stack([olda(k, False) for k in range(n_pay)]), hdr_vals, payload)
    out["OLD-A+F"] = symfec(np.stack([olda(k, True) for k in range(n_pay)]), hdr_vals, payload)
    # 三基线
    sav = np.zeros((n_pay, N)); tri = np.zeros((n_pay, N)); uni = np.zeros((n_pay, N))
    for k in range(n_pay):
        st = (off + k) * NF
        sav[k] = np.abs(sav_demod(samples=x, start_sample=st, sf=SF, os_factor=OS).combined_spectrum) ** 2
        tri[k] = trim_demod(samples=x, start_sample=st, sf=SF, os_factor=OS).metric
    train = tuple(UniChirpTrainingSymbol(start_sample=k * NF, raw_fft_bin=0, abs_symbol_index=float(k))
                  for k in range(PRE_N))
    model, _o = build_unichirp_phase_model(samples=x, training_symbols=train, sf=SF,
                                           os_factor=OS, config=UNI_CFG)
    for k in range(n_pay):
        r = uni_demod(samples=x, start_sample=(off + k) * NF, sf=SF, os_factor=OS,
                      phase_rad=model.predict(off + k), config=UNI_CFG)
        uni[k] = r.metric
    out["SAVAUX"] = symfec(sav, hdr_vals, payload)
    out["TRIMMER"] = symfec(tri, hdr_vals, payload)
    out["UNICHIRP"] = symfec(uni, hdr_vals, payload)
    return out

def selftest():
    print("SELFTEST: wide 臂, +10 dB, 全链应 PDR=1")
    res = run_packet("wide", 0.0, 10 ** (-10 / 10.0))
    print("  kap=0.00:", res)
    if not all(res.values()):
        raise SystemExit("SELFTEST A FAILED")
    res = run_packet("wide", 0.25, 10 ** (-10 / 10.0))
    print("  kap=0.25:", res)
    if not all(res.values()):
        raise SystemExit("SELFTEST B FAILED")

def main():
    selftest()
    n_pkt = 60
    snrs = [-34, -32, -30, -28, -26]
    kap_modes = {"k25": 0.25, "k50": 0.5, "drift": None}
    arms = ["wide", "chan"]
    chains = ["OLD-A", "OLD-A+F", "SAVAUX", "TRIMMER", "UNICHIRP", "NEW-0", "BCJR"]
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
    with open(r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\arbitration_20260928\results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("\n%.0fs elapsed -> results.json" % (time.time() - t0))

if __name__ == "__main__":
    main()

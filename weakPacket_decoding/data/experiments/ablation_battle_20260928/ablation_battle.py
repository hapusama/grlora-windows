# -*- coding: utf-8 -*-
r"""
2026-09-28 验证脚本 III：消融（C1/C2a/C2b/Trellis/软FEC）+ 与仓库基线 battle。

我方链（值域 T，直接可比）：
  OLD-A  chip 网格单相位 + N-FFT                    （frame_sync 风格）
  OLD-B  4 相位功率平均                              （朴素副本叠加）
  NEW-0  OS 去斜 + 补零 16N 细网格，只在 −4s 精确读  （C1：相位对齐细网格，无 κ 处理）
  NEW-1  + ±2 细bin 窗 max                           （C2a：免估计 κ-GLRT）
  NEW-2  + 幅值比 κ̂ + Dirichlet 相干重组              （C2b：反相指纹）
  TREL-N NEW 细网格 + 5 态 κ 漂移 Viterbi            （跨符号联合，coded_trellis 家族代表）
  ORACLE 真κ 相干 ML（上界）

仓库基线（真导入，bin 域，经 (512−bin)%N 映射到值域）：
  SAVAUX  weak_decoder/baselines/savaux_oversampled（TIM'22 Eq.34-37 branch 相位合并）
  TRIMMER weak_decoder/baselines/loratrimmer（MobiCom'24 切分能量凝聚）
  软证据：SAVAUX 用 combined_spectrum 功率，TRIMMER 用逐候选 metric。

条件完全一致：同一信号、同一噪声、同一 FEC stand-in（Chase-16 软 / 最近码字硬）。
可控量：κ ∈ {0, 0.25, 0.5, drift(+0.02/sym from 0.1)}；SF10/OS=4；120 包/点。
"""
import sys
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
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
PAY_START = 8          # 前 8 个 preamble 符号（值 512=对方约定 bin0），payload 从第 8 个起
GUARD = 1              # 包尾 guard 符号（UniChirp 的带限滤波需要符号后余量）
PRE_AMBLE_S = 512      # 我方约定下使对方 raw_fft_bin=0 的符号值
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")  # 本测试床无整数 CFO，禁用其连续校正
RNG = np.random.default_rng(20260928)
BIN2VAL = (-np.arange(N)) % N          # OLD 链 bin->值
BASE2VAL = (512 - np.arange(N)) % N    # 仓库基线 bin->值（探针校准常数 512）

U_GRID = np.arange(NF) / OS
REF_OS = np.exp(-1j * 2 * np.pi * U_GRID ** 2 / (2 * N))
DOWN_CHIP = np.exp(-1j * np.pi * np.arange(N) ** 2 / N)
M_PAD = 16 * N

def sym_wave(u, s):
    return np.exp(1j * 2 * np.pi * (u ** 2 / (2 * N) - s * u / N))

def dirich(x, L, M):
    x = np.asarray(x, dtype=float)
    a = np.pi * x / M
    sa = np.sin(a)
    ok = np.abs(sa) > 1e-12
    val = np.where(ok, np.sin(a * L) / (L * np.where(ok, sa, 1.0)), 1.0)
    return np.exp(1j * np.pi * x * (L - 1) / M) * val

def gray_decode(v):
    v = v ^ (v >> 16); v = v ^ (v >> 8); v = v ^ (v >> 4); v = v ^ (v >> 2); v = v ^ (v >> 1)
    return v

VAL = np.array([gray_decode(s) for s in np.arange(N)])
BITMASK = np.stack([(VAL >> i) & 1 for i in range(10)], axis=0).astype(bool)

CW = []
for d in range(16):
    b = np.array([(d >> i) & 1 for i in range(4)], dtype=np.int8)
    cw = np.concatenate([b, [b[0] ^ b[1] ^ b[2], b[1] ^ b[2] ^ b[3], b[0] ^ b[2] ^ b[3]]])
    cw = np.concatenate([cw, [cw.sum() & 1]])
    CW.append(cw)
CW = np.array(CW)
NSYM_PKT = 8

def make_packet(s_vals, kaps, sigma2):
    x = np.concatenate([sym_wave(U_GRID - kaps[k], s_vals[k]) for k in range(len(s_vals))])
    return x + (RNG.standard_normal(x.size) + 1j * RNG.standard_normal(x.size)) * np.sqrt(sigma2 / 2)

def bit_llrs(T):
    eps = 1e-9 * T.max(axis=1, keepdims=True) + 1e-30
    num = np.tensordot(BITMASK.astype(float), T + eps, axes=(1, 1))
    den = np.tensordot((~BITMASK).astype(float), T + eps, axes=(1, 1))
    return (np.log(num) - np.log(den)).T

def fec_pdr(llrs, payload):
    ts = th = 0
    for c in range(10):
        l = llrs[:, c]
        hard = (l > 0).astype(np.int8)
        d_soft = CW[int(np.argmax((2 * CW.astype(float) - 1) @ l))]
        d_hard = CW[int(np.argmin((CW != hard).sum(axis=1)))]
        ts += int(np.sum(d_soft[:4] == payload[c]))
        th += int(np.sum(d_hard[:4] == payload[c]))
    return ts, th

def gen_payload():
    return RNG.integers(0, 2, (10, 4))

def bits_to_symbols(payload):
    code = CW[payload @ np.array([1, 2, 4, 8])]
    flat = code.reshape(-1)
    sb = np.zeros((NSYM_PKT, 10), dtype=np.int64)
    for c in range(10):
        for j in range(NSYM_PKT):
            sb[j, c] = flat[c * 8 + j]
    vals = (sb * (1 << np.arange(10))).sum(axis=1)
    return vals ^ (vals >> 1)

# ---------- 我方接收机的每符号输出 ----------
def my_new_stats(x, k):
    """返回 (T0_exact, T1_winmax, T2_fingerprint, m_window)。值域索引。"""
    y = x[k * NF:(k + 1) * NF] * REF_OS
    Z = np.fft.fft(y, n=M_PAD)
    mag2 = np.abs(Z) ** 2
    base = (-4 * np.arange(N)) % M_PAD
    win = (base[:, None] + np.arange(-2, 3)[None, :]) % M_PAD
    m = mag2[win]                                   # (N,5) 列 d ↔ κ=(2−d)/4
    row = np.arange(N)
    T0 = m[:, 2]
    T1 = m.max(axis=1)
    d_star = np.argmax(m, axis=1)
    left = np.where(d_star > 0, m[row, np.maximum(d_star - 1, 0)], -1.0)
    right = np.where(d_star < 4, m[row, np.minimum(d_star + 1, 4)], -1.0)
    use_right = right >= left
    m2 = np.where(use_right, right, left)
    j1 = win[row, d_star]
    j2 = np.where(use_right, win[row, np.minimum(d_star + 1, 4)],
                  win[row, np.maximum(d_star - 1, 0)])
    delta = np.clip(m2 / (T1 + 1e-12), 0.0, 0.9)
    x1 = np.where(use_right, delta, -delta)
    x2 = x1 - np.where(use_right, 1.0, -1.0)
    w1 = np.conj(dirich(x1, NF, M_PAD)); w2 = np.conj(dirich(x2, NF, M_PAD))
    T2 = np.abs(w1 * Z[j1] + w2 * Z[j2]) ** 2 / (np.abs(w1) ** 2 + np.abs(w2) ** 2 + 1e-30)
    return T0, T1, T2, m

def viterbi_kappa(ms):
    """ms: (NSYM, N, 5) 窗口能量。返回逐符号 κ* 列索引路径。"""
    nsym = len(ms)
    logr = np.array([np.log(ms[j].max(axis=0) + 1e-30) for j in range(nsym)])  # (NSYM,5)
    acc = logr[0].copy()
    back = np.zeros((nsym, 5), dtype=int)
    for j in range(1, nsym):
        cand = np.stack([acc[np.maximum(np.arange(5) - 1, 0)],
                         acc,
                         acc[np.minimum(np.arange(5) + 1, 4)]])   # |Δd|<=1
        choice = np.argmax(cand, axis=0)
        back[j] = np.array([np.argmax(cand[:, d]) for d in range(5)]) - 1
        acc = cand[choice, np.arange(5)] + logr[j]
    path = np.zeros(nsym, dtype=int)
    path[-1] = int(np.argmax(acc))
    for j in range(nsym - 1, 0, -1):
        path[j - 1] = np.clip(path[j] + back[j, path[j]], 0, 4)
    return path

def main():
    n_pkt = 120
    snrs = [-26, -24, -22, -20, -18, -16]
    kap_modes = {"k0": 0.0, "k25": 0.25, "k50": 0.5, "drift": None}
    chains = ["OLD-A", "OLD-B", "SAVAUX", "TRIMMER", "UNICHIRP",
              "NEW-0", "NEW-1", "NEW-2", "TREL-N", "ORACLE"]
    results = {}
    t_start = time.time()

    for km, kap0 in kap_modes.items():
        for snr_db in snrs:
            sigma2 = 10 ** (-snr_db / 10.0)
            err = {c: 0 for c in chains}
            pdr_s = {c: 0 for c in chains}
            pdr_h = {c: 0 for c in chains}
            n_sym = 0
            for _ in range(n_pkt):
                payload = gen_payload()
                s_vals = bits_to_symbols(payload)
                if kap0 is not None:
                    kaps_all = np.full(PAY_START + NSYM_PKT + GUARD, kap0)
                else:
                    kaps_all = 0.1 + 0.02 * np.arange(PAY_START + NSYM_PKT + GUARD)
                kaps_pay = kaps_all[PAY_START:PAY_START + NSYM_PKT]
                # 整包 = 8 preamble(值512,已知) + 8 payload + 1 guard；κ 连续作用于全包
                s_all = np.concatenate([np.full(PAY_START, PRE_AMBLE_S), s_vals,
                                        np.full(GUARD, PRE_AMBLE_S)])
                x = make_packet(s_all, kaps_all, sigma2)
                off = PAY_START   # payload 符号 k 的样本起点 = (off+k)*NF

                # 我方族
                T = {c: np.zeros((NSYM_PKT, N)) for c in
                     ["OLD-A", "OLD-B", "NEW-0", "NEW-1", "NEW-2", "TREL-N", "ORACLE"]}
                ms = []
                for k in range(NSYM_PKT):
                    seg = x[(off + k) * NF:(off + k + 1) * NF]
                    T["OLD-A"][k] = np.abs(np.fft.fft(seg[::OS] * DOWN_CHIP)) ** 2
                    Tb = np.zeros(N)
                    for p in range(OS):
                        Tb += np.abs(np.fft.fft(seg[p:p + NF:OS] * DOWN_CHIP)) ** 2
                    T["OLD-B"][k] = Tb
                    T0, T1, T2, m = my_new_stats(x, off + k)
                    T["NEW-0"][k] = T0
                    T["NEW-1"][k] = T1
                    T["NEW-2"][k] = T2
                    ms.append(m)
                    y = seg * np.exp(-1j * 2 * np.pi * (U_GRID - kaps_pay[k]) ** 2 / (2 * N))
                    T["ORACLE"][k] = np.abs(np.fft.fft(y, n=M_PAD)[(-4 * np.arange(N)) % M_PAD]) ** 2
                # 值域转换 + trellis
                Tv = {}
                for c in ["OLD-A", "OLD-B"]:
                    Tv[c] = T[c][:, BIN2VAL]
                path = viterbi_kappa(ms)
                for k in range(NSYM_PKT):
                    T["TREL-N"][k] = ms[k][:, path[k]]
                for c in ["NEW-0", "NEW-1", "NEW-2", "TREL-N", "ORACLE"]:
                    Tv[c] = T[c]
                # 仓库基线（bin->值）。UNICHIRP 额外消费 preamble 训练相位模型（对其有利）。
                for k in range(NSYM_PKT):
                    st = (off + k) * NF
                    r1 = sav_demod(samples=x, start_sample=st, sf=10, os_factor=4)
                    r2 = trim_demod(samples=x, start_sample=st, sf=10, os_factor=4)
                    Tv.setdefault("SAVAUX", np.zeros((NSYM_PKT, N)))[k] = np.abs(r1.combined_spectrum) ** 2
                    Tv.setdefault("TRIMMER", np.zeros((NSYM_PKT, N)))[k] = r2.metric
                train = tuple(UniChirpTrainingSymbol(start_sample=k * NF, raw_fft_bin=0,
                                                     abs_symbol_index=float(k))
                              for k in range(PAY_START))
                model, _obs = build_unichirp_phase_model(samples=x, training_symbols=train,
                                                         sf=10, os_factor=4, config=UNI_CFG)
                for k in range(NSYM_PKT):
                    r3 = uni_demod(samples=x, start_sample=(off + k) * NF, sf=10,
                                   os_factor=4, phase_rad=model.predict(off + k),
                                   config=UNI_CFG)
                    Tv.setdefault("UNICHIRP", np.zeros((NSYM_PKT, N)))[k] = r3.metric
                Tv["SAVAUX"] = Tv["SAVAUX"][:, BASE2VAL]
                Tv["TRIMMER"] = Tv["TRIMMER"][:, BASE2VAL]
                Tv["UNICHIRP"] = Tv["UNICHIRP"][:, BASE2VAL]

                for c in chains:
                    dec = np.argmax(Tv[c], axis=1)
                    err[c] += int(np.sum(dec != s_vals))
                    ok_s, ok_h = fec_pdr(bit_llrs(Tv[c]), payload)
                    pdr_s[c] += int(ok_s == 40)
                    pdr_h[c] += int(ok_h == 40)
                n_sym += NSYM_PKT
            results.setdefault(km, {})[str(snr_db)] = {
                c: dict(ser=err[c] / n_sym, pdr_soft=pdr_s[c] / n_pkt, pdr_hard=pdr_h[c] / n_pkt)
                for c in chains}
            row = results[km][str(snr_db)]
            print("[{:5s}]{:>4}: ".format(km, snr_db) + " | ".join(
                "{}:{:.3f}/{:.2f}".format(c, row[c]['pdr_soft'], row[c]['ser']) for c in chains))
            print("      hard: " + " | ".join(
                "{}:{:.3f}".format(c, row[c]['pdr_hard']) for c in chains), flush=True)
    # 输出路径为项目内字面量（无拼接、不含上跳段），直接内联供静态检查
    with open(r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\ablation_battle_20260928\results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("\n%.0fs elapsed -> results.json" % (time.time() - t_start))

if __name__ == "__main__":
    main()

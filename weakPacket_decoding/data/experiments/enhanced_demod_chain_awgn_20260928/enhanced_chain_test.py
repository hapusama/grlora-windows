# -*- coding: utf-8 -*-
r"""
2026-09-28 验证脚本 II：用已发现的泄漏性质组装增强解码链，可控 AWGN 压测。

链（同一物理信号、同一噪声，只换解调层）：
  OLD-A  chip 网格单相位抽取 + N-FFT argmax        （frame_sync 风格旧链）
  OLD-B  4 相位抽取 + 功率谱平均 argmax             （朴素"过采样副本叠加"）
  NEW-1  OS 域去斜 + 4N 细网格 FFT + 5 细bin 窗口 max（κ-GLRT，粗）
  NEW-2  OS 域去斜 + 4N 细网格 FFT + 反相指纹相干重组（幅值比→κ̂→Dirichlet 匹配权重）
  ORACLE 真κ 相干 ML（上界参考）

关键性质的使用：
  P1 整数 chip 对齐 + OS 域处理 = 相位对齐合并（未滤波噪声下 ~ +6 dB 相对单相位）
  P2 反相指纹: m2/m1 = δ/(1−δ) → κ̂；Dirichlet 核相位 = 相干重组权重
  P3 细网格窗口 = 免估计 κ-GLRT（含 SFO 漂移）

FEC stand-in（非完整 LoRa FEC，仅用于链间公平对比）：
  40 payload bits → 10 x Hamming(8,4)+总体校验 (d_min=4) → 80 code bits
  → 对角交织到 8 个 SF10 符号 → Gray 映射成符号值。
  解调出 T(s) 全谱 → bit LLR（边缘化）→ Chase-16 MAP；硬版=最近码字。
  PDR = payload 40 bit 全对。

可控量：per-sample SNR（os 域 iid 噪声，口径同 oversampling_accounting），
kappa 模式 {0, 0.25, 0.5, drift(+0.02/symbol from 0.1)}。只依赖 numpy。
"""
import json
import numpy as np

N = 1024          # SF10
OS = 4
NF = N * OS       # 细网格点数
RNG = np.random.default_rng(20260928)

# 输出文件：项目内字面量路径，无任何路径拼接
OUT_JSON = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\enhanced_demod_chain_awgn_20260928\results.json"

RESULTS = {}

# ---------- 波形 ----------
U_GRID = np.arange(NF) / OS                       # 本地 chip 网格时刻 u_j = j/4

def sym_wave(u, s):
    return np.exp(1j * 2 * np.pi * (u ** 2 / (2 * N) - s * u / N))

REF_OS = np.exp(-1j * 2 * np.pi * U_GRID ** 2 / (2 * N))   # OS 去斜参考（chip 网格对齐）
DOWN_CHIP = np.exp(-1j * np.pi * np.arange(N) ** 2 / N)

def decimate_fft(x_os, m0):
    return np.fft.fft(x_os[m0:m0 + N * OS:OS] * DOWN_CHIP)

def dirich(x, L, M):
    """长 L 样本窗、补零到 M 点的 Dirichlet 核。x = 音位置 − bin（M 网格单位）。"""
    x = np.asarray(x, dtype=float)
    a = np.pi * x / M
    sa = np.sin(a)
    ok = np.abs(sa) > 1e-12
    val = np.where(ok, np.sin(a * L) / (L * np.where(ok, sa, 1.0)), 1.0)
    return np.exp(1j * np.pi * x * (L - 1) / M) * val

# ---------- Gray / FEC stand-in ----------
def gray_decode(v):
    v = v ^ (v >> 16); v = v ^ (v >> 8); v = v ^ (v >> 4); v = v ^ (v >> 2); v = v ^ (v >> 1)
    return v

SYM = np.arange(N)
VAL = np.array([gray_decode(s) for s in SYM])      # 符号值 -> 10bit 原值
BITMASK = np.stack([(VAL >> i) & 1 for i in range(10)], axis=0).astype(bool)  # (10, N)

# Hamming(8,4)+总校验 码表（d_min=4，16 码字）
CW = []
for d in range(16):
    b = np.array([(d >> i) & 1 for i in range(4)], dtype=np.int8)
    p1 = b[0] ^ b[1] ^ b[2]; p2 = b[1] ^ b[2] ^ b[3]; p3 = b[0] ^ b[2] ^ b[3]
    cw = np.concatenate([b, [p1, p2, p3]])
    cw = np.concatenate([cw, [cw.sum() & 1]])
    CW.append(cw)
CW = np.array(CW)                                   # (16, 8)

NSYM_PKT = 8
BIN2VAL = (-np.arange(N)) % N          # T 向量按 bin 索引排；BIN2VAL[b] = 该 bin 对应的符号值

def make_packet(s_vals, kaps, sigma2):
    """OS 域整包信号：每符号一段 NF 样本 + iid 复高斯噪声。"""
    xs = [sym_wave(U_GRID - kaps[k], s_vals[k]) for k in range(len(s_vals))]
    x = np.concatenate(xs)
    x = x + (RNG.standard_normal(x.size) + 1j * RNG.standard_normal(x.size)) * np.sqrt(sigma2 / 2)
    return x

# ---------- 各解调链：返回每符号 T(s) 全谱向量（未归一能量） ----------
def demod_old_a(x, k):
    y = x[k * NF:(k + 1) * NF:OS] * DOWN_CHIP
    return np.abs(np.fft.fft(y)) ** 2

def demod_old_b(x, k):
    T = np.zeros(N)
    m0 = k * NF
    for p in range(OS):
        T += np.abs(decimate_fft(x[m0:], p)) ** 2
    return T

def demod_new(x, k):
    """返回 (T1 窗口max, T2 指纹相干)。补零到 16N：细网格 = 主 bin 的 1/4。"""
    M = 16 * N
    y = x[k * NF:(k + 1) * NF] * REF_OS
    Z = np.fft.fft(y, n=M)
    mag2 = np.abs(Z) ** 2
    base = (-4 * np.arange(N)) % M
    win = (base[:, None] + np.arange(-2, 3)[None, :]) % M      # (N,5)，覆盖 κ∈(−0.5,0.5]
    m = mag2[win]
    row = np.arange(N)
    d_star = np.argmax(m, axis=1)
    m1 = m[row, d_star]
    left = np.where(d_star > 0, m[row, np.maximum(d_star - 1, 0)], -1.0)
    right = np.where(d_star < 4, m[row, np.minimum(d_star + 1, 4)], -1.0)
    use_right = right >= left
    m2 = np.where(use_right, right, left)
    j1 = win[row, d_star]
    j2 = np.where(use_right, win[row, np.minimum(d_star + 1, 4)],
                  win[row, np.maximum(d_star - 1, 0)])
    delta = np.clip(m2 / (m1 + m2 + 1e-12), 0.0, 0.9)
    # 音位置 p̂ = j1 + δ（use_right）或 j1 − δ；x = p̂ − j
    x1 = np.where(use_right, delta, -delta)
    x2 = x1 - np.where(use_right, 1.0, -1.0)
    w1 = np.conj(dirich(x1, NF, M)); w2 = np.conj(dirich(x2, NF, M))
    wnorm = np.abs(w1) ** 2 + np.abs(w2) ** 2 + 1e-30
    T2 = np.abs(w1 * Z[j1] + w2 * Z[j2]) ** 2 / wnorm
    return m1, T2

def demod_oracle(x, k, kap):
    y = x[k * NF:(k + 1) * NF] * np.exp(-1j * 2 * np.pi * (U_GRID - kap) ** 2 / (2 * N))
    Z = np.fft.fft(y, n=16 * N)
    return np.abs(Z[(-4 * np.arange(N)) % (16 * N)]) ** 2

# ---------- 软/硬 FEC ----------
def bit_llrs(T):
    """T: (NSYM, N) -> llrs: (NSYM, 10)。"""
    eps = 1e-9 * T.max(axis=1, keepdims=True) + 1e-30
    num = np.tensordot(BITMASK.astype(float), T + eps, axes=(1, 1))   # (10, NSYM)
    den = np.tensordot((~BITMASK).astype(float), T + eps, axes=(1, 1))
    return (np.log(num) - np.log(den)).T

def fec_pdr(llrs, payload):
    """交织：codebit(c,j) -> 符号 j 的位 c。返回 (soft/hard 正确 payload bit 数)。"""
    total_s = total_h = 0
    for c in range(10):
        l = llrs[:, c]
        hard = (l > 0).astype(np.int8)
        score = (2 * CW.astype(float) - 1) @ l
        d_soft = CW[int(np.argmax(score))]
        dist = (CW != hard).sum(axis=1)
        d_hard = CW[int(np.argmin(dist))]
        total_s += int(np.sum(d_soft[:4] == payload[c]))
        total_h += int(np.sum(d_hard[:4] == payload[c]))
    return total_s, total_h

def gen_payload():
    return RNG.integers(0, 2, (10, 4))

def bits_to_symbols(payload):
    nib = payload @ np.array([1, 2, 4, 8])   # 每行 4 bit -> nibble
    code = CW[nib]                            # (10, 8)
    flat = code.reshape(-1)                   # 80 bit：flat[c*8+j] = codeword c 的 bit j
    sym_bits = np.zeros((NSYM_PKT, 10), dtype=np.int64)
    for c in range(10):
        for j in range(NSYM_PKT):
            sym_bits[j, c] = flat[c * 8 + j]
    vals = (sym_bits * (1 << np.arange(10))).sum(axis=1)
    return vals ^ (vals >> 1)                 # gray encode

# ---------- 实验 ----------
def main():
    n_pkt = 120
    snrs = [-26, -24, -22, -20, -18, -16]
    kap_modes = {"k0": 0.0, "k25": 0.25, "k50": 0.5, "drift": None}
    chains = ["OLD-A", "OLD-B", "NEW-1", "NEW-2", "ORACLE"]

    for km, kap0 in kap_modes.items():
        for snr_db in snrs:
            sigma2 = 10 ** (-snr_db / 10.0)
            err = {c: 0 for c in chains}
            pdr_soft = {c: 0 for c in chains}
            pdr_hard = {c: 0 for c in chains}
            n_sym = 0
            for _ in range(n_pkt):
                payload = gen_payload()
                s_vals = bits_to_symbols(payload)
                kaps = (np.full(NSYM_PKT, kap0) if kap0 is not None
                        else 0.1 + 0.02 * np.arange(NSYM_PKT))
                x = make_packet(s_vals, kaps, sigma2)
                Ts = {c: np.zeros((NSYM_PKT, N)) for c in chains}
                for k in range(NSYM_PKT):
                    Ts["OLD-A"][k] = demod_old_a(x, k)
                    Ts["OLD-B"][k] = demod_old_b(x, k)
                    t1, t2 = demod_new(x, k)
                    Ts["NEW-1"][k] = t1
                    Ts["NEW-2"][k] = t2
                    Ts["ORACLE"][k] = demod_oracle(x, k, kaps[k])
                for c in chains:
                    # OLD 链的 T 按 bin 索引（bin ≈ −符号值），需翻转到符号值域；
                    # NEW/ORACLE 的 T 本来就按假设的符号值索引，直接用。
                    Tv = Ts[c][:, BIN2VAL] if c in ("OLD-A", "OLD-B") else Ts[c]
                    dec = np.argmax(Tv, axis=1)
                    err[c] += int(np.sum(dec != s_vals))
                    ok_s, ok_h = fec_pdr(bit_llrs(Tv), payload)
                    pdr_soft[c] += int(ok_s == 40)
                    pdr_hard[c] += int(ok_h == 40)
                n_sym += NSYM_PKT
            RESULTS.setdefault(km, {})[str(snr_db)] = {
                c: dict(ser=err[c] / n_sym,
                        pdr_soft=pdr_soft[c] / n_pkt,
                        pdr_hard=pdr_hard[c] / n_pkt)
                for c in chains}
            row = RESULTS[km][str(snr_db)]
            print("[{:6s}] SNR {:>4}: ".format(km, snr_db) + " | ".join(
                "{} S {:.4f} sP {:.3f} hP {:.3f}".format(
                    c, row[c]['ser'], row[c]['pdr_soft'], row[c]['pdr_hard'])
                for c in chains))
    with open(OUT_JSON, "w") as f:
        json.dump(RESULTS, f, indent=1, default=float)
    print("\nsaved -> results.json")

if __name__ == "__main__":
    main()

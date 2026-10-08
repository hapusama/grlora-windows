# -*- coding: utf-8 -*-
"""红队攻击 (e)(f)(g)：LDRO 校验位消歧 —— 规格 + 三臂消融 + 口径。

(e) 规格已核对（interleaver_impl.cc）：LDRO/首块行 = sf_app 数据位(MSB 侧)
+ 1 校验位在 2^1 + LSB 恒 0。A2 说"校验位=LSB"是错的；但修正后结论更强：
相邻 bin 候选经真实 RX 映射（单 fold，Gray 性质）恰差一个 w-bit 于位置
t=b-1 尾随 1 数 → t=0 踩恒零位、t=1 踩校验位、t>=2 踩单个数据位
→ 行校验全部确定性击杀，100%（非 A2 说的 50%）。

(f) 三臂消融（SF9 LDRO CR4/8, 1 块 = 8 符号 x 7 码字, genie κ̂, 纯 AWGN）：
  A0 现状: 4 精细 bin 非相干边缘化（divisor-4 demod，丢校验位）
  A1 硬剪枝: 只用唯一合法精细 bin（恒零+校验全用）
  A2 软惩罚: 非法 bin 权重 0.5
  报 CER10% 门限差。

(g) 口径：见打印。

运行：py -3.12 rt_f_ldro_check.py
"""
import numpy as np
import time

rng = np.random.default_rng(20261003)
SF = 9
N = 1 << SF
NS = 8            # cw_len=8 symbols per block
SFA = SF - 2      # 7 codewords
T = np.arange(N)

Gm = np.array([[1, 0, 0, 0, 1, 1, 1, 0],
               [0, 1, 0, 0, 1, 1, 0, 1],
               [0, 0, 1, 0, 1, 0, 1, 1],
               [0, 0, 0, 1, 0, 1, 1, 1]])
info = np.arange(16)
cb = (((info[:, None] >> np.arange(4)[::-1]) & 1) @ Gm) % 2     # [16,8]
cbint = cb @ (1 << np.arange(7, -1, -1))

def cumfold9(v):
    g = v.copy()
    for sh in range(1, SF):
        g ^= v >> sh
    return g

# 候选表: d(7bit 行数据) -> 每符号 w -> bin；及同 coarse 组的 4 个精细 bin
D = np.arange(1 << SFA)
dbits = ((D[:, None] >> np.arange(SFA - 1, -1, -1)) & 1)        # [128,7] MSB-first
def w_of(dbits, i):
    """symbol i 的 w：row[j]=d_j, row[7]=parity, row[8]=0"""
    par = dbits.sum(1) % 2
    row = np.concatenate([dbits, par[:, None], np.zeros((len(D), 1), int)], axis=1)
    return row @ (1 << np.arange(SF - 1, -1, -1))
W = w_of(dbits, 0)                                              # 行结构不随 i 变
BINvalid = (cumfold9(W) + 1) % N                                # [128]
# 每个 d 的 coarse 组 4 精细 bin（合法 + 3 非法）
COARSE = (W >> 2)
FINE = (COARSE[:, None] << 2) + np.arange(4)[None, :]           # [128,4]
FINEBIN = (cumfold9(FINE) + 1) % N                              # [128,4]

def gen(kappa, gamma_db, n):
    """真码块信号。返回 y[n,8,N], 真码字索引 [n,7]"""
    ci = rng.integers(0, 16, size=(n, SFA))
    cw = cb[ci]                                                 # [n,7,8]
    bins = np.zeros((n, NS), dtype=np.int64)
    for i in range(NS):
        row = np.zeros((n, SF), dtype=np.int8)
        for j in range(SFA):
            row[:, j] = cw[:, (i - j - 1) % SFA, i]
        row[:, SFA] = row[:, :SFA].sum(1) % 2
        w = row @ (1 << np.arange(SF - 1, -1, -1))
        bins[:, i] = (cumfold9(w) + 1) % N
    ph = rng.uniform(0, 2 * np.pi, n)
    sig = np.exp(2j * np.pi * (bins[..., None] + kappa) * T / N + ph[:, None, None])
    sg = 10 ** (-gamma_db / 10.0)
    y = sig + np.sqrt(sg / 2) * (rng.standard_normal(sig.shape)
                                 + 1j * rng.standard_normal(sig.shape))
    return y, ci

def cer(y, ci, gamma_db, arm, kappa, genie=True):
    sigma = 10 ** (-gamma_db / 20.0)
    lam0 = N * 10 ** (gamma_db / 10.0)
    z = np.exp(-2j * np.pi * (kappa if genie else 0.0) * T / N)
    Z = np.fft.fft(y * z[None, None, :], axis=-1) / (sigma * np.sqrt(N))  # [n,8,N]
    t = 2.0 * np.sqrt(lam0)
    def li0(x):
        x = np.abs(x)
        return np.where(x > 30, x - 0.5 * np.log(2 * np.pi * np.maximum(x, 30)),
                        np.log(np.i0(np.minimum(x, 30)) + 1e-300))
    n = len(y)
    if arm == 'A1':
        zv = Z[:, :, BINvalid]
        L = li0(t * np.abs(zv)) - lam0                          # [n,8,128]
    elif arm == 'A0':
        zv = Z[:, :, FINEBIN]
        s = li0(t * np.abs(zv)) - lam0
        mx = s.max(-1, keepdims=True)
        L = (mx[..., 0] + np.log(0.25 * np.exp(s - mx).sum(-1)))  # [n,8,128]
    else:  # A2 软惩罚: 合法 x1, 非法 x0.5
        zv = Z[:, :, FINEBIN]
        s = li0(t * np.abs(zv)) - lam0
        wgt = np.array([0.7, 0.1, 0.1, 0.1])
        mx = s.max(-1, keepdims=True)
        L = mx[..., 0] + np.log((wgt * np.exp(s - mx)).sum(-1))
    # 符号后验 -> 行 bit 后验 -> 码字判决
    pz = np.exp(L - L.max(-1, keepdims=True))
    pz /= pz.sum(-1, keepdims=True)                             # [n,8,128]
    pb = pz @ dbits                                              # P(row_j=1) [n,8,7]
    err = 0
    for m in range(SFA):
        ll = np.zeros((n, 16))
        for i in range(NS):
            j = (i - 1 - m) % SFA
            p1 = np.clip(pb[:, i, j], 1e-12, 1 - 1e-12)
            ll += np.log(np.where(cb[None, :, i] == 1, p1[:, None], 1 - p1[:, None]))
        err += int(np.sum(np.argmax(ll, axis=1) != ci[:, m]))
    return err / (n * SFA)


def eval_cer(arm, kappa, gamma_db, n, genie=True):
    y, ci = gen(kappa, gamma_db, n)
    return cer(y, ci, gamma_db, arm, kappa, genie=genie)


def threshold(arm, kappa, lo=-36.0, hi=-22.0, genie=True):
    for _ in range(5):
        mid = 0.5 * (lo + hi)
        if eval_cer(arm, kappa, mid, 200, genie) > 0.10:
            lo = mid
        else:
            hi = mid
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        if eval_cer(arm, kappa, mid, 400, genie) > 0.10:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

t0 = time.time()
for genie, lab in ((False, "无 khat（真实跨踞歧义）"), (True, "genie khat（对照）")):
    print("== (f) SF9 LDRO CR4/8 单块三臂 (%s) ==" % lab)
    for kappa in (0.45, 0.35):
        row = {}
        for arm in ('A0', 'A2', 'A1'):
            row[arm] = threshold(arm, kappa, genie=genie)
            print(" kappa=%+.2f  %s CER10%% = %+.2f dB" % (kappa, arm, row[arm]), flush=True)
        print("   Δ(硬剪枝-现状) = %+.2f dB ; Δ(软惩罚-现状) = %+.2f dB"
              % (row['A1'] - row['A0'], row['A2'] - row['A0']), flush=True)
print("elapsed %.1f s" % (time.time() - t0))

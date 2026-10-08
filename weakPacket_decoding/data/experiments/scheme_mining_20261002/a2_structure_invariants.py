# -*- coding: utf-8 -*-
"""A2 视角数值核查（2026-10-02）：帧级结构恒等式快验（v2，修正映射方向）。

项目真实约定（payload_codec.py 复刻 gr-lora_sdr）：
  TX: interleaved 值 w -> 发射 bin s = fold(w) + 1   (fold = XOR-fold, gray_demap_symbols)
  RX: bin b -> w = unfold(b - 1)
故相邻 bin (b, b+1) 的混淆 -> 值对 (unfold(b-1), unfold(b))，
bit 重量 = |unfold((b-1)^b)|，unfold 线性 -> 只依赖 b 的尾随 1 游程。

核查项：
1. 相邻 bin ±1 错的值域 bit 重量分布（种子2前提定量版）。
2. LDRO/首块符号 LSB=行校验位时，相邻 bin 混淆被校验位确定性消歧的比例
   （新挖恒等式候选）。
3. 种子1指纹：单符号值错 -> 对角交织命中码字剖面（闭式 mod(i-j-1, sf_app) 验证）。
"""
import numpy as np

rng = np.random.default_rng(20261002)
sf = 10
N = 1 << sf
mask = N - 1


def fold(v, sf):
    g = v
    for shift in range(1, sf):
        g ^= v >> shift
    return g


def unfold(g, sf):
    w = g
    shift = 1
    while (1 << shift) < (1 << sf):
        w ^= w >> shift
        shift += 1
    return w


def int2bool_msbf(v, n):
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def bool2int_msbf(bits):
    x = 0
    for b in bits:
        x = (x << 1) | int(b)
    return x


def interleave_block(codewords, sf_app, parity=False):
    """对角交织 inter_bin[i][j] = cw_bin[mod(i-j-1, sf_app)][i]，MSB-first。"""
    cw_len = len(codewords)
    cw_bin = [int2bool_msbf(c, cw_len) for c in codewords]
    out = []
    for i in range(cw_len):
        row = np.zeros(sf_app + 1, dtype=np.uint8)
        for j in range(sf_app):
            row[j] = cw_bin[(i - j - 1) % sf_app][i]
        if parity:
            row[sf_app] = row[:sf_app].sum() % 2  # 行校验位 = 行 LSB
        out.append(row)
    return out


# ---------- 1. 相邻 bin ±1 错 -> interleaved 值域 bit 重量 ----------
b = np.arange(N)
w0 = np.array([unfold((int(x) - 1) & mask, sf) for x in b])  # 真值 bin b 的值
w1 = np.array([unfold(int(x) & mask, sf) for x in b])        # 错判 bin b+1 的值
d = np.array([int(a) ^ int(bb) for a, bb in zip(w0, w1)])
wt = np.array([bin(x).count("1") for x in d])
print("[1] bin±1 错 -> 值域 bit 重量: mean=%.3f  P(1)=%.4f P(2)=%.4f P(3)=%.4f"
      % (wt.mean(), (wt == 1).mean(), (wt == 2).mean(), (wt == 3).mean()))

# ---------- 2. 校验位消歧比例（LDRO/首块符号） ----------
# 行 LSB = 其余 sf_app bit 异或 => 两候选恰在 LSB 差 1 bit 时，恰一候选合法。
resolvable = (wt == 1) & ((d & 1) == 1)
print("[2] LDRO/首块符号校验位可确定性消歧相邻 bin 混淆的比例 = %.4f"
      % resolvable.mean())
# 单 bit 翻（任意位）的比例 = 交织-汉明可纠的单符号错上限
print("    单 bit 翻转(任意位, 汉明可纠)比例 = %.4f" % (wt == 1).mean())

# ---------- 3. 种子1指纹：单符号错 -> 码字命中剖面 ----------
sf_app, cw_len = 8, 8   # 首块/header: SF10 -> sf_app=8, cw_len=8
codewords = list(rng.integers(0, 1 << cw_len, sf_app))
blk = interleave_block(codewords, sf_app, parity=True)
sym = [bool2int_msbf(r) for r in blk]

for err, label in ((1, "值错+1"), (7, "值错低位风暴")):
    i0 = 3
    e = (sym[i0] + err) % (1 << (sf_app + 1))
    diff = int2bool_msbf(e, sf_app + 1) ^ blk[i0]
    hits = {}
    for j in range(sf_app + 1):
        if diff[j]:
            hits.setdefault((i0 - j - 1) % sf_app, []).append(j)
    print("    符号%d %s -> 命中(码字:[翻转位]) %s，每码字 bit 数 %s"
          % (i0, label, hits, {k: len(v) for k, v in hits.items()}))
print("    [验证] 命中码字集合 = {mod(i0-j-1, sf_app) : 行内翻转位 j}，闭式成立")

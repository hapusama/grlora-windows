# -*- coding: utf-8 -*-
"""红队攻击 (i)：半 bin 改写的码学验证 —— 真实链（nibble Hamming + 对角交织
+ CR4/5..4/8 + LDRO 行校验/恒零位）下，跨踞对 (bin b, b±1) 的两个候选
是否可能同时是有效码。逐 CR 档报 p = P(两候选皆过码)。

真实约定（interleaver_impl.cc / payload_codec.py 已核对）：
  payload 块: sf_app=sf 个码字, cw_len=4+cr, 无校验行
  LDRO/首块: sf_app=sf-2, row[sf_app]=行校验位(2^1), row[sf-1]恒0(LSB)
  TX bin s = (fold(w)+1) mod N ; RX w = unfold((b-1) mod N)

运行：py -3.12 rt_i_halbin_code.py
"""
import numpy as np

rng = np.random.default_rng(20261002)


def encode_hamming_vec(nibbles, cr_app):
    """向量化 encode_hamming_nibble。nibbles: [B] -> codewords [B] (cw_len bits)"""
    d = ((nibbles[:, None] >> np.arange(4)[::-1]) & 1)  # [B,4] d3..d0
    if cr_app != 1:
        p0 = d[:, 3] ^ d[:, 2] ^ d[:, 1]
        p1 = d[:, 2] ^ d[:, 1] ^ d[:, 0]
        p2 = d[:, 3] ^ d[:, 2] ^ d[:, 0]
        p3 = d[:, 3] ^ d[:, 1] ^ d[:, 0]
        full = (d @ (1 << np.arange(7, 3, -1))) | (p0 << 3) | (p1 << 2) | (p2 << 1) | p3
        return full >> (4 - cr_app), 4 + cr_app
    p4 = d[:, 0] ^ d[:, 1] ^ d[:, 2] ^ d[:, 3]
    return (d @ (1 << np.arange(4, 0, -1))) | p4, 5


def bits_of(x, n):
    return ((x[:, None] >> np.arange(n - 1, -1, -1)) & 1)  # MSB-first [B,n]


def fold(v, sf):
    g = v.copy()
    for shift in range(1, sf):
        g ^= v >> shift
    return g


def unfold(g, sf):
    """真实 RX gray 逆映射：单步 fold（gray_mapping_impl.cc: x ^ (x>>1)）。
    注意：a2_structure_invariants 的累积 unfold 不是 RX 链用的映射。"""
    return g ^ (g >> 1)


def block_symbols(cw, cw_len, sf_app, sf, parity_row):
    """cw: [B, sf_app] codeword ints -> symbols [B, cw_len] (含可选校验行)"""
    B = cw.shape[0]
    cb = np.zeros((B, sf_app, 8), dtype=np.int8)   # 最多 8 bit 码字
    cb[:, :, :cw_len] = bits_of(cw.ravel(), cw_len).reshape(B, sf_app, cw_len)
    nrow = sf if parity_row else sf  # 行寄存器总是 sf 位
    out = np.zeros((B, cw_len), dtype=np.int64)
    for i in range(cw_len):
        row = np.zeros((B, sf), dtype=np.int8)
        for j in range(sf_app):
            row[:, j] = cb[:, (i - j - 1) % sf_app, i]
        if parity_row:
            row[:, sf_app] = row[:, :sf_app].sum(1) % 2   # row[sf-1] 恒 0
        out[:, i] = row @ (1 << np.arange(sf - 1, -1, -1))
    return out


def check_valid(syms, cw_len, sf_app, sf, codebook, parity_row):
    """从符号值反抽码字并检查有效性（含 LDRO 行校验/恒零）。syms: [B,cw_len]"""
    B = syms.shape[0]
    sb = bits_of(syms.ravel(), sf).reshape(B, cw_len, sf)
    ok = np.ones(B, dtype=bool)
    if parity_row:
        ok &= np.all(sb[:, :, sf - 1] == 0, axis=1)
        ok &= np.all(sb[:, :, sf_app] == sb[:, :, :sf_app].sum(2) % 2, axis=1)
    for m in range(sf_app):
        cwv = np.zeros(B, dtype=np.int64)
        for i in range(cw_len):
            j = (i - 1 - m) % sf_app
            cwv |= sb[:, i, j].astype(np.int64) << (cw_len - 1 - i)
        ok &= np.isin(cwv, codebook)
    return ok


def run_case(sf, cr, ldro, n_blocks=100000, label=""):
    n = 1 << sf
    sf_app = sf - 2 if ldro else sf
    cw_len = 4 + cr
    parity_row = ldro
    nib = rng.integers(0, 16, size=(n_blocks, sf_app))
    cws, _ = encode_hamming_vec(nib.ravel(), cr if not ldro else cr)
    # 注意：LDRO 时所有块都是 sf_app=sf-2 结构，码字长度仍按 cr（首块 cr=4 由首块规则决定，此处按 cr 通行）
    cw = cws.reshape(n_blocks, sf_app)
    codebook = np.array(sorted(set(encode_hamming_vec(np.arange(16), cr)[0].tolist())))
    syms = block_symbols(cw, cw_len, sf_app, sf, parity_row)
    bins = (fold(syms, sf) + 1) % n
    tot = 0
    both = 0
    wrap = 0
    for direction in (+1, -1):
        b2 = (bins + direction) % n
        w2 = unfold((b2 - 1) % n, sf)
        for i in range(cw_len):
            alt = syms.copy()
            alt[:, i] = w2[:, i]
            ok = check_valid(alt, cw_len, sf_app, sf, codebook, parity_row)
            both += int(ok.sum())
            tot += n_blocks
            wrap += int(np.sum(((bins[:, i] == 0) & (direction == -1)) |
                               ((bins[:, i] == n - 1) & (direction == 1))))
    print(" %-22s cw_len=%d sf_app=%d parity_row=%d : p = %d/%d = %.6f (wrap-bin 事件 %d)"
          % (label, cw_len, sf_app, parity_row, both, tot, both / tot, wrap))


if __name__ == "__main__":
    print("== (i) 真实链跨踞对 (b, b±1) 双有效概率 p，逐 CR/LDRO ==")
    for cr, name in ((1, "CR4/5"), (2, "CR4/6"), (3, "CR4/7"), (4, "CR4/8")):
        run_case(8, cr, False, label="SF8 payload %s" % name)
    run_case(8, 4, True, label="SF8 首块(LDRO 结构)")
    run_case(11, 4, True, label="SF11 LDRO CR4/8")
    run_case(12, 4, True, label="SF12 LDRO CR4/8")
    run_case(10, 1, True, label="SF10 LDRO CR4/5")

# 理论注解：w 的翻转 bit j 作用在码字 m=(i-j-1)%sf_app 的 bit i 上；
# d=2^{tz+1}-1 的 bit 集 {0..tz} -> 不同 j -> 不同码字 -> 每码字恰好 1 bit 翻
# -> 任何 dmin>=2 的码（所有 LoRa CR）都检测 -> p 应为 0（非 wrap）。

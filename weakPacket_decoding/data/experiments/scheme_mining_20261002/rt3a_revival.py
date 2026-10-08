# -*- coding: utf-8 -*-
"""Round 3 复活轮 agent A：两段式获取（前导放松门限 + 解码层 CRC 仲裁）复活条件量化。

红队死刑依据（rt_b_sysroc）：
  1) 记账混单位：W=1e3 窗、FAR_sys=1e-6 下 per-cell 只能 6.55e-5（不是 1e-2），
     一致记账预算增益仅 +2.66dB。
  2) 解码门钳死：SF8/CR4/5/N=30 全帧 CRC 通过 0.9 的门 = -11.11dB（per-sample），
     比 stage-1 frontier（-20.8/-23.1）浅 9-12dB -> 系统被拉回，净 -7dB。
死因疑点：rt_b 的"解码门"是弱解码（逐符号硬判 + (1-SER)^30 模型）。本脚本用
真实 codec（对角交织 + CR 汉明 + grlora CRC16，与 payload_codec 对齐并 roundtrip
验证）实测三档门的深度与 H0 误过率，给出系统级净 dB 裁决。

门设计（同噪声实现喂所有门）：
  A   弱门（锚点）：genie kappa=0 FFT argmax -> 硬 deinterleave -> gr-lora 硬汉明
      （cr=1 不纠错、cr=4 奇偶+syndrome 纠 1 个数据位，复刻 hamming_dec_impl）
      -> CRC16。应复现 rt_b 的 -11.11dB。
  B   强门（margF 级）：前 8 符号帧池化 kappa-hat/phi-hat + 16 点 kappa 网格
      对数域边缘化软符号度量（相干；genie lambda0 标度，诚实标注）-> 精确码字
      MAP（16 合法码字）-> CRC16（唯一判决，L=1）。
  B+  Chase 式列表：B 的 MAP 之上翻转低置信码字（单翻 12 + 双翻 C(6,2)=15，
      名义 L=28）-> 任一过 CRC 即通过。

配置：SF8 M=256，CR4/5 N=30（6 块x5 符号，22B payload+2B CRC）；
CR4/8 N=32（4 块x8 符号满块平铺，14B+2B——N=30 不满块，取最近平铺并声明）。

运行：py -3.12 rt3a_revival.py          （全量，约 20 分钟）
      py -3.12 rt3a_revival.py smoke    （codec 验证 + 小样本冒烟）
"""
import sys
import time
import json
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")

M, NP, KGRID = 256, 8, 16
W, TOPK = 1000, 20
kap = np.arange(KGRID) / KGRID
mm = np.arange(M)
TW = np.exp(-2j * np.pi * np.outer(kap, mm) / M)      # (16, 256)

# ---- 灰度/fold 查找表：TX bin=(cumfold(v)+1)&255 ; RX v=fold1((bin-1)&255) ----
CUMFOLD = np.zeros(M, dtype=np.int64)
for v in range(M):
    g = v
    for sh in range(1, 8):
        g ^= v >> sh
    CUMFOLD[v] = g
TXBIN = (CUMFOLD + 1) & 255
FOLD1 = np.array([x ^ (x >> 1) for x in range(M)], dtype=np.int64)   # RX 单步 fold
PERM = np.stack([(TXBIN - 16 * g) % M for g in range(KGRID)])       # (16, 256)
BPRE = np.array([(-16 * g) % M for g in range(KGRID)])              # 前导峰 bin

# ---- grlora CRC16 线性化（init=0 => GF(2) 线性）----
from weak_decoder.decoding.payload_codec import crc16 as ref_crc16  # noqa: E402

_CRC_BASIS = {}


def crc_basis(nbits):
    """长度 nbits 消息的单比特 CRC 基（GF(2) 线性，init=0）。"""
    if nbits not in _CRC_BASIS:
        nb = (nbits + 7) // 8
        b = np.zeros(nbits, dtype=np.int64)
        for i in range(nbits):
            msg = bytearray(nb)
            msg[i // 8] |= 1 << (7 - i % 8)
            b[i] = ref_crc16(msg)
        _CRC_BASIS[nbits] = b
    return _CRC_BASIS[nbits]


def crc16_bits(bits):                                 # bits: (n, nbits) 0/1
    crs = crc_basis(bits.shape[1])
    return np.bitwise_xor.reduce(np.where(bits > 0, crs[None, :], 0), axis=1)

# ---- 白化序列（前 24 byte 足够）----
WHIT = np.array([
    0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE1, 0xC2, 0x85, 0x0B, 0x17, 0x2F, 0x5E,
    0xBC, 0x78, 0xF1, 0xE3, 0xC6, 0x8D, 0x1A, 0x34, 0x68, 0xD0, 0xA0, 0x40],
    dtype=np.int64)

CFGS = {
    "cr45": dict(cr=1, BLK=5, NB=6, N=30, plen=22),
    "cr48": dict(cr=4, BLK=8, NB=4, N=32, plen=14),
}


def cw_bits_of_nibbles(nib):
    """nib: (...,4)=[d3,d2,d1,d0] -> 码字比特 MSB->LSB=[d0,d1,d2,d3,校验]。
    注意 payload_codec encode_hamming_nibble 把数据位反序放进码字（decode 端
    hamming_decode_hard 再反序取回），此处精确复刻。"""
    d3, d2, d1, d0 = nib[..., 0], nib[..., 1], nib[..., 2], nib[..., 3]
    bits45 = np.stack([d0, d1, d2, d3, d3 ^ d2 ^ d1 ^ d0], -1)
    bits48 = np.stack([d0, d1, d2, d3,
                       d0 ^ d1 ^ d2, d1 ^ d2 ^ d3,
                       d0 ^ d1 ^ d3, d0 ^ d2 ^ d3], -1)
    return bits45, bits48


VALID_CW = {}
for cr in (1, 4):
    nib = np.arange(16)[:, None] >> np.array([3, 2, 1, 0]) & 1
    b45, b48 = cw_bits_of_nibbles(nib)
    VALID_CW[cr] = (b45 if cr == 1 else b48)          # (16, cw_len)


def gen_payload(rng, n, cfg):
    """随机 payload -> 白化 nibble 流 + grlora CRC nibble（与 payload_codec 对齐）。"""
    plen = cfg["plen"]
    payload = rng.integers(0, 256, (n, plen)).astype(np.int64)
    offs = np.arange(plen)
    low = (payload & 0xF) ^ (WHIT[offs] & 0xF)
    high = (payload >> 4) ^ ((WHIT[offs] >> 4) & 0xF)
    wnib = np.empty((n, 2 * plen), dtype=np.int64)
    wnib[:, 0::2], wnib[:, 1::2] = low, high
    crc_v = np.array([
        ref_crc16(bytes(int(x) for x in row[:plen - 2]))
        ^ int(row[plen - 1]) ^ (int(row[plen - 2]) << 8)
        for row in payload], dtype=np.int64)
    cnib = np.stack([crc_v & 0xF, (crc_v >> 4) & 0xF,
                     (crc_v >> 8) & 0xF, (crc_v >> 12) & 0xF], -1)
    return np.concatenate([wnib, cnib], 1), payload


def nib_stream_to_bins(nib, cfg):
    """(n, NB*8) nibbles -> (n, N) TX bins（对角交织 + cumfold+1）。"""
    cr, BLK, NB = cfg["cr"], cfg["BLK"], cfg["NB"]
    n = nib.shape[0]
    nibbits = (nib.reshape(n, NB, 8)[..., None] >> np.array([3, 2, 1, 0])) & 1
    bits45, bits48 = cw_bits_of_nibbles(nibbits)
    xb = (bits45 if cr == 1 else bits48)                      # (n, NB, 8, 4+cr)
    out = np.zeros((n, NB * BLK), dtype=np.int64)
    for i in range(BLK):
        for m in range(8):
            out[:, i::BLK] |= xb[:, :, m, i].astype(np.int64) << (7 - ((i - 1 - m) % 8))
    return TXBIN[out]


def rx_vals_to_cwbits(vals, cfg):
    """(n, N) RX 值 v（8bit）-> (n, NB*8, cw_len) 码字比特。"""
    cr, BLK, NB = cfg["cr"], cfg["BLK"], cfg["NB"]
    n = vals.shape[0]
    vb = (vals[..., None] >> np.arange(7, -1, -1)) & 1      # (n, N, 8) MSB first
    cwb = np.zeros((n, NB * 8, 4 + cr), dtype=np.int64)
    for b in range(NB):
        for i in range(BLK):
            for m in range(8):
                cwb[:, b * 8 + m, i] = vb[:, b * BLK + i, (i - 1 - m) % 8]
    return cwb


def hard_hamming_nibbles(cwb, cfg):
    """gr-lora hamming_dec_impl 硬路径：cr=1 不纠错；cr=4 奇偶偶=>直取，
    奇=>syndrome 纠 1 个数据位（syn 5/7/3/6 -> 码字位 3/2/1/0），其余不纠。
    码字位 MSB->LSB = [d0,d1,d2,d3,校验...]；nibble = c3c2c1c0（MSB->LSB）。"""
    if cfg["cr"] == 1:
        c = cwb[..., :5]
        return (c[..., 3] << 3) | (c[..., 2] << 2) | (c[..., 1] << 1) | c[..., 0]
    par = cwb.sum(-1) & 1
    c = cwb.copy()
    c0, c1, c2, c3, c4, c5, c6 = [cwb[..., k] for k in range(7)]
    syn = ((c0 ^ c1 ^ c2 ^ c4) + 2 * (c1 ^ c2 ^ c3 ^ c5)
           + 4 * (c0 ^ c1 ^ c3 ^ c6))
    p = par.astype(bool)
    c[..., 3] ^= (syn == 5) & p
    c[..., 2] ^= (syn == 7) & p
    c[..., 1] ^= (syn == 3) & p
    c[..., 0] ^= (syn == 6) & p
    return (c[..., 3] << 3) | (c[..., 2] << 2) | (c[..., 1] << 1) | c[..., 0]


def crc_check(nib_dec, cfg):
    """(n, NB*8) 解出 nibble 流 -> (CRC bool, dewhitened payload)。"""
    plen = cfg["plen"]
    by = (nib_dec[:, 0::2] & 0xF) | ((nib_dec[:, 1::2] & 0xF) << 4)
    by[:, :plen] ^= WHIT[None, :plen]                  # dewhiten payload 部分
    payload, crc_part = by[:, :plen], by[:, plen:plen + 2]
    bits = ((payload[:, None, :] >> np.arange(7, -1, -1)[None, :, None]) & 1)
    bits = bits.transpose(0, 2, 1).reshape(len(by), 8 * plen)   # byte-major
    crc_v = crc16_bits(bits[:, :8 * (plen - 2)]).astype(np.int64)
    crc_v ^= payload[:, plen - 1]
    crc_v ^= payload[:, plen - 2] << 8
    crc_rx = crc_part[:, 0] | (crc_part[:, 1] << 8)
    return crc_v == crc_rx, payload


# ---------------------------------------------------------------------------
# 帧生成与三门
# ---------------------------------------------------------------------------

def gen_frames(rng, n, gamma_db, cfg):
    """-> (y, payload, t1)：y (n, 8+N, 256)；t1 = rt_a 口径前导统计量。"""
    sigma = 10.0 ** (-gamma_db / 20.0)
    nib, payload = gen_payload(rng, n, cfg)
    bins = nib_stream_to_bins(nib, cfg)
    nu = np.concatenate([np.zeros((n, NP), dtype=np.int64), bins], 1)
    w = sigma * ((rng.standard_normal((n, NP + cfg["N"], M))
                  + 1j * rng.standard_normal((n, NP + cfg["N"], M))) / np.sqrt(2))
    phi = rng.uniform(0, 2 * np.pi, n)
    y = w + np.exp(2j * np.pi * nu[..., None] * mm[None, None, :] / M
                   + 1j * phi[:, None, None])
    sc = 1.0 / (sigma * np.sqrt(M))
    zp = np.einsum('psm,gm->psg', y[:, :NP, :], TW) * sc
    t1 = np.max(np.abs(zp.sum(1)) ** 2, axis=-1)
    return y, payload, t1


def preamble_kappa_phi(ypre, sigma, lam0):
    """帧池化 kappa-hat/phi-hat：S_g = sum_p FFT_g[p, b_pre(g)]，
    w_g ~ 相干累积似然权重（genie lambda0 标度）。"""
    sc = 1.0 / (sigma * np.sqrt(M))
    n = ypre.shape[0]
    S = np.zeros((n, KGRID), complex)
    for g in range(KGRID):
        Z = np.fft.fft(ypre * TW[g], axis=-1) * sc
        S[:, g] = Z[:, :, BPRE[g]].sum(1)
    c = NP * lam0 / (1.0 + NP * lam0)
    lw = c * np.abs(S) ** 2 / (NP * NP)
    lw -= lw.max(1, keepdims=True)
    lnw = lw - np.log(np.exp(lw).sum(1, keepdims=True))
    ghat = np.argmax(np.abs(S), 1)
    return lnw, np.angle(S[np.arange(n), ghat])


def gate_A(y, sigma, cfg):
    """弱门：genie kappa=0 argmax + 硬汉明 + CRC16。"""
    sc = 1.0 / (sigma * np.sqrt(M))
    Z = np.fft.fft(y[:, NP:, :] * TW[0], axis=-1) * sc
    vals = FOLD1[(np.argmax(np.abs(Z), -1) - 1) & 255]
    nib = hard_hamming_nibbles(rx_vals_to_cwbits(vals, cfg), cfg)
    ok, _ = crc_check(nib, cfg)
    return ok


def gate_B(y, sigma, cfg, lam0, list_mode=False):
    """margF 强门：软度量 -> bit LLR -> 码字 MAP(+Chase 列表) -> CRC16。"""
    cr, BLK, NB = cfg["cr"], cfg["BLK"], cfg["NB"]
    n = y.shape[0]
    lnw, phi_hat = preamble_kappa_phi(y[:, :NP, :], sigma, lam0)
    sc = 1.0 / (sigma * np.sqrt(M))
    S = y.shape[1] - NP
    a0 = np.float32(np.sqrt(lam0))
    rot = np.exp(-1j * phi_hat)[:, None, None]
    # kappa 边缘化（对数域 LSE，流式 max 重标定）
    mx = np.full((n, S), -np.inf)
    acc = np.zeros((n, S, M), np.float32)
    for g in range(KGRID):
        Z = np.fft.fft(y[:, NP:, :] * TW[g], axis=-1) * sc
        xg = (a0 * (Z * rot).real.astype(np.float32))[:, :, PERM[g]] \
            + lnw[:, g][:, None, None].astype(np.float32)
        mnew = np.maximum(mx, xg.max(-1))
        with np.errstate(invalid="ignore"):
            acc *= np.exp(np.where(np.isfinite(mx), mx - mnew, 0.0))[:, :, None]
        acc += np.exp(xg - mnew[:, :, None])
        mx = mnew
    L = np.log(acc) + mx[:, :, None]                       # (n, S, 256)
    # bit LLR（值域边缘化）
    mxv = L.max(-1, keepdims=True)
    e = np.exp(L - mxv)
    LLR = np.zeros((n, S, 8))
    for j in range(8):
        msk = ((np.arange(M) >> (7 - j)) & 1).astype(bool)
        LLR[:, :, j] = np.log(e[:, :, msk].sum(-1)) - np.log(e[:, :, ~msk].sum(-1))
    # 对角 gather -> (n, NB, 8m, cw_len)
    cwl = 4 + cr
    IDX_S = np.zeros((NB, cwl), int)
    IDX_J = np.zeros((8, NB, cwl), int)
    for b in range(NB):
        for i in range(cwl):
            IDX_S[b, i] = b * BLK + i
            for m in range(8):
                IDX_J[m, b, i] = (i - 1 - m) % 8
    Lp = LLR[:, IDX_S.reshape(-1)]                          # (n, NB*cwl, 8)
    Lcw = Lp[:, np.arange(NB * cwl)[None, :], IDX_J.reshape(8, -1)]
    Lcw = Lcw.reshape(n, 8, NB, cwl).transpose(0, 2, 1, 3)  # (n, NB, 8m, cwl)
    scores = Lcw @ VALID_CW[cr].T.astype(np.float32)        # (n, NB, 8m, 16)
    order = np.argsort(-scores, -1)
    top1 = np.take_along_axis(scores, order[..., :1], -1)[..., 0]
    top2 = np.take_along_axis(scores, order[..., 1:2], -1)[..., 0]
    base = order[..., 0].reshape(n, -1)                     # MAP 码字索引=nibble
    ok, _ = crc_check(base, cfg)
    if not list_mode:
        return ok
    # Chase-lite：单翻 12 个最低置信码字 + 最低 6 个的两两翻转（名义 L=28）
    marg = (top1 - top2).reshape(n, -1)
    ordm = np.argsort(marg, 1)
    second = order.reshape(n, -1, 16)[:, :, 1]
    sec_at = np.take_along_axis(second, ordm, 1)            # 各位的次优码字
    for a in range(12):
        cand = base.copy()
        np.put_along_axis(cand, ordm[:, a:a + 1], sec_at[:, a:a + 1], 1)
        okx, _ = crc_check(cand, cfg)
        ok |= okx
    for a in range(6):
        for b2 in range(a + 1, 6):
            cand = base.copy()
            np.put_along_axis(cand, ordm[:, a:a + 1], sec_at[:, a:a + 1], 1)
            np.put_along_axis(cand, ordm[:, b2:b2 + 1], sec_at[:, b2:b2 + 1], 1)
            okx, _ = crc_check(cand, cfg)
            ok |= okx
    return ok


# ---------------------------------------------------------------------------
# 验证：对齐 payload_codec
# ---------------------------------------------------------------------------
def validate_codec():
    from weak_decoder.decoding.payload_codec import (
        interleave_codewords, hamming_encode_nibbles, decode_payload_symbols)
    rng = np.random.default_rng(2026)
    ok_all = True
    for name, cfg in CFGS.items():
        # V1: 对角交织映射 vs 参考（6 个 dummy 前缀使第二块为 payload 块）
        v1 = True
        for _ in range(4):
            nib8 = rng.integers(0, 16, 8)
            stream = np.concatenate([np.zeros(6, int), nib8])
            cws = hamming_encode_nibbles([int(x) for x in stream], sf=8, cr=cfg["cr"])
            ref_sym = interleave_codewords(cws, sf=8, cr=cfg["cr"], ldro=False,
                                           frame_len_nibbles=14)
            ref_bins = np.array([TXBIN[int(v) & 255]
                                 for v in ref_sym[8:8 + cfg["BLK"]]])
            nib_full = np.zeros((1, cfg["NB"] * 8), dtype=np.int64)
            nib_full[0, :8] = nib8
            my_bins = nib_stream_to_bins(nib_full, cfg)[0][:cfg["BLK"]]
            v1 &= np.array_equal(ref_bins, my_bins)
        ok_all &= v1
        # V2: 无噪全链 roundtrip：我的 TX -> 参考 decode_payload_symbols
        nib, payload = gen_payload(rng, 24, cfg)
        bins = nib_stream_to_bins(nib, cfg)
        vals = (bins - 1) & 255
        v2 = True
        for k in range(24):
            r = decode_payload_symbols([int(v) for v in vals[k]], sf=8, cr=cfg["cr"],
                                       ldro=False, payload_len=cfg["plen"],
                                       has_crc=True, crc_mode="grlora")
            v2 &= r.crc_valid and bytes(r.payload_bytes) == bytes(int(x) for x in payload[k])
        ok_all &= v2
        # V3: 我的 RX 硬链无噪 roundtrip
        nibd = hard_hamming_nibbles(rx_vals_to_cwbits(FOLD1[vals], cfg), cfg)
        okm, plm = crc_check(nibd, cfg)
        ok_all &= bool(okm.all()) and bool((plm == payload).all())
        # V4: CRC 线性化 vs 参考
        v4 = True
        for _ in range(8):
            msg = bytes(int(x) for x in rng.integers(0, 256, cfg["plen"]))
            arr = np.frombuffer(msg, np.uint8)
            bt = ((arr[None, :] >> np.arange(7, -1, -1)[None, :, None]) & 1)
            bt = bt.transpose(0, 2, 1).reshape(-1)              # byte-major
            v4 &= int(crc16_bits(bt[None, :])[0]) == ref_crc16(msg)
        ok_all &= v4
        print("validate %-4s: V1 interleave=%s  V2 ref-RX=%s  V3 my-RX=%s  V4 crc=%s"
              % (name, v1, v2, okm.all(), v4), flush=True)
    return ok_all


# ---------------------------------------------------------------------------
def pd09(curve, gammas, lvl=0.9):
    """curve 随 gamma 单调升 -> 穿越点线性插值。"""
    for i in range(len(gammas) - 1):
        if curve[i] < lvl <= curve[i + 1]:
            return float(gammas[i] + (gammas[i + 1] - gammas[i])
                         * (lvl - curve[i]) / max(curve[i + 1] - curve[i], 1e-9))
    return None


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    t0 = time.time()
    print("== codec validation（对齐 weak_decoder payload_codec）==")
    assert validate_codec(), "codec 对齐失败"
    rep = {"validated": True, "seed": 31337}
    rng = np.random.default_rng(31337)

    GAM1 = np.array([-13.0, -12.0, -11.0]) if smoke else np.arange(-17.5, -9.99, 0.5)
    NFR = 300 if smoke else 2000

    print("\n== M1 gate depth（P(CRC)=0.9 门深, per-sample dB, n=%d/gamma）==" % NFR)
    rep["gate_depth"] = {}
    for name, cfg in CFGS.items():
        res = {"A": [], "B": [], "Blist": []}
        for g in GAM1:
            y, payload, t1 = gen_frames(rng, NFR, float(g), cfg)
            sig = 10.0 ** (-g / 20.0)
            lam0 = M * 10 ** (g / 10.0)
            res["A"].append(float(gate_A(y, sig, cfg).mean()))
            res["B"].append(float(gate_B(y, sig, cfg, lam0).mean()))
            res["Blist"].append(float(gate_B(y, sig, cfg, lam0, True).mean()))
            print(" %-4s g=%+.2f  A=%.3f B=%.3f B+=%.3f"
                  % (name, g, res["A"][-1], res["B"][-1], res["Blist"][-1]), flush=True)
        rep["gate_depth"][name] = {
            "gammas": [float(x) for x in GAM1], "curves": res,
            "pd09": {k: {"0.9": pd09(res[k], GAM1), "0.5": pd09(res[k], GAM1, 0.5)}
                     for k in res},
            "pd_at_frontier": {
                "m208": {k: float(np.interp(-20.8, GAM1, res[k])) for k in res},
                "m181": {k: float(np.interp(-18.1, GAM1, res[k])) for k in res}}}
        print(" %-4s Pd.9 门深: A=%s B=%s B+=%s | Pd.5: B=%s"
              % (name, rep["gate_depth"][name]["pd09"]["A"]["0.9"],
                 rep["gate_depth"][name]["pd09"]["B"]["0.9"],
                 rep["gate_depth"][name]["pd09"]["Blist"]["0.9"],
                 rep["gate_depth"][name]["pd09"]["B"]["0.5"]), flush=True)

    print("\n== M2 H0 q_gate（纯噪声 CRC 误过率直接 MC，CR4/5）==")
    qn = 3000 if smoke else 100000
    cfg45 = CFGS["cr45"]
    rep["q_h0"] = {}
    for tag, g_assume in (("lam14", -12.0), ("lam2", -20.8)):
        lam0 = M * 10 ** (g_assume / 10.0)
        n = qn if tag == "lam14" else max(qn // 3, 1000)
        rng2 = np.random.default_rng(555 + int(lam0 * 10))
        cnt = {"A": 0, "B": 0, "Blist": 0}
        done, CH = 0, 250
        while done < n:
            c = min(CH, n - done)
            w = ((rng2.standard_normal((c, NP + cfg45["N"], M))
                  + 1j * rng2.standard_normal((c, NP + cfg45["N"], M))) / np.sqrt(2))
            cnt["A"] += int(gate_A(w, 1.0, cfg45).sum())
            u = gate_B(w, 1.0, cfg45, lam0)
            l = gate_B(w, 1.0, cfg45, lam0, True)
            cnt["B"] += int(u.sum())
            cnt["Blist"] += int(l.sum())
            done += c
        rep["q_h0"][tag] = {"n": n, "assume_gamma_db": g_assume,
                            "count": dict(cnt), "q": {k: v / n for k, v in cnt.items()}}
        print(" %s（假设 g=%+.1f, lam0=%.1f）n=%d : qA=%.2e qB=%.2e qB+=%.2e"
              % (tag, g_assume, lam0, n, cnt["A"] / n, cnt["B"] / n, cnt["Blist"] / n),
              flush=True)

    print("\n== M3 stage-1 frontier（复用 rt_a 生成器 + rt_b 指数尾阈值模型）==")
    import rt_a_tail as A
    q1, q2 = 65.62, 84.84
    b_exp = (q2 - q1) / np.log(10.0)

    def thr_exp(far):
        return q2 + b_exp * np.log(1e-4 / far)

    def frontier_thr(thr, lo, hi, iters=7, n=1500, seed=555):
        for it in range(iters):
            mid = 0.5 * (lo + hi)
            pd = float(np.mean(A.t1_h1(mid, n, seed + 7919 * it) > thr))
            if pd >= 0.9:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    qb_u = max(rep["q_h0"]["lam14"]["q"]["B"], 2.0 ** -16)
    qb_l = max(rep["q_h0"]["lam14"]["q"]["Blist"], 28 * 2.0 ** -16)
    f_u, f_l = 1e-6 / (W * qb_u), 1e-6 / (W * qb_l)
    f_base_r = frontier_thr(thr_exp(1e-9), -21, -15)
    f_u_r = frontier_thr(thr_exp(f_u), -24, -17)
    f_l_r = frontier_thr(thr_exp(f_l), -23, -16)
    print(" 基线 per-cell 1e-9 : frontier=%+.3f dB   (rt_b=-18.14)" % f_base_r)
    print(" A'(unique) q=%.2e -> f_cell=%.2e : frontier=%+.3f dB (rt_b=-20.80)"
          % (qb_u, f_u, f_u_r))
    print(" A'(list28) q=%.2e -> f_cell=%.2e : frontier=%+.3f dB"
          % (qb_l, f_l, f_l_r))
    rep["frontier"] = {"base_1e9": f_base_r, "unique": f_u_r, "list": f_l_r,
                       "f_cell": {"unique": f_u, "list": f_l},
                       "q_used": {"unique": qb_u, "list": qb_l}}

    print("\n== M3b 系统 ROC 联合 MC（list 版 A'：thr=%s, top-K=%d, W=%d）=="
          % (round(thr_exp(f_l), 1), TOPK, W))
    noise_pool = A.t1_h0(60000, 31)
    anchor = rep["gate_depth"]["cr45"]["pd09"]["Blist"]["0.9"]
    gam_sys = [-12.6, -12.1] if (smoke or anchor is None) else [
        round(anchor + d, 3) for d in (0.5, 0.0, -0.5)]
    rng3 = np.random.default_rng(777)
    thr_l = thr_exp(f_l)
    sysres = []
    for g in gam_sys:
        y, payload, t1 = gen_frames(rng3, 2000, g, cfg45)
        lam0 = M * 10 ** (g / 10.0)
        bl = gate_B(y, 10.0 ** (-g / 20.0), cfg45, lam0, True)
        idx = rng3.integers(0, len(noise_pool), size=(2000, W - 1))
        n_above = np.sum(noise_pool[idx] > t1[:, None], 1)
        rank_ok = n_above < TOPK
        sysres.append(dict(gamma=float(g),
                           pd_sys=float(np.mean((t1 > thr_l) & rank_ok & bl)),
                           pd_det=float(np.mean((t1 > thr_l) & rank_ok)),
                           pd_crc=float(np.mean(bl))))
        print(" g=%+.2f: P(det&rank)=%.3f P(crc|B+)=%.3f P_sys=%.3f"
              % (g, sysres[-1]["pd_det"], sysres[-1]["pd_crc"], sysres[-1]["pd_sys"]),
              flush=True)
    rep["system_joint"] = sysres
    sysf = pd09([s["pd_sys"] for s in sysres], [s["gamma"] for s in sysres])
    rep["system_frontier_A_list"] = sysf
    gB45 = rep["gate_depth"]["cr45"]["pd09"]["Blist"]["0.9"]
    gB48 = rep["gate_depth"]["cr48"]["pd09"]["Blist"]["0.9"]
    print(" A' 系统 frontier@Pd.9 = %s dB（门深 B+ cr45=%s, cr48=%s）" % (sysf, gB45, gB48))
    if sysf is not None:
        print(" 净 dB（基线 %+.3f - A' %+.3f）= %+.2f dB" % (f_base_r, sysf, f_base_r - sysf))

    rep["elapsed_s"] = time.time() - t0
    with open("rt3a_revival_results.json", "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("saved rt3a_revival_results.json (%.1f s)" % rep["elapsed_s"])


if __name__ == "__main__":
    main()

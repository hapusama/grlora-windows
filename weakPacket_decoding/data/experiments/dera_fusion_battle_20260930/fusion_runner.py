# -*- coding: utf-8 -*-
r"""2026-09-30 DeRa×LoRaTrimmer 联合 battle：DT-FUSE 联合链入列（9 链）。

用户指令（2026-09-30）：把 DeRa 与 LoRaTrimmer 联合起来和我们的方案 battle。
本 runner 在 dera_battle_20260929（runner v2，8 链）物理零改动的基础上：

  - 新增 **DT-FUSE 联合链** = DERA Stage-2 相干行 + 非相干行（= LoRaTrimmer
    度量）的等增益合并：
        M_fuse[c] = |F_c + e^{-jφ̂0} T_c|² + (|F_c|² + |T_c|²)
    即神招辩论 §1 统一度量 Λ(c) 在衰减因子 ρ=1/2 的工作点（φ̂0 复用 DeRa
    Stage-2 ML 公共相位，无新估计器、零调参）。它代表"est-compensate 阵营
    把 DeRa 与 LoRaTrimmer 两台解码器联合"的最强单判据实现。
  - 我方三链（NEW-0/TREL-5/BCJR-5）改 import `weak_decoder.decoding.
    kappa_trellis`（协议要求，不得内联；与 dera_battle 内联版数值逐位一致，
    preflight_check.py 启动自校验）。
  - 噪声种子派生沿用 **20260929** 常数（刻意不改），与 dera_battle_20260929
    的 v2 战表构成**同种子配对**：8 条共享链应逐位复现，DT-FUSE 为新列。

铁律不变：纯 AWGN、同一实现喂所有链、同一段全先验对齐信号、整包 SNR 口径、
δ 干净信号冻结、GT 只来自干净原生解。DT-FUSE 与其余链拿到同一段信号、
同一份先验、同一判据（judge_crc_fast）。

R2 红线备案（神招辩论 §8.1-7）：DERA port v2 缺 Algorithm 1 Stage 3 与检测
级 CFO——本实验是状态测量，不构成任何"vs DeRa 胜利声明"。
"""
import sys
import csv
import json
import os
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, SymFECSymbolEvidence, build_symfec_symbol_evidence,
    decode_symfec_payload_from_evidences)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator
from weak_decoder.chirp import build_upchirp
from weak_decoder.decoding.kappa_trellis import KappaTrellisDemodulator
from weak_decoder.decoding.kappa_trellis import trellis as kt_trellis

SF, N, OS, NF = 10, 1024, 4, 4096
FEC_CFG = SymFECConfig()
ROT5 = np.stack([np.exp(2j * np.pi * ((2 - d) / 4.0) * np.arange(N) / N) for d in range(5)])
MASK = np.abs(np.fft.fftfreq(NF)) <= 0.125 + 40.0 / NF
REF_OS = np.conj(build_upchirp(SF, symbol_id=0, os_factor=OS))
DERA = DeRaDemodulator(SF, OS)
KT = KappaTrellisDemodulator(sf=SF, os_factor=OS, n_cols=5, radius=1)

CHAINS = ["PLAIN", "OLD-A", "TRIMMER", "DERA", "DT-FUSE", "SAVAUX",
          "NEW-0", "TREL-5", "BCJR-5"]

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dera_fusion_battle_20260930"
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

SOURCES = [
    ("0_0_0_10_14_8",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_8_header_first_frames.csv"),
    ("0_0_0_10_14_16",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_frames.csv"),
    ("0_0_0_10_14_32",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_32.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_32_header_first_frames.csv"),
]

G = {}  # worker 内的全局帧缓存


# ---------------- 与 dera_battle_20260929 完全相同的信号/先验处理 ----------------
def frac_delay(x, samples):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x))
    return np.fft.ifft(X * np.exp(-2j * np.pi * f * samples))


def seg_aligned(iq, r, psym):
    hs = int(r["header_start_sample"])
    seg = np.asarray(iq[hs - 8 * NF: hs + (8 + psym + 1) * NF], dtype=np.complex128)
    n = np.arange(len(seg))
    f = int(r["source_grlora_cfo_int"]) + float(r["source_grlora_cfo_frac"])
    seg = seg * np.exp(-2j * np.pi * f * n / NF)
    return frac_delay(seg, -float(r["source_grlora_payload_sto_frac"]) * OS)


def snr_parts(seg):
    X = np.fft.fftshift(np.fft.fft(seg))
    p = np.abs(X) ** 2 / len(seg)
    f = np.fft.fftshift(np.fft.fftfreq(len(seg))) * 500000.0
    ob = (np.abs(f) > 70000) & (np.abs(f) < 240000)
    n0 = float(np.mean(p[ob]))
    total = float(np.mean(np.abs(seg) ** 2))
    return max(total - n0, 1e-30), n0


def wm(x, k):
    """内联细格窗（仅 build_frames 的 δ 冻结沿用，与 dera_battle 逐位一致；
    战表链一律走 kappa_trellis 模块）。"""
    seg = np.fft.ifft(np.fft.fft(x[k * NF:(k + 1) * NF]) * MASK)
    y = seg * REF_OS
    chips = np.stack([y[p:p + NF:OS] for p in range(OS)])
    inp = (chips[None, :, :] * ROT5[:, None, :]).reshape(20, N)
    P = np.abs(np.fft.fft(inp, axis=1)) ** 2
    return P.reshape(5, 4, N).mean(axis=1).T


def olda_rows(seg, k):
    s = seg[k * NF:(k + 1) * NF].copy()
    return np.abs(np.fft.fft(s[::OS] * REF_OS[::OS])) ** 2


def _lse(x):
    m = x.max()
    return m + np.log(np.sum(np.exp(x - m)))


# ---------------- 向量化证据（与逐 bin 版数值一致，启动自校验） ----------------
def fast_evidence(power_row, symbol_index, floor_db=30.0, top_count=8):
    db = 10.0 * np.log10(np.maximum(power_row, 1e-30))
    rel = np.maximum(db - float(np.max(db)), -abs(floor_db))
    # payload 域：value=(bin-1)%N 单射 => score_by_value[v] = rel[(v+1)%N]
    score_by_value = np.roll(rel, -1)
    best_raw = np.roll(np.arange(N), -1)
    argmax_bin = int(np.argmax(rel))
    order = np.argsort(rel)[::-1][:max(1, int(top_count))]
    return SymFECSymbolEvidence(
        symbol_index=int(symbol_index), raw_scores=rel,
        score_by_value=score_by_value, best_raw_bin_by_value=best_raw,
        argmax_raw_bin=argmax_bin,
        argmax_symbol_value=(argmax_bin - 1) % N,
        peak_margin_db=float(rel[order[0]] - rel[order[1]]) if N > 1 else 0.0,
        top_raw_bins=tuple(int(b) for b in order),
        top_symbol_values=tuple((int(b) - 1) % N for b in order))


def judge_crc_fast(rows_bin, delta, gt_hdr, plen, cr_i):
    """rows_bin: (psym, N) bin 域能量行；值 v 的音在 bin v+delta。

    与 exp1 路径逐位等价：exp1 = roll(rows,-delta) 再 roll_evidence(+1)
    再 builder (bin-1)，净移位 = -(delta-1)；此处一次性完成。
    """
    rc = np.roll(rows_bin, -(int(delta) - 1), axis=1)
    evs = [fast_evidence(rc[k], k) for k in range(rc.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=cr_i, ldro=False, config=FEC_CFG,
        header_symbol_values=list(gt_hdr), payload_len=plen,
        has_crc=True, crc_mode="grlora")
    if res.payload_decode is None:
        return False
    return bool(res.payload_decode.crc_valid)


# ---------------- 各链谱行 ----------------
def chain_rows(seg, psym):
    rows = {}
    rows["OLD-A"] = np.stack([olda_rows(seg, 16 + k) for k in range(psym)])
    tri = np.zeros((psym, N)); sav = np.zeros((psym, N))
    for k in range(psym):
        st = (16 + k) * NF
        tri[k] = trim_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                            cfo_int=0).metric
        sav[k] = np.abs(sav_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                                  cfo_int=0).combined_spectrum) ** 2
    rows["TRIMMER"] = tri
    rows["SAVAUX"] = sav
    _s1d, _cohd, _ncd = DERA.demod_payload_all(seg, 16, psym)
    rows["DERA"] = _cohd
    rows["DT-FUSE"] = _cohd + _ncd   # 联合链：相干 + 非相干 等增益（ρ=1/2 工作点）
    # 我方三链：kappa_trellis 模块（协议要求 import，不内联）；单次 grids 共享
    ms = KT.grids(seg, 16, psym)
    rows["NEW-0"] = np.stack([m[:, 2] for m in ms])
    lam = kt_trellis.emission_prominence(ms)
    path = kt_trellis.viterbi(lam, KT.radius)
    rows["TREL-5"] = np.stack([ms[k][:, path[k]] for k in range(psym)])
    post = kt_trellis.forward_backward(lam, KT.radius)
    rows["BCJR-5"] = np.stack([ms[k] @ post[k] for k in range(psym)])
    return rows


# ---------------- 帧静态数据（与 dera_battle 逐位一致） ----------------
def build_frames():
    frames = []
    for cap, bin_path, csv_path in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            psym = int(r["payload_symbol_count"])
            res = demod_symbol_sequence(
                samples=np.asarray(iq, dtype=np.complex64),
                header_start_sample=int(r["header_start_sample"]), sf=SF, os_factor=OS,
                cfo_int=int(r["source_grlora_cfo_int"]),
                cfo_frac=float(r["source_grlora_cfo_frac"]),
                sfo_hat=float(r["source_grlora_sfo_hat"]),
                sfo_cum_initial=float(r.get("source_grlora_branch_sfo_cum_initial") or 0.0),
                header_count=8, payload_count=psym, payload_ldro=False)
            gt_hdr = [x.symbol_value for x in res[:8]]
            gt = [x.symbol_value for x in res[8:]]
            dec = decode_explicit_frame_symbols(gt_hdr, gt, sf=SF, bw=125000.0, ldro_mode=2)
            assert dec.header.header_valid and dec.payload.crc_valid
            seg = seg_aligned(iq, r, psym)
            S, N0 = snr_parts(seg)
            ms = [wm(seg, 16 + k) for k in range(psym)]
            rows_now = np.stack([m[:, 2] for m in ms])
            d = [(int(np.argmax(rows_now[k])) - g) % N for k, g in enumerate(gt)]
            delta = int(np.bincount(d).argmax())
            frames.append(dict(gt_hdr=gt_hdr, gt=gt, seg=seg, S=S, N0=N0,
                               psym=psym, plen=int(r["payload_len"]),
                               cr=int(r["cr"]), delta=delta))
    return frames


def init_worker():
    global G
    G["frames"] = build_frames()
    # 快路径自校验：与逐 bin 版一致
    rng = np.random.default_rng(0)
    row = rng.exponential(1.0, N)
    a = fast_evidence(row, 0)
    b = build_symfec_symbol_evidence(row, sf=SF, symbol_index=0, ldro=False)
    assert np.allclose(a.score_by_value, b.score_by_value), "快证据不一致"
    assert np.allclose(a.best_raw_bin_by_value % N, b.best_raw_bin_by_value % N)


def run_unit(u):
    level, seed, fi = u  # level: None 或目标 SNR dB
    f = G["frames"][fi]
    seg = f["seg"]
    if level is not None:
        rng = np.random.default_rng((20260929 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi) % (2 ** 31))
        p_add = max(f["S"] / 10 ** (level / 10.0) - f["N0"], 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    rows_ = chain_rows(seg, f["psym"])
    out = {}
    for c in CHAINS:
        cc = f["delta"]
        rc = rows_["OLD-A"] if c == "PLAIN" else rows_[c]
        hard = [(int(np.argmax(rc[k])) - cc) % N for k in range(f["psym"])]
        sym_err = int(sum(int(h != g) for h, g in zip(hard, f["gt"])))
        if c == "PLAIN":
            try:
                dec = decode_explicit_frame_symbols(f["gt_hdr"], hard, sf=SF,
                                                    bw=125000.0, ldro_mode=2)
                crc_ok = bool(dec.header.header_valid and dec.payload.crc_valid)
            except Exception:
                crc_ok = False
        else:
            crc_ok = judge_crc_fast(rc, cc, f["gt_hdr"], f["plen"], f["cr"])
        out[c] = {"sym_err": sym_err, "crc_fail": int(not crc_ok)}
    return {"level": level, "seed": seed, "frame": fi,
            "sym_tot": f["psym"], "chains": out}


def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["level"], r["seed"], r["frame"]))
            except Exception:
                pass
        print("断点恢复：已有 %d 个完成单元" % len(done), flush=True)

    units = [(None, 0, fi) for fi in range(28)]
    for lv in range(-16, -27, -1):
        for sd in range(3):
            for fi in range(28):
                units.append((lv, sd, fi))
    units = [u for u in units if (u[0], u[1], u[2]) not in done]
    print("待跑 %d 单元（%d workers）" % (len(units), os.cpu_count()), flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, os.cpu_count() - 1),
                  initializer=init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units, chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 56 == 0:
                    print("  %d/%d 单元完成 (%.0fs)" % (i + 1, len(units),
                                                        time.time() - t0), flush=True)

    # 汇总
    agg = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["sym_tot"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\n战表（SER / PER；档内单元数见行尾）")
    for key in ["native"] + ["%+d" % v for v in range(-16, -27, -1)]:
        if key not in agg:
            continue
        a = agg[key]
        n_pkt = sum(1 for line in open(CKPT, encoding="utf-8")
                    if ("native" if json.loads(line)["level"] is None
                        else "%+d" % json.loads(line)["level"]) == key)
        parts = " | ".join("%s %.3f/%.3f" % (c, a[c][0] / max(a[c][1], 1),
                                             a[c][2] / n_pkt) for c in CHAINS)
        print("[%8s] %s  (n=%d)" % (key, parts, n_pkt), flush=True)
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

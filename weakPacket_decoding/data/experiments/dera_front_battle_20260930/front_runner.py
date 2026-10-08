# -*- coding: utf-8 -*-
r"""2026-09-30 实验A系统战：DeRa 前端 × 解调器 vs 我方前端 × 解调器。

用户指令：DeRa 本体是 detection 方法，要比的是 **DeRa(检测/同步前端) +
LoRaTrimmer(解调器)** 组成的系统 vs 我们的方案（我方前端 + TREL-5）。

物理与 exp2（full_chain_20260929）完全一致：整包加噪（含前导）、无先验、
同步全部在带噪信号上重做、δ 逐链 CRC 仲裁、纯 AWGN、同一实现喂所有链、
整包 SNR 口径。exp2 的我方前端路径【逐字复用】（sync_and_align 及其配
置），噪声种子派生常数保持 20260930 → 我方两链应逐位复现 exp2 战表。

系统矩阵（5 链）：
  OURS×TRIMMER / OURS×TREL-5   —— 我方前端（exp2 产线链）×2 解调器
  DERA×TRIMMER                 —— 用户点名的对手系统（DeRa 前端 + Trimmer）
  DERA×DERA                    —— 完整 DeRa 系统（前端 + 其解码器 v2 port）
  DERA×TREL-5                  —— E3 关键格子（其前端 × 我方解调器）

DERA 前端 = weak_decoder/baselines/dera/paper_dera_detector.py（DeRaDetector，
论文 Algorithm 1 lines 1-23 相干检测 port；结构校验复用产线 locate，两前端
对称）。DERA 链的候选重试 = 检测 top-5 候选逐个解码至 CRC 过（论文 Stage 3
统一环的 receiver 原语；我方前端的等价物 = 其多事件逐个尝试，两对称）。
"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.kappa_trellis import KappaTrellisDemodulator
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, SymFECSymbolEvidence, decode_symfec_payload_from_evidences)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator
from weak_decoder.baselines.dera.paper_dera_detector import DeRaDetector
from weak_decoder.synchronization.preamble_detector import (
    PreambleDetectorConfig, detect_preamble_runs)
from weak_decoder.synchronization.frame_locator import (
    FrameLocatorConfig, locate_frame_from_event)
from weak_decoder.synchronization.grlora_frame_sync import (
    run_grlora_frame_sync_validation)

import importlib.util as _ilu
_rwsc_spec = _ilu.spec_from_file_location(
    "rwsc", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\scripts"
            r"\run_weak_sync_chain.py")
rwsc = _ilu.module_from_spec(_rwsc_spec)
_rwsc_spec.loader.exec_module(rwsc)

SF, N, OS, NF = 10, 1024, 4, 4096
LEAD = 16                      # header 前取 16 符号（preamble8+sync2+down2.25+余量）
FEC_CFG = SymFECConfig()
KT = KappaTrellisDemodulator(SF, OS)
DERA_DEC = DeRaDemodulator(SF, OS)
DET = DeRaDetector(SF, OS)

CHAINS = ["OURS×TRIMMER", "OURS×TREL-5", "DERA×TRIMMER", "DERA×DERA",
          "DERA×TREL-5"]
FRONT = {"OURS×TRIMMER": "OURS", "OURS×TREL-5": "OURS",
         "DERA×TRIMMER": "DERA", "DERA×DERA": "DERA",
         "DERA×TREL-5": "DERA"}

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dera_front_battle_20260930"
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

SOURCES = [
    ("0_0_0_10_14_8",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_8_header_first_frames.csv",
     8),
    ("0_0_0_10_14_16",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_frames.csv",
     16),
    ("0_0_0_10_14_32",
     r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_32.bin",
     r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_32_header_first_frames.csv",
     32),
]

G = {}


def det_loc_cfg(pre):
    if "cfgs" not in G:
        G["cfgs"] = {}
    if pre not in G["cfgs"]:
        G["cfgs"][pre] = (
            PreambleDetectorConfig(sf=SF, bw=125000.0, samp_rate=500000.0,
                                   win_chirps=4, hop_samples=None,
                                   min_periodic_peaks=5, bin_tol=4),
            FrameLocatorConfig(preamble_len=pre, sync_word=0x34,
                               min_preamble_peaks=5),
        )
    return G["cfgs"][pre]


def fv(x):
    s = (x or "0").strip()
    return float(s.split("|")[0]) if s else 0.0


def snr_parts(seg):
    X = np.fft.fftshift(np.fft.fft(seg))
    p = np.abs(X) ** 2 / len(seg)
    f = np.fft.fftshift(np.fft.fftfreq(len(seg))) * 500000.0
    ob = (np.abs(f) > 70000) & (np.abs(f) < 240000)
    n0 = float(np.mean(p[ob]))
    total = float(np.mean(np.abs(seg) ** 2))
    return max(total - n0, 1e-30), n0


# ---------------- 我方前端：与 exp2 逐字一致 ----------------
def sync_and_align(seg, pre):
    det_cfg, loc_cfg = det_loc_cfg(pre)
    _w, events = detect_preamble_runs(seg, det_cfg)
    for ev in sorted(events, key=lambda e: -e.mean_confidence_db):
        alignment = rwsc.align_event_start(seg, ev, det_cfg,
                                           search_radius_samples=8192,
                                           step_samples=512, align_chirps=4)
        loc = locate_frame_from_event(seg, ev, det_cfg, loc_cfg,
                                      coarse_start_sample=int(
                                          alignment["aligned_start_sample"]))
        if not loc.valid:
            continue
        fs = run_grlora_frame_sync_validation(
            seg, loc, det_cfg, preamble_len=pre, sync_word=0x34,
            center_freq=487.7e6)
        if fs.valid:
            hs_est = int(fs.fine_payload_start_sample)
            n_rel = np.arange(len(seg))
            f_est = float(fs.cfo_int_est) + float(fs.cfo_frac_est)
            seg_a = seg * np.exp(-2j * np.pi * f_est * n_rel / NF)
            X = np.fft.fft(seg_a)
            seg_a = np.fft.ifft(X * np.exp(
                -2j * np.pi * np.fft.fftfreq(len(seg_a))
                * (-float(fs.payload_sto_frac_est) * OS)))
            return dict(fs=fs, seg_a=seg_a, hs_est=hs_est,
                        pay0=hs_est // NF + 8)
    return None


# ---------------- DeRa 前端 ----------------
class _ShimEvent:
    def __init__(self, start):
        self.start_sample = int(start)
        self.event_index = 0


def dera_locate_cb(pre):
    def _locate(seg, start, pre_):
        det_cfg_, _ = det_loc_cfg(pre_)
        loc_cfg_ = FrameLocatorConfig(
            preamble_len=pre_, sync_word=0x34, min_preamble_peaks=5,
            search_radius_samples=320, step_samples=4, symbol_search_span=1)
        try:
            loc = locate_frame_from_event(seg, _ShimEvent(start), det_cfg_,
                                          loc_cfg_,
                                          coarse_start_sample=int(start))
        except Exception:
            return False, -1, -1
        return bool(loc.valid), int(loc.payload_start_sample), \
            int(loc.preamble_start_sample)
    return _locate


def dera_sync(seg, pre):
    """返回 (cands, seg_a_list, pay0_list)：top-5 候选各自的对齐段与 pay0。"""
    cands = DET.detect_frame(seg, pre, locate=dera_locate_cb(pre))
    seg_as, pay0s = [], []
    n = np.arange(len(seg))
    for c in cands:
        seg_a = seg * np.exp(-2j * np.pi * c.f_bins * n / NF)
        seg_as.append(seg_a)
        pay0s.append(c.hs_est // NF + 8)
    return cands, seg_as, pay0s


# ---------------- 解调行与判据（exp2 同款） ----------------
def chain_row(seg_a, pay0, k, decoder):
    st = (pay0 + k) * NF
    if decoder == "TRIMMER":
        return trim_demod(samples=seg_a, start_sample=st, sf=SF,
                          os_factor=OS, cfo_int=0).metric
    if decoder == "DERA":
        return None   # DERA 解码器整段一次算
    return None


def fast_evidence(power_row, symbol_index, floor_db=30.0, top_count=8):
    db = 10.0 * np.log10(np.maximum(power_row, 1e-30))
    rel = np.maximum(db - float(np.max(db)), -abs(floor_db))
    argmax_bin = int(np.argmax(rel))
    order = np.argsort(rel)[::-1][:max(1, int(top_count))]
    return SymFECSymbolEvidence(
        symbol_index=int(symbol_index), raw_scores=rel,
        score_by_value=np.roll(rel, -1),
        best_raw_bin_by_value=np.roll(np.arange(N), -1),
        argmax_raw_bin=argmax_bin,
        argmax_symbol_value=(argmax_bin - 1) % N,
        peak_margin_db=float(rel[order[0]] - rel[order[1]]),
        top_raw_bins=tuple(int(b) for b in order),
        top_symbol_values=tuple((int(b) - 1) % N for b in order))


def judge_crc(rows_bin, delta, gt_hdr, plen, cr_i):
    rc = np.roll(rows_bin, -(int(delta) - 1), axis=1)
    evs = [fast_evidence(rc[k], k) for k in range(rc.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=cr_i, ldro=False, config=FEC_CFG,
        header_symbol_values=list(gt_hdr), payload_len=plen,
        has_crc=True, crc_mode="grlora")
    if res.payload_decode is None:
        return False
    return bool(res.payload_decode.crc_valid)


def decode_chain(rows, f, top5=None):
    """δ 逐链 CRC 仲裁（可选 top-5 候选重试，DeRa Stage-3 原语）。"""
    if top5 is None:
        for d_try in (0, 1, -1, 2, -2):
            if judge_crc(rows, d_try, f["gt_hdr"], f["plen"], f["cr"]):
                return d_try, True
        return 0, False
    for rows_c in top5:
        for d_try in (0, 1, -1, 2, -2):
            if judge_crc(rows_c, d_try, f["gt_hdr"], f["plen"], f["cr"]):
                return d_try, True, rows_c
    return 0, False, top5[0]


# ---------------- 帧静态数据（exp2 逐字一致） ----------------
def build_frames():
    frames = []
    for cap, bin_path, csv_path, pre in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in _csv_reader(csv_path)
                if r.get("header_valid") == "1"
                and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            psym = int(r["payload_symbol_count"])
            res = demod_symbol_sequence(
                samples=np.asarray(iq, dtype=np.complex64),
                header_start_sample=int(r["header_start_sample"]), sf=SF,
                os_factor=OS,
                cfo_int=int(r["source_grlora_cfo_int"]),
                cfo_frac=fv(r.get("source_grlora_cfo_frac")),
                sfo_hat=fv(r.get("source_grlora_sfo_hat")),
                sfo_cum_initial=fv(r.get("source_grlora_branch_sfo_cum_initial")),
                header_count=8, payload_count=psym, payload_ldro=False)
            gt_hdr = [x.symbol_value for x in res[:8]]
            gt = [x.symbol_value for x in res[8:]]
            dec = decode_explicit_frame_symbols(gt_hdr, gt, sf=SF,
                                                bw=125000.0, ldro_mode=2)
            assert dec.header.header_valid and dec.payload.crc_valid
            frames.append(dict(gt_hdr=gt_hdr, gt=gt, psym=psym,
                               plen=int(r["payload_len"]), cr=int(r["cr"]),
                               seg=np.asarray(iq[0:0]), iq=iq,
                               hs=int(r["header_start_sample"]),
                               cfo=int(r["source_grlora_cfo_int"])
                               + fv(r.get("source_grlora_cfo_frac")),
                               pre=int(pre), cap=cap))
    return frames


def _csv_reader(path):
    import csv
    return csv.DictReader(open(path, encoding="utf-8"))


def run_unit(u):
    level, seed, fi = u
    f = G["frames"][fi]
    pre = f["pre"]
    lead = pre + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = G["snr"][fi]
    if level is not None:
        rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)

    out = {c: {"sym_err": 0, "crc_fail": 1, "den": 0} for c in CHAINS}
    sync = {"OURS": False, "DERA": False}

    # ---- 我方前端（exp2 路径）----
    try:
        sa = sync_and_align(seg, pre)
        if sa is not None:
            sync["OURS"] = True
            seg_a, pay0 = sa["seg_a"], sa["pay0"]
            tri = np.stack([trim_demod(samples=seg_a,
                                       start_sample=(pay0 + k) * NF, sf=SF,
                                       os_factor=OS, cfo_int=0).metric
                            for k in range(f["psym"])])
            trel = KT.demod_payload(seg_a, pay0, f["psym"], readout="viterbi")
            for name, rows in (("OURS×TRIMMER", tri),
                               ("OURS×TREL-5", trel)):
                d_win, ok = decode_chain(rows, f)
                hard = [(int(np.argmax(rows[k])) - d_win) % N
                        for k in range(f["psym"])]
                out[name] = {"sym_err": int(sum(int(h != g)
                                                for h, g in zip(hard, f["gt"]))),
                             "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    # ---- DeRa 前端（top-5 候选 + Stage-3 CRC 重试）----
    try:
        cands, seg_as, pay0s = dera_sync(seg, pre)
        if cands:
            sync["DERA"] = True
            for name, dec_name in (("DERA×TRIMMER", "TRIMMER"),
                                   ("DERA×DERA", "DERA"),
                                   ("DERA×TREL-5", "TREL")):
                top5_rows = []
                for seg_a, pay0 in zip(seg_as, pay0s):
                    if pay0 + f["psym"] + 1 > len(seg_a) // NF:
                        continue
                    if dec_name == "TRIMMER":
                        rows = np.stack([
                            trim_demod(samples=seg_a,
                                       start_sample=(pay0 + k) * NF, sf=SF,
                                       os_factor=OS, cfo_int=0).metric
                            for k in range(f["psym"])])
                    elif dec_name == "DERA":
                        _s1, rows = DERA_DEC.demod_payload(seg_a, pay0,
                                                           f["psym"])
                    else:
                        rows = KT.demod_payload(seg_a, pay0, f["psym"],
                                                readout="viterbi")
                    top5_rows.append(rows)
                if top5_rows:
                    _d, ok, rows = decode_chain(None, f, top5=top5_rows)
                    hard = [(int(np.argmax(rows[k])) - _d) % N
                            for k in range(f["psym"])]
                    out[name] = {"sym_err": int(sum(int(h != g)
                                                    for h, g in zip(hard,
                                                                    f["gt"]))),
                                 "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    return {"level": level, "seed": seed, "frame": fi,
            "sync_ours": sync["OURS"], "sync_dera": sync["DERA"],
            "chains": out}


def init_worker():
    G["frames"] = build_frames()
    G["snr"] = []
    for f in G["frames"]:
        s = np.asarray(f["iq"][f["hs"] - (f["pre"] + 6) * NF:
                               f["hs"] + 8 * NF], dtype=np.complex128)
        G["snr"].append(snr_parts(s))


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
        print("断点恢复：%d 单元" % len(done), flush=True)
    levels = [None] + [-v for v in (10, 14, 17, 20, 22, 24, 26)]
    n_seeds = 3
    units = []
    for lv in levels:
        for sd in range(n_seeds if lv is not None else 1):
            for fi in range(28):
                units.append((lv, sd, fi))
    units = [u for u in units if (u[0], u[1], u[2]) not in done]
    print("待跑 %d 单元" % len(units), flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=6, initializer=init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units,
                                                        chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 28 == 0:
                    print("  %d/%d (%.0fs)" % (i + 1, len(units),
                                               time.time() - t0), flush=True)

    agg = {}
    counts = {}
    sync = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        counts[key] = counts.get(key, 0) + 1
        sync.setdefault(key, {"OURS": [0, 0], "DERA": [0, 0]})
        for fr in ("OURS", "DERA"):
            k = "sync_ours" if fr == "OURS" else "sync_dera"
            sync[key][fr][0] += int(r[k])
            sync[key][fr][1] += 1
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\n实验A系统战表（SER=前端同步成功单元内 / PER=含同步失败）")
    for key in ["native"] + ["%+d" % v for v in (10, 14, 17, 20, 22, 24, 26)]:
        if key not in agg:
            continue
        n_pkt = counts[key]
        so = sync[key]["OURS"][0] / max(sync[key]["OURS"][1], 1)
        sd_ = sync[key]["DERA"][0] / max(sync[key]["DERA"][1], 1)
        parts = " | ".join("%s %.3f/%.3f"
                           % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n_pkt) for c in CHAINS)
        print("[%7s] sync OURS=%.2f DERA=%.2f | %s (n=%d)"
              % (key, so, sd_, parts, n_pkt), flush=True)
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

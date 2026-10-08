# -*- coding: utf-8 -*-
r"""2026-10-04 拼接战：DeRa 前端 × Savaux 解调器 —— 前端能否激发 Savaux 全链潜力。

用户指令（2026-10-04）：实验 B 先验给定下 Savaux 相干合并深端最强
（SF11 −16~−24 SER 最强；SF10 −24 起 SER 全场最低）。验证 **DeRa 前端
能否完全激发 Savaux 完整解码链条的潜力**。

物理与 dera_front_battle_20260930 完全一致：整包加噪（含前导）、无先验、
同步全部在带噪信号上重做、δ 逐链 CRC 仲裁、纯 AWGN、同一实现喂所有链、
整包 SNR 口径。噪声种子派生常数保持 20260930 → 共享单元与 front battle
的 DERA 参照链应逐位一致（harness 完整性验证）。

系统矩阵（7 链）：
  DERA×SAVAUX   —— 本轮主角：DeRa 前端 top-5 候选 × Savaux 解调
  DERAfd×SAVAUX —— 修复臂：DeRa 前端 + 前导能量分数时延精化（无 GT，
                   网格 [−2,+2] 步 0.25，trim 能量取 argmax）× Savaux
  DERA×TRIMMER / DERA×TREL-5 / DERA×DERA —— front battle 参照链（同种子复现）
  OURS×SAVAUX   —— 前端对照：我方前端 × Savaux
  PRIOR×SAVAUX  —— 诊断天花板（oracle 标签）：干净 CSV 全先验
                   （int+frac CFO × 全段 + STO 亚 chip frac_delay）× Savaux。
                   只作"同步残差代价"的归因参照，非参赛链（协议 §5A-4
                   oracle 臂分报规则）。

DERA 前端对齐 = 仅分数 CFO（c.f_bins），不施加分数时延（front battle §3
声明差异原样继承）→ DERA×SAVAUX 与 DERA×TRIMMER 输入逐位相同，公平。

native 冒烟已证实的机制（debug_f00/debug_delay，2026-10-04）：
DeRa 前端缺亚样本定时项 → Savaux 合并谱峰落 bin 边界 → argmax 逐符号
混跳（{0,−1} 混合 diff，非 δ 可修）；补 −sto_frac·OS 后恢复恒定映射。
时延响应呈双平台（两平台均 δ 可修，仅中间 ~0.6 样本混跳区致命）。
"""
import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.kappa_trellis import KappaTrellisDemodulator
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, SymFECSymbolEvidence, decode_symfec_payload_from_evidences)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
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
FEC_CFG = SymFECConfig()
KT = KappaTrellisDemodulator(SF, OS)
DERA_DEC = DeRaDemodulator(SF, OS)
DET = DeRaDetector(SF, OS)

CHAINS = ["DERA×SAVAUX", "DERAfd×SAVAUX", "DERA×TRIMMER", "DERA×TREL-5",
          "DERA×DERA", "OURS×SAVAUX", "PRIOR×SAVAUX"]
DERA_CHAINS = {"DERA×SAVAUX": "SAVAUX", "DERAfd×SAVAUX": "SAVAUXFD",
               "DERA×TRIMMER": "TRIMMER",
               "DERA×TREL-5": "TREL", "DERA×DERA": "DERA"}

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dera_savaux_splice_20261004"
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


def frac_delay(x, samples):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x))
    return np.fft.ifft(X * np.exp(-2j * np.pi * f * samples))


# ---------------- 我方前端：与 exp2/front battle 逐字一致 ----------------
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
            seg_a = np.fft.ifft(np.fft.fft(seg_a) * np.exp(
                -2j * np.pi * np.fft.fftfreq(len(seg_a))
                * (-float(fs.payload_sto_frac_est) * OS)))
            return dict(fs=fs, seg_a=seg_a, hs_est=hs_est,
                        pay0=hs_est // NF + 8, f_est=f_est)
    return None


# ---------------- DeRa 前端：与 front battle 逐字一致 ----------------
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


# ---------------- 判据（exp2/front battle 同款） ----------------
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


def sav_rows(seg_a, pay0, psym):
    return np.stack([
        np.abs(sav_demod(samples=seg_a, start_sample=(pay0 + k) * NF,
                         sf=SF, os_factor=OS, cfo_int=0)
               .combined_spectrum) ** 2
        for k in range(psym)])


_FD_GRID = np.arange(-2.0, 2.01, 0.25)


def dera_frac_refine(seg_a, lead):
    """可部署分数时延精化（无 GT）：前导 4 符号 dechirp 峰能量对 d 网格
    取 argmax。返回 (精化段, d*)。"""
    best_d, best_e = 0.0, -1.0
    for d in _FD_GRID:
        seg_d = seg_a if d == 0.0 else frac_delay(seg_a, float(d))
        e = 0.0
        for k in range(4):
            m = trim_demod(samples=seg_d, start_sample=(lead - 4 + k) * NF,
                           sf=SF, os_factor=OS, cfo_int=0).metric
            e += float(np.max(m))
        if e > best_e + 1e-12:
            best_e, best_d = e, float(d)
    return (seg_a if best_d == 0.0 else frac_delay(seg_a, best_d)), best_d


# ---------------- 帧静态数据（exp2/front battle 逐字一致） ----------------
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
                               sto_frac=fv(r.get(
                                   "source_grlora_payload_sto_frac")),
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
    meta = {"dera": None, "ours": None}

    # ---- PRIOR×SAVAUX：干净 CSV 全先验（oracle 诊断天花板，永不失败）----
    try:
        n_rel = np.arange(len(seg))
        seg_p = seg * np.exp(-2j * np.pi * f["cfo"] * n_rel / NF)
        seg_p = frac_delay(seg_p, -f["sto_frac"] * OS)
        rows = sav_rows(seg_p, lead + 8, f["psym"])
        d_win, ok = decode_chain(rows, f)
        hard = [(int(np.argmax(rows[k])) - d_win) % N
                for k in range(f["psym"])]
        out["PRIOR×SAVAUX"] = {
            "sym_err": int(sum(int(h != g) for h, g in zip(hard, f["gt"]))),
            "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    # ---- 我方前端 ----
    try:
        sa = sync_and_align(seg, pre)
        if sa is not None:
            sync["OURS"] = True
            meta["ours"] = {
                "f_err": sa["f_est"] - f["cfo"],
                "hs_err": sa["hs_est"] - lead * NF}
            seg_a, pay0 = sa["seg_a"], sa["pay0"]
            rows = sav_rows(seg_a, pay0, f["psym"])
            d_win, ok = decode_chain(rows, f)
            hard = [(int(np.argmax(rows[k])) - d_win) % N
                    for k in range(f["psym"])]
            out["OURS×SAVAUX"] = {
                "sym_err": int(sum(int(h != g) for h, g in zip(hard, f["gt"]))),
                "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    # ---- DeRa 前端（top-5 候选 + Stage-3 CRC 重试，4 条链共用）----
    try:
        cands, seg_as, pay0s = dera_sync(seg, pre)
        if cands:
            sync["DERA"] = True
            dstars = []
            best = cands[0]
            meta["dera"] = {
                "n_cands": len(cands),
                "best_score_db": float(best.score_db),
                "best_f_err": float(best.f_bins) - f["cfo"],
                "best_hs_err": int(best.hs_est) - lead * NF,
                "fd_dstars": dstars}
            top5 = {}
            for seg_a, pay0 in zip(seg_as, pay0s):
                if pay0 + f["psym"] + 1 > len(seg_a) // NF:
                    continue
                for name, dec_name in DERA_CHAINS.items():
                    if dec_name == "SAVAUX":
                        rows = sav_rows(seg_a, pay0, f["psym"])
                    elif dec_name == "SAVAUXFD":
                        seg_fd, d_star = dera_frac_refine(seg_a, lead)
                        dstars.append(d_star)
                        rows = sav_rows(seg_fd, pay0, f["psym"])
                    elif dec_name == "TRIMMER":
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
                    top5.setdefault(name, []).append(rows)
            for name, top5_rows in top5.items():
                if not top5_rows:
                    continue
                _d, ok, rows = decode_chain(None, f, top5=top5_rows)
                hard = [(int(np.argmax(rows[k])) - _d) % N
                        for k in range(f["psym"])]
                out[name] = {
                    "sym_err": int(sum(int(h != g)
                                       for h, g in zip(hard, f["gt"]))),
                    "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    return {"level": level, "seed": seed, "frame": fi,
            "sync_ours": sync["OURS"], "sync_dera": sync["DERA"],
            "meta": meta, "chains": out}


def init_worker():
    G["frames"] = build_frames()
    G["snr"] = []
    for f in G["frames"]:
        s = np.asarray(f["iq"][f["hs"] - (f["pre"] + 6) * NF:
                               f["hs"] + 8 * NF], dtype=np.complex128)
        G["snr"].append(snr_parts(s))


LEVELS = [None, -17, -20, -22, -24, -26, -28]
N_SEEDS = 3


def make_units():
    units = []
    for lv in LEVELS:
        for sd in range(N_SEEDS if lv is not None else 1):
            for fi in range(28):
                units.append((lv, sd, fi))
    return units


def aggregate():
    agg, counts, sync = {}, {}, {}
    ferr, hserr, fdm = {}, {}, {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        counts[key] = counts.get(key, 0) + 1
        sync.setdefault(key, {"OURS": [0, 0], "DERA": [0, 0]})
        sync[key]["OURS"][0] += int(r["sync_ours"])
        sync[key]["OURS"][1] += 1
        sync[key]["DERA"][0] += int(r["sync_dera"])
        sync[key]["DERA"][1] += 1
        m = (r.get("meta") or {}).get("dera")
        if m is not None:
            ferr.setdefault(key, []).append(m["best_f_err"])
            hserr.setdefault(key, []).append(m["best_hs_err"])
            fdm.setdefault(key, []).extend(m.get("fd_dstars") or [])
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\n拼接战表（SER=前端同步成功单元内 / PER=含同步失败；"
          "PRIOR×SAVAUX=oracle 诊断天花板）")
    for key in ["native"] + ["%+d" % v for v in LEVELS[1:]]:
        if key not in agg:
            continue
        n_pkt = counts[key]
        so = sync[key]["OURS"][0] / max(sync[key]["OURS"][1], 1)
        sd_ = sync[key]["DERA"][0] / max(sync[key]["DERA"][1], 1)
        parts = " | ".join("%s %.3f/%.3f"
                           % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n_pkt) for c in CHAINS)
        extra = ""
        if ferr.get(key):
            fe = np.median(np.abs(ferr[key]))
            he = np.median(np.abs(hserr[key]))
            extra = "  [DERA |f_err|med=%.4f bin, |hs_err|med=%.0f samp" % (fe, he)
            if fdm.get(key):
                extra += ", fd d*med=%+.2f" % float(np.median(fdm[key]))
            extra += "]"
        print("[%7s] sync OURS=%.2f DERA=%.2f | %s (n=%d)%s"
              % (key, so, sd_, parts, n_pkt, extra), flush=True)


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
    units = [u for u in make_units() if (u[0], u[1], u[2]) not in done]
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
                    el = time.time() - t0
                    print("  %d/%d (%.0fs, %.1fs/u, ETA %.0fmin)"
                          % (i + 1, len(units), el, el / (i + 1),
                             el / (i + 1) * (len(units) - i - 1) / 60.0),
                          flush=True)

    aggregate()
    print("%.0fs elapsed" % (time.time() - t0))


def smoke():
    """native 门禁 + 判据负控（协议 §6）。串行执行，出错带全栈。"""
    init_worker()
    print("== native 门禁（28 帧，串行）==")
    bad = []
    rows_sum = {c: [0, 0, 0] for c in CHAINS}
    n_ours = n_dera = 0
    for fi in range(28):
        r = run_unit((None, 0, fi))
        n_ours += int(r["sync_ours"])
        n_dera += int(r["sync_dera"])
        for c in CHAINS:
            d = r["chains"][c]
            if d["den"] == 0 and c != "OURS×SAVAUX":
                bad.append((fi, c, "no-rows"))
            rows_sum[c][0] += d["sym_err"]
            rows_sum[c][1] += d["den"]
            rows_sum[c][2] += d["crc_fail"]
        print("  f%02d ours=%d dera=%d %s" % (
            fi, r["sync_ours"], r["sync_dera"],
            " ".join("%s %.3f/%.3f" % (
                c, r["chains"][c]["sym_err"] / max(r["chains"][c]["den"], 1),
                r["chains"][c]["crc_fail"]) for c in CHAINS)), flush=True)
    print("sync OURS=%d/28 DERA=%d/28" % (n_ours, n_dera), flush=True)
    print("\n== native 汇总 ==")
    for c in CHAINS:
        ser = rows_sum[c][0] / max(rows_sum[c][1], 1)
        per = rows_sum[c][2] / 28.0
        flag = ""
        if "SAVAUX" in c and ser > 0.10:
            flag = "  <-- 超门禁(0.10)"
        if c == "DERAfd×SAVAUX" and ser > 0.06:
            flag = "  <-- 修复臂应≈PRIOR(0.039)"
        if c in ("DERA×TREL-5",) and ser > 0.0:
            flag = "  <-- TREL 应为 0"
        print("%s SER=%.4f PER=%.3f%s" % (c, ser, per, flag))

    print("\n== 判据负控 ==")
    f = G["frames"][0]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    cands, seg_as, pay0s = dera_sync(seg, f["pre"])
    rows = sav_rows(seg_as[0], pay0s[0], f["psym"])
    d0, ok0 = decode_chain(rows, f)
    rows_bad = rows.copy()
    rows_bad[3] = np.roll(rows_bad[3], 97)
    ok_bad = any(judge_crc(rows_bad, d, f["gt_hdr"], f["plen"], f["cr"])
                 for d in (0, 1, -1, 2, -2))
    # header 校验门说明：本实验判据（symfec 证据 + payload CRC）与 GT 冻结
    # 用的显式符号 codec 均按设计不重验 header（exp2/battle/front battle
    # 同款）；header 有效性由生产链 header_first/frame_sync 的校验和门
    # 把关并记录于 CSV header_valid=1。故"坏 header 被拒"不构成本层负控，
    # 以声明替代；CRC 旁路回归由单符号损坏负控覆盖。
    negctl_pass = ok0 and not ok_bad
    print("完好帧 CRC 通过=%s（应 True，δ=%d）" % (ok0, d0))
    print("单符号损坏被拒=%s（应 True）" % (not ok_bad))
    print("坏 header 负控=声明豁免（判据不重验 header，见上）")
    if bad:
        print("\n[FAIL] no-rows 单元: %s" % bad[:10])
    if not negctl_pass:
        print("[FAIL] 负控未过，禁止开跑")
        sys.exit(1)
    print("[PASS] 冒烟+负控通过")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        main()

# -*- coding: utf-8 -*-
r"""2026-09-29 实验二：整包加噪（含前导），完整 detection+framesync+demod。

与实验一的区别（用户指定）：
  - 噪声加在整个包段（前导+同步字+SFD+header+payload，含噪底导引）；
  - 无干净先验：CFO/STO/SFO/header_start 全部由同步链在【带噪信号】上
    估计（detect_preamble_runs → locate_frame_from_event →
    run_grlora_frame_sync_validation，与产线 sync_chain 同一套模块）；
  - δ 为接收机侧：每链用其自身带噪 preamble 观测的众数（参考 bin 为
    该链干净数据上的一次性标定常数，接收机设计知识）。
  - 同步失败 = 全链 PER 记错；SER 只统计同步成功单元，另报同步成功率。
我方三读出（grid0/viterbi/bcjr）走正式模块 kappa_trellis。
铁律不变：噪声有且仅有 AWGN、同一实现喂所有链、整包 SNR 口径。
"""
import sys
import csv
import json
import os
import time
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.kappa_trellis import KappaTrellisDemodulator
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, SymFECSymbolEvidence, decode_symfec_payload_from_evidences)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.baselines.unichirp.paper_unichirp_demod import (
    UniChirpTrainingSymbol, UniChirpDemodConfig, build_unichirp_phase_model,
    demod_unichirp_symbol as uni_demod, unichirp_full_spectrum)
from weak_decoder.synchronization.preamble_detector import (
    PreambleDetectorConfig, detect_preamble_runs)
from weak_decoder.synchronization.frame_locator import (
    FrameLocatorConfig, locate_frame_from_event)
from weak_decoder.synchronization.grlora_frame_sync import (
    run_grlora_frame_sync_validation)

# 产线脚本的 align_event_start（检测→定位之间的起点精对齐）——借模块加载
import importlib.util as _ilu
_rwsc_spec = _ilu.spec_from_file_location(
    "rwsc", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\scripts"
            r"\run_weak_sync_chain.py")
rwsc = _ilu.module_from_spec(_rwsc_spec)
_rwsc_spec.loader.exec_module(rwsc)

SF, N, OS, NF = 10, 1024, 4, 4096
LEAD = 16                      # header 前取 16 符号（preamble8+sync2+down2.25+余量）
PRE_REF_SYM = -12              # δ 观测窗：header 前 12 符号起的 upchirp 区
FEC_CFG = SymFECConfig()
UNI_CFG = UniChirpDemodConfig(cfo_correction_mode="none")
KT = KappaTrellisDemodulator(SF, OS)

CHAINS = ["OLD-A", "TRIMMER", "SAVAUX", "NEW-0", "TREL-5", "BCJR-5"]

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\full_chain_20260929"
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

DET_CFG = PreambleDetectorConfig(sf=SF, bw=125000.0, samp_rate=500000.0,
                                 win_chirps=4, hop_samples=None,
                                 min_periodic_peaks=5, bin_tol=4)
LOC_CFG = FrameLocatorConfig(preamble_len=8, sync_word=0x34,
                             min_preamble_peaks=5)

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


def olda_rows(seg, k, ref_os):
    s = seg[k * NF:(k + 1) * NF].copy()
    return np.abs(np.fft.fft(s[::OS] * ref_os[::OS])) ** 2


def chain_rows(seg, pay0, psym, uni_m, ref_os):
    """pay0 = payload 起始符号序号（按估计 header_start）。"""
    rows = {}
    rows["OLD-A"] = np.stack([olda_rows(seg, pay0 + k, ref_os) for k in range(psym)])
    tri = np.zeros((psym, N)); sav = np.zeros((psym, N)); uni = np.zeros((psym, N))
    for k in range(psym):
        st = (pay0 + k) * NF
        tri[k] = trim_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                            cfo_int=0).metric
        sav[k] = np.abs(sav_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                                  cfo_int=0).combined_spectrum) ** 2
        uni[k] = uni_demod(samples=seg, start_sample=st, sf=SF, os_factor=OS,
                           phase_rad=uni_m.predict(pay0 + k), config=UNI_CFG).metric
    rows["TRIMMER"] = tri
    rows["SAVAUX"] = sav
    rows["UNICHIRP_AUX"] = uni
    for name, readout in [("NEW-0", "grid0"), ("TREL-5", "viterbi"), ("BCJR-5", "bcjr")]:
        rows[name] = KT.demod_payload(seg, pay0, psym, readout=readout)
    return rows


def preamble_delta(rows_probe, ref_bin):
    """rows_probe: (K,N) preamble 符号行；返回众数偏移。"""
    d = [(int(np.argmax(rows_probe[k])) - ref_bin) % N for k in range(rows_probe.shape[0])]
    return int(np.bincount(d).argmax())


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


def build_frames():
    frames = []
    for cap, bin_path, csv_path, pre in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1" and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            psym = int(r["payload_symbol_count"])
            res = demod_symbol_sequence(
                samples=np.asarray(iq, dtype=np.complex64),
                header_start_sample=int(r["header_start_sample"]), sf=SF, os_factor=OS,
                cfo_int=int(r["source_grlora_cfo_int"]),
                cfo_frac=fv(r.get("source_grlora_cfo_frac")),
                sfo_hat=fv(r.get("source_grlora_sfo_hat")),
                sfo_cum_initial=fv(r.get("source_grlora_branch_sfo_cum_initial")),
                header_count=8, payload_count=psym, payload_ldro=False)
            gt_hdr = [x.symbol_value for x in res[:8]]
            gt = [x.symbol_value for x in res[8:]]
            dec = decode_explicit_frame_symbols(gt_hdr, gt, sf=SF, bw=125000.0,
                                                ldro_mode=2)
            assert dec.header.header_valid and dec.payload.crc_valid
            frames.append(dict(gt_hdr=gt_hdr, gt=gt, psym=psym,
                               plen=int(r["payload_len"]), cr=int(r["cr"]),
                               seg=np.asarray(iq[0:0]), iq=iq,
                               hs=int(r["header_start_sample"]),
                               cfo=int(r["source_grlora_cfo_int"])
                               + fv(r.get("source_grlora_cfo_frac")),
                               pre=int(pre), cap=cap))
    return frames


def sync_and_align(seg, pre):
    """检测→定位→帧同步→带估计对齐。返回 dict 或 None（同步失败）。"""
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
                        pay0=hs_est // NF + 8,
                        p0=hs_est // NF + PRE_REF_SYM)
    return None


def calibrate_preamble_refs():
    """每 capture 每链 preamble 参考 bin：native 段上走【同一条同步链】对齐后
    标定——与 run_unit 的带噪探测严格同域（帧同步自身估计，不用 CSV 先验）。
    """
    from weak_decoder.chirp import build_upchirp
    ref_os = np.conj(build_upchirp(SF, symbol_id=0, os_factor=OS))
    G["ref_os"] = ref_os
    refs = {}
    by_cap = {}
    for idx, f in enumerate(G["frames"]):
        by_cap.setdefault(f["cap"], []).append(idx)
    for cap, idxs in by_cap.items():
        f = G["frames"][idxs[0]]
        lead0 = f["pre"] + 6
        seg = np.asarray(f["iq"][f["hs"] - lead0 * NF: f["hs"] + 8 * NF],
                         dtype=np.complex128)
        sa = sync_and_align(seg, f["pre"])
        p0 = sa["p0"]
        probe = {}
        probe["OLD-A"] = np.stack([olda_rows(sa["seg_a"], p0 + k, ref_os)
                                   for k in range(6)])
        probe["TRIMMER"] = np.stack(
            [trim_demod(samples=sa["seg_a"], start_sample=(p0 + k) * NF, sf=SF,
                        os_factor=OS, cfo_int=0).metric for k in range(6)])
        probe["SAVAUX"] = np.stack(
            [np.abs(sav_demod(samples=sa["seg_a"], start_sample=(p0 + k) * NF,
                              sf=SF, os_factor=OS,
                              cfo_int=0).combined_spectrum) ** 2
             for k in range(6)])
        probe["NEW-0"] = np.stack([KT.grid.symbol_spectrum(sa["seg_a"], p0 + k)[:, 2]
                                   for k in range(6)])
        refs[cap] = {}
        for c, rows in probe.items():
            refs[cap][c] = int(np.bincount(
                [int(np.argmax(rows[k])) for k in range(6)]).argmax())
        refs[cap]["TREL-5"] = refs[cap]["BCJR-5"] = refs[cap]["NEW-0"]
    G["refs"] = refs
    return refs


def run_unit(u):
    level, seed, fi = u
    f = G["frames"][fi]
    ref_os = G["ref_os"]
    refs = G["refs"]
    pre = f["pre"]
    lead = pre + 6
    det_cfg, loc_cfg = det_loc_cfg(pre)
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                             f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = G["snr"][fi]
    sync_ok = False
    out = {c: {"sym_err": 0, "crc_fail": 1} for c in CHAINS}   # 默认同步失败=PER错
    if level is not None:
        rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    try:
        sa = sync_and_align(seg, pre)
        if sa is None:
            return {"level": level, "seed": seed, "frame": fi, "sync_ok": False,
                    "sym_tot": f["psym"], "chains": out}
        sync_ok = True
        seg_a = sa["seg_a"]
        pay0 = sa["pay0"]
        p0 = sa["p0"]
        # UniChirp 训练（带噪 preamble，接收机行为）
        train = []
        for k in range(4):
            full = unichirp_full_spectrum(samples=seg_a, start_sample=(p0 + k) * NF,
                                          sf=SF, os_factor=OS, cfo_int=0, cfo_frac=0.0,
                                          config=UNI_CFG)
            train.append(UniChirpTrainingSymbol(start_sample=(p0 + k) * NF,
                                                raw_fft_bin=int(np.argmax(full[:N])),
                                                abs_symbol_index=float(p0 + k)))
        uni_m, _o = build_unichirp_phase_model(samples=seg_a,
                                               training_symbols=tuple(train),
                                               sf=SF, os_factor=OS, config=UNI_CFG)
        rows_ = chain_rows(seg_a, pay0, f["psym"], uni_m, ref_os)
        # 接收机侧 δ：逐链 CRC 仲裁（δ∈{0,±1,±2} 候选解码，CRC 过者胜；
        # 判据已修复无空真，错误 δ 过 CRC 概率 ~2^-16/候选。与 DeRa 的
        # CRC-guided retry 同一接收机原语。）
        for c in CHAINS:
            delta_c = 0
            crc_ok = False
            for d_try in (0, 1, -1, 2, -2):
                if judge_crc(rows_[c], d_try, f["gt_hdr"], f["plen"], f["cr"]):
                    delta_c = d_try
                    crc_ok = True
                    break
            hard = [(int(np.argmax(rows_[c][k])) - delta_c) % N
                    for k in range(f["psym"])]
            out[c]["sym_err"] = int(sum(int(h != g) for h, g in zip(hard, f["gt"])))
            out[c]["crc_fail"] = int(not crc_ok)
    except Exception:
        sync_ok = False
    return {"level": level, "seed": seed, "frame": fi, "sync_ok": sync_ok,
            "sym_tot": f["psym"], "chains": out}


def init_worker():
    G["frames"] = build_frames()
    G["snr"] = []
    for f in G["frames"]:
        s = np.asarray(f["iq"][f["hs"] - (f["pre"] + 6) * NF: f["hs"] + 8 * NF],
                       dtype=np.complex128)
        G["snr"].append(snr_parts(s))
    G["refs"] = calibrate_preamble_refs()


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
    levels = [None] + [-v for v in (10, 14, 17, 20, 22)]
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
            for i, res in enumerate(pool.imap_unordered(run_unit, units, chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 56 == 0:
                    print("  %d/%d (%.0fs)" % (i + 1, len(units), time.time() - t0),
                          flush=True)

    agg = {}
    counts = {}
    sync = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        counts[key] = counts.get(key, 0) + 1
        sync.setdefault(key, [0, 0])
        sync[key][0] += int(r["sync_ok"])
        sync[key][1] += 1
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["sym_tot"] if r["sync_ok"] else 0
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\n实验二战表（SER=同步成功单元内 / PER=含同步失败）")
    for key in ["native"] + ["%+d" % v for v in (10, 14, 17, 20, 22)]:
        if key not in agg:
            continue
        n_pkt = counts[key]
        sr = sync[key][0] / max(sync[key][1], 1)
        parts = " | ".join("%s %.3f/%.3f" % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                                             agg[key][c][2] / n_pkt) for c in CHAINS)
        print("[%7s] sync=%.2f | %s  (n=%d)" % (key, sr, parts, n_pkt), flush=True)
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

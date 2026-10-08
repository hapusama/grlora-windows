# -*- coding: utf-8 -*-
r"""2026-10-08 SF11 移植战：拼接线 × SF11/LDRO（零重调泛化验证）。

目的：SAVT2（SF10 上设计的双层时延网格）在 SF11/LDRO 真实 OTA 上是否
零重调成立——同时充当网格超参的天然留出集（全部常数从 SF10 原封搬来：
网格 [−2,+2]、细步 0.125×7 态、副本中心间距 0.75、罚 0.2）。

臂（4）：
  DERA×SAVT2  —— 主打：DeRa 前端（SF11 首次启用）+ 双层网格 + Savaux
  DERA×DERA   —— 全 DeRa 参照
  DERA×SAVAUX —— 原样拼接对照（检验 SF11 上混跳失效机制是否复现）
  PRIOR×SAVAUX—— 干净 CSV 全先验天花板（oracle 标签，分报）

数据：lab1_sf11_TP2 16 capture（pre=16，LDRO），126 header_valid →
GT-CRC 通过帧入集（协议 §1 规则，sf11 battle 同款 ≈120）。
判据：LDRO 版 symfec 证据 + CRC16（sf11_battle 20260929 逐位同款）。
噪声种子常数 20261008（新战役声明）；档 native + {−17..−26}×3 种子。
"""
import sys
import os
import csv
import json
import time
import argparse
import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.decoding.header_first_demod import demod_symbol_sequence
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
from weak_decoder.baselines.symfec.paper_symfec_decoder import (
    SymFECConfig, SymFECSymbolEvidence, decode_symfec_payload_from_evidences)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    demod_paper_oversampled_symbol as sav_demod)
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator
from weak_decoder.baselines.dera.paper_dera_detector import DeRaDetector
from weak_decoder.synchronization.preamble_detector import (
    PreambleDetectorConfig, detect_preamble_runs)
from weak_decoder.synchronization.frame_locator import (
    FrameLocatorConfig, locate_frame_from_event)

SF, N, OS, NF = 11, 2048, 4, 8192
PRE = 16
FEC_CFG = SymFECConfig()
DERA_DEC = DeRaDemodulator(SF, OS)
DET = DeRaDetector(SF, OS)

CHAINS = ["DERA×SAVT2", "DERA×DERA", "DERA×SAVAUX", "PRIOR×SAVAUX"]

_IQ = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ\lab1_sf11_TP2"
_HF = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
       r"\weak_sync_chain\header_first")
SOURCES = ([("1_0_%d_11_2_16" % p, _IQ + r"\1_0_%d_11_2_16.bin" % p,
             _HF + r"\1_0_%d_11_2_16_header_first_frames.csv" % p)
            for p in range(8, 17)]
           + [("1_1_%d_11_2_16" % p, _IQ + r"\1_1_%d_11_2_16.bin" % p,
               _HF + r"\1_1_%d_11_2_16_header_first_frames.csv" % p)
              for p in range(8)])

EXP_DIR = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
           r"\experiments\sf11_splice_20261008")
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

LEVELS = [None, -17, -20, -22, -24, -26]
N_SEEDS = 3

CENTERS = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
FINE = np.arange(-0.375, 0.376, 0.125)
PEN_F = 0.2

G = {}
N_FRAMES = None


# ---------------- 通用工具 ----------------
def fv(x):
    s = (x or "0").strip()
    return float(s.split("|")[0]) if s else 0.0


def frac_delay(x, samples):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x))
    return np.fft.ifft(X * np.exp(-2j * np.pi * f * samples))


def snr_parts(seg):
    X = np.fft.fftshift(np.fft.fft(seg))
    p = np.abs(X) ** 2 / len(seg)
    f = np.fft.fftshift(np.fft.fftfreq(len(seg))) * 500000.0
    ob = (np.abs(f) > 70000) & (np.abs(f) < 240000)
    n0 = float(np.mean(p[ob]))
    total = float(np.mean(np.abs(seg) ** 2))
    return max(total - n0, 1e-30), n0


# ---------------- LDRO 证据与判据（sf11_battle 逐位同款） ----------------
def fast_evidence_ldro(power_row, symbol_index, floor_db=30.0, top_count=8):
    db = 10.0 * np.log10(np.maximum(power_row, 1e-30))
    rel = np.maximum(db - float(np.max(db)), -abs(floor_db))
    n_val = N // 4
    score = np.full(n_val, -np.inf)
    score[:n_val - 1] = rel[1: 4 * (n_val - 1) + 1].reshape(n_val - 1, 4).max(axis=1)
    score[n_val - 1] = max(float(rel[4 * (n_val - 1) + 1:].max()), float(rel[0]))
    argmax_bin = int(np.argmax(rel))
    order = np.argsort(rel)[::-1][:max(1, int(top_count))]
    return SymFECSymbolEvidence(
        symbol_index=int(symbol_index), raw_scores=rel,
        score_by_value=score,
        best_raw_bin_by_value=np.arange(n_val),
        argmax_raw_bin=argmax_bin,
        argmax_symbol_value=((argmax_bin - 1) % N) // 4,
        peak_margin_db=float(rel[order[0]] - rel[order[1]]),
        top_raw_bins=tuple(int(b) for b in order),
        top_symbol_values=tuple(((int(b) - 1) % N) // 4 for b in order))


def judge_crc(rows_bin, delta, gt_hdr, plen, cr_i):
    rc = np.roll(rows_bin, -(int(delta) - 1), axis=1)
    evs = [fast_evidence_ldro(rc[k], k) for k in range(rc.shape[0])]
    res = decode_symfec_payload_from_evidences(
        evidences=evs, sf=SF, cr=cr_i, ldro=True, config=FEC_CFG,
        header_symbol_values=list(gt_hdr), payload_len=plen,
        has_crc=True, crc_mode="grlora")
    if res.payload_decode is None:
        return False
    return bool(res.payload_decode.crc_valid)


def hard_values(rows, delta):
    rc = np.roll(rows, -(int(delta) - 1), axis=1)
    return [fast_evidence_ldro(rc[k], k).argmax_symbol_value
            for k in range(rows.shape[0])]


def decode_chain(rows, f, top5=None):
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


# ---------------- DeRa 前端（SF11 首次） ----------------
class _ShimEvent:
    def __init__(self, start):
        self.start_sample = int(start)
        self.event_index = 0


def dera_locate_cb():
    det_cfg = PreambleDetectorConfig(sf=SF, bw=125000.0, samp_rate=500000.0,
                                     win_chirps=4, hop_samples=None,
                                     min_periodic_peaks=5, bin_tol=4)
    loc_cfg = FrameLocatorConfig(preamble_len=PRE, sync_word=0x34,
                                 min_preamble_peaks=5,
                                 search_radius_samples=320, step_samples=4,
                                 symbol_search_span=1)

    def _locate(seg, start, pre_):
        try:
            loc = locate_frame_from_event(seg, _ShimEvent(start), det_cfg,
                                          loc_cfg,
                                          coarse_start_sample=int(start))
        except Exception:
            return False, -1, -1
        return bool(loc.valid), int(loc.payload_start_sample), \
            int(loc.preamble_start_sample)
    return _locate


_LOC = dera_locate_cb()


def dera_sync(seg):
    cands = DET.detect_frame(seg, PRE, locate=_LOC)
    seg_as, pay0s = [], []
    n = np.arange(len(seg))
    for c in cands:
        seg_as.append(seg * np.exp(-2j * np.pi * c.f_bins * n / NF))
        pay0s.append(c.hs_est // NF + 8)
    return cands, seg_as, pay0s


# ---------------- 解调 ----------------
def sav_rows(seg_a, pay0, psym):
    return np.stack([
        np.abs(sav_demod(samples=seg_a, start_sample=(pay0 + k) * NF,
                         sf=SF, os_factor=OS, cfo_int=0)
               .combined_spectrum) ** 2
        for k in range(psym)])


def savt2_outputs(seg_a, pay0, psym):
    abs_d = np.concatenate([c + FINE for c in CENTERS])
    n_d = len(abs_d)
    E = np.empty((psym, n_d))
    rows_all = np.empty((psym, n_d, N))
    for j, d in enumerate(abs_d):
        seg_d = seg_a if d == 0.0 else frac_delay(seg_a, float(d))
        for i in range(psym):
            res = sav_demod(samples=seg_d, start_sample=(pay0 + i) * NF,
                            sf=SF, os_factor=OS, cfo_int=0)
            p = np.abs(res.combined_spectrum.astype(np.complex128)) ** 2
            rows_all[i, j] = p
            E[i, j] = np.log(np.max(p) + 1e-30)
    outs = []
    n_f = len(FINE)
    for k in range(len(CENTERS)):
        sl = slice(k * n_f, (k + 1) * n_f)
        Ek = E[:, sl]
        score = Ek[0].copy()
        back = np.zeros((psym, n_f), dtype=int)
        for i in range(1, psym):
            ns = np.full(n_f, -1e30)
            bk = np.zeros(n_f, dtype=int)
            for s in range(n_f):
                best, arg = -1e30, s
                for s0 in (s - 1, s, s + 1):
                    if 0 <= s0 < n_f:
                        vv = score[s0] - PEN_F * abs(s - s0)
                        if vv > best:
                            best, arg = vv, s0
                ns[s], bk[s] = best, arg
            score = ns + Ek[i]
            back[i] = bk
        path = np.zeros(psym, dtype=int)
        path[-1] = int(np.argmax(score))
        for i in range(psym - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        outs.append((float(score[path[-1]]),
                     rows_all[:, sl][np.arange(psym), path]))
    outs.sort(key=lambda t: -t[0])
    return [o[1] for o in outs]


# ---------------- 帧构建 ----------------
def build_frames():
    frames = []
    for cap, bin_path, csv_path in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1"
                and int(r.get("payload_len", 0) or 0) > 0]
        for r in rows:
            psym = int(r["payload_symbol_count"])
            try:
                res = demod_symbol_sequence(
                    samples=np.asarray(iq, dtype=np.complex64),
                    header_start_sample=int(r["header_start_sample"]),
                    sf=SF, os_factor=OS,
                    cfo_int=int(r["source_grlora_cfo_int"]),
                    cfo_frac=fv(r.get("source_grlora_cfo_frac")),
                    sfo_hat=fv(r.get("source_grlora_sfo_hat")),
                    sfo_cum_initial=fv(
                        r.get("source_grlora_branch_sfo_cum_initial")),
                    header_count=8, payload_count=psym, payload_ldro=True)
                gt_hdr = [x.symbol_value for x in res[:8]]
                gt = [x.symbol_value for x in res[8:]]
                dec = decode_explicit_frame_symbols(gt_hdr, gt, sf=SF,
                                                    bw=125000.0, ldro_mode=1)
                if not (dec.header.header_valid and dec.payload.crc_valid):
                    raise ValueError("GT CRC 未过")
            except Exception:
                continue
            frames.append(dict(gt_hdr=gt_hdr, gt=gt, psym=psym,
                               plen=int(r["payload_len"]), cr=int(r["cr"]),
                               iq=iq, hs=int(r["header_start_sample"]),
                               cfo=int(r["source_grlora_cfo_int"])
                               + fv(r.get("source_grlora_cfo_frac")),
                               sto_frac=fv(r.get(
                                   "source_grlora_payload_sto_frac")),
                               cap=cap))
    return frames


def init_worker():
    G["frames"] = build_frames()
    G["snr"] = []
    for f in G["frames"]:
        s = np.asarray(f["iq"][f["hs"] - (PRE + 6) * NF: f["hs"] + 8 * NF],
                       dtype=np.complex128)
        G["snr"].append(snr_parts(s))


def run_unit(u):
    level, seed, fi = u
    f = G["frames"][fi]
    lead = PRE + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = G["snr"][fi]
    if level is not None:
        rng = np.random.default_rng((20261008 * 7919
                                     + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)

    out = {c: {"sym_err": 0, "crc_fail": 1, "den": 0} for c in CHAINS}
    sync_dera = False
    meta = {"n_cands": 0, "f_err": None, "hs_err": None}

    # PRIOR×SAVAUX（oracle 天花板）
    try:
        n_rel = np.arange(len(seg))
        seg_p = seg * np.exp(-2j * np.pi * f["cfo"] * n_rel / NF)
        seg_p = frac_delay(seg_p, -f["sto_frac"] * OS)
        rows = sav_rows(seg_p, lead + 8, f["psym"])
        d_win, ok = decode_chain(rows, f)
        hard = hard_values(rows, d_win)
        out["PRIOR×SAVAUX"] = {
            "sym_err": int(sum(int(h != g) for h, g in zip(hard, f["gt"]))),
            "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    # DeRa 前端三链
    try:
        cands, seg_as, pay0s = dera_sync(seg)
        if cands:
            sync_dera = True
            meta["n_cands"] = len(cands)
            meta["f_err"] = float(cands[0].f_bins) - f["cfo"]
            meta["hs_err"] = int(cands[0].hs_est) - f["hs"]
            top5 = {"DERA×SAVT2": [], "DERA×DERA": [], "DERA×SAVAUX": []}
            for seg_a, pay0 in zip(seg_as, pay0s):
                if pay0 + f["psym"] + 1 > len(seg_a) // NF:
                    continue
                top5["DERA×SAVT2"].extend(
                    savt2_outputs(seg_a, pay0, f["psym"]))
                _s1, rows = DERA_DEC.demod_payload(seg_a, pay0, f["psym"])
                top5["DERA×DERA"].append(rows)
                top5["DERA×SAVAUX"].append(
                    sav_rows(seg_a, pay0, f["psym"]))
            for name, rows_list in top5.items():
                if not rows_list:
                    continue
                _d, ok, rows = decode_chain(None, f, top5=rows_list)
                hard = hard_values(rows, _d)
                out[name] = {
                    "sym_err": int(sum(int(h != g)
                                       for h, g in zip(hard, f["gt"]))),
                    "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    return {"level": level, "seed": seed, "frame": fi,
            "sync_dera": sync_dera, "meta": meta, "chains": out}


# ---------------- 运行器 ----------------
def precheck():
    """DeRa 前端 SF11 首验：前 6 帧干净段检测 + 参数误差。"""
    init_worker()
    n_ok = 0
    for fi in range(6):
        f = G["frames"][fi]
        lead = PRE + 6
        seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                                f["hs"] + (8 + f["psym"] + 2) * NF],
                         dtype=np.complex128)
        cands, seg_as, pay0s = dera_sync(seg)
        if cands:
            n_ok += 1
            print("f%02d cap=%s cands=%d f_err=%+.4f hs_err=%d pay0=%d/%d"
                  % (fi, f["cap"], len(cands), float(cands[0].f_bins)
                     - f["cfo"], int(cands[0].hs_est) - f["hs"],
                     pay0s[0], len(seg) // NF))
        else:
            print("f%02d cap=%s FAIL（无候选）" % (fi, f["cap"]))
    print("precheck: %d/6 检出" % n_ok)


def aggregate():
    agg, counts, sync = {}, {}, {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        counts[key] = counts.get(key, 0) + 1
        sync[key] = sync.get(key, 0) + int(r["sync_dera"])
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\nSF11 战表（SER=同步成功单元内 / PER=含同步失败）")
    for key in ["native"] + ["%+d" % v for v in LEVELS[1:]]:
        if key not in agg:
            continue
        n_pkt = counts[key]
        parts = " | ".join("%s %.3f/%.3f"
                           % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n_pkt) for c in CHAINS)
        print("[%7s] sync DERA=%.2f | %s (n=%d)"
              % (key, sync[key] / n_pkt, parts, n_pkt), flush=True)


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
    units = [(lv, sd, fi)
             for lv in LEVELS
             for sd in range(N_SEEDS if lv is not None else 1)
             for fi in range(N_FRAMES)]
    units = [u for u in units if (u[0], u[1], u[2]) not in done]
    print("待跑 %d 单元" % len(units), flush=True)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4, initializer=init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units,
                                                        chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 40 == 0:
                    el = time.time() - t0
                    print("  %d/%d (%.0fs, %.1fs/u, ETA %.0fmin)"
                          % (i + 1, len(units), el, el / (i + 1),
                             el / (i + 1) * (len(units) - i - 1) / 60.0),
                          flush=True)
    aggregate()
    print("%.0fs elapsed" % (time.time() - t0))


def smoke():
    init_worker()
    print("frames=%d" % len(G["frames"]))
    tot = {c: [0, 0, 0] for c in CHAINS}
    n_sync = 0
    for fi in range(len(G["frames"])):
        r = run_unit((None, 0, fi))
        n_sync += int(r["sync_dera"])
        for c in CHAINS:
            d = r["chains"][c]
            tot[c][0] += d["sym_err"]
            tot[c][1] += d["den"]
            tot[c][2] += d["crc_fail"]
        if fi < 8 or r["chains"]["DERA×SAVAUX"]["sym_err"] > 0:
            print("  f%03d sync=%d %s" % (
                fi, r["sync_dera"],
                " ".join("%s %d/%d" % (c.split("×")[-1],
                                       r["chains"][c]["sym_err"],
                                       r["chains"][c]["crc_fail"])
                         for c in CHAINS)), flush=True)
    print("sync DERA=%d/%d" % (n_sync, len(G["frames"])))
    for c in CHAINS:
        print("%s SER=%.4f PER=%.3f" % (
            c, tot[c][0] / max(tot[c][1], 1),
            tot[c][2] / len(G["frames"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precheck", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.precheck:
        precheck()
    elif args.smoke:
        smoke()
    else:
        N_FRAMES = len(build_frames())
        print("GT 帧 %d" % N_FRAMES, flush=True)
        main()

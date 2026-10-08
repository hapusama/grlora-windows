# -*- coding: utf-8 -*-
r"""sys_cert_battle v2 — cert 检测 + 符号网格 f̂ 复测（对齐精度修复）。

v1 发现（sys_checkpoint.jsonl）：cert fallback 恢复 221 单元、同步轴全档
≥ DeRa 前端，但恢复包 SER .29~.90——根因=f̂ 在粗对齐网格（512 步）上
测量，时频模糊（256 样本偏移 ⇒ tone 移 64 bin）打飞 f̂。

v2 修复（轻量，~15 行）：cert 选中事件后，locate 落定符号网格，再在
**located 网格上复测 f̂**——复用 DeRa port 的精化原语（DET._score_rows，
与 DeRa 前端内部同一套），两前端同精化、检测是唯一变量（公平性声明）。

链（新跑 2 条，OURS/DERA 行复用 v1 checkpoint 同种子合并）：
  CERT2×TREL-5   cert v2 前端 × 我方解码器
  CERT2×DERA-dec cert v2 前端 × DeRa 解码器（直射 DERA×DERA）

预注册判据：
  P1 前端 superiority：CERT2×TREL-5 PER ≤ DERA×TREL-5 @−20/−22/−24；
  P2 检测 superiority：CERT2×DERA-dec PER ≤ DERA×DERA @−20/−22
     （= 战胜 DeRa 完整系统的中段）。
"""
import sys
import os
import json
import time

import numpy as np

EX = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments"
sys.path.insert(0, EX + r"\fullfield_cert_detect_20261003")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "fr", EX + r"\dera_front_battle_20260930\front_runner.py")
FR = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(FR)

import ff_runner as R

SF, N, OS, NF = FR.SF, FR.N, FR.OS, FR.NF
EXP_DIR = EX + r"\sys_cert_battle_20261003"
CKPT = os.path.join(EXP_DIR, "sys_checkpoint_v4.jsonl")
LEVELS = [None] + [-v for v in (20, 22, 24, 26)]
N_SEEDS = 3
N_EVENT_TRY = 3


def cert_event_score(seg, start, pre):
    """部署式 cert 统计量（同 v1：无 GT 锚，κ̂ 前 4 窗+bank，三段幅度和）。"""
    if start < 0 or start + (pre + 4) * NF > len(seg):
        return None
    cols, z = [], []
    for j in range(pre + 4):
        ref = R.DET.up_ref if j >= pre + 2 else R.DET.down_ref
        row = R.DET.signed_spectrum(seg, start + j * NF, ref)
        mag = np.abs(row)
        k = int(np.argmax(mag))
        if k < R.DET.NOISE_GUARD or k > len(mag) - R.DET.NOISE_GUARD:
            return None
        cols.append(k)
        z.append(complex(row[k]))
    row0 = np.abs(R.DET.signed_spectrum(seg, start, R.DET.down_ref))
    far = np.ones(len(row0), dtype=bool)
    for k in set(cols):
        far[max(0, k - 16):k + 17] = False
    sig2 = float(np.median(row0[far] ** 2))
    if sig2 <= 0:
        return None
    z = np.array(z)
    z_est = z[:4]
    F = np.abs(np.fft.fft(z_est, 256))
    q0 = int(np.argmax(F))
    best, kap = 0.0, 0.0
    for db in range(-4, 5):
        kt = (q0 + db) / 256
        kt = kt - 1.0 if kt > 0.5 else kt
        v = float(np.abs(np.sum(z_est * np.exp(
            -1j * 2 * np.pi * kt * np.arange(4)))) ** 2)
        if v > best:
            best, kap = v, kt
    w = np.exp(-1j * 2 * np.pi * kap * np.arange(pre + 4, dtype=float))
    zw = z * w
    seg1 = float(np.abs(np.sum(zw[4:pre]))) / np.sqrt(max(pre - 4, 1))
    seg2 = float(np.abs(np.sum(zw[pre:pre + 2]))) / np.sqrt(2)
    zd = zw[pre + 2:]
    seg3 = max(float(np.abs(np.sum(zd))),
               float(np.abs(np.sum(np.conj(zd))))) / np.sqrt(2)
    k_up = (float(np.median(cols[:4])) - N) / 2.0
    return dict(score=seg1 + seg2 + seg3,
                f_coarse=round(k_up - kap) + kap, kappa=float(kap))


def remeasure_f_on_grid(seg, hs_est, pre):
    """located 符号网格上的 f̂ 复测（DeRa port 精化原语 DET._score_rows）。"""
    hs_sym = int(round(hs_est / NF))
    pre_meas = (hs_sym - pre - 4) * NF - NF // 4
    if pre_meas < 0:
        return None
    try:
        rows = np.stack([R.DET.signed_spectrum(seg, pre_meas + i * NF,
                                               R.DET.down_ref)
                         for i in range(pre)])
    except Exception:
        return None
    m = R.DET._score_rows(rows, pre, pre_meas, 0)
    if m is None:
        return None
    return round(m["k_up"] - m["kappa"]) + m["kappa"]


def cert_sync_v2(seg, pre, det_cfg, loc_cfg):
    """OURS 失败后的 cert v4 fallback。

    v4 两处修复（诊断驱动）：① locate 用 DeRa 前端同款紧配置
    （radius=320/step=4/span=1）；② pay0 双网格假设 {0,+1}——诊断实测
    fallback 的 payload 网格比 DeRa 低 1 符号（f̂ 偏差由 δ 仲裁吸收）。
    返回候选对齐列表（≤3 事件 × 2 网格），由 run_unit 逐个解码至 CRC。
    """
    n = np.arange(len(seg))
    _w, events = FR.detect_preamble_runs(seg, det_cfg)
    loc_cfg_tight = FR.FrameLocatorConfig(
        preamble_len=pre, sync_word=0x34, min_preamble_peaks=5,
        search_radius_samples=320, step_samples=4, symbol_search_span=1)
    scored = []
    for ev in events:
        try:
            al = FR.rwsc.align_event_start(seg, ev, det_cfg,
                                           search_radius_samples=8192,
                                           step_samples=512,
                                           align_chirps=4)
        except Exception:
            continue
        s0 = int(al["aligned_start_sample"])
        sc = cert_event_score(seg, s0, pre)
        if sc is not None:
            scored.append((sc["score"], ev, al, sc, s0))
    scored.sort(key=lambda t: -t[0])
    cands = []
    for sc, ev, al, info, s0 in scored[:N_EVENT_TRY]:
        try:
            k_up0, k_dn0 = R.DET._k_up_k_dn(seg, pre, s0)
            u = k_up0 - k_dn0
            e_corr = int(round(2.0 * (u - R.DET.BRACKET_U0))) \
                if abs(u) <= 200 else 0
        except Exception:
            e_corr = 0
        s_fixed = s0 - e_corr
        loc = None
        for s_try in (s_fixed, s0):
            try:
                cand = FR.locate_frame_from_event(
                    seg, ev, det_cfg, loc_cfg_tight,
                    coarse_start_sample=int(s_try))
            except Exception:
                continue
            if cand.valid:
                loc = cand
                break
        if loc is None:
            continue
        hs_est = int(loc.payload_start_sample)
        f_bins = remeasure_f_on_grid(seg, hs_est, pre)
        if f_bins is None:
            f_bins = info["f_coarse"]
        seg_a = seg * np.exp(-2j * np.pi * f_bins * n / NF)
        p0 = hs_est // NF + 8
        for dp in (0, 1):
            cands.append(dict(seg_a=seg_a, pay0=p0 + dp, f_bins=f_bins,
                              hs_est=hs_est))
    return cands


def run_unit(u):
    level, seed, fi = u
    f = FR.G["frames"][fi]
    pre = f["pre"]
    lead = pre + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                             f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = FR.G["snr"][fi]
    if level is not None:
        rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100)
                                     * 131 + seed * 17 + fi * 7919)
                                    % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) \
            * np.sqrt(p_add / 2.0)

    out = {"CERT2×TREL-5": {"sym_err": 0, "crc_fail": 1, "den": 0},
           "CERT2×DERA": {"sym_err": 0, "crc_fail": 1, "den": 0}}
    sync = {"OURS": False, "CERT2": False}
    det_cfg, loc_cfg = FR.det_loc_cfg(pre)

    sa = None
    try:
        sa = FR.sync_and_align(seg, pre)
        sync["OURS"] = sa is not None
    except Exception:
        sa = None
    sa_c = sa
    cands = []
    if sa is None:
        try:
            cands = cert_sync_v2(seg, pre, det_cfg, loc_cfg)
            if cands:
                sync["CERT2"] = True
        except Exception:
            cands = []
    if sa is not None:
        sync["CERT2"] = True                 # CERT2 ⊇ OURS
        try:
            rows = FR.KT.demod_payload(sa["seg_a"], sa["pay0"],
                                       f["psym"], readout="viterbi")
            d_win, ok = FR.decode_chain(rows, f)
            hard = [(int(np.argmax(rows[k])) - d_win) % N
                    for k in range(f["psym"])]
            out["CERT2×TREL-5"] = {
                "sym_err": int(sum(int(h != g) for h, g in
                                   zip(hard, f["gt"]))),
                "crc_fail": int(not ok), "den": f["psym"]}
            _s1, drows = FR.DERA_DEC.demod_payload(sa["seg_a"],
                                                   sa["pay0"], f["psym"])
            d2, ok2 = FR.decode_chain(drows, f)
            hard2 = [(int(np.argmax(drows[k])) - d2) % N
                     for k in range(f["psym"])]
            out["CERT2×DERA"] = {
                "sym_err": int(sum(int(h != g) for h, g in
                                   zip(hard2, f["gt"]))),
                "crc_fail": int(not ok2), "den": f["psym"]}
        except Exception:
            pass
    else:
        # fallback 候选逐个解码至 CRC（pay0 双网格假设含在内）
        best_trel = best_dera = None
        for cand in cands[:6]:
            try:
                if best_trel is None or not best_trel[1]:
                    rows = FR.KT.demod_payload(cand["seg_a"], cand["pay0"],
                                               f["psym"], readout="viterbi")
                    d_win, ok = FR.decode_chain(rows, f)
                    hard = [(int(np.argmax(rows[k])) - d_win) % N
                            for k in range(f["psym"])]
                    rec = {"sym_err": int(sum(int(h != g) for h, g in
                                              zip(hard, f["gt"]))),
                           "crc_fail": int(not ok), "den": f["psym"]}
                    if best_trel is None or (best_trel[1] and not ok):
                        best_trel = (rec, ok)
                if best_dera is None or not best_dera[1]:
                    _s1, drows = FR.DERA_DEC.demod_payload(
                        cand["seg_a"], cand["pay0"], f["psym"])
                    d2, ok2 = FR.decode_chain(drows, f)
                    hard2 = [(int(np.argmax(drows[k])) - d2) % N
                             for k in range(f["psym"])]
                    rec2 = {"sym_err": int(sum(int(h != g) for h, g in
                                               zip(hard2, f["gt"]))),
                            "crc_fail": int(not ok2), "den": f["psym"]}
                    if best_dera is None or (best_dera[1] and not ok2):
                        best_dera = (rec2, ok2)
                if best_trel is not None and best_trel[1] \
                        and best_dera is not None and best_dera[1]:
                    break
            except Exception:
                continue
        if best_trel is not None:
            out["CERT2×TREL-5"] = best_trel[0]
        if best_dera is not None:
            out["CERT2×DERA"] = best_dera[0]
    return {"level": level, "seed": seed, "frame": fi,
            "sync_ours": sync["OURS"], "sync_cert2": sync["CERT2"],
            "chains": out}


def init_worker():
    FR.init_worker()


def main():
    t0 = time.time()
    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["level"], r["seed"], r["frame"]))
            except Exception:
                pass
        print("断点恢复：%d" % len(done), flush=True)
    units = []
    for lv in LEVELS:
        for sd in range(N_SEEDS if lv is not None else 1):
            for fi in range(28):
                units.append((lv, sd, fi))
    units = [u for u in units if (u[0], u[1], u[2]) not in done]
    print("待跑 %d" % len(units), flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=6, initializer=init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(run_unit, units,
                                                        chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 56 == 0:
                    print("  %d/%d (%.0fs)" % (i + 1, len(units),
                                               time.time() - t0), flush=True)
    # 汇总：CERT2 新跑 + v1 checkpoint 的 OURS/DERA 行 + front_battle 的
    # DERA×DERA 行（同种子派生，逐位可比）
    agg, counts, sync = {}, {}, {}
    for src, keys in ((CKPT, ("CERT2×TREL-5", "CERT2×DERA")),
                      (EX + r"\sys_cert_battle_20261003\sys_checkpoint.jsonl",
                       ("OURS×TREL-5", "DERA×TREL-5")),
                      (EX + r"\dera_front_battle_20260930\checkpoint.jsonl",
                       ("DERA×DERA",))):
        for line in open(src, encoding="utf-8"):
            r = json.loads(line)
            key = "native" if r["level"] is None else "%+d" % r["level"]
            counts.setdefault(key, 0)
            sync.setdefault(key, {"OURS": [0, 0], "CERT2": [0, 0],
                                  "DERA": [0, 0]})
            if src == CKPT:
                counts[key] += 1
                sync[key]["OURS"][0] += int(r["sync_ours"])
                sync[key]["OURS"][1] += 1
                sync[key]["CERT2"][0] += int(r["sync_cert2"])
                sync[key]["CERT2"][1] += 1
            elif src.endswith("sys_checkpoint.jsonl"):
                if key in ("native", "-20", "-22", "-24", "-26"):
                    sync[key]["DERA"][0] += int(r["sync_dera"])
                    sync[key]["DERA"][1] += 1
            for c in keys:
                a = agg.setdefault(key, {}).setdefault(
                    c, [0, 0, 0])
                a[0] += r["chains"][c]["sym_err"]
                a[1] += r["chains"][c]["den"]
                a[2] += r["chains"][c]["crc_fail"]
    print("\n===== v2 系统战表（SER/PER，TREL-5/DERA 解码器）=====")
    order = ["CERT2×TREL-5", "CERT2×DERA", "OURS×TREL-5",
             "DERA×TREL-5", "DERA×DERA"]
    for key in ["native", "-20", "-22", "-24", "-26"]:
        if key not in counts or counts[key] == 0:
            continue
        n = counts[key]
        parts = []
        for c in order:
            if c in agg.get(key, {}) and agg[key][c][2] > 0:
                parts.append("%s %.3f/%.3f" % (c.split("×")[1],
                                               agg[key][c][0]
                                               / max(agg[key][c][1], 1),
                                               agg[key][c][2] / n))
            elif c in agg.get(key, {}):
                parts.append("%s —/1.0" % c.split("×")[1])
        sr = {k: (round(v[0] / v[1], 2) if v[1] else "—")
              for k, v in sync[key].items()}
        print("[%6s] n=%d | %s | sync OURS/CERT2/DERA = %s"
              % (key, n, " | ".join(parts), sr), flush=True)
    print("%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

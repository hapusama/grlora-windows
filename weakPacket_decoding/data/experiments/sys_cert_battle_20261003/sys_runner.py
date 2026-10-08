# -*- coding: utf-8 -*-
r"""sys_cert_battle_20261003 — E3-lite：cert 检测统计量接入真实管线的系统级对比。

用户质询：机制级 cert8 未连接真实 decoding 工作与 DeRa 系统比较 → 本轮补充。

设计（复用 dera_front_battle_20260930/front_runner.py 全部基础设施，
噪声种子派生同常数 20260930 → OURS×TREL-5 行与其战表逐位可比）：

  三前端 × 固定解码器 TREL-5：
    OURS  我方产线前端（exp2 逐字：scan→align→locate→GRLora 验证）
    CERT  = OURS 管线；失败时 cert fallback：
            detect_preamble_runs 事件 → align_event_start 粗对齐 →
            cert 部署式统计量排序（κ̂ 从前 4 窗估计（与确认样本不相交，
            bank±4）；确认=后半前导+sync+SFD 三段幅度和，无类型模型、
            无 GT 锚——部署形态）→ top-3 事件 locate（结构校验）→
            Eq.18 类 f̂=round(k_up−κ̂)+κ̂ → TREL-5 → δ 仲裁 CRC。
            无门限：CRC 是终审，误事件自然失败计入 PER。
    DERA  DeRa 前端 port（交叉验证过，top-5 候选重试=其 Stage-3 原生）

  公平性：CERT ⊇ OURS（先走 OURS 全路径，失败才 fallback）→ CERT−OURS
  差 = fallback 的边际恢复量，即机制级净胜在真实管线中的兑现。
  指标：sync 率 / PER（含同步失败）/ cert 恢复数。

档位：native + −20/−22/−24/−26 × 3 种子 × 28 帧 = 448 单元。
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

import ff_runner as R          # DET.signed_spectrum / 参考啁啾
SF, N, OS, NF = FR.SF, FR.N, FR.OS, FR.NF
M = 256
EXP_DIR = EX + r"\sys_cert_battle_20261003"
CKPT = os.path.join(EXP_DIR, "sys_checkpoint.jsonl")
LEVELS = [None] + [-v for v in (20, 22, 24, 26)]
N_SEEDS = 3


def cert_event_score(seg, start, pre):
    """部署式 cert 统计量（无 GT 锚）：返回 (score, f_bins, kappa) 或 None。"""
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
    rows_mag = np.stack([np.abs(R.DET.signed_spectrum(
        seg, start + j * NF, R.DET.down_ref if j < pre + 2
        else R.DET.up_ref)) for j in range(pre)]) if False else None
    # σ̂²：前导行的远列（用第一行近似——部署形态，声明）
    row0 = np.abs(R.DET.signed_spectrum(seg, start, R.DET.down_ref))
    far = np.ones(len(row0), dtype=bool)
    for k in set(cols):
        far[max(0, k - 16):k + 17] = False
    sig2 = float(np.median(row0[far] ** 2))
    if sig2 <= 0:
        return None
    z = np.array(z)
    # stage-1：κ̂ 从前 4 窗（与确认样本不相交），bank ±4
    z_est = z[:4]
    F = np.abs(np.fft.fft(z_est, M))
    q0 = int(np.argmax(F))
    best, kap = 0.0, 0.0
    for db in range(-4, 5):
        kap_t = (q0 + db) / M
        kap_t = kap_t - 1.0 if kap_t > 0.5 else kap_t
        v = float(np.abs(np.sum(z_est * np.exp(
            -1j * 2 * np.pi * kap_t * np.arange(4)))) ** 2)
        if v > best:
            best, kap = v, kap_t
    j_idx = np.arange(pre + 4, dtype=float)
    w = np.exp(-1j * 2 * np.pi * kap * j_idx)
    zw = z * w
    k1 = pre - 4
    seg1 = float(np.abs(np.sum(zw[4:pre]))) / np.sqrt(k1)
    seg2 = float(np.abs(np.sum(zw[pre:pre + 2]))) / np.sqrt(2)
    zd = zw[pre + 2:]
    seg3 = max(float(np.abs(np.sum(zd))),
               float(np.abs(np.sum(np.conj(zd))))) / np.sqrt(2)
    score = seg1 + seg2 + seg3
    # Eq.18 类 f̂：k_up=前导列中位（0.5 格 signed）
    k_up = (float(np.median(cols[:4])) - N) / 2.0
    f_bins = round(k_up - kap) + kap
    return dict(score=score, f_bins=float(f_bins), kappa=float(kap))


def cert_fallback(seg, pre, det_cfg, loc_cfg):
    """OURS 失败后的 cert fallback：返回 OURS 返回值同构 dict 或 None。"""
    _w, events = FR.detect_preamble_runs(seg, det_cfg)
    scored = []
    n = np.arange(len(seg))
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
            scored.append((sc["score"], ev, al, sc))
    scored.sort(key=lambda t: -t[0])
    for sc, ev, al, info in scored[:3]:
        try:
            loc = FR.locate_frame_from_event(
                seg, ev, det_cfg, loc_cfg,
                coarse_start_sample=int(al["aligned_start_sample"]))
        except Exception:
            continue
        if not loc.valid:
            continue
        hs_est = int(loc.payload_start_sample)
        seg_a = seg * np.exp(-2j * np.pi * info["f_bins"] * n / NF)
        return dict(seg_a=seg_a, hs_est=hs_est, pay0=hs_est // NF + 8,
                    kappa=info["kappa"], f_bins=info["f_bins"],
                    recovered=True)
    return None


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
        rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)

    out = {"OURS×TREL-5": {"sym_err": 0, "crc_fail": 1, "den": 0},
           "CERT×TREL-5": {"sym_err": 0, "crc_fail": 1, "den": 0},
           "DERA×TREL-5": {"sym_err": 0, "crc_fail": 1, "den": 0}}
    sync = {"OURS": False, "CERT": False, "DERA": False}
    cert_recovered = 0

    det_cfg, loc_cfg = FR.det_loc_cfg(pre)

    # ---- OURS 前端 ----
    sa = None
    try:
        sa = FR.sync_and_align(seg, pre)
        if sa is not None:
            sync["OURS"] = True
            rows = FR.KT.demod_payload(sa["seg_a"], sa["pay0"],
                                       f["psym"], readout="viterbi")
            d_win, ok = FR.decode_chain(rows, f)
            hard = [(int(np.argmax(rows[k])) - d_win) % N
                    for k in range(f["psym"])]
            out["OURS×TREL-5"] = {
                "sym_err": int(sum(int(h != g) for h, g in
                                   zip(hard, f["gt"]))),
                "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        sa = None

    # ---- CERT 前端 = OURS + fallback ----
    sa_c = sa
    if sa is None:
        try:
            fb = cert_fallback(seg, pre, det_cfg, loc_cfg)
            if fb is not None:
                sa_c = fb
                cert_recovered = 1
        except Exception:
            sa_c = None
    if sa_c is not None:
        sync["CERT"] = True
        try:
            rows = FR.KT.demod_payload(sa_c["seg_a"], sa_c["pay0"],
                                       f["psym"], readout="viterbi")
            d_win, ok = FR.decode_chain(rows, f)
            hard = [(int(np.argmax(rows[k])) - d_win) % N
                    for k in range(f["psym"])]
            out["CERT×TREL-5"] = {
                "sym_err": int(sum(int(h != g) for h, g in
                                   zip(hard, f["gt"]))),
                "crc_fail": int(not ok), "den": f["psym"]}
        except Exception:
            pass

    # ---- DERA 前端 ----
    try:
        cands, seg_as, pay0s = FR.dera_sync(seg, pre)
        if cands:
            sync["DERA"] = True
            top5_rows = []
            for seg_a, pay0 in zip(seg_as, pay0s):
                if pay0 + f["psym"] + 1 > len(seg_a) // NF:
                    continue
                top5_rows.append(FR.KT.demod_payload(
                    seg_a, pay0, f["psym"], readout="viterbi"))
            if top5_rows:
                _d, ok, rows = FR.decode_chain(None, f, top5=top5_rows)
                hard = [(int(np.argmax(rows[k])) - _d) % N
                        for k in range(f["psym"])]
                out["DERA×TREL-5"] = {
                    "sym_err": int(sum(int(h != g) for h, g in
                                       zip(hard, f["gt"]))),
                    "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    return {"level": level, "seed": seed, "frame": fi,
            "sync_ours": sync["OURS"], "sync_cert": sync["CERT"],
            "sync_dera": sync["DERA"],
            "cert_recovered": cert_recovered, "chains": out}


def init_worker():
    FR.init_worker()


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
    units = []
    for lv in LEVELS:
        for sd in range(N_SEEDS if lv is not None else 1):
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
    rec_total = 0
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        counts[key] = counts.get(key, 0) + 1
        sync.setdefault(key, {"OURS": 0, "CERT": 0, "DERA": 0})
        for fr_ in ("OURS", "CERT", "DERA"):
            sync[key][fr_] += int(r["sync_" + fr_.lower()])
        rec_total += int(r["cert_recovered"])
        a = agg.setdefault(key, {c: [0, 0, 0] for c in
                                 ("OURS×TREL-5", "CERT×TREL-5",
                                  "DERA×TREL-5")})
        for c in a:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\n系统战表（TREL-5 固定；SER=同步成功单元内 / PER=含同步失败）")
    for key in ["native"] + ["%+d" % -v for v in (20, 22, 24, 26)]:
        if key not in agg:
            continue
        n = counts[key]
        parts = []
        for c in ("OURS×TREL-5", "CERT×TREL-5", "DERA×TREL-5"):
            ser = agg[key][c][0] / max(agg[key][c][1], 1)
            per = agg[key][c][2] / n
            parts.append("%.3f/%.3f" % (ser, per))
        sr = {k: round(sync[key][k] / n, 2)
              for k in ("OURS", "CERT", "DERA")}
        print("[%7s] n=%d | OURS %.3f/%.3f | CERT %.3f/%.3f | "
              "DERA %.3f/%.3f | sync %s | cert恢复 %d"
              % (key, n, parts[0][0:0] or 0, 0, 0, 0, 0, 0, sr,
                 rec_total if key == list(agg)[-1] else 0)
              if False else
              "[%7s] n=%d | OURS %s | CERT %s | DERA %s | sync %s"
              % (key, n, parts[0], parts[1], parts[2], sr), flush=True)
    print("cert 恢复单元总数：%d" % rec_total)
    print("%.0fs elapsed" % (time.time() - t0))


if __name__ == "__main__":
    main()

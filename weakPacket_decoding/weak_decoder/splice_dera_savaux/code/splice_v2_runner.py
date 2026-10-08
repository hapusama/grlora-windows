# -*- coding: utf-8 -*-
r"""2026-10-05 v2：fd × γ-链 —— 拼接线的最强形态候选。

设计依据（validate_v2/validate_v2b + 既有战表）：
  ① 混跳区 = wrap 边界的时域错位：频域 κ 网格（SAVK 整数重权 / SAVK2
     时域旋转）修不了（SAVK2 仅收窄 + max 税），fd 时延搜索不可替代；
  ② γ-链（我方解码器）在 DeRa 前端下已对 DERA 解码器平到略胜
     （gamma_link_battle_20260930：−24 SER .116 vs .150），但从未跑过
     fd 精化（fd 2026-10-04 才发明）——native 2/980 错疑为时延残差帧；
  ③ fusion 已证伪（DT-FUSE ≤ DERA 单链），不做融合臂。

臂（5）：
  DERA×GAMMA    —— gamma_a_runner 同种子复现锚（逐位对齐 2026-09-30）
  DERAfd×GAMMA  —— 主打：fd 精化（前导能量 17 点时延网格）× γ-链
  DERAfd×SAVAUX —— v1 锚（与 dera_savaux_splice_20261004 同种子配对）
  DERA×DERA     —— 全 DeRa 参照
  PRIOR×GAMMA   —— 干净 CSV 全先验天花板（oracle 标签，分报）

物理/协议与 v1 完全一致：整包加噪、无先验、δ 逐链 CRC 仲裁、种子常数
20260930。档 native + {−17,−20,−22,−24,−26}×3 种子 = 448 单元
（−28 略去：v1 全链 PER 1.0，偏差声明）。
"""
import sys
import os
import json
import time
import argparse
import numpy as np

import importlib.util as _ilu
_sr_spec = _ilu.spec_from_file_location(
    "sr", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_runner.py")
sr = _ilu.module_from_spec(_sr_spec)
_sr_spec.loader.exec_module(sr)

from weak_decoder.decoding.gamma_link import GammaLinkDemodulator

SF, N, OS, NF = sr.SF, sr.N, sr.OS, sr.NF
GAM = GammaLinkDemodulator(SF, OS)
DERA_DEC = sr.DERA_DEC

CHAINS = ["DERA×GAMMA", "DERAfd×GAMMA", "DERAfd×SAVAUX", "DERA×DERA",
          "PRIOR×GAMMA"]

EXP_DIR = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dera_fd_gamma_20261005"
CKPT = os.path.join(EXP_DIR, "checkpoint.jsonl")

LEVELS = [None, -17, -20, -22, -24, -26]
N_SEEDS = 3


def run_unit(u):
    level, seed, fi = u
    f = sr.G["frames"][fi]
    pre = f["pre"]
    lead = pre + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = sr.G["snr"][fi]
    if level is not None:
        rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)

    out = {c: {"sym_err": 0, "crc_fail": 1, "den": 0} for c in CHAINS}
    sync_dera = False
    meta = {"n_cands": 0, "fd_dstars": []}

    # ---- PRIOR×GAMMA（oracle 天花板）----
    try:
        n_rel = np.arange(len(seg))
        seg_p = seg * np.exp(-2j * np.pi * f["cfo"] * n_rel / NF)
        seg_p = sr.frac_delay(seg_p, -f["sto_frac"] * OS)
        rows = GAM.demod_payload(seg_p, lead + 8, f["psym"])
        d_win, ok = sr.decode_chain(rows, f)
        hard = [(int(np.argmax(rows[k])) - d_win) % N
                for k in range(f["psym"])]
        out["PRIOR×GAMMA"] = {
            "sym_err": int(sum(int(h != g) for h, g in zip(hard, f["gt"]))),
            "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    # ---- DeRa 前端（4 链共用）----
    try:
        cands, seg_as, pay0s = sr.dera_sync(seg, pre)
        if cands:
            sync_dera = True
            meta["n_cands"] = len(cands)
            top5 = {c: [] for c in CHAINS if c != "PRIOR×GAMMA"}
            for seg_a, pay0 in zip(seg_as, pay0s):
                if pay0 + f["psym"] + 1 > len(seg_a) // NF:
                    continue
                top5["DERA×GAMMA"].append(
                    GAM.demod_payload(seg_a, pay0, f["psym"]))
                seg_fd, d_star = sr.dera_frac_refine(seg_a, lead)
                meta["fd_dstars"].append(d_star)
                top5["DERAfd×GAMMA"].append(
                    GAM.demod_payload(seg_fd, pay0, f["psym"]))
                top5["DERAfd×SAVAUX"].append(
                    sr.sav_rows(seg_fd, pay0, f["psym"]))
                _s1, rows = DERA_DEC.demod_payload(seg_a, pay0, f["psym"])
                top5["DERA×DERA"].append(rows)
            for name, rows_list in top5.items():
                if not rows_list:
                    continue
                _d, ok, rows = sr.decode_chain(None, f, top5=rows_list)
                hard = [(int(np.argmax(rows[k])) - _d) % N
                        for k in range(f["psym"])]
                out[name] = {
                    "sym_err": int(sum(int(h != g)
                                       for h, g in zip(hard, f["gt"]))),
                    "crc_fail": int(not ok), "den": f["psym"]}
    except Exception:
        pass

    return {"level": level, "seed": seed, "frame": fi,
            "sync_dera": sync_dera, "meta": meta, "chains": out}


def init_worker():
    sr.init_worker()


def aggregate():
    agg, counts, sync = {}, {}, {}
    fdm = {}
    for line in open(CKPT, encoding="utf-8"):
        r = json.loads(line)
        key = "native" if r["level"] is None else "%+d" % r["level"]
        counts[key] = counts.get(key, 0) + 1
        sync[key] = sync.get(key, 0) + int(r["sync_dera"])
        fdm.setdefault(key, []).extend((r.get("meta") or {}).get("fd_dstars")
                                       or [])
        a = agg.setdefault(key, {c: [0, 0, 0] for c in CHAINS})
        for c in CHAINS:
            a[c][0] += r["chains"][c]["sym_err"]
            a[c][1] += r["chains"][c]["den"]
            a[c][2] += r["chains"][c]["crc_fail"]
    print("\nv2 战表（SER=同步成功单元内 / PER=含同步失败；PRIOR=oracle 分报）")
    for key in ["native"] + ["%+d" % v for v in LEVELS[1:]]:
        if key not in agg:
            continue
        n_pkt = counts[key]
        parts = " | ".join("%s %.3f/%.3f"
                           % (c, agg[key][c][0] / max(agg[key][c][1], 1),
                              agg[key][c][2] / n_pkt) for c in CHAINS)
        extra = ("  [fd d*med=%+.2f]" % float(np.median(fdm[key]))
                 if fdm.get(key) else "")
        print("[%7s] sync DERA=%.2f | %s (n=%d)%s"
              % (key, sync[key] / n_pkt, parts, n_pkt, extra), flush=True)


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
             for fi in range(28)]
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
                    el = time.time() - t0
                    print("  %d/%d (%.0fs, %.1fs/u, ETA %.0fmin)"
                          % (i + 1, len(units), el, el / (i + 1),
                             el / (i + 1) * (len(units) - i - 1) / 60.0),
                          flush=True)
    aggregate()
    print("%.0fs elapsed" % (time.time() - t0))


def smoke():
    """native 门禁 + 负控。锚点：DERA×GAMMA≈.002/.000（gamma 战表）、
    DERAfd×SAVAUX=.000/.000（v1）、DERA×DERA=.000/.000。"""
    init_worker()
    print("== native 门禁（28 帧，串行）==")
    tot = {c: [0, 0, 0] for c in CHAINS}
    n_sync = 0
    for fi in range(28):
        r = run_unit((None, 0, fi))
        n_sync += int(r["sync_dera"])
        line = []
        for c in CHAINS:
            d = r["chains"][c]
            tot[c][0] += d["sym_err"]
            tot[c][1] += d["den"]
            tot[c][2] += d["crc_fail"]
            line.append("%s %.3f/%d" % (c.split("×")[-1],
                                        d["sym_err"] / max(d["den"], 1),
                                        d["crc_fail"]))
        print("  f%02d sync=%d %s" % (fi, r["sync_dera"], " | ".join(line)),
              flush=True)
    print("sync DERA=%d/28" % n_sync)
    print("\n== native 汇总 ==")
    for c in CHAINS:
        print("%s SER=%.4f PER=%.3f" % (c, tot[c][0] / max(tot[c][1], 1),
                                        tot[c][2] / 28.0))

    print("\n== 判据负控（PRIOR×GAMMA 行）==")
    f = sr.G["frames"][0]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    n_rel = np.arange(len(seg))
    seg_p = seg * np.exp(-2j * np.pi * f["cfo"] * n_rel / NF)
    seg_p = sr.frac_delay(seg_p, -f["sto_frac"] * OS)
    rows = GAM.demod_payload(seg_p, lead + 8, f["psym"])
    d0, ok0 = sr.decode_chain(rows, f)
    rows_bad = rows.copy()
    rows_bad[3] = np.roll(rows_bad[3], 97)
    ok_bad = any(sr.judge_crc(rows_bad, d, f["gt_hdr"], f["plen"], f["cr"])
                 for d in (0, 1, -1, 2, -2))
    print("完好帧 CRC 通过=%s（δ=%d）；单符号损坏被拒=%s"
          % (ok0, d0, not ok_bad))
    if not (ok0 and not ok_bad):
        print("[FAIL] 负控未过")
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

# -*- coding: utf-8 -*-
r"""DERAfd×SAVAUX PER 失败分类学（taxonomy）runner — 2026-10-05。

对拼接战（dera_savaux_splice_20261004）DERAfd×SAVAUX 链的失败单元归类：

  cls 0 = 成功（top-5 候选(fd 精化) × δ∈{0,±1,±2} CRC 仲裁有一通过）
  cls 1 = sync-fail：dera_sync 无候选（或无可用候选/前端异常）
  cls 2 = coverage-fail 解调崩：全候选 CRC 败 且 最优单候选 raw SER > 0.3
  cls 3 = coverage-fail FEC 边缘：全候选 CRC 败 但 最优单候选 raw SER ≤ 0.3

raw SER 口径 = mode 冻结（debug_delay.py ser_diff 的推广）：
  d_k = (argmax(rows[k]) − (gt_k+1)) mod N；δ_c = mode(d)；raw SER = mean(d ≠ δ_c)。
  干净行 diff 为常数（{0:35} 或 {1023:35}），mode 即该候选固有 δ；
  raw SER 度量"逐符号混跳/散射"，与常数偏移无关。
  "最优单候选" = 逐候选冻结 δ=其行 mode 后 raw SER 最低者。

cls 2 附加归因：对 top-1 候选在 [−2,+2]×0.125 网格扫 mode 冻结 raw SER
取最优 d_opt，记录 |d*−d_opt|（fd 能量目标平坦性检验）。

噪声逐位复用原实验（种子派生常数 20260930 同源）；断点续跑
tax_checkpoint.jsonl。档 {−20,−22,−24} × seeds {0,1,2} × 28 帧 = 252 单元。
"""
import sys
import os
import json
import time
import argparse
import traceback
import importlib.util as ilu
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(os.path.dirname(_HERE), "code", "splice_runner.py")
_spec = ilu.spec_from_file_location("sr", _RUNNER)
sr = ilu.module_from_spec(_spec)
sys.modules["sr"] = sr
_spec.loader.exec_module(sr)

CKPT = os.path.join(_HERE, "tax_checkpoint.jsonl")
ORIG_CKPT = sr.CKPT            # 原实验 checkpoint（交叉验证）
LEVELS = (-20, -22, -24)
SEEDS = (0, 1, 2)
NFRAMES = 28
SER_COLLAPSE = 0.30
SCAN_GRID = np.arange(-2.0, 2.0 + 1e-9, 0.125)   # 33 点，含 0.0

N, NF = sr.N, sr.NF


def make_noisy(fi, level, seed):
    """与 run_unit 逐位一致的整包加噪段。"""
    f = sr.G["frames"][fi]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = sr.G["snr"][fi]
    rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100) * 131
                                 + seed * 17 + fi * 7919) % (2 ** 31))
    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
    seg = seg + (rng.standard_normal(len(seg))
                 + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    return f, lead, seg


def wrap_half(x):
    """把 [0,N) diff 包到 [−N/2, N/2) 便于阅读。"""
    x = int(x) % N
    return x - N if x >= N // 2 else x


def raw_ser_mode(rows, gt):
    """mode 冻结 raw SER：返回 (ser, mode, mode_frac)。"""
    psym = len(gt)
    d = np.array([(int(np.argmax(rows[k])) - (gt[k] + 1)) % N
                  for k in range(psym)])
    vals, cnts = np.unique(d, return_counts=True)
    i = int(np.argmax(cnts))
    mode = int(vals[i])
    return float(np.mean(d != mode)), mode, float(cnts[i] / psym)


def scan_dopt(seg_a, pay0, psym, gt):
    """top-1 候选时延细扫：mode 冻结 raw SER(d)，d∈[−2,+2] 步 0.125。
    tie-break: ser 最小，再取 |d| 最小。返回 (d_opt, ser_opt, curve)。"""
    best_key, best = None, None
    curve = []
    for d in SCAN_GRID:
        d = float(d)
        s = seg_a if d == 0.0 else sr.frac_delay(seg_a, d)
        rows = sr.sav_rows(s, pay0, psym)
        ser, _mode, _mf = raw_ser_mode(rows, gt)
        curve.append(round(ser, 4))
        key = (ser, abs(d))
        if best_key is None or key < best_key:
            best_key, best = key, (d, ser)
    return best[0], best[1], curve


def tax_unit(u):
    level, seed, fi = u
    t0 = time.time()
    rec = {"level": level, "seed": seed, "frame": fi}
    try:
        f, lead, seg = make_noisy(fi, level, seed)
        rec["psym"] = f["psym"]
        try:
            cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
        except Exception as e:
            rec.update(cls=1, reason="sync_exc:%s" % type(e).__name__,
                       n_cands=-1)
            rec["sec"] = round(time.time() - t0, 1)
            return rec
        rec["n_cands"] = len(cands)
        if not cands:
            rec.update(cls=1, reason="no_cands")
            rec["sec"] = round(time.time() - t0, 1)
            return rec

        # ---- 可用候选：fd 精化 + Savaux 行（与 run_unit SAVAUXFD 逐字同序）----
        usable = []      # (ci, seg_a, pay0, d_star, rows)
        for ci, (seg_a, pay0) in enumerate(zip(seg_as, pay0s)):
            if pay0 + f["psym"] + 1 > len(seg_a) // NF:
                continue
            seg_fd, d_star = sr.dera_frac_refine(seg_a, lead)
            rows = sr.sav_rows(seg_fd, pay0, f["psym"])
            usable.append((ci, seg_a, pay0, float(d_star), rows))
        if not usable:
            rec.update(cls=1, reason="no_usable_cands")
            rec["sec"] = round(time.time() - t0, 1)
            return rec

        # ---- δ 逐候选 CRC 仲裁（decode_chain top5 同序）----
        win = None
        for ci, _sa, _p0, _ds, rows in usable:
            for d_try in (0, 1, -1, 2, -2):
                if sr.judge_crc(rows, d_try, f["gt_hdr"], f["plen"], f["cr"]):
                    win = (ci, d_try)
                    break
            if win:
                break

        cand_info = []
        for ci, _sa, _p0, d_star, rows in usable:
            ser, mode, mf = raw_ser_mode(rows, f["gt"])
            cand_info.append({"ci": ci, "d_star": d_star,
                              "raw_ser": round(ser, 4),
                              "mode": wrap_half(mode),
                              "mode_frac": round(mf, 3)})
        rec["n_usable"] = len(usable)
        rec["cands"] = cand_info

        if win is not None:
            rec.update(cls=0, win_ci=win[0], win_d=win[1])
            rec["sec"] = round(time.time() - t0, 1)
            return rec

        # ---- 全候选 CRC 败：最优单候选（mode 冻结 raw SER 最低，先到先得）----
        best_ci = min(range(len(usable)),
                      key=lambda i: (cand_info[i]["raw_ser"], i))
        rec["best_ci"] = usable[best_ci][0]
        rec["best_raw_ser"] = cand_info[best_ci]["raw_ser"]

        if rec["best_raw_ser"] > SER_COLLAPSE:
            # ---- cls 2：top-1 候选事后最优时延扫描 ----
            ci0, seg_a0, pay00, d_star0, _rows0 = usable[0]
            d_opt, ser_opt, curve = scan_dopt(seg_a0, pay00, f["psym"],
                                              f["gt"])
            # d* 落 0.25 网格，曲线 0.125 网格 → 直接重算 d* 点保证一致
            s_ds = seg_a0 if d_star0 == 0.0 else sr.frac_delay(seg_a0,
                                                               d_star0)
            ser_ds, _m, _mf = raw_ser_mode(sr.sav_rows(s_ds, pay00,
                                                       f["psym"]), f["gt"])
            rec.update(cls=2, scan_ci=ci0, d_star_top1=d_star0,
                       d_opt=round(float(d_opt), 3),
                       ser_opt=round(float(ser_opt), 4),
                       ser_at_dstar=round(float(ser_ds), 4),
                       d_gap=round(abs(d_star0 - float(d_opt)), 3),
                       ser_curve=curve)
        else:
            rec.update(cls=3)
        rec["sec"] = round(time.time() - t0, 1)
        return rec
    except Exception:
        rec.update(cls=-1, reason=traceback.format_exc(limit=3))
        rec["sec"] = round(time.time() - t0, 1)
        return rec


def load_mine():
    last = {}
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            try:
                r = json.loads(line)
                last[(r["level"], r["seed"], r["frame"])] = r
            except Exception:
                pass
    return last


def load_orig():
    orig = {}
    for line in open(ORIG_CKPT, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["level"] in LEVELS:
            orig[(r["level"], r["seed"], r["frame"])] = int(
                r["chains"]["DERAfd×SAVAUX"]["crc_fail"])
    return orig


def summarize():
    mine = load_mine()
    recs = [mine[k] for k in sorted(mine)]
    print("== taxonomy units=%d ==" % len(recs), flush=True)

    orig = load_orig()
    mism = [(k, mine[k]["cls"], orig[k]) for k in mine
            if k in orig and (mine[k]["cls"] not in (0, 1, 2, 3)
                              or (mine[k]["cls"] != 0) != bool(orig[k]))]
    print("crosscheck vs orig DERAfd crc_fail: mismatches=%d / %d"
          % (len(mism), sum(1 for k in mine if k in orig)), flush=True)
    for m in mism[:10]:
        print("  MISMATCH %s cls=%s orig_fail=%s" % m, flush=True)

    print("\n== per-level taxonomy ==")
    for lv in LEVELS:
        sub = [r for r in recs if r["level"] == lv]
        c = [0, 0, 0, 0, 0]   # success, cls1, cls2, cls3, other
        for r in sub:
            c[r["cls"] if r["cls"] in (0, 1, 2, 3) else 4] += 1
        of = sum(orig.get((r["level"], r["seed"], r["frame"]), 0)
                 for r in sub)
        print("[%d] n=%d success=%d cls1(sync)=%d cls2(demod)=%d "
              "cls3(fec-edge)=%d other=%d | orig_fail=%d %s"
              % (lv, len(sub), c[0], c[1], c[2], c[3], c[4], of,
                 "OK" if of == c[1] + c[2] + c[3] + c[4] else "DIFF!"),
              flush=True)

    print("\n== cls2: |d* - d_opt| (top-1 cand) ==")
    gaps = [r for r in recs if r["cls"] == 2]
    if gaps:
        g = np.array([r["d_gap"] for r in gaps])
        print("n=%d med=%.3f max=%.3f" % (len(g), float(np.median(g)),
                                          float(g.max())), flush=True)
        vals, cnts = np.unique(g, return_counts=True)
        print("dist:", {("%.3f" % v): int(c)
                        for v, c in zip(vals, cnts)}, flush=True)
        so = np.array([r["ser_opt"] for r in gaps])
        sd = np.array([r["ser_at_dstar"] for r in gaps])
        print("ser_opt med=%.3f max=%.3f | ser@d* med=%.3f max=%.3f"
              % (float(np.median(so)), float(so.max()),
                 float(np.median(sd)), float(sd.max())), flush=True)
    else:
        print("n=0", flush=True)

    print("\n== cls3: best-cand raw SER histogram ==")
    sers = np.array([r["best_raw_ser"] for r in recs if r["cls"] == 3])
    if len(sers):
        bins = np.arange(0.0, 0.301, 0.05)
        h, _ = np.histogram(sers, bins=bins)
        for i in range(len(h)):
            print("  [%.2f,%.2f): %d %s"
                  % (bins[i], bins[i + 1], h[i], "#" * int(h[i])),
                  flush=True)
        print("  med=%.3f" % float(np.median(sers)), flush=True)
    else:
        print("n=0", flush=True)

    print("\n== -24: per-frame failures (seed list) ==")
    f24 = [r for r in recs if r["level"] == -24 and r["cls"] != 0]
    cnt = {}
    for r in f24:
        cnt.setdefault(r["frame"], []).append(
            (r["seed"], r["cls"], r.get("reason", "")))
    top = sorted(cnt.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    print("failing units=%d / 84 across %d frames"
          % (len(f24), len(cnt)), flush=True)
    for fr, lst in top[:10]:
        print("  frame %02d: n=%d %s"
              % (fr, len(lst),
                 [(s, "cls%d" % c, rsn) for s, c, rsn in sorted(lst)]),
              flush=True)
    print("\n== cls1 reasons ==")
    rs = {}
    for r in recs:
        if r["cls"] == 1:
            rs[r["reason"]] = rs.get(r["reason"], 0) + 1
    print(rs or "n=0", flush=True)


def main():
    t0 = time.time()
    done = set(load_mine())
    units = [(lv, sd, fi) for lv in LEVELS for sd in SEEDS
             for fi in range(NFRAMES) if (lv, sd, fi) not in done]
    print("resume: done=%d todo=%d" % (len(done), len(units)), flush=True)
    if not units:
        summarize()
        return

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4, initializer=sr.init_worker) as pool:
        with open(CKPT, "a", encoding="utf-8") as fh:
            for i, res in enumerate(pool.imap_unordered(tax_unit, units,
                                                        chunksize=1)):
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                if (i + 1) % 12 == 0 or i + 1 == len(units):
                    el = time.time() - t0
                    print("  %d/%d (%.0fs, %.1fs/u, ETA %.1fmin) last=%s"
                          % (i + 1, len(units), el, el / (i + 1),
                             el / (i + 1) * (len(units) - i - 1) / 60.0,
                             (res["level"], res["seed"], res["frame"],
                              res["cls"])), flush=True)
    print("ALL DONE %.0fs" % (time.time() - t0), flush=True)
    summarize()


def smoke():
    """串行单单元验证 + 与原 checkpoint 同单元对照。"""
    sr.init_worker()
    orig = load_orig()
    for u in ((-20, 0, 0), (-24, 0, 0), (-22, 1, 5)):
        t0 = time.time()
        r = tax_unit(u)
        ok = orig.get(u)
        print("unit %s -> cls=%s (%.1fs) | orig crc_fail=%s %s"
              % (u, r.get("cls"), time.time() - t0, ok,
                 json.dumps({k: v for k, v in r.items()
                             if k not in ("cands", "ser_curve")})),
              flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        main()

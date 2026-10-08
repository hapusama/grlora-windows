# -*- coding: utf-8 -*-
"""M2b 探针 2：KMIRROR 拟合 + native 冒烟 + dep2/dep3a/dep3b 同段对照。

① o_sfd 镜像律 K̄ per-capture 拟合（28 帧干净域，δ∈{0,0.082}）；
② native 冒烟：双段采集 28/28（|锚err|<0.5 bin vs 注入真模板）；
③ 同噪段三臂对照（score 同 K_c 归一 ⇒ 可直比）+ δ̂ 伪峰率 + 相干度
   （score_H1 中位 / cert 参考 = 有效相干度代理）。
→ m2b_probe2.json
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d1_core as C
import d1_battle as A
import d2_core as D
import m2_core as M
import m2b_core as W

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_CONST = 20261003
DELTAS = (0.0, 0.02, 0.082)
LEVELS = (None, -26, -30, -34)


def add_noise(seg, S, N0, level, seed, salt, di):
    rng = np.random.default_rng((SEED_CONST * 7919
                                 + (int(level) + 100) * 131
                                 + seed * 17 + salt * 7919
                                 + di * 104729) % (2 ** 31))
    p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
    return seg + ((rng.standard_normal(len(seg))
                   + 1j * rng.standard_normal(len(seg)))
                  * np.sqrt(p_add / 2.0))


def main():
    t0 = time.time()
    frames = A.build_frames()
    res = {}

    # ---- ① KMIRROR 拟合（干净注入帧，o_sfd 3 行中位）----
    kfit = {}
    for pre in (8, 16, 32):
        ks = []
        for f in frames:
            if f["pre"] != pre:
                continue
            for dl in (0.0, 0.082):
                eps = D.eps_of_delta(dl)
                inj = D.resample_sfo(
                    f["seg"], f["lead"] - (pre + 4.25) * W.NF,
                    eps) if dl else f["seg"]
                tmq = D.clean_template_q(inj, f["lead"], pre)
                o_sfd = np.median([tmq["o"][rw]
                                   for rw in (pre + 2, pre + 3, pre + 4)])
                k = o_sfd + 2.0 * tmq["nu0"]
                k = ((k + W.SFD_HALF_SPAN) % W.N) - W.SFD_HALF_SPAN
                ks.append(k)
        ks = np.array(ks)
        m = np.angle(np.mean(np.exp(1j * ks / W.N * 2 * np.pi))) \
            / (2 * np.pi) * W.N      # 圆均值（wrap 域 N）
        res["kfit_raw_P%d" % pre] = [float(x) for x in ks]
        res["kfit_P%d" % pre] = float(m)
        e = np.angle(np.exp(1j * (ks - m) / W.N * 2 * np.pi)) \
            / (2 * np.pi) * W.N
        res["kfit_std_P%d" % pre] = float(np.std(e))
        kfit[pre] = m
    print("KMIRROR 拟合：", {k: round(v, 2) for k, v in kfit.items()},
          "std:", {p: round(res["kfit_std_P%d" % p], 3) for p in kfit})

    # ---- ②③ native 冒烟（全帧）+ 同段对照（帧子集 1/3）----
    by_cap = {}
    for i, f in enumerate(frames):
        by_cap.setdefault(f["pre"], []).append(i)
    sub = set()
    for pre, lst in by_cap.items():
        sub.update(lst[::3])
    smoke = []
    cmp_rows = []
    for di, dl in enumerate(DELTAS):
        for level in LEVELS:
            for seed in (0, 1, 2):
                for fi, f in enumerate(frames):
                    if level is not None and fi not in sub:
                        continue
                    pre = f["pre"]
                    eps = D.eps_of_delta(dl)
                    inj = D.resample_sfo(
                        f["seg"], f["lead"] - (pre + 4.25) * W.NF,
                        eps) if dl else f["seg"]
                    S, N0 = A.snr_parts(inj)
                    seg = inj if level is None else add_noise(
                        inj, S, N0, level, seed, f["hs"] % 4099, di)
                    seg = seg[None, :]
                    if level is None:
                        acq = W.acquire(seg, f["lead"], pre)
                        tmq = D.clean_template_q(inj, f["lead"], pre)
                        e = acq["nu0h"][0] - (tmq["nu0"] + acq["c_e"]
                                              * tmq["delta"])
                        smoke.append(dict(frame=fi, pre=pre, delta=dl,
                                          nu_err=float(e),
                                          ok=1.0 if abs(e) < 0.5 else 0.0,
                                          acq=float(acq["acq_score"][0])))
                    s2, d2, k2, ph2, acq2 = M.score_dep2(
                        seg, f["lead"], pre)
                    sa, da, ka, bksa, acqa = W.score_dep3(
                        seg, f["lead"], pre, mode="dep3a")
                    sb, db, kb, bksb, acqb = W.score_dep3(
                        seg, f["lead"], pre, mode="dep3b")
                    cmp_rows.append(dict(
                        frame=fi, pre=pre, delta=dl, level=level, seed=seed,
                        dep2=float(s2[0]), dep3a=float(sa[0]),
                        dep3b=float(sb[0]),
                        d2=float(d2[0]), da=float(da[0]), db=float(db[0]),
                        acq=float(acqa["acq_score"][0])))
    res["smoke"] = smoke
    res["cmp"] = cmp_rows
    ok = np.array([s["ok"] for s in smoke])
    print("native 冒烟：%d/%d 采集 |err|<0.5" % (int(ok.sum()), len(ok)))
    for pre in (8, 16, 32):
        sel = [s for s in smoke if s["pre"] == pre]
        print("  P=%d: %d/%d" % (pre, int(sum(s["ok"] for s in sel)),
                                 len(sel)))

    print("\n同段 score 对照（dB，中位 [p25,p75]）+ δ̂ 伪峰率：")
    for pre in (8, 16, 32):
        for dl in DELTAS:
            for level in LEVELS[1:]:
                sel = [r for r in cmp_rows
                       if r["pre"] == pre and r["delta"] == dl
                       and r["level"] == level]
                if not sel:
                    continue
                v2 = 10 * np.log10(np.array([r["dep2"] for r in sel]))
                va = 10 * np.log10(np.array([r["dep3a"] for r in sel]))
                vb = 10 * np.log10(np.array([r["dep3b"] for r in sel]))
                p2 = np.mean([abs(r["d2"] - dl) > 0.04 for r in sel])
                pa = np.mean([abs(r["da"] - dl) > 0.04 for r in sel])
                pb = np.mean([abs(r["db"] - dl) > 0.04 for r in sel])
                print(" P%2d δ=%.3f lv=%d n=%3d | dep2 %6.2f  dep3a %6.2f"
                      " (%+.2f)  dep3b %6.2f (%+.2f) | 伪峰 %.2f→%.2f→%.2f"
                      % (pre, dl, level, len(sel), np.median(v2),
                         np.median(va), np.median(va) - np.median(v2),
                         np.median(vb), np.median(vb) - np.median(v2),
                         p2, pa, pb))
    json.dump(res, open(os.path.join(HERE, "m2b_probe2.json"), "w"),
              indent=1)
    print("%.0fs → m2b_probe2.json" % (time.time() - t0))


if __name__ == "__main__":
    main()

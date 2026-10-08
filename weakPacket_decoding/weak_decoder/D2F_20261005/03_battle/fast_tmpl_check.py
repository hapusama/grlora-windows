# -*- coding: utf-8 -*-
"""fast_tmpl 验收实验（2026-10-05）：HANDOFF §4 验收硬标准 ①②③。

  ① 统计量无损：oracle 参数（δ=GT 注入、τ̂₀=GT 码值 payload est_tau0、
     ν̂₀=dep4 锚）下，fast（修复→port 相干行）vs ref_template 完整模板
     z_full——逐符号输出 SNR 差（fast−ref，读 mode(argmax−gt−1) bin：
     去斜 STO 免疫下整数记账随修复翻转、链上 Δ0 合法吸收，对拍须读
     真实音位）。PASS 预注册（单侧）：全体帧中位 ≥ −0.1dB 且 ≥26/28
     帧的帧内中位 ≥ −0.1dB（防重复补偿/结构损失）；正方向 = port ML
     φ̂₀ 自适应胜开环模板（行1 定律 0.8~3.2dB），单列不改判。
  ② ν̂₀ 约定：dep4 锚直旋（修复后不居中不补偿走动）native 解码
     SER=0（0.146bin 生死线，F5②）。
  ③ 盲 τ̂₀ 进估计环：前导盲估（code0 公知，无 GT）vs oracle 分布 +
     盲 τ̂ native 解码（Fresnel 旁瓣命中率，F5①）。

场景：SF10 OTA 28 帧 × δ=0.02 注入（sfo_sto 同款构造/历元）+ δ=0
零回归 sanity。单进程（28 帧轻量）。
→ 04_results/fast_tmpl_check.jsonl
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\weak_decoder\D2F_20261005")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\weak_decoder\D2F_20261005\01_core")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\dera_front_battle_20260930")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\e2e_final_20261004")

import d2_core as D2                          # noqa: E402
import m3p_core as M3P                        # noqa: E402
import front_runner as FR                     # noqa: E402
import ref_template as RT                     # noqa: E402
import fast_tmpl_core as FT                   # noqa: E402
from weak_decoder.baselines.dera.paper_dera_demod import DeRaDemodulator

NF, N, OS = 4096, 1024, 4
DELTA = 0.02
DD = DeRaDemodulator(10, 4)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "04_results", "fast_tmpl_check.jsonl")


def main():
    t0 = time.time()
    frames = FR.build_frames()
    out_fh = open(OUT, "a", encoding="utf-8")
    for fi, f in enumerate(frames):
        pre, psym = f["pre"], f["psym"]
        lead = pre + 6
        tail = (8 + psym + 2) * NF + 64
        seg0 = np.asarray(f["iq"][f["hs"] - lead * NF: f["hs"] + tail],
                          dtype=np.complex128)
        origin = (lead - (pre + 4.25)) * NF
        hs_abs = lead * NF
        m0 = (lead + 8) * NF
        gt1 = np.asarray(f["gt"]) + 1
        eps = D2.eps_of_delta(DELTA)
        seg_inj = D2.resample_sfo(seg0, origin, eps)
        S, n0 = FR.snr_parts(seg_inj[: lead * NF + 8 * NF])

        det_c = M3P.dep4_blind_detect(seg_inj, pre)
        rec = dict(frame=fi, pre=pre, delta=DELTA,
                   det_ok=det_c is not None)
        if det_c is None:
            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            print("f%02d dep4 miss" % fi, flush=True)
            continue
        nu0 = float(det_c["nu0h"])
        dhat_c = float(det_c["dhat"])
        rec.update(nu0=nu0, dhat=dhat_c)

        # oracle τ̂₀（GT 码值 payload est，注入段上）+ native 历史口径
        tau0_or = RT.est_tau0(seg_inj, m0, gt1, nu0, eps, origin, None)
        # ① 统计量无损（同参数对拍）
        g = FT.stat_gap_oracle(seg_inj, f, m0, nu0, DELTA, origin,
                               tau0_or, DD, n0)
        gap_med = float(np.median(g["gap"]))
        # ③ 前导盲 τ̂₀
        tau_pre = FT.blind_tau0_pre(seg_inj, hs_abs, pre, nu0, DELTA,
                                    origin)
        # ②③ native 解码：oracle / blind 两型（A 列，port 结构不动）
        r_or = FT.fast_arm(seg_inj, f, hs_abs, m0, nu0, dhat_c, origin,
                           DD, ladder=(DELTA,), tau_mode="oracle",
                           tau0_oracle=tau0_or)
        r_bl = FT.fast_arm(seg_inj, f, hs_abs, m0, nu0, dhat_c, origin,
                           DD, tau_mode="blind")
        # δ=0 零回归 sanity（native 段，梯全开）
        det0 = M3P.dep4_blind_detect(seg0, pre)
        r_d0 = None
        if det0 is not None:
            r_d0 = FT.fast_arm(seg0, f, hs_abs, m0, float(det0["nu0h"]),
                               float(det0["dhat"]), origin, DD,
                               tau_mode="blind")
        rec.update(
            tau0_oracle=float(tau0_or), tau_pre=float(tau_pre),
            tau_err=float(tau_pre - tau0_or),
            gap_med=gap_med,
            gap_p10=float(np.percentile(g["gap"], 10)),
            gap_p90=float(np.percentile(g["gap"], 90)),
            mode_off=int(g["mode_off"]),
            snr_fast_med=float(np.median(g["snr_fast"])),
            snr_ref_med=float(np.median(g["snr_ref"])),
            e_or=dict(ok=r_or["ok"], ser=r_or["ser"], dc=r_or["delta_c"],
                      tau=float(r_or["tau"])),
            e_bl=dict(ok=r_bl["ok"], ser=r_bl["ser"], dc=r_bl["delta_c"],
                      tau=float(r_bl["tau"])),
            e_d0=(dict(ok=r_d0["ok"], ser=r_d0["ser"], dc=r_d0["delta_c"])
                  if r_d0 else None),
        )
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()
        print("f%02d gap_med %+.3f tau_err %+.2f e_or %d e_bl %d e_d0 %s"
              % (fi, gap_med, tau_pre - tau0_or, r_or["ok"], r_bl["ok"],
                 r_d0["ok"] if r_d0 else -1), flush=True)
    out_fh.close()

    # ---- 汇总 ----
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")
            if l.strip()]
    rows = [r for r in rows if r.get("delta") == DELTA]
    full = [r for r in rows if "gap_med" in r]
    print("\n=== 验收① 统计量无损（%d 帧，δ=0.02 oracle 参数）===" % len(full))
    meds = np.array([r["gap_med"] for r in full])
    print("帧中位差的分布：med %+.3f IQR [%+.3f,%+.3f] min %+.3f max %+.3f dB"
          % (np.median(meds), np.percentile(meds, 25),
             np.percentile(meds, 75), meds.min(), meds.max()))
    n_in = int(np.sum(meds >= -0.1))
    print("帧内中位 ≥ −0.1dB：%d/%d  → %s"
          % (n_in, len(meds), "PASS" if n_in >= 26 else "FAIL"))
    print("mode(argmax−gt−1) 分布：%s" % sorted(
        __import__("collections").Counter(
            r.get("mode_off") for r in full).items()))
    print("\n=== 验收③ 盲 τ̂₀（前导 code0 vs GT payload oracle）===")
    terr = np.array([r["tau_err"] for r in full])
    print("|Δτ| med %.2f p90 %.2f max %.2f 采样；|Δτ|>1 采样帧数 %d"
          % (np.median(np.abs(terr)), np.percentile(np.abs(terr), 90),
             np.abs(terr).max(), int(np.sum(np.abs(terr) > 1.0))))
    print("oracle τ̂₀ 范围 [%.2f, %.2f]" % (min(r["tau0_oracle"] for r in full),
                                           max(r["tau0_oracle"] for r in full)))
    print("\n=== 验收②③ native 解码（A 列 port，结构不动）===")
    ok_or = sum(r["e_or"]["ok"] for r in full)
    ok_bl = sum(r["e_bl"]["ok"] for r in full)
    d0 = [r for r in rows if r.get("e_d0")]
    ok_d0 = sum(r["e_d0"]["ok"] for r in d0)
    print("δ=0.02 oracle τ̂：%d/%d   盲 τ̂：%d/%d   δ=0 盲梯：%d/%d"
          % (ok_or, len(full), ok_bl, len(full), ok_d0, len(d0)))
    cas = [(r["frame"], r["tau_err"]) for r in full if not r["e_bl"]["ok"]]
    if cas:
        print("盲 τ̂ native 失败帧（Fresnel 嫌疑）：%s" % cas)
    print("\n%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()

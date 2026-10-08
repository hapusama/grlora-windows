# -*- coding: utf-8 -*-
"""D1 native 冒烟：28 帧 OTA SF10——OURS-cert 检出 28/28（门限=CFAR 闭式
@FAR=1e-3）+ DeRa 臂同窗对照 + κ̂/δ̂ 与 γ-链 preflight（δ=0.002~0.024）量级
核对 + keystone vs δ 网格在真实数据（前导+sync 上行场）一致性抽查。"""
import csv
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
                 r"\data\experiments\keystone_battle_20261003")
import d1_core as C

HF = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
      r"\header_first")
USRP = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ"
SOURCES = [(p, "%s/0_0_0_10_14_%d.bin" % (USRP, p),
            "%s/0_0_0_10_14_%d_header_first_frames.csv" % (HF, p))
           for p in (8, 16, 32)]


def main():
    t0 = time.time()
    cseg, chs = C.synth_frame(pre=8, noise=False)
    tm0 = C.clean_template(cseg, chs, 8)
    idx, bb = C.frame_slots(8)
    thr = C.cfar_threshold(1e-3, C.upcrossing_c(idx, bb))
    det_n = 0
    det_dera = 0
    rows = []
    for pre, bin_path, csv_path in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rws = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
               if r.get("header_valid") == "1"
               and int(r.get("payload_len", 0) or 0) > 0]
        for r in rws:
            hs = int(r["header_start_sample"])
            lead = (pre + 6) * C.NF
            seg = np.asarray(iq[hs - lead:hs + 8 * C.NF], dtype=np.complex128)
            # 段原点平移到 hs（field_windows 以绝对 hs 取窗——这里传入相对段：
            # 构造 seg 使 hs' = lead）
            tm = C.clean_template(seg, lead, pre)
            s_cert = C.score_ours(seg, lead, pre, tm, "cert")
            s_dep = C.score_ours(seg, lead, pre, tm, "dep")
            s_der = C.score_dera(seg, lead, pre)
            hit = s_cert["score_db"] > 10 * np.log10(thr)
            det_n += int(hit)
            det_dera += int(s_der["score_db"] > 13.0)   # port 原门限
            rows.append(dict(pre=pre, hs=hs, cert=round(s_cert["score_db"], 2),
                             dep=round(s_dep["score_db"], 2),
                             dera=round(s_der["score_db"], 2),
                             kappa_cert=round(s_cert["kappa"], 3),
                             kappa_dera=round(s_der["kappa"], 3),
                             delta=round(tm["delta"], 5),
                             conj=int(tm["conj_dn"]), hit=int(hit)))
            print("%2d %6d cert=%6.2f dep=%6.2f dera=%6.2f κc=%+6.3f "
                  "κd=%+6.3f δ̂=%+8.5f conj=%d %s"
                  % (pre, hs, s_cert["score_db"], s_dep["score_db"],
                     s_der["score_db"], s_cert["kappa"], s_der["kappa"],
                     tm["delta"], tm["conj_dn"], "HIT" if hit else "MISS"),
                  flush=True)
    D = np.abs([r["delta"] for r in rows])
    print("\n== native 冒烟 ==")
    print("OURS-cert 检出 %d/28（闭式门限 %.2f dB @FAR=1e-3）"
          % (det_n, 10 * np.log10(thr)))
    print("DeRa 臂（13dB port 门限）检出 %d/28" % det_dera)
    print("cert vs dep 差（中位）%.2f dB | cert vs dera 差（中位）%.2f dB"
          % (np.median([r["cert"] - r["dep"] for r in rows]),
             np.median([r["cert"] - r["dera"] for r in rows])))
    print("δ̂：中位 %.5f  90%% %.5f  max %.5f（γ-preflight 参考 0.002~0.024）"
          % (np.median(D), np.percentile(D, 90), D.max()))

    # ---- keystone vs 网格：真实数据上行场（前导+sync，P=8 两帧抽查）----
    iq = np.memmap(SOURCES[0][1], dtype=np.complex64, mode="r")
    rws = [r for r in csv.DictReader(open(SOURCES[0][2], encoding="utf-8"))
           if r.get("header_valid") == "1"
           and int(r.get("payload_len", 0) or 0) > 0][:2]
    for r in rws:
        hs = int(r["header_start_sample"])
        lead = 14 * C.NF
        seg = np.asarray(iq[hs - lead:hs + 8 * C.NF], dtype=np.complex128)
        tm = C.clean_template(seg, lead, 8)
        # δ 网格（全场含 SFD 镜像）vs keystone（12 整窗，1/4 弃）
        sg = C.score_grid(seg, lead, 8, tm)
        sk = C.score_keystone(seg, lead, 8, tm)
        print("keystone=网格抽查（native P=8 hs=%d）：grid=%.2f ks=%.2f "
              "Δ=%.2f dB δ̂g=%+.3f δ̂k=%+.3f"
              % (hs, sg["score_db"], sk["score_db"],
                 sk["score_db"] - sg["score_db"], sg["delta"], sk["delta"]),
              flush=True)
    json.dump(rows, open("d1_smoke_results.json", "w"), indent=1)
    print("→ d1_smoke_results.json  (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()

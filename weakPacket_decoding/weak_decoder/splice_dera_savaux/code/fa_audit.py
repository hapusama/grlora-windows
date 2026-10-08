# -*- coding: utf-8 -*-
"""空信道虚警审计（红队二轮 #5）v2：修复定义顺序与窗计数。

对 3 个 capture 的帧间静默段切与帧段同长的窗，灌全链：DeRa 检测（pre=8）
→ SAVT2 → δ CRC 仲裁。计数：总窗数 / FA-detect / FA-decode。
"""
import importlib.util as ilu
import csv
import numpy as np

sr_spec = ilu.spec_from_file_location(
    "sr", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_runner.py")
sr = ilu.module_from_spec(sr_spec)
sr_spec.loader.exec_module(sr)
sr.init_worker()
G = sr.G
NF, N, OS, SF_ = sr.NF, sr.N, sr.OS, sr.SF

_v4b_spec = ilu.spec_from_file_location(
    "v4b", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
           r"\splice_dera_savaux\code\splice_v4b_runner.py")
v4b = ilu.module_from_spec(_v4b_spec)
_v4b_spec.loader.exec_module(v4b)

TAGS = ("8", "16", "32")


def frame_for(tag):
    for f in G["frames"]:
        if f["cap"].endswith("_" + tag):
            return f
    return None


def hs_list(tag):
    path = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
            r"\weak_sync_chain\header_first\0_0_0_10_14_%s_header_first_frames.csv" % tag)
    return sorted(int(r["header_start_sample"]) for r
                  in csv.DictReader(open(path, encoding="utf-8"))
                  if r.get("header_valid") == "1")


total_det = total_dec = total_win = 0
for tag in TAGS:
    f = frame_for(tag)
    if f is None:
        continue
    iq = f["iq"]
    psym = f["psym"]
    seg_len = (f["pre"] + 6 + 8 + psym + 2) * NF
    dummy_hdr = [0] * 8
    rows = hs_list(tag)
    n_tot = len(iq)
    ends = [h + seg_len for h in rows]
    gaps = []
    if rows and rows[0] >= seg_len + 4 * NF:
        gaps.append((0, rows[0] - 4 * NF))
    for e, s_next in zip(ends, rows[1:]):
        if s_next - e >= seg_len + 8 * NF:
            gaps.append((e + 4 * NF, s_next - 4 * NF))
    if rows and n_tot - ends[-1] >= seg_len + 4 * NF:
        gaps.append((ends[-1] + 4 * NF, n_tot - 4 * NF))
    wlen = seg_len
    n_w = n_det = n_dec = 0
    for g0, g1 in gaps:
        for w0 in range(int(g0), int(g1) - wlen + 1, wlen):
            n_w += 1
            w = np.asarray(iq[w0:w0 + wlen], dtype=np.complex128)
            cands, seg_as, pay0s = sr.dera_sync(w, 8)
            if not cands:
                continue
            n_det += 1
            outs = []
            for seg_a, pay0 in zip(seg_as, pay0s):
                if pay0 + psym + 1 > len(seg_a) // NF:
                    continue
                outs.extend(v4b.savt2_outputs(seg_a, pay0, psym))
            if not outs:
                continue
            hit = False
            for rows_m in outs:
                for d in (0, 1, -1, 2, -2):
                    if sr.judge_crc(rows_m, d, dummy_hdr, f["plen"], f["cr"]):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                n_dec += 1
    per = (tag, len(gaps), n_w, n_det, n_dec)
    print("cap %s: gaps=%d 总窗=%d FA-detect=%d FA-decode=%d" % per, flush=True)
    total_win += n_w
    total_det += n_det
    total_dec += n_dec

print("\n总计: 噪声窗=%d  FA-detect=%d (%.2e/窗)  FA-decode=%d (%.2e/窗)"
      % (total_win, total_det, total_det / max(total_win, 1), total_dec,
         total_dec / max(total_win, 1)))

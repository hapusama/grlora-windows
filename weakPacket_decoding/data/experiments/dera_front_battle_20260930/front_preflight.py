# -*- coding: utf-8 -*-
"""DeRa 检测级 port v2 preflight：native 门禁 + 估计精度 + FA + 深SNR 预览。

门槛（不坑 baseline 红线）：干净 OTA 28/28 检出（locate 校验过）；
f̂ 与 CSV 先验差 ≤0.5 bin；DeRa 前端对齐下 native 解码 SER 与 exp2 同量级
（TRIMMER ≤.06、TREL ≈0）；纯噪声段零候选。
用法：D:\mysoft2\miniconda3\envs\gr-lora\python.exe front_preflight.py
"""
import importlib.util
import sys

import numpy as np

HERE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
        r"\data\experiments\dera_front_battle_20260930")
E2 = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
      r"\data\experiments\full_chain_20260929")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


e2 = _load("exp2_runner", E2 + r"\exp2_runner.py")

from weak_decoder.baselines.dera.paper_dera_detector import DeRaDetector
from weak_decoder.baselines.loratrimmer.paper_loratrimmer_demod import (
    demod_loratrimmer_symbol as trim_demod)
from weak_decoder.synchronization.frame_locator import (
    FrameLocatorConfig, locate_frame_from_event)

SF, N, OS, NF = e2.SF, e2.N, e2.OS, e2.NF
DET = DeRaDetector(SF, OS)


class _ShimEvent:
    def __init__(self, start):
        self.start_sample = int(start)
        self.event_index = 0


def make_locate(pre):
    def _locate(seg_, start, pre_):
        det_cfg_, _ = e2.det_loc_cfg(pre_)
        loc_cfg = FrameLocatorConfig(preamble_len=pre_, sync_word=0x34,
                                     min_preamble_peaks=5,
                                     search_radius_samples=320,
                                     step_samples=4, symbol_search_span=1)
        try:
            loc = locate_frame_from_event(seg_, _ShimEvent(start), det_cfg_,
                                          loc_cfg,
                                          coarse_start_sample=int(start))
        except Exception:
            return False, -1, -1
        return bool(loc.valid), int(loc.payload_start_sample), \
            int(loc.preamble_start_sample)
    return _locate


def seg_of(f):
    pre = f["pre"]
    lead = pre + 6
    return np.asarray(f["iq"][f["hs"] - lead * NF:
                             f["hs"] + (8 + f["psym"] + 2) * NF],
                      dtype=np.complex128)


def align_deRa(seg, cand):
    n = np.arange(len(seg))
    seg_a = seg * np.exp(-2j * np.pi * cand.f_bins * n / NF)
    if cand.grid_shift:
        X = np.fft.fft(seg_a)
        f = np.fft.fftfreq(len(seg_a))
        seg_a = np.fft.ifft(X * np.exp(2j * np.pi * f * cand.grid_shift))
    return seg_a


def chain_rows_light(seg_a, pay0, psym):
    tri = np.stack([trim_demod(samples=seg_a, start_sample=(pay0 + k) * NF,
                               sf=SF, os_factor=OS, cfo_int=0).metric
                    for k in range(psym)])
    trel = e2.KT.demod_payload(seg_a, pay0, psym, readout="viterbi")
    return {"TRIMMER": tri, "TREL-5": trel}


def ser_with_crc(rows, f):
    d_win = 0
    for d_try in (0, 1, -1, 2, -2):
        if e2.judge_crc(rows, d_try, f["gt_hdr"], f["plen"], f["cr"]):
            d_win = d_try
            break
    hard = [(int(np.argmax(rows[k])) - d_win) % N for k in range(rows.shape[0])]
    return sum(int(h != g) for h, g in zip(hard, f["gt"])), d_win


print("building frames ...", flush=True)
frames = e2.build_frames()
print("%d frames" % len(frames), flush=True)

# ---------------- 1) native 检测 + 估计精度 + 端到端解码 ----------------
ok_cnt = 0
f_err, k_err, hs_err = [], [], []
ser_r = ser_t = 0
for idx, f in enumerate(frames):
    pre = f["pre"]
    seg = seg_of(f)
    cands = DET.detect_frame(seg, pre, locate=make_locate(pre))
    if not cands:
        print("  frame %2d pre=%d  NO CANDIDATE" % (idx, pre))
        ser_r += f["psym"]
        ser_t += f["psym"]
        continue
    c = cands[0]
    csv_f = f["cfo"]
    f_err.append(c.f_bins - csv_f)
    k_err.append(((c.kappa - (csv_f % 1.0) + 0.5) % 1.0) - 0.5)
    hs_err.append(c.hs_est - (pre + 6) * NF)   # 段坐标，未平移
    ok_cnt += 1
    seg_a = align_deRa(seg, c)
    pay0 = c.hs_est // NF + 8
    rows = chain_rows_light(seg_a, pay0, f["psym"])
    e_r, _ = ser_with_crc(rows["TRIMMER"], f)
    e_t, _ = ser_with_crc(rows["TREL-5"], f)
    ser_r += e_r
    ser_t += e_t
    if idx < 4 or idx == 13 or idx == 27:
        print("  frame %2d pre=%2d score=%5.1fdB f̂=%+8.3f (csv %+8.3f, "
              "err %+.3f) κ̂err %+.3f hs_err %+d | nSER T %.3f R %.3f"
              % (idx, pre, c.score_db, c.f_bins, csv_f, c.f_bins - csv_f,
                 k_err[-1], hs_err[-1], e_t / f["psym"], e_r / f["psym"]))

tot = sum(f["psym"] for f in frames)
print("\nnative 检出（locate 校验过）：%d/%d" % (ok_cnt, len(frames)))
print("f̂ 误差 bins: max |err| = %.3f" % max(abs(x) for x in f_err))
print("κ̂ 误差: max |err| = %.3f" % max(abs(x) for x in k_err))
print("hs 误差样本(段坐标): %s" % sorted(set(hs_err)))
print("native 解码（DeRa 前端）：TRIMMER %d/%d=%.4f  TREL-5 %d/%d=%.4f"
      % (ser_r, tot, ser_r / tot, ser_t, tot, ser_t / tot))

# ---------------- 2) 纯噪声 FA 检查 ----------------
rng = np.random.default_rng(7)
fa = 0
for t in range(3):
    z = (rng.standard_normal(59 * NF) + 1j * rng.standard_normal(59 * NF))
    c = DET.detect_frame(z, 8, locate=None)
    fa += len(c)
print("纯噪声段候选总数：%d（应=0）" % fa)

# ---------------- 3) 深 SNR 检出预览（seed0，前 10 帧） ----------------
for lv in (-20, -22, -24, -26):
    hit = 0
    for idx, f in enumerate(frames[:10]):
        pre = f["pre"]
        lead = pre + 6
        seg = seg_of(f)
        S, N0 = e2.snr_parts(np.asarray(
            f["iq"][f["hs"] - lead * NF: f["hs"] + 8 * NF],
            dtype=np.complex128))
        rng = np.random.default_rng((20260930 * 7919 + (lv + 100) * 131
                                     + idx * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (lv / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
        cands = DET.detect_frame(seg, pre, locate=make_locate(pre))
        hit += int(bool(cands))
    print("−%ddB 检出预览：%d/10" % (lv, hit))

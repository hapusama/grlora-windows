# -*- coding: utf-8 -*-
"""m1_probe_ota.py — OTA κ 标定 + native 冒烟探针（battle 前置）。

1. 干净信号上 γ-链 κ̂（我的 WLS 复刻）vs 直接音位 frac（branch0 谱峰
   对 GT 的残余）——标定 κ 约定（γ 口径 ↔ 抽头口径）；
2. OURS-Dir native（干净，γ κ̂）逐帧 SER —— 必须 ≈0；
3. OURS-Dir native κ̂=0 对照（看 κ 条件化在 native 的作用）；
4. 判据负控：损坏一个符号的行 → judge 必须拒。
"""
import importlib.util
import sys

import numpy as np

HERE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
        r"\data\experiments\dirichlet_demod_20261004")
BATTLE = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"
          r"\data\experiments\dera_battle_20260929")
sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
sys.path.insert(0, HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


br = _load("battle_runner_m1probe", BATTLE + r"\battle_runner.py")

from m1_dirichlet import (DirichletDemodulator, DeraDirTapsDemodulator,
                          gamma_kappa_track)
from weak_decoder.decoding.kappa_trellis import KappaTrellisDemodulator

SF, N, OS, NF = br.SF, br.N, br.OS, br.NF
DIR = DirichletDemodulator(SF, OS)
DDT = DeraDirTapsDemodulator(SF, OS)
KT = KappaTrellisDemodulator(SF, OS)


def main():
    frames = br.build_frames()
    print("frames: %d" % len(frames), flush=True)
    REF_OS = br.REF_OS
    for fi in range(0, len(frames), 5):     # 抽 6 帧探针
        f = frames[fi]
        seg, psym, gt, delta = f["seg"], f["psym"], f["gt"], f["delta"]
        # γ-链 κ̂（干净）
        kap = gamma_kappa_track(seg, 16, psym, SF, OS)
        # 直接音位 frac：branch0 谱峰 − GT − δ
        fr = []
        for k in range(psym):
            w = seg[(16 + k) * NF:(17 + k) * NF]
            X0 = np.fft.fft(w[0::OS] * np.conj(REF_OS)[0::OS])
            fr.append((int(np.argmax(np.abs(X0))) - gt[k] - delta))
        fr = np.array(fr, dtype=float)
        frac_dir = np.angle(np.mean(np.exp(2j * np.pi * fr / N))) * N / (2 * np.pi)
        # OURS native：γ κ̂ vs κ̂=0
        rows_k = DIR.demod_payload(seg, 16, psym, kappa_hat=kap)
        rows_0 = DIR.demod_payload(seg, 16, psym, kappa_hat=None)
        hard_k = [(int(np.argmax(rows_k[i])) - delta) % N for i in range(psym)]
        hard_0 = [(int(np.argmax(rows_0[i])) - delta) % N for i in range(psym)]
        ser_k = np.mean([h != g for h, g in zip(hard_k, gt)])
        ser_0 = np.mean([h != g for h, g in zip(hard_0, gt)])
        # DDT native
        rows_dt = DDT.demod_payload(seg, 16, psym, kappa_hat=kap)
        hard_dt = [(int(np.argmax(rows_dt[i])) - delta) % N for i in range(psym)]
        ser_dt = np.mean([h != g for h, g in zip(hard_dt, gt)])
        # TREL native 对照
        rows_t = KT.demod_payload(seg, 16, psym, readout="viterbi")
        hard_t = [(int(np.argmax(rows_t[i])) - delta) % N for i in range(psym)]
        ser_t = np.mean([h != g for h, g in zip(hard_t, gt)])
        # 判据（γ κ̂ 版）+ 负控
        crc_ok = br.judge_crc_fast(rows_k, delta, f["gt_hdr"], f["plen"], f["cr"])
        bad = rows_k.copy()
        bad[0] = np.roll(bad[0], 7)
        crc_bad = br.judge_crc_fast(bad, delta, f["gt_hdr"], f["plen"], f["cr"])
        print("f%02d psym=%2d | γκ̂[mean]=%+.3f drift-range=%.3f | direct-frac=%+.3f"
              " | SER OURS(γκ̂)=%.4f OURS(0)=%.4f DT=%.4f TREL=%.4f | CRC %s negctl %s"
              % (fi, psym, kap.mean(), kap.max() - kap.min(), frac_dir,
                 ser_k, ser_0, ser_dt, ser_t, crc_ok, crc_bad), flush=True)


if __name__ == "__main__":
    main()

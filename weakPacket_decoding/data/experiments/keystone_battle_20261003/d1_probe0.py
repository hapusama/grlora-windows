# -*- coding: utf-8 -*-
"""D1 probe0：12.25 已知场结构探测（native OTA，SF10 28 帧）v3（干净算术）。"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")
from weak_decoder.chirp import build_upchirp

SF, N, OS, NF = 10, 1024, 4, 4096
NFFT = 2 * NF
UP = build_upchirp(SF, 0, OS).astype(np.complex64)
DOWN = np.conj(UP)
HF = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
      r"\header_first")
USRP = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ"
SOURCES = [(p, "%s/0_0_0_10_14_%d.bin" % (USRP, p),
            "%s/0_0_0_10_14_%d_header_first_frames.csv" % (HF, p))
           for p in (8, 16, 32)]
N_FINE = 256


def signed_spec(iq, start, ref, length=NF):
    w = np.asarray(iq[start:start + length], dtype=np.complex64)
    X = np.fft.fft(w * ref[:length], NFFT)
    return np.concatenate((X[NFFT - N:], X[:N]))


def interp_peak(mag, idx):
    if idx <= 0 or idx >= len(mag) - 1:
        return float(idx)
    a, b, c = mag[idx - 1], mag[idx], mag[idx + 1]
    d = a - 2 * b + c
    return idx + 0.5 * (a - c) / d if d != 0 else float(idx)


def dtft_at(row_window, ref, p_bin, length):
    """连续位置 p_bin（bin 域）的精确 DTFT：⟨y·ref, e^{-j2πpn/NF}⟩。"""
    n = np.arange(length)
    return np.dot(np.asarray(row_window, dtype=np.complex128) * ref[:length],
                  np.exp(-2j * np.pi * p_bin * n / NF))


def field_windows(hs, pre):
    base = hs - int((pre + 4.25) * NF)
    out = []
    for j in range(pre + 4):
        ref = DOWN if j < pre + 2 else UP
        out.append((base + j * NF, ref, NF))
    out.append((base + (pre + 4) * NF, UP, NF // 4))
    return out


def main():
    recs = []
    print("frame   P   nu0      o_s0   o_s1   o_d0   o_d1   o_dq   conj  "
          "delta_hat   kappa  cohM(dB) [theta_type deg: s0 s1 d0 d1 dq]")
    for pre, bin_path, csv_path in SOURCES:
        iq = np.memmap(bin_path, dtype=np.complex64, mode="r")
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if r.get("header_valid") == "1"
                and int(r.get("payload_len", 0) or 0) > 0]
        for fi, r in enumerate(rows):
            hs = int(r["header_start_sample"])
            wins = field_windows(hs, pre)
            pos = []
            for s, ref, ln in wins:
                X = signed_spec(iq, s, ref, ln)
                m = np.abs(X)
                pos.append((interp_peak(m, int(np.argmax(m))) - N) / 2.0)
            nu0 = float(np.median(pos[:pre]))
            offs = [pos[j] - nu0 for j in range(pre, pre + 5)]
            # δ̂：pre+sync 分数音位（去整偏移）线性拟合
            yy = [pos[j] - nu0 - (0 if j < pre else round(offs[j - pre]))
                  for j in range(pre + 2)]
            delta = float(np.polyfit(np.arange(pre + 2, dtype=float), yy, 1)[0])
            # 相位与类型残差：连续位置 DTFT（整窗 exact）
            idx = np.arange(pre + 5, dtype=float)
            idx[-1] = pre + 4.25

            def phases(conj_dn):
                ph = []
                for j in range(pre + 5):
                    pj = nu0 + (offs[j - pre] if j >= pre else 0.0) \
                        + idx[j] * delta
                    v = dtft_at(iq[wins[j][0]:wins[j][0] + wins[j][2]],
                                wins[j][1], pj, wins[j][2])
                    if conj_dn and j >= pre + 2:
                        v = np.conj(v)
                    ph.append(np.angle(v))
                ph = np.unwrap(np.array(ph))
                sl, ic = np.polyfit(idx, ph, 1)
                res = ph - (sl * idx + ic)
                kap = sl / (2 * np.pi)
                kap = kap - round(kap)
                coh = float(np.abs(np.fft.fft(np.exp(1j * res), N_FINE)).max())
                return kap, res, coh
            kc, rc, cc = phases(True)
            kn, rn, cn = phases(False)
            conj = cc > cn
            kap, res, coh = (kc, rc, cc) if conj else (kn, rn, cn)
            th = np.degrees(res[pre:pre + 5])
            print("%4s/%02d %2d %+8.3f  %+6.2f %+6.2f %+6.2f %+6.2f %+6.2f "
                  "  %d   %+8.5f %+7.4f  %5.1f  [%6.1f %6.1f %6.1f %6.1f "
                  "%6.1f]" % (pre, fi, pre, nu0, *offs, conj, delta, kap,
                              10 * np.log10(max(cc, cn) / min(cc, cn)), *th))
            recs.append(dict(pre=pre, fi=fi, nu0=nu0, offs=offs, conj=conj,
                             delta=delta, kappa=kap, theta=th.tolist()))
    O = np.array([r["offs"] for r in recs])
    TH = np.array([r["theta"] for r in recs])
    D = np.abs(np.array([r["delta"] for r in recs]))
    print("\n== 汇总 ==")
    print("offs (s0,s1,d0,d1,dq)：mean", np.round(O.mean(0), 3),
          "\n                    std", np.round(O.std(0), 4))
    print("conj 帧占比：%.0f%%" % (100 * np.mean([r["conj"] for r in recs])))
    print("θ_type 跨帧 std(deg)：", np.round(TH.std(0), 1))
    print("δ̂（bin/符）：中位 %.5f  90%% %.5f  max %.5f | γ-preflight 参考 "
          "0.002~0.024" % (np.median(D), np.percentile(D, 90), D.max()))
    json.dump(recs, open("d1_probe0_results.json", "w"), indent=1)
    print("→ d1_probe0_results.json")


if __name__ == "__main__":
    main()

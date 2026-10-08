# -*- coding: utf-8 -*-
"""A3 主核查 v2（修正版）：
1) 前导码相干平均谱形 → 幅度域单径 Dirichlet 拟合（多径检验）+ 3ν/5ν 谐波检验（量化纹理）；
2) 前导码逐 upchirp 分数 ν_k → 帧内漂移 δ（bins/符，无需 GT）+ 幅度 CV（衰落恒定性）；
3) payload 峰 SNR 帧首 vs 帧中（AGC/滤波瞬态检验）；
4) dechirp 域逐 bin 噪声底 σ_b 谱形（边缘滚降检验 → 1/σ² 加权合法性）；
5) 前导相位残差（去线性 CFO 后）= 相位游走。
"""
import numpy as np
import csv
import os

HF = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
      r"\header_first")
USRP = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ"


def upchirp(N):
    n = np.arange(N)
    return np.exp(2j * np.pi * n ** 2 / (2 * N))


def dq(nu, N):
    """幅度域 Dirichlet 核 |D| 于偏移 nu（矢量）。"""
    num = np.sin(np.pi * nu)
    den = np.sin(np.pi * nu / N)
    return np.abs(np.where(np.abs(den) < 1e-14, 1e-14, num) /
                  np.where(np.abs(den) < 1e-14, 1e-14, den))


def analyze(tag, bp, cp, SF, P, limit=99, verbose=3):
    N = 1 << SF
    NF = 4 * N
    OS = 4
    ref = np.conj(upchirp(N))
    iq = np.memmap(bp, dtype=np.complex64, mode="r")
    rows = [r for r in csv.DictReader(open(cp, encoding="utf-8"))
            if r.get("header_valid") == "1"][:limit]
    print("== %s  SF=%d pre=%d  帧数=%d" % (tag, SF, P, len(rows)))
    out = []
    for fi, r in enumerate(rows):
        hs = int(r["header_start_sample"])
        cfo = int(r["source_grlora_cfo_int"]) + float(r["source_grlora_cfo_frac"])
        starts = [hs - int((4.25 + P - k) * NF) + NF // 4 for k in range(P)]
        if starts[0] < 0:
            continue

        def spec(st):
            seg = np.asarray(iq[st:st + NF], dtype=np.complex128)
            seg = seg * np.exp(-2j * np.pi * cfo * np.arange(NF) / NF)
            return np.fft.fft(seg[::OS][:N] * ref)

        S = np.stack([spec(st) for st in starts])      # P × N
        pkbin = int(np.bincount(np.argmax(np.abs(S), axis=1)).argmax())
        # —— 谱形：幅度域 ——
        align = S / (S[:, pkbin] / np.abs(S[:, pkbin]))[:, None]
        B = np.abs(align.mean(axis=0))                  # 相干平均幅度谱
        d = (np.arange(N) - pkbin + N // 2) % N - N // 2
        sel = np.abs(d) <= 16
        dd = d[sel].astype(float)
        best = None
        for kap in np.linspace(-0.5, 0.5, 101):
            g = dq(kap - dd, N)
            g /= np.linalg.norm(g)
            c = float(np.dot(B[sel], g))
            res = float(np.sum((B[sel] - c * g) ** 2))
            if best is None or res < best[0]:
                best = (res, kap, c)
        _, kap, c = best
        nu = pkbin + kap
        kern = dq(kap - dd, N)
        g1 = c * kern / np.linalg.norm(kern)
        excess = float(np.sum(np.maximum(B[sel] - g1, 0)) / np.sum(g1))
        fiterr = float(np.sum((B[sel] - g1) ** 2) / np.sum(g1 ** 2))
        # 谐波检验：3ν / 5ν 位置（mod N）
        harm = []
        for mult in (3, 5):
            hb = (pkbin * mult) % N
            # 修正：谐波出现在 3ν 处, ν=nu → bin = round(3nu) mod N
            hb = int(round(nu * mult)) % N
            harm.append(float(B[hb] / B[pkbin]))
        # —— 前导分数 ν_k 漂移（抛物线）——
        nuk = []
        for k in range(P):
            b = int(np.argmax(np.abs(S[k])))
            X = S[k]
            l, m_, rr = np.abs(X[(b - 1) % N]), np.abs(X[b]), np.abs(X[(b + 1) % N])
            off = 0.5 * (l - rr) / (l - 2 * m_ + rr + 1e-30)
            nuk.append(b + np.clip(off, -0.5, 0.5))
        nuk = np.array(nuk)
        w = np.polyfit(np.arange(P), nuk, 1)
        drift = float(w[0])
        dres = float(np.std(nuk - np.polyval(w, np.arange(P))))
        # —— 幅度 CV（衰落恒定；κ 相同 → 无 scalloping 混淆）——
        amps = np.abs(S[:, pkbin])
        cv_amp = float(amps.std() / amps.mean())
        # 相位游走（去线性）
        ph = np.unwrap(np.angle(S[:, pkbin]))
        wph = np.polyfit(np.arange(P), ph, 1)
        phres = float(np.std(ph - np.polyval(wph, np.arange(P))))
        # —— payload 峰SNR 帧首/中段 + 逐bin噪声底 ——
        psym = int(r["payload_symbol_count"])
        snrk, noise_bins = [], []
        for k in range(psym):
            st = hs + k * NF
            seg = np.asarray(iq[st:st + NF], dtype=np.complex128)
            seg = seg * np.exp(-2j * np.pi * cfo * np.arange(NF) / NF)
            X = np.fft.fft(seg[::OS][:N] * ref)
            p2 = np.abs(X) ** 2
            b = int(np.argmax(p2))
            snrk.append(p2[b] / np.median(p2))
            noise_bins.append(p2)
        snrk = np.array(snrk)
        NB = np.stack(noise_bins)
        prof = np.median(NB, axis=0)                   # 逐 bin 噪声底（payload 中位）
        # 真峰剔除后重算中位（去符号峰污染）：按列去掉 top 10%
        prof2 = np.median(np.sort(NB, axis=0)[: max(int(psym * 0.9), 1)], axis=0)
        pk_pos = np.argmax(prof2)
        # 噪声谱形：以峰为中心的相对轮廓（±200 bin）
        off = (np.arange(N) - pk_pos + N // 2) % N - N // 2
        core = np.abs(off) <= 200
        rel = prof2[core] / np.median(prof2[core])
        edge_db = 10 * np.log10(np.mean(prof2[np.abs(off) >= 460]) /
                                np.median(prof2[core])) if (np.abs(off) >= 460).any() else float("nan")
        n13 = max(len(snrk) // 3, 1)
        rec = dict(tag=tag, excess=excess, fiterr=fiterr, nu=nu - pkbin,
                   h3=harm[0], h5=harm[1], drift=drift, dres=dres,
                   cv_amp=cv_amp, phres=phres,
                   head=float(snrk[:n13].mean()), mid=float(snrk[n13:2 * n13].mean()),
                   edge_db=edge_db, profile=rel, offsets=off[core])
        out.append(rec)
        if fi < verbose:
            print("  f%02d κ0=%+.3f 谱形excess=%.3f fiterr=%.2e h3=%.3f h5=%.3f "
                  "δ=%+.4f dσ=%.4f 幅CV=%.3f phw=%.3f 首SNR%.0f 中%.0f 边沿%.1f dB"
                  % (fi, nu - pkbin, excess, fiterr, harm[0], harm[1],
                     drift, dres, cv_amp, phres, rec["head"], rec["mid"], edge_db))
    del iq
    return out


jobs = [("SF10_8", os.path.join(USRP, "0_0_0_10_14_8.bin"),
         os.path.join(HF, "0_0_0_10_14_8_header_first_frames.csv"), 10, 8, 99),
        ("SF10_16", os.path.join(USRP, "0_0_0_10_14_16.bin"),
         os.path.join(HF, "0_0_0_10_14_16_header_first_frames.csv"), 10, 16, 99),
        ("SF10_32", os.path.join(USRP, "0_0_0_10_14_32.bin"),
         os.path.join(HF, "0_0_0_10_14_32_header_first_frames.csv"), 10, 32, 99),
        ("SF11_1_0_8", os.path.join(USRP, "lab1_sf11_TP2", "1_0_8_11_2_16.bin"),
         os.path.join(HF, "1_0_8_11_2_16_header_first_frames.csv"), 11, 16, 8),
        ("SF11_1_1_3", os.path.join(USRP, "lab1_sf11_TP2", "1_1_3_11_2_16.bin"),
         os.path.join(HF, "1_1_3_11_2_16_header_first_frames.csv"), 11, 16, 8)]

allr = []
for tag, bp, cp, sf, p, lim in jobs:
    allr += analyze(tag, bp, cp, sf, p, lim)

print("\n===== 汇总（%d 帧）=====" % len(allr))
for key, name in (("excess", "单径excess(幅度域)"), ("fiterr", "Dirichlet拟合err"),
                  ("h3", "3ν谐波/主峰"), ("h5", "5ν谐波/主峰"),
                  ("drift", "δ[bin/符](前导)"), ("dres", "ν残差σ[bin]"),
                  ("cv_amp", "前导幅度CV"), ("phres", "相位游走σ[rad]"),
                  ("head", "帧首SNR"), ("mid", "中段SNR"), ("edge_db", "噪声边沿dB")):
    v = np.array([r[key] for r in allr if r[key] == r[key]])
    if len(v):
        print("%-16s: 中位 %+.4f  P10 %+.4f  P90 %+.4f" %
              (name, np.median(v), np.percentile(v, 10), np.percentile(v, 90)))

# 噪声谱形平均轮廓（每帧 align 到自身峰, 中位合成）
ref_prof = np.median(np.stack([r["profile"] for r in allr[:40]]), axis=0)
offs = allr[0]["offsets"]
print("\n噪声谱形（峰对齐, 相对中心中位, dB）:")
marks = [-200, -150, -100, -50, 0, 50, 100, 150, 200]
print("  " + "  ".join("%+d:%+.1f" % (o, 10 * np.log10(ref_prof[np.argmin(np.abs(offs - o))]))
                        for o in marks))

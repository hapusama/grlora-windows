# -*- coding: utf-8 -*-
"""A3 主核查：前导码相干谱形（多径/相位噪声）+ 帧内 κ 漂移 + 衰落恒定 + 帧首质量。

方法（全只读）：
- 用 CSV 的 header_start_sample 定位帧；前导码上行 chirp 窗取在
  hs-(4.25+P-k)NF+NF/4 处（窗整段落在一个 upchirp 内）。
- 每个 upchirp：抽 4 相位之一（N 点，段内连续），乘 conj(upchirp) dechirp，
  N 点 FFT。所有前导码符号同值 → 峰 bin 相同 → 相干叠加得 P 倍增益谱形。
- 单径 Dirichlet 最小二乘拟合 ±16 bin；报告残差与第二径能量上限。
- payload 符号：argmax + 抛物线分数估计 κ_k → 线性漂移 δ̂、残差 σ；
  峰功率 P_k → 帧内起伏；帧首 vs 帧中质量。
"""
import numpy as np
import csv
import os

HF = (r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain"
      r"\header_first")
USRP = r"D:\Desktop\proj\gr-lora_sdr\data\USRP_IQ"


def upchirp(N):
    n = np.arange(N)
    return np.exp(2j * np.pi * (n ** 2 + n) / (2 * N) - 1j * np.pi * n / N)


def analyze(tag, bp, cp, SF, P, files_limit=99):
    N = 1 << SF
    NF = 4 * N
    OS = 4
    ref = np.conj(upchirp(N))
    iq = np.memmap(bp, dtype=np.complex64, mode="r")
    rows = [r for r in csv.DictReader(open(cp, encoding="utf-8"))
            if r.get("header_valid") == "1"][:files_limit]
    print("== %s  SF=%d pre=%d  帧数=%d" % (tag, SF, P, len(rows)))
    for fi, r in enumerate(rows):
        hs = int(r["header_start_sample"])
        cfo = int(r["source_grlora_cfo_int"]) + float(r["source_grlora_cfo_frac"])
        # CFO 已由 battle_runner 惯例补偿（Hz，500k 采样）
        # —— 前导码相干叠加 ——
        starts = [hs - int((4.25 + P - k) * NF) + NF // 4 for k in range(P)]
        if starts[0] < 0:
            continue
        spec, amps, phs = [], [], []
        for st in starts:
            seg = np.asarray(iq[st:st + NF], dtype=np.complex128)
            n = np.arange(NF)
            seg = seg * np.exp(-2j * np.pi * cfo * n / NF)
            y = seg[::OS][:N] * ref
            X = np.fft.fft(y)
            spec.append(X)
            amps.append(np.abs(X).max())
        S = np.stack(spec)                       # P × N
        pkbin = int(np.bincount(np.argmax(np.abs(S), axis=1)).argmax())
        # 相位对齐后相干叠加（每行除以自己峰的相位 → 消除 CFO 线性相位）
        align = S / (S[:, pkbin] / np.abs(S[:, pkbin]))[:, None]
        A = np.abs(align.mean(axis=0)) ** 2
        # 单径 Dirichlet 拟合（±16 bin 邻域, 幅度域闭式核）
        j = np.arange(N)
        d = (j - pkbin + N // 2) % N - N // 2   # 相对峰的带符号 bin 距
        sel = np.abs(d) <= 16
        dd = d[sel]

        def dirich(ν):
            # |X[j0+m]| ∝ |sin(π(ν-m)) / sin(π(ν-m)/N)| (N 偶极归一)
            m = dd - (pkbin - pkbin)            # m 即 dd
            num = np.sin(np.pi * (ν - m))
            den = np.sin(np.pi * (ν - m) / N)
            return np.abs(num / np.where(np.abs(den) < 1e-12, 1e-12, den))

        best = None
        for ν in np.linspace(pkbin - 0.5, pkbin + 0.5, 81):
            g = dirich(ν)
            c = float(np.dot(A[sel], g) / np.dot(g, g))
            res = float(np.sum((A[sel] - c * g) ** 2))
            if best is None or res < best[0]:
                best = (res, ν, c)
        res1, nu, c1 = best
        g1 = c1 * dirich(nu)
        # 残差能量比（±16 内超出单径的部分）
        excess = float(np.sum(np.maximum(A[sel] - g1, 0)) / np.sum(g1))
        # 相干增益检验：|sum| vs Σ|.|（相位漂移/相位噪声会降低相干性）
        coh = float(np.abs(align.mean(axis=0))[pkbin] / np.mean(np.abs(S[:, pkbin])))
        # 相位残差（去线性 CFO 相位后）
        ph = np.angle(S[:, pkbin])
        t = np.arange(P)
        w = np.polyfit(t, np.unwrap(ph), 1)
        phres = np.std(np.unwrap(ph) - np.polyval(w, t))
        # —— payload 符号分析 ——
        psym = int(r["payload_symbol_count"])
        kap, pkp, snrk = [], [], []
        allpk = []
        for k in range(-8, psym):
            st = hs + k * NF
            if st < 0 or st + NF > len(iq):
                continue
            seg = np.asarray(iq[st:st + NF], dtype=np.complex128)
            n = np.arange(NF)
            seg = seg * np.exp(-2j * np.pi * cfo * n / NF)
            y = seg[::OS][:N] * ref
            X = np.fft.fft(y)
            p2 = np.abs(X) ** 2
            b = int(np.argmax(p2))
            allpk.append((k, p2[b]))
            if k >= 0:
                kap.append(b)
                pkp.append(p2[b])
                med = np.median(p2)
                snrk.append(p2[b] / med)
        kap = np.array(kap, float)
        pkp = np.array(pkp)
        snrk = np.array(snrk)
        # 抛物线分数偏移
        kap_frac = []
        Xb = []
        for k in range(len(kap)):
            st = hs + k * NF
            seg = np.asarray(iq[st:st + NF], dtype=np.complex128)
            seg = seg * np.exp(-2j * np.pi * cfo * np.arange(NF) / NF)
            y = seg[::OS][:N] * ref
            X = np.fft.fft(y)
            b = int(kap[k])
            l, m_, rr = np.abs(X[(b - 1) % N]), np.abs(X[b]), np.abs(X[(b + 1) % N])
            off = 0.5 * (l - rr) / (l - 2 * m_ + rr + 1e-30)
            kap_frac.append(b + np.clip(off, -0.5, 0.5))
            Xb.append(X)
        kap_frac = np.array(kap_frac)
        t2 = np.arange(len(kap_frac))
        w2 = np.polyfit(t2, kap_frac, 1)
        resid = kap_frac - np.polyval(w2, t2)
        drift = w2[0]
        cv = float(pkp.std() / pkp.mean())
        # 帧首(前1/3) vs 中段 质量
        n13 = max(len(snrk) // 3, 1)
        head_snr, mid_snr = snrk[:n13].mean(), snrk[n13:2 * n13].mean()
        if fi < 6 or fi == len(rows) - 1:
            print("  f%02d: κ峰bin=%3d 拟合ν=%+.3f 单径残差能量=%.2e 相干度=%.3f "
                  "phres=%.3f rad" % (fi, pkbin, nu - pkbin, excess, coh, phres))
            print("       δ̂=%+.5f bin/符 残差σ=%.4f bin  峰功率CV=%.3f  帧首SNR%.1f 中段%.1f"
                  % (drift, resid.std(), cv, head_snr, mid_snr))
        yield dict(tag=tag, excess=excess, coh=coh, phres=phres, drift=drift,
                   rstd=float(resid.std()), cv=cv, head=head_snr, mid=mid_snr,
                   kappa0=float(kap_frac[0]) if len(kap_frac) else np.nan)


jobs = [("SF10_8", os.path.join(USRP, "0_0_0_10_14_8.bin"),
         os.path.join(HF, "0_0_0_10_14_8_header_first_frames.csv"), 10, 8),
        ("SF10_16", os.path.join(USRP, "0_0_0_10_14_16.bin"),
         os.path.join(HF, "0_0_0_10_14_16_header_first_frames.csv"), 10, 16),
        ("SF10_32", os.path.join(USRP, "0_0_0_10_14_32.bin"),
         os.path.join(HF, "0_0_0_10_14_32_header_first_frames.csv"), 10, 32),
        ("SF11_1_0_8", os.path.join(USRP, "lab1_sf11_TP2", "1_0_8_11_2_16.bin"),
         os.path.join(HF, "1_0_8_11_2_16_header_first_frames.csv"), 11, 16, 6),
        ("SF11_1_1_3", os.path.join(USRP, "lab1_sf11_TP2", "1_1_3_11_2_16.bin"),
         os.path.join(HF, "1_1_3_11_2_16_header_first_frames.csv"), 11, 16, 6)]

allr = []
for jj in jobs:
    for rec in analyze(*jj):
        allr.append(rec)

print("\n===== 汇总（%d 帧）=====" % len(allr))
for key, name in (("excess", "单径残差能量比"), ("coh", "前导相干度"),
                  ("phres", "相位残差σ[rad]"), ("drift", "δ̂[bin/符]"),
                  ("rstd", "κ残差σ[bin]"), ("cv", "峰功率CV"),
                  ("head", "帧首SNR"), ("mid", "中段SNR")):
    v = np.array([r[key] for r in allr if r[key] == r[key]])
    print("%-14s: 中位 %+.4f  P10 %+.4f  P90 %+.4f" %
          (name, np.median(v), np.percentile(v, 10), np.percentile(v, 90)))

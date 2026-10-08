# -*- coding: utf-8 -*-
"""v3 验证：fd 判据升级 —— 前导相干模板分数 vs v1 能量分数。

v1（incumbent）：SFD 段 4 符号 trim 能量求和（native 完美，深端选错平台）。
v3（候选）：真前导 upchirp 最后 8 符号 × Savaux 合并谱，非相干累积定公共
bin ĉ，再对复数序列 {X_p[ĉ]} 做零填 FFT 取 max|·|²（跨符号相干积累，
自动吸收线性相位漂移）。

判据测试集：native 28 帧（不得回归）+ 分类学 14 个 fd 选点失误单元
（目标：d* 落到 d_opt 平台，raw SER 显著下降）。
"""
import importlib.util as ilu
import json
import numpy as np

sr_spec = ilu.spec_from_file_location(
    "sr", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_runner.py")
sr = ilu.module_from_spec(sr_spec)
sr_spec.loader.exec_module(sr)
sr.init_worker()
G = sr.G
NF, N, OS, SF_ = sr.NF, sr.N, sr.OS, sr.SF

_GRID = np.arange(-2.0, 2.01, 0.25)


def fd_v1(seg_a, lead):
    return sr.dera_frac_refine(seg_a, lead)[1]


def fd_v3(seg_a, pre):
    """前导相干模板分数：返回 d*。用最后 min(pre,8) 个 upchirp。"""
    n_sym = min(int(pre), 8)
    p0 = int(pre) - n_sym
    best_d, best_s = 0.0, -1.0
    for d in _GRID:
        seg_d = seg_a if d == 0.0 else sr.frac_delay(seg_a, float(d))
        acc = np.zeros(N)
        combs = []
        for k in range(n_sym):
            res = sr.sav_demod(samples=seg_d, start_sample=(p0 + k) * NF,
                               sf=SF_, os_factor=OS, cfo_int=0)
            comb = res.combined_spectrum.astype(np.complex128)
            combs.append(comb)
            acc += np.abs(comb) ** 2
        c_hat = int(np.argmax(acc))
        x = np.array([c[c_hat] for c in combs])
        X = np.fft.fft(x, 4 * n_sym)
        score = float(np.max(np.abs(X) ** 2))
        if score > best_s + 1e-12:
            best_s, best_d = score, float(d)
    return best_d


def raw_ser_at(seg_a, pay0, f, d):
    seg_d = seg_a if d == 0.0 else sr.frac_delay(seg_a, float(d))
    rows = sr.sav_rows(seg_d, pay0, f["psym"])
    dd = [(int(np.argmax(rows[k])) - g) % N for k, g in enumerate(f["gt"])]
    mode = int(np.bincount(dd).argmax())
    return float(np.mean(np.array(dd) != mode))


def noisy_unit(level, seed, fi):
    f = G["frames"][fi]
    lead = f["pre"] + 6
    seg = np.asarray(f["iq"][f["hs"] - lead * NF:
                            f["hs"] + (8 + f["psym"] + 2) * NF],
                     dtype=np.complex128)
    S, N0 = G["snr"][fi]
    if level is not None:
        rng = np.random.default_rng((20260930 * 7919 + (int(level) + 100) * 131
                                     + seed * 17 + fi * 7919) % (2 ** 31))
        p_add = max(S / 10 ** (level / 10.0) - N0, 1e-30)
        seg = seg + (rng.standard_normal(len(seg))
                     + 1j * rng.standard_normal(len(seg))) * np.sqrt(p_add / 2.0)
    return f, seg, lead


print("== A: native 28 帧（d*_v1 / d*_v3 → SAVAUX raw SER）==")
nat_bad_v1 = nat_bad_v3 = 0
for fi in range(28):
    f, seg, lead = noisy_unit(None, 0, fi)
    cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
    if not cands:
        continue
    d1 = fd_v1(seg_as[0], lead)
    d3 = fd_v3(seg_as[0], f["pre"])
    s1 = raw_ser_at(seg_as[0], pay0s[0], f, d1)
    s3 = raw_ser_at(seg_as[0], pay0s[0], f, d3)
    nat_bad_v1 += s1 > 0
    nat_bad_v3 += s3 > 0
    flag = "  <-- v3 回归!" if s3 > s1 else ""
    if s1 > 0 or s3 > 0 or fi < 6:
        print("  f%02d d*%.2f→SER %.3f | d*%.2f→SER %.3f%s"
              % (fi, d1, s1, d3, s3, flag))
print("native 坏帧数: v1=%d v3=%d（应均为 0）" % (nat_bad_v1, nat_bad_v3))

print("\n== B: 14 个 fd 选点失误单元（taxonomy cls2）==")
tax = [json.loads(l) for l in open(
    r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
    r"\splice_dera_savaux\analysis\tax_checkpoint.jsonl", encoding="utf-8")]
cls2 = [r for r in tax if r.get("cls") == 2]
imp = 0
for r in cls2:
    lv, sd, fi = r["level"], r["seed"], r["frame"]
    f, seg, lead = noisy_unit(lv, sd, fi)
    cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
    if not cands:
        print("  (%d,%d,f%02d) sync-fail?" % (lv, sd, fi))
        continue
    d1 = fd_v1(seg_as[0], lead)
    d3 = fd_v3(seg_as[0], f["pre"])
    s1 = raw_ser_at(seg_as[0], pay0s[0], f, d1)
    s3 = raw_ser_at(seg_as[0], pay0s[0], f, d3)
    d_opt = r.get("d_opt", 99.0)
    imp += int(s3 < s1)
    print("  (%d,s%d,f%02d) d_opt=%+.2f | v1: d*=%+.2f SER=%.3f | "
          "v3: d*=%+.2f SER=%.3f %s"
          % (lv, sd, fi, d_opt, d1, s1, d3, s3,
             "<-- 改善" if s3 < s1 else ("=" if s3 == s1 else "回归")))
print("改善 %d/14" % imp)

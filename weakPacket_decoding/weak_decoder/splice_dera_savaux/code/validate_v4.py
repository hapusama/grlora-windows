# -*- coding: utf-8 -*-
"""v4-a：整包时延网格（D2F 网格思想 → Savaux 对齐）。

fd_c 的局限：只用前导 8 符号、整包一个静态时延。v4-a 把对齐搬进
序列决策：时延格 D=[−2,+2]×0.25（17 态），逐符号发射 = Savaux 合并谱
峰功率（对齐好→峰集中；混跳区→峰跨 bin 分裂，掉 2-3dB），转移 =
|Δd|≤1 步（SFO 漂移模型）+ 小步罚。Viterbi 输出逐符号时延轨迹，
每符号取该态的合并谱行 → δ 仲裁。

对照：bare（无对齐）/ fd_c（静态）/ TRELLIS（整包网格）。
测试集：native 28 帧 + v3 战役 DERAfdc×SAVAUX 的残余失败单元
（sync 成功但 CRC 失败，−22/−24/−26）。
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

D_GRID = np.arange(-2.0, 2.01, 0.25)          # 17 态
STEP_PEN = 0.4                                 # 每步移动罚（log 域）


def fd_coh(seg_a, pre):
    """v3 前导相干模板 fd（对照用，与 splice_v3_runner.fd_coh 相同）。"""
    n_sym = min(int(pre), 8)
    p0 = int(pre) - n_sym
    best_d, best_s = 0.0, -1.0
    for d in D_GRID:
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
        score = float(np.max(np.abs(np.fft.fft(x, 4 * n_sym)) ** 2))
        if score > best_s + 1e-12:
            best_s, best_d = score, float(d)
    return best_d


def trellis_rows(seg_a, pay0, psym):
    """整包时延网格：返回 (rows, delay轨迹)。发射=Savaux 合并谱峰功率。"""
    n_s = len(D_GRID)
    segs = [seg_a if d == 0.0 else sr.frac_delay(seg_a, float(d))
            for d in D_GRID]
    E = np.empty((psym, n_s))
    rows_all = np.empty((psym, n_s, N))
    for i in range(psym):
        for s in range(n_s):
            res = sr.sav_demod(samples=segs[s], start_sample=(pay0 + i) * NF,
                               sf=SF_, os_factor=OS, cfo_int=0)
            p = np.abs(res.combined_spectrum.astype(np.complex128)) ** 2
            rows_all[i, s] = p
            E[i, s] = np.log(np.max(p) + 1e-30)
    # Viterbi：转移 |Δs|≤1，罚 STEP_PEN×|Δs|
    score = E[0].copy()
    back = np.zeros((psym, n_s), dtype=int)
    for i in range(1, psym):
        ns = np.full(n_s, -1e30)
        bk = np.zeros(n_s, dtype=int)
        for s in range(n_s):
            best, arg = -1e30, s
            for s0 in (s - 1, s, s + 1):
                if 0 <= s0 < n_s:
                    v = score[s0] - STEP_PEN * abs(s - s0)
                    if v > best:
                        best, arg = v, s0
            ns[s], bk[s] = best, arg
        score = ns + E[i]
        back[i] = bk
    path = np.zeros(psym, dtype=int)
    path[-1] = int(np.argmax(score))
    for i in range(psym - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    rows = rows_all[np.arange(psym), path]
    return rows, D_GRID[path]


def raw_ser(rows, gt):
    d = [(int(np.argmax(rows[k])) - g) % N for k, g in enumerate(gt)]
    mode = int(np.bincount(d).argmax())
    return float(np.mean(np.array(d) != mode))


def eval_unit(level, seed, fi):
    """返回三法对比 dict。"""
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
    cands, seg_as, pay0s = sr.dera_sync(seg, f["pre"])
    if not cands:
        return None
    seg_a, pay0 = seg_as[0], pay0s[0]
    out = {}
    # bare
    out["bare"] = raw_ser(sr.sav_rows(seg_a, pay0, f["psym"]), f["gt"])
    # fd_c 静态
    d_c = fd_coh(seg_a, f["pre"])
    seg_c = seg_a if d_c == 0.0 else sr.frac_delay(seg_a, d_c)
    rows_c = sr.sav_rows(seg_c, pay0, f["psym"])
    out["fdc"] = raw_ser(rows_c, f["gt"])
    _d, ok_c = sr.decode_chain(rows_c, f)
    out["fdc_crc"] = bool(ok_c)
    # trellis
    rows_t, traj = trellis_rows(seg_a, pay0, f["psym"])
    out["trellis"] = raw_ser(rows_t, f["gt"])
    _d, ok_t = sr.decode_chain(rows_t, f)
    out["trellis_crc"] = bool(ok_t)
    out["traj_span"] = float(traj.max() - traj.min())
    return out


print("== A: native 28 帧 ==")
b = {"bare": 0, "fdc": 0, "trellis": 0}
for fi in range(28):
    r = eval_unit(None, 0, fi)
    if r is None:
        continue
    for k in ("bare", "fdc", "trellis"):
        b[k] += r[k] > 0
    if r["bare"] > 0 or r["trellis"] > 0 or fi < 4:
        print("  f%02d bare=%.3f fdc=%.3f trellis=%.3f traj_span=%.2f"
              % (fi, r["bare"], r["fdc"], r["trellis"], r["traj_span"]))
print("native 坏帧: bare=%d fdc=%d trellis=%d（fdc 应 0，trellis 应 ≤fdc）"
      % (b["bare"], b["fdc"], b["trellis"]))

print("\n== B: v3 残余失败单元（DERAfdc×SAVAUX crc_fail=1 且 sync=1）==")
ck = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dera_fdc_gamma_20261005\checkpoint.jsonl"
units = []
for line in open(ck, encoding="utf-8"):
    r = json.loads(line)
    if r["level"] in (-22, -24, -26) and r.get("sync_dera") \
            and r["chains"].get("DERAfdc×SAVAUX", {}).get("crc_fail") == 1:
        units.append((r["level"], r["seed"], r["frame"]))
print("目标单元数:", len(units))
resc_c = resc_t = tot = 0
sers = {"fdc": [], "trellis": []}
for lv, sd, fi in units:
    r = eval_unit(lv, sd, fi)
    if r is None:
        continue
    tot += 1
    resc_c += r["fdc_crc"]
    resc_t += r["trellis_crc"]
    sers["fdc"].append(r["fdc"])
    sers["trellis"].append(r["trellis"])
print("可评估 %d 单元：CRC 救回 fdc=%d trellis=%d" % (tot, resc_c, resc_t))
for k in ("fdc", "trellis"):
    a = np.array(sers[k])
    print("  %-8s raw SER: mean=%.3f med=%.3f 归零率=%.2f"
          % (k, a.mean(), np.median(a), float(np.mean(a == 0))))

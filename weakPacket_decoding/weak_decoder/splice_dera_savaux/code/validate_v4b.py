# -*- coding: utf-8 -*-
"""v4-b：偏移副本版——每个粗偏移副本单独维护一个细时延网格。

结构：5 个粗副本（中心 {−1.5,−0.75,0,+0.75,+1.5}）× 各自 7 态细格
（±0.375，步 0.125；绝对时延覆盖 [−1.875,+1.875]@0.125，分辨率是整包
版的 2 倍）。每副本独立 Viterbi（转移 ±1 细步 + 0.2 罚）产出整包行；
副本按路径得分排序后进跨副本 δ CRC 仲裁（同款 Stage-3 原语）。

判定集：native 28 帧（零回归门）+ v4 战役 DERA×SAVT 的残余失败单元
（sync=1 且 CRC 失败，−22/−24/−26）——SAVT2 在 SAVT 之上的净增量。
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

CENTERS = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
FINE = np.arange(-0.375, 0.376, 0.125)          # 7 态
PEN_F = 0.2

_v4_spec = ilu.spec_from_file_location(
    "v4", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\weak_decoder"
          r"\splice_dera_savaux\code\splice_v4_runner.py")
v4 = ilu.module_from_spec(_v4_spec)
_v4_spec.loader.exec_module(v4)                  # 复用整包版做对照


def savt2_outputs(seg_a, pay0, psym):
    """返回按路径得分降序的副本行列表 + 每副本 (center, path_score)。"""
    abs_d = np.concatenate([CENTERS[k] + FINE for k in range(len(CENTERS))])
    n_d = len(abs_d)
    E = np.empty((psym, n_d))
    rows_all = np.empty((psym, n_d, N))
    for j, d in enumerate(abs_d):
        seg_d = seg_a if d == 0.0 else sr.frac_delay(seg_a, float(d))
        for i in range(psym):
            res = sr.sav_demod(samples=seg_d, start_sample=(pay0 + i) * NF,
                               sf=SF_, os_factor=OS, cfo_int=0)
            p = np.abs(res.combined_spectrum.astype(np.complex128)) ** 2
            rows_all[i, j] = p
            E[i, j] = np.log(np.max(p) + 1e-30)
    outs = []
    n_f = len(FINE)
    for k in range(len(CENTERS)):
        sl = slice(k * n_f, (k + 1) * n_f)
        Ek = E[:, sl]
        score = Ek[0].copy()
        back = np.zeros((psym, n_f), dtype=int)
        for i in range(1, psym):
            ns = np.full(n_f, -1e30)
            bk = np.zeros(n_f, dtype=int)
            for s in range(n_f):
                best, arg = -1e30, s
                for s0 in (s - 1, s, s + 1):
                    if 0 <= s0 < n_f:
                        vv = score[s0] - PEN_F * abs(s - s0)
                        if vv > best:
                            best, arg = vv, s0
                ns[s], bk[s] = best, arg
            score = ns + Ek[i]
            back[i] = bk
        path = np.zeros(psym, dtype=int)
        path[-1] = int(np.argmax(score))
        for i in range(psym - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        rows_k = rows_all[:, sl][np.arange(psym), path]
        outs.append((float(score[path[-1]]), CENTERS[k], rows_k))
    outs.sort(key=lambda t: -t[0])
    return outs


def raw_ser(rows, gt):
    d = [(int(np.argmax(rows[k])) - g) % N for k, g in enumerate(gt)]
    mode = int(np.bincount(d).argmax())
    return float(np.mean(np.array(d) != mode))


def eval_unit(level, seed, fi):
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
    rows_t = v4.trellis_rows(seg_a, pay0, f["psym"])
    out["savt"] = raw_ser(rows_t, f["gt"])
    _d, ok = sr.decode_chain(rows_t, f)
    out["savt_crc"] = bool(ok)
    outs = savt2_outputs(seg_a, pay0, f["psym"])
    rows2 = outs[0][2]
    out["savt2"] = raw_ser(rows2, f["gt"])
    _d, ok2, _rows2 = sr.decode_chain(None, f, top5=[o[2] for o in outs])
    out["savt2_crc"] = bool(ok2)
    out["best_center"] = outs[0][1]
    return out


print("== A: native 28 帧 ==")
bad = {"savt": 0, "savt2": 0}
for fi in range(28):
    r = eval_unit(None, 0, fi)
    if r is None:
        continue
    bad["savt"] += r["savt"] > 0
    bad["savt2"] += r["savt2"] > 0
    if r["savt"] > 0 or r["savt2"] > 0 or fi < 4:
        print("  f%02d savt=%.3f savt2=%.3f best_c=%+.2f"
              % (fi, r["savt"], r["savt2"], r["best_center"]))
print("native 坏帧: savt=%d savt2=%d（应均 0）" % (bad["savt"], bad["savt2"]))

print("\n== B: v4 残余失败单元（DERA×SAVT crc_fail=1 且 sync=1）==")
ck = r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data\experiments\dera_savt_20261005\checkpoint.jsonl"
units = []
for line in open(ck, encoding="utf-8"):
    r = json.loads(line)
    if r["level"] in (-22, -24, -26) and r.get("sync_dera") \
            and r["chains"].get("DERA×SAVT", {}).get("crc_fail") == 1:
        units.append((r["level"], r["seed"], r["frame"]))
print("目标单元数:", len(units))
resc = {"savt": 0, "savt2": 0}
sers = {"savt": [], "savt2": []}
tot = 0
for lv, sd, fi in units:
    r = eval_unit(lv, sd, fi)
    if r is None:
        continue
    tot += 1
    resc["savt"] += r["savt_crc"]
    resc["savt2"] += r["savt2_crc"]
    sers["savt"].append(r["savt"])
    sers["savt2"].append(r["savt2"])
print("可评估 %d 单元：CRC 救回 savt=%d savt2=%d"
      % (tot, resc["savt"], resc["savt2"]))
for k in ("savt", "savt2"):
    a = np.array(sers[k])
    print("  %-6s raw SER: mean=%.3f med=%.3f 归零率=%.2f"
          % (k, a.mean(), np.median(a), float(np.mean(a == 0))))

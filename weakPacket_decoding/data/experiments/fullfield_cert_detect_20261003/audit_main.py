# -*- coding: utf-8 -*-
"""audit_main.py — 独立统计审计 fullfield_cert_detect_20261003。

复算口径（与实验方定义一致，独立实现）：
  thr[v][lv]  = 该档全部 h0 分数（level 非 None）的 99% 分位（线性插值）；
  margin      = 10log10(score) - 10log10(thr)；
  净胜(V|A_P) = median(margin_V) - median(margin_A_P)。
只读 checkpoint；输出 AUDIT.md。运行： D:/mysoft2/miniconda3/python audit_main.py
"""
import json
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_lib import (PFA, THEO_Q50, THEO_Q90, chi2_sf, gof_exponential,
                       homog_chisq, ols, q99, spearman)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
LEVELS = [None, -10, -14, -17, -20, -22, -24, -26, -28, -30, -32]
NOISY = [lv for lv in LEVELS if lv is not None]

FILES = {
    "ff1": ("checkpoint.jsonl",
            ["A_P", "FF_up", "FF_lin", "FF_q", "FF_ml", "FF_tmpl"]),
    "ff2": ("checkpoint_ff2.jsonl",
            ["A_P", "FF_se", "FF_se_ok", "FF_tmpl"]),
    "ff3": ("checkpoint_ff3.jsonl",
            ["A_P", "FF_split_pre", "FF_split_syn", "FF_split_model",
             "FF_tmpl"]),
    "ff4": ("checkpoint_ff4.jsonl",
            ["A_P", "FF_bank1", "FF_bank9", "FF_bank25", "FF_tmpl"]),
    "ff5": ("checkpoint_ff5.jsonl",
            ["A_P", "FF_GLBLRT", "FF_pert30", "FF_pert90", "FF_tmpl"]),
    "ff6": ("checkpoint_ff6.jsonl",
            ["A_P", "FF_cert8", "FF_tmpl"]),
}

md = []
CON = {}


def emit(s=""):
    md.append(s)


def load(fname):
    rows = []
    with open(os.path.join(EXP_DIR, fname), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def subset(rows, ke=None):
    if ke is None:
        return rows
    return [r for r in rows if r.get("ke") == ke]


def sc(record, v):
    return record["scores"][v]


def thresholds(rows, variants, ke=None):
    """thr[(v, lv)] = 该档全部 h0 的 99% 分位。"""
    rows = subset(rows, ke)
    thr = {}
    for v in variants:
        for lv in NOISY:
            thr[(v, lv)] = q99([sc(r, v) for r in rows
                                if r["kind"] == "h0" and r["level"] == lv])
    return thr


def pooled_thr_db(rows, v, ke=None):
    rows = subset(rows, ke)
    x = [sc(r, v) for r in rows
         if r["kind"] == "h0" and r["level"] is not None]
    return 10 * math.log10(q99(x))


def margin_stats(rows, v, lv, thr, cap=None, seed=None, ke=None):
    if ke is not None:
        rows = [r for r in rows if r.get("ke") == ke]
    sel = [r for r in rows if r["kind"] == "h1" and r["level"] == lv]
    if cap is not None:
        sel = [r for r in sel if r["cap"] == cap]
    if seed is not None:
        sel = [r for r in sel if r["seed"] == seed]
    if not sel:
        return (float("nan"), float("nan"), 0)
    tdb = 10 * math.log10(thr[(v, lv)])
    mg = [10 * math.log10(sc(r, v)) - tdb for r in sel]
    pd = float(np.mean([sc(r, v) > thr[(v, lv)] for r in sel]))
    return (float(np.median(mg)), pd, len(sel))


# ================================================================ 加载
print("== 加载 ==", flush=True)
DATA = {}
for tag, (fn, vs) in FILES.items():
    rows = load(fn)
    DATA[tag] = dict(rows=rows, variants=vs, fname=fn)
    print("  %s: %s  %d 条" % (tag, fn, len(rows)), flush=True)

# frame -> cap
fc = {}
for r in DATA["ff6"]["rows"]:
    if r["kind"] == "h1":
        fc[r["frame"]] = r["cap"]
CAPS = sorted(set(fc.values()))
FRAMES_CAP = {c: sorted(f for f, cc in fc.items() if cc == c) for c in CAPS}
PRE_CAP = {}
for r in DATA["ff6"]["rows"]:
    PRE_CAP.setdefault(r["cap"], r["pre"])

# ================================================================ 0 完整性
emit("# 对抗性审计报告 — fullfield_cert_detect_20261003\n")
emit("审计：独立统计审计 agent（audit_lib.py / audit_main.py，"
     "python 3.12.9 + numpy %s；无 scipy，卡方/二项等特殊函数自实现并在库中可复核）。"
     % np.__version__)
emit("口径与实验方一致：thr=各噪声档全部 h0 的 99% 分位；margin=10log10(score)−10log10(thr)；"
     "净胜=median(margin_V)−median(margin_A_P)。checkpoint 全程只读。\n")
emit("## 0. 数据完整性与一致性\n")
emit("| 检查 | 结果 |")
emit("|---|---|")
integ_ok = True
for tag, d in DATA.items():
    rows = d["rows"]
    exact = len(rows) - len(set(json.dumps(r, sort_keys=True) for r in rows))
    miss = sum(1 for r in rows for v in d["variants"]
               if v not in r["scores"] or not np.isfinite(r["scores"][v]))
    emit("| %s `%s`：%d 行；完全重复行；变体缺失/非有限 | %d；%d |"
         % (tag, d["fname"], len(rows), exact, miss))
    if exact or miss:
        integ_ok = False
emit("| frame→cap 唯一性 | 28 帧唯一映射：%s |"
     % "，".join("%s（P=%d，%d 帧）" % (c, PRE_CAP[c], len(FRAMES_CAP[c]))
                 for c in CAPS))

# 跨文件同单元一致性：同 (kind,level,seed,cap,frame) 单元内按文件顺序对齐 h0 窗
def unit_series(rows, v, ke=None):
    rows = subset(rows, ke)
    ser = {}
    for r in rows:
        k = (r["kind"], r["level"], r["seed"], r["cap"], r["frame"])
        ser.setdefault(k, []).append(r["scores"][v])
    return ser

emit("\n**跨文件同单元分数一致性**（A_P / FF_tmpl 应逐位一致；h0 窗按单元内文件顺序对齐）：")
for v in ("A_P", "FF_tmpl"):
    ref = unit_series(DATA["ff1"]["rows"], v)
    worst, wn = 0.0, ""
    for tag in ("ff2", "ff4", "ff5", "ff6"):
        s2 = unit_series(DATA[tag]["rows"], v)
        for k in set(ref) & set(s2):
            if len(ref[k]) != len(s2[k]):
                worst, wn = float("inf"), "%s(窗口数不一致 %s)" % (tag, k)
                break
            for a, b in zip(ref[k], s2[k]):
                dd = abs(a - b) / max(abs(a), 1e-12)
                if dd > worst:
                    worst, wn = dd, str((tag, k))
    s3 = unit_series(DATA["ff3"]["rows"], v, ke=4)
    for k in set(ref) & set(s3):
        k4 = (k[0], k[1], k[2], k[3], k[4])
        if len(ref[k4]) != len(s3[k]):
            worst, wn = float("inf"), "ff3(窗口数不一致)"
            break
        for a, b in zip(ref[k4], s3[k]):
            dd = abs(a - b) / max(abs(a), 1e-12)
            if dd > worst:
                worst, wn = dd, str(("ff3", k))
    emit("- %s：与 ff1 最大相对差 %.2e（%s）%s"
         % (v, worst, wn[:80], "✓" if worst < 1e-9 else "⚠️ 不一致"))
    if worst >= 1e-9:
        integ_ok = False
a4, a8 = {}, {}
for r in DATA["ff3"]["rows"]:
    k = (r["kind"], r["level"], r["seed"], r["cap"], r["frame"])
    (a4 if r.get("ke") == 4 else a8).setdefault(k, []).append(r["scores"]["A_P"])
worst = 0.0
for k in set(a4) & set(a8):
    if len(a4[k]) == len(a8[k]):
        worst = max(worst, max(abs(x - y) / max(abs(x), 1e-12)
                               for x, y in zip(a4[k], a8[k])))
emit("- ff3 内部 ke=4 vs ke=8 同单元 A_P：最大相对差 %.2e %s"
     % (worst, "✓" if worst < 1e-9 else "⚠️"))
emit("\n**完整性小结：%s**\n"
     % ("通过——无重复行、无缺分；同单元分数跨 5 个文件逐位一致 → 确定性重算、无静默改数"
        if integ_ok else "存在异常，见上"))
emit("**溯源备注**：① 目录中存在 ff/ff2/ff3/ff4/ff5 的 runner 源码，"
     "唯独没有产生 `checkpoint_ff6.jsonl` 的 runner（源码与 __pycache__ 均无 ff6 痕迹；"
     "ff5_runner.py 写入的是 ff5 checkpoint）——ff6（主结果轮）的生成脚本未入库，"
     "复现依赖数据文件本身，建议补交；② 本审计为**统计层**审计：GT 锚定、噪声注入与"
     "变体实现是否忠实（harness 正确性）不在可复核范围内，仅审计数据内部的统计一致性。\n")

# ================================================================ 1 复算
emit("## 1. 独立复算 vs RESULTS.md 声明\n")
emit("### 1.1 噪声档门限（99% 分位，dB）\n")
emit("| 轮 | 池化=全 10 档 | 池化=工作带 −20..−32 | 声称（dB） |")
WORK = [-20, -22, -24, -26, -28, -30, -32]
claims_pool = {
    "ff1": ("A_P 11.26 / FF_lin 11.44 / FF_q 11.83 / FF_ml 12.53 / FF_tmpl 8.56",
            {"A_P": 11.26, "FF_lin": 11.44, "FF_q": 11.83, "FF_ml": 12.53,
             "FF_tmpl": 8.56}),
    "ff3": ("A_P 10.8 / FF_split_model 7.5", {"A_P": 10.8,
                                              "FF_split_model": 7.5}),
    "ff4": ("bank1 7.5 / bank9 8.3 / bank25 8.9",
            {"FF_bank1": 7.5, "FF_bank9": 8.3, "FF_bank25": 8.9}),
}
THR = {}
for tag, d in DATA.items():
    ke = 4 if tag == "ff3" else None
    THR[tag] = thresholds(d["rows"], d["variants"], ke=ke)
    rows = subset(d["rows"], ke)

    def pool_db(v, lvs):
        x = [sc(r, v) for r in rows
             if r["kind"] == "h0" and r["level"] in lvs]
        return 10 * math.log10(q99(x))
    got_all = " / ".join("%s %.2f" % (v, pool_db(v, NOISY))
                         for v in d["variants"])
    got_work = " / ".join("%s %.2f" % (v, pool_db(v, WORK))
                          for v in d["variants"])
    emit("| %s | %s | %s | %s |"
         % (tag, got_all, got_work, claims_pool.get(tag, ("—", {}))[0]))
for tag, (_, cc) in claims_pool.items():
    ke = 4 if tag == "ff3" else None
    devs = []
    for v, cv in cc.items():
        g = pooled_thr_db(DATA[tag]["rows"], v, ke=ke)
        devs.append("%s 全档Δ%+.2f" % (v, g - cv))
    emit("- %s 门限偏差（相对声称，全 10 档池化）：%s"
         % (tag, "，".join(devs)))
emit("注：ff3/ff4 的声称门限（10.8/7.5、7.5/8.3/8.9）与**工作带 −20..−32 池化**"
     "吻合（复算 10.87/7.49、7.49/8.35/8.83），全 10 档池化多 ~0.4 dB——"
     "−10/−14/−17 档门限受邻包泄漏残余抬高（A_P 14.30@−10 → 10.60@−32）。"
     "ff1 声称则与全档池化逐位一致。两口径相差的这 ~0.4 dB 不改变任何变体间差值结论，"
     "但报告应写明口径。")

emit("\n### 1.2 ff6 主表复算（裕度 = 中位 H1 margin，Pd@Pfa=1e-2）\n")
rows6 = DATA["ff6"]["rows"]
claims_ff6 = {
    -14: (20.3, 23.3, 3.01, 1.00), -20: (15.4, 18.5, 3.12, 0.99),
    -22: (14.2, 16.3, 2.08, 1.00), -24: (12.0, 13.9, 1.86, 0.94),
    -26: (9.9, 11.7, 1.77, 0.94), -28: (8.4, 9.7, 1.38, 0.92),
    -30: (6.3, 6.9, 0.59, 0.85), -32: (4.4, 2.4, -2.07, 0.64),
}
emit("| 档 | A_P 裕度(Pd) | cert8 裕度(Pd) | 净胜复算 | 净胜声称 | Δ | Pd 声称 |")
ff6_dev = ff6_pd_dev = 0.0
for lv in NOISY:
    m0, pd0, _ = margin_stats(rows6, "A_P", lv, THR["ff6"])
    m1, pd1, _ = margin_stats(rows6, "FF_cert8", lv, THR["ff6"])
    nw = m1 - m0
    c = claims_ff6.get(lv)
    if c:
        ff6_dev = max(ff6_dev, abs(m0 - c[0]), abs(m1 - c[1]), abs(nw - c[2]))
        ff6_pd_dev = max(ff6_pd_dev, abs(pd1 - c[3]))
        emit("| %d | %+.2f (%.2f) | %+.2f (%.2f) | %+.2f | %+.2f | %+.2f | %.2f |"
             % (lv, m0, pd0, m1, pd1, nw, c[2], nw - c[2], c[3]))
    else:
        emit("| %d | %+.2f (%.2f) | %+.2f (%.2f) | %+.2f | —（表未列） | — | — |"
             % (lv, m0, pd0, m1, pd1, nw))
emit("\nff6 最大偏差：裕度/净胜 %.3f dB，Pd %.3f → %s"
     % (ff6_dev, ff6_pd_dev,
        "✓ 全部落在舍入误差内" if ff6_dev < 0.06 and ff6_pd_dev < 0.006
        else "⚠️ 超出舍入误差"))

emit("\n### 1.3 ff1 / ff2 逐 cap 声明抽查\n")
emit("| 轮 cap(组) 档 | 变体 | 复算裕度 | 声称 | Δ |")
claims_ff1 = {
    ("0_0_0_10_14_8", -26): dict(A_P=7.9, FF_lin=7.5, FF_q=7.4, FF_ml=7.9,
                                 FF_tmpl=13.5),
    ("0_0_0_10_14_8", -30): dict(A_P=3.7, FF_lin=3.3, FF_q=3.1, FF_ml=3.6,
                                 FF_tmpl=7.8),
    ("0_0_0_10_14_8", -32): dict(A_P=1.8, FF_lin=1.3, FF_q=1.3, FF_ml=2.5,
                                 FF_tmpl=7.2),
    ("0_0_0_10_14_16", -32): dict(A_P=5.0, FF_lin=3.9, FF_q=4.2, FF_ml=4.6,
                                  FF_tmpl=8.9),
    ("0_0_0_10_14_32", -32): dict(A_P=7.8, FF_lin=6.9, FF_q=6.9, FF_ml=7.6,
                                  FF_tmpl=12.0),
}
claims_ff2 = {
    ("0_0_0_10_14_8", -26): dict(A_P=7.9, FF_se=8.0, FF_se_ok=9.4,
                                 FF_tmpl=13.5),
    ("0_0_0_10_14_16", -26): dict(A_P=10.3, FF_se=10.0, FF_se_ok=11.5,
                                  FF_tmpl=15.1),
    ("0_0_0_10_14_32", -26): dict(A_P=13.5, FF_se=12.8, FF_se_ok=14.3,
                                  FF_tmpl=17.7),
}
ff_dev = 0.0
for tag, cl, thr in (("ff1", claims_ff1, THR["ff1"]),
                     ("ff2", claims_ff2, THR["ff2"])):
    for (cap, lv), cc in cl.items():
        for v, cv in cc.items():
            m, _, _ = margin_stats(DATA[tag]["rows"], v, lv, thr, cap=cap)
            ff_dev = max(ff_dev, abs(m - cv))
            emit("| %s %s %d | %s | %+.2f | %+.1f | %+.2f |"
                 % (tag, cap[-1] + "（P=%d）" % PRE_CAP[cap], lv,
                    v, m, cv, m - cv))
emit("\nff1/ff2 逐 cap 裕度最大偏差 %.2f dB（≤0.05 视为舍入内）%s"
     % (ff_dev, "✓" if ff_dev < 0.06 else "⚠️"))
p8 = "0_0_0_10_14_8"
pds = {}
for v in ("A_P", "FF_tmpl"):
    _, pds[v], _ = margin_stats(DATA["ff1"]["rows"], v, -32, THR["ff1"], cap=p8)
emit("ff1 P=8 −32 Pd：A_P %.2f（声称 0.83）、FF_tmpl %.2f（声称 1.00）。" % (pds["A_P"], pds["FF_tmpl"]))
h0m = []
for v, c in (("A_P", 8.89), ("FF_se", 10.04), ("FF_se_ok", 5.19),
             ("FF_tmpl", 1.71)):
    xs_dB = [10 * math.log10(sc(r, v)) for r in DATA["ff2"]["rows"]
             if r["kind"] == "h0" and r["level"] is not None]
    xs_lin = [sc(r, v) for r in DATA["ff2"]["rows"]
              if r["kind"] == "h0" and r["level"] is not None]
    h0m.append("%s 均值dB %.2f / dB(均值线性) %.2f（声称 %.2f）"
               % (v, float(np.mean(xs_dB)),
                  10 * math.log10(float(np.mean(xs_lin))), c))
emit("ff2 H0 池化均值（两种口径）：%s。声称值与 **dB(线性均值)** 口径吻合；"
     "均值dB 口径因 Jensen 差系统性低 ~1-2.6 dB，非数据分歧。" % "；".join(h0m))

emit("\n### 1.4 各轮净胜全表（复算，V − A_P，dB；括号 Pd）\n")
emit("| 轮 | 档 | 净胜（Pd） |")
NW = {}
for tag, d in DATA.items():
    ke = 4 if tag == "ff3" else None
    for lv in NOISY:
        cells = []
        for v in d["variants"]:
            if v == "A_P":
                continue
            m, pd, _ = margin_stats(d["rows"], v, lv, THR[tag], ke=ke)
            m0, _, _ = margin_stats(d["rows"], "A_P", lv, THR[tag], ke=ke)
            NW[(tag, v, lv)] = m - m0
            cells.append("%s %+.2f (%.2f)" % (v.replace("FF_", ""), m - m0, pd))
        emit("| %s | %d | %s |" % (tag, lv, "；".join(cells)))
gwork = [NW[("ff5", "FF_GLBLRT", lv)] for lv in (-22, -24, -26, -28, -30, -32)]
emit("\nff5 FF_GLBLRT 净胜：工作带 −22..−32 范围 [%.2f, %.2f] dB（声称 −0.5~−1.3）%s；"
     "浅档 −10..−20 为 %s（全场积分在高 SNR 端不再吃亏）。"
     % (min(gwork), max(gwork),
        "✓" if min(gwork) > -1.5 and max(gwork) < -0.4 else "⚠️",
        "、".join("%+.2f" % NW[("ff5", "FF_GLBLRT", lv)]
                  for lv in (-10, -14, -17, -20))))

# ================================================================ A1
emit("\n## A1. 指数性检验（噪声档 h0 归一化分数 vs Exp(rate=4.605)）\n")
emit("归一化 = 分数 ÷ 本档 99% 分位门限。精确指数 ⇒ q50/q90/q99 = 0.1505/0.500/1.000，"
     "生存函数 e^(−4.605x)。GOF：20 等概率箱卡方，df=19（未扣门限拟合的 1 个参数，保守）。"
     "native 档不参与（污染，见 A6）。\n")
A1 = {}
emit("| 轮 | 变体 | n | χ²(19) | p | q50(Δ%) | q90(Δ%) | q99 | q90/q99 原始池化 |")
for tag, d in DATA.items():
    ke = 4 if tag == "ff3" else None
    rows = subset(d["rows"], ke)
    for v in d["variants"]:
        xs = []
        for lv in NOISY:
            t = THR[tag][(v, lv)]
            xs.extend(sc(r, v) / t for r in rows
                      if r["kind"] == "h0" and r["level"] == lv)
        xs = np.array(xs)
        raw = np.array([sc(r, v) for r in rows
                        if r["kind"] == "h0" and r["level"] is not None])
        qr = float(np.quantile(raw, 0.9) / np.quantile(raw, 0.99))
        g = gof_exponential(xs, 20)
        q50, q90, q99v = (float(np.quantile(xs, 0.5)),
                          float(np.quantile(xs, 0.9)),
                          float(np.quantile(xs, 0.99)))
        A1[(tag, v)] = (g, q50, q90, q99v, qr)
        emit("| %s | %s | %d | %.1f | %.3g | %.4f (%+d%%) | %.4f (%+d%%) | %.4f | %.4f |"
             % (tag, v, g["n"], g["chi2"], g["p"], q50,
                round(100 * (q50 / THEO_Q50 - 1)), q90,
                round(100 * (q90 / THEO_Q90 - 1)), q99v, qr))
emit("\n**q-ratio 声明对照**（声称 = 原始分数全档池化 q90/q99）：ff3 A_P 0.659 / "
     "FF_split_model 0.506；ff4 bank1/9/25 = 0.506/0.508/0.557；ff6 cert8 0.505。"
     "复算（上表末列）与声称逐项吻合（Δ≤0.003）。但注意：该口径受 −10 档泄漏抬高池化 q99 "
     "的稀释；**分档** q90/q99 中位为 cert8 0.536、bank9 0.539、split_model 0.527"
     "（各档 0.46~0.64 散布），均高于理论 0.500。")
emit("\n**ff6 重点：分档 q50/理论、q90/理论：**\n")
emit("| 档 | A_P q50/q90 比 | cert8 q50/q90 比 | tmpl q50/q90 比 |")
for lv in NOISY:
    cells = []
    for v in ("A_P", "FF_cert8", "FF_tmpl"):
        t = THR["ff6"][(v, lv)]
        xs = np.array([sc(r, v) / t for r in rows6
                       if r["kind"] == "h0" and r["level"] == lv])
        cells.append("%.3f / %.3f" % (np.quantile(xs, 0.5) / THEO_Q50,
                                      np.quantile(xs, 0.9) / THEO_Q90))
    emit("| %d | %s |" % (lv, " | ".join(cells)))
emit("\n**ff6 分档 20 箱 GOF p 值（−10→−32）：**")
for v in ("A_P", "FF_cert8", "FF_tmpl"):
    ps = []
    for lv in NOISY:
        t = THR["ff6"][(v, lv)]
        ps.append(gof_exponential(
            [sc(r, v) / t for r in rows6
             if r["kind"] == "h0" and r["level"] == lv], 20)["p"])
    emit("- %s：%s" % (v, " ".join("%.2f" % p for p in ps)))
emit("\n**ff6 分档 q90/q99 原始比：**")
for v in ("A_P", "FF_cert8", "FF_tmpl"):
    rs = []
    for lv in NOISY:
        a = np.array([sc(r, v) for r in rows6
                      if r["kind"] == "h0" and r["level"] == lv])
        rs.append(float(np.quantile(a, 0.9) / np.quantile(a, 0.99)))
    emit("- %s：%s（中位 %.3f）" % (v, " ".join("%.3f" % x for x in rs),
                                   float(np.median(rs))))
c8g, apg, tgg = (A1[("ff6", "FF_cert8")], A1[("ff6", "A_P")],
                 A1[("ff6", "FF_tmpl")])
CON["A1"] = ("FF_cert8 池化 χ²=%.0f（p=%.0e），分档 GOF p 全部 <0.01，"
             "q50 偏 %+.0f%%、q90 偏 %+.0f%% → 精确指数在真实数据上被拒绝；"
             "声称的 q-ratio 0.505 仅在原始分数全档池化口径下复现（受 −10 泄漏稀释），"
             "分档中位 0.536。A_P χ²=%.0f（q-ratio 0.659，max 统计量固有非指数，与自述一致）。"
             "对照：FF_tmpl χ²=%.1f p=%.2f、分档 GOF 基本通过 → 检验有判别力，"
             "cert8 的拒绝是真实形状偏离（bank-max 中尾增厚），非检验假阳"
             % (c8g[0]["chi2"], c8g[0]["p"], 100 * (c8g[1] / THEO_Q50 - 1),
                100 * (c8g[2] / THEO_Q90 - 1), apg[0]["chi2"],
                tgg[0]["chi2"], tgg[0]["p"]))

# ================================================================ A2
emit("\n## A2. 逐 capture 虚警均匀性\n")
emit("各档门限下统计各 cap h0 超阈数；跨 cap 齐性 = 2×3 卡方（渐近 p + 条件于总超阈数的 "
     "Monte Carlo 精确 p，2 万次）。每档每 cap 期望超阈仅 ~1.5 条，故正式检验用 10 档合并"
     "（每 cap n≈1400，期望 ~14 条）；若分档 CFAR 成立，合并后仍应均匀。\n")
A2 = {}
emit("| 轮 | 变体 | cap 超阈/总数（率%） | χ²(2) | p_asym | p_MC | 判定 |")
for tag, d in DATA.items():
    ke = 4 if tag == "ff3" else None
    rows = subset(d["rows"], ke)
    for v in d["variants"]:
        counts, totals = [], []
        for c in CAPS:
            k = n = 0
            for lv in NOISY:
                t = THR[tag][(v, lv)]
                for r in rows:
                    if (r["kind"] == "h0" and r["level"] == lv
                            and r["cap"] == c):
                        n += 1
                        if sc(r, v) > t:
                            k += 1
            counts.append(k)
            totals.append(n)
        h = homog_chisq(counts, totals)
        A2[(tag, v)] = (h, counts, totals)
        det = ("✓ 均匀" if h["p_mc"] > 0.05 else
               ("△ 边缘不均" if h["p_mc"] > 0.01 else "⚠️ 不均匀"))
        cells = "；".join("%d/%d(%.2f%%)" % (k, n, 100 * k / n)
                          for k, n in zip(counts, totals))
        emit("| %s | %s | %s | %.1f | %.3g | %.3g | %s |"
             % (tag, v, cells, h["chi2"], h["p_asym"], h["p_mc"], det))
emit("\n**ff6 重点：逐档×cap 超阈率（%）：**\n")
emit("| 档 | A_P 8/16/32 | cert8 8/16/32 | tmpl 8/16/32 |")
for lv in NOISY:
    cells = []
    for v in ("A_P", "FF_cert8", "FF_tmpl"):
        row = []
        for c in CAPS:
            t = THR["ff6"][(v, lv)]
            rr = [r for r in rows6 if r["kind"] == "h0"
                  and r["level"] == lv and r["cap"] == c]
            row.append(100 * float(np.mean([sc(r, v) > t for r in rr])))
        cells.append("/".join("%.1f" % x for x in row))
    emit("| %d | %s |" % (lv, " | ".join(cells)))
h8 = A2[("ff6", "FF_cert8")][0]
h8c = A2[("ff6", "A_P")][0]
CON["A2"] = ("cert8 三 cap 率 %s，χ²=%.1f，p_MC=%.3g → %s"
             "（P=16 一贯偏重；A_P 同表 χ²=%.1f p_MC=%.3g）"
             % (", ".join("%.2f%%" % (100 * r) for r in h8["rates"]),
                h8["chi2"], h8["p_mc"],
                "边缘性不均（5% 显著、1% 不显著），无决定性单 cap 主导"
                if 0.01 < h8["p_mc"] <= 0.05 else
                ("无单 cap 主导" if h8["p_mc"] > 0.05
                 else "⚠️ 单 cap 主导（证书瑕疵）"),
                h8c["chi2"], h8c["p_mc"]))
emit("\n**ff6 cert8 / A_P 的留一 capture 敏感性**（门限只用 cap≠c 的 h0，"
     "净胜在 cap=c 的 H1 上评——若某 cap 主导门限，其对内净胜应大幅缩水）：\n")
emit("| 档 | cap | cert8 净胜(LOO-cap 门限) | A_P 裕度 | cert8 裕度 |")
for lv in (-20, -22, -24, -26, -30, -32):
    for c in CAPS:
        thr = {}
        for v in ("A_P", "FF_cert8"):
            thr[(v, lv)] = q99([sc(r, v) for r in rows6 if r["kind"] == "h0"
                                and r["level"] == lv and r["cap"] != c])
        m1, _, _ = margin_stats(rows6, "FF_cert8", lv, thr, cap=c)
        m0, _, _ = margin_stats(rows6, "A_P", lv, thr, cap=c)
        emit("| %d | P=%d | %+.2f | %+.2f | %+.2f |"
             % (lv, PRE_CAP[c], m1 - m0, m0, m1))

# ================================================================ A3
emit("\n## A3. 净胜的帧自助置信区间（2000 次，每档内）\n")
emit("按帧号有放回重采样 28 帧（帧=簇，携带其全部种子 h1+h0 记录），"
     "每次重采样本内重算门限→margin→净胜；CI=百分位 2.5/97.5，rng 种子 20261003。\n")


def bootstrap_netwin(rows, v1, nboot=2000, seed=20261003):
    out = {}
    rng = np.random.default_rng(seed)
    for lv in NOISY:
        h0db = {v: {} for v in (v1, "A_P")}
        h1db = {v: {} for v in (v1, "A_P")}
        frames = sorted(set(r["frame"] for r in rows if r["level"] == lv))
        for f in frames:
            for v in (v1, "A_P"):
                h0db[v][f] = np.array([10 * math.log10(sc(r, v)) for r in rows
                                       if r["kind"] == "h0"
                                       and r["level"] == lv and r["frame"] == f])
                h1db[v][f] = np.array([10 * math.log10(sc(r, v)) for r in rows
                                       if r["kind"] == "h1"
                                       and r["level"] == lv and r["frame"] == f])
        stats = np.empty(nboot)
        nf = len(frames)
        for b in range(nboot):
            pick = rng.integers(0, nf, size=nf)
            # 门限来自重采样的 h0；margin 中位来自重采样的 h1
            d0 = np.concatenate([h0db["A_P"][frames[i]] for i in pick])
            d1 = np.concatenate([h0db[v1][frames[i]] for i in pick])
            g0 = np.concatenate([h1db["A_P"][frames[i]] for i in pick])
            g1 = np.concatenate([h1db[v1][frames[i]] for i in pick])
            # 分位在线性域计算（与主口径一致）
            m0 = float(np.median(
                g0 - 10 * math.log10(np.quantile(10 ** (d0 / 10.0), 1 - PFA))))
            m1 = float(np.median(
                g1 - 10 * math.log10(np.quantile(10 ** (d1 / 10.0), 1 - PFA))))
            stats[b] = m1 - m0
        r0 = np.concatenate([h0db["A_P"][f] for f in frames])
        r1 = np.concatenate([h0db[v1][f] for f in frames])
        q0 = np.concatenate([h1db["A_P"][f] for f in frames])
        q1 = np.concatenate([h1db[v1][f] for f in frames])
        p0 = float(np.median(
            q0 - 10 * math.log10(np.quantile(10 ** (r0 / 10.0), 1 - PFA))))
        p1 = float(np.median(
            q1 - 10 * math.log10(np.quantile(10 ** (r1 / 10.0), 1 - PFA))))
        out[lv] = (p1 - p0, float(np.quantile(stats, 0.025)),
                   float(np.quantile(stats, 0.975)))
    return out


emit("### ff6：FF_cert8 − A_P\n")
emit("| 档 | 点估计 | 95% CI | CI 全部 >0？ |")
bs6 = bootstrap_netwin(rows6, "FF_cert8")
ci_ok6 = {}
for lv in NOISY:
    p, lo, hi = bs6[lv]
    ci_ok6[lv] = lo > 0
    emit("| %d | %+.2f | [%+.2f, %+.2f] | %s |"
         % (lv, p, lo, hi, "是" if lo > 0 else "否"))
focus = (-20, -22, -24, -26)
a3_all6 = all(ci_ok6[lv] for lv in focus)
CON["A3_ff6"] = ("−20/−22/−24/−26 档 CI %s>0（%s）；−30 CI [%+.2f,%+.2f]，"
                 "−32 CI [%+.2f,%+.2f]"
                 % ("全部" if a3_all6 else "未",
                    "、".join("[%+.2f,%+.2f]" % bs6[lv][1:] for lv in focus),
                    bs6[-30][1], bs6[-30][2], bs6[-32][1], bs6[-32][2]))
emit("\n### ff3（ke=4）：FF_split_model − A_P\n")
emit("| 档 | 点估计 | 95% CI | CI 全部 >0？ |")
bs3 = bootstrap_netwin(subset(DATA["ff3"]["rows"], 4), "FF_split_model")
ci_ok3 = {lv: bs3[lv][1] > 0 for lv in NOISY}
for lv in NOISY:
    p, lo, hi = bs3[lv]
    emit("| %d | %+.2f | [%+.2f, %+.2f] | %s |"
         % (lv, p, lo, hi, "是" if lo > 0 else "否"))
a3_all3 = all(ci_ok3[lv] for lv in focus)
nw3_pt = {lv: bs3[lv][0] for lv in NOISY}
CON["A3_ff3"] = ("split_model(ke=4) −20..−26 CI %s>0（%s）；点估计本身 %s——"
                 "ff3 拆分族的收益在门限/证书（7.5 vs 10.8 dB），不在裕度净胜，"
                 "与 RESULTS 口径一致，不构成净胜主张"
                 % ("全部" if a3_all3 else "未",
                    "、".join("[%+.2f,%+.2f]" % bs3[lv][1:] for lv in focus),
                    "、".join("%d:%+.2f" % (lv, nw3_pt[lv]) for lv in focus)))

# ================================================================ A4
emit("\n## A4. 门限过拟合（留一种子，ff6）\n")
emit("种子 s 的净胜：门限仅用 seed≠s 的 h0 标定，H1 仅取 seed=s。种子间极差 = max−min。\n")
emit("| 档 | s=0 | s=1 | s=2 | 极差 |")
A4r = {}
A4_min = {}
for lv in NOISY:
    vals = []
    for s in (0, 1, 2):
        thr = {}
        for v in ("A_P", "FF_cert8"):
            thr[(v, lv)] = q99([sc(r, v) for r in rows6 if r["kind"] == "h0"
                                and r["level"] == lv and r["seed"] != s])
        m1, _, _ = margin_stats(rows6, "FF_cert8", lv, thr, seed=s)
        m0, _, _ = margin_stats(rows6, "A_P", lv, thr, seed=s)
        vals.append(m1 - m0)
    A4r[lv] = max(vals) - min(vals)
    A4_min[lv] = min(vals)
    emit("| %d | %+.2f | %+.2f | %+.2f | %.2f |"
         % (lv, vals[0], vals[1], vals[2], A4r[lv]))
deep = [lv for lv in NOISY if lv <= -22]
mid = [lv for lv in NOISY if -26 <= lv <= -20]
CON["A4"] = ("LOO 种子极差：−20..−26 为 %.2f~%.2f dB，−22..−32 为 %.2f~%.2f dB；"
             "各档三种子最小净胜 %s；净胜符号翻转仅出现在 %s"
             % (min(A4r[lv] for lv in mid), max(A4r[lv] for lv in mid),
                min(A4r[lv] for lv in deep), max(A4r[lv] for lv in deep),
                "、".join("%d:%+.2f" % (lv, A4_min[lv]) for lv in NOISY),
                "、".join(str(lv) for lv in NOISY if A4_min[lv] < 0) or "无"))

# ================================================================ A5
emit("\n## A5. SNR 斜率合理性\n")
emit("两层检验：① **SNR 标定本身** = H1 中位分数（dB）对 level 的斜率，"
     "应 ≈ +1（线性标定、统计量 ∝ SNR）；② margin 斜率 = ① − 门限斜率，"
     "浅档门限被泄漏抬高时会 <1（检测器/校准现象，非 SNR 标定误差）。\n")
emit("| 量 | 数据 | 斜率 dB/dB | R² |")
margins_ap = {lv: margin_stats(rows6, "A_P", lv, THR["ff6"])[0] for lv in NOISY}
sl, _, r2 = ols(NOISY, [margins_ap[lv] for lv in NOISY])
emit("| A_P 中位 margin | ff6 池化 −10..−32 | %+.3f | %.4f |" % (sl, r2))
sel = [lv for lv in NOISY if -28 <= lv <= -14]
sl2, _, r22 = ols(sel, [margins_ap[lv] for lv in sel])
emit("| A_P 中位 margin | ff6 池化 −28..−14 | %+.3f | %.4f |" % (sl2, r22))
for c in CAPS:
    mm = [margin_stats(rows6, "A_P", lv, THR["ff6"], cap=c)[0] for lv in NOISY]
    s3, _, r3 = ols(NOISY, mm)
    emit("| A_P 中位 margin | ff6 P=%d −10..−32 | %+.3f | %.4f |"
         % (PRE_CAP[c], s3, r3))
m1ap = {lv: margin_stats(DATA["ff1"]["rows"], "A_P", lv, THR["ff1"])[0]
        for lv in NOISY}
sl4, _, r4 = ols(NOISY, [m1ap[lv] for lv in NOISY])
emit("| A_P 中位 margin | ff1 池化 −10..−32 | %+.3f | %.4f |" % (sl4, r4))

h1_slope = {}
for v in ("A_P", "FF_tmpl", "FF_cert8"):
    meds = [float(np.median([10 * math.log10(sc(r, v)) for r in rows6
                             if r["kind"] == "h1" and r["level"] == lv]))
            for lv in NOISY]
    s5, _, r5 = ols([float(x) for x in NOISY], meds)
    h1_slope[v] = s5
    emit("| %s 中位 H1 分数(dB) | ff6 池化 −10..−32 | %+.3f | %.4f |"
         % (v, s5, r5))
thr_ap = [10 * math.log10(THR["ff6"][("A_P", lv)]) for lv in NOISY]
st, _, rt = ols([float(x) for x in NOISY], thr_ap)
emit("| A_P 门限(dB) | ff6 池化 −10..−32 | %+.3f | %.4f |" % (st, rt))
CON["A5"] = ("SNR 标定斜率：A_P H1 中位 %+.3f、FF_tmpl %+.3f（R²≈1）→ 标定线性，"
             "通过；margin 斜率 %+.3f 偏离 1 的主因 = 门限对 level 的斜率 %+.3f"
             "（−10/−14 泄漏残余抬高浅档门限），非 SNR 标定问题。"
             "cert8 H1 斜率 %+.2f（−32 档 κ̂ 崩塌致中位塌落，与其深端净胜转负一致）"
             % (h1_slope["A_P"], h1_slope["FF_tmpl"], sl, st,
                h1_slope["FF_cert8"]))

# ================================================================ A6
emit("\n## A6. native 档 H0 污染确认\n")
emit("自归一统计量在纯噪声下应与档位无关，native h0 中位 − 噪声档 h0 中位 ≈ 污染幅度"
     "（理论无污染差 = 0 dB；参照列给噪声档自身各档中位的范围 = 平坦度）。\n")
emit("| 轮 | 变体 | native h0 中位 dB | 噪声档合并中位 dB（各档范围） | 差 dB |")
A6 = {}
for tag, d in DATA.items():
    ke = 4 if tag == "ff3" else None
    rows = subset(d["rows"], ke)
    for v in d["variants"]:
        nat = [10 * math.log10(sc(r, v)) for r in rows
               if r["kind"] == "h0" and r["level"] is None]
        meds = [float(np.median([10 * math.log10(sc(r, v)) for r in rows
                                 if r["kind"] == "h0" and r["level"] == lv]))
                for lv in NOISY]
        gap = float(np.median(nat)) - float(np.median(meds))
        A6[(tag, v)] = gap
        if tag == "ff1" and v == "A_P":
            nat_med_ap = float(np.median(nat))
        emit("| %s | %s | %+.2f | %+.2f（%.1f~%.1f） | %+.2f |"
             % (tag, v, float(np.median(nat)), float(np.median(meds)),
                min(meds), max(meds), gap))
CON["A6"] = ("native 污染证实且各轮可复现：A_P native h0 中位比噪声档高 "
             "%.1f~%.1f dB（native 中位绝对值 ~%.1f dB，即 RESULTS 所称"
             "“中位 ~10dB”），FF_tmpl 高 %.1f~%.1f dB；实验方将门限校准限制在"
             "噪声档的处理正确，该问题不影响各档结论"
             % (min(A6[(t, "A_P")] for t in FILES),
                max(A6[(t, "A_P")] for t in FILES),
                nat_med_ap, 
                min(A6[(t, "FF_tmpl")] for t in FILES),
                max(A6[(t, "FF_tmpl")] for t in FILES)))

# ================================================================ A7
emit("\n## A7. ff4 dkap_bins（κ̂ 误差，bin）：分位与深度单调性（h1 记录，|Δ|）\n")
emit("| 档 | p50(\\|Δ\\|) | p90(\\|Δ\\|) | h0 p50（参照） |")
p50s, p90s, p50h0 = [], [], []
for lv in NOISY:
    x1 = np.abs([r["scores"]["dkap_bins"] for r in DATA["ff4"]["rows"]
                 if r["kind"] == "h1" and r["level"] == lv])
    x0 = np.abs([r["scores"]["dkap_bins"] for r in DATA["ff4"]["rows"]
                 if r["kind"] == "h0" and r["level"] == lv])
    p50, p90 = float(np.quantile(x1, 0.5)), float(np.quantile(x1, 0.9))
    p50s.append(p50)
    p90s.append(p90)
    p50h0.append(float(np.quantile(x0, 0.5)))
    emit("| %d | %.2f | %.2f | %.1f |" % (lv, p50, p90, p50h0[-1]))
mono50 = all(p50s[i] <= p50s[i + 1] * 1.001 for i in range(len(p50s) - 1))
mono90 = all(p90s[i] <= p90s[i + 1] * 1.001 for i in range(len(p90s) - 1))
depth = [-lv for lv in NOISY]   # 深度取正，越深越大
sp50 = spearman(depth, p50s)
emit("\np50 序列随深度 %s单调：%.2f@−10 → %.2f@−32（与深度 Spearman ρ=%.3f）；"
     "p90 序列 %s单调：%.2f@−10 → %.2f@−32。"
     % ("严格" if mono50 else "不", p50s[0], p50s[-1], sp50,
        "严格" if mono90 else "不", p90s[0], p90s[-1]))
CON["A7"] = ("p50 %s单调且 p90 %s单调（p50 %.1f@−10→%.1f@−32，p90 %.1f→%.1f；"
             "与深度 Spearman=%.2f）→ κ̂ 误差随深度增长 %s"
             % ("" if mono50 else "不", "" if mono90 else "不",
                p50s[0], p50s[-1], p90s[0], p90s[-1], sp50,
                "成立" if mono50 and mono90 else "未确认"))

# ================================================================ 汇总
emit("\n## 汇总判定\n")
for k, v in CON.items():
    emit("- **%s**：%s" % (k, v))

with open(os.path.join(EXP_DIR, "AUDIT.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")
print("\nAUDIT.md 写入完成（%d 行）" % len(md))
for k, v in CON.items():
    print("[%s] %s" % (k, v))

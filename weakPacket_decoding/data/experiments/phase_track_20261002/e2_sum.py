# -*- coding: utf-8 -*-
"""E2 汇总：①E1 遗产聚合表 ②三臂 SER/PER 配对表 ③相位红利与 C 捕获率
④预注册判定 + 失败模式诊断。输出 e2_summary.json + 控制台战表。"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C

ROOT = C.ROOT
ARMS = ("A0", "A", "B", "C")


def e1_legacy_table():
    r = json.load(open(os.path.join(ROOT, "e1_snr_results.json"), encoding="utf-8"))
    rows = {}
    hdr = ["native"] + [str(s) for s in C.SNR_LEVELS]
    keys = [("前导测量地板 l0_pre", "l0_pre"),
            ("前导 M1 拟合 genie_pre_m1", "genie_pre_m1"),
            ("全帧 M2c+kind genie_pre", "genie_pre"),
            ("genie_pay(GT 未建模 det)", "genie_pay"),
            ("因果 k=8 causal8_pre", "causal8_pre"),
            ("因果 k=8 causal8_pay", "causal8_pay"),
            ("γ_PCM 地板", "gamma_pcm_sigma"),
            ("V1V2 y 地板", "y_sigma")]
    out = {}
    for lab, k in keys:
        out[lab] = {h: (r.get(h, {}).get(k, {}) or {}).get("med") for h in hdr}
    kap = {h: r.get(h, {}).get("kap_wls_rmse") for h in hdr}
    out["κ̂_WLS RMSE"] = kap
    return out


def paired_table():
    recs = []
    for l in open(os.path.join(ROOT, "e2_units.jsonl"), encoding="utf-8"):
        recs.append(json.loads(l))
    recs = [r for r in recs if "skip" not in r]
    lvls = ["native"] + [str(s) for s in C.SNR_LEVELS]
    table = {}
    for lab in lvls:
        sub = [r for r in recs
               if (("native" if r["snr"] is None else str(r["snr"])) == lab)]
        if not sub:
            continue
        d = {"n_units": len(sub)}
        tot = sum(r["sym_tot"] for r in sub)
        for a in ARMS:
            ser = sum(r[a]["sym_err"] for r in sub) / max(tot, 1)
            per = sum(r[a]["crc_fail"] for r in sub) / max(len(sub), 1)
            d[a] = dict(SER=ser, PER=per)
        # 配对 SER 差（每单元差再平均，同噪声实现配对）
        for a, b in (("B", "A"), ("C", "A"), ("C", "A0"), ("B", "A0"),
                     ("A", "A0"), ("C", "B")):
            dd = [ (r[a]["sym_err"] - r[b]["sym_err"]) / r["sym_tot"] for r in sub]
            d["dSER_%s-%s" % (a, b)] = float(np.mean(dd))
            dp = [r[a]["crc_fail"] - r[b]["crc_fail"] for r in sub]
            d["dPER_%s-%s" % (a, b)] = float(np.mean(dp))
            d["win_%s-%s" % (a, b)] = int(sum(1 for x in dd if x < 0))
            d["lose_%s-%s" % (a, b)] = int(sum(1 for x in dd if x > 0))
        ph = np.array([r["phase_rmse"] for r in sub if r.get("phase_rmse") == r.get("phase_rmse")])
        d["phase_rmse_med"] = float(np.median(ph)) if len(ph) else None
        pc = np.array([r["phase_post_conc"] for r in sub])
        d["phase_post_conc_med"] = float(np.median(pc))
        gc = np.array([r["genie_conc"] for r in sub])
        d["genie_conc_med"] = float(np.median(gc))
        # τ̂ 命中率（|τ̂_C − τ̂_B| ≤ 0.2）
        hit = [abs(r["tau_hat"] - r["tau_hat_B"]) <= 0.2 for r in sub]
        d["tau_hit_rate"] = float(np.mean(hit))
        table[lab] = d
    return table


def threshold_shift(table, arm_c, arm_b, target=0.10):
    """10% SER 门限移位（dB）：线性插值 SER(SNR) 交 10%。"""
    xs, ys = [], []
    for lab in [str(s) for s in C.SNR_LEVELS]:
        if lab in table:
            xs.append(float(lab))
            ys.append(table[lab][arm_c]["SER"] if False else table[lab][arm_c]["SER"])
    # arm_c 为 'C' 等：直接用 SER
    def cross(x, y):
        for i in range(len(y) - 1):
            if y[i] >= target >= y[i + 1] or (y[i] >= target and y[i + 1] < target):
                if y[i] == y[i + 1]:
                    continue
                t = (y[i] - target) / (y[i] - y[i + 1])
                return x[i] + t * (x[i + 1] - x[i])
        return None
    a = cross(xs, [table[str(int(x))][arm_c]["SER"] for x in xs])
    b = cross(xs, [table[str(int(x))][arm_b]["SER"] for x in xs])
    return a, b, (b - a if (a is not None and b is not None) else None)


def main():
    e1t = e1_legacy_table()
    tab = paired_table()
    # 判定
    thr = {}
    for pair in (("C", "A"), ("C", "A0"), ("B", "A"), ("B", "A0")):
        thr["%s-%s" % pair] = threshold_shift(tab, pair[0], pair[1])[2]
    # C 捕获率 = dSER(C−A) / dSER(B−A)（正增益档）
    cap = {}
    for lab in tab:
        if lab == "native":
            continue
        ba = tab[lab]["dSER_B-A"]
        ca = tab[lab]["dSER_C-A"]
        cap[lab] = ca / ba if ba < -1e-4 else None
    summary = dict(e1_legacy=e1t, table=tab, threshold_shift_db=thr,
                   capture_rate=cap)
    json.dump(summary, open(os.path.join(ROOT, "e2_summary.json"), "w",
                            encoding="utf-8"), indent=1, ensure_ascii=False)
    # 控制台
    print("=== E1 遗产聚合（中位，rad）===")
    for lab, row in e1t.items():
        print("%-28s" % lab, {k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in row.items()})
    print("\n=== E2 三臂战表（SER/PER，n=单元数）===")
    for lab, d in tab.items():
        print("[%7s] n=%4d  " % (lab, d["n_units"]) + "  ".join(
            "%s %.4f/%.3f" % (a, d[a]["SER"], d[a]["PER"]) for a in ARMS)
            + "  phRMSE=%.2f τhit=%.2f" % (d["phase_rmse_med"] or -1,
                                           d["tau_hit_rate"]))
        print("            配对 dSER: C−A %+.4f  B−A %+.4f  C−B %+.4f  A−A0 %+.4f  "
              "C−A0 %+.4f (win/lose C−A %d/%d)" % (
                  d["dSER_C-A"], d["dSER_B-A"], d["dSER_C-B"], d["dSER_A-A0"],
                  d["dSER_C-A0"], d["win_C-A"], d["lose_C-A"]))
    print("\n10%% SER 门限移位 (dB):", thr)
    print("C 捕获率 (C−A)/(B−A):", cap)


if __name__ == "__main__":
    main()

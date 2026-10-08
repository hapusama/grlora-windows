# -*- coding: utf-8 -*-
"""E3 汇总：①配对验证（A/B vs e2_units）②战表（全集+公平域 19 帧）
③门限移位/捕获率 ④迭代健康度 ⑤归因分解（κ-line vs 相位）⑥预注册判定。
输出 e3_summary.json + 控制台战表。"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_common as C
import e3_common as D3

ROOT = C.ROOT
ARMS = ("A", "B", "D", "DA", "DNP", "E8", "E16")
FAIR19 = {0, 1, 2, 3, 4, 6, 10, 11, 12, 13, 14, 15, 16, 20, 22, 23, 24, 25, 26}


def lab(r):
    return "native" if r["snr"] is None else str(r["snr"])


def load():
    recs = [json.loads(l) for l in open(os.path.join(ROOT, "e3_units.jsonl"),
                                        encoding="utf-8")]
    return [r for r in recs if "skip" not in r]


def paired_verify(recs):
    e2 = {}
    for l in open(os.path.join(ROOT, "e2_units.jsonl"), encoding="utf-8"):
        rr = json.loads(l)
        e2[(rr["gid"], rr["snr"], rr["seed"])] = rr
    n = okA = okB = 0
    for r in recs:
        ref = e2.get((r["gid"], r["snr"], r["seed"]))
        if ref is None:
            continue
        n += 1
        okA += int(r["A"]["sym_err"] == ref["A"]["sym_err"]
                   and r["A"]["crc_fail"] == ref["A"]["crc_fail"])
        okB += int(r["B"]["sym_err"] == ref["B"]["sym_err"]
                   and r["B"]["crc_fail"] == ref["B"]["crc_fail"])
    return dict(n_checked=n, A_bitexact=okA, B_bitexact=okB)


def table(recs, dom):
    lvls = ["native"] + [str(s) for s in C.SNR_LEVELS]
    tab = {}
    for lv in lvls:
        sub = [r for r in recs if lab(r) == lv and (dom is None or r["gid"] in dom)]
        if not sub:
            continue
        d = {"n_units": len(sub)}
        tot = sum(r["sym_tot"] for r in sub)
        for a in ARMS:
            ser = sum(r[a]["sym_err"] for r in sub) / max(tot, 1)
            per = sum(r[a]["crc_fail"] for r in sub) / max(len(sub), 1)
            d[a] = dict(SER=ser, PER=per)
        for a in ("D", "DA", "DNP", "E8", "E16", "B"):
            dd = [(r[a]["sym_err"] - r["A"]["sym_err"]) / r["sym_tot"] for r in sub]
            dp = [r[a]["crc_fail"] - r["A"]["crc_fail"] for r in sub]
            d["dSER_%s-A" % a] = float(np.mean(dd))
            d["dPER_%s-A" % a] = float(np.mean(dp))
            d["win_%s-A" % a] = int(sum(1 for x in dd if x < 0))
            d["lose_%s-A" % a] = int(sum(1 for x in dd if x > 0))
        # 健康度：D 末轮 τ̂ 命中（vs probe5 GT）、gate、轮次
        hits = []
        gates = []
        rounds_fin = []
        loores = []
        for r in sub:
            dg = r.get("D_diag") or {}
            rounds = dg.get("rounds") or []
            tr = [x for x in rounds if x.get("tau") is not None]
            if tr:
                hits.append(int(abs(tr[-1]["tau"] - D3.TAU_GT[r["gid"]]) <= 0.2))
            if "gate_rate" in dg:
                gates.append(dg["gate_rate"])
            if rounds:
                rounds_fin.append(len(rounds) - 1)
            if "loo_res" in dg:
                loores.append(dg["loo_res"])
        d["tau_hit_gt"] = float(np.mean(hits)) if hits else None
        d["gate_med"] = float(np.median(gates)) if gates else None
        d["loo_res_med"] = float(np.median(loores)) if loores else None
        d["D_rounds_dist"] = {int(k): int(v) for k, v in zip(
            *np.unique(rounds_fin, return_counts=True))} if rounds_fin else None
        tab[lv] = d
    return tab


def threshold(tab, arm, target=0.10):
    """10% SER 门限（dB）：SER 随档位加深而**上升**，找上升穿越点。"""
    xs = [float(s) for s in C.SNR_LEVELS if str(s) in tab]
    ys = [tab[str(int(x))][arm]["SER"] for x in xs]
    for i in range(len(ys) - 1):
        if ys[i] < target <= ys[i + 1]:
            t = (target - ys[i]) / (ys[i + 1] - ys[i])
            return xs[i] + t * (xs[i + 1] - xs[i])
    if ys[-1] < target:
        return "beyond-%d" % abs(int(xs[-1]))        # 比最深档还深
    return None


def capture(tab, lv, arm="D"):
    ba = tab[lv]["dSER_B-A"]
    da = tab[lv]["dSER_%s-A" % arm]
    return da / ba if ba < -1e-4 else None


def main():
    recs = load()
    pv = paired_verify(recs)
    print("=== ①配对验证 ===")
    print(json.dumps(pv))
    out = dict(paired_verify=pv)
    for dom, nm in ((None, "ALL28"), (FAIR19, "FAIR19")):
        tab = table(recs, dom)
        out[nm] = tab
        print("\n=== ②%s 战表（SER/PER）===" % nm)
        for lv, d in tab.items():
            print("[%7s] n=%4d " % (lv, d["n_units"]) + "  ".join(
                "%s %.4f/%.3f" % (a, d[a]["SER"], d[a]["PER"]) for a in ARMS))
            print("            dSER: B−A %+.4f  D−A %+.4f  DA−A %+.4f  DNP−A %+.4f "
                  "E8−A %+.4f  E16−A %+.4f | D w/l %d:%d  τhit=%.2f gate=%.2f 轮次=%s"
                  % (d["dSER_B-A"], d["dSER_D-A"], d["dSER_DA-A"], d["dSER_DNP-A"],
                     d["dSER_E8-A"], d["dSER_E16-A"], d["win_D-A"], d["lose_D-A"],
                     d["tau_hit_gt"] if d["tau_hit_gt"] is not None else -1,
                     d["gate_med"] if d["gate_med"] is not None else -1,
                     d["D_rounds_dist"]))
        th = {a: threshold(tab, a) for a in ARMS}
        out[nm + "_threshold10"] = th
        print("  10%%SER 门限:", {a: (round(v, 2) if isinstance(v, float) else v)
                               for a, v in th.items()})
        out[nm + "_capture"] = {lv: capture(tab, lv) for lv in tab if lv != "native"}
        print("  捕获率 (X−A)/(B−A):", {k: (round(v, 3) if v else v)
                                    for k, v in out[nm + "_capture"].items()})
    # ⑤归因（公平域 −22/−24 深端）
    att = {}
    for dom, nm in ((FAIR19, "FAIR19"),):
        for lv in ("-22", "-24"):
            sub = [r for r in recs if lab(r) == lv and r["gid"] in dom]
            if not sub:
                continue
            tot = sum(r["sym_tot"] for r in sub)
            att[lv] = {
                "A": sum(r["A"]["sym_err"] for r in sub) / tot,
                "D": sum(r["D"]["sym_err"] for r in sub) / tot,
                "DNP": sum(r["DNP"]["sym_err"] for r in sub) / tot,
                "B": sum(r["B"]["sym_err"] for r in sub) / tot,
                "E8": sum(r["E8"]["sym_err"] for r in sub) / tot,
                "E16": sum(r["E16"]["sym_err"] for r in sub) / tot,
                "kap_gain(DNP−A)": sum(r["DNP"]["sym_err"] for r in sub) / tot
                - sum(r["A"]["sym_err"] for r in sub) / tot,
                "phase_gain(D−DNP)": sum(r["D"]["sym_err"] for r in sub) / tot
                - sum(r["DNP"]["sym_err"] for r in sub) / tot,
            }
    out["attribution"] = att
    print("\n=== ⑤归因分解（FAIR19）===")
    for lv, a in att.items():
        print(lv, {k: round(v, 4) for k, v in a.items()})
    # ⑥健康度补充：逐轮 SER、jitter 相关、D 每档 PER 捕获
    for lv in ("-22", "-24"):
        sub = [r for r in recs if lab(r) == lv and r["gid"] in FAIR19]
        if not sub:
            continue
        for it in (0, 1, 2):
            sers = [r["D_diag"]["rounds"][it]["SER"] for r in sub
                    if len(r["D_diag"]["rounds"]) > it]
            if sers:
                print("  FAIR19 %s D round%d SER(mean)=%.4f (n=%d)"
                      % (lv, it, float(np.mean(sers)), len(sers)))
        # jitter vs D−A 相关
        dd = {}
        for r in sub:
            dd.setdefault(D3.INC_COR[r["gid"]], []).append(
                (r["D"]["sym_err"] - r["A"]["sym_err"]) / r["sym_tot"])
        js = sorted(dd)
        print("  jitter 分桶 dSER(D−A):",
              {("%.2f" % j): round(float(np.mean(dd[j])), 4) for j in js})
    json.dump(out, open(os.path.join(ROOT, "e3_summary.json"), "w",
                        encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n写入 e3_summary.json")


if __name__ == "__main__":
    main()

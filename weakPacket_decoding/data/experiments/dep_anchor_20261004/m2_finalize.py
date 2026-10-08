# -*- coding: utf-8 -*-
"""M2 收尾：跑 ROC 并打印战表/配对/锚/δ̂ 摘要（供 RESULTS.md 填数）。"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"D:\mysoft2\miniconda3\envs\gr-lora\python.exe"


def main():
    subprocess.run([PY, os.path.join(HERE, "m2_roc.py")], check=True)
    r = json.load(open(os.path.join(HERE, "m2_roc_results.json")))
    print("\n==== Pd09 战表（FAR=1e-3）====")
    for k, v in r["pd09_table"].items():
        print("%s | cert %.2f | dep-old %.2f | dera %.2f | dep2 %.2f | dep2c %.2f"
              " || Δdep2−dera %+.2f | Δdep2c−dera %+.2f | Δcert−dera %+.2f"
              " | thr: dep2 %.2f dep2c %.2f dera %.2f"
              % (k, v["cert"], v["dep"], v["dera"], v["dep2"], v["dep2c"],
                 v["dep2_minus_dera"], v["dep2c_minus_dera"],
                 v["cert_minus_dera"], v["thr_dep2_db"], v["thr_dep2c_db"],
                 v["thr_dera_db"]))
    print("\n==== 配对 w/l（vs dera）====")
    for k, v in r["paired_wl"].items():
        print("%s | dep2 %dW/%dL | dep2c %dW/%dL" % (k, v["win"], v["loss"],
                                                     v["dep2c_win"],
                                                     v["dep2c_loss"]))
    if "pd09_byP_delta0" in r:
        print("\n==== per-P（δ=0）====")
        print(json.dumps(r["pd09_byP_delta0"], indent=0))
    print("\n==== H0 不变性 ====")
    print(json.dumps(r["h0_invariance"], indent=0))
    print("\n==== 锚（battle H1 记录）====")
    print(json.dumps(r.get("anchor", {}), indent=0))
    print("\n==== δ̂ 分布 ====")
    print(json.dumps(r.get("dhat_dist", {}), indent=0))


if __name__ == "__main__":
    main()

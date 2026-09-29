"""Plot coset transfer results and quantify paired differences."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/"data/experiments/paper_audit_20260911/coset_sync"


def read(name):
    with (ROOT/name).open(newline="") as f:return list(csv.DictReader(f))


def main():
    rows=read("summary.csv")
    trials=read("joint_known_trials.csv")
    pairs=[]
    for snr in [-18,-14,-10]:
        proposed={r["trial"]:r for r in trials if r["method"]=="optimized" and int(r["snr_db"])==snr}
        for method in ["uniform","fixed_03","random_1"]:
            baseline={r["trial"]:r for r in trials if r["method"]==method and int(r["snr_db"])==snr}
            fixes=sum(int(proposed[k]["correct"]) and not int(baseline[k]["correct"]) for k in proposed)
            breaks=sum(not int(proposed[k]["correct"]) and int(baseline[k]["correct"]) for k in proposed)
            pairs.append(dict(snr_db=snr,baseline=method,fixes=fixes,breaks=breaks,
                mcnemar_exact_p=binomtest(fixes,fixes+breaks).pvalue if fixes+breaks else 1.0))
    (ROOT/"paired_comparisons.json").write_text(json.dumps(pairs,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,pilot in zip(axes,["preamble_only","preamble_equal_energy","joint_known"]):
        for method in ["full","uniform","fixed_03","random_1","optimized"]:
            rr=[r for r in rows if r["pilot"]==pilot and r["method"]==method]
            ax.plot([int(r["snr_db"]) for r in rr],[int(r["correct"])/int(r["trials"]) for r in rr],"o-",label=method)
        ax.set(xlabel="Injected ADC-white SNR (dB)",ylabel="Correct delay AND CFO fraction",
            title=pilot.replace("_"," ")+"; synthetic SF7",ylim=(0,1.02))
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle("Fixed L=8, K=2 cosets: offline design, 256 held-out states per SNR")
    fig.tight_layout();fig.savefig(ROOT/"coset_sync.png",dpi=170);fig.savefig(ROOT/"coset_sync.svg")
    print(json.dumps(pairs,indent=2))


if __name__=="__main__":main()

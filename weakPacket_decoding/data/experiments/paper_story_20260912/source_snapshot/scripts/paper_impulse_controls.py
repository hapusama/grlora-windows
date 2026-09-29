"""Reconstruct paired observations and test simple clipping/blanking controls.

Threshold is fixed at 2.5 times median-radius Rayleigh scale; no labels are
used for scale, threshold, clipping, blanking, or symbol selection.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weak_decoder.os_lora.experiments.archive import evaluate_robust_sparse_demod as ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"data/experiments/paper_audit_20260911/robust_impulse")
    args=parser.parse_args()
    with (args.source/"symbols.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    datasets=list(ref.DEFAULT_DATASETS)
    samples={name:np.memmap(ref.dataset_paths(name)[0],dtype=np.complex64,mode="r") for name in datasets}
    packets={name:{int(p["packet_index"]):p for p in ref.load_packets(ref.dataset_paths(name)[1])} for name in datasets}
    output=[]
    for row in rows:
        name=row["dataset"]
        di=datasets.index(name)
        packet=packets[name][int(row["packet_index"])]
        sf,osr=int(row["sf"]),int(row["os_factor"])
        start=int(row["demod_start_sample"])
        clean=np.asarray(samples[name][start:start+(1<<sf)*osr])
        seed_args=(int(row["seed"]),di,int(row["packet_index"]),int(row["payload_symbol_index"]))
        noisy=ref.noise_samples(clean=clean,snr_db=float(row["snr_db"]),
            seed=ref._trial_seed(*seed_args,stream=1),signal_reference_power=float(row["signal_reference_power"]),
            noise_shape="white",os_factor=osr)
        observed,_=ref._add_sparse_impulses(samples=noisy,signal_power=float(row["signal_reference_power"]),
            fraction=float(row["impulse_fraction"]),isr_db=float(row["impulse_isr_db"] or 35),
            layout="random",seed=ref._trial_seed(*seed_args,stream=2))
        radius=np.abs(observed)
        threshold=2.5*float(np.median(radius))/math.sqrt(math.log(2))
        clipped=observed*np.minimum(1,threshold/np.maximum(radius,1e-30))
        blanked=np.where(radius>threshold,0,observed)
        result={k:row[k] for k in ["dataset","seed","packet_index","payload_symbol_index","snr_db","gt_bin","savaux_error","robust_error"]}
        for method,values in [("verify",observed),("clip",clipped),("blank",blanked)]:
            tick=time.perf_counter()
            spectrum,*_=ref.paper_oversampled_spectrum(samples=values,start_sample=0,sf=sf,os_factor=osr,
                cfo_int=int(packet["cfo_int"]),cfo_frac=float(packet["cfo_frac"]),header_start_sample=0,cfo_correction_mode="symbol")
            selected=int(np.argmax(np.abs(spectrum)**2))
            if method=="verify":
                assert selected==int(row["savaux_bin"]), (name,selected,row["savaux_bin"])
            else:
                result[method+"_error"]=int(selected!=int(row["gt_bin"]))
                result[method+"_seconds"]=time.perf_counter()-tick
        output.append(result)
    with (args.source/"simple_controls.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    summary={method:sum(int(r[method+"_error"]) for r in output) for method in ["savaux","robust","clip","blank"]}
    summary.update(symbols=len(output),threshold_scale=2.5,all_paired_savaux_decisions_verified=True)
    (args.source/"simple_controls_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()

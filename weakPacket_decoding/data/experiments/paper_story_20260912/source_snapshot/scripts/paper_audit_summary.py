"""Aggregate frozen audit CSVs and render standalone evidence plots."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/"data/experiments/paper_audit_20260911"


def read(path):
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    rows=read(ROOT/"acquisition/trials.csv")
    detectors=sorted(set(r["detector"] for r in rows))
    gated=[]
    for det in detectors:
        calibration=[r for r in rows if r["kind"]=="noise" and int(r["seed"])<92064 and r["detector"]==det]
        threshold=max(float(r["max_event_share"]) for r in calibration)
        test=[r for r in rows if r["kind"]=="noise" and int(r["seed"])>=92064 and r["detector"]==det]
        false=sum(float(r["max_event_share"])>threshold for r in test)
        for snr in [-32,-30,-28,-26]:
            trials=[r for r in rows if r["kind"]=="ota" and float(r["channel_snr_db"])==snr and r["detector"]==det]
            gated.append(dict(detector=det,channel_snr_db=snr,threshold=threshold,
                noise_calibration_trials=len(calibration),noise_test_trials=len(test),noise_test_false_trials=false,
                ota_trials=len(trials),raw_hits=sum(int(r["true_basin_detected"]) for r in trials),
                gated_hits=sum(float(r["max_true_share"])>threshold for r in trials)))
    write(ROOT/"acquisition/gated_summary.csv",gated)
    sparse=read(ROOT/"sparse_sampling/trials.csv")
    methods=[k[:-6] for k in sparse[0] if k.endswith("_error")]
    sparse_summary=[]
    for snr in ["clean","-10","-16","-22"]:
        rr=[r for r in sparse if r["snr_db"]==snr]
        for method in methods:
            sparse_summary.append(dict(snr_db=snr,method=method,symbols=len(rr),
                errors=sum(int(r[method+"_error"]) for r in rr),
                ser=np.mean([int(r[method+"_error"]) for r in rr])))
    write(ROOT/"sparse_sampling/summary.csv",sparse_summary)
    packets=read(ROOT/"sparse_sampling/packet_trials.csv")
    packet_summary=[]
    for snr in ["clean","-10","-16","-22"]:
        for method in methods:
            rr=[r for r in packets if r["snr_db"]==snr and r["method"]==method]
            packet_summary.append(dict(snr_db=snr,method=method,trials=len(rr),
                crc_pass=sum(int(r["crc_valid"]) for r in rr),
                exact_payload_match=sum(int(r["exact_payload_match"]) for r in rr),
                crc_and_exact=sum(int(r["crc_valid"]) and int(r["exact_payload_match"]) for r in rr)))
    write(ROOT/"sparse_sampling/packet_summary.csv",packet_summary)
    impulse=[]
    for scenario in ["robust_impulse","robust_awgn"]:
        rr=read(ROOT/scenario/"simple_controls.csv")
        for snr in sorted(set(r["snr_db"] for r in rr)):
            selected=[r for r in rr if r["snr_db"]==snr]
            for method in ["savaux","robust","clip","blank"]:
                impulse.append(dict(scenario=scenario,snr_db=snr,method=method,symbols=len(selected),
                    errors=sum(int(r[method+"_error"]) for r in selected),
                    ser=np.mean([int(r[method+"_error"]) for r in selected])))
    write(ROOT/"impulse_summary.csv",impulse)
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    names={"legacy_m4":"Noncoherent M=4","noncoherent_m12":"Noncoherent M=12",
        "chip_phase_m12":"Coherent 125k","uniform_250k_m12":"Coherent 250k","virtual_simo_m12":"Coherent 1M"}
    for det,label in names.items():
        rr=[r for r in gated if r["detector"]==det]
        axes[0].plot([r["channel_snr_db"] for r in rr],[r["raw_hits"]/r["ota_trials"] for r in rr],"o-",label=label)
    axes[0].set(xlabel="Injected channel SNR (dB)",ylabel="Preamble basin hit rate",title="8 OTA packets x 8 new noise seeds")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.25)
    impulse_rows=[r for r in impulse if r["scenario"]=="robust_impulse"]
    axes[1].bar([r["method"] for r in impulse_rows],[100*r["ser"] for r in impulse_rows],color=["#777777","#dc813a","#4090ba","#42996d"])
    axes[1].set(ylabel="Symbol disagreement (%)",title="4% impulses, ISR 35 dB; n=2940")
    for i,r in enumerate(impulse_rows):axes[1].text(i,100*r["ser"]+.15,str(r["errors"]),ha="center")
    for method in ["uniform_256","nonuniform_256","nonuniform_512","uniform_1024","full_4096"]:
        rr=[r for r in sparse_summary if r["method"]==method and r["snr_db"]!="clean"]
        axes[2].plot([float(r["snr_db"]) for r in rr],[r["ser"] for r in rr],"o-",label=method)
    axes[2].set(xlabel="Injected ADC-white SNR (dB)",ylabel="Symbol disagreement rate",title="Oracle-sync sampling budget; n=2940/point")
    axes[2].legend(fontsize=8);axes[2].grid(alpha=.25)
    fig.tight_layout();fig.savefig(ROOT/"evidence.png",dpi=170);fig.savefig(ROOT/"evidence.svg")
    fig2,ax=plt.subplots(figsize=(8.5,5))
    for method in ["uniform_256","nonuniform_256","nonuniform_512","uniform_1024","full_4096"]:
        rr=[r for r in packet_summary if r["method"]==method and r["snr_db"]!="clean"]
        ax.plot([float(r["snr_db"]) for r in rr],[r["crc_and_exact"]/r["trials"] for r in rr],"o-",label=method)
    ax.set(xlabel="Injected ADC-white SNR (dB)",ylabel="CRC-valid exact payload recovery fraction",
        title="Conditional recovery: clean timing, CFO and header supplied\n28 OTA packets x 3 new noise seeds; not end-to-end PDR")
    ax.legend();ax.grid(alpha=.25);fig2.tight_layout();fig2.savefig(ROOT/"payload_recovery.png",dpi=170)
    print(json.dumps(dict(gated=gated,sparse=sparse_summary,impulse=impulse),indent=2))


if __name__=="__main__":main()

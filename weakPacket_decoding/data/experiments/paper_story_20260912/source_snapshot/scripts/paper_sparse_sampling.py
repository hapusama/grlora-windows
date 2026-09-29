"""Equal-observation-budget uniform/random sampling on frozen OTA symbols.

All 1024 symbol candidates are scored using only selected sample values.
Timing and CFO are supplied by clean synchronization: this is an oracle-sync
demodulation feasibility experiment, not a sub-Nyquist packet receiver.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
from pathlib import Path
import csv
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weak_decoder.baselines import common
from weak_decoder.os_lora.system.nonuniform_sampling import prepare_dechirped_symbol
from weak_decoder.decoding.phase_templates import dechirped_candidate_template
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols
from weak_decoder.decoding.header_first_demod import bin_to_grlora_symbol


def main():
    out=ROOT/"data/experiments/paper_audit_20260911/sparse_sampling"
    out.mkdir(parents=True,exist_ok=True)
    sf,r=10,4
    n=1<<sf
    templates=np.stack([dechirped_candidate_template(sf,r,k) for k in range(n)])
    rng=np.random.default_rng(91400)
    patterns={}
    for count in [256,512,1024]:
        patterns[f"uniform_{count}"]=np.arange(count)*(n*r//count)
        patterns[f"nonuniform_{count}"]=np.sort(rng.choice(n*r,count,replace=False))
    patterns["full_4096"]=np.arange(n*r)
    banks={name:np.conj(templates[:,idx]).copy() for name,idx in patterns.items()}
    # Full matched-filter implementation validation against noiseless templates.
    sanity={}
    for name,bank in banks.items():
        scores=np.abs(bank@templates[:,patterns[name]].T)**2
        correct=np.diag(scores)
        ties=np.sum(scores>=correct[:,None]*(1-1e-5),axis=1)
        sanity[name]=dict(mean_equivalent_candidates=float(ties.mean()),max_equivalent_candidates=int(ties.max()))
        if name.startswith("nonuniform"):
            assert np.all(ties==1)
    results=[]
    packet_results=[]
    for di,dataset in enumerate(common.DEFAULT_DATASETS):
        iq_path,csv_path=common.dataset_paths(dataset)
        iq=np.memmap(iq_path,dtype=np.complex64,mode="r")
        packets=common.load_packets(csv_path)
        power,*_=common.signal_reference_power(samples=iq,packets=packets,mode="payload",explicit_power=None)
        for snr,noise_seed in [(None,91401)]+[(s,t) for s in [-10,-16,-22] for t in [91401,91402,91403]]:
            columns=[]; labels=[]
            for packet in packets:
                for item in packet["payload_symbols"]:
                    start=int(item["start_sample"])+r//2
                    clean=np.asarray(iq[start:start+n*r])
                    if len(clean)!=n*r: continue
                    seed=int(np.random.SeedSequence([noise_seed,di,int(packet["packet_index"]),int(item["payload_symbol_index"])]).generate_state(1)[0])
                    observed=clean if snr is None else common.noise_samples(clean=clean,snr_db=snr,seed=seed,
                        signal_reference_power=power,noise_shape="white",os_factor=r)
                    columns.append(prepare_dechirped_symbol(observed,0,sf,r,int(packet["cfo_int"]),
                        float(packet["cfo_frac"]),0,"symbol"))
                    labels.append((packet["packet_index"],item["payload_symbol_index"],int(item["gt_bin"])))
            values=np.stack(columns,axis=1)
            predictions={}
            for name,bank in banks.items():
                scores=np.abs(bank@values[patterns[name],:])**2
                # Uniform dictionaries have exact aliases. Quantized relative
                # scores impose deterministic first-bin ties, not roundoff wins.
                scores=np.round(scores/np.maximum(scores.max(axis=0),1e-30),5)
                predictions[name]=scores.argmax(axis=0)
            for i,(packet,symbol,gt) in enumerate(labels):
                row=dict(dataset=dataset,snr_db="clean" if snr is None else snr,seed=noise_seed,
                    packet_index=packet,payload_symbol_index=symbol,gt_bin=gt)
                row.update({name+"_error":int(pred[i]!=gt) for name,pred in predictions.items()})
                row.update({name+"_bin":int(pred[i]) for name,pred in predictions.items()})
                results.append(row)
            for packet in packets:
                indexes=[i for i,label in enumerate(labels) if label[0]==packet["packet_index"]]
                header=[s["symbol_value"] for s in packet["header_symbols"]]
                reference=decode_explicit_frame_symbols(header,[s["symbol_value"] for s in packet["payload_symbols"]],
                    sf,125000,int(packet["ldro"]),"grlora")
                assert reference.payload.crc_valid
                for name,pred in predictions.items():
                    decoded=decode_explicit_frame_symbols(header,[bin_to_grlora_symbol(int(pred[i]),sf,False,bool(packet["ldro"])) for i in indexes],
                        sf,125000,int(packet["ldro"]),"grlora")
                    packet_results.append(dict(dataset=dataset,snr_db="clean" if snr is None else snr,seed=noise_seed,
                        packet_index=packet["packet_index"],method=name,crc_valid=int(decoded.payload.crc_valid),
                        exact_payload_match=int(decoded.payload.payload_bytes==reference.payload.payload_bytes),
                        clean_header_and_sync_supplied=1))
            print(dataset,snr,len(labels),{name:int(np.sum(pred!=np.array([x[2] for x in labels]))) for name,pred in predictions.items()},flush=True)
    with (out/"trials.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    with (out/"packet_trials.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(packet_results[0]));w.writeheader();w.writerows(packet_results)
    (out/"config.json").write_text(json.dumps(dict(sf=sf,bw=125000,source_rate=500000,
        pattern_seed=91400,noise_seeds=[91401,91402,91403],patterns={k:v.tolist() for k,v in patterns.items()},
        sanity=sanity,truth="frozen clean FFT labels; not independent transmitted labels",
        scope="oracle clean timing/CFO and header; white ADC noise; no noisy packet synchronization; exhaustive matched filter",
        crc_mode="grlora; all 28 frozen clean references pass; sx1276 mode passes 0/28",
        tie_rule="relative power rounded to 5 decimal places; first maximum"),indent=2))


if __name__=="__main__": main()

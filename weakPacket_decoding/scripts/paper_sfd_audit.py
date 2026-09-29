"""Parallel paired SFD audit with shared deterministic acquisition work.

Memoization only reuses alignment/locator results for identical noisy inputs;
it never reuses a clean coordinate for noisy synchronization.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weak_decoder.os_lora.experiments import evaluate_virtual_phase_sfd_sync_ota as ref


def work(path):
    repo=ROOT.parent.parent/"lora-rfsr-savaux"
    ota=repo/"data/reference_phy/rfsr_db"
    module=ref.base._load_single_packet_sync_module(repo)
    original_sync=module.run_single_packet_sync
    last=[]
    def remember(*args,**kwargs):
        value=original_sync(*args,**kwargs)
        last[:]=[value]
        return value
    module.run_single_packet_sync=remember
    clean=ref.base._clean_packet_worker(dict(dataset_repo=str(repo),ota_root=str(ota),metadata_path=path,symbols_per_packet=32))
    packet=clean["packet"]
    if packet is None:return clean["audit"],[],[]
    clean_frame=last[0].frame_sync
    module.run_single_packet_sync=original_sync
    cache={}
    for name in ["align_event_start","locate_frame_from_event"]:
        original=getattr(module,name)
        def cached(*args,_fn=original,_name=name,**kwargs):
            key=(_name,repr(args[1:]),repr(sorted(kwargs.items())))
            if key not in cache:cache[key]=_fn(*args,**kwargs)
            return cache[key]
        setattr(module,name,cached)
    samples=np.fromfile(packet["iq_path"],dtype="<c8")
    config=ref.base._sync_config(module,packet["center_frequency_hz"])
    trials=[];symbols=[]
    for seed in [91301,91302]:
        cache.clear()
        rng=np.random.default_rng(np.random.SeedSequence((seed,int(packet["reference_id"]),73013)))
        power=float(packet["signal_power"])*ref.base.N_BINS/10**1.3
        noisy=(samples+ref.base._unit_lora_band_awgn(rng,samples.size)*math.sqrt(power)).astype(np.complex64)
        chip=ref._sync_with_sfd_mode(module,noisy,config,"chip")
        virtual=ref._sync_with_sfd_mode(module,noisy,config,"virtual_phase")
        trial_id=f"{packet['packet_id']}:13:{seed}"
        trial=dict(trial_id=trial_id,packet_id=packet["packet_id"],reference_id=packet["reference_id"],
            esn0_db=13.0,seed=seed,clean_cfo_int=int(clean_frame.cfo_int_est),
            clean_cfo_total_bins=float(clean_frame.cfo_total_est),clean_payload_start_sample=int(clean_frame.fine_payload_start_sample))
        trial.update(ref._frame_fields("chip",chip,clean_frame))
        trial.update(ref._frame_fields("virtual",virtual,clean_frame))
        trials.append(trial)
        symbols.extend(ref._evaluate_payloads(noisy,packet,chip,virtual,trial_id,13.0,seed))
    # Workers may process another packet: restore module state completely.
    for name in ["align_event_start","locate_frame_from_event"]:
        setattr(module,name,getattr(module,name).__kwdefaults__["_fn"])
    module.run_grlora_frame_sync_validation=ref.run_grlora_frame_sync_validation
    return clean["audit"],trials,symbols


def main():
    out=ROOT/"data/experiments/paper_audit_20260911/sfd"
    out.mkdir(parents=True,exist_ok=True)
    ota=ROOT.parent.parent/"lora-rfsr-savaux/data/reference_phy/rfsr_db"
    paths=list(ref.base._packet_metadata_paths(ota))[:24]
    audits=[];trials=[];symbols=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(work,str(p)) for p in paths]
        for i,f in enumerate(as_completed(futures),1):
            a,t,s=f.result();audits.append(a);trials.extend(t);symbols.extend(s)
            ref._write_csv(out/"clean_sync_audit.csv",audits)
            ref._write_csv(out/"packet_trials.csv",trials)
            ref._write_csv(out/"symbol_trials.csv",symbols)
            print(f"completed {i}/{len(paths)} packets",flush=True)
    summary=ref._summary(trials,symbols,[13.0])
    ref._write_csv(out/"summary.csv",summary)
    ref._build_report(out,summary)
    (out/"config.json").write_text(json.dumps(dict(max_packets=24,seeds=[91301,91302],esn0_db=13,
        symbols_per_packet=32,noise="whole-packet bandlimited AWGN",memoization="identical-input alignment and locator only",
        scope="conditional on clean-admitted packets and clean-correct symbols; not CRC PDR"),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()

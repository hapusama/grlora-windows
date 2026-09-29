"""Held-out whole-packet receiver comparison with strong synchronization baselines.

No expected payload or clean timing enters a noisy receiver. Clean references
are used for SNR calibration, the explicitly labelled oracle, and scoring.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial
from pathlib import Path
from dataclasses import replace
import argparse
import json
import math
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weak_decoder.os_lora.experiments import evaluate_decoder_aware_crc_pdr_ota as base
from weak_decoder.synchronization.grlora_frame_sync import run_grlora_frame_sync_validation


def work(task):
    path,snrs,seeds=task
    repo=ROOT.parent.parent/"lora-rfsr-savaux"
    ota=repo/"data/reference_phy/rfsr_db"
    module=base.load_single_packet_sync_module(repo)
    module.run_grlora_frame_sync_validation=partial(run_grlora_frame_sync_validation,sfd_cfo_mode="chip")
    clean=base._clean_worker(dict(dataset_repo=str(repo),ota_root=str(ota),metadata_path=path))
    packet=clean["packet"]
    if packet is None:return clean["audit"],[],[]
    samples=np.fromfile(packet["iq_path"],dtype="<c8")
    expected=bytes.fromhex(packet["expected_frame_hex"])
    config=base._sync_config(module,packet["center_frequency_hz"])
    original={name:getattr(module,name) for name in ["align_event_start","locate_frame_from_event","detect_preamble_runs"]}
    cache={}
    for name,fn in original.items():
        def cached(*args,_fn=fn,_name=name,**kwargs):
            key=(_name,repr(args[1:]),repr(sorted(kwargs.items())))
            if key not in cache:cache[key]=_fn(*args,**kwargs)
            return cache[key]
        setattr(module,name,cached)
    trials=[];attempt_rows=[]
    for seed in seeds:
        rng=np.random.default_rng(np.random.SeedSequence((seed,packet["reference_id"],9122026)))
        prefix=int(rng.integers(0,base.SYMBOL_SAMPLES))
        padded=np.pad(samples,(prefix,base.DEMOD_TAIL_SAMPLES))
        noise=base.unit_lora_band_awgn(rng,len(padded),os_factor=8)
        oracle_frame=base._clean_frame_sync(packet)
        oracle_frame.fine_payload_start_sample+=prefix
        for snr in snrs:
            cache.clear()
            power=packet["signal_power"]*base.N_BINS/10**(snr/10)
            noisy=(padded+noise*np.sqrt(power)).astype(np.complex64)
            decoded_cache={}
            def decode(frame,soft=False):
                if frame is None:return None,0.0
                key=(soft,)+tuple(getattr(frame,k) for k in ["fine_payload_start_sample","cfo_int_est","cfo_frac_est","sfo_hat","sfo_cum_initial"])
                if key not in decoded_cache:
                    tick=time.perf_counter()
                    value=(base.decode_soft_hamming_sync_candidate(noisy,frame,sf=12,bw_hz=125000,
                        os_factor=8,ldro_mode=1,crc_mode="grlora",allow_gate_failed_candidate=True)
                        if soft else base._decode(noisy,frame,allow_gate_failed_candidate=True))
                    decoded_cache[key]=(value,time.perf_counter()-tick)
                return decoded_cache[key]
            def accepted(value):
                return bool(value is not None and value.header_valid and value.header.has_crc and value.crc_valid)
            methods={};sync_results={};lists={};sync_times={};list_times={}
            for mode in ["chip","virtual_phase"]:
                tick=time.perf_counter()
                module.run_grlora_frame_sync_validation=partial(run_grlora_frame_sync_validation,sfd_cfo_mode=mode)
                sync=module.run_single_packet_sync(noisy,config)
                sync_times[mode]=time.perf_counter()-tick
                sync_results[mode]=sync
                frame=sync.frame_sync
                short="chip" if mode=="chip" else "full"
                methods[short+"_strict"]=[frame] if sync.synchronized else []
                methods[short+"_soft"]=[frame] if sync.accepted("decoder_aware") else []
                methods[short+"_crc1"]=[frame] if frame is not None else []
                tick=time.perf_counter()
                candidates=base.build_ambiguity_ridge_sync_list(noisy,frame,sf=12,bw_hz=125000,os_factor=8,
                    center_frequency_hz=packet["center_frequency_hz"],preamble_symbols=16,sync_word=0x12,top_k=4,sfd_peak_pool=32).candidates
                list_times[mode]=time.perf_counter()-tick
                lists[short]=list(candidates)
                methods[short+"_list4"]=list(candidates)
            methods["hybrid4"]=[]
            seen=set()
            for frame in [sync_results["virtual_phase"].frame_sync,sync_results["chip"].frame_sync]+lists["full"]+lists["chip"]:
                if frame is None:continue
                # Avoid spending a CRC attempt on an identical CFO/start pair.
                key=(frame.fine_payload_start_sample,round(float(frame.cfo_int_est)+float(frame.cfo_frac_est),6))
                if key in seen:continue
                seen.add(key);methods["hybrid4"].append(frame)
                if len(methods["hybrid4"])==4:break
            methods["oracle"]=[oracle_frame]
            for name in ["chip_strict","full_strict","chip_crc1","full_crc1","chip_list4","full_list4","hybrid4","oracle"]:
                methods[name+"_softfec"]=methods[name]
            for method,frames in methods.items():
                selected=None;attempts=0;decode_time=0.0
                for index,frame in enumerate(frames):
                    value,elapsed=decode(frame,soft=method.endswith("_softfec"))
                    attempts+=1;decode_time+=elapsed
                    crc=accepted(value)
                    exact=bool(crc and value.payload_bytes==expected)
                    attempt_rows.append(dict(packet_id=packet["packet_id"],seed=seed,esn0_db=snr,method=method,
                        index=index,header_valid=int(value.header_valid),crc_accept=int(crc),exact=int(exact),
                        payload_start=int(frame.fine_payload_start_sample),cfo_bins=float(frame.cfo_int_est)+float(frame.cfo_frac_est)))
                    if crc:selected=value;break
                crc=accepted(selected)
                exact=bool(crc and selected.payload_bytes==expected)
                trials.append(dict(packet_id=packet["packet_id"],reference_id=packet["reference_id"],seed=seed,
                    esn0_db=snr,channel_snr_db=snr-10*math.log10(base.N_BINS),prefix_samples=prefix,method=method,
                    crc_accept=int(crc),exact=int(exact),false_delivery=int(crc and not exact),attempts=attempts,
                    decode_seconds=decode_time,chip_sync_seconds=sync_times["chip"],
                    additional_full_sync_seconds=sync_times["virtual_phase"],
                    chip_list_seconds=list_times["chip"],full_list_seconds=list_times["virtual_phase"],
                    chip_estimate_available=int(sync_results["chip"].frame_sync is not None),
                    full_estimate_available=int(sync_results["virtual_phase"].frame_sync is not None)))
            print(f"done {packet['packet_id']} EsN0={snr} seed={seed}",flush=True)
    for name,fn in original.items():setattr(module,name,fn)
    module.run_grlora_frame_sync_validation=run_grlora_frame_sync_validation
    return clean["audit"],trials,attempt_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offset",type=int,default=96)
    p.add_argument("--packets",type=int,default=24)
    p.add_argument("--seeds",default="91601,91602")
    p.add_argument("--snrs",default="12,13,14")
    p.add_argument("--workers",type=int,default=3)
    p.add_argument("--output",type=Path,default=ROOT/"data/experiments/paper_story_20260912")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths=list(base.packet_metadata_paths(ROOT.parent.parent/"lora-rfsr-savaux/data/reference_phy/rfsr_db"))[args.offset:args.offset+args.packets]
    snrs=[float(x) for x in args.snrs.split(",")];seeds=[int(x) for x in args.seeds.split(",")]
    (args.output/"config.json").write_text(json.dumps(dict(offset=args.offset,requested_packets=len(paths),
        metadata=[str(p) for p in paths],snrs=snrs,seeds=seeds,
        prefix="uniform 0..32767 extra samples; noise added once to whole padded IQ",
        scope="held-out packets from same lab session; clean CRC eligibility audited separately",
        selection="first valid-header, CRC-enabled, CRC-pass candidate; expected bytes only score afterwards",
        policies="chip/full strict, soft admission, unrestricted single CRC; chip/full ridge4; frozen union hybrid4; oracle; matched soft-FEC ablations",
        list_note="existing chip-phase SFD alternative generator for both lists; only initial sync point differs"),indent=2))
    audits=[];trials=[];attempts=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,(str(path),snrs,seeds)) for path in paths]
        for i,f in enumerate(as_completed(futures),1):
            a,t,c=f.result();audits.append(a);trials.extend(t);attempts.extend(c)
            base.write_csv_rows(args.output/"clean_audit.csv",audits)
            base.write_csv_rows(args.output/"trials.csv",trials)
            base.write_csv_rows(args.output/"attempts.csv",attempts)
            print(f"completed {i}/{len(paths)} packets",flush=True)


if __name__=="__main__":main()

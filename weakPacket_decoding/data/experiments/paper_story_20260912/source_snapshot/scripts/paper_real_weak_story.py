"""Re-decode natural weak SF10 captures from shared received-IQ locator events.

The historical event/locator pool is shared and never obtained from clean IQ.
Missed events cannot be measured here, so outputs are recovery counts, not PDR.
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weak_decoder.synchronization.preamble_detector import PreambleDetectorConfig
from weak_decoder.synchronization.grlora_frame_sync import run_grlora_frame_sync_validation
from weak_decoder.os_lora.system.decoder_aware_crc import decode_savaux_sync_candidate
from weak_decoder.os_lora.system.soft_hamming_crc import decode_soft_hamming_sync_candidate
from weak_decoder.os_lora.system.ambiguity_ridge_list import build_ambiguity_ridge_sync_list
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols


def read(path):
    with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))


def write(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    out=ROOT/"data/experiments/paper_story_20260912/natural_weak"
    out.mkdir(parents=True,exist_ok=True)
    gt_path=ROOT/"data/groundtruth/branch4_fixed/high_snr/sf10_bw125_fs500_pre32_sw34_r001_fft_bin_groundtruth.csv"
    gt=read(gt_path)
    reference=decode_explicit_frame_symbols([int(r["groundtruth_symbol"]) for r in gt if r["stage"]=="header"],
        [int(r["groundtruth_symbol"]) for r in gt if r["stage"]=="payload"],10,125000,2,"grlora")
    assert reference.header.header_valid and reference.payload.crc_valid
    expected=reference.payload.payload_bytes
    rows=[];audits=[]
    config=PreambleDetectorConfig(10,125000,500000,4,4096,29,2)
    for capture in range(1,8):
        folder="real_low_snr_20260717" if capture<=3 else "real_low_snr_20260718_low4_low6" if capture==4 else "real_low_snr_20260719_low5_low7"
        sync_path=ROOT/f"data/experiments/{folder}/low{capture}_win4/sync.csv"
        source=read(sync_path)
        iq_path=ROOT/f"USRP_collector/data/branch4_fixed/low_snr/sf10_bw125_fs500_pre32_sw34_low{capture}.bin"
        iq=np.memmap(iq_path,dtype=np.complex64,mode="r")
        audits.append(dict(capture=capture,iq_path=str(iq_path),sync_path=str(sync_path),events=len(source),
            seconds=len(iq)/500000,iq_sha256=hashlib.file_digest(iq_path.open("rb"),"sha256").hexdigest() if hasattr(hashlib,"file_digest") else "not_available"))
        for row in source:
            location=SimpleNamespace(event_index=int(row["event_index"]),preamble_ref_bin=int(row["preamble_ref_bin"]),
                preamble_start_sample=int(row["located_preamble_start_sample"]))
            methods={};frames={};lists={}
            for mode,name in [("chip","chip"),("virtual_phase","full")]:
                frame=run_grlora_frame_sync_validation(iq,location,config,32,0x34,center_freq=487.7e6,sfd_cfo_mode=mode)
                frames[name]=frame
                methods[name+"_strict"]=[frame] if int(row["frame_valid"]) and frame.valid else []
                methods[name+"_crc1"]=[frame]
                lists[name]=list(build_ambiguity_ridge_sync_list(iq,frame,sf=10,bw_hz=125000,os_factor=4,
                    center_frequency_hz=487.7e6,preamble_symbols=32,sync_word=0x34,top_k=4,sfd_peak_pool=32).candidates)
                methods[name+"_list4"]=lists[name]
            methods["hybrid4"]=[];seen=set()
            for frame in [frames["full"],frames["chip"]]+lists["full"]+lists["chip"]:
                key=(frame.fine_payload_start_sample,round(frame.cfo_int_est+frame.cfo_frac_est,6))
                if key in seen:continue
                seen.add(key);methods["hybrid4"].append(frame)
                if len(methods["hybrid4"])==4:break
            for name,ff in list(methods.items()):methods[name+"_softfec"]=ff
            cache={}
            for method,ff in methods.items():
                selected=None;attempts=0
                for frame in ff:
                    soft=method.endswith("_softfec")
                    key=(soft,)+tuple(getattr(frame,k) for k in ["fine_payload_start_sample","cfo_int_est","cfo_frac_est","sfo_hat","sfo_cum_initial"])
                    if key not in cache:
                        fn=decode_soft_hamming_sync_candidate if soft else decode_savaux_sync_candidate
                        cache[key]=fn(iq,frame,sf=10,bw_hz=125000,os_factor=4,ldro_mode=2,crc_mode="grlora",allow_gate_failed_candidate=True)
                    value=cache[key];attempts+=1
                    if value.header_valid and value.header.has_crc and value.crc_valid:selected=value;break
                crc=selected is not None
                rows.append(dict(capture=capture,event_index=int(row["event_index"]),method=method,crc_accept=int(crc),
                    exact=int(crc and selected.payload_bytes==expected),reference_mismatch=int(crc and selected.payload_bytes!=expected),
                    payload_sha256=hashlib.sha256(selected.payload_bytes).hexdigest() if crc else "",attempts=attempts,
                    baseline_start_matches_history=int(frames["chip"].fine_payload_start_sample==int(row["grlora_fine_payload_start_sample"])),
                    baseline_cfo_error_vs_history=frames["chip"].cfo_total_est-float(row["grlora_cfo_total_est"])))
        write(out/"trials.csv",rows);write(out/"capture_audit.csv",audits)
        print(f"finished natural low{capture}: {len(source)} events",flush=True)
    (out/"config.json").write_text(json.dumps(dict(no_added_noise=True,sf=10,bw=125000,sample_rate=500000,
        preamble=32,sync_word="0x34",crc_mode="grlora",reference_csv=str(gt_path),
        scope="fixed received-IQ locator pool, original win4 detection; not independent arrival/PDR audit",
        groundtruth="CRC-valid high-SNR fixed-payload consensus; used only after receiver decisions"),indent=2))


if __name__=="__main__":main()

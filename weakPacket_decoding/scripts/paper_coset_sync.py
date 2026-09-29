"""Adapt arXiv:2605.18189 Algorithm 1 to LoRa known-sequence acquisition.

This is a synthetic, single-path transfer experiment, not a 5G reproduction or
an OTA receiver claim. Offline pattern design sees noiseless training states;
all receivers share held-out arrival/CFO/noise trials and equal search grids.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
from itertools import combinations
from pathlib import Path
import csv
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weak_decoder.chirp import build_upchirp


def write(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def build_pilot(kind):
    up=build_upchirp(7,0,2);down=np.conj(up)
    if kind=="preamble_only":return np.tile(up,4)
    if kind=="preamble_equal_energy":return np.concatenate([np.tile(up,8),up[:64]])
    return np.concatenate([np.tile(up,4),build_upchirp(7,8,2),build_upchirp(7,16,2),down,down,down[:64]])


def experiment(kind,out):
    pilot=build_pilot(kind)
    delays=np.arange(0,129,2)
    cfos=np.arange(-64,65,dtype=float)
    dd=np.repeat(delays,len(cfos));ff=np.tile(cfos,len(delays))
    length=len(pilot)+128
    times=np.arange(length)
    atoms=np.empty((len(dd),length),np.complex64)
    for i,d in enumerate(delays):
        shifted=np.zeros(length,np.complex64);shifted[d:d+len(pilot)]=pilot
        atoms[i*len(cfos):(i+1)*len(cfos)]=shifted[None,:]*np.exp(2j*np.pi*cfos[:,None]*times[None,:]/256)
    train=[(d,f) for d in [32,64,96] for f in [-32,0,32]]
    trainidx=[int(np.flatnonzero((dd==d)&(ff==f))[0]) for d,f in train]
    components=[];energies=[]
    for c in range(8):
        a=atoms[:,c::8]
        components.append(np.conj(a)@a[trainidx].T)
        energies.append(np.sum(np.abs(a)**2,axis=1))
    components=np.stack(components);energies=np.stack(energies)
    designs=[]
    for cs in combinations(range(8),2):
        energy=energies[list(cs)].sum(axis=0)
        score=np.abs(components[list(cs)].sum(axis=0))**2/energy[:,None]
        balance=float(energy.min()/energy.max())
        spr=[]
        for j,(d,f) in enumerate(train):
            # One grid-index guard in each dimension, as in the source paper.
            outside=(np.abs(dd-d)>2)|(np.abs(ff-f)>1)
            spr.append(float(score[outside,j].max()/score[trainidx[j],j]))
        designs.append(dict(pilot=kind,cosets=" ".join(map(str,cs)),spr=float(np.mean(spr)),
            balance=balance,cost=float(np.mean(spr)/balance)))
    designs.sort(key=lambda x:(x["cost"],x["cosets"]))
    write(out/f"{kind}_design.csv",designs)
    best=tuple(map(int,designs[0]["cosets"].split()))
    patterns=dict(full=tuple(range(8)),uniform=(0,4),fixed_03=(0,3),optimized=best)
    for j in range(4):
        patterns[f"random_{j}"]=tuple(sorted(np.random.default_rng(91500+j).choice(8,2,replace=False).tolist()))
    rng=np.random.default_rng(91501)
    trial_count=256
    true_delays=rng.integers(0,129,size=trial_count)
    true_cfos=rng.integers(-48,49,size=trial_count).astype(float)
    # Half the trials are off the delay grid; CFO has independent fine residual.
    true_cfos+=rng.uniform(-.04,.04,size=trial_count)
    signals=np.zeros((length,trial_count),np.complex64)
    for j,(d,f) in enumerate(zip(true_delays,true_cfos)):
        signals[d:d+len(pilot),j]=pilot
        signals[:,j]*=np.exp(2j*np.pi*f*times/256+1j*rng.uniform(-np.pi,np.pi))
    noise=((rng.normal(size=signals.shape)+1j*rng.normal(size=signals.shape))/np.sqrt(2)).astype(np.complex64)
    null=((rng.normal(size=(length,256))+1j*rng.normal(size=(length,256)))/np.sqrt(2)).astype(np.complex64)
    rows=[];nullrows=[];summaries=[]
    for method,cs in patterns.items():
        idx=np.flatnonzero(np.isin(times%8,cs))
        bank=np.conj(atoms[:,idx]).copy()
        energy=np.sum(np.abs(bank)**2,axis=1)
        nullpower=np.abs(bank@null[idx])**2/energy[:,None]
        nullmax=nullpower.max(axis=0)
        threshold=float(nullmax[:128].max())
        for j,value in enumerate(nullmax):nullrows.append(dict(pilot=kind,method=method,trial=j,
            split="calibration" if j<128 else "test",score=float(value),threshold=threshold))
        for snr in [-18,-14,-10]:
            variance=10**(-snr/10)
            observed=signals[idx]+noise[idx]*np.sqrt(variance)
            tick=time.perf_counter()
            power=np.abs(bank@observed)**2/energy[:,None]
            elapsed=time.perf_counter()-tick
            selected=power.argmax(axis=0)
            correct=(np.abs(dd[selected]-true_delays)<=2)&(np.abs(ff[selected]-true_cfos)<=1)
            detected=power.max(axis=0)/variance>threshold
            for j in range(trial_count):rows.append(dict(pilot=kind,method=method,cosets=" ".join(map(str,cs)),snr_db=snr,
                trial=j,true_delay=int(true_delays[j]),true_cfo_bins=float(true_cfos[j]),
                estimated_delay=int(dd[selected[j]]),estimated_cfo_bins=float(ff[selected[j]]),
                peak_statistic=float(power[:,j].max()/variance),
                correct=int(correct[j]),detected=int(detected[j]),correct_and_detected=int(correct[j] and detected[j])))
            summaries.append(dict(pilot=kind,method=method,cosets=" ".join(map(str,cs)),snr_db=snr,trials=trial_count,
                correct=int(correct.sum()),correct_and_detected=int((correct&detected).sum()),
                false_alarm_test=int(np.sum(nullmax[128:]>threshold)),noise_test_trials=128,
                delay_rmse_samples=float(np.sqrt(np.mean((dd[selected]-true_delays)**2))),
                cfo_rmse_bins=float(np.sqrt(np.mean((ff[selected]-true_cfos)**2))),
                retained_samples=len(idx),correlator_seconds_per_trial=elapsed/trial_count))
        print(kind,method,cs,flush=True)
    write(out/f"{kind}_trials.csv",rows);write(out/f"{kind}_noise.csv",nullrows)
    return designs,summaries


def main():
    out=ROOT/"data/experiments/paper_audit_20260911/coset_sync"
    out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for kind in ["preamble_only","preamble_equal_energy","joint_known"]:
        _,rows=experiment(kind,out);summaries.extend(rows)
        write(out/"summary.csv",summaries)
    (out/"config.json").write_text(json.dumps(dict(sf=7,bw=125000,fs=250000,L=8,K=2,
        design="all 28 patterns; mean sidelobe/peak divided by min/max retained energy",
        train_delays=[32,64,96],train_cfo_bins=[-32,0,32],delay_grid="0:2:128 samples",cfo_grid="-64:1:64 LoRa bins",
        test_delays="uniform integers 0..128",test_cfo="uniform integer -48..48 plus Uniform(-.04,.04)",
        test_seed=91501,pattern_baseline_seeds=[91500,91501,91502,91503],trials_per_snr=256,
        noise="ADC-white proper complex Gaussian; true noise variance supplied for CFAR statistic",
        negative_controls="128 calibration plus 128 holdout noise-only trials per waveform; paired across patterns",
        scope="synthetic single path; broad CFO stress; no OTA/PDR claim; block acquisition with known pilot count",
        source="https://arxiv.org/html/2605.18189v1"),indent=2))


if __name__=="__main__":main()

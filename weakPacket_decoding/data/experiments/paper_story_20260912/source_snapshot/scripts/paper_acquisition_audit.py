"""Independent-seed acquisition audit, including noise-only negative controls.

Run from any directory. Existing receiver parameters are frozen. The exact
full FFT replaces its algebraically equivalent polyphase implementation only
inside this experiment; equivalence is checked before launching workers.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weak_decoder.os_lora.experiments import evaluate_virtual_simo_preamble_acquisition_ota as old
from weak_decoder.synchronization import preamble_detector as pd


def full_fft(values, os_factor):
    return np.fft.fft(np.asarray(values, np.complex64).reshape(-1)).astype(np.complex64)


def evaluate(task):
    pd.virtual_simo_preamble_spectrum = full_fft
    kind, audit, seed, snrs = task
    if kind == "ota":
        rid = int(audit["reference_id"])
        pid = audit["packet_id"]
        root = ROOT.parent.parent / "lora-rfsr-savaux/data/reference_phy/rfsr_db"
        meta = json.loads((root / "metadata" / f"{pid}.json").read_text())
        clean = np.fromfile(root / meta["ota"]["relative_path"], dtype="<c8")
        start = int(audit["fine_payload_start_sample"]) - round(20.25 * old.CHIRP_SAMPLES)
        rng = np.random.default_rng(np.random.SeedSequence((seed, rid, 73013)))
        noise = old.base._unit_lora_band_awgn(rng, clean.size)
    else:
        rid, pid, start = -1, "noise_only", -100000000
        rng = np.random.default_rng(np.random.SeedSequence((seed, 91877)))
        noise = old.base._unit_lora_band_awgn(rng, old.SCAN_CHIRPS * old.CHIRP_SAMPLES)
        clean = np.zeros_like(noise)
    rows = []
    for snr in snrs:
        power = float(audit["signal_power"]) / 10 ** (snr / 10) if kind == "ota" else 1.0
        noisy = np.asarray(clean + noise * math.sqrt(power), np.complex64)
        specs = list(old._detector_specs())
        # Actual retained ADC observations; same elapsed window and same noise.
        specs += [dict(name=f"uniform_{rate}k_m12", mode="virtual_simo",
                       window_chirps=12, min_run=5, decimation=dec)
                  for rate, dec in [(250, 4), (500, 2)]]
        for spec in specs:
            dec = spec.get("decimation", 1)
            config = pd.PreambleDetectorConfig(sf=12, bw=125000,
                samp_rate=1000000 / dec, win_chirps=spec["window_chirps"],
                hop_samples=old.CHIRP_SAMPLES // dec,
                min_periodic_peaks=spec["min_run"], bin_tol=2)
            tick = time.perf_counter()
            windows, events = pd.detect_preamble_runs(noisy[::dec], config,
                sample_limit=old.SCAN_CHIRPS * old.CHIRP_SAMPLES // dec,
                mode=spec["mode"])
            true = [e for e in events if kind == "ota" and
                    start-old.CHIRP_SAMPLES <= e.start_sample*dec <= start+6*old.CHIRP_SAMPLES]
            rows.append(dict(kind=kind, packet_id=pid, reference_id=rid, seed=seed,
                channel_snr_db=snr, detector=spec["name"],
                retained_fraction=(1/8 if spec["mode"] == "chip_phase" else 1/dec),
                true_basin_detected=int(bool(true)), event_count=len(events),
                false_event_count=len(events)-len(true),
                max_event_share=max((e.max_peak_share for e in events), default=0),
                max_true_share=max((e.max_peak_share for e in true), default=0),
                elapsed_seconds=time.perf_counter()-tick))
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="91001,91002,91003,91004,91005,91006,91007,91008")
    p.add_argument("--snrs", default="-32,-30,-28,-26")
    p.add_argument("--noise-trials", type=int, default=128)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--output", type=Path, default=ROOT/"data/experiments/paper_audit_20260911/acquisition")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT/"data/experiments/noisy_framesync_headroom_ota_awgn_20260821/clean_sync_audit.csv"
    audits = old._read_csv(audit_path)
    seeds = [int(s) for s in args.seeds.split(",")]
    snrs = [float(s) for s in args.snrs.split(",")]
    rng = np.random.default_rng(99101)
    x = (rng.normal(size=(12,32768)) + 1j*rng.normal(size=(12,32768))).astype(np.complex64)
    a = pd.virtual_simo_preamble_spectrum(x,8)
    b = full_fft(x,8)
    relative = float(np.linalg.norm(a-b)/np.linalg.norm(a))
    assert relative < 1e-6 and a.argmax() == b.argmax()
    tasks = [("ota", audit, seed, snrs) for audit in audits for seed in seeds]
    tasks += [("noise", {}, 92000+i, [0.0]) for i in range(args.noise_trials)]
    config = dict(seeds=seeds, channel_snr_db=snrs, ota_packets=len(audits),
        noise_trials=args.noise_trials, noise_scan_seconds=old.SCAN_CHIRPS*old.CHIRP_SAMPLES/1e6,
        fft_equivalence_relative_error=relative,
        clean_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        truth_usage="post-hoc basin labelling; no payload or boundary enters detector",
        scope="OTA plus injected channel-bandlimited noise; conditional on eight clean-admitted captures")
    (args.output/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate, task) for task in tasks]
        for i,f in enumerate(as_completed(futures),1):
            rows.extend(f.result())
            write_csv(args.output/"trials.csv",rows)
            if i%8 == 0 or i == len(tasks):
                print(f"completed {i}/{len(tasks)} tasks",flush=True)
    summary=[]
    for kind,snr,det in sorted(set((r["kind"],r["channel_snr_db"],r["detector"]) for r in rows)):
        subset=[r for r in rows if (r["kind"],r["channel_snr_db"],r["detector"]) == (kind,snr,det)]
        summary.append(dict(kind=kind, channel_snr_db=snr, detector=det,trials=len(subset),
            hits=sum(r["true_basin_detected"] for r in subset),
            false_events=sum(r["false_event_count"] for r in subset),
            false_trials=sum(r["false_event_count"]>0 for r in subset)))
    write_csv(args.output/"summary.csv",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()

"""Verify paired replay coverage and plot recovery counts and decode calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    configs, tables = {}, {}
    for group in ("replay", "validation"):
        configs[group] = json.loads((out / group / "config.json").read_text())
        tables[group] = json.loads((out / group / "summary.json").read_text())
        rows = read_csv(out / group / "trials.csv")
        expected_count = len(configs[group]["metadata"]) * len(configs[group]["seeds"]) * len(configs[group]["snrs"])
        for summary in tables[group]:
            selected = [r for r in rows if r["method"] == summary["method"]]
            keys = {(r["packet_id"], int(r["seed"]), float(r["esn0_db"])) for r in selected}
            if len(selected) != expected_count or len(keys) != expected_count:
                raise ValueError(f"incomplete or duplicate trials for {group}/{summary['method']}")
            if sum(int(r["exact"]) for r in selected) != summary["correct"]:
                raise ValueError("summary does not match trial data")
    if set(configs["replay"]["metadata"]) & set(configs["validation"]["metadata"]):
        raise ValueError("physical packet overlap")
    if set(configs["replay"]["seeds"]) & set(configs["validation"]["seeds"]):
        raise ValueError("noise seed overlap")
    hashes = [{key.replace("\\", "/"): value for key, value in configs[group]["source_hashes"].items()}
              for group in ("replay", "validation")]
    source_key = "weak_decoder/os_lora/system/residual_cfo.py"
    if hashes[0][source_key] != hashes[1][source_key]:
        raise ValueError("receiver changed between runs")
    old = read_csv(ROOT / "data/experiments/paper_story_20260912/frequency_holdout/trials.csv")
    new = read_csv(out / "replay/trials.csv")
    for before, after in (("full_crc1_softfec", "single_soft"), ("full_cfo3_softfec", "fixed_cfo3_soft")):
        def decisions(rows, method):
            return {(r["packet_id"], r["seed"], float(r["esn0_db"])):
                    (int(r["exact"]), int(r["false_delivery"]), int(r["attempts"]))
                    for r in rows if r["method"] == method}
        if decisions(old, before) != decisions(new, after):
            raise ValueError("historical replay mismatch")

    gt_path = ROOT / "data/groundtruth/branch4_fixed/high_snr/sf10_bw125_fs500_pre32_sw34_r001_fft_bin_groundtruth.csv"
    gt = read_csv(gt_path)
    reference = decode_explicit_frame_symbols(
        [int(r["groundtruth_symbol"]) for r in gt if r["stage"] == "header"],
        [int(r["groundtruth_symbol"]) for r in gt if r["stage"] == "payload"],
        10, 125000, 2, "grlora",
    )
    if not reference.header.header_valid or not reference.payload.crc_valid:
        raise ValueError("natural capture scoring reference is invalid")
    natural = []
    for capture in range(1, 8):
        for policy in ("fixed", "preamble_single"):
            path = out / f"natural_low{capture}_{policy}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            crc = sum(r["crc_accept"] for r in rows)
            exact = sum(r.get("payload_hex") == reference.payload.payload_bytes.hex() for r in rows)
            natural.append(dict(capture=capture, policy=policy, events=len(rows),
                                correct=exact, crc_accepted=crc, false_delivery=crc-exact,
                                attempts=sum(len(r["attempts"]) for r in rows)))
    (out / "natural_summary.json").write_text(json.dumps(natural, indent=2), encoding="utf-8")

    methods = ["single_soft", "fixed_cfo3_soft", "preamble_single_soft", "preamble_cfo3_soft"]
    labels = ["Original\none decode", "Fixed CFO\nup to 3", "Preamble\none decode", "Preamble\nup to 3"]
    colors = ["#697586", "#3478c5", "#d98a24", "#298666"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, group, title in zip(axes[:2], ("replay", "validation"),
                               ("Seen packets: exact replay", "Same session: new packets + noise")):
        selected = {r["method"]: r for r in tables[group]}
        correct = [selected[m]["correct"] for m in methods]
        ax.bar(range(4), correct, color=colors)
        for index, method in enumerate(methods):
            row = selected[method]
            ax.text(index, row["correct"] + 1, f"{row['correct']}/{row['trials']}\n{row['mean_attempts']*row['trials']:.0f} calls", ha="center", fontsize=9)
        ax.axhline(selected["oracle_soft"]["correct"], color="#926bb6", linestyle="--", label="Clean-sync oracle")
        ax.set_xticks(range(4), labels, fontsize=9)
        ax.set(title=title, ylabel="CRC-valid, byte-exact trials", ylim=(0, max(r["trials"] for r in selected.values()) * 1.08))
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=.15)
    totals = {policy: {field: sum(r[field] for r in natural if r["policy"] == policy)
                       for field in ("correct", "attempts", "events")}
              for policy in ("fixed", "preamble_single")}
    axes[2].bar([0, 1], [totals[p]["correct"] for p in totals], color=[colors[1], colors[2]])
    for index, policy in enumerate(totals):
        r = totals[policy]
        axes[2].text(index, r["correct"]+1, f"{r['correct']}/{r['events']} events\n{r['attempts']} calls", ha="center", fontsize=10)
    axes[2].set_xticks([0, 1], ["Fixed CFO\nup to 3", "Preamble\none decode"])
    axes[2].set(title="Natural SF10: seven recordings", ylabel="Byte-exact packets in shared events", ylim=(0, 76))
    axes[2].text(.03, .92, "Last three recordings: no detected events\nTransmitted count unknown; not PDR", transform=axes[2].transAxes, fontsize=8, va="top")
    fig.suptitle("Residual CFO recovery: same soft PHY decoder; no payload truth in candidate selection")
    fig.tight_layout()
    fig.savefig(out / "recovery_evidence.png", dpi=180)
    fig.savefig(out / "recovery_evidence.svg")
    plt.close(fig)

    # Check and archive the exact files listed in each run's source manifest.
    for group in configs:
        for key, expected_hash in configs[group]["source_hashes"].items():
            source = ROOT / key
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"source changed since {group}: {key}")
            target = out / "source_snapshot" / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    external = ROOT.parent.parent / "lora-rfsr-savaux/weak_decoder/synchronization/single_packet.py"
    shutil.copy2(external, out / "source_snapshot/external_single_packet.py")
    utility_hashes = {}
    for source in (Path(__file__), ROOT / "scripts/recover_weak_packets.py"):
        target = out / "source_snapshot/scripts" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        utility_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    validation = dict(full_trial_grids=True, physical_packets_disjoint=True, noise_seeds_disjoint=True,
                      frozen_receiver_hash=hashes[0][source_key], historical_decisions_match=True,
                      source_manifest_matches=True,
                      external_wrapper_post_run_sha256=hashlib.sha256(external.read_bytes()).hexdigest(),
                      utility_post_run_sha256=utility_hashes,
                      natural_scoring_reference_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                      natural=totals)
    (out / "integrity.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

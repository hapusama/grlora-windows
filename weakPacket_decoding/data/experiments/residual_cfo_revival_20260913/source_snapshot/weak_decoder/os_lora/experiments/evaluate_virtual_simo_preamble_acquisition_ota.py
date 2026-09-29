"""Evaluate single-packet virtual-SIMO LoRa preamble acquisition under AWGN.

The experiment separates acquisition from absolute frame synchronization.  A
detected event is labelled with the clean packet boundary only after detection;
the boundary is never provided to any detector.

Three paired detectors use the same noisy packet:

* ``legacy_m4``: four per-chirp FFT powers are added noncoherently;
* ``noncoherent_m12``: the same statistic with a 12-chirp window;
* ``chip_phase_m12``: a 12-chirp coherent FFT using one ADC phase;
* ``virtual_simo_m12``: 12 repeated LoRa chirps are integrated coherently while
  all OSR polyphase branches are aligned with their deterministic steering law.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
WEAK_PACKET_ROOT = SCRIPT_PATH.parents[3]
GR_LORA_ROOT = SCRIPT_PATH.parents[4]
WORKSPACE_ROOT = GR_LORA_ROOT.parent
if str(WEAK_PACKET_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_PACKET_ROOT))

from weak_decoder.os_lora.experiments import (  # noqa: E402
    analyze_noisy_framesync_headroom_ota as base,
)
from weak_decoder.synchronization.preamble_detector import (  # noqa: E402
    PreambleDetectorConfig,
    detect_preamble_runs,
)


SF = 12
N_BINS = 1 << SF
BW_HZ = 125_000.0
SAMPLE_RATE_HZ = 1_000_000.0
OS_FACTOR = 8
CHIRP_SAMPLES = N_BINS * OS_FACTOR
PREAMBLE_SYMBOLS = 16
SCAN_CHIRPS = 32


def _parse_list(text: str, cast: type) -> tuple[Any, ...]:
    values = tuple(cast(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return values


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _detector_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "legacy_m4",
            "mode": "noncoherent",
            "window_chirps": 4,
            "min_run": 13,
        },
        {
            "name": "noncoherent_m12",
            "mode": "noncoherent",
            "window_chirps": 12,
            "min_run": 5,
        },
        {
            "name": "chip_phase_m12",
            "mode": "chip_phase",
            "window_chirps": 12,
            "min_run": 5,
        },
        {
            "name": "virtual_simo_m12",
            "mode": "virtual_simo",
            "window_chirps": 12,
            "min_run": 5,
        },
    )


def _event_is_in_true_preamble_basin(event: Any, preamble_start: int) -> bool:
    # The scan grid is not aligned to the packet.  Permit the first window to
    # contain up to one chirp of leading noise and permit a 12-chirp event run
    # to start within the five fully-contained positions plus one transition.
    return bool(
        int(preamble_start) - CHIRP_SAMPLES
        <= int(event.start_sample)
        <= int(preamble_start) + 6 * CHIRP_SAMPLES
    )


def _summarize(
    rows: Sequence[dict[str, Any]], channel_snr_values: Sequence[float]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for channel_snr_db in channel_snr_values:
        for spec in _detector_specs():
            selected = [
                row
                for row in rows
                if math.isclose(
                    float(row["channel_snr_db"]),
                    float(channel_snr_db),
                    abs_tol=1e-12,
                )
                and str(row["detector"]) == str(spec["name"])
            ]
            output.append(
                {
                    "channel_snr_db": float(channel_snr_db),
                    "esn0_db": float(channel_snr_db)
                    + 10.0 * math.log10(N_BINS),
                    "detector": str(spec["name"]),
                    "trials": len(selected),
                    "true_basin_hits": sum(
                        int(row["true_basin_detected"]) for row in selected
                    ),
                    "true_basin_detection_rate": float(
                        np.mean(
                            [int(row["true_basin_detected"]) for row in selected],
                            dtype=np.float64,
                        )
                    )
                    if selected
                    else float("nan"),
                    "false_events": sum(int(row["false_event_count"]) for row in selected),
                    "trials_with_false_event": sum(
                        int(int(row["false_event_count"]) > 0) for row in selected
                    ),
                    "mean_true_run_length": float(
                        np.mean(
                            [
                                int(row["max_true_run_length"])
                                for row in selected
                                if int(row["true_basin_detected"])
                            ],
                            dtype=np.float64,
                        )
                    )
                    if any(int(row["true_basin_detected"]) for row in selected)
                    else float("nan"),
                }
            )
    return output


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-repo", type=Path, default=WORKSPACE_ROOT / "lora-rfsr-savaux"
    )
    parser.add_argument(
        "--clean-audit",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "noisy_framesync_headroom_ota_awgn_20260821"
        / "clean_sync_audit.csv",
    )
    parser.add_argument("--channel-snr-db", default="-30")
    parser.add_argument("--seeds", default="20260821,20260822")
    parser.add_argument("--max-packets", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "virtual_simo_preamble_acquisition_ota_awgn_20260827",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    dataset_repo = args.dataset_repo.resolve()
    ota_root = dataset_repo / "data" / "reference_phy" / "rfsr_db"
    audits = _read_csv(args.clean_audit.resolve())[: int(args.max_packets)]
    channel_snr_values = _parse_list(args.channel_snr_db, float)
    seeds = _parse_list(args.seeds, int)
    rows: list[dict[str, Any]] = []

    for audit in audits:
        packet_id = str(audit["packet_id"])
        reference_id = int(audit["reference_id"])
        metadata = json.loads(
            (ota_root / "metadata" / f"{packet_id}.json").read_text(
                encoding="utf-8"
            )
        )
        clean = np.fromfile(
            ota_root / str(metadata["ota"]["relative_path"]),
            dtype=np.dtype("<c8"),
        )
        clean_preamble_start = int(audit["fine_payload_start_sample"]) - int(
            round(20.25 * CHIRP_SAMPLES)
        )
        for channel_snr_db in channel_snr_values:
            esn0_db = float(channel_snr_db) + 10.0 * math.log10(N_BINS)
            noise_power = float(audit["signal_power"]) * N_BINS / (
                10.0 ** (esn0_db / 10.0)
            )
            for seed in seeds:
                rng = np.random.default_rng(
                    np.random.SeedSequence((int(seed), reference_id, 73013))
                )
                noise = base._unit_lora_band_awgn(rng, clean.size)
                noisy = np.asarray(
                    clean + noise * math.sqrt(noise_power), dtype=np.complex64
                )
                for spec in _detector_specs():
                    config = PreambleDetectorConfig(
                        sf=SF,
                        bw=BW_HZ,
                        samp_rate=SAMPLE_RATE_HZ,
                        win_chirps=int(spec["window_chirps"]),
                        hop_samples=CHIRP_SAMPLES,
                        min_periodic_peaks=int(spec["min_run"]),
                        bin_tol=2,
                    )
                    _windows, events = detect_preamble_runs(
                        noisy,
                        config,
                        sample_limit=SCAN_CHIRPS * CHIRP_SAMPLES,
                        mode=str(spec["mode"]),
                    )
                    true_events = [
                        event
                        for event in events
                        if _event_is_in_true_preamble_basin(
                            event, clean_preamble_start
                        )
                    ]
                    false_events = [event for event in events if event not in true_events]
                    rows.append(
                        {
                            "packet_id": packet_id,
                            "reference_id": reference_id,
                            "seed": int(seed),
                            "channel_snr_db": float(channel_snr_db),
                            "esn0_db": float(esn0_db),
                            "detector": str(spec["name"]),
                            "window_chirps": int(spec["window_chirps"]),
                            "mode": str(spec["mode"]),
                            "event_count": len(events),
                            "true_basin_detected": int(bool(true_events)),
                            "true_event_count": len(true_events),
                            "false_event_count": len(false_events),
                            "max_true_run_length": max(
                                (int(event.window_count) for event in true_events),
                                default=0,
                            ),
                            "clean_preamble_start_sample": clean_preamble_start,
                        }
                    )
                print(
                    f"completed {packet_id}, SNR {channel_snr_db:g} dB, seed {seed}",
                    flush=True,
                )

    summary = _summarize(rows, channel_snr_values)
    output_dir = args.output_dir.resolve()
    _write_csv(output_dir / "trials.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset_repo": str(dataset_repo),
                "clean_audit": str(args.clean_audit.resolve()),
                "channel_snr_db": list(channel_snr_values),
                "esn0_db": [
                    float(value) + 10.0 * math.log10(N_BINS)
                    for value in channel_snr_values
                ],
                "seeds": list(seeds),
                "max_packets": int(args.max_packets),
                "noise_model": "complex AWGN flat in the 125 kHz LoRa channel",
                "truth_usage": "post-hoc acquisition-basin labelling only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

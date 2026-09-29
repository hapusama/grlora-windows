"""PC-1/PC-2: coherent preamble candidates vs noisy FrameSync availability.

Reproduces the official 560-trial AWGN protocol bit-for-bit (same
SeedSequence/noise calibration) and replaces only the synchronization stage
with the coherent preamble candidate generator, then measures:

* PC-1 availability: does the candidate list contain a decodable point
  (hard decoder, CRC + exact payload), and at which rank;
* PC-2 end-to-end: arbitration PDR with the soft decoder over the same list.

The coarse payload-start hint comes from the clean capture (documented
assumption: coarse packet detection is given; this experiment isolates the
estimation stage).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
WEAK_PACKET_ROOT = SCRIPT_PATH.parents[3]
if str(WEAK_PACKET_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_PACKET_ROOT))

from weak_decoder.os_lora.experiment_support.noisy_framesync_ota import (  # noqa: E402
    load_json,
    packet_metadata_paths,
    unit_lora_band_awgn,
    write_csv_rows,
)
from weak_decoder.os_lora.system.decoder_aware_crc import (  # noqa: E402
    decode_savaux_sync_candidate,
)
from weak_decoder.os_lora.system.soft_hamming_crc import (  # noqa: E402
    decode_soft_hamming_sync_candidate,
)
from weak_decoder.os_lora.system.ambiguity_ridge_list import (  # noqa: E402
    arbitrate_sync_list_with_crc,
)
from weak_decoder.os_lora.system.preamble_coherent_candidates import (  # noqa: E402
    coherent_preamble_candidates,
)

SF = 12
BW_HZ = 125_000.0
OS_FACTOR = 8
N_BINS = 1 << SF
DEMOD_TAIL_SAMPLES = 256
LDRO_MODE = 1
CRC_MODE = "grlora"


def _trial_row(task: dict[str, Any]) -> dict[str, Any]:
    packet = task["packet"]
    esn0_db = float(task["esn0_db"])
    seed = int(task["seed"])

    clean = np.fromfile(Path(str(packet["iq_path"])), dtype=np.dtype("<c8"))
    rng = np.random.default_rng(
        np.random.SeedSequence((seed, int(packet["reference_id"]), 73013))
    )
    noise_power = float(packet["signal_power"]) * N_BINS / (
        10.0 ** (esn0_db / 10.0)
    )
    noise = unit_lora_band_awgn(rng, clean.size, os_factor=OS_FACTOR)
    noisy = np.asarray(clean + noise * math.sqrt(noise_power), dtype=np.complex64)
    tail = unit_lora_band_awgn(rng, DEMOD_TAIL_SAMPLES, os_factor=OS_FACTOR)
    decode_samples = np.concatenate(
        [noisy, tail * math.sqrt(noise_power)]
    ).astype(np.complex64)
    expected = bytes.fromhex(str(packet["expected_frame_hex"]))

    started = time.perf_counter()
    candidates = coherent_preamble_candidates(
        noisy,
        sf=SF,
        os_factor=OS_FACTOR,
        payload_start_hint=int(packet["clean_fine_payload_start_sample"]),
    )
    candidate_ms = 1e3 * (time.perf_counter() - started)

    # PC-1: soft-decode along the list until CRC+exact success (soft is the
    # verifier that stays comparable to the official soft-oracle baseline;
    # hard decoding is dead below ~13 dB even with perfect sync).
    hit_rank = 0
    hard_attempts = 0
    decode_started = time.perf_counter()
    for index, candidate in enumerate(candidates, start=1):
        hard_attempts = index
        decoded = decode_soft_hamming_sync_candidate(
            decode_samples,
            candidate.as_frame_sync(),
            sf=SF,
            bw_hz=BW_HZ,
            os_factor=OS_FACTOR,
            ldro_mode=LDRO_MODE,
            crc_mode=CRC_MODE,
            allow_gate_failed_candidate=True,
        )
        if (
            decoded.header_valid
            and decoded.crc_valid
            and decoded.payload_bytes == expected
        ):
            hit_rank = index
            break
    hard_ms = 1e3 * (time.perf_counter() - decode_started)

    # PC-2: soft arbitration over the same list (early stop on CRC).
    arbitration_started = time.perf_counter()
    arbitration = arbitrate_sync_list_with_crc(
        decode_samples,
        [candidate.as_frame_sync() for candidate in candidates],
        sf=SF,
        bw_hz=BW_HZ,
        os_factor=OS_FACTOR,
        ldro_mode=LDRO_MODE,
        crc_mode=CRC_MODE,
        require_payload_crc=True,
        stop_on_crc=True,
        decoder_mode="soft_hamming",
    )
    arbitration_ms = 1e3 * (time.perf_counter() - arbitration_started)
    selected = arbitration.selected
    soft_exact = bool(
        selected is not None
        and selected.decode.header_valid
        and selected.decode.crc_valid
        and selected.decode.payload_bytes == expected
    )

    primary = candidates[0] if candidates else None
    return {
        "trial_id": f"{packet['packet_id']}:{esn0_db:g}:{seed}",
        "packet_id": str(packet["packet_id"]),
        "reference_id": int(packet["reference_id"]),
        "esn0_db": esn0_db,
        "seed": seed,
        "candidate_count": len(candidates),
        "coherent_hit": int(hit_rank > 0),
        "coherent_hit_rank": hit_rank,
        "hard_decode_attempts": hard_attempts,
        "soft_arbitration_delivered": int(soft_exact),
        "soft_arbitration_attempts": len(arbitration.attempts),
        "primary_cfo_total_bins": ""
        if primary is None
        else float(primary.cfo_total_bins),
        "clean_cfo_total_bins": float(packet["clean_cfo_total_bins"]),
        "candidate_ms": candidate_ms,
        "hard_ms": hard_ms,
        "arbitration_ms": arbitration_ms,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-repo",
        type=Path,
        default=WEAK_PACKET_ROOT.parent.parent / "lora-rfsr-savaux",
    )
    parser.add_argument(
        "--clean-audit",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "ambiguity_ridge_soft_hamming_ota_awgn_10seeds_20260820"
        / "clean_sync_audit.csv",
    )
    parser.add_argument("--esn0-db", default="10,11,12,13")
    parser.add_argument(
        "--seeds",
        default="20260821,20260822,20260823,20260824,20260825,20260826,20260827,20260828,20260829,20260830",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "preamble_coherent_candidates_ota_20260915",
    )
    args = parser.parse_args(argv)

    ota_root = args.dataset_repo.resolve() / "data" / "reference_phy" / "rfsr_db"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    esn0_values = [float(v) for v in str(args.esn0_db).split(",") if v.strip()]
    seeds = [int(v) for v in str(args.seeds).split(",") if v.strip()]

    with open(args.clean_audit, newline="", encoding="utf-8") as handle:
        audit = {row["packet_id"]: row for row in csv.DictReader(handle)}

    packets: list[dict[str, Any]] = []
    for metadata_path in list(packet_metadata_paths(ota_root))[:8]:
        metadata = load_json(metadata_path)
        packet_id = str(metadata["ota_id"])
        row = audit.get(packet_id)
        if row is None or int(row["full_payload_exact"]) != 1:
            continue
        reference = load_json(
            ota_root.parent
            / "metadata"
            / f"{int(metadata['reference']['reference_id']):06d}.json"
        )
        packets.append(
            {
                "packet_id": packet_id,
                "reference_id": int(metadata["reference"]["reference_id"]),
                "iq_path": str(ota_root / str(metadata["ota"]["relative_path"])),
                "signal_power": float(row["signal_power"]),
                "clean_cfo_total_bins": float(row["cfo_total_bins"]),
                "clean_fine_payload_start_sample": int(
                    row["fine_payload_start_sample"]
                ),
                "expected_frame_hex": str(reference["packet"]["frame_hex"]),
            }
        )
    if len(packets) != 8:
        raise RuntimeError(f"expected 8 admitted packets, got {len(packets)}")

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        tasks = [
            {"packet": packet, "esn0_db": esn0, "seed": seed}
            for esn0 in esn0_values
            for seed in seeds
            for packet in packets
        ]
        futures = [executor.submit(_trial_row, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 20 == 0 or index == len(futures):
                print(f"completed {index}/{len(futures)} trials", flush=True)

    rows.sort(
        key=lambda r: (float(r["esn0_db"]), int(r["seed"]), int(r["reference_id"]))
    )
    write_csv_rows(output_dir / "packet_trials.csv", rows)

    summary: list[dict[str, Any]] = []
    for esn0 in esn0_values:
        selected = [r for r in rows if float(r["esn0_db"]) == esn0]
        summary.append(
            {
                "esn0_db": esn0,
                "trials": len(selected),
                "coherent_hit_rate": sum(int(r["coherent_hit"]) for r in selected)
                / len(selected),
                "mean_hit_rank": float(
                    np.mean(
                        [
                            int(r["coherent_hit_rank"])
                            for r in selected
                            if int(r["coherent_hit_rank"]) > 0
                        ]
                        or [0]
                    )
                ),
                "soft_arbitration_pdr": sum(
                    int(r["soft_arbitration_delivered"]) for r in selected
                )
                / len(selected),
                "mean_candidate_ms": float(
                    np.mean([float(r["candidate_ms"]) for r in selected])
                ),
                "mean_hard_ms": float(
                    np.mean([float(r["hard_ms"]) for r in selected])
                ),
                "mean_arbitration_ms": float(
                    np.mean([float(r["arbitration_ms"]) for r in selected])
                ),
            }
        )
    write_csv_rows(output_dir / "summary.csv", summary)
    config = {
        "sf": SF,
        "bw_hz": BW_HZ,
        "os_factor": OS_FACTOR,
        "esn0_db": esn0_values,
        "seeds": seeds,
        "noise_protocol": "identical to evaluate_decoder_aware_crc_pdr_ota",
        "hint": "clean capture payload start (coarse detection given)",
        "decoder_hard": "savaux hard via decode_savaux_sync_candidate",
        "decoder_soft": "soft_hamming arbitration",
        "rows": len(rows),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

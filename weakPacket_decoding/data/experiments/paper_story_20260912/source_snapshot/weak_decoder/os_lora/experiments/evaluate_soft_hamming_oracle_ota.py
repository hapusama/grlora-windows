"""Evaluate Savaux hard versus soft-Hamming decoding at oracle sync.

The runner reuses packet/SNR/seed/noise-power rows from a whole-packet AWGN
experiment.  It reproduces the identical noisy IQ but intentionally skips the
expensive noisy FrameSync: both decoders receive the clean synchronization
coordinate.  This isolates decoder sensitivity from synchronization gains.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
WEAK_PACKET_ROOT = SCRIPT_PATH.parents[3]
GR_LORA_ROOT = SCRIPT_PATH.parents[4]
WORKSPACE_ROOT = GR_LORA_ROOT.parent
if str(WEAK_PACKET_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_PACKET_ROOT))

from weak_decoder.os_lora.experiment_support.noisy_framesync_ota import (  # noqa: E402
    load_json,
    packet_metadata_paths,
    parse_csv_list,
    read_csv_rows,
    unit_lora_band_awgn,
    write_csv_rows,
)
from weak_decoder.os_lora.system.soft_hamming_crc import (  # noqa: E402
    decode_soft_hamming_sync_candidate,
)


SF = 12
N_BINS = 1 << SF
BW_HZ = 125_000.0
OS_FACTOR = 8
LDRO_MODE = 1
CRC_MODE = "grlora"
DEMOD_TAIL_SAMPLES = 256


def _split_cfo(total_bins: float) -> tuple[int, float]:
    integer = int(math.floor(float(total_bins) + 0.5))
    return integer, float(total_bins - integer)


def _metadata_by_reference(ota_root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in packet_metadata_paths(ota_root):
        metadata = load_json(path)
        result.setdefault(int(metadata["reference"]["reference_id"]), path)
    return result


def _worker(task: dict[str, Any]) -> list[dict[str, Any]]:
    baselines = [dict(row) for row in task["baselines"]]
    if not baselines:
        return []
    first_baseline = baselines[0]
    ota_root = Path(str(task["ota_root"]))
    metadata = load_json(Path(str(task["metadata_path"])))
    clean = dict(task["clean_sync"])
    reference_id = int(first_baseline["reference_id"])
    reference = load_json(
        ota_root.parent / "metadata" / f"{reference_id:06d}.json"
    )
    expected = bytes.fromhex(str(reference["packet"]["frame_hex"]))
    samples = np.fromfile(
        ota_root / str(metadata["ota"]["relative_path"]),
        dtype=np.dtype("<c8"),
    )
    seed = int(first_baseline["seed"])
    if any(
        int(row["reference_id"]) != reference_id or int(row["seed"]) != seed
        for row in baselines
    ):
        raise ValueError("grouped baselines must share reference_id and seed")
    rng = np.random.default_rng(
        np.random.SeedSequence((seed, reference_id, 73013))
    )
    unit_noise = unit_lora_band_awgn(
        rng,
        samples.size,
        os_factor=OS_FACTOR,
    )
    unit_tail = unit_lora_band_awgn(
        rng,
        DEMOD_TAIL_SAMPLES,
        os_factor=OS_FACTOR,
    )
    cfo_int, cfo_frac = _split_cfo(float(clean["cfo_total_bins"]))
    frame_sync = SimpleNamespace(
        valid=True,
        fine_payload_start_sample=int(clean["fine_payload_start_sample"]),
        cfo_int_est=cfo_int,
        cfo_frac_est=cfo_frac,
        sfo_hat=float(clean["sfo_hat"]),
        sfo_cum_initial=float(clean["sfo_cum_initial"]),
    )
    output: list[dict[str, Any]] = []
    for baseline in baselines:
        scale = math.sqrt(float(baseline["added_noise_power"]))
        noisy = np.asarray(samples + unit_noise * scale, dtype=np.complex64)
        tail = np.asarray(unit_tail * scale, dtype=np.complex64)
        decode_samples = np.concatenate([noisy, tail]).astype(np.complex64)
        decoded = decode_soft_hamming_sync_candidate(
            decode_samples,
            frame_sync,
            sf=SF,
            bw_hz=BW_HZ,
            os_factor=OS_FACTOR,
            ldro_mode=LDRO_MODE,
            crc_mode=CRC_MODE,
            allow_gate_failed_candidate=False,
        )
        soft_accept = bool(decoded.header_valid and decoded.crc_valid)
        soft_exact = bool(soft_accept and decoded.payload_bytes == expected)
        hard_exact = bool(int(baseline["oracle_packet_delivered"]))
        output.append(
            {
                "trial_id": str(baseline["trial_id"]),
                "reference_id": reference_id,
                "esn0_db": float(baseline["esn0_db"]),
                "seed": seed,
                "hard_oracle_packet_delivered": int(hard_exact),
                "soft_oracle_packet_delivered": int(soft_exact),
                "soft_crc_accept": int(soft_accept),
                "soft_crc_false_delivery": int(
                    soft_accept and not soft_exact
                ),
                "soft_fix": int(soft_exact and not hard_exact),
                "soft_break": int(hard_exact and not soft_exact),
                "soft_decode_status": str(decoded.status),
                "soft_header_valid": int(decoded.header_valid),
                "soft_crc_valid": int(decoded.crc_valid),
                "soft_payload_hex": decoded.payload_bytes.hex(),
            }
        )
    return output


def _summary(
    rows: Sequence[dict[str, Any]],
    esn0_values: Sequence[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for esn0_db in esn0_values:
        selected = [
            row for row in rows if float(row["esn0_db"]) == float(esn0_db)
        ]
        if not selected:
            continue
        count = len(selected)
        output.append(
            {
                "esn0_db": float(esn0_db),
                "trials": count,
                "hard_oracle_crc_pdr": float(
                    np.mean(
                        [
                            int(row["hard_oracle_packet_delivered"])
                            for row in selected
                        ]
                    )
                ),
                "soft_oracle_crc_pdr": float(
                    np.mean(
                        [
                            int(row["soft_oracle_packet_delivered"])
                            for row in selected
                        ]
                    )
                ),
                "soft_fixes": sum(int(row["soft_fix"]) for row in selected),
                "soft_breaks": sum(int(row["soft_break"]) for row in selected),
                "soft_crc_false_deliveries": sum(
                    int(row["soft_crc_false_delivery"]) for row in selected
                ),
            }
        )
    return output


def _write_report(
    output_dir: Path,
    summary: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# Oracle-sync Savaux hard versus soft-Hamming CRC",
        "",
        "Both decoders use identical noisy IQ and clean synchronization.",
        "",
        "| Es/N0 | trials | hard PDR | soft-Hamming PDR | fixes | breaks | false CRC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {esn0_db:g} | {trials} | {hard_oracle_crc_pdr:.3f} | "
            "{soft_oracle_crc_pdr:.3f} | {soft_fixes} | {soft_breaks} | "
            "{soft_crc_false_deliveries} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Expected bytes are used only after CRC for exact-delivery audit.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate bounded soft-Hamming decoding at oracle sync."
    )
    parser.add_argument(
        "--dataset-repo",
        type=Path,
        default=WORKSPACE_ROOT / "lora-rfsr-savaux",
    )
    parser.add_argument("--ota-root", type=Path, default=None)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--esn0-db", default="10,11,12,13,14,15,16")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    dataset_repo = args.dataset_repo.resolve()
    ota_root = (
        args.ota_root.resolve()
        if args.ota_root is not None
        else dataset_repo / "data" / "reference_phy" / "rfsr_db"
    )
    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    esn0_values = parse_csv_list(args.esn0_db, float)
    requested = {float(value) for value in esn0_values}
    baselines = [
        row
        for row in read_csv_rows(baseline_dir / "packet_trials.csv")
        if float(row["esn0_db"]) in requested
    ]
    clean_by_reference = {
        int(row["reference_id"]): row
        for row in read_csv_rows(baseline_dir / "clean_sync_audit.csv")
        if int(row["full_payload_exact"])
    }
    metadata = _metadata_by_reference(ota_root)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in baselines:
        key = (int(row["reference_id"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    tasks = [
        {
            "baselines": sorted(
                group,
                key=lambda row: float(row["esn0_db"]),
            ),
            "ota_root": str(ota_root),
            "metadata_path": str(metadata[reference_id]),
            "clean_sync": clean_by_reference[reference_id],
        }
        for (reference_id, _seed), group in grouped.items()
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if index % 8 == 0 or index == len(futures):
                print(
                    f"completed soft-Hamming noise groups: {index}/{len(futures)}",
                    flush=True,
                )
    rows.sort(
        key=lambda row: (
            float(row["esn0_db"]),
            int(row["seed"]),
            int(row["reference_id"]),
        )
    )
    summary = _summary(rows, esn0_values)
    write_csv_rows(output_dir / "packet_trials.csv", rows)
    write_csv_rows(output_dir / "summary.csv", summary)
    _write_report(output_dir, summary)
    config = {
        "baseline_dir": str(baseline_dir),
        "esn0_db": list(esn0_values),
        "workers": int(args.workers),
        "decoder": "Savaux likelihood -> bit marginal -> 16-word Hamming ML -> PHY CRC",
        "synchronization": "clean/oracle coordinate",
        "ground_truth_usage": "post-CRC exact-delivery audit only",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

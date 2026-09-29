"""K-copy independent-noise replay: how far below the single-packet cliff?

Simulates K retransmissions of the same packet by adding K independent
band-limited AWGN realizations to one clean OTA capture (optimistic vs real
retransmissions: no inter-copy channel variation).  Each copy is synchronized
independently with PC-Ridge (candidate 0), then the per-symbol Savaux power
spectra of all copies are noncoherently averaged and decoded once through the
soft-Hamming + CRC chain.  This measures the operating-point shift
10*log10(K) buys and the per-copy sync floor of PC-Ridge.

Scale conversion: the SER tools' "-20 dB" axis is per-sample SNR of the
1 MS/s signal; the Es/N0 protocol here satisfies per-sample SNR = Es/N0 -
10*log10(2^SF) = Es/N0 - 36.1 dB.
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
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.decoding.header_first_demod import (  # noqa: E402
    advance_symbol_cursor,
    decode_explicit_header,
)
from weak_decoder.decoding.payload_codec import (  # noqa: E402
    decode_explicit_frame_symbols,
)
from weak_decoder.os_lora.system.preamble_coherent_candidates import (  # noqa: E402
    coherent_preamble_axes_power,
    coherent_preamble_candidates,
    coherent_preamble_complex_stack,
    shared_bu_from_axes_power,
)
from weak_decoder.os_lora.system.soft_hamming_crc import (  # noqa: E402
    soft_repair_interleaver_block,
)

SF = 12
BW_HZ = 125_000.0
OS_FACTOR = 8
N_BINS = 1 << SF
DEMOD_TAIL_SAMPLES = 256
LDRO_MODE = 1
CRC_MODE = "grlora"


def _copy_frame_sync(samples: np.ndarray, hint: int) -> Any | None:
    candidates = coherent_preamble_candidates(
        samples,
        sf=SF,
        os_factor=OS_FACTOR,
        payload_start_hint=int(hint),
    )
    return candidates[0].as_frame_sync() if candidates else None


def _symbol_power(
    samples: np.ndarray,
    frame_sync: Any,
    symbol_index: int,
    header_start: int,
    payload_ldro: bool,
) -> np.ndarray | None:
    """Per-symbol Savaux power spectrum under one copy's sync coordinates."""

    n_s = N_BINS * OS_FACTOR
    cursor = int(frame_sync.fine_payload_start_sample)
    sfo_cum = float(frame_sync.sfo_cum_initial)
    for _ in range(int(symbol_index)):
        cursor, sfo_cum, _adjust = advance_symbol_cursor(
            cursor,
            sf=SF,
            os_factor=OS_FACTOR,
            sfo_cum=sfo_cum,
            sfo_hat=float(frame_sync.sfo_hat),
        )
    s0 = cursor + OS_FACTOR // 2
    if s0 < 0 or s0 + n_s > samples.size:
        return None
    spectrum = paper_oversampled_spectrum(
        samples,
        s0,
        sf=SF,
        os_factor=OS_FACTOR,
        cfo_int=int(frame_sync.cfo_int_est),
        cfo_frac=float(frame_sync.cfo_frac_est),
        header_start_sample=int(header_start) + OS_FACTOR // 2,
        cfo_correction_mode="continuous",
    )[0]
    return np.abs(spectrum).astype(np.float64) ** 2


def _symbol_complex(
    samples: np.ndarray,
    frame_sync: Any,
    symbol_index: int,
    header_start: int,
    payload_ldro: bool,
) -> np.ndarray | None:
    """Complex per-symbol Savaux spectrum (for coherent combining)."""

    n_s = N_BINS * OS_FACTOR
    cursor = int(frame_sync.fine_payload_start_sample)
    sfo_cum = float(frame_sync.sfo_cum_initial)
    for _ in range(int(symbol_index)):
        cursor, sfo_cum, _adjust = advance_symbol_cursor(
            cursor,
            sf=SF,
            os_factor=OS_FACTOR,
            sfo_cum=sfo_cum,
            sfo_hat=float(frame_sync.sfo_hat),
        )
    s0 = cursor + OS_FACTOR // 2
    if s0 < 0 or s0 + n_s > samples.size:
        return None
    return paper_oversampled_spectrum(
        samples,
        s0,
        sf=SF,
        os_factor=OS_FACTOR,
        cfo_int=int(frame_sync.cfo_int_est),
        cfo_frac=float(frame_sync.cfo_frac_est),
        header_start_sample=int(header_start) + OS_FACTOR // 2,
        cfo_correction_mode="continuous",
    )[0].astype(np.complex128)


def decode_multicopy_soft(
    copies: Sequence[np.ndarray],
    frame_syncs: Sequence[Any],
    weights: Sequence[complex] | None = None,
) -> dict[str, Any]:
    """Cross-copy combining + single soft-Hamming/CRC decode.

    ``weights=None`` averages power spectra (noncoherent); with per-copy
    phase weights it coherently sums complex spectra first.
    """

    n_copies = len(copies)
    if n_copies == 0 or any(fs is None for fs in frame_syncs):
        return {"status": "sync_missing", "crc_valid": False, "payload": None}

    def averaged_power(symbol_index: int, payload_ldro: bool) -> np.ndarray | None:
        if weights is None:
            total = None
            for samples, fs in zip(copies, frame_syncs):
                power = _symbol_power(
                    samples,
                    fs,
                    symbol_index=symbol_index,
                    header_start=int(frame_syncs[0].fine_payload_start_sample),
                    payload_ldro=payload_ldro,
                )
                if power is None:
                    return None
                total = power if total is None else total + power
            return total / float(n_copies)
        combined = None
        for index, (samples, fs) in enumerate(zip(copies, frame_syncs)):
            spectrum = _symbol_complex(
                samples,
                fs,
                symbol_index=symbol_index,
                header_start=int(frame_syncs[0].fine_payload_start_sample),
                payload_ldro=payload_ldro,
            )
            if spectrum is None:
                return None
            term = spectrum * weights[index]
            combined = term if combined is None else combined + term
        # Coherent: complex amplitudes sum BEFORE the modulus — the K^2
        # signal gain vs K noise gain is the entire point.
        return np.abs(combined) ** 2

    header_powers = [averaged_power(i, False) for i in range(8)]
    if any(p is None for p in header_powers):
        return {"status": "truncated", "crc_valid": False, "payload": None}
    repaired_header = soft_repair_interleaver_block(
        header_powers,
        sf=SF,
        is_header=True,
        cr=4,
        ldro=False,
    )
    header = decode_explicit_header(
        repaired_header.symbol_values,
        sf=SF,
        bw=BW_HZ,
        ldro_mode=LDRO_MODE,
    )
    if not header.header_valid or not 1 <= int(header.cr) <= 4:
        return {
            "status": "header_invalid",
            "crc_valid": False,
            "payload": None,
            "header": header,
        }

    payload_powers = [
        averaged_power(8 + i, bool(header.ldro))
        for i in range(int(header.payload_symbol_count))
    ]
    if any(p is None for p in payload_powers):
        return {"status": "truncated", "crc_valid": False, "payload": None}
    cw_len = int(header.cr) + 4
    payload_values: list[int] = []
    for offset in range(0, len(payload_powers), cw_len):
        block = payload_powers[offset : offset + cw_len]
        if len(block) != cw_len:
            return {"status": "block_truncated", "crc_valid": False, "payload": None}
        payload_values.extend(
            soft_repair_interleaver_block(
                block,
                sf=SF,
                is_header=False,
                cr=int(header.cr),
                ldro=bool(header.ldro),
            ).symbol_values
        )
    frame = decode_explicit_frame_symbols(
        repaired_header.symbol_values,
        payload_values,
        sf=SF,
        bw=BW_HZ,
        ldro_mode=LDRO_MODE,
        crc_mode=CRC_MODE,
    )
    return {
        "status": "ok",
        "crc_valid": bool(frame.payload.crc_valid),
        "payload": frame.payload.payload_bytes,
        "header": header,
    }


def _shared_frame_sync(
    noisy_copies: Sequence[np.ndarray], hint: int, coherent: bool = False
) -> tuple[Any, float, list[complex] | None]:
    """Cross-copy shared b_u (average 2-D coherent power, then peak).

    With ``coherent`` also returns per-copy phase weights taken from the
    complex chirp-axis stack at the shared (axis, bin) peak, for cross-copy
    coherent spectrum combining.
    """

    total = None
    stacks: list[np.ndarray] | None = [] if coherent else None
    for noisy in noisy_copies:
        if coherent:
            stack, _w0 = coherent_preamble_complex_stack(
                noisy,
                sf=SF,
                os_factor=OS_FACTOR,
                payload_start_hint=int(hint),
            )
            stacks.append(stack)
            power2d = np.abs(stack) ** 2
        else:
            power2d, _w0 = coherent_preamble_axes_power(
                noisy,
                sf=SF,
                os_factor=OS_FACTOR,
                payload_start_hint=int(hint),
            )
        total = power2d if total is None else total + power2d
    avg = total / float(len(noisy_copies))
    b_u = shared_bu_from_axes_power(avg)
    weights: list[complex] | None = None
    if coherent:
        per_bin = np.max(avg, axis=0)
        per_axis = np.argmax(avg, axis=0)
        b_star = int(np.argmax(per_bin))
        axis_star = int(per_axis[b_star])
        ref = stacks[0][axis_star, b_star]
        ref_angle = np.angle(ref) if abs(ref) > 0 else 0.0
        weights = [
            np.exp(
                -1j
                * (
                    np.angle(stacks[k][axis_star, b_star]) - ref_angle
                )
            )
            for k in range(len(stacks))
        ]
    cfo = float(b_u) % float(N_BINS)
    cfo_int = int(math.floor(cfo + 0.5))
    signed = cfo if cfo < N_BINS / 2 else cfo - N_BINS
    frame_sync = __import__("types").SimpleNamespace(
        valid=True,
        fine_payload_start_sample=int(hint),
        cfo_int_est=cfo_int,
        cfo_frac_est=cfo - cfo_int,
        sfo_hat=signed * BW_HZ / 487_700_000.0,
        sfo_cum_initial=0.0,
    )
    return frame_sync, float(b_u), weights


def _trial_row(task: dict[str, Any]) -> dict[str, Any]:
    packet = task["packet"]
    esn0_db = float(task["esn0_db"])
    seed = int(task["seed"])
    n_copies = int(task["n_copies"])
    shared_sync = bool(task.get("shared_sync", False))
    combining = str(task.get("combining", "power"))

    clean = np.fromfile(Path(str(packet["iq_path"])), dtype=np.dtype("<c8"))
    hint = int(packet["clean_fine_payload_start_sample"])
    noise_power = float(packet["signal_power"]) * N_BINS / (
        10.0 ** (esn0_db / 10.0)
    )
    copies: list[np.ndarray] = []
    noisy_copies: list[np.ndarray] = []
    for copy_index in range(n_copies):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                (seed, int(packet["reference_id"]), 73013, copy_index)
            )
        )
        noise = unit_lora_band_awgn(rng, clean.size, os_factor=OS_FACTOR)
        noisy = np.asarray(
            clean + noise * math.sqrt(noise_power), dtype=np.complex64
        )
        tail = unit_lora_band_awgn(rng, DEMOD_TAIL_SAMPLES, os_factor=OS_FACTOR)
        decode_samples = np.concatenate(
            [noisy, tail * math.sqrt(noise_power)]
        ).astype(np.complex64)
        noisy_copies.append(noisy)
        copies.append(decode_samples)

    sync_ok = 0
    b_u_errors: list[float] = []
    weights = None
    if shared_sync:
        frame_sync, b_u, weights = _shared_frame_sync(
            noisy_copies, hint, coherent=(combining == "coherent")
        )
        frame_syncs = [frame_sync] * n_copies
        sync_ok = n_copies
        err = (b_u - float(packet["clean_cfo_total_bins"]) - 0.2) % N_BINS
        b_u_errors.append(err if err < N_BINS / 2 else err - N_BINS)
    else:
        frame_syncs = []
        for noisy in noisy_copies:
            fs = _copy_frame_sync(noisy, hint)
            frame_syncs.append(fs)
            if fs is not None:
                sync_ok += 1
                cfo = float(fs.cfo_int_est) + float(fs.cfo_frac_est)
                b_u = cfo + 0.2  # sto of the anchor member is ~-0.2 chips
                err = (b_u - (float(packet["clean_cfo_total_bins"]) - 0.2)) % N_BINS
                b_u_errors.append(err if err < N_BINS / 2 else err - N_BINS)

    started = time.perf_counter()
    result = decode_multicopy_soft(copies, frame_syncs, weights)
    decode_ms = 1e3 * (time.perf_counter() - started)
    expected = bytes.fromhex(str(packet["expected_frame_hex"]))
    exact = bool(
        result["crc_valid"] and result["payload"] == expected
    )
    return {
        "trial_id": f"{packet['packet_id']}:{esn0_db:g}:{seed}:{n_copies}",
        "packet_id": str(packet["packet_id"]),
        "esn0_db": esn0_db,
        "seed": seed,
        "copies": n_copies,
        "shared_sync": int(shared_sync),
        "sync_ok": sync_ok,
        "b_u_error_mean": float(np.mean(b_u_errors)) if b_u_errors else "",
        "decoded": int(exact),
        "crc_only": int(bool(result["crc_valid"]) and not exact),
        "status": str(result["status"]),
        "decode_ms": decode_ms,
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
    parser.add_argument("--esn0-db", default="2,4,6,8,10")
    parser.add_argument("--seeds", default="20260821,20260822,20260823")
    parser.add_argument("--copies", default="1,2,4,8")
    parser.add_argument(
        "--shared-sync",
        action="store_true",
        help="Average the 2-D coherent preamble power across copies before "
        "peak-finding (cross-copy assisted sync).",
    )
    parser.add_argument(
        "--combining",
        choices=("power", "coherent"),
        default="power",
        help="power: average per-symbol power spectra; coherent: weight "
        "complex spectra by per-copy preamble phase before summing.",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "multicopy_replay_ota_20260915",
    )
    args = parser.parse_args(argv)

    ota_root = args.dataset_repo.resolve() / "data" / "reference_phy" / "rfsr_db"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    esn0_values = [float(v) for v in str(args.esn0_db).split(",") if v.strip()]
    seeds = [int(v) for v in str(args.seeds).split(",") if v.strip()]
    copy_counts = [int(v) for v in str(args.copies).split(",") if v.strip()]

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
    tasks = [
        {
            "packet": packet,
            "esn0_db": esn0,
            "seed": seed,
            "n_copies": n,
            "shared_sync": bool(args.shared_sync),
            "combining": str(args.combining),
        }
        for esn0 in esn0_values
        for seed in seeds
        for n in copy_counts
        for packet in packets
    ]
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [executor.submit(_trial_row, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 20 == 0 or index == len(futures):
                print(f"completed {index}/{len(futures)} trials", flush=True)
            # Checkpoint so a worker crash cannot lose the whole grid.
            if index % 80 == 0:
                write_csv_rows(output_dir / "packet_trials_partial.csv", rows)

    rows.sort(
        key=lambda r: (
            float(r["esn0_db"]),
            int(r["copies"]),
            int(r["seed"]),
            r["packet_id"],
        )
    )
    write_csv_rows(output_dir / "packet_trials.csv", rows)

    summary: list[dict[str, Any]] = []
    for esn0 in esn0_values:
        for n in copy_counts:
            selected = [
                r
                for r in rows
                if float(r["esn0_db"]) == esn0 and int(r["copies"]) == n
            ]
            if not selected:
                continue
            summary.append(
                {
                    "esn0_db": esn0,
                    "copies": n,
                    "trials": len(selected),
                    "sync_rate": sum(int(r["sync_ok"]) for r in selected)
                    / len(selected),
                    "pdr": sum(int(r["decoded"]) for r in selected)
                    / len(selected),
                    "crc_only": sum(int(r["crc_only"]) for r in selected),
                    "mean_decode_ms": float(
                        np.mean([float(r["decode_ms"]) for r in selected])
                    ),
                }
            )
    write_csv_rows(output_dir / "summary.csv", summary)
    config = {
        "sf": SF,
        "esn0_db": esn0_values,
        "seeds": seeds,
        "copies": copy_counts,
        "per_sample_snr_db": [e - 36.1 for e in esn0_values],
        "combining": "noncoherent power averaging of Savaux spectra",
        "sync": "PC-Ridge candidate 0 per copy",
        "note": "independent-noise replay is optimistic vs real retransmissions",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

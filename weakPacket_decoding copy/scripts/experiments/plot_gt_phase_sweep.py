#!/usr/bin/env python3
"""Plot preamble/header/payload GT-bin phase trajectories across SNRs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WEAK_ROOT.parent
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_downchirp, build_upchirp  # noqa: E402
from weak_decoder.grlora_frame_sync import _extract_chip_rate_chirps  # noqa: E402
from weak_decoder.preamble_detector import PreambleDetectorConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    dataset = "0_0_0_10_14_16"
    default_noisy = WEAK_ROOT / "data" / "low_snr_gt_bin" / f"{dataset}_m22_m27_sto_input"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=dataset)
    parser.add_argument("--packet", type=int, default=10)
    parser.add_argument("--snrs", type=float, nargs="+", default=[-22.0, -23.0, -24.0, -25.0, -26.0, -27.0])
    parser.add_argument("--noisy-dir", type=Path, default=default_noisy)
    parser.add_argument(
        "--front-grid",
        choices=("fine_payload_backtrack", "frame_sync_raw", "synced_preamble", "fine_preamble", "payload_backtrack"),
        default="fine_payload_backtrack",
        help="Sampling grid for preamble/sync/SFD phase extraction.",
    )
    parser.add_argument(
        "--clean-iq",
        type=Path,
        default=None,
        help="Clean IQ file used to lock SFD bins on the common fine grid.",
    )
    parser.add_argument(
        "--framesync-peaks-csv",
        type=Path,
        default=None,
        help="Frame-sync peak CSV used by --front-grid frame_sync_raw.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "phase_line" / "gt_preamble_header_payload_sweep",
    )
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else float(default)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _infer_params(dataset: str) -> tuple[int, int]:
    parts = dataset.split("_")
    return int(parts[3]), int(parts[-1])


def _symbol_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    return int(start_sample) + int(os_factor // 2) + int(os_factor) * np.arange(n_bins, dtype=np.int64)


def _phase_at_raw_bin(
    samples: np.ndarray,
    row: dict[str, str],
    header_start_sample: int,
) -> tuple[float, float, int]:
    sf = _int(row, "sf", 10)
    os_factor = _int(row, "os_factor", 4)
    n_bins = 1 << sf
    start_sample = _int(row, "start_sample")
    raw_bin = _int(row, "raw_fft_bin", 0) % n_bins
    indexes = _symbol_indexes(start_sample, sf, os_factor)
    symbol = np.asarray(samples[indexes], dtype=np.complex64)
    cfo_total = _float(row, "cfo_int", 0.0) + _float(row, "cfo_frac", 0.0)
    relative_chip_start = float(start_sample - int(header_start_sample)) / float(os_factor)
    cfo_common_phase = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
    symbol = (symbol * np.exp(-1j * cfo_common_phase)).astype(np.complex64)
    downchirp = build_downchirp(sf, cfo_int=_int(row, "cfo_int", 0), cfo_frac=_float(row, "cfo_frac", 0.0))
    spectrum = np.fft.fft(symbol * downchirp).astype(np.complex64)
    value = complex(spectrum[raw_bin])
    return float(math.atan2(value.imag, value.real)), float(abs(value)), int(raw_bin)


def _center_symbol_phase(
    samples: np.ndarray,
    start_sample: int,
    raw_bin: int,
    config: PreambleDetectorConfig,
    downchirp: np.ndarray,
    cfo_total: float,
    header_start: int,
) -> tuple[float, float]:
    n_bins = int(config.n_bins)
    os_factor = int(config.os_factor)
    chirp = _extract_chip_rate_chirps(samples, int(start_sample), config, 1, sample_correction=0)[0]
    rel = float(int(start_sample) - int(header_start)) / float(os_factor)
    cfo_phase = float(2.0 * math.pi * float(cfo_total) * rel / n_bins)
    chirp = (chirp * np.exp(-1j * cfo_phase)).astype(np.complex64)
    spectrum = np.fft.fft(chirp * downchirp).astype(np.complex64)
    value = complex(spectrum[int(raw_bin) % n_bins])
    return float(math.atan2(value.imag, value.real)), float(abs(value))


def _center_symbol_phase_with_reference(
    samples: np.ndarray,
    start_sample: int,
    raw_bin: int,
    config: PreambleDetectorConfig,
    reference: np.ndarray,
    cfo_total: float,
    header_start: int,
) -> tuple[float, float]:
    n_bins = int(config.n_bins)
    os_factor = int(config.os_factor)
    chirp = _extract_chip_rate_chirps(samples, int(start_sample), config, 1, sample_correction=0)[0]
    rel = float(int(start_sample) - int(header_start)) / float(os_factor)
    cfo_phase = float(2.0 * math.pi * float(cfo_total) * rel / n_bins)
    chirp = (chirp * np.exp(-1j * cfo_phase)).astype(np.complex64)
    spectrum = np.fft.fft(chirp * reference).astype(np.complex64)
    value = complex(spectrum[int(raw_bin) % n_bins])
    return float(math.atan2(value.imag, value.real)), float(abs(value))


def _center_symbol_peak_phase(
    samples: np.ndarray,
    start_sample: int,
    config: PreambleDetectorConfig,
    reference: np.ndarray,
    cfo_total: float,
    header_start: int,
) -> tuple[float, float, int]:
    n_bins = int(config.n_bins)
    os_factor = int(config.os_factor)
    chirp = _extract_chip_rate_chirps(samples, int(start_sample), config, 1, sample_correction=0)[0]
    rel = float(int(start_sample) - int(header_start)) / float(os_factor)
    cfo_phase = float(2.0 * math.pi * float(cfo_total) * rel / n_bins)
    chirp = (chirp * np.exp(-1j * cfo_phase)).astype(np.complex64)
    spectrum = np.fft.fft(chirp * reference).astype(np.complex64)
    power = np.abs(spectrum).astype(np.float64) ** 2
    raw_bin = int(np.argmax(power))
    value = complex(spectrum[raw_bin])
    return float(math.atan2(value.imag, value.real)), float(abs(value)), raw_bin


def _full_rate_symbol_phase(
    samples: np.ndarray,
    start_sample: int,
    raw_bin: int,
    reference: np.ndarray,
) -> tuple[float, float]:
    n_fft = int(reference.size)
    start = int(start_sample)
    stop = start + n_fft
    if start < 0 or stop > samples.size:
        raise ValueError(f"frame-sync symbol at {start_sample} exceeds input sample range.")
    segment = np.asarray(samples[start:stop], dtype=np.complex64)
    spectrum = np.fft.fft(segment * reference).astype(np.complex64)
    value = complex(spectrum[int(raw_bin) % n_fft])
    return float(math.atan2(value.imag, value.real)), float(abs(value))


def _fine_payload_backtrack_start(sync_row: dict[str, str], preamble_len: int, chirp_samples: int) -> int:
    header_start = _int(sync_row, "grlora_fine_payload_start_sample")
    return int(round(float(header_start) - (float(preamble_len) + 4.25) * int(chirp_samples)))


def _front_grid_preamble_start(sync_row: dict[str, str], preamble_len: int, chirp_samples: int, front_grid: str) -> int:
    grid_name = str(front_grid)
    if grid_name == "synced_preamble":
        return _int(sync_row, "grlora_synced_preamble_start_sample")
    if grid_name == "fine_preamble":
        return _int(sync_row, "grlora_fine_preamble_start_sample")
    if grid_name in {"fine_payload_backtrack", "payload_backtrack"}:
        return _fine_payload_backtrack_start(sync_row, preamble_len, chirp_samples)
    raise ValueError(f"unknown front grid: {front_grid}")


def _packet_symbol_index(start_sample: int, preamble_start: int, chirp_samples: int) -> float:
    return float(int(start_sample) - int(preamble_start)) / float(chirp_samples)


def _lock_common_grid_sfd_bins(
    clean_samples: np.ndarray,
    sync_row: dict[str, str],
    sf: int,
    preamble_len: int,
    config: PreambleDetectorConfig,
) -> list[int]:
    chirp_samples = int(config.chirp_samples)
    n_bins = int(config.n_bins)
    header_start = _int(sync_row, "grlora_fine_payload_start_sample")
    preamble_start = _fine_payload_backtrack_start(sync_row, preamble_len, chirp_samples)
    cfo_total = _float(sync_row, "grlora_cfo_int_est", 0.0) + _float(sync_row, "grlora_cfo_frac_est", 0.0)
    sfd_ref = np.conjugate(
        build_downchirp(
            sf,
            cfo_int=_int(sync_row, "grlora_cfo_int_est", 0),
            cfo_frac=_float(sync_row, "grlora_cfo_frac_est", 0.0),
        )
    ).astype(np.complex64)
    bins: list[int] = []
    for sfd_idx in range(2):
        start_sample = preamble_start + (int(preamble_len) + 2 + sfd_idx) * chirp_samples
        chirp = _extract_chip_rate_chirps(clean_samples, int(start_sample), config, 1, sample_correction=0)[0]
        rel = float(int(start_sample) - int(header_start)) / float(config.os_factor)
        cfo_phase = float(2.0 * math.pi * float(cfo_total) * rel / n_bins)
        chirp = (chirp * np.exp(-1j * cfo_phase)).astype(np.complex64)
        spectrum = np.fft.fft(chirp * sfd_ref).astype(np.complex64)
        bins.append(int(np.argmax(np.abs(spectrum) ** 2)))
    return bins


def _preamble_sync_rows(
    samples: np.ndarray,
    sync_row: dict[str, str],
    sf: int,
    preamble_len: int,
    os_factor: int,
    front_grid: str,
    sfd_gt_bins: list[int] | None = None,
) -> list[dict[str, Any]]:
    n_bins = 1 << int(sf)
    config = PreambleDetectorConfig(
        sf=sf,
        samp_rate=500000.0,
        bw=125000.0,
        win_chirps=max(2, int(preamble_len)),
        hop_samples=None,
        min_periodic_peaks=2,
        bin_tol=2,
    )
    chirp_samples = int(config.chirp_samples)
    header_start = _int(sync_row, "grlora_fine_payload_start_sample")
    grid_name = str(front_grid)
    preamble_start = _front_grid_preamble_start(sync_row, preamble_len, chirp_samples, grid_name)
    downchirp = build_downchirp(
        sf,
        cfo_int=_int(sync_row, "grlora_cfo_int_est", 0),
        cfo_frac=_float(sync_row, "grlora_cfo_frac_est", 0.0),
    )
    sfd_ref = np.conjugate(downchirp).astype(np.complex64)
    cfo_total = _float(sync_row, "grlora_cfo_int_est", 0.0) + _float(sync_row, "grlora_cfo_frac_est", 0.0)
    rows: list[dict[str, Any]] = []
    for idx in range(int(preamble_len)):
        symbol_start = preamble_start + idx * chirp_samples
        phase, amp = _center_symbol_phase(
            samples=samples,
            start_sample=symbol_start,
            raw_bin=0,
            config=config,
            downchirp=downchirp,
            cfo_total=cfo_total,
            header_start=header_start,
        )
        rows.append(
            {
                "stage": "preamble",
                "symbol_index": float(idx),
                "phase": phase,
                "amp": amp,
                "raw_bin": 0,
                "start_sample": int(symbol_start),
                "sample_offset": int(os_factor // 2),
                "reference": "downchirp_cfo_corrected",
                "front_grid": grid_name,
            }
        )
    sync_bins = [
        _int(sync_row, "grlora_sync1_expected_signed_bin", 24) % n_bins,
        _int(sync_row, "grlora_sync2_expected_signed_bin", 32) % n_bins,
    ]
    for sync_idx, raw_bin in enumerate(sync_bins):
        symbol_start = preamble_start + (int(preamble_len) + sync_idx) * chirp_samples
        phase, amp = _center_symbol_phase(
            samples=samples,
            start_sample=symbol_start,
            raw_bin=raw_bin,
            config=config,
            downchirp=downchirp,
            cfo_total=cfo_total,
            header_start=header_start,
        )
        rows.append(
            {
                "stage": "sync_word",
                "symbol_index": float(preamble_len + sync_idx),
                "phase": phase,
                "amp": amp,
                "raw_bin": int(raw_bin),
                "start_sample": int(symbol_start),
                "sample_offset": int(os_factor // 2),
                "reference": "downchirp_cfo_corrected",
                "front_grid": grid_name,
            }
        )
    for sfd_idx in range(2):
        symbol_start = preamble_start + (int(preamble_len) + 2 + sfd_idx) * chirp_samples
        if sfd_gt_bins is None:
            phase, amp, raw_bin = _center_symbol_peak_phase(
                samples=samples,
                start_sample=symbol_start,
                config=config,
                reference=sfd_ref,
                cfo_total=cfo_total,
                header_start=header_start,
            )
            reference_label = "sfd_cfo_corrected_upchirp_noisy_peak"
        else:
            raw_bin = int(sfd_gt_bins[sfd_idx]) % n_bins
            phase, amp = _center_symbol_phase_with_reference(
                samples=samples,
                start_sample=symbol_start,
                raw_bin=raw_bin,
                config=config,
                reference=sfd_ref,
                cfo_total=cfo_total,
                header_start=header_start,
            )
            reference_label = "sfd_cfo_corrected_upchirp_clean_locked_bin"
        rows.append(
            {
                "stage": "sfd",
                "symbol_index": float(preamble_len + 2 + sfd_idx),
                "phase": phase,
                "amp": amp,
                "raw_bin": int(raw_bin),
                "start_sample": int(symbol_start),
                "sample_offset": int(os_factor // 2),
                "reference": reference_label,
                "front_grid": grid_name,
            }
        )
    return rows


def _frame_sync_raw_rows(
    samples: np.ndarray,
    framesync_rows: list[dict[str, str]],
    sf: int,
    os_factor: int,
) -> list[dict[str, Any]]:
    down_ref_os = np.conjugate(build_upchirp(sf, symbol_id=0, os_factor=os_factor)).astype(np.complex64)
    up_ref_os = build_upchirp(sf, symbol_id=0, os_factor=os_factor).astype(np.complex64)
    rows: list[dict[str, Any]] = []
    for peak_row in sorted(framesync_rows, key=lambda r: _float(r, "symbol_index")):
        stage_raw = str(peak_row.get("stage", "")).strip()
        if stage_raw == "sync":
            stage = "sync_word"
        elif stage_raw in {"preamble", "sfd"}:
            stage = stage_raw
        else:
            continue
        reference = up_ref_os if stage == "sfd" else down_ref_os
        raw_bin = _int(peak_row, "peak_bin", 0)
        phase, amp = _full_rate_symbol_phase(
            samples=samples,
            start_sample=_int(peak_row, "start_sample"),
            raw_bin=raw_bin,
            reference=reference,
        )
        rows.append(
            {
                "stage": stage,
                "symbol_index": _float(peak_row, "symbol_index"),
                "phase": phase,
                "amp": amp,
                "raw_bin": int(raw_bin),
                "start_sample": _int(peak_row, "start_sample"),
                "sample_offset": 0,
                "reference": "frame_sync_raw_upchirp_ref" if stage == "sfd" else "frame_sync_raw_downchirp_ref",
                "front_grid": "frame_sync_raw",
            }
        )
    return rows


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    rmse = float(math.sqrt(np.mean((y - pred) ** 2)))
    return coef, pred, r2, rmse


def _linear_fit_or_nan(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    if x.size < 2 or y.size < 2:
        return np.asarray([float("nan"), float("nan")]), np.full_like(y, float("nan")), float("nan"), float("nan")
    return _linear_fit(x, y)


def _branch_match(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.size == 0 or target.size == 0:
        return pred
    shift = 2.0 * math.pi * round(float(np.mean(target - pred)) / (2.0 * math.pi))
    return pred + shift


def main() -> int:
    args = parse_args()
    dataset = str(args.dataset)
    packet_id = int(args.packet)
    sf, preamble_len = _infer_params(dataset)
    symbol_csv = WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / f"{dataset}_header_first_symbols.csv"
    sync_csv = WEAK_ROOT / "data" / "weak_sync_chain" / "sync_chain" / f"{dataset}_sync_chain.csv"
    framesync_peaks_csv = Path(args.framesync_peaks_csv) if args.framesync_peaks_csv else (
        WEAK_ROOT / "data" / "weak_sync_chain" / "framesync_peaks" / f"{dataset}_framesync_peaks.csv"
    )
    noisy_dir = Path(args.noisy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    front_grid = str(args.front_grid)
    if front_grid == "payload_backtrack":
        front_grid = "fine_payload_backtrack"

    symbol_rows_all = _read_csv(symbol_csv)
    packet_symbol_rows = [row for row in symbol_rows_all if _int(row, "packet_index", -1) == packet_id]
    header_rows = sorted([row for row in packet_symbol_rows if row.get("stage") == "header"], key=lambda r: _int(r, "stage_symbol_index"))
    payload_rows = sorted([row for row in packet_symbol_rows if row.get("stage") == "payload"], key=lambda r: _int(r, "stage_symbol_index"))
    if not header_rows or not payload_rows:
        raise ValueError(f"packet {packet_id} missing header/payload rows in {symbol_csv}")
    header_start = _int(header_rows[0], "start_sample")
    os_factor = _int(header_rows[0], "os_factor", 4)

    sync_rows = _read_csv(sync_csv)
    sync_matches = [row for row in sync_rows if _int(row, "packet_index", -1) == packet_id]
    if not sync_matches:
        raise ValueError(f"packet {packet_id} missing sync row in {sync_csv}")
    sync_row = sync_matches[0]
    framesync_packet_rows: list[dict[str, str]] = []
    if front_grid == "frame_sync_raw":
        framesync_rows = _read_csv(framesync_peaks_csv)
        framesync_packet_rows = [row for row in framesync_rows if _int(row, "packet_index", -1) == packet_id]
        if not framesync_packet_rows:
            raise ValueError(f"packet {packet_id} missing frame-sync peak rows in {framesync_peaks_csv}")

    detector_config = PreambleDetectorConfig(
        sf=sf,
        samp_rate=500000.0,
        bw=125000.0,
        win_chirps=max(2, int(preamble_len)),
        hop_samples=None,
        min_periodic_peaks=2,
        bin_tol=2,
    )
    sfd_gt_bins: list[int] | None = None
    if front_grid == "fine_payload_backtrack":
        clean_iq = Path(args.clean_iq) if args.clean_iq else PROJECT_ROOT / "data" / "USRP_IQ" / f"{dataset}.bin"
        clean_samples = np.fromfile(clean_iq, dtype=np.complex64)
        if clean_samples.size == 0:
            raise ValueError(f"empty or missing clean IQ: {clean_iq}")
        sfd_gt_bins = _lock_common_grid_sfd_bins(
            clean_samples,
            sync_row,
            sf=sf,
            preamble_len=preamble_len,
            config=detector_config,
        )

    snrs = [float(v) for v in args.snrs]
    summary_rows: list[dict[str, Any]] = []
    per_symbol_rows: list[dict[str, Any]] = []
    plot_payload: dict[float, dict[str, Any]] = {}

    for snr in snrs:
        snr_tag = f"m{abs(int(snr))}"
        noisy_path = noisy_dir / f"{dataset}_snr_{snr_tag}dB.bin"
        samples = np.fromfile(noisy_path, dtype=np.complex64)
        if samples.size == 0:
            raise ValueError(f"empty or missing IQ: {noisy_path}")

        if front_grid == "frame_sync_raw":
            rows = _frame_sync_raw_rows(
                samples,
                framesync_packet_rows,
                sf=sf,
                os_factor=os_factor,
            )
        else:
            rows: list[dict[str, Any]] = []
            rows.extend(
                _preamble_sync_rows(
                    samples,
                    sync_row,
                    sf=sf,
                    preamble_len=preamble_len,
                    os_factor=os_factor,
                    front_grid=front_grid,
                    sfd_gt_bins=sfd_gt_bins,
                )
            )
            common_preamble_start = _front_grid_preamble_start(
                sync_row,
                preamble_len,
                int(detector_config.chirp_samples),
                front_grid,
            )
            for row in header_rows:
                phase, amp, raw_bin = _phase_at_raw_bin(samples, row, header_start_sample=header_start)
                rows.append(
                    {
                        "stage": "header",
                        "symbol_index": _packet_symbol_index(
                            _int(row, "start_sample"),
                            common_preamble_start,
                            int(detector_config.chirp_samples),
                        ),
                        "phase": phase,
                        "amp": amp,
                        "raw_bin": raw_bin,
                        "start_sample": _int(row, "start_sample"),
                        "sample_offset": int(os_factor // 2),
                        "reference": "downchirp_cfo_corrected",
                        "front_grid": "fine_payload_backtrack",
                    }
                )
            for row in payload_rows:
                phase, amp, raw_bin = _phase_at_raw_bin(samples, row, header_start_sample=header_start)
                rows.append(
                    {
                        "stage": "payload_gt",
                        "symbol_index": _packet_symbol_index(
                            _int(row, "start_sample"),
                            common_preamble_start,
                            int(detector_config.chirp_samples),
                        ),
                        "phase": phase,
                        "amp": amp,
                        "raw_bin": raw_bin,
                        "start_sample": _int(row, "start_sample"),
                        "sample_offset": int(os_factor // 2),
                        "reference": "downchirp_cfo_corrected",
                        "front_grid": "fine_payload_backtrack",
                    }
                )

        rows.sort(key=lambda item: float(item["symbol_index"]))
        down_rows = [row for row in rows if row["stage"] != "sfd"]
        down_phases = np.asarray([float(row["phase"]) for row in down_rows], dtype=np.float64)
        down_unwrapped = np.unwrap(down_phases)
        for row, phase_unwrap in zip(down_rows, down_unwrapped):
            row["phase_unwrap"] = float(phase_unwrap)

        sfd_rows = [row for row in rows if row["stage"] == "sfd"]
        if sfd_rows:
            sfd_unwrapped = np.unwrap(np.asarray([float(row["phase"]) for row in sfd_rows], dtype=np.float64))
            neighbor = [
                float(row["phase_unwrap"])
                for row in down_rows
                if 15.0 <= float(row["symbol_index"]) <= 21.5 and "phase_unwrap" in row
            ]
            if neighbor:
                shift = 2.0 * math.pi * round(float(np.mean(neighbor) - np.mean(sfd_unwrapped)) / (2.0 * math.pi))
                sfd_unwrapped = sfd_unwrapped + shift
            for row, phase_unwrap in zip(sfd_rows, sfd_unwrapped):
                row["phase_unwrap"] = float(phase_unwrap)

        for row in rows:
            row["target_snr_db"] = float(snr)
            row["packet_index"] = packet_id
            row["dataset"] = dataset
            per_symbol_rows.append(row)

        fit_rows = [row for row in rows if row["stage"] != "sfd"]
        x = np.asarray([float(row["symbol_index"]) for row in fit_rows], dtype=np.float64)
        y = np.asarray([float(row["phase_unwrap"]) for row in fit_rows], dtype=np.float64)
        pre_header_mask = np.asarray([row["stage"] in ("preamble", "header") for row in fit_rows], dtype=bool)
        pre_sync_header_mask = np.asarray([row["stage"] in ("preamble", "sync_word", "header") for row in fit_rows], dtype=bool)
        payload_mask = np.asarray([row["stage"] == "payload_gt" for row in fit_rows], dtype=bool)
        pre_header_coef, _pre_header_fit_local, pre_header_r2, pre_header_rmse = _linear_fit_or_nan(x[pre_header_mask], y[pre_header_mask])
        pre_sync_header_coef, _pre_sync_header_fit_local, pre_sync_header_r2, pre_sync_header_rmse = _linear_fit_or_nan(
            x[pre_sync_header_mask], y[pre_sync_header_mask]
        )
        payload_coef, payload_fit, payload_r2, payload_rmse = _linear_fit_or_nan(x[payload_mask], y[payload_mask])
        if np.any(payload_mask) and np.all(np.isfinite(pre_header_coef)):
            pre_header_pred_payload = pre_header_coef[0] * x[payload_mask] + pre_header_coef[1]
            pre_header_pred_payload = _branch_match(pre_header_pred_payload, y[payload_mask])
            payload_resid = y[payload_mask] - pre_header_pred_payload
            payload_rmse_from_pre_header = float(math.sqrt(np.mean(payload_resid**2)))
        else:
            payload_rmse_from_pre_header = float("nan")
        if np.any(payload_mask) and np.all(np.isfinite(pre_sync_header_coef)):
            pre_sync_header_pred_payload = pre_sync_header_coef[0] * x[payload_mask] + pre_sync_header_coef[1]
            pre_sync_header_pred_payload = _branch_match(pre_sync_header_pred_payload, y[payload_mask])
            payload_resid_pre_sync_header = y[payload_mask] - pre_sync_header_pred_payload
            payload_rmse_from_pre_sync_header = float(math.sqrt(np.mean(payload_resid_pre_sync_header**2)))
        else:
            payload_rmse_from_pre_sync_header = float("nan")

        summary_rows.append(
            {
                "dataset": dataset,
                "packet_index": packet_id,
                "target_snr_db": float(snr),
                "front_grid": front_grid,
                "pre_header_slope_pi_per_symbol": float(pre_header_coef[0] / math.pi),
                "payload_slope_pi_per_symbol": float(payload_coef[0] / math.pi),
                "slope_delta_pi_per_symbol": float(payload_coef[0] / math.pi - pre_header_coef[0] / math.pi),
                "pre_header_fit_r2": float(pre_header_r2),
                "pre_header_fit_rmse_pi": float(pre_header_rmse / math.pi),
                "pre_sync_header_slope_pi_per_symbol": float(pre_sync_header_coef[0] / math.pi),
                "pre_sync_header_fit_r2": float(pre_sync_header_r2),
                "pre_sync_header_fit_rmse_pi": float(pre_sync_header_rmse / math.pi),
                "payload_gt_fit_r2": float(payload_r2),
                "payload_gt_fit_rmse_pi": float(payload_rmse / math.pi),
                "payload_rmse_from_pre_header_pi": float(payload_rmse_from_pre_header / math.pi),
                "payload_rmse_from_pre_sync_header_pi": float(payload_rmse_from_pre_sync_header / math.pi),
            }
        )
        plot_payload[snr] = {
            "rows": rows,
            "x": x,
            "y": y,
            "pre_header_coef": pre_header_coef,
            "payload_coef": payload_coef,
            "summary": summary_rows[-1],
        }

    suffix = front_grid
    _write_csv(out_dir / f"{dataset}_packet_{packet_id:03d}_{suffix}_gt_phase_points.csv", per_symbol_rows)
    _write_csv(out_dir / f"{dataset}_packet_{packet_id:03d}_{suffix}_gt_phase_summary.csv", summary_rows)
    _plot_grid(
        dataset,
        packet_id,
        snrs,
        plot_payload,
        out_dir / f"{dataset}_packet_{packet_id:03d}_{suffix}_gt_phase_snr_m22_m27.png",
        front_grid=front_grid,
    )
    (out_dir / f"{dataset}_packet_{packet_id:03d}_{suffix}_manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "packet": packet_id,
                "snrs": snrs,
                "symbol_csv": str(symbol_csv),
                "sync_csv": str(sync_csv),
                "framesync_peaks_csv": str(framesync_peaks_csv),
                "noisy_dir": str(noisy_dir),
                "preamble_len": preamble_len,
                "sf": sf,
                "front_grid": front_grid,
                "sfd_gt_bins": sfd_gt_bins,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(out_dir)
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _plot_grid(
    dataset: str,
    packet_id: int,
    snrs: list[float],
    data: dict[float, dict[str, Any]],
    out_path: Path,
    front_grid: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.8), dpi=170, sharex=True)
    axes_flat = axes.ravel()
    stage_style = {
        "preamble": dict(marker="o", linestyle="-", color="#2f7fbd", label="preamble bin0", markersize=4.5),
        "sync_word": dict(marker="s", linestyle="-", color="#d08b28", label="sync word expected bin", markersize=5.0),
        "sfd": dict(marker="^", linestyle="-", color="#c44e52", label="SFD bin", markersize=5.2),
        "header": dict(marker="D", linestyle="-", color="#8b5fbf", label="header GT bin", markersize=4.2),
        "payload_gt": dict(marker="o", linestyle="-", color="#2f8f4e", label="payload GT bin", markersize=4.5),
    }
    for ax, snr in zip(axes_flat, snrs):
        item = data[float(snr)]
        rows = item["rows"]
        present_stages = {str(row["stage"]) for row in rows}
        for stage, style in stage_style.items():
            xs = [float(row["symbol_index"]) for row in rows if row["stage"] == stage]
            ys = [float(row["phase_unwrap"]) / math.pi for row in rows if row["stage"] == stage]
            if not xs:
                continue
            ax.plot(xs, ys, **style)

        s = item["summary"]
        payload_r2 = float(s.get("payload_gt_fit_r2", float("nan")))
        if math.isfinite(payload_r2):
            title_suffix = f"payload GT R2={payload_r2:.3f}"
        else:
            title_suffix = "frame-sync front fields"
        ax.set_title(
            f"SNR {snr:.0f} dB | {title_suffix}",
            fontsize=10.5,
        )
        ax.grid(True, alpha=0.28)
        ax.axvline(16, color="#777777", linewidth=0.8, alpha=0.35)
        ax.axvline(18, color="#777777", linewidth=0.8, alpha=0.35)
        ax.axvline(20.25, color="#777777", linewidth=0.8, alpha=0.35)
        if "payload_gt" in present_stages:
            ax.axvline(28.25, color="#777777", linewidth=0.8, alpha=0.35)
        ax.text(2, ax.get_ylim()[0] + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "preamble", color="#2f7fbd", fontsize=8)
        ax.text(16.2, ax.get_ylim()[0] + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "sync", color="#d08b28", fontsize=8)
        ax.text(18.2, ax.get_ylim()[0] + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "SFD", color="#c44e52", fontsize=8)
        if "header" in present_stages:
            ax.text(21, ax.get_ylim()[0] + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "header", color="#8b5fbf", fontsize=8)
        if "payload_gt" in present_stages:
            ax.text(31, ax.get_ylim()[0] + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "payload", color="#2f8f4e", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("unwrapped phase / pi")
    for ax in axes[-1, :]:
        ax.set_xlabel("packet-local symbol index")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        f"{dataset} packet {packet_id}: phase across SNR, front grid = {front_grid}",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

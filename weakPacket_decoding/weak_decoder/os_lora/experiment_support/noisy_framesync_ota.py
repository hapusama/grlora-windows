"""Shared I/O and AWGN helpers for whole-packet OTA FrameSync experiments."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np


_SYNC_MODULE: ModuleType | None = None


def parse_csv_list(text: str, cast: type) -> tuple[Any, ...]:
    values = tuple(cast(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return values


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    # ``utf-8-sig`` also accepts ordinary UTF-8 while stripping the BOM that
    # Windows PowerShell 5 adds when Export-Csv is used with ``-Encoding utf8``.
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_single_packet_sync_module(dataset_repo: Path) -> ModuleType:
    """Load the sibling dataset repository's single-packet wrapper lazily."""

    global _SYNC_MODULE
    if _SYNC_MODULE is not None:
        return _SYNC_MODULE
    source = Path(dataset_repo) / "weak_decoder" / "synchronization" / "single_packet.py"
    if not source.is_file():
        raise FileNotFoundError(f"missing clean synchronization wrapper: {source}")
    name = "weak_decoder.synchronization._shared_noisy_ota_single_packet"
    if name in sys.modules:
        _SYNC_MODULE = sys.modules[name]
        return _SYNC_MODULE
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load synchronization wrapper: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _SYNC_MODULE = module
    return module


def packet_metadata_paths(ota_root: Path) -> Iterable[Path]:
    for path in sorted((Path(ota_root) / "metadata").glob("*_fulltrim.json")):
        metadata = load_json(path)
        if int(metadata.get("view", {}).get("adc_phase_2m", -1)) != 0:
            continue
        if not bool(metadata.get("packet", {}).get("crc_valid", False)):
            continue
        yield path


def unit_lora_band_awgn(
    rng: np.random.Generator,
    count: int,
    *,
    os_factor: int,
) -> np.ndarray:
    """Return unit-power complex AWGN flat only inside the LoRa B-wide band."""

    length = int(count)
    white = (
        rng.standard_normal(length).astype(np.float32)
        + 1j * rng.standard_normal(length).astype(np.float32)
    ) / np.float32(math.sqrt(2.0))
    frequency = np.fft.fft(white)
    passband_bins = int(round(length / int(os_factor)))
    if passband_bins % 2:
        passband_bins -= 1
    half = passband_bins // 2
    frequency[half : length - half] = 0.0
    noise = np.fft.ifft(frequency)
    noise *= math.sqrt(float(length) / passband_bins)
    return noise.astype(np.complex64)


def make_sync_config(
    module: ModuleType,
    center_frequency_hz: float,
    *,
    sf: int,
    bw_hz: float,
    sample_rate_hz: float,
    preamble_symbols: int,
    sync_word: int,
) -> Any:
    return module.SinglePacketSyncConfig(
        sf=int(sf),
        bw_hz=float(bw_hz),
        sample_rate_hz=float(sample_rate_hz),
        center_frequency_hz=float(center_frequency_hz),
        preamble_symbols=int(preamble_symbols),
        sync_word=int(sync_word),
    )


def sync_gate_failures(sync_result: Any, config: Any) -> tuple[str, ...]:
    failures: list[str] = []
    location = sync_result.frame_location
    frame_sync = sync_result.frame_sync
    if location is None:
        failures.append("frame_location_missing")
    else:
        min_preamble = max(3, int(config.preamble_symbols) - int(config.bin_tolerance))
        if int(location.preamble_stable_count) < min_preamble:
            failures.append("locator_preamble_stability")
        if int(location.sync1_distance) > int(config.sync_bin_tolerance):
            failures.append("locator_sync1")
        if int(location.sync2_distance) > int(config.sync_bin_tolerance):
            failures.append("locator_sync2")
        if int(location.sfd_bin_distance) > int(config.sfd_bin_tolerance):
            failures.append("locator_sfd")
    if frame_sync is None:
        failures.append("frame_sync_missing")
    else:
        if int(frame_sync.preamble_bin0_count) != int(frame_sync.preamble_peak_count):
            failures.append("framesync_preamble_all_bin0")
        if not bool(frame_sync.netid_valid):
            failures.append("framesync_netid")
    return tuple(failures)


__all__ = [
    "load_json",
    "load_single_packet_sync_module",
    "make_sync_config",
    "packet_metadata_paths",
    "parse_csv_list",
    "read_csv_rows",
    "sync_gate_failures",
    "unit_lora_band_awgn",
    "write_csv_rows",
]

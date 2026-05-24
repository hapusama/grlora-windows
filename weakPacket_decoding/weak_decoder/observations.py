"""Extract per-symbol complex FFT candidates from detected packet windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import math

import numpy as np

from .chirp import bin_to_grlora_symbol, build_downchirp, dechirp_fft, gray_value, positive_mod


@dataclass
class Candidate:
    """One FFT-bin candidate for a symbol."""

    bin: int
    grlora_symbol: int
    gray: int
    value_real: float
    value_imag: float
    magnitude: float
    power: float
    phase: float
    rank: int


@dataclass
class SymbolObservation:
    """Top-K observation for one LoRa symbol after SFD."""

    symbol_index: int
    absolute_sample: int
    is_header: bool
    confidence_db: float
    total_power: float
    noise_power_est: float
    candidates: list[Candidate] = field(default_factory=list)


@dataclass
class PacketObservation:
    """All per-symbol observations extracted from one packet."""

    input_file: str
    frame_count: int
    sf: int
    bw: float
    sample_rate: float
    os_factor: int
    samples_per_symbol: int
    n_bins: int
    symbol_start_sample: int
    requested_symbols: int
    extracted_symbols: int
    cfo: float
    sto: float
    sfo: float
    ldro: bool
    packet_metadata: dict[str, Any]
    symbols: list[SymbolObservation] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        """Convert nested dataclasses to JSON-safe dictionaries."""
        return asdict(self)


def resolve_ldro(sf: int, bw: float, ldro_mode: int) -> bool:
    """Resolve low-data-rate optimization using gr-lora_sdr's threshold."""
    if int(ldro_mode) == 0:
        return False
    if int(ldro_mode) == 1:
        return True
    return ((1 << int(sf)) * 1e3 / float(bw)) > 16.0


def _top_indices(power: np.ndarray, top_k: int) -> np.ndarray:
    top_k = max(1, min(int(top_k), int(power.size)))
    if top_k == power.size:
        return np.argsort(power)[::-1]
    idx = np.argpartition(power, -top_k)[-top_k:]
    return idx[np.argsort(power[idx])[::-1]]


def _decimate_symbol(raw_symbol: np.ndarray, os_factor: int) -> np.ndarray:
    """Convert raw oversampled samples to one sample per chip."""
    os_factor = max(1, int(os_factor))
    if os_factor == 1:
        return np.asarray(raw_symbol, dtype=np.complex64)
    offset = os_factor // 2
    return np.asarray(raw_symbol[offset::os_factor], dtype=np.complex64)


def extract_packet_observation(
    input_file: Path,
    packet: dict[str, Any],
    *,
    top_k: int = 8,
    max_symbols: int | None = None,
    start_symbol: int = 0,
) -> PacketObservation:
    """Extract Top-K complex FFT candidates from one detected packet.

    ``packet['end_sample']`` is treated as the first header symbol start. This
    matches the local frame_sync metadata, where start/end cover preamble,
    sync word, SFD, and stop at the first symbol emitted to fft_demod.
    """
    input_file = Path(input_file)
    samples = np.memmap(input_file, dtype=np.complex64, mode="r")

    sf = int(packet.get("sf", 10))
    n_bins = 1 << sf
    bw = float(packet.get("bw", 125000.0))
    sample_rate = float(packet.get("sample_rate", packet.get("samp_rate", 500000.0)))
    os_factor = max(1, int(round(sample_rate / bw)))
    samples_per_symbol = int(packet.get("samples_per_symbol", n_bins * os_factor))
    symbol_start = int(packet.get("end_sample", packet.get("packet_start_sample", 0)))
    total_symbols = int(packet.get("payload_symbols", 0))
    if total_symbols <= 0:
        total_symbols = max(0, (int(packet.get("packet_end_sample", symbol_start)) - symbol_start) // samples_per_symbol)
    if max_symbols is not None:
        total_symbols = min(total_symbols, int(max_symbols))
    total_symbols = max(0, total_symbols - int(start_symbol))

    cfo = float(packet.get("cfo", 0.0) or 0.0)
    cfo_int = int(math.floor(cfo)) if cfo >= 0 else int(math.ceil(cfo))
    cfo_frac = cfo - cfo_int
    downchirp = build_downchirp(sf, cfo_int=cfo_int, cfo_frac=cfo_frac)
    ldro = resolve_ldro(sf, bw, int(packet.get("ldro_mode", 2)))

    observations: list[SymbolObservation] = []
    for local_index in range(total_symbols):
        symbol_index = int(start_symbol) + local_index
        absolute_sample = symbol_start + symbol_index * samples_per_symbol
        raw_stop = absolute_sample + samples_per_symbol
        if absolute_sample < 0 or raw_stop > samples.size:
            break
        symbol_samples = _decimate_symbol(samples[absolute_sample:raw_stop], os_factor)
        if symbol_samples.size < n_bins:
            break
        symbol_samples = symbol_samples[:n_bins]
        bins = dechirp_fft(symbol_samples, downchirp)
        power = np.abs(bins) ** 2
        total_power = float(np.sum(power, dtype=np.float64))
        top_idx = _top_indices(power, top_k)
        strongest = float(power[top_idx[0]])
        second = float(power[top_idx[1]]) if top_idx.size > 1 else 0.0
        confidence_db = 10.0 * math.log10((strongest + 1e-30) / (second + 1e-30))
        noise_est = (total_power - strongest) / max(1, n_bins - 1)

        is_header = symbol_index < 8
        candidates: list[Candidate] = []
        for rank, bin_index in enumerate(top_idx.tolist(), start=1):
            value = complex(bins[bin_index])
            gr_symbol = bin_to_grlora_symbol(bin_index, sf, is_header=is_header, ldro=ldro)
            candidates.append(
                Candidate(
                    bin=int(bin_index),
                    grlora_symbol=int(gr_symbol),
                    gray=int(gray_value(gr_symbol)),
                    value_real=float(value.real),
                    value_imag=float(value.imag),
                    magnitude=float(abs(value)),
                    power=float(power[bin_index]),
                    phase=float(math.atan2(value.imag, value.real)),
                    rank=int(rank),
                )
            )
        observations.append(
            SymbolObservation(
                symbol_index=int(symbol_index),
                absolute_sample=int(absolute_sample),
                is_header=bool(is_header),
                confidence_db=float(confidence_db),
                total_power=float(total_power),
                noise_power_est=float(noise_est),
                candidates=candidates,
            )
        )

    return PacketObservation(
        input_file=str(input_file),
        frame_count=int(packet.get("frame_count", 0)),
        sf=sf,
        bw=bw,
        sample_rate=sample_rate,
        os_factor=os_factor,
        samples_per_symbol=samples_per_symbol,
        n_bins=n_bins,
        symbol_start_sample=int(symbol_start),
        requested_symbols=int(total_symbols),
        extracted_symbols=len(observations),
        cfo=cfo,
        sto=float(packet.get("sto", 0.0) or 0.0),
        sfo=float(packet.get("sfo", 0.0) or 0.0),
        ldro=bool(ldro),
        packet_metadata=dict(packet),
        symbols=observations,
    )

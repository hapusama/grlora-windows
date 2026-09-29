"""CRC-aided ambiguity-ridge list synchronization for oversampled LoRa.

The gr-lora_sdr-style synchronizer obtains integer CFO from the SFD2 FFT.  A
single noisy Argmax can select the wrong SFD peak and move both CFO and the
payload boundary to another point on LoRa's CFO/STO ambiguity ridge.  This
module retains a small list of SFD peak hypotheses and couples their timing as

    delta_payload_start_samples = os_factor * delta_cfo_bins.

Only the resulting Top-K candidates enter Savaux/FEC/CRC.  No two-dimensional
CFO-by-STO grid and no payload ground truth are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from ...chirp import build_upchirp, positive_mod, signed_fft_bin
from ...synchronization.frame_locator import sync_word_to_symbols
from .decoder_aware_crc import (
    DecoderAwarePacketResult,
    decode_savaux_sync_candidate,
)
from .soft_hamming_crc import decode_soft_hamming_sync_candidate


def _grlora_round(value: float) -> int:
    number = float(value)
    if number > 0.0:
        return int(number + 0.5)
    return int(math.ceil(number - 0.5))


def _wrap_half(value: float) -> float:
    return float((float(value) + 0.5) % 1.0 - 0.5)


def _split_cfo(total_bins: float) -> tuple[int, float]:
    integer = int(math.floor(float(total_bins) + 0.5))
    return integer, float(total_bins - integer)


@dataclass(frozen=True)
class AmbiguityRidgeCandidate:
    """One coupled CFO/STO coordinate consumable by the packet decoder."""

    candidate_index: int
    source: str
    sfd_peak_rank: int
    sfd_peak_signed_bin: int
    sfd_relative_power_db: float
    cfo_int_est: int
    cfo_frac_est: float
    cfo_total_est: float
    cfo_hz_est: float
    sfo_hat: float
    sfo_cum_initial: float
    fine_preamble_start_sample: int
    fine_payload_start_sample: int
    netid1_est: int
    netid2_est: int
    netid_offset: int
    netid_valid: bool
    netid_margin_db: float
    delta_cfo_bins: float
    delta_payload_samples: int
    ridge_residual_bins: float
    valid: bool = True


@dataclass(frozen=True)
class AmbiguityRidgeListResult:
    """Ranked list and diagnostics from the SFD peak pool."""

    candidates: tuple[AmbiguityRidgeCandidate, ...]
    unique_integer_cfo_pool: int
    sfd_peaks_examined: int


@dataclass(frozen=True)
class CrcListAttempt:
    candidate: AmbiguityRidgeCandidate
    decode: DecoderAwarePacketResult
    crc_accepted: bool


@dataclass(frozen=True)
class CrcListArbitrationResult:
    attempts: tuple[CrcListAttempt, ...]
    selected_index: int | None

    @property
    def selected(self) -> CrcListAttempt | None:
        if self.selected_index is None:
            return None
        return self.attempts[int(self.selected_index)]

    @property
    def crc_success(self) -> bool:
        return self.selected is not None


def _extract_chip_symbol(
    samples: np.ndarray,
    *,
    start_sample: int,
    n_bins: int,
    os_factor: int,
    sample_correction: int,
) -> np.ndarray:
    first = int(start_sample) + int(os_factor // 2) - int(sample_correction)
    indexes = first + int(os_factor) * np.arange(int(n_bins), dtype=np.int64)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= np.asarray(samples).size:
        raise ValueError("not enough samples for chip-rate synchronization symbol")
    return np.asarray(samples[indexes], dtype=np.complex64)


def _estimate_sto_frac(
    chip_chirps: np.ndarray,
    downchirp: np.ndarray,
) -> float:
    chirps = np.asarray(chip_chirps, dtype=np.complex64)
    _symbols, n_bins = chirps.shape
    spectra = np.fft.fft(
        chirps * downchirp[np.newaxis, :], n=2 * int(n_bins), axis=1
    )
    power = np.sum(np.abs(spectra) ** 2, axis=0, dtype=np.float64)
    k0 = int(np.argmax(power))
    y_minus = float(power[(k0 - 1) % (2 * n_bins)])
    y0 = float(power[k0])
    y_plus = float(power[(k0 + 1) % (2 * n_bins)])
    u = 64.0 * n_bins / 406.5506497
    v = u * 2.4674
    denom = u * (y_plus + y_minus) + v * y0
    wa = 0.0 if abs(denom) <= 1e-300 else (y_plus - y_minus) / denom
    ka = wa * n_bins / math.pi
    residual = math.fmod((k0 + ka) / 2.0, 1.0)
    return float(residual - (1.0 if residual > 0.5 else 0.0))


def _sfo_correction_vector(
    symbol_count: int,
    n_bins: int,
    bw_hz: float,
    fs_prime_hz: float,
) -> np.ndarray:
    total = int(symbol_count) * int(n_bins)
    n = np.arange(total, dtype=np.float64)
    q = np.floor(n / int(n_bins))
    r = np.mod(n, int(n_bins))
    fs = float(bw_hz)
    fs_prime = float(fs_prime_hz)
    phase = (
        (r**2)
        / (2.0 * int(n_bins))
        * ((float(bw_hz) / fs_prime) ** 2 - (float(bw_hz) / fs) ** 2)
        + (
            q
            * (
                (float(bw_hz) / fs_prime) ** 2
                - (float(bw_hz) / fs_prime)
            )
            + float(bw_hz) / 2.0 * (1.0 / fs - 1.0 / fs_prime)
        )
        * r
    )
    return np.exp(-2j * np.pi * phase).astype(np.complex64)


def _refine_sto_for_cfo(
    samples: np.ndarray,
    frame_sync: Any,
    *,
    cfo_int: int,
    cfo_frac: float,
    sfo_hat: float,
    n_bins: int,
    os_factor: int,
    bw_hz: float,
) -> float:
    """Repeat gr-lora_sdr's coupled STO refinement for one integer-CFO peak."""

    symbol_count = int(frame_sync.up_symbols_used)
    first = int(frame_sync.synced_preamble_start_sample) + int(os_factor // 2)
    total_chips = symbol_count * int(n_bins)
    indexes = first + int(os_factor) * np.arange(total_chips, dtype=np.int64)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= np.asarray(samples).size:
        return float(frame_sync.sto_frac_initial)
    chirps = np.asarray(samples[indexes], dtype=np.complex64).reshape(
        symbol_count, int(n_bins)
    )
    flat_n = np.arange(total_chips, dtype=np.float64)
    fractional = np.exp(
        -2j * np.pi * float(cfo_frac) * flat_n / int(n_bins)
    ).astype(np.complex64)
    refined_flat = (chirps.reshape(-1) * fractional).astype(np.complex64)
    refined_flat = np.roll(
        refined_flat, -positive_mod(int(cfo_int), int(n_bins))
    )
    integer = np.exp(
        -2j * np.pi * float(cfo_int) * flat_n / int(n_bins)
    ).astype(np.complex64)
    refined = (refined_flat * integer).reshape(symbol_count, int(n_bins))
    fs_prime = float(bw_hz) * (1.0 - float(sfo_hat) / int(n_bins))
    refined = (
        refined.reshape(-1)
        * _sfo_correction_vector(
            symbol_count, int(n_bins), float(bw_hz), fs_prime
        )
    ).reshape(symbol_count, int(n_bins))
    downchirp = np.conjugate(
        build_upchirp(int(round(math.log2(n_bins))))
    ).astype(np.complex64)
    sto_refined = _estimate_sto_frac(refined, downchirp)
    sto_initial = float(frame_sync.sto_frac_initial)
    if abs(sto_initial - sto_refined) <= float(os_factor - 1) / float(os_factor):
        return float(sto_refined)
    return sto_initial


def _netid_evidence(
    samples: np.ndarray,
    *,
    synced_preamble_start_sample: int,
    preamble_symbols: int,
    n_bins: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    netid_sto_frac: float,
    sync_word: int,
) -> tuple[int, int, int, bool, float]:
    start = (
        int(synced_preamble_start_sample)
        + int(preamble_symbols) * int(n_bins) * int(os_factor)
        + int(cfo_int) * int(os_factor)
    )
    correction = _grlora_round(float(netid_sto_frac) * int(os_factor))
    total_chips = 2 * int(n_bins)
    first = start + int(os_factor // 2) - correction
    indexes = first + int(os_factor) * np.arange(total_chips, dtype=np.int64)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= np.asarray(samples).size:
        return -1, -1, 0, False, -math.inf
    chirps = np.asarray(samples[indexes], dtype=np.complex64).reshape(2, int(n_bins))

    flat_n = np.arange(total_chips, dtype=np.float64)
    integer_correction = np.exp(
        -2j * np.pi * float(cfo_int) * flat_n / int(n_bins)
    ).astype(np.complex64)
    chirps = (chirps.reshape(-1) * integer_correction).reshape(2, int(n_bins))
    symbol_n = np.arange(int(n_bins), dtype=np.float64)
    fractional_correction = np.exp(
        -2j * np.pi * float(cfo_frac) * symbol_n / int(n_bins)
    ).astype(np.complex64)
    chirps = (chirps * fractional_correction[np.newaxis, :]).astype(np.complex64)
    downchirp = np.conjugate(build_upchirp(int(round(math.log2(n_bins))))).astype(
        np.complex64
    )
    spectra = np.fft.fft(chirps * downchirp[np.newaxis, :], axis=1)
    power = np.abs(spectra).astype(np.float64) ** 2
    bins = np.argmax(power, axis=1).astype(np.int64)
    margins: list[float] = []
    for row, selected in enumerate(bins):
        first_power = float(power[row, int(selected)])
        second_power = float(np.partition(power[row], -2)[-2])
        margins.append(
            10.0 * math.log10((first_power + 1e-30) / (second_power + 1e-30))
        )

    sync1_expected, sync2_expected = sync_word_to_symbols(int(sync_word))
    netid1 = int(bins[0])
    netid2 = int(bins[1])
    offset = int(netid1 - int(sync1_expected))
    valid = bool(
        abs(offset) <= 2
        and positive_mod(netid2 - offset, n_bins)
        == positive_mod(int(sync2_expected), n_bins)
    )
    return netid1, netid2, offset, valid, float(np.mean(margins))


def _candidate_from_cfo_integer(
    samples: np.ndarray,
    frame_sync: Any,
    *,
    cfo_int: int,
    sfd_peak_rank: int,
    sfd_peak_signed_bin: int,
    sfd_relative_power_db: float,
    source: str,
    n_bins: int,
    os_factor: int,
    bw_hz: float,
    center_frequency_hz: float,
    preamble_symbols: int,
    sync_word: int,
) -> AmbiguityRidgeCandidate:
    cfo_frac = float(frame_sync.cfo_frac_est)
    cfo_total = float(int(cfo_int) + cfo_frac)
    sfo_hat = float(cfo_total * float(bw_hz) / float(center_frequency_hz))
    sto_used = _refine_sto_for_cfo(
        samples,
        frame_sync,
        cfo_int=int(cfo_int),
        cfo_frac=cfo_frac,
        sfo_hat=sfo_hat,
        n_bins=int(n_bins),
        os_factor=int(os_factor),
        bw_hz=float(bw_hz),
    )
    sto_correction = _grlora_round(sto_used * int(os_factor))
    netid_sto = _wrap_half(sto_used + sfo_hat * int(preamble_symbols))
    payload_sto = _wrap_half(
        sto_used + sfo_hat * (int(preamble_symbols) + 4.25)
    )
    payload_correction = _grlora_round(payload_sto * int(os_factor))
    netid1, netid2, netid_offset, netid_valid, netid_margin = _netid_evidence(
        samples,
        synced_preamble_start_sample=int(frame_sync.synced_preamble_start_sample),
        preamble_symbols=int(preamble_symbols),
        n_bins=int(n_bins),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=cfo_frac,
        netid_sto_frac=netid_sto,
        sync_word=int(sync_word),
    )
    fine_preamble = int(
        int(frame_sync.synced_preamble_start_sample) - sto_correction
    )
    fine_payload = int(
        int(frame_sync.synced_payload_start_sample)
        + int(os_factor) * int(cfo_int)
        - (int(os_factor) * int(netid_offset) if netid_valid else 0)
        - payload_correction
    )
    sfo_cum = float(
        (payload_sto * int(os_factor) - payload_correction) / int(os_factor)
    )
    delta_cfo = float(cfo_total - float(frame_sync.cfo_total_est))
    delta_samples = int(fine_payload - int(frame_sync.fine_payload_start_sample))
    return AmbiguityRidgeCandidate(
        candidate_index=-1,
        source=str(source),
        sfd_peak_rank=int(sfd_peak_rank),
        sfd_peak_signed_bin=int(sfd_peak_signed_bin),
        sfd_relative_power_db=float(sfd_relative_power_db),
        cfo_int_est=int(cfo_int),
        cfo_frac_est=cfo_frac,
        cfo_total_est=cfo_total,
        cfo_hz_est=float(cfo_total * float(bw_hz) / int(n_bins)),
        sfo_hat=sfo_hat,
        sfo_cum_initial=sfo_cum,
        fine_preamble_start_sample=fine_preamble,
        fine_payload_start_sample=fine_payload,
        netid1_est=netid1,
        netid2_est=netid2,
        netid_offset=netid_offset,
        netid_valid=netid_valid,
        netid_margin_db=netid_margin,
        delta_cfo_bins=delta_cfo,
        delta_payload_samples=delta_samples,
        ridge_residual_bins=float(delta_cfo - delta_samples / int(os_factor)),
    )


def build_ambiguity_ridge_sync_list(
    samples: np.ndarray,
    frame_sync: Any | None,
    *,
    sf: int,
    bw_hz: float,
    os_factor: int,
    center_frequency_hz: float,
    preamble_symbols: int = 16,
    sync_word: int = 0x12,
    top_k: int = 4,
    sfd_peak_pool: int = 32,
) -> AmbiguityRidgeListResult:
    """Build a Top-K 1D synchronization list from alternative SFD peaks.

    Candidate zero is always the original FrameSync point estimate.  The
    remaining slots follow descending SFD power; net-ID is retained as soft
    diagnostics rather than another hard gate. Reserving the baseline
    coordinate makes list synchronization monotone with respect to the
    existing decoder when CRC arbitration is used.
    """

    if frame_sync is None:
        return AmbiguityRidgeListResult((), 0, 0)
    if int(top_k) <= 0 or int(sfd_peak_pool) <= 0:
        raise ValueError("top_k and sfd_peak_pool must be positive")
    if int(os_factor) <= 0:
        raise ValueError("os_factor must be positive")
    n_bins = 1 << int(sf)
    values = np.asarray(samples, dtype=np.complex64)

    sfd2_start = int(frame_sync.synced_sfd_start_sample) + n_bins * int(os_factor)
    sfd2 = _extract_chip_symbol(
        values,
        start_sample=sfd2_start,
        n_bins=n_bins,
        os_factor=int(os_factor),
        sample_correction=int(frame_sync.sto_sample_correction),
    )
    n = np.arange(n_bins, dtype=np.float64)
    frac_correction = np.exp(
        -2j * np.pi * float(frame_sync.cfo_frac_est) * n / n_bins
    ).astype(np.complex64)
    upchirp = build_upchirp(int(sf), symbol_id=0, os_factor=1)
    spectrum = np.fft.fft(sfd2 * frac_correction * upchirp)
    power = np.abs(spectrum).astype(np.float64) ** 2
    order = np.argsort(power)[::-1][: min(int(sfd_peak_pool), n_bins)]
    strongest = float(power[int(order[0])]) if order.size else 0.0

    by_integer: dict[int, tuple[int, int, float]] = {}
    for rank, raw_bin in enumerate(order, start=1):
        signed = signed_fft_bin(int(raw_bin), n_bins)
        cfo_int = int(math.floor(float(signed) / 2.0))
        relative_db = float(
            10.0
            * math.log10((float(power[int(raw_bin)]) + 1e-30) / (strongest + 1e-30))
        )
        if cfo_int not in by_integer:
            by_integer[cfo_int] = (int(rank), int(signed), relative_db)

    selected_int, _selected_frac = _split_cfo(float(frame_sync.cfo_total_est))
    selected_rank, selected_signed, selected_db = by_integer.get(
        int(selected_int), (0, int(frame_sync.down_val_signed_bin), 0.0)
    )
    selected = _candidate_from_cfo_integer(
        values,
        frame_sync,
        cfo_int=int(selected_int),
        sfd_peak_rank=int(selected_rank),
        sfd_peak_signed_bin=int(selected_signed),
        sfd_relative_power_db=float(selected_db),
        source="selected",
        n_bins=n_bins,
        os_factor=int(os_factor),
        bw_hz=float(bw_hz),
        center_frequency_hz=float(center_frequency_hz),
        preamble_symbols=int(preamble_symbols),
        sync_word=int(sync_word),
    )

    # ``by_integer`` preserves descending SFD power insertion order.  Build
    # only the K-1 alternatives that can reach the decoder; candidate-specific
    # STO/SFO and net-ID refinement is therefore O(K), not O(SFD-pool).
    alternative_specs = [
        (cfo_int, rank, signed, relative_db)
        for cfo_int, (rank, signed, relative_db) in by_integer.items()
        if int(cfo_int) != int(selected_int)
    ][: max(0, int(top_k) - 1)]
    alternatives = [
        _candidate_from_cfo_integer(
            values,
            frame_sync,
            cfo_int=int(cfo_int),
            sfd_peak_rank=int(rank),
            sfd_peak_signed_bin=int(signed),
            sfd_relative_power_db=float(relative_db),
            source="sfd_list",
            n_bins=n_bins,
            os_factor=int(os_factor),
            bw_hz=float(bw_hz),
            center_frequency_hz=float(center_frequency_hz),
            preamble_symbols=int(preamble_symbols),
            sync_word=int(sync_word),
        )
        for cfo_int, rank, signed, relative_db in alternative_specs
    ]
    ranked = [selected] + alternatives
    indexed = tuple(
        AmbiguityRidgeCandidate(
            **{**item.__dict__, "candidate_index": index}
        )
        for index, item in enumerate(ranked)
    )
    return AmbiguityRidgeListResult(
        candidates=indexed,
        unique_integer_cfo_pool=len(by_integer),
        sfd_peaks_examined=len(order),
    )


def arbitrate_sync_list_with_crc(
    samples: np.ndarray,
    candidates: Sequence[AmbiguityRidgeCandidate],
    *,
    sf: int,
    bw_hz: float,
    os_factor: int,
    ldro_mode: int = 2,
    crc_mode: str = "grlora",
    require_payload_crc: bool = True,
    stop_on_crc: bool = True,
    decoder_mode: str = "savaux",
) -> CrcListArbitrationResult:
    """Select the first PHY-CRC success in ranked candidate order.

    The operational default stops immediately after a CRC success.  Offline
    audits can set ``stop_on_crc=False`` to evaluate every listed coordinate.
    """

    mode = str(decoder_mode).lower()
    if mode == "savaux":
        decoder = decode_savaux_sync_candidate
    elif mode == "soft_hamming":
        decoder = decode_soft_hamming_sync_candidate
    else:
        raise ValueError(f"unknown list decoder mode: {decoder_mode}")

    attempts: list[CrcListAttempt] = []
    selected_index: int | None = None
    for candidate in candidates:
        decoded = decoder(
            samples,
            candidate,
            sf=int(sf),
            bw_hz=float(bw_hz),
            os_factor=int(os_factor),
            ldro_mode=int(ldro_mode),
            crc_mode=str(crc_mode),
            allow_gate_failed_candidate=True,
        )
        has_crc = bool(
            decoded.header is not None and decoded.header.has_crc
        )
        accepted = bool(
            decoded.header_valid
            and decoded.crc_valid
            and (has_crc or not bool(require_payload_crc))
        )
        attempts.append(
            CrcListAttempt(
                candidate=candidate,
                decode=decoded,
                crc_accepted=accepted,
            )
        )
        if accepted and selected_index is None:
            selected_index = len(attempts) - 1
            if bool(stop_on_crc):
                break
    return CrcListArbitrationResult(
        attempts=tuple(attempts),
        selected_index=selected_index,
    )


__all__ = [
    "AmbiguityRidgeCandidate",
    "AmbiguityRidgeListResult",
    "CrcListArbitrationResult",
    "CrcListAttempt",
    "arbitrate_sync_list_with_crc",
    "build_ambiguity_ridge_sync_list",
]

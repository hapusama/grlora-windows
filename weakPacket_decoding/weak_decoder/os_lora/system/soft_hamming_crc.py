"""Soft-Hamming packet decoding for oversampled LoRa synchronization lists.

Savaux already supplies the full matched-filter spectrum for every symbol.
The ordinary packet path discards that evidence with an Argmax before LoRa's
interleaver and Hamming decoder.  This module converts each spectrum into bit
log-likelihoods, performs maximum-likelihood decoding over the 16 valid
Hamming codewords, reconstructs the corresponding LoRa symbols, and finally
uses the unchanged explicit-header/FEC/dewhitening/PHY-CRC chain.

The search is bounded: 16 codewords per decoded nibble.  It does not enumerate
payloads, use expected bytes, or accept a packet without its payload CRC.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import numpy as np

from ...baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)
from ...chirp import bin_to_grlora_symbol
from ...decoding.header_first_demod import (
    HeaderDecodeResult,
    advance_symbol_cursor,
    bits_to_int,
    decode_explicit_header,
    int_to_bits_msb,
)
from ...decoding.payload_codec import (
    ExplicitFrameDecodeResult,
    decode_explicit_frame_symbols,
    encode_hamming_nibble,
)
from .decoder_aware_crc import (
    DecoderAwarePacketResult,
    DecoderAwareSymbolDecision,
)


@dataclass(frozen=True)
class SoftHammingBlockResult:
    """One interleaver block after soft Hamming-codeword selection."""

    symbol_values: tuple[int, ...]
    decoded_nibbles: tuple[int, ...]
    codewords: tuple[int, ...]
    mean_codeword_margin: float


@dataclass(frozen=True)
class _SymbolEvidence:
    decision: DecoderAwareSymbolDecision
    power: np.ndarray


def _empty_result(
    status: str,
    *,
    candidate_used: bool,
    header: HeaderDecodeResult | None = None,
    symbols: tuple[DecoderAwareSymbolDecision, ...] = (),
    error: str | None = None,
) -> DecoderAwarePacketResult:
    return DecoderAwarePacketResult(
        status=str(status),
        candidate_used=bool(candidate_used),
        header=header,
        frame=None,
        symbols=tuple(symbols),
        error=error,
    )


def _logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array))
    if not np.isfinite(maximum):
        return -math.inf
    return float(maximum + math.log(float(np.sum(np.exp(array - maximum)))))


def _gray_to_binary(value: int) -> int:
    gray = int(value)
    binary = gray
    shift = 1
    while (gray >> shift) > 0:
        binary ^= gray >> shift
        shift += 1
    return int(binary)


def _semantic_log_scores(power: np.ndarray, divisor: int) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("spectrum power must be a non-empty vector")
    group = int(divisor)
    if group <= 0 or values.size % group != 0:
        raise ValueError("invalid LoRa symbol grouping divisor")

    # The median of exponential noise power is N0*ln(2).  Normalizing by this
    # estimate keeps likelihood magnitudes comparable across packet symbols.
    noise_power = max(
        float(np.median(values)) / math.log(2.0),
        np.finfo(np.float64).tiny,
    )
    raw_scores = values / noise_power
    semantic_count = values.size // group
    semantic = np.full(semantic_count, -math.inf, dtype=np.float64)
    for raw_bin, score in enumerate(raw_scores):
        symbol = ((int(raw_bin) - 1) % values.size) // group
        semantic[symbol] = max(float(semantic[symbol]), float(score))
    semantic -= float(np.max(semantic))
    return np.maximum(semantic, -80.0)


def _bit_log_probabilities(
    power: np.ndarray,
    *,
    sf_app: int,
    divisor: int,
) -> np.ndarray:
    scores = _semantic_log_scores(power, divisor)
    expected = 1 << int(sf_app)
    if scores.size != expected:
        raise ValueError(
            f"semantic score count {scores.size} does not match 2^SFapp={expected}"
        )
    symbols = np.arange(expected, dtype=np.int64)
    gray = symbols ^ (symbols >> 1)
    output = np.empty((int(sf_app), 2), dtype=np.float64)
    for bit_index in range(int(sf_app)):
        shift = int(sf_app) - 1 - bit_index
        bits = (gray >> shift) & 1
        output[bit_index, 0] = _logsumexp(scores[bits == 0])
        output[bit_index, 1] = _logsumexp(scores[bits == 1])
        output[bit_index] -= _logsumexp(output[bit_index])
    return output


def soft_repair_interleaver_block(
    spectrum_powers: Sequence[np.ndarray],
    *,
    sf: int,
    is_header: bool,
    cr: int,
    ldro: bool,
) -> SoftHammingBlockResult:
    """Soft-decode one LoRa interleaver block over valid Hamming words."""

    sf_value = int(sf)
    cr_app = 4 if bool(is_header) else int(cr)
    cw_len = 8 if bool(is_header) else 4 + cr_app
    if not 1 <= cr_app <= 4:
        raise ValueError("LoRa code rate must be in [1, 4]")
    if len(spectrum_powers) != cw_len:
        raise ValueError(
            f"interleaver block needs {cw_len} spectra, got {len(spectrum_powers)}"
        )
    sf_app = sf_value - 2 if bool(is_header) or bool(ldro) else sf_value
    divisor = 4 if bool(is_header) or bool(ldro) else 1
    bit_logs = np.stack(
        [
            _bit_log_probabilities(
                power,
                sf_app=sf_app,
                divisor=divisor,
            )
            for power in spectrum_powers
        ],
        axis=0,
    )

    selected_nibbles: list[int] = []
    selected_codewords: list[int] = []
    margins: list[float] = []
    for row in range(sf_app):
        candidate_scores = np.empty(16, dtype=np.float64)
        candidate_codewords = np.empty(16, dtype=np.int64)
        for nibble in range(16):
            codeword = encode_hamming_nibble(nibble, cr_app=cr_app)
            bits = int_to_bits_msb(codeword, cw_len)
            score = 0.0
            for column, bit in enumerate(bits):
                symbol_bit = (column - row - 1) % sf_app
                score += float(bit_logs[column, symbol_bit, int(bool(bit))])
            candidate_scores[nibble] = score
            candidate_codewords[nibble] = codeword
        order = np.argsort(candidate_scores)[::-1]
        best = int(order[0])
        selected_nibbles.append(best)
        selected_codewords.append(int(candidate_codewords[best]))
        margins.append(
            float(candidate_scores[best] - candidate_scores[int(order[1])])
        )

    codeword_bits = [
        int_to_bits_msb(codeword, cw_len) for codeword in selected_codewords
    ]
    repaired_symbols: list[int] = []
    for column in range(cw_len):
        gray_bits = [
            codeword_bits[(column - bit_index - 1) % sf_app][column]
            for bit_index in range(sf_app)
        ]
        repaired_symbols.append(_gray_to_binary(bits_to_int(gray_bits)))
    return SoftHammingBlockResult(
        symbol_values=tuple(repaired_symbols),
        decoded_nibbles=tuple(selected_nibbles),
        codewords=tuple(selected_codewords),
        mean_codeword_margin=float(np.mean(margins)),
    )


def _demod_symbol_evidence(
    samples: np.ndarray,
    *,
    start_sample: int,
    header_start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    stage: str,
    frame_symbol_index: int,
    stage_symbol_index: int,
    payload_ldro: bool,
) -> _SymbolEvidence:
    origin_shift = int(os_factor) // 2
    spectrum, _branches, _phase = paper_oversampled_spectrum(
        samples=np.asarray(samples, dtype=np.complex64),
        start_sample=int(start_sample) + origin_shift,
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=int(header_start_sample) + origin_shift,
        cfo_correction_mode="continuous",
    )
    power = np.abs(spectrum).astype(np.float64) ** 2
    raw_bin = int(np.argmax(power))
    first = float(power[raw_bin])
    second = float(np.partition(power, -2)[-2]) if power.size > 1 else 0.0
    is_header = str(stage) == "header"
    decision = DecoderAwareSymbolDecision(
        stage=str(stage),
        frame_symbol_index=int(frame_symbol_index),
        stage_symbol_index=int(stage_symbol_index),
        start_sample=int(start_sample),
        raw_fft_bin=raw_bin,
        symbol_value=bin_to_grlora_symbol(
            raw_bin,
            sf=int(sf),
            is_header=is_header,
            ldro=bool(payload_ldro),
        ),
        peak_margin_db=float(
            10.0 * math.log10((first + 1e-30) / (second + 1e-30))
        ),
    )
    return _SymbolEvidence(decision=decision, power=power)


def _replace_decisions(
    evidence: Sequence[_SymbolEvidence],
    symbol_values: Sequence[int],
) -> list[DecoderAwareSymbolDecision]:
    if len(evidence) != len(symbol_values):
        raise ValueError("evidence and repaired symbol counts differ")
    return [
        replace(item.decision, symbol_value=int(symbol))
        for item, symbol in zip(evidence, symbol_values, strict=True)
    ]


def decode_soft_hamming_sync_candidate(
    samples: np.ndarray,
    frame_sync: Any | None,
    *,
    sf: int,
    bw_hz: float,
    os_factor: int,
    ldro_mode: int = 2,
    crc_mode: str = "grlora",
    allow_gate_failed_candidate: bool = False,
    max_payload_symbols: int = 512,
) -> DecoderAwarePacketResult:
    """Decode one sync coordinate with Savaux likelihoods and soft Hamming."""

    if frame_sync is None:
        return _empty_result("sync_missing", candidate_used=False)
    if not bool(getattr(frame_sync, "valid", False)) and not bool(
        allow_gate_failed_candidate
    ):
        return _empty_result("sync_gate_rejected", candidate_used=False)
    if int(os_factor) <= 0:
        raise ValueError("os_factor must be positive")
    required = (
        "fine_payload_start_sample",
        "cfo_int_est",
        "cfo_frac_est",
        "sfo_hat",
        "sfo_cum_initial",
    )
    missing = [name for name in required if not hasattr(frame_sync, name)]
    if missing:
        return _empty_result(
            "sync_fields_missing",
            candidate_used=False,
            error="missing FrameSync fields: " + ", ".join(missing),
        )

    values = np.asarray(samples, dtype=np.complex64)
    header_start = int(frame_sync.fine_payload_start_sample)
    cursor = header_start
    sfo_cum = float(frame_sync.sfo_cum_initial)
    header_evidence: list[_SymbolEvidence] = []
    try:
        for index in range(8):
            header_evidence.append(
                _demod_symbol_evidence(
                    values,
                    start_sample=cursor,
                    header_start_sample=header_start,
                    sf=int(sf),
                    os_factor=int(os_factor),
                    cfo_int=int(frame_sync.cfo_int_est),
                    cfo_frac=float(frame_sync.cfo_frac_est),
                    stage="header",
                    frame_symbol_index=index,
                    stage_symbol_index=index,
                    payload_ldro=False,
                )
            )
            cursor, sfo_cum, _adjust = advance_symbol_cursor(
                cursor,
                sf=int(sf),
                os_factor=int(os_factor),
                sfo_cum=sfo_cum,
                sfo_hat=float(frame_sync.sfo_hat),
            )
    except ValueError as exc:
        return _empty_result(
            "truncated_header",
            candidate_used=True,
            symbols=tuple(item.decision for item in header_evidence),
            error=str(exc),
        )

    repaired_header = soft_repair_interleaver_block(
        [item.power for item in header_evidence],
        sf=int(sf),
        is_header=True,
        cr=4,
        ldro=False,
    )
    decisions = _replace_decisions(
        header_evidence,
        repaired_header.symbol_values,
    )
    header = decode_explicit_header(
        repaired_header.symbol_values,
        sf=int(sf),
        bw=float(bw_hz),
        ldro_mode=int(ldro_mode),
    )
    if not header.header_valid:
        return _empty_result(
            "header_invalid",
            candidate_used=True,
            header=header,
            symbols=tuple(decisions),
        )
    # The five-bit explicit-header checksum does not constrain the CR field to
    # LoRa's legal range.  At very low SNR a checksum-valid random header can
    # therefore carry CR=0 or CR>4.  Treat it as an invalid candidate instead
    # of letting the bounded Hamming decoder raise and abort the whole list.
    if not 1 <= int(header.cr) <= 4:
        return _empty_result(
            "header_cr_invalid",
            candidate_used=True,
            header=header,
            symbols=tuple(decisions),
            error=f"decoded LoRa code rate is {header.cr}",
        )
    if not 0 <= int(header.payload_symbol_count) <= int(max_payload_symbols):
        return _empty_result(
            "payload_count_invalid",
            candidate_used=True,
            header=header,
            symbols=tuple(decisions),
            error=f"decoded payload symbol count is {header.payload_symbol_count}",
        )

    payload_evidence: list[_SymbolEvidence] = []
    try:
        for payload_index in range(int(header.payload_symbol_count)):
            frame_index = 8 + payload_index
            payload_evidence.append(
                _demod_symbol_evidence(
                    values,
                    start_sample=cursor,
                    header_start_sample=header_start,
                    sf=int(sf),
                    os_factor=int(os_factor),
                    cfo_int=int(frame_sync.cfo_int_est),
                    cfo_frac=float(frame_sync.cfo_frac_est),
                    stage="payload",
                    frame_symbol_index=frame_index,
                    stage_symbol_index=payload_index,
                    payload_ldro=bool(header.ldro),
                )
            )
            cursor, sfo_cum, _adjust = advance_symbol_cursor(
                cursor,
                sf=int(sf),
                os_factor=int(os_factor),
                sfo_cum=sfo_cum,
                sfo_hat=float(frame_sync.sfo_hat),
            )
    except ValueError as exc:
        return _empty_result(
            "truncated_payload",
            candidate_used=True,
            header=header,
            symbols=tuple(decisions)
            + tuple(item.decision for item in payload_evidence),
            error=str(exc),
        )

    cw_len = int(header.cr) + 4
    repaired_payload_values: list[int] = []
    for block_start in range(0, len(payload_evidence), cw_len):
        block = payload_evidence[block_start : block_start + cw_len]
        if len(block) != cw_len:
            return _empty_result(
                "payload_block_truncated",
                candidate_used=True,
                header=header,
                symbols=tuple(decisions)
                + tuple(item.decision for item in payload_evidence),
            )
        repaired = soft_repair_interleaver_block(
            [item.power for item in block],
            sf=int(sf),
            is_header=False,
            cr=int(header.cr),
            ldro=bool(header.ldro),
        )
        repaired_payload_values.extend(repaired.symbol_values)
    decisions.extend(
        _replace_decisions(payload_evidence, repaired_payload_values)
    )
    frame: ExplicitFrameDecodeResult = decode_explicit_frame_symbols(
        header_symbol_values=repaired_header.symbol_values,
        payload_symbol_values=repaired_payload_values,
        sf=int(sf),
        bw=float(bw_hz),
        ldro_mode=int(ldro_mode),
        crc_mode=str(crc_mode),
    )
    return DecoderAwarePacketResult(
        status="ok",
        candidate_used=True,
        header=header,
        frame=frame,
        symbols=tuple(decisions),
    )


__all__ = [
    "SoftHammingBlockResult",
    "decode_soft_hamming_sync_candidate",
    "soft_repair_interleaver_block",
]

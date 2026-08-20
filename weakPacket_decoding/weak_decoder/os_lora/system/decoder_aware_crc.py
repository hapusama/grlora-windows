"""Decoder-aware Savaux packet decoding for weak FrameSync candidates.

The ordinary synchronization path rejects a ``GrloraFrameSyncResult`` when
one of its validation gates fails.  For weak packets, the estimated timing,
CFO, and SFO can still be useful.  This module keeps the default strict
behaviour, but offers an explicit opt-in that lets a caller demodulate such a
candidate and defer the final decision to the PHY header and CRC.

No reference symbols or expected payload bytes are used here.  Ground truth
comparison belongs in the experiment layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ...baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)
from ...chirp import bin_to_grlora_symbol
from ...decoding.header_first_demod import (
    HeaderDecodeResult,
    advance_symbol_cursor,
    decode_explicit_header,
)
from ...decoding.payload_codec import (
    ExplicitFrameDecodeResult,
    decode_explicit_frame_symbols,
)


@dataclass(frozen=True)
class DecoderAwareSymbolDecision:
    """One hard Savaux decision retained for packet-level diagnostics."""

    stage: str
    frame_symbol_index: int
    stage_symbol_index: int
    start_sample: int
    raw_fft_bin: int
    symbol_value: int
    peak_margin_db: float


@dataclass(frozen=True)
class DecoderAwarePacketResult:
    """Result of decoding one synchronization candidate through PHY CRC."""

    status: str
    candidate_used: bool
    header: HeaderDecodeResult | None
    frame: ExplicitFrameDecodeResult | None
    symbols: tuple[DecoderAwareSymbolDecision, ...]
    error: str | None = None

    @property
    def header_valid(self) -> bool:
        return bool(self.header is not None and self.header.header_valid)

    @property
    def crc_valid(self) -> bool:
        return bool(self.frame is not None and self.frame.payload.crc_valid)

    @property
    def payload_bytes(self) -> bytes:
        return b"" if self.frame is None else self.frame.payload.payload_bytes

    @property
    def header_symbols(self) -> tuple[DecoderAwareSymbolDecision, ...]:
        return tuple(item for item in self.symbols if item.stage == "header")

    @property
    def payload_symbols(self) -> tuple[DecoderAwareSymbolDecision, ...]:
        return tuple(item for item in self.symbols if item.stage == "payload")


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


def _demod_symbol(
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
) -> DecoderAwareSymbolDecision:
    # FrameSync reports an interval boundary.  Savaux's q=0 branch origin is
    # the chip-centre phase used by the gr-lora_sdr hard demodulator.
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
    return DecoderAwareSymbolDecision(
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


def decode_savaux_sync_candidate(
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
    """Decode one FrameSync estimate through Savaux, FEC, whitening, and CRC.

    ``allow_gate_failed_candidate`` defaults to ``False`` so existing strict
    behaviour is preserved.  A decoder-aware caller may set it only after its
    own candidate policy has established that the locator and net-ID evidence
    are usable.  The decoded header still controls CR, LDRO, payload length,
    and the number of payload symbols; metadata is never consulted.
    """

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
    decisions: list[DecoderAwareSymbolDecision] = []

    try:
        for index in range(8):
            decisions.append(
                _demod_symbol(
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
            symbols=tuple(decisions),
            error=str(exc),
        )

    header = decode_explicit_header(
        [item.symbol_value for item in decisions],
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
    if not 0 <= int(header.payload_symbol_count) <= int(max_payload_symbols):
        return _empty_result(
            "payload_count_invalid",
            candidate_used=True,
            header=header,
            symbols=tuple(decisions),
            error=f"decoded payload symbol count is {header.payload_symbol_count}",
        )

    try:
        for payload_index in range(int(header.payload_symbol_count)):
            frame_index = 8 + payload_index
            decisions.append(
                _demod_symbol(
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
            symbols=tuple(decisions),
            error=str(exc),
        )

    frame = decode_explicit_frame_symbols(
        header_symbol_values=[
            item.symbol_value for item in decisions if item.stage == "header"
        ],
        payload_symbol_values=[
            item.symbol_value for item in decisions if item.stage == "payload"
        ],
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
    "DecoderAwarePacketResult",
    "DecoderAwareSymbolDecision",
    "decode_savaux_sync_candidate",
]

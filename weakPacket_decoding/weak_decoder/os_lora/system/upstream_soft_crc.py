"""Faithful port of gr-lora_sdr native soft decoding for sync-candidate lists.

This module ports the upstream ``soft_decoding=True, max_log_approx=True``
chain from ``lib/fft_demod_impl.cc::get_LLRs``, ``lib/deinterleaver_impl.cc``
and ``lib/hamming_dec_impl.cc`` so the same Savaux matched-filter spectra used
by the other list decoders can be decoded with the reference receiver's own
soft metric.  It exists to answer the HANDOFF_20260820 Step-1 fairness
question: how much of the ridge+soft gain is attributable to soft decoding
that upstream already ships.

Only numpy is required; ``log(I0)`` is evaluated with a series/asymptotic
split.  Nibble conventions differ between upstream and the shared decoding
helpers: upstream data nibbles are the bit-reversal of the wire nibbles used
by :func:`encode_hamming_nibble`; the codeword values themselves coincide
after cropping the LUT entries to ``cw_len`` bits.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ...decoding.header_first_demod import bits_to_int, int_to_bits_msb
from .decoder_aware_crc import DecoderAwarePacketResult
from .soft_hamming_crc import (
    SoftHammingBlockResult,
    _gray_to_binary,
    decode_soft_hamming_sync_candidate,
)

# Hard-coded 16-codeword LUTs from lib/hamming_dec_impl.cc.
_CW_LUT = (0, 23, 45, 58, 78, 89, 99, 116, 139, 156, 166, 177, 197, 210, 232, 255)
_CW_LUT_CR5 = (
    0,
    24,
    40,
    48,
    72,
    80,
    96,
    120,
    136,
    144,
    160,
    184,
    192,
    216,
    232,
    240,
)

# fft_demod_impl.cc switches to the linear |Y|^2*N fallback above this.
_BESSEL_ARG_CLIP = 713.0


def log_i0(x):
    """Evaluate log(I0(x)) for non-negative arrays with numpy only.

    Power series for x <= 20 and the standard asymptotic expansion above,
    matching boost::math::cyl_bessel_i(0, x) to well below the LLR precision
    that the max-log demodulator can resolve.
    """

    values = np.asarray(x, dtype=np.float64)
    result = np.empty_like(values)
    small = values <= 20.0
    xs = values[small]
    if xs.size:
        total = np.ones_like(xs)
        term = np.ones_like(xs)
        half_sq = (xs * 0.5) ** 2
        for k in range(1, 256):
            term = term * half_sq / float(k * k)
            total = total + term
            if bool(np.all(term <= total * 1e-18)):
                break
        result[small] = np.log(total)
    xl = values[~small]
    if xl.size:
        inv = 1.0 / xl
        correction = (
            inv / 8.0
            + 9.0 * inv * inv / 128.0
            + 225.0 * inv**3 / 3072.0
            + 11025.0 * inv**4 / 98304.0
        )
        result[~small] = xl - 0.5 * np.log(2.0 * np.pi * xl) + np.log1p(correction)
    return result


def upstream_symbol_bit_llrs(
    power: np.ndarray,
    *,
    sf: int,
    reduced_rate: bool,
) -> np.ndarray:
    """Max-log Bessel LLRs exactly as fft_demod_impl.cc::get_LLRs.

    ``power`` is |FFT|^2 of the dechirped symbol with one entry per bin
    (2**sf bins); any common scale cancels.  Returns ``sf`` LLRs ordered
    MSB-first, the convention the upstream soft deinterleaver consumes.
    """

    values = np.asarray(power, dtype=np.float64)
    n_bins = values.size
    if n_bins != (1 << int(sf)) or n_bins == 0:
        raise ValueError(
            f"spectrum must hold 2**sf bins, got {n_bins} for sf={sf}"
        )

    # Per-symbol SNR split with n_adjacent_bins = 1, including the upstream
    # modulus (N-1) quirk that also captures the farthest wrap-around bin.
    symbol_idx = int(np.argmax(values))
    offsets = np.abs(np.arange(n_bins) - symbol_idx)
    signal_mask = np.mod(offsets, n_bins - 1) < 2
    signal_energy = float(np.sum(values[signal_mask]))
    noise_energy = float(np.sum(values[~signal_mask]))
    ps_est = max(signal_energy / n_bins, np.finfo(np.float64).tiny)
    pn_est = max(noise_energy / (n_bins - 3), np.finfo(np.float64).tiny)

    # Upstream normalizes |Y|^2 to |Y|^2 * N before the Bessel argument.
    mag_sq = values * n_bins
    bessel_arg = math.sqrt(ps_est) / pn_est * np.sqrt(mag_sq)
    if bool(np.any(bessel_arg >= _BESSEL_ARG_CLIP)):
        log_likelihoods = mag_sq
    else:
        log_likelihoods = log_i0(bessel_arg)

    divisor = 4 if reduced_rate else 1
    symbols = (((np.arange(n_bins) - 1) % n_bins) // divisor).astype(np.int64)
    symbols ^= symbols >> 1  # Gray demap, folded into get_LLRs

    llrs = np.zeros(int(sf), dtype=np.float64)
    for bit in range(int(sf)):
        ones_mask = ((symbols >> bit) & 1) == 1
        max_one = (
            float(np.max(log_likelihoods[ones_mask]))
            if bool(np.any(ones_mask))
            else 0.0
        )
        max_zero = (
            float(np.max(log_likelihoods[~ones_mask]))
            if bool(np.any(~ones_mask))
            else 0.0
        )
        llrs[int(sf) - 1 - bit] = max_one - max_zero
    return llrs


def _bit_reverse4(value: int) -> int:
    nibble = int(value) & 0x0F
    return ((nibble & 1) << 3) | ((nibble & 2) << 1) | ((nibble & 4) >> 1) | (
        (nibble & 8) >> 3
    )


def upstream_soft_repair_interleaver_block(
    spectrum_powers: Sequence[np.ndarray],
    *,
    sf: int,
    is_header: bool,
    cr: int,
    ldro: bool,
) -> SoftHammingBlockResult:
    """Decode one interleaver block through the upstream soft chain.

    Same contract as :func:`soft_repair_interleaver_block` but every stage is
    the gr-lora_sdr formula: Bessel-I0 max-log bit LLRs, the upstream soft
    deinterleaver permutation, and the 16-entry LUT Hamming scorer.  Margins
    are reported in upstream max-log |LLR| units, not log-likelihood units.
    """

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

    bit_llrs = np.stack(
        [
            upstream_symbol_bit_llrs(
                power,
                sf=sf_value,
                reduced_rate=bool(is_header) or bool(ldro),
            )
            for power in spectrum_powers
        ],
        axis=0,
    )  # (cw_len, sf) MSB-first per symbol, as fft_demod outputs them

    # deinterleaver_impl.cc soft branch: keep the low sf_app LLRs, then
    # deinter_bin[mod(i - j - 1, sf_app)][i] = inter_bin[i][j].
    inter_bin = bit_llrs[:, sf_value - sf_app :]
    deinter_bin = np.empty((sf_app, cw_len), dtype=np.float64)
    for i in range(cw_len):
        for j in range(sf_app):
            deinter_bin[(i - j - 1) % sf_app, i] = inter_bin[i, j]

    # hamming_dec_impl.cc soft branch over the hard-coded LUTs.
    lut = _CW_LUT_CR5 if cr_app == 1 else _CW_LUT
    cw_bits = np.empty((16, cw_len), dtype=np.int64)
    for entry in range(16):
        cropped = lut[entry] >> (8 - cw_len)
        for j in range(cw_len):
            cw_bits[entry, j] = (cropped >> (cw_len - 1 - j)) & 1

    abs_llr = np.abs(deinter_bin)  # (sf_app, cw_len)
    hard_bit = (deinter_bin > 0).astype(np.int64)
    match = cw_bits[None, :, :] == hard_bit[:, None, :]  # (sf_app, 16, cw_len)
    scores = np.sum(
        np.where(match, abs_llr[:, None, :], -abs_llr[:, None, :]),
        axis=2,
    )  # (sf_app, 16)
    order = np.argsort(scores, axis=1)[:, ::-1]
    best_entries = order[:, 0]
    second_entries = order[:, 1]
    rows = np.arange(sf_app)
    margins = scores[rows, best_entries] - scores[rows, second_entries]

    selected_codewords = [
        int(lut[int(entry)] >> (8 - cw_len)) for entry in best_entries
    ]
    # Upstream data nibble = LUT >> 4; the shared codec uses the bit-reversed
    # wire convention (see hamming_dec_impl.cc line 131 doing the same).
    selected_nibbles = [
        _bit_reverse4(int(lut[int(entry)] >> 4)) for entry in best_entries
    ]

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


def decode_upstream_soft_sync_candidate(
    samples: np.ndarray,
    frame_sync,
    **kwargs,
) -> DecoderAwarePacketResult:
    """Decode one sync candidate with the upstream gr-lora soft chain.

    Accepts the same arguments as
    :func:`decode_soft_hamming_sync_candidate` and reuses its whole
    header/FEC/dewhitening/CRC orchestration, swapping only the per-block
    soft repair for the ported upstream metric.
    """

    return decode_soft_hamming_sync_candidate(
        samples,
        frame_sync,
        repair_block=upstream_soft_repair_interleaver_block,
        **kwargs,
    )


__all__ = [
    "decode_upstream_soft_sync_candidate",
    "log_i0",
    "upstream_soft_repair_interleaver_block",
    "upstream_symbol_bit_llrs",
]

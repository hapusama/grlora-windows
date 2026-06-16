"""Two-stage weak-packet decoder using LoRa codec-block constraints.

This module is intentionally single-packet and PHY-only:

* no payload template
* no counter model
* no cross-packet joint prior
* no application-layer byte trajectory

The decoder starts after weak sync + header-first have already succeeded.  It
keeps the full complex FFT evidence for each payload symbol, converts that into
symbol-value likelihoods, builds Hamming-codeword nibble lists per interleaver
block, then runs a small beam search over codec-valid candidates.  Final
payload candidates are re-encoded through the full local LoRa PHY codec and
rescored against the original FFT evidence, including CRC symbols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import time
from typing import Iterable, Sequence

import numpy as np

from .chirp import bin_to_grlora_symbol, positive_mod
from .header_first_demod import bits_to_int, int_to_bits_msb
from .payload_codec import (
    WHITENING_SEQ,
    crc16,
    encode_explicit_frame_symbols,
    encode_hamming_nibble,
    explicit_header_tail_nibbles,
    gray_demap_symbols,
    nibbles_to_dewhitened_bytes,
    payload_symbols_to_nibbles,
    reencoded_payload_known_prefix_symbols,
)
from .phase_guided_demod import (
    PhaseGuidedPayloadConfig,
    PhaseLine,
    _score_payload_symbol_prior_candidate,
)


def _logsumexp(values: np.ndarray) -> float:
    if values.size == 0:
        return float("-inf")
    vmax = float(np.max(values))
    if not math.isfinite(vmax):
        return vmax
    return float(vmax + math.log(float(np.sum(np.exp(values - vmax)))))


def _trimmed_sum(values: Sequence[float], trim_fraction: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("-inf")
    trim_count = int(math.floor(arr.size * max(0.0, min(0.45, float(trim_fraction)))))
    if trim_count > 0 and arr.size - trim_count >= 1:
        arr = np.sort(arr)[trim_count:]
    return float(np.sum(arr))


def _symbol_alphabet_size(sf: int, ldro: bool) -> int:
    return 1 << (int(sf) - 2 if bool(ldro) else int(sf))


def _symbol_value_to_raw_bin(symbol_value: int, sf: int, ldro: bool) -> int:
    n_bins = 1 << int(sf)
    divisor = 4 if bool(ldro) else 1
    return int((int(symbol_value) * divisor + 1) % n_bins)


def _payload_codewords_to_symbol_values(
    codewords: Sequence[int],
    sf: int,
    cr: int,
    ldro: bool,
) -> tuple[int, ...]:
    """Project one payload interleaver block of codewords to demod symbol values."""

    sf_i = int(sf)
    cr_i = int(cr)
    cw_len = cr_i + 4
    sf_app = sf_i - 2 if bool(ldro) else sf_i
    if len(codewords) != sf_app:
        raise ValueError(f"payload block needs {sf_app} codewords, got {len(codewords)}")

    cw_bin = [int_to_bits_msb(int(cw), cw_len) for cw in codewords]
    interleaved: list[int] = []
    for i in range(cw_len):
        row = [False for _ in range(sf_i)]
        for j in range(sf_app):
            row[j] = cw_bin[positive_mod(i - j - 1, sf_app)][i]
        if bool(ldro):
            row[sf_app] = bool(sum(1 for bit in row[:sf_app] if bit) % 2)
        interleaved.append(bits_to_int(row))

    raw_bins = gray_demap_symbols(interleaved, sf=sf_i)
    divisor = 4 if bool(ldro) else 1
    n_bins = 1 << sf_i
    return tuple(int(((raw_bin - 1) % n_bins) // divisor) for raw_bin in raw_bins)


def _expected_payload_symbol_count(
    payload_len: int,
    sf: int,
    cr: int,
    has_crc: bool,
    ldro: bool,
    crc_mode: str,
) -> int:
    """Return the PHY payload-symbol count implied by the decoded header."""

    dummy_payload = bytes(int(payload_len))
    _header_symbols, payload_symbols = encode_explicit_frame_symbols(
        payload=dummy_payload,
        sf=sf,
        cr=cr,
        has_crc=has_crc,
        ldro=ldro,
        crc_mode=crc_mode,
    )
    return int(len(payload_symbols))


def _verify_payload_crc(
    payload: bytes,
    crc_bytes: bytes,
    has_crc: bool,
    crc_mode: str,
) -> tuple[bool, int, int]:
    if not bool(has_crc):
        return True, 0, 0
    if len(crc_bytes) < 2:
        return False, 0, 0
    if str(crc_mode).lower() == "sx1276":
        computed = crc16(payload)
    else:
        if len(payload) < 2:
            computed = crc16(payload)
        else:
            computed = crc16(payload[: max(0, len(payload) - 2)])
            computed ^= int(payload[-1])
            computed ^= int(payload[-2]) << 8
    received = int(crc_bytes[0]) + (int(crc_bytes[1]) << 8)
    return bool(received == computed), int(computed), int(received)


@dataclass(frozen=True)
class TwoStageWeakConfig:
    """Tunable parameters for the two-stage decoder."""

    top_k_metrics: int = 32
    amplitude_floor_db: float = 24.0
    phase_weight: float = 0.10
    bit_metric: str = "max"  # "max" or "logsumexp"
    nibble_candidates_per_codeword: int = 4
    row_beam_width: int = 256
    block_candidate_limit: int = 64
    global_beam_width: int = 64
    final_candidate_limit: int = 128
    block_trim_fraction: float = 0.20
    projection_trim_fraction: float = 0.10
    projection_score_weight: float = 1.00
    trajectory_score_weight: float = 0.00
    crc_observed_bonus: float = 4.0
    crc_candidate_min_evidence_margin: float = 0.0
    crc_candidate_max_beam_rank: int = 2048
    crc_mode: str = "grlora"
    argmax_fallback_on_crc_failure: bool = True


@dataclass(frozen=True)
class SymbolLikelihood:
    """Full FFT evidence compressed into symbol-value likelihoods."""

    symbol_index: int
    raw_scores: np.ndarray
    score_by_value: np.ndarray
    best_raw_bin_by_value: np.ndarray
    argmax_bin: int
    argmax_symbol_value: int
    top_bins: tuple[int, ...]
    top_scores: tuple[float, ...]
    phase_predicted: float

    def score_symbol(self, symbol_value: int) -> float:
        value = int(symbol_value)
        if 0 <= value < self.score_by_value.size:
            return float(self.score_by_value[value])
        return float("-inf")

    def raw_rank(self, raw_bin: int) -> int:
        raw_bin = int(raw_bin)
        if raw_bin < 0 or raw_bin >= self.raw_scores.size:
            return -1
        order = np.argsort(self.raw_scores)[::-1]
        matches = np.where(order == raw_bin)[0]
        return int(matches[0] + 1) if matches.size else -1


@dataclass(frozen=True)
class NibbleCandidate:
    nibble: int
    codeword: int
    approx_score: float


@dataclass(frozen=True)
class CodewordList:
    block_index: int
    codeword_index: int
    candidates: tuple[NibbleCandidate, ...]


@dataclass(frozen=True)
class BlockCandidate:
    block_index: int
    nibbles: tuple[int, ...]
    codewords: tuple[int, ...]
    symbol_values: tuple[int, ...]
    raw_bins: tuple[int, ...]
    approx_score: float
    projection_score: float
    total_score: float


@dataclass(order=True)
class _BeamState:
    sort_key: float
    nibbles: tuple[int, ...] = field(compare=False)
    codewords: tuple[int, ...] = field(compare=False)
    symbol_values: tuple[int, ...] = field(compare=False)
    raw_bins: tuple[int, ...] = field(compare=False)
    score: float = field(compare=False)


@dataclass(frozen=True)
class PayloadBeamCandidate:
    payload_bytes: bytes
    crc_bytes_observed: bytes
    observed_crc_valid: bool
    crc_computed: int
    crc_received: int
    frame_nibbles: tuple[int, ...]
    projected_payload_symbol_values: tuple[int, ...]
    projected_raw_bins: tuple[int, ...]
    payload_symbol_values: tuple[int, ...]
    raw_bins: tuple[int, ...]
    block_score: float
    projection_score: float
    trajectory_score: float
    total_score: float
    beam_rank: int
    selection_source: str = "beam"


@dataclass(frozen=True)
class TwoStageWeakResult:
    success: bool
    selected: PayloadBeamCandidate | None
    candidates: tuple[PayloadBeamCandidate, ...]
    argmax_symbol_values: tuple[int, ...]
    argmax_raw_bins: tuple[int, ...]
    likelihoods: tuple[SymbolLikelihood, ...]
    codeword_lists: tuple[CodewordList, ...]
    block_candidates: tuple[tuple[BlockCandidate, ...], ...]
    timings_ms: dict[str, float]
    metrics: dict[str, float | int]
    error: str = ""


def build_symbol_likelihood(
    spectrum: np.ndarray,
    symbol_index: int,
    sf: int,
    ldro: bool,
    predicted_phase_rad: float,
    config: TwoStageWeakConfig,
    raw_score_override: np.ndarray | None = None,
) -> SymbolLikelihood:
    """Build raw-bin and demod-symbol likelihoods from one FFT spectrum."""

    spec = np.asarray(spectrum, dtype=np.complex64)
    n_bins = 1 << int(sf)
    alphabet = _symbol_alphabet_size(sf, ldro)
    if spec.size != n_bins:
        raise ValueError(f"spectrum has {spec.size} bins, expected {n_bins}")

    power = np.abs(spec).astype(np.float64) ** 2
    if raw_score_override is None:
        max_power = float(np.max(power)) if power.size else 0.0
        rel_db = 10.0 * np.log10((power + 1e-30) / (max_power + 1e-30))
        floor_db = max(1.0, float(config.amplitude_floor_db))
        amp_score = np.maximum(rel_db, -floor_db) / floor_db

        phases = np.angle(spec)
        residual = np.angle(np.exp(1j * (phases - float(predicted_phase_rad))))
        phase_penalty = 0.5 * (np.cos(residual) - 1.0)
        raw_scores = amp_score + float(config.phase_weight) * phase_penalty
        argmax_metric = power
    else:
        raw_scores = np.asarray(raw_score_override, dtype=np.float64)
        if raw_scores.size != n_bins:
            raise ValueError(
                f"raw_score_override has {raw_scores.size} bins, expected {n_bins}"
            )
        finite = np.isfinite(raw_scores)
        if not np.any(finite):
            raw_scores = np.zeros(n_bins, dtype=np.float64)
        else:
            floor = float(np.min(raw_scores[finite])) - 1.0
            raw_scores = np.where(finite, raw_scores, floor)
        argmax_metric = raw_scores
    raw_scores = raw_scores - float(np.max(raw_scores))

    score_by_value = np.full(alphabet, -np.inf, dtype=np.float64)
    best_raw_bin_by_value = np.full(alphabet, -1, dtype=np.int32)
    for raw_bin, score in enumerate(raw_scores):
        symbol_value = bin_to_grlora_symbol(raw_bin, sf=sf, is_header=False, ldro=ldro)
        if 0 <= symbol_value < alphabet and float(score) > float(score_by_value[symbol_value]):
            score_by_value[symbol_value] = float(score)
            best_raw_bin_by_value[symbol_value] = int(raw_bin)

    order = np.argsort(raw_scores)[::-1]
    top_count = min(int(config.top_k_metrics), int(order.size))
    top_bins = tuple(int(v) for v in order[:top_count])
    top_scores = tuple(float(raw_scores[v]) for v in top_bins)
    argmax_bin = int(np.argmax(argmax_metric))
    argmax_symbol_value = bin_to_grlora_symbol(argmax_bin, sf=sf, is_header=False, ldro=ldro)
    return SymbolLikelihood(
        symbol_index=int(symbol_index),
        raw_scores=raw_scores,
        score_by_value=score_by_value,
        best_raw_bin_by_value=best_raw_bin_by_value,
        argmax_bin=argmax_bin,
        argmax_symbol_value=int(argmax_symbol_value),
        top_bins=top_bins,
        top_scores=top_scores,
        phase_predicted=float(predicted_phase_rad),
    )


def _bit_scores_for_symbol(
    likelihood: SymbolLikelihood,
    bit_position: int,
    sf_app: int,
    metric: str,
) -> tuple[float, float]:
    values = np.arange(likelihood.score_by_value.size, dtype=np.int32)
    finite = np.isfinite(likelihood.score_by_value)
    if not np.any(finite):
        return 0.0, 0.0
    # RX deinterleaving consumes gray_mapping(symbol_value), not the demod
    # symbol value directly.  This mirrors header_first_demod.gray_demapping().
    gray_values = values ^ (values >> 1)
    bit_mask = ((gray_values >> (int(sf_app) - 1 - int(bit_position))) & 1).astype(bool)
    scores0 = likelihood.score_by_value[finite & ~bit_mask]
    scores1 = likelihood.score_by_value[finite & bit_mask]
    if str(metric).lower() == "logsumexp":
        return _logsumexp(scores0), _logsumexp(scores1)
    return (
        float(np.max(scores0)) if scores0.size else float("-inf"),
        float(np.max(scores1)) if scores1.size else float("-inf"),
    )


def enumerate_codeword_lists_for_block(
    block_index: int,
    block_likelihoods: Sequence[SymbolLikelihood],
    sf: int,
    cr: int,
    ldro: bool,
    config: TwoStageWeakConfig,
) -> tuple[CodewordList, ...]:
    """Stage 1: enumerate 16 legal Hamming nibbles for each codeword row."""

    cw_len = int(cr) + 4
    sf_app = int(sf) - 2 if bool(ldro) else int(sf)
    if len(block_likelihoods) != cw_len:
        raise ValueError(f"block needs {cw_len} symbol likelihoods")

    out: list[CodewordList] = []
    for row in range(sf_app):
        bit_scores: list[tuple[float, float]] = []
        for bit_idx in range(cw_len):
            symbol_bit_pos = positive_mod(bit_idx - row - 1, sf_app)
            bit_scores.append(
                _bit_scores_for_symbol(
                    block_likelihoods[bit_idx],
                    symbol_bit_pos,
                    sf_app,
                    metric=config.bit_metric,
                )
            )

        candidates: list[NibbleCandidate] = []
        for nibble in range(16):
            codeword = encode_hamming_nibble(nibble, cr_app=int(cr))
            bits = int_to_bits_msb(codeword, cw_len)
            score = 0.0
            for idx, bit in enumerate(bits):
                score += bit_scores[idx][1 if bit else 0]
            candidates.append(
                NibbleCandidate(
                    nibble=int(nibble),
                    codeword=int(codeword),
                    approx_score=float(score),
                )
            )
        candidates.sort(key=lambda item: item.approx_score, reverse=True)
        out.append(
            CodewordList(
                block_index=int(block_index),
                codeword_index=int(row),
                candidates=tuple(candidates[: max(1, int(config.nibble_candidates_per_codeword))]),
            )
        )
    return tuple(out)


def _score_projected_symbol_values(
    symbol_values: Sequence[int],
    likelihoods: Sequence[SymbolLikelihood],
    trim_fraction: float,
) -> float:
    scores = [
        likelihood.score_symbol(symbol_value)
        for symbol_value, likelihood in zip(symbol_values, likelihoods)
    ]
    return _trimmed_sum(scores, trim_fraction)


def enumerate_block_candidates(
    block_index: int,
    block_likelihoods: Sequence[SymbolLikelihood],
    sf: int,
    cr: int,
    ldro: bool,
    config: TwoStageWeakConfig,
) -> tuple[tuple[CodewordList, ...], tuple[BlockCandidate, ...]]:
    """Stage 1/2 bridge: row-wise nibble lists, then block-level beam search."""

    codeword_lists = enumerate_codeword_lists_for_block(
        block_index=block_index,
        block_likelihoods=block_likelihoods,
        sf=sf,
        cr=cr,
        ldro=ldro,
        config=config,
    )
    beam: list[tuple[tuple[int, ...], tuple[int, ...], float]] = [((), (), 0.0)]
    for codeword_list in codeword_lists:
        next_beam: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []
        for nibbles, codewords, score in beam:
            for candidate in codeword_list.candidates:
                next_beam.append(
                    (
                        nibbles + (candidate.nibble,),
                        codewords + (candidate.codeword,),
                        float(score + candidate.approx_score),
                    )
                )
        next_beam.sort(key=lambda item: item[2], reverse=True)
        beam = next_beam[: max(1, int(config.row_beam_width))]

    block_candidates: list[BlockCandidate] = []
    for nibbles, codewords, approx_score in beam:
        try:
            symbol_values = _payload_codewords_to_symbol_values(
                codewords,
                sf=sf,
                cr=cr,
                ldro=ldro,
            )
        except ValueError:
            continue
        projection_score = _score_projected_symbol_values(
            symbol_values,
            block_likelihoods,
            trim_fraction=config.block_trim_fraction,
        )
        raw_bins = tuple(_symbol_value_to_raw_bin(v, sf=sf, ldro=ldro) for v in symbol_values)
        total_score = float(projection_score)
        block_candidates.append(
            BlockCandidate(
                block_index=int(block_index),
                nibbles=tuple(int(v) for v in nibbles),
                codewords=tuple(int(v) for v in codewords),
                symbol_values=tuple(int(v) for v in symbol_values),
                raw_bins=tuple(int(v) for v in raw_bins),
                approx_score=float(approx_score),
                projection_score=float(projection_score),
                total_score=total_score,
            )
        )
    block_candidates.sort(key=lambda item: item.total_score, reverse=True)
    return codeword_lists, tuple(block_candidates[: max(1, int(config.block_candidate_limit))])


def _beam_blocks(
    block_candidates: Sequence[Sequence[BlockCandidate]],
    global_beam_width: int,
) -> list[_BeamState]:
    beam_width = max(1, int(global_beam_width))
    beam = [
        _BeamState(
            sort_key=0.0,
            nibbles=(),
            codewords=(),
            symbol_values=(),
            raw_bins=(),
            score=0.0,
        )
    ]
    for candidates in block_candidates:
        top_heap: list[tuple[float, int, _BeamState, BlockCandidate]] = []
        counter = 0
        for state in beam:
            for block in candidates:
                score = float(state.score + block.total_score)
                if len(top_heap) < beam_width:
                    heapq.heappush(top_heap, (score, counter, state, block))
                    counter += 1
                    continue
                if score <= top_heap[0][0]:
                    # Candidates are sorted descending within each block list,
                    # so the rest for this state cannot enter the top-N beam.
                    break
                heapq.heapreplace(top_heap, (score, counter, state, block))
                counter += 1
        top_items = sorted(top_heap, key=lambda item: item[0], reverse=True)
        beam = [
            _BeamState(
                sort_key=-float(score),
                nibbles=state.nibbles + block.nibbles,
                codewords=state.codewords + block.codewords,
                symbol_values=state.symbol_values + block.symbol_values,
                raw_bins=state.raw_bins + block.raw_bins,
                score=float(score),
            )
            for score, _counter, state, block in top_items
        ]
        if not beam:
            break
    return beam


def _score_full_payload_projection(
    payload_bytes: bytes,
    likelihoods: Sequence[SymbolLikelihood],
    sf: int,
    cr: int,
    has_crc: bool,
    ldro: bool,
    crc_mode: str,
    trim_fraction: float,
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    _header_symbols, payload_symbols = encode_explicit_frame_symbols(
        payload=payload_bytes,
        sf=sf,
        cr=cr,
        has_crc=has_crc,
        ldro=ldro,
        crc_mode=crc_mode,
    )
    payload_symbols = tuple(int(v) for v in payload_symbols[: len(likelihoods)])
    score_count = min(len(payload_symbols), len(likelihoods))
    score = _score_projected_symbol_values(
        payload_symbols[:score_count],
        likelihoods[:score_count],
        trim_fraction,
    )
    raw_bins = tuple(_symbol_value_to_raw_bin(v, sf=sf, ldro=ldro) for v in payload_symbols)
    return float(score), payload_symbols, raw_bins


def _score_payload_phase_trajectory(
    payload_symbols: Sequence[int],
    trajectory_spectra: Sequence[np.ndarray],
    payload_abs_indices: Sequence[float],
    line: PhaseLine,
    sf: int,
    ldro: bool,
    config: TwoStageWeakConfig,
) -> float:
    """Score whether a projected payload candidate forms a smooth phase track."""

    if float(config.trajectory_score_weight) <= 0.0:
        return 0.0
    if len(payload_symbols) == 0 or len(trajectory_spectra) == 0:
        return 0.0
    if int(getattr(line, "anchor_count", 0)) < 2:
        return 0.0

    limit = min(len(payload_symbols), len(trajectory_spectra), len(payload_abs_indices))
    if limit <= 0:
        return 0.0
    phase_config = PhaseGuidedPayloadConfig()
    score, _meta = _score_payload_symbol_prior_candidate(
        expected_symbols=tuple(int(v) for v in payload_symbols[:limit]),
        score_positions=set(range(limit)),
        payload_spectra=tuple(trajectory_spectra[:limit]),
        payload_dechirped=(),
        payload_abs_indices=tuple(float(v) for v in payload_abs_indices[:limit]),
        line=line,
        sf=int(sf),
        ldro=bool(ldro),
        config=phase_config,
        preamble_profile=None,
        preamble_profile_quality=0.0,
        kappa_line=line,
    )
    return float(score) if math.isfinite(float(score)) else 0.0


def _make_argmax_fallback_candidate(
    likelihoods: Sequence[SymbolLikelihood],
    trajectory_spectra: Sequence[np.ndarray],
    payload_abs_indices: Sequence[float],
    header_tail: Sequence[int],
    phase_line: PhaseLine,
    sf: int,
    cr: int,
    payload_len: int,
    has_crc: bool,
    ldro: bool,
    config: TwoStageWeakConfig,
) -> PayloadBeamCandidate | None:
    """Build the active-evidence hard-argmax payload as an explicit fallback.

    If none of the codec-list candidates satisfies CRC, the weak-packet decoder
    should abstain from destructive repair and report the hard-decision path for
    the currently supplied likelihood.  With center evidence this is the
    traditional FFT argmax path; with phase-gated raw scores it is the
    phase-gated top-1 path.  This fallback is not given access to GT or payload
    templates.
    """

    if not likelihoods:
        return None
    argmax_symbols = tuple(int(item.argmax_symbol_value) for item in likelihoods)
    argmax_raw_bins = tuple(int(item.argmax_bin) for item in likelihoods)
    frame_nibbles = tuple(int(v) for v in header_tail) + tuple(
        payload_symbols_to_nibbles(
            argmax_symbols,
            sf=int(sf),
            cr=int(cr),
            ldro=bool(ldro),
        )
    )
    payload, crc_bytes = nibbles_to_dewhitened_bytes(
        frame_nibbles,
        payload_len=int(payload_len),
        has_crc=bool(has_crc),
    )
    if len(payload) < int(payload_len):
        return None
    observed_crc_valid, crc_computed, crc_received = _verify_payload_crc(
        payload,
        crc_bytes,
        has_crc=has_crc,
        crc_mode=config.crc_mode,
    )
    projection_score, projected_symbols, projected_raw_bins = _score_full_payload_projection(
        payload_bytes=payload,
        likelihoods=likelihoods,
        sf=sf,
        cr=cr,
        has_crc=has_crc,
        ldro=ldro,
        crc_mode=config.crc_mode,
        trim_fraction=config.projection_trim_fraction,
    )
    direct_score = _score_projected_symbol_values(
        argmax_symbols,
        likelihoods,
        trim_fraction=0.0,
    )
    trajectory_score = _score_payload_phase_trajectory(
        payload_symbols=projected_symbols,
        trajectory_spectra=trajectory_spectra,
        payload_abs_indices=payload_abs_indices,
        line=phase_line,
        sf=sf,
        ldro=ldro,
        config=config,
    )
    total_score = float(
        direct_score
        + config.projection_score_weight * projection_score
        + config.trajectory_score_weight * trajectory_score
    )
    if observed_crc_valid:
        total_score += float(config.crc_observed_bonus)
    return PayloadBeamCandidate(
        payload_bytes=bytes(payload),
        crc_bytes_observed=bytes(crc_bytes),
        observed_crc_valid=bool(observed_crc_valid),
        crc_computed=int(crc_computed),
        crc_received=int(crc_received),
        frame_nibbles=tuple(int(v) for v in frame_nibbles),
        projected_payload_symbol_values=projected_symbols,
        projected_raw_bins=projected_raw_bins,
        payload_symbol_values=argmax_symbols,
        raw_bins=argmax_raw_bins,
        block_score=float(direct_score),
        projection_score=float(projection_score),
        trajectory_score=float(trajectory_score),
        total_score=float(total_score),
        beam_rank=-1,
        selection_source="argmax_fallback",
    )


def _candidate_evidence_score(
    candidate: PayloadBeamCandidate,
    config: TwoStageWeakConfig,
) -> float:
    return float(
        candidate.block_score
        + config.projection_score_weight * candidate.projection_score
        + config.trajectory_score_weight * candidate.trajectory_score
    )


def _accept_crc_candidate(
    candidate: PayloadBeamCandidate,
    argmax_fallback: PayloadBeamCandidate | None,
    config: TwoStageWeakConfig,
) -> bool:
    if not candidate.observed_crc_valid:
        return False
    max_rank = int(config.crc_candidate_max_beam_rank)
    if max_rank >= 0 and int(candidate.beam_rank) > max_rank:
        return False
    if argmax_fallback is None:
        return True
    margin = float(config.crc_candidate_min_evidence_margin)
    return (
        _candidate_evidence_score(candidate, config)
        >= _candidate_evidence_score(argmax_fallback, config) + margin
    )


def decode_two_stage_weak_payload(
    payload_spectra: Sequence[np.ndarray],
    header_symbol_values: Sequence[int],
    phase_line: PhaseLine | None,
    payload_symbol_start_abs_index: float,
    sf: int,
    cr: int,
    payload_len: int,
    has_crc: bool,
    ldro: bool,
    config: TwoStageWeakConfig | None = None,
    raw_score_overrides: Sequence[np.ndarray] | None = None,
    trajectory_spectra: Sequence[np.ndarray] | None = None,
) -> TwoStageWeakResult:
    """Decode one packet payload from full FFT evidence."""

    cfg = config or TwoStageWeakConfig()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if not payload_spectra:
        return TwoStageWeakResult(
            success=False,
            selected=None,
            candidates=(),
            argmax_symbol_values=(),
            argmax_raw_bins=(),
            likelihoods=(),
            codeword_lists=(),
            block_candidates=(),
            timings_ms={},
            metrics={},
            error="no payload spectra",
        )

    line = phase_line or PhaseLine()
    trajectory_source_spectra = (
        list(trajectory_spectra)
        if trajectory_spectra is not None
        else list(payload_spectra)
    )
    payload_abs_indices = [
        float(payload_symbol_start_abs_index) + float(idx)
        for idx in range(len(payload_spectra))
    ]
    likelihoods: list[SymbolLikelihood] = []
    overrides = list(raw_score_overrides) if raw_score_overrides is not None else None
    for idx, spectrum in enumerate(payload_spectra):
        predicted = line.predict(float(payload_symbol_start_abs_index) + float(idx))
        score_override = None
        if overrides is not None and idx < len(overrides):
            score_override = np.asarray(overrides[idx], dtype=np.float64)
        likelihoods.append(
            build_symbol_likelihood(
                spectrum=spectrum,
                symbol_index=idx,
                sf=sf,
                ldro=ldro,
                predicted_phase_rad=predicted,
                config=cfg,
                raw_score_override=score_override,
            )
        )
    timings["likelihood_ms"] = (time.perf_counter() - t0) * 1000.0

    expected_payload_symbols = _expected_payload_symbol_count(
        payload_len=payload_len,
        sf=sf,
        cr=cr,
        has_crc=has_crc,
        ldro=ldro,
        crc_mode=cfg.crc_mode,
    )
    if expected_payload_symbols > 0:
        likelihoods = likelihoods[: min(len(likelihoods), expected_payload_symbols)]
        trajectory_source_spectra = trajectory_source_spectra[: len(likelihoods)]
        payload_abs_indices = payload_abs_indices[: len(likelihoods)]

    cw_len = int(cr) + 4
    full_symbol_count = (len(likelihoods) // cw_len) * cw_len
    if full_symbol_count <= 0:
        return TwoStageWeakResult(
            success=False,
            selected=None,
            candidates=(),
            argmax_symbol_values=tuple(item.argmax_symbol_value for item in likelihoods),
            argmax_raw_bins=tuple(item.argmax_bin for item in likelihoods),
            likelihoods=tuple(likelihoods),
            codeword_lists=(),
            block_candidates=(),
            timings_ms=timings,
            metrics={},
            error="not enough symbols for one interleaver block",
        )
    likelihoods = likelihoods[:full_symbol_count]
    trajectory_source_spectra = trajectory_source_spectra[:full_symbol_count]
    payload_abs_indices = payload_abs_indices[:full_symbol_count]

    t1 = time.perf_counter()
    all_codeword_lists: list[CodewordList] = []
    block_candidates: list[tuple[BlockCandidate, ...]] = []
    for block_index, start in enumerate(range(0, full_symbol_count, cw_len)):
        block_likelihoods = likelihoods[start:start + cw_len]
        codeword_lists, candidates = enumerate_block_candidates(
            block_index=block_index,
            block_likelihoods=block_likelihoods,
            sf=sf,
            cr=cr,
            ldro=ldro,
            config=cfg,
        )
        all_codeword_lists.extend(codeword_lists)
        if candidates:
            block_candidates.append(candidates)
    timings["block_search_ms"] = (time.perf_counter() - t1) * 1000.0

    if not block_candidates:
        return TwoStageWeakResult(
            success=False,
            selected=None,
            candidates=(),
            argmax_symbol_values=tuple(item.argmax_symbol_value for item in likelihoods),
            argmax_raw_bins=tuple(item.argmax_bin for item in likelihoods),
            likelihoods=tuple(likelihoods),
            codeword_lists=tuple(all_codeword_lists),
            block_candidates=(),
            timings_ms=timings,
            metrics={},
            error="no block candidates",
        )

    t2 = time.perf_counter()
    block_beam = _beam_blocks(block_candidates, cfg.global_beam_width)
    timings["global_beam_ms"] = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    try:
        header_tail = explicit_header_tail_nibbles(header_symbol_values, sf=sf)
    except Exception:
        header_tail = ()

    payload_candidates: list[PayloadBeamCandidate] = []
    for rank, state in enumerate(block_beam[: max(1, int(cfg.final_candidate_limit))]):
        frame_nibbles = tuple(header_tail) + tuple(state.nibbles)
        payload, crc_bytes = nibbles_to_dewhitened_bytes(
            frame_nibbles,
            payload_len=int(payload_len),
            has_crc=bool(has_crc),
        )
        if len(payload) < int(payload_len):
            continue
        observed_crc_valid, crc_computed, crc_received = _verify_payload_crc(
            payload,
            crc_bytes,
            has_crc=has_crc,
            crc_mode=cfg.crc_mode,
        )
        projection_score, projected_symbols, projected_raw_bins = _score_full_payload_projection(
            payload_bytes=payload,
            likelihoods=likelihoods,
            sf=sf,
            cr=cr,
            has_crc=has_crc,
            ldro=ldro,
            crc_mode=cfg.crc_mode,
            trim_fraction=cfg.projection_trim_fraction,
        )
        trajectory_score = _score_payload_phase_trajectory(
            payload_symbols=projected_symbols,
            trajectory_spectra=trajectory_source_spectra,
            payload_abs_indices=payload_abs_indices,
            line=line,
            sf=sf,
            ldro=ldro,
            config=cfg,
        )
        total_score = float(
            state.score
            + cfg.projection_score_weight * projection_score
            + cfg.trajectory_score_weight * trajectory_score
        )
        if observed_crc_valid:
            total_score += float(cfg.crc_observed_bonus)
        payload_candidates.append(
            PayloadBeamCandidate(
                payload_bytes=bytes(payload),
                crc_bytes_observed=bytes(crc_bytes),
                observed_crc_valid=bool(observed_crc_valid),
                crc_computed=int(crc_computed),
                crc_received=int(crc_received),
                frame_nibbles=tuple(int(v) for v in frame_nibbles),
                projected_payload_symbol_values=projected_symbols,
                projected_raw_bins=projected_raw_bins,
                payload_symbol_values=tuple(int(v) for v in state.symbol_values),
                raw_bins=tuple(int(v) for v in state.raw_bins),
                block_score=float(state.score),
                projection_score=float(projection_score),
                trajectory_score=float(trajectory_score),
                total_score=float(total_score),
                beam_rank=int(rank),
                selection_source="beam",
            )
        )
    payload_candidates.sort(key=lambda item: item.total_score, reverse=True)
    beam_payload_candidates = tuple(payload_candidates)
    argmax_fallback = _make_argmax_fallback_candidate(
        likelihoods=likelihoods,
        trajectory_spectra=trajectory_source_spectra,
        payload_abs_indices=payload_abs_indices,
        header_tail=header_tail,
        phase_line=line,
        sf=sf,
        cr=cr,
        payload_len=payload_len,
        has_crc=has_crc,
        ldro=ldro,
        config=cfg,
    )
    if argmax_fallback is not None:
        payload_candidates.append(argmax_fallback)
    timings["final_projection_ms"] = (time.perf_counter() - t3) * 1000.0
    timings["total_ms"] = (time.perf_counter() - t0) * 1000.0

    crc_valid_beam = [
        item
        for item in beam_payload_candidates
        if _accept_crc_candidate(item, argmax_fallback, cfg)
    ]
    if crc_valid_beam:
        selected = max(crc_valid_beam, key=lambda item: item.total_score)
    elif bool(cfg.argmax_fallback_on_crc_failure) and argmax_fallback is not None:
        selected = argmax_fallback
    else:
        selected = payload_candidates[0] if payload_candidates else None
    if selected is not None:
        payload_candidates.sort(
            key=lambda item: (item is selected, item.total_score),
            reverse=True,
        )
    metrics: dict[str, float | int] = {
        "payload_symbol_count": int(len(likelihoods)),
        "expected_payload_symbol_count": int(expected_payload_symbols),
        "block_count": int(len(block_candidates)),
        "codeword_list_count": int(len(all_codeword_lists)),
        "candidate_payload_count": int(len(beam_payload_candidates)),
        "candidate_payload_count_with_fallback": int(len(payload_candidates)),
        "observed_crc_valid_count": int(sum(1 for item in beam_payload_candidates if item.observed_crc_valid)),
        "accepted_crc_valid_count": int(len(crc_valid_beam)),
        "selected_source_is_argmax_fallback": int(
            selected is not None and selected.selection_source == "argmax_fallback"
        ),
        "avg_block_candidates": float(np.mean([len(items) for items in block_candidates])) if block_candidates else 0.0,
    }

    return TwoStageWeakResult(
        success=selected is not None,
        selected=selected,
        candidates=tuple(payload_candidates),
        argmax_symbol_values=tuple(item.argmax_symbol_value for item in likelihoods),
        argmax_raw_bins=tuple(item.argmax_bin for item in likelihoods),
        likelihoods=tuple(likelihoods),
        codeword_lists=tuple(all_codeword_lists),
        block_candidates=tuple(block_candidates),
        timings_ms=timings,
        metrics=metrics,
        error="" if selected is not None else "no final payload candidate",
    )


def summarize_with_ground_truth(
    result: TwoStageWeakResult,
    gt_raw_bins: Sequence[int] | None = None,
    sf: int = 10,
    ldro: bool = False,
    top_k: int = 32,
) -> dict[str, float | int]:
    """Compute evaluation-only metrics.  Ground truth is never used by decoding."""

    if gt_raw_bins is None:
        return {}
    gt = [int(v) for v in gt_raw_bins]
    n = min(len(gt), len(result.likelihoods))
    if n <= 0:
        return {}

    argmax_errors = 0
    selected_errors = 0
    argmax_symbol_errors = 0
    selected_symbol_errors = 0
    topk_hits = 0
    rank_sum = 0.0
    rank_count = 0
    selected_raw = tuple(result.selected.raw_bins) if result.selected is not None else ()
    selected_symbols = tuple(result.selected.payload_symbol_values) if result.selected is not None else ()

    for idx in range(n):
        likelihood = result.likelihoods[idx]
        gt_bin = gt[idx]
        gt_symbol = bin_to_grlora_symbol(gt_bin, sf=sf, is_header=False, ldro=ldro)
        argmax_errors += int(likelihood.argmax_bin != gt_bin)
        argmax_symbol_errors += int(likelihood.argmax_symbol_value != gt_symbol)
        if idx < len(selected_raw):
            selected_errors += int(int(selected_raw[idx]) != gt_bin)
        else:
            selected_errors += 1
        if idx < len(selected_symbols):
            selected_symbol_errors += int(int(selected_symbols[idx]) != gt_symbol)
        else:
            selected_symbol_errors += 1
        rank = likelihood.raw_rank(gt_bin)
        if rank > 0:
            rank_sum += float(rank)
            rank_count += 1
            topk_hits += int(rank <= int(top_k))

    return {
        "gt_compared_symbols": int(n),
        "argmax_raw_ser": float(argmax_errors / n),
        "two_stage_raw_ser": float(selected_errors / n),
        "argmax_symbol_ser": float(argmax_symbol_errors / n),
        "two_stage_symbol_ser": float(selected_symbol_errors / n),
        "topk_recall": float(topk_hits / n),
        "mean_gt_rank": float(rank_sum / rank_count) if rank_count else 0.0,
        "ser_delta_raw_argmax_minus_two_stage": float((argmax_errors - selected_errors) / n),
        "ser_delta_symbol_argmax_minus_two_stage": float((argmax_symbol_errors - selected_symbol_errors) / n),
    }


__all__ = [
    "BlockCandidate",
    "CodewordList",
    "NibbleCandidate",
    "PayloadBeamCandidate",
    "SymbolLikelihood",
    "TwoStageWeakConfig",
    "TwoStageWeakResult",
    "build_symbol_likelihood",
    "decode_two_stage_weak_payload",
    "enumerate_block_candidates",
    "enumerate_codeword_lists_for_block",
    "summarize_with_ground_truth",
]

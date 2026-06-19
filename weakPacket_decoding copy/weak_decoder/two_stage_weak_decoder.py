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
from itertools import combinations, product
import math
import time
from typing import Iterable, Sequence

import numpy as np

from .chirp import bin_to_grlora_symbol, positive_mod
from .header_first_demod import (
    bits_to_int,
    deinterleave_hard,
    gray_demapping,
    hamming_decode_hard,
    int_to_bits_msb,
)
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
    score_projected_payload_symbols,
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
    block_symbol_seed_top_m: int = 0
    block_symbol_seed_deep_top_l: int = 0
    block_symbol_seed_max_deep_positions: int = 1
    block_symbol_seed_quota: int = 0
    block_symbol_seed_max_combinations: int = 50000
    global_beam_width: int = 64
    global_rank_diverse_top_r: int = 0
    global_rank_diverse_beam_width: int = 0
    global_rank_cost_max: float = 0.0
    global_rank_cost_state_limit: int = 0
    final_candidate_limit: int = 128
    block_trim_fraction: float = 0.20
    projection_trim_fraction: float = 0.10
    projection_score_weight: float = 1.00
    trajectory_score_weight: float = 0.00
    crc_observed_bonus: float = 4.0
    crc_candidate_min_evidence_margin: float = 0.0
    crc_candidate_max_beam_rank: int = 64
    crc_candidate_high_rank_max_beam_rank: int = 256
    crc_candidate_high_rank_min_evidence_margin: float = 4.0
    crc_selection_policy: str = "best_score"  # "best_score", "earliest_rank", "score_unless_low_margin"
    crc_non_earliest_min_evidence_margin: float = 0.50
    crc_failure_selection_policy: str = "argmax_fallback"  # "argmax_fallback", "best_score", "small_change_best_score"
    crc_failure_max_symbol_changes: int = 2
    crc_failure_min_evidence_margin: float = -0.50
    crc_state_search_block_top_r: int = 0
    crc_state_search_state_limit: int = 20000
    crc_state_search_keep_per_key: int = 1
    crc_state_search_max_candidates: int = 256
    crc_state_search_min_evidence_margin: float = 0.0
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
    source: str = "row_beam"


@dataclass(order=True)
class _BeamState:
    sort_key: float
    nibbles: tuple[int, ...] = field(compare=False)
    codewords: tuple[int, ...] = field(compare=False)
    symbol_values: tuple[int, ...] = field(compare=False)
    raw_bins: tuple[int, ...] = field(compare=False)
    score: float = field(compare=False)
    source: str = field(default="score_beam", compare=False)


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


def _make_block_candidate(
    block_index: int,
    nibbles: Sequence[int],
    codewords: Sequence[int],
    approx_score: float,
    block_likelihoods: Sequence[SymbolLikelihood],
    sf: int,
    cr: int,
    ldro: bool,
    config: TwoStageWeakConfig,
    source: str = "row_beam",
) -> BlockCandidate | None:
    try:
        symbol_values = _payload_codewords_to_symbol_values(
            codewords,
            sf=sf,
            cr=cr,
            ldro=ldro,
        )
    except ValueError:
        return None
    projection_score = _score_projected_symbol_values(
        symbol_values,
        block_likelihoods,
        trim_fraction=config.block_trim_fraction,
    )
    raw_bins = tuple(_symbol_value_to_raw_bin(v, sf=sf, ldro=ldro) for v in symbol_values)
    return BlockCandidate(
        block_index=int(block_index),
        nibbles=tuple(int(v) for v in nibbles),
        codewords=tuple(int(v) for v in codewords),
        symbol_values=tuple(int(v) for v in symbol_values),
        raw_bins=tuple(int(v) for v in raw_bins),
        approx_score=float(approx_score),
        projection_score=float(projection_score),
        total_score=float(projection_score),
        source=str(source),
    )


def _symbol_seed_block_candidates(
    block_index: int,
    block_likelihoods: Sequence[SymbolLikelihood],
    sf: int,
    cr: int,
    ldro: bool,
    config: TwoStageWeakConfig,
) -> list[BlockCandidate]:
    """Decode Top-M symbol tuples into extra legal block candidates.

    The row-wise Hamming beam scores each codeword independently.  Near the
    noise limit, the correct row values can all be present but their joint
    combination can still be trimmed before projection rescoring.  This seed
    path starts from the symbol evidence itself, decodes each Top-M tuple
    through the normal hard deinterleaver/Hamming path, then projects the
    resulting legal codewords back to symbols before merging with the normal
    block candidates.
    """

    top_m = int(getattr(config, "block_symbol_seed_top_m", 0))
    if top_m <= 1 or not block_likelihoods:
        return []
    cw_len = int(cr) + 4
    if len(block_likelihoods) != cw_len:
        return []
    max_combinations = max(1, int(getattr(config, "block_symbol_seed_max_combinations", 50000)))
    effective_top_m = max(1, int(top_m))
    while effective_top_m > 1 and int(effective_top_m) ** int(cw_len) > max_combinations:
        effective_top_m -= 1
    if effective_top_m <= 1:
        return []

    value_lists: list[tuple[int, ...]] = []
    deep_value_lists: list[tuple[int, ...]] = []
    score_by_value: list[dict[int, float]] = []
    for likelihood in block_likelihoods:
        finite = np.isfinite(likelihood.score_by_value)
        if not np.any(finite):
            return []
        order = np.argsort(likelihood.score_by_value)[::-1]
        values = tuple(int(v) for v in order[: max(1, min(effective_top_m, order.size))] if finite[int(v)])
        if not values:
            return []
        value_lists.append(values)
        deep_top_l = max(effective_top_m, int(getattr(config, "block_symbol_seed_deep_top_l", 0)))
        max_deep_positions = max(0, int(getattr(config, "block_symbol_seed_max_deep_positions", 1)))
        while (
            deep_top_l > effective_top_m
            and max_deep_positions > 0
            and (
                int(effective_top_m) ** int(cw_len)
                + int(cw_len) * int(deep_top_l) * (int(effective_top_m) ** max(0, int(cw_len) - 1))
            )
            > max_combinations
        ):
            deep_top_l -= 1
        deep_values = tuple(int(v) for v in order[: max(1, min(deep_top_l, order.size))] if finite[int(v)])
        deep_value_lists.append(deep_values if deep_values else values)
        score_by_value.append({int(v): float(likelihood.score_symbol(int(v))) for v in deep_values})

    out: list[BlockCandidate] = []
    sf_app = int(sf) - 2 if bool(ldro) else int(sf)
    tuple_seen: set[tuple[int, ...]] = set()

    def _symbol_tuple_iter() -> Iterable[tuple[int, ...]]:
        for values in product(*value_lists):
            yield tuple(int(v) for v in values)

        deep_top_l = max(effective_top_m, int(getattr(config, "block_symbol_seed_deep_top_l", 0)))
        max_deep_positions = max(0, int(getattr(config, "block_symbol_seed_max_deep_positions", 1)))
        if deep_top_l <= effective_top_m or max_deep_positions <= 0:
            return
        for depth in range(1, min(cw_len, max_deep_positions) + 1):
            for deep_positions in combinations(range(cw_len), depth):
                lists = list(value_lists)
                for deep_pos in deep_positions:
                    lists[deep_pos] = tuple(int(v) for v in deep_value_lists[deep_pos][:deep_top_l])
                for values in product(*lists):
                    yield tuple(int(v) for v in values)

    for symbol_values in _symbol_tuple_iter():
        if symbol_values in tuple_seen:
            continue
        tuple_seen.add(symbol_values)
        try:
            gray_values = gray_demapping(symbol_values)
            codewords = tuple(
                int(v)
                for v in deinterleave_hard(
                    gray_values,
                    sf=int(sf),
                    is_header=False,
                    cr=int(cr),
                    ldro=bool(ldro),
                )
            )
            if len(codewords) != sf_app:
                continue
            nibbles = tuple(
                int(v) & 0xF
                for v in hamming_decode_hard(
                    codewords,
                    is_header=False,
                    cr=int(cr),
                )
            )
            if len(nibbles) != sf_app:
                continue
        except Exception:
            continue
        approx_score = sum(
            score_by_value[idx].get(int(symbol_value), float("-inf"))
            for idx, symbol_value in enumerate(symbol_values)
        )
        candidate = _make_block_candidate(
            block_index=block_index,
            nibbles=nibbles,
            codewords=codewords,
            approx_score=float(approx_score),
            block_likelihoods=block_likelihoods,
            sf=sf,
            cr=cr,
            ldro=ldro,
            config=config,
            source="symbol_seed",
        )
        if candidate is not None:
            out.append(candidate)
    return out


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

    row_candidates: list[BlockCandidate] = []
    for nibbles, codewords, approx_score in beam:
        candidate = _make_block_candidate(
            block_index=block_index,
            nibbles=nibbles,
            codewords=codewords,
            approx_score=approx_score,
            block_likelihoods=block_likelihoods,
            sf=sf,
            cr=cr,
            ldro=ldro,
            config=config,
        )
        if candidate is not None:
            row_candidates.append(candidate)

    seed_candidates: list[BlockCandidate] = []
    if int(getattr(config, "block_symbol_seed_top_m", 0)) > 1:
        seed_candidates.extend(
            _symbol_seed_block_candidates(
                block_index=block_index,
                block_likelihoods=block_likelihoods,
                sf=sf,
                cr=cr,
                ldro=ldro,
                config=config,
            )
        )

    def _dedup(candidates: Sequence[BlockCandidate]) -> list[BlockCandidate]:
        dedup: dict[tuple[int, ...], BlockCandidate] = {}
        for candidate in candidates:
            key = tuple(candidate.nibbles)
            previous = dedup.get(key)
            if previous is None or float(candidate.total_score) > float(previous.total_score):
                dedup[key] = candidate
        out = list(dedup.values())
        out.sort(key=lambda item: item.total_score, reverse=True)
        return out

    row_candidates = _dedup(row_candidates)
    seed_candidates = _dedup(seed_candidates)
    limit = max(1, int(config.block_candidate_limit))
    seed_quota = max(0, min(limit, int(getattr(config, "block_symbol_seed_quota", 0))))
    if seed_quota <= 0 or not seed_candidates:
        block_candidates = row_candidates[:limit]
    else:
        selected: dict[tuple[int, ...], BlockCandidate] = {}
        for candidate in seed_candidates[:seed_quota]:
            key = tuple(candidate.nibbles)
            selected[key] = candidate
        for candidate in row_candidates:
            key = tuple(candidate.nibbles)
            previous = selected.get(key)
            if previous is not None:
                if float(candidate.total_score) > float(previous.total_score):
                    selected[key] = candidate
                continue
            if len(selected) >= limit:
                continue
            selected[key] = candidate
        block_candidates = list(selected.values())
    block_candidates.sort(key=lambda item: item.total_score, reverse=True)
    return codeword_lists, tuple(block_candidates[:limit])


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


def _rank_diverse_block_states(
    block_candidates: Sequence[Sequence[BlockCandidate]],
    top_r: int,
    beam_width: int,
) -> list[_BeamState]:
    """Add low-rank-cost block combinations that a pure score beam can miss."""

    rank_limit = max(0, int(top_r))
    if rank_limit <= 0 or not block_candidates:
        return []
    width = max(1, int(beam_width))
    usable = [tuple(candidates[:rank_limit]) for candidates in block_candidates if candidates]
    if len(usable) != len(block_candidates):
        return []

    seen: set[tuple[int, ...]] = set()
    beam: list[tuple[float, float, _BeamState]] = [
        (
            0.0,
            0.0,
            _BeamState(
                sort_key=0.0,
                nibbles=(),
                codewords=(),
                symbol_values=(),
                raw_bins=(),
                score=0.0,
                source="rank_diverse",
            ),
        )
    ]
    for candidates in usable:
        next_beam: list[tuple[float, float, _BeamState]] = []
        for cost, _neg_score, state in beam:
            for rank, block in enumerate(candidates):
                score = float(state.score + block.total_score)
                rank_cost = float(cost + math.log2(float(rank + 1)))
                next_beam.append(
                    (
                        rank_cost,
                        -score,
                        _BeamState(
                            sort_key=float(rank_cost),
                            nibbles=state.nibbles + block.nibbles,
                            codewords=state.codewords + block.codewords,
                            symbol_values=state.symbol_values + block.symbol_values,
                            raw_bins=state.raw_bins + block.raw_bins,
                            score=score,
                            source="rank_diverse",
                        ),
                    )
                )
        next_beam.sort(key=lambda item: (item[0], item[1]))
        beam = next_beam[:width]
        if not beam:
            break

    states: list[_BeamState] = []
    for _cost, _neg_score, state in beam:
        nibbles: list[int] = []
        codewords: list[int] = []
        symbol_values: list[int] = []
        raw_bins: list[int] = []
        nibbles.extend(state.nibbles)
        codewords.extend(state.codewords)
        symbol_values.extend(state.symbol_values)
        raw_bins.extend(state.raw_bins)
        key = tuple(int(v) for v in state.nibbles)
        if key in seen:
            continue
        seen.add(key)
        states.append(
            _BeamState(
                sort_key=state.sort_key,
                nibbles=tuple(nibbles),
                codewords=tuple(codewords),
                symbol_values=tuple(symbol_values),
                raw_bins=tuple(raw_bins),
                score=float(state.score),
                source="rank_diverse",
            )
        )
    return states


def _rank_cost_block_states(
    block_candidates: Sequence[Sequence[BlockCandidate]],
    top_r: int,
    max_cost: float,
    state_limit: int,
) -> list[_BeamState]:
    """Enumerate cross-block combinations under a rank-cost budget."""

    rank_limit = max(0, int(top_r))
    cost_limit = float(max_cost)
    limit = max(0, int(state_limit))
    if rank_limit <= 0 or cost_limit <= 0.0 or limit <= 0 or not block_candidates:
        return []
    usable = [tuple(candidates[:rank_limit]) for candidates in block_candidates if candidates]
    if len(usable) != len(block_candidates):
        return []

    choices: list[list[tuple[float, BlockCandidate]]] = []
    for candidates in usable:
        block_choices: list[tuple[float, BlockCandidate]] = []
        for rank, block in enumerate(candidates):
            rank_cost = math.log2(float(rank + 1))
            if rank_cost <= cost_limit:
                block_choices.append((rank_cost, block))
        if not block_choices:
            return []
        block_choices.sort(key=lambda item: (item[0], -float(item[1].total_score)))
        choices.append(block_choices)

    states: list[_BeamState] = []
    emitted_nibbles: set[tuple[int, ...]] = set()
    visited_patterns: set[tuple[int, ...]] = set()
    start_pattern = tuple(0 for _ in choices)
    heap: list[tuple[float, int, tuple[int, ...]]] = [(0.0, 0, start_pattern)]
    visited_patterns.add(start_pattern)
    counter = 1

    def _pattern_cost(pattern: Sequence[int]) -> float:
        return float(sum(choices[idx][rank][0] for idx, rank in enumerate(pattern)))

    while heap and len(states) < limit:
        cost, _counter, pattern = heapq.heappop(heap)
        if float(cost) > cost_limit:
            break

        nibbles: list[int] = []
        codewords: list[int] = []
        symbol_values: list[int] = []
        raw_bins: list[int] = []
        score = 0.0
        for block_index, rank in enumerate(pattern):
            block = choices[block_index][rank][1]
            nibbles.extend(block.nibbles)
            codewords.extend(block.codewords)
            symbol_values.extend(block.symbol_values)
            raw_bins.extend(block.raw_bins)
            score += float(block.total_score)
        key = tuple(int(v) for v in nibbles)
        if key not in emitted_nibbles:
            emitted_nibbles.add(key)
            states.append(
                _BeamState(
                    sort_key=float(cost),
                    nibbles=tuple(nibbles),
                    codewords=tuple(codewords),
                    symbol_values=tuple(symbol_values),
                    raw_bins=tuple(raw_bins),
                    score=float(score),
                    source="rank_cost",
                )
            )

        for block_index, rank in enumerate(pattern):
            next_rank = int(rank) + 1
            if next_rank >= len(choices[block_index]):
                continue
            next_pattern_list = list(pattern)
            next_pattern_list[block_index] = next_rank
            next_pattern = tuple(next_pattern_list)
            if next_pattern in visited_patterns:
                continue
            next_cost = _pattern_cost(next_pattern)
            if next_cost > cost_limit:
                continue
            visited_patterns.add(next_pattern)
            heapq.heappush(heap, (float(next_cost), counter, next_pattern))
            counter += 1
    return states


def _crc16_update_byte(crc: int, byte: int) -> int:
    """Incremental CRC-16(poly=0x1021, init=0) update for one byte."""

    out = int(crc) & 0xFFFF
    new_byte = int(byte) & 0xFF
    for _ in range(8):
        if (((out & 0x8000) >> 8) ^ (new_byte & 0x80)):
            out = ((out << 1) ^ 0x1021) & 0xFFFF
        else:
            out = (out << 1) & 0xFFFF
        new_byte = (new_byte << 1) & 0xFF
    return int(out)


def _crc_state_search_states(
    block_candidates: Sequence[Sequence[BlockCandidate]],
    header_tail: Sequence[int],
    payload_len: int,
    has_crc: bool,
    crc_mode: str,
    config: TwoStageWeakConfig,
) -> list[_BeamState]:
    """Search block combinations while pruning equivalent byte/CRC states.

    This is an experimental PHY-only backend.  It treats decoded nibbles as a
    stream, folds complete payload bytes through dewhitening and CRC, and keeps
    only the strongest paths for identical byte-boundary states.  It avoids any
    payload template or ground-truth information.
    """

    top_r = max(0, int(getattr(config, "crc_state_search_block_top_r", 0)))
    if top_r <= 0 or not block_candidates:
        return []
    state_limit = max(1, int(getattr(config, "crc_state_search_state_limit", 20000)))
    keep_per_key = max(1, int(getattr(config, "crc_state_search_keep_per_key", 1)))
    max_candidates = max(1, int(getattr(config, "crc_state_search_max_candidates", 256)))
    has_crc_bool = bool(has_crc)
    payload_len_i = int(payload_len)
    total_data_bytes = payload_len_i + (2 if has_crc_bool else 0)
    total_data_nibbles = total_data_bytes * 2
    header = tuple(int(v) & 0xF for v in header_tail)

    @dataclass(frozen=True)
    class _CrcState:
        nibbles: tuple[int, ...]
        codewords: tuple[int, ...]
        symbol_values: tuple[int, ...]
        raw_bins: tuple[int, ...]
        score: float
        byte_index: int
        pending_low: int
        crc: int
        prev_payload_tail: tuple[int, ...]

    def _advance_stream(
        state: _CrcState,
        stream_nibbles: Sequence[int],
        absolute_start: int,
    ) -> tuple[int, int, int, tuple[int, ...]]:
        byte_index = int(state.byte_index)
        pending_low = int(state.pending_low)
        crc_value = int(state.crc)
        prev_tail = tuple(int(v) & 0xFF for v in state.prev_payload_tail)

        for rel_offset, value in enumerate(stream_nibbles):
            offset = int(absolute_start) + int(rel_offset)
            nibble = int(value) & 0xF
            if offset % 2 == 0:
                pending_low = nibble
                continue

            raw_byte = ((nibble & 0xF) << 4) | (pending_low & 0xF)
            if byte_index < payload_len_i:
                whiten = int(WHITENING_SEQ[byte_index])
                payload_byte = raw_byte ^ whiten
                if str(crc_mode).lower() == "sx1276":
                    crc_value = _crc16_update_byte(crc_value, payload_byte)
                else:
                    if byte_index >= 2:
                        crc_value = _crc16_update_byte(crc_value, prev_tail[0])
                    prev_tail = (prev_tail + (payload_byte,))[-2:]
            byte_index += 1
        return byte_index, pending_low, crc_value, prev_tail

    init_state = _CrcState(
        nibbles=(),
        codewords=(),
        symbol_values=(),
        raw_bins=(),
        score=0.0,
        byte_index=0,
        pending_low=-1,
        crc=0,
        prev_payload_tail=(),
    )
    init_byte_index, init_pending_low, init_crc, init_prev_tail = _advance_stream(
        init_state,
        header,
        absolute_start=0,
    )

    def _advance(
        state: _CrcState,
        block: BlockCandidate,
    ) -> _CrcState | None:
        block_nibbles = tuple(int(v) & 0xF for v in block.nibbles)
        nibbles = state.nibbles + block_nibbles
        frame_nibbles = header + nibbles
        if len(frame_nibbles) > total_data_nibbles:
            nibbles = nibbles[: max(0, total_data_nibbles - len(header))]
            block_nibbles = block_nibbles[: max(0, total_data_nibbles - len(header) - len(state.nibbles))]
            frame_nibbles = header + nibbles

        byte_index, pending_low, crc_value, prev_tail = _advance_stream(
            state,
            block_nibbles,
            absolute_start=len(header) + len(state.nibbles),
        )

        if len(frame_nibbles) == total_data_nibbles and byte_index >= total_data_bytes:
            payload, crc_bytes = nibbles_to_dewhitened_bytes(
                frame_nibbles,
                payload_len=payload_len_i,
                has_crc=has_crc_bool,
            )
            if len(payload) < payload_len_i:
                return None
            crc_ok, _computed, _received = _verify_payload_crc(
                payload,
                crc_bytes,
                has_crc=has_crc_bool,
                crc_mode=str(crc_mode),
            )
            if not crc_ok:
                return None

        return _CrcState(
            nibbles=tuple(int(v) for v in nibbles),
            codewords=state.codewords + tuple(int(v) for v in block.codewords),
            symbol_values=state.symbol_values + tuple(int(v) for v in block.symbol_values),
            raw_bins=state.raw_bins + tuple(int(v) for v in block.raw_bins),
            score=float(state.score + block.total_score),
            byte_index=byte_index,
            pending_low=pending_low,
            crc=crc_value,
            prev_payload_tail=prev_tail,
        )

    def _key(state: _CrcState) -> tuple[int, int, int, tuple[int, ...], int]:
        # Only merge at byte boundaries.  Mid-byte states keep pending_low in
        # the key to avoid mixing incompatible nibble streams.
        parity = (len(header) + len(state.nibbles)) & 1
        return (
            int(state.byte_index),
            int(parity),
            int(state.pending_low) if parity else -1,
            tuple(int(v) for v in state.prev_payload_tail),
            int(state.crc),
        )

    states = [
        _CrcState(
            nibbles=(),
            codewords=(),
            symbol_values=(),
            raw_bins=(),
            score=0.0,
            byte_index=int(init_byte_index),
            pending_low=int(init_pending_low),
            crc=int(init_crc),
            prev_payload_tail=tuple(init_prev_tail),
        )
    ]
    for candidates in block_candidates:
        choices = tuple(candidates[:top_r])
        if not choices:
            return []
        buckets: dict[tuple[int, int, int, tuple[int, ...], int], list[_CrcState]] = {}
        for state in states:
            for block in choices:
                next_state = _advance(state, block)
                if next_state is None:
                    continue
                buckets.setdefault(_key(next_state), []).append(next_state)
        merged: list[_CrcState] = []
        for bucket in buckets.values():
            bucket.sort(key=lambda item: item.score, reverse=True)
            merged.extend(bucket[:keep_per_key])
        merged.sort(key=lambda item: item.score, reverse=True)
        states = merged[:state_limit]
        if not states:
            break

    out: list[_BeamState] = []
    seen: set[tuple[int, ...]] = set()
    for state in sorted(states, key=lambda item: item.score, reverse=True):
        frame_nibbles = header + state.nibbles
        payload, crc_bytes = nibbles_to_dewhitened_bytes(
            frame_nibbles,
            payload_len=payload_len_i,
            has_crc=has_crc_bool,
        )
        if len(payload) < payload_len_i:
            continue
        crc_ok, _computed, _received = _verify_payload_crc(
            payload,
            crc_bytes,
            has_crc=has_crc_bool,
            crc_mode=str(crc_mode),
        )
        if not crc_ok:
            continue
        key = tuple(int(v) for v in state.nibbles)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            _BeamState(
                sort_key=-float(state.score),
                nibbles=tuple(int(v) for v in state.nibbles),
                codewords=tuple(int(v) for v in state.codewords),
                symbol_values=tuple(int(v) for v in state.symbol_values),
                raw_bins=tuple(int(v) for v in state.raw_bins),
                score=float(state.score),
                source="crc_state",
            )
        )
        if len(out) >= max_candidates:
            break
    return out


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
    score, _meta = score_projected_payload_symbols(
        projected_symbols=tuple(int(v) for v in payload_symbols[:limit]),
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


def _raw_bin_change_count(
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    return sum(int(int(a) != int(b)) for a, b in zip(left, right))


def _accept_crc_candidate(
    candidate: PayloadBeamCandidate,
    argmax_fallback: PayloadBeamCandidate | None,
    config: TwoStageWeakConfig,
) -> bool:
    if not candidate.observed_crc_valid:
        return False
    if argmax_fallback is None:
        evidence_margin = float("inf")
    else:
        evidence_margin = (
            _candidate_evidence_score(candidate, config)
            - _candidate_evidence_score(argmax_fallback, config)
        )
    if argmax_fallback is None:
        min_margin_ok = True
    else:
        min_margin_ok = bool(
            evidence_margin >= float(config.crc_candidate_min_evidence_margin)
        )
    if not min_margin_ok:
        return False

    rank = int(candidate.beam_rank)
    max_rank = int(config.crc_candidate_max_beam_rank)
    if max_rank < 0 or rank <= max_rank:
        return True

    high_rank_max = int(getattr(config, "crc_candidate_high_rank_max_beam_rank", -1))
    if high_rank_max < 0 or rank > high_rank_max:
        if candidate.selection_source == "crc_state":
            state_margin = float(getattr(config, "crc_state_search_min_evidence_margin", 0.0))
            return bool(evidence_margin >= state_margin)
        return False
    high_rank_margin = float(getattr(config, "crc_candidate_high_rank_min_evidence_margin", float("inf")))
    return bool(evidence_margin >= high_rank_margin)


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

    expected_payload_symbol_count = _expected_payload_symbol_count(
        payload_len=payload_len,
        sf=sf,
        cr=cr,
        has_crc=has_crc,
        ldro=ldro,
        crc_mode=cfg.crc_mode,
    )
    if expected_payload_symbol_count > 0:
        likelihoods = likelihoods[: min(len(likelihoods), expected_payload_symbol_count)]
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
    rank_diverse_states = _rank_diverse_block_states(
        block_candidates,
        top_r=int(getattr(cfg, "global_rank_diverse_top_r", 0)),
        beam_width=int(getattr(cfg, "global_rank_diverse_beam_width", 0) or getattr(cfg, "global_beam_width", 64)),
    )
    rank_cost_states = _rank_cost_block_states(
        block_candidates,
        top_r=int(getattr(cfg, "global_rank_diverse_top_r", 0)),
        max_cost=float(getattr(cfg, "global_rank_cost_max", 0.0)),
        state_limit=int(getattr(cfg, "global_rank_cost_state_limit", 0)),
    )
    if rank_diverse_states:
        seen_nibbles = {tuple(state.nibbles) for state in block_beam}
        for state in rank_diverse_states:
            key = tuple(state.nibbles)
            if key in seen_nibbles:
                continue
            block_beam.append(state)
            seen_nibbles.add(key)
        block_beam.sort(key=lambda item: item.score, reverse=True)
    if rank_cost_states:
        seen_nibbles = {tuple(state.nibbles) for state in block_beam}
        for state in rank_cost_states:
            key = tuple(state.nibbles)
            if key in seen_nibbles:
                continue
            block_beam.append(state)
            seen_nibbles.add(key)
        block_beam.sort(key=lambda item: item.score, reverse=True)
    timings["global_beam_ms"] = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    try:
        header_tail = explicit_header_tail_nibbles(header_symbol_values, sf=sf)
    except Exception:
        header_tail = ()

    t_crc_state = time.perf_counter()
    crc_state_states = _crc_state_search_states(
        block_candidates,
        header_tail=header_tail,
        payload_len=int(payload_len),
        has_crc=bool(has_crc),
        crc_mode=str(cfg.crc_mode),
        config=cfg,
    )
    timings["crc_state_search_ms"] = (time.perf_counter() - t_crc_state) * 1000.0

    payload_candidates: list[PayloadBeamCandidate] = []
    final_states: list[_BeamState] = list(block_beam[: max(1, int(cfg.final_candidate_limit))])
    if rank_diverse_states:
        seen_nibbles = {tuple(state.nibbles) for state in final_states}
        for state in rank_diverse_states:
            key = tuple(state.nibbles)
            if key in seen_nibbles:
                continue
            final_states.append(state)
            seen_nibbles.add(key)
    if crc_state_states:
        seen_nibbles = {tuple(state.nibbles) for state in final_states}
        for state in crc_state_states:
            key = tuple(state.nibbles)
            if key in seen_nibbles:
                continue
            final_states.append(state)
            seen_nibbles.add(key)
    if rank_cost_states:
        seen_nibbles = {tuple(state.nibbles) for state in final_states}
        for state in rank_cost_states:
            key = tuple(state.nibbles)
            if key in seen_nibbles:
                continue
            final_states.append(state)
            seen_nibbles.add(key)

    for rank, state in enumerate(final_states):
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
                selection_source=str(state.source),
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
        selection_policy = str(getattr(cfg, "crc_selection_policy", "best_score"))
        best_score_crc = max(crc_valid_beam, key=lambda item: item.total_score)
        earliest_crc = min(crc_valid_beam, key=lambda item: item.beam_rank)
        if selection_policy == "earliest_rank":
            selected = earliest_crc
        elif selection_policy == "score_unless_low_margin":
            margin = (
                _candidate_evidence_score(best_score_crc, cfg)
                - (_candidate_evidence_score(argmax_fallback, cfg) if argmax_fallback is not None else 0.0)
            )
            if (
                int(best_score_crc.beam_rank) != int(earliest_crc.beam_rank)
                and margin < float(cfg.crc_non_earliest_min_evidence_margin)
            ):
                selected = earliest_crc
            else:
                selected = best_score_crc
        else:
            selected = best_score_crc
    elif bool(cfg.argmax_fallback_on_crc_failure) and argmax_fallback is not None:
        failure_policy = str(getattr(cfg, "crc_failure_selection_policy", "argmax_fallback"))
        if failure_policy == "best_score" and beam_payload_candidates:
            selected = max(beam_payload_candidates, key=lambda item: item.total_score)
        elif failure_policy == "small_change_best_score" and beam_payload_candidates:
            max_changes = int(getattr(cfg, "crc_failure_max_symbol_changes", 2))
            min_margin = float(getattr(cfg, "crc_failure_min_evidence_margin", -0.50))
            fallback_evidence = _candidate_evidence_score(argmax_fallback, cfg)
            conservative = [
                item
                for item in beam_payload_candidates
                if _raw_bin_change_count(item.raw_bins, argmax_fallback.raw_bins) <= max_changes
                and _candidate_evidence_score(item, cfg) >= fallback_evidence + min_margin
            ]
            selected = max(conservative, key=lambda item: item.total_score) if conservative else argmax_fallback
        else:
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
        "expected_payload_symbol_count": int(expected_payload_symbol_count),
        "block_count": int(len(block_candidates)),
        "codeword_list_count": int(len(all_codeword_lists)),
        "candidate_payload_count": int(len(beam_payload_candidates)),
        "candidate_payload_count_with_fallback": int(len(payload_candidates)),
        "rank_diverse_state_count": int(len(rank_diverse_states)),
        "rank_cost_state_count": int(len(rank_cost_states)),
        "crc_state_state_count": int(len(crc_state_states)),
        "final_state_count": int(len(final_states)),
        "observed_crc_valid_count": int(sum(1 for item in beam_payload_candidates if item.observed_crc_valid)),
        "accepted_crc_valid_count": int(len(crc_valid_beam)),
        "accepted_crc_min_beam_rank": int(min((item.beam_rank for item in crc_valid_beam), default=-1)),
        "accepted_crc_max_beam_rank": int(max((item.beam_rank for item in crc_valid_beam), default=-1)),
        "accepted_crc_best_evidence_margin": float(
            max(
                (
                    _candidate_evidence_score(item, cfg)
                    - (_candidate_evidence_score(argmax_fallback, cfg) if argmax_fallback is not None else 0.0)
                    for item in crc_valid_beam
                ),
                default=0.0,
            )
        ),
        "selected_evidence_margin": float(
            (
                _candidate_evidence_score(selected, cfg)
                - (_candidate_evidence_score(argmax_fallback, cfg) if argmax_fallback is not None else 0.0)
            )
            if selected is not None and selected.selection_source != "argmax_fallback"
            else 0.0
        ),
        "selected_is_earliest_accepted_crc": int(
            selected is not None
            and selected.selection_source != "argmax_fallback"
            and bool(crc_valid_beam)
            and int(selected.beam_rank) == min(item.beam_rank for item in crc_valid_beam)
        ),
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

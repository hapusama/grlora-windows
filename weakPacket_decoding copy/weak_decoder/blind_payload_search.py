"""Single-packet blind payload search for low-SNR LoRa decoding.

This module implements the core blind decoder that uses ONLY LoRa PHY layer priors:
  - Header-first decoded parameters (payload_len, CR, CRC, LDRO)
  - FFT amplitude/phase evidence from payload symbols
  - Header-based phase line for phase prediction
  - LoRa coding constraints (Hamming parity, interleaver structure)
  - CRC validation

NO cross-packet priors, NO payload templates, NO application-layer patterns.

Architecture:
  1. Per-symbol confidence analysis → low-confidence symbols get candidate sets
  2. Block-wise search over LoRa interleaver blocks (CR+4 symbols per block)
  3. Beam search to connect blocks while respecting token budget
  4. CRC validation for final payload selection

Key innovation: Multi-symbol joint decoding with LoRa code constraints beats
single-symbol argmax at low SNR, without requiring cross-packet session data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .chirp import bin_to_grlora_symbol, positive_mod, signed_fft_bin
from .header_first_demod import deinterleave_hard, gray_demapping, hamming_decode_hard
from .payload_codec import decode_payload_symbols, PayloadDecodeResult
from .phase_guided_demod import (
    PhaseLine,
    build_candidate_set,
    score_candidates_phase_guided,
)


@dataclass
class SymbolConfidence:
    """Confidence metrics for one payload symbol."""

    symbol_index: int
    argmax_bin: int
    argmax_symbol_value: int

    # Confidence indicators
    top1_power: float
    top2_power: float
    peak_gap_db: float  # top1 - top2 in dB
    peak_ratio: float  # top1 / top2
    spectral_entropy: float
    energy_ratio: float  # argmax_power / total_power

    # Phase consistency (relative to header-based phase line)
    phase_residual: float
    phase_score: float  # cos(phase_residual)

    # Overall confidence [0, 1]
    confidence: float
    is_high_confidence: bool


@dataclass
class SymbolCandidate:
    """One candidate bin for a payload symbol."""

    raw_bin: int
    symbol_value: int
    amplitude: float
    power: float
    phase: float
    phase_residual: float
    phase_score: float
    amp_score: float
    combined_score: float
    rank_by_amp: int
    rank_by_phase: int
    rank_by_combined: int


@dataclass
class BlockCandidate:
    """One candidate configuration for an interleaver block."""

    block_index: int
    symbol_values: tuple[int, ...]  # CR+4 symbol values
    raw_bins: tuple[int, ...]

    # Scores
    fft_score: float  # Combined amplitude + phase score
    phase_smoothness: float  # How well phases fit a line
    hamming_parity_valid: bool
    hamming_syndrome_cost: float

    total_score: float


@dataclass
class PayloadCandidate:
    """One complete payload candidate across all blocks."""

    symbol_values: tuple[int, ...]
    raw_bins: tuple[int, ...]

    # Block-level breakdown
    block_scores: tuple[float, ...]

    # Overall scores
    total_fft_score: float
    phase_line_rmse_pi: float
    phase_line_r2: float
    hamming_valid_ratio: float

    # Payload decode result
    payload_decode: Optional[PayloadDecodeResult]
    crc_valid: bool

    total_score: float


@dataclass
class BlindSearchConfig:
    """Configuration for blind payload search."""

    # Confidence threshold to trigger candidate search
    confidence_threshold: float = 0.5

    # Candidate set sizes
    amplitude_topk: int = 16
    argmax_window: int = 3
    phase_topk: int = 8
    combined_topk: int = 16

    # Scoring weights
    phase_weight: float = 0.85
    fft_score_weight: float = 0.40
    phase_smooth_weight: float = 0.35
    hamming_parity_weight: float = 0.25

    # Block search
    max_candidates_per_symbol: int = 4  # Keep top-N candidates per low-conf symbol
    block_beam_width: int = 64  # Beam width for block-level search

    # Global beam search
    global_beam_width: int = 16  # Beam width across blocks

    # CRC bonus
    crc_valid_bonus: float = 2.0


def compute_symbol_confidence(
    spectrum: np.ndarray,
    predicted_phase_rad: float,
    config: BlindSearchConfig,
) -> SymbolConfidence:
    """Compute confidence metrics for one symbol to decide if candidate search is needed."""

    power = np.abs(spectrum) ** 2
    sorted_indices = np.argsort(power)[::-1]

    argmax_bin = int(sorted_indices[0])
    top1_power = float(power[argmax_bin])
    top2_power = float(power[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0

    # Peak gap and ratio
    peak_gap_db = float(10.0 * math.log10((top1_power + 1e-30) / (top2_power + 1e-30)))
    peak_ratio = float(top1_power / (top2_power + 1e-30))

    # Energy ratio
    total_power = float(np.sum(power))
    energy_ratio = float(top1_power / (total_power + 1e-30))

    # Spectral entropy
    prob = power / (total_power + 1e-30)
    prob_nz = prob[prob > 1e-12]
    spectral_entropy = float(-np.sum(prob_nz * np.log2(prob_nz)))

    # Phase consistency
    argmax_phase = float(np.angle(spectrum[argmax_bin]))
    phase_residual = float(np.angle(np.exp(1j * (argmax_phase - predicted_phase_rad))))
    phase_score = float(np.cos(phase_residual))

    # Overall confidence (weighted combination)
    # High confidence when: strong peak, low entropy, good phase match
    peak_quality = min(1.0, peak_gap_db / 10.0)  # Normalize to [0, 1]
    entropy_quality = max(0.0, 1.0 - spectral_entropy / 8.0)  # Lower entropy = better
    phase_quality = max(0.0, phase_score)
    energy_quality = min(1.0, energy_ratio * 5.0)

    confidence = float(
        0.30 * peak_quality +
        0.25 * entropy_quality +
        0.30 * phase_quality +
        0.15 * energy_quality
    )

    is_high_confidence = confidence >= config.confidence_threshold

    return SymbolConfidence(
        symbol_index=-1,  # Set by caller
        argmax_bin=argmax_bin,
        argmax_symbol_value=-1,  # Set by caller
        top1_power=top1_power,
        top2_power=top2_power,
        peak_gap_db=peak_gap_db,
        peak_ratio=peak_ratio,
        spectral_entropy=spectral_entropy,
        energy_ratio=energy_ratio,
        phase_residual=phase_residual,
        phase_score=phase_score,
        confidence=confidence,
        is_high_confidence=is_high_confidence,
    )


def build_symbol_candidate_set(
    spectrum: np.ndarray,
    predicted_phase_rad: float,
    sf: int,
    ldro: bool,
    config: BlindSearchConfig,
) -> list[SymbolCandidate]:
    """Build candidate set for one low-confidence symbol.

    Combines multiple selection strategies:
      1. Amplitude topK
      2. Argmax ±W window
      3. Phase-residual topK
      4. Combined score topK

    Returns unique candidates ranked by combined score.
    """

    n_bins = 1 << sf
    power = np.abs(spectrum) ** 2
    phases = np.angle(spectrum)

    # Get candidate bins using phase_guided_demod's method
    candidate_bins, candidate_values = build_candidate_set(
        spectrum,
        top_l=max(config.amplitude_topk, config.combined_topk, 64),
        argmax_window_radius=config.argmax_window,
        energy_threshold_db=12.0,
        n_bins=n_bins,
    )

    if len(candidate_bins) == 0:
        return []

    # Score candidates
    scores, residuals, cand_phases, amp_factors = score_candidates_phase_guided(
        candidate_values,
        candidate_bins,
        predicted_phase_rad,
        spectrum_power=power,
        phase_weight=config.phase_weight,
    )

    # Build candidate records
    candidates: dict[int, SymbolCandidate] = {}

    # Rank by amplitude
    amp_ranks = {int(candidate_bins[i]): i for i in np.argsort(amp_factors)[::-1]}

    # Rank by phase score
    phase_scores = np.cos(residuals)
    phase_ranks = {int(candidate_bins[i]): i for i in np.argsort(phase_scores)[::-1]}

    # Rank by combined score
    combined_ranks = {int(candidate_bins[i]): i for i in np.argsort(scores)[::-1]}

    for i, raw_bin in enumerate(candidate_bins):
        bin_idx = int(raw_bin)
        symbol_value = bin_to_grlora_symbol(bin_idx, sf=sf, is_header=False, ldro=ldro)

        # Only keep if in one of the topK sets
        include = False
        if amp_ranks[bin_idx] < config.amplitude_topk:
            include = True
        if phase_ranks[bin_idx] < config.phase_topk:
            include = True
        if combined_ranks[bin_idx] < config.combined_topk:
            include = True

        if not include:
            continue

        # Use best candidate for this symbol_value (handles LDRO aliasing)
        if symbol_value in [c.symbol_value for c in candidates.values()]:
            existing = next(c for c in candidates.values() if c.symbol_value == symbol_value)
            if scores[i] <= existing.combined_score:
                continue

        candidates[bin_idx] = SymbolCandidate(
            raw_bin=bin_idx,
            symbol_value=symbol_value,
            amplitude=float(np.abs(spectrum[bin_idx])),
            power=float(power[bin_idx]),
            phase=float(cand_phases[i]),
            phase_residual=float(residuals[i]),
            phase_score=float(phase_scores[i]),
            amp_score=float(amp_factors[i]),
            combined_score=float(scores[i]),
            rank_by_amp=amp_ranks[bin_idx],
            rank_by_phase=phase_ranks[bin_idx],
            rank_by_combined=combined_ranks[bin_idx],
        )

    # Return top candidates by combined score
    sorted_candidates = sorted(candidates.values(), key=lambda c: c.combined_score, reverse=True)
    return sorted_candidates[:config.max_candidates_per_symbol]


def check_hamming_parity(symbol_values: Sequence[int], sf: int, cr: int, ldro: bool) -> tuple[bool, float]:
    """Check if a block of symbols satisfies Hamming parity constraints.

    For CR=4/5 (cr=1), each deinterleaved codeword has even parity.

    Returns:
        (all_valid, valid_ratio)
    """
    if cr != 1:
        return True, 1.0  # Only CR=1 has simple parity check

    try:
        gray_symbols = gray_demapping(symbol_values)
        codewords = deinterleave_hard(
            gray_symbols,
            sf=sf,
            is_header=False,
            cr=cr,
            ldro=ldro,
        )
    except Exception:
        return False, 0.0

    if not codewords:
        return False, 0.0

    valid_count = 0
    for cw in codewords:
        bits = int(cw) & ((1 << (cr + 4)) - 1)
        if bits.bit_count() % 2 == 0:
            valid_count += 1

    valid_ratio = valid_count / len(codewords)
    all_valid = valid_count == len(codewords)

    return all_valid, valid_ratio


def score_block_candidate(
    symbol_values: Sequence[int],
    candidate_records: Sequence[list[SymbolCandidate]],
    sf: int,
    cr: int,
    ldro: bool,
    config: BlindSearchConfig,
) -> float:
    """Score one block candidate configuration.

    Combines:
      - FFT evidence (amplitude + phase)
      - Phase smoothness across symbols
      - Hamming parity bonus
    """

    # FFT score: average of per-symbol combined scores
    fft_scores = []
    for sym_idx, sym_val in enumerate(symbol_values):
        # Find matching candidate
        cands = candidate_records[sym_idx]
        matching = [c for c in cands if c.symbol_value == sym_val]
        if matching:
            fft_scores.append(matching[0].combined_score)
        else:
            fft_scores.append(0.0)  # Penalty for missing candidate

    fft_score = float(np.mean(fft_scores)) if fft_scores else 0.0

    # Hamming parity check
    parity_valid, parity_ratio = check_hamming_parity(symbol_values, sf, cr, ldro)
    parity_score = parity_ratio

    # Combined score
    total = (
        config.fft_score_weight * fft_score +
        config.hamming_parity_weight * parity_score
    )

    return float(total)


def search_block_candidates(
    block_index: int,
    candidate_records: list[list[SymbolCandidate]],
    sf: int,
    cr: int,
    ldro: bool,
    config: BlindSearchConfig,
) -> list[BlockCandidate]:
    """Search for best candidate configurations for one interleaver block.

    Uses beam search to explore symbol combinations within the block.
    Each block has CR+4 symbols (e.g., 5 symbols for CR=1).
    """

    block_len = cr + 4
    if len(candidate_records) != block_len:
        return []

    # Beam search over block
    # State: (partial_symbol_values, partial_bins, score)
    beam: list[tuple[tuple[int, ...], tuple[int, ...], float]] = [((), (), 0.0)]

    for sym_pos in range(block_len):
        next_beam: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []

        for partial_syms, partial_bins, partial_score in beam:
            # Try each candidate for this symbol position
            for cand in candidate_records[sym_pos]:
                new_syms = partial_syms + (cand.symbol_value,)
                new_bins = partial_bins + (cand.raw_bin,)

                # Incremental score update
                new_score = partial_score + cand.combined_score

                # Add Hamming bonus at the end
                if sym_pos == block_len - 1:
                    parity_valid, parity_ratio = check_hamming_parity(new_syms, sf, cr, ldro)
                    new_score += config.hamming_parity_weight * parity_ratio

                next_beam.append((new_syms, new_bins, new_score))

        # Keep top beam_width candidates
        next_beam.sort(key=lambda x: x[2], reverse=True)
        beam = next_beam[:config.block_beam_width]

    # Convert top beam entries to BlockCandidate
    results = []
    for syms, bins, score in beam[:config.block_beam_width]:
        parity_valid, _ = check_hamming_parity(syms, sf, cr, ldro)

        results.append(BlockCandidate(
            block_index=block_index,
            symbol_values=syms,
            raw_bins=bins,
            fft_score=score,
            phase_smoothness=0.0,  # TODO: compute if needed
            hamming_parity_valid=parity_valid,
            hamming_syndrome_cost=0.0,
            total_score=score,
        ))

    return results


def blind_payload_search(
    payload_spectra: Sequence[np.ndarray],
    payload_abs_indices: Sequence[float],
    phase_line: PhaseLine,
    sf: int,
    cr: int,
    ldro: bool,
    payload_len: int,
    has_crc: bool,
    config: BlindSearchConfig,
) -> list[PayloadCandidate]:
    """Execute blind payload search across all symbols.

    Steps:
      1. Compute confidence for each symbol
      2. Build candidate sets for low-confidence symbols
      3. High-confidence symbols fixed to argmax
      4. Block-wise search within each interleaver block
      5. Beam search to connect blocks
      6. Decode and CRC-validate final candidates

    Returns:
        List of payload candidates, sorted by total_score descending.
    """

    n_bins = 1 << sf
    n_symbols = len(payload_spectra)

    if n_symbols == 0:
        return []

    # Step 1: Analyze confidence and build candidate sets
    confidences: list[SymbolConfidence] = []
    candidate_sets: list[list[SymbolCandidate]] = []

    for k, (spectrum, abs_idx) in enumerate(zip(payload_spectra, payload_abs_indices)):
        predicted_phase = phase_line.predict(abs_idx)

        conf = compute_symbol_confidence(spectrum, predicted_phase, config)
        conf.symbol_index = k
        conf.argmax_symbol_value = bin_to_grlora_symbol(
            conf.argmax_bin, sf=sf, is_header=False, ldro=ldro
        )
        confidences.append(conf)

        if conf.is_high_confidence:
            # High confidence: fix to argmax
            candidate_sets.append([
                SymbolCandidate(
                    raw_bin=conf.argmax_bin,
                    symbol_value=conf.argmax_symbol_value,
                    amplitude=float(np.abs(spectrum[conf.argmax_bin])),
                    power=conf.top1_power,
                    phase=float(np.angle(spectrum[conf.argmax_bin])),
                    phase_residual=conf.phase_residual,
                    phase_score=conf.phase_score,
                    amp_score=1.0,
                    combined_score=conf.phase_score,
                    rank_by_amp=0,
                    rank_by_phase=0,
                    rank_by_combined=0,
                )
            ])
        else:
            # Low confidence: build candidate set
            candidates = build_symbol_candidate_set(
                spectrum, predicted_phase, sf, ldro, config
            )
            candidate_sets.append(candidates if candidates else [])

    # Step 2: Block-wise search
    block_len = cr + 4
    block_candidates_by_block: list[list[BlockCandidate]] = []

    for block_start in range(0, n_symbols, block_len):
        block_end = min(block_start + block_len, n_symbols)

        if block_end - block_start < block_len:
            # Incomplete block at end - skip for now
            break

        block_cands = candidate_sets[block_start:block_end]

        # Search this block
        block_results = search_block_candidates(
            block_index=len(block_candidates_by_block),
            candidate_records=block_cands,
            sf=sf,
            cr=cr,
            ldro=ldro,
            config=config,
        )

        block_candidates_by_block.append(block_results)

    if not block_candidates_by_block:
        return []

    # Step 3: Beam search across blocks
    # State: (block_choices: list[BlockCandidate], total_score)
    beam: list[tuple[list[BlockCandidate], float]] = [([], 0.0)]

    for block_results in block_candidates_by_block:
        next_beam: list[tuple[list[BlockCandidate], float]] = []

        for path, path_score in beam:
            for block_cand in block_results:
                new_path = path + [block_cand]
                new_score = path_score + block_cand.total_score
                next_beam.append((new_path, new_score))

        # Keep top beam_width
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:config.global_beam_width]

    # Step 4: Assemble and decode final candidates
    payload_candidates: list[PayloadCandidate] = []

    for path, path_score in beam:
        # Concatenate symbol values from all blocks
        all_symbol_values = []
        all_bins = []
        block_scores = []

        for block_cand in path:
            all_symbol_values.extend(block_cand.symbol_values)
            all_bins.extend(block_cand.raw_bins)
            block_scores.append(block_cand.total_score)

        # Decode payload
        payload_decode = None
        crc_valid = False
        try:
            payload_decode = decode_payload_symbols(
                all_symbol_values,
                sf=sf,
                cr=cr,
                ldro=ldro,
                payload_len=payload_len,
                has_crc=has_crc,
                crc_mode="grlora",
            )
            crc_valid = payload_decode.crc_valid
        except Exception:
            pass

        # Total score with CRC bonus
        total_score = path_score
        if crc_valid:
            total_score += config.crc_valid_bonus

        payload_candidates.append(PayloadCandidate(
            symbol_values=tuple(all_symbol_values),
            raw_bins=tuple(all_bins),
            block_scores=tuple(block_scores),
            total_fft_score=path_score,
            phase_line_rmse_pi=0.0,  # TODO: compute if needed
            phase_line_r2=0.0,
            hamming_valid_ratio=sum(bc.hamming_parity_valid for bc in path) / len(path),
            payload_decode=payload_decode,
            crc_valid=crc_valid,
            total_score=total_score,
        ))

    # Sort by total score
    payload_candidates.sort(key=lambda c: c.total_score, reverse=True)

    return payload_candidates

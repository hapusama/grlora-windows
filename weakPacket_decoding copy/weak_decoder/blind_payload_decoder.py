"""Block-wise beam search decoder for low-SNR LoRa payload recovery.

This module implements single-packet blind decoding using only PHY-layer priors:
  - Header-based phase line for candidate set construction
  - FFT amplitude and phase evidence
  - LoRa encoding constraints (Hamming/parity check)
  - CRC validation

No payload templates, no cross-packet joint decoding, no affine byte priors.

Architecture:
  1. Per-symbol confidence analysis → candidate sets
  2. Block-wise LoRa code-constrained search (one interleaver block at a time)
  3. Beam search across blocks to limit combinatorial explosion
  4. CRC-guided final selection

Each interleaver block (CR=1 → 5 LoRa symbols) is decoded independently,
with beam search connecting blocks. High-confidence symbols are fixed to argmax;
low-confidence symbols explore candidates constructed from:
  - Amplitude topK
  - Argmax ±W window
  - Phase-residual topK
  - Combined (amplitude + phase) topK
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .chirp import bin_to_grlora_symbol, positive_mod
from .header_first_demod import (
    deinterleave_hard,
    gray_demapping,
    hamming_decode_hard,
)
from .payload_codec import (
    PayloadDecodeResult,
    crc16,
    WHITENING_SEQ,
)
from .phase_guided_demod import (
    PhaseLine,
    score_candidates_phase_guided,
)


# ── Confidence Metrics ──────────────────────────────────────────────────────

@dataclass
class SymbolConfidence:
    """Per-symbol confidence metrics for candidate set construction."""
    symbol_index: int
    argmax_bin: int
    argmax_power: float
    second_power: float
    total_power: float

    # Confidence indicators
    peak_margin_db: float  # argmax vs second-best power gap
    peak_ratio: float      # argmax / second-best
    energy_ratio: float    # argmax / total
    spectral_entropy: float

    # Phase alignment
    predicted_phase_rad: float
    argmax_phase_residual: float
    argmax_phase_score: float  # cos(phase_residual)

    # Combined confidence score
    confidence: float

    @property
    def is_high_confidence(self) -> bool:
        """High confidence: trust argmax, no candidate search needed."""
        # Heuristic: strong amplitude peak + good phase alignment
        return (
            self.peak_margin_db > 6.0 and
            self.peak_ratio > 4.0 and
            self.argmax_phase_score > 0.7
        )


def compute_symbol_confidence(
    spectrum: np.ndarray,
    predicted_phase_rad: float,
) -> SymbolConfidence:
    """Compute confidence metrics for one symbol's FFT spectrum."""
    power = np.abs(spectrum) ** 2
    sorted_indices = np.argsort(power)[::-1]

    argmax_bin = int(sorted_indices[0])
    argmax_power = float(power[argmax_bin])
    second_power = float(power[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
    total_power = float(np.sum(power))

    # Amplitude metrics
    peak_margin_db = 10.0 * math.log10((argmax_power + 1e-30) / (second_power + 1e-30))
    peak_ratio = argmax_power / (second_power + 1e-30)
    energy_ratio = argmax_power / (total_power + 1e-30)

    # Spectral entropy (normalized)
    prob = power / (total_power + 1e-30)
    prob = prob[prob > 1e-30]
    entropy = -float(np.sum(prob * np.log2(prob)))
    max_entropy = math.log2(spectrum.size)
    spectral_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    # Phase alignment
    argmax_phase = float(np.angle(spectrum[argmax_bin]))
    argmax_phase_residual = float(np.angle(np.exp(1j * (argmax_phase - predicted_phase_rad))))
    argmax_phase_score = float(np.cos(argmax_phase_residual))

    # Combined confidence (weighted average)
    # Higher is more confident
    amp_conf = min(peak_margin_db / 12.0, 1.0)  # normalize to [0, 1]
    phase_conf = (argmax_phase_score + 1.0) / 2.0  # map [-1, 1] to [0, 1]
    entropy_conf = 1.0 - spectral_entropy  # low entropy = high confidence

    confidence = 0.5 * amp_conf + 0.3 * phase_conf + 0.2 * entropy_conf

    return SymbolConfidence(
        symbol_index=0,  # Will be set by caller
        argmax_bin=argmax_bin,
        argmax_power=argmax_power,
        second_power=second_power,
        total_power=total_power,
        peak_margin_db=peak_margin_db,
        peak_ratio=peak_ratio,
        energy_ratio=energy_ratio,
        spectral_entropy=spectral_entropy,
        predicted_phase_rad=predicted_phase_rad,
        argmax_phase_residual=argmax_phase_residual,
        argmax_phase_score=argmax_phase_score,
        confidence=confidence,
    )


# ── Candidate Set Construction ──────────────────────────────────────────────

@dataclass
class SymbolCandidates:
    """Candidate bins for one low-confidence symbol."""
    symbol_index: int
    confidence: SymbolConfidence
    candidate_bins: np.ndarray  # Shape: (n_candidates,)
    candidate_scores: np.ndarray  # Combined amplitude + phase scores
    candidate_phases: np.ndarray

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_bins)


def build_symbol_candidates(
    spectrum: np.ndarray,
    confidence: SymbolConfidence,
    amplitude_topk: int = 8,
    argmax_window: int = 4,
    phase_topk: int = 8,
    combined_topk: int = 16,
    phase_weight: float = 0.85,
) -> SymbolCandidates:
    """Build candidate set for one low-confidence symbol.

    Union of:
      - Amplitude topK
      - Argmax ±W window
      - Phase-residual topK
      - Amplitude+phase combined topK
    """
    n_bins = spectrum.size
    power = np.abs(spectrum) ** 2

    candidates = set()

    # (1) Amplitude topK
    sorted_by_amp = np.argsort(power)[::-1]
    candidates.update(int(b) for b in sorted_by_amp[:amplitude_topk])

    # (2) Argmax ±W window
    for delta in range(-argmax_window, argmax_window + 1):
        candidates.add((confidence.argmax_bin + delta) % n_bins)

    # (3) Phase-residual topK
    phases = np.angle(spectrum)
    phase_residuals = np.angle(np.exp(1j * (phases - confidence.predicted_phase_rad)))
    phase_scores = np.cos(phase_residuals)
    sorted_by_phase = np.argsort(phase_scores)[::-1]
    candidates.update(int(b) for b in sorted_by_phase[:phase_topk])

    # (4) Combined (amplitude + phase) topK
    amp_factors = power / (float(np.max(power)) + 1e-30)
    combined_scores = phase_weight * phase_scores + (1.0 - phase_weight) * amp_factors
    sorted_by_combined = np.argsort(combined_scores)[::-1]
    candidates.update(int(b) for b in sorted_by_combined[:combined_topk])

    # Finalize candidate set
    candidate_bins = np.array(sorted(candidates), dtype=np.int32)
    candidate_scores = combined_scores[candidate_bins]
    candidate_phases = phases[candidate_bins]

    return SymbolCandidates(
        symbol_index=confidence.symbol_index,
        confidence=confidence,
        candidate_bins=candidate_bins,
        candidate_scores=candidate_scores,
        candidate_phases=candidate_phases,
    )


# ── Block-wise Decoding ─────────────────────────────────────────────────────

@dataclass
class BlockCandidate:
    """One candidate decoding for an interleaver block."""
    block_index: int
    symbol_bins: tuple[int, ...]  # Raw FFT bins for symbols in this block
    nibbles: tuple[int, ...]  # Decoded nibbles (after gray, deinterleave, Hamming)

    # Scores
    amplitude_score: float
    phase_score: float
    combined_score: float
    hamming_valid: bool  # Did Hamming decode succeed?

    @property
    def total_score(self) -> float:
        """Total score for beam search ranking."""
        base = self.combined_score
        # Bonus for valid Hamming decoding
        if self.hamming_valid:
            base += 0.5
        return base


def decode_lora_block(
    symbol_bins: Sequence[int],
    sf: int,
    cr: int,
) -> Tuple[tuple[int, ...], bool]:
    """Decode one LoRa interleaver block: symbols → nibbles.

    Returns:
        (nibbles, hamming_valid)
    """
    n_bins = 1 << sf

    # Bins → gray symbols
    gray_symbols = tuple(bin_to_grlora_symbol(int(b), n_bins) for b in symbol_bins)

    # Gray demapping
    nibbles_with_parity = gray_demapping(gray_symbols)

    # Deinterleaving
    deinterleaved = deinterleave_hard(nibbles_with_parity, cr)

    # Hamming decode
    decoded_nibbles, errors = hamming_decode_hard(deinterleaved, cr)
    hamming_valid = all(e == 0 for e in errors)

    return decoded_nibbles, hamming_valid


def score_block_candidate(
    symbol_bins: Sequence[int],
    spectra: Sequence[np.ndarray],
    predicted_phases: Sequence[float],
    phase_weight: float = 0.85,
) -> Tuple[float, float, float]:
    """Score a block candidate using FFT evidence.

    Returns:
        (amplitude_score, phase_score, combined_score)
    """
    amp_scores = []
    phase_scores = []

    for bin_idx, spectrum, pred_phase in zip(symbol_bins, spectra, predicted_phases):
        power = np.abs(spectrum) ** 2
        max_power = float(np.max(power))
        amp_score = float(power[bin_idx]) / (max_power + 1e-30)

        phase = float(np.angle(spectrum[bin_idx]))
        phase_residual = float(np.angle(np.exp(1j * (phase - pred_phase))))
        phase_score = float(np.cos(phase_residual))

        amp_scores.append(amp_score)
        phase_scores.append(phase_score)

    avg_amp = float(np.mean(amp_scores))
    avg_phase = float(np.mean(phase_scores))
    combined = phase_weight * avg_phase + (1.0 - phase_weight) * avg_amp

    return avg_amp, avg_phase, combined


def enumerate_block_candidates(
    block_symbol_indices: Sequence[int],
    all_symbol_candidates: dict[int, SymbolCandidates],
    fixed_bins: dict[int, int],
    spectra: Sequence[np.ndarray],
    predicted_phases: Sequence[float],
    sf: int,
    cr: int,
    max_candidates: int = 1000,
    phase_weight: float = 0.85,
) -> list[BlockCandidate]:
    """Enumerate candidates for one interleaver block.

    Args:
        block_symbol_indices: Symbol indices in this block (e.g., [0, 1, 2, 3, 4] for CR=1)
        all_symbol_candidates: Map from symbol_index → SymbolCandidates
        fixed_bins: Map from symbol_index → fixed bin (for high-confidence symbols)
        spectra: FFT spectra for symbols in this block
        predicted_phases: Predicted phases for symbols in this block
        sf: Spreading factor
        cr: Coding rate
        max_candidates: Max candidates to generate (prune by score)
        phase_weight: Phase weight for scoring

    Returns:
        List of BlockCandidate, sorted by total_score (descending)
    """
    # Separate fixed and variable symbols
    variable_indices = [i for i in block_symbol_indices if i not in fixed_bins]

    if not variable_indices:
        # All symbols fixed: only one candidate
        symbol_bins = tuple(fixed_bins[i] for i in block_symbol_indices)
        amp_score, phase_score, combined_score = score_block_candidate(
            symbol_bins, spectra, predicted_phases, phase_weight
        )
        nibbles, hamming_valid = decode_lora_block(symbol_bins, sf, cr)

        return [BlockCandidate(
            block_index=0,
            symbol_bins=symbol_bins,
            nibbles=nibbles,
            amplitude_score=amp_score,
            phase_score=phase_score,
            combined_score=combined_score,
            hamming_valid=hamming_valid,
        )]

    # Enumerate variable symbol combinations
    # Use greedy pruning: only keep top-scoring partial combinations
    from itertools import product

    candidate_lists = []
    for idx in variable_indices:
        if idx in all_symbol_candidates:
            candidate_lists.append(all_symbol_candidates[idx].candidate_bins)
        else:
            # Fallback: use argmax
            candidate_lists.append(np.array([0], dtype=np.int32))  # Placeholder

    # Combinatorial explosion check
    total_combinations = 1
    for clist in candidate_lists:
        total_combinations *= len(clist)

    if total_combinations > max_candidates:
        # Prune: only use top candidates per symbol
        pruned_lists = []
        for idx, clist in zip(variable_indices, candidate_lists):
            if idx in all_symbol_candidates:
                sym_cand = all_symbol_candidates[idx]
                # Sort by score, keep top
                k = min(len(clist), max(4, max_candidates // len(variable_indices)))
                top_indices = np.argsort(sym_cand.candidate_scores)[::-1][:k]
                pruned_lists.append(clist[top_indices])
            else:
                pruned_lists.append(clist)
        candidate_lists = pruned_lists

    # Generate all combinations
    candidates = []
    for combo in product(*candidate_lists):
        # Build full symbol_bins
        symbol_bins_list = []
        for idx in block_symbol_indices:
            if idx in fixed_bins:
                symbol_bins_list.append(fixed_bins[idx])
            else:
                var_pos = variable_indices.index(idx)
                symbol_bins_list.append(int(combo[var_pos]))

        symbol_bins = tuple(symbol_bins_list)

        # Score
        amp_score, phase_score, combined_score = score_block_candidate(
            symbol_bins, spectra, predicted_phases, phase_weight
        )

        # Decode
        nibbles, hamming_valid = decode_lora_block(symbol_bins, sf, cr)

        candidates.append(BlockCandidate(
            block_index=0,
            symbol_bins=symbol_bins,
            nibbles=nibbles,
            amplitude_score=amp_score,
            phase_score=phase_score,
            combined_score=combined_score,
            hamming_valid=hamming_valid,
        ))

        if len(candidates) >= max_candidates:
            break

    # Sort by total_score (descending)
    candidates.sort(key=lambda c: c.total_score, reverse=True)
    return candidates[:max_candidates]


# ── Beam Search Across Blocks ───────────────────────────────────────────────

@dataclass(order=True)
class BeamState:
    """One partial payload decoding state in the beam."""
    # Ordering by negative score for min-heap
    priority: float = field(compare=True)

    block_index: int = field(compare=False)
    nibbles: tuple[int, ...] = field(compare=False)  # Accumulated nibbles so far
    symbol_bins: tuple[int, ...] = field(compare=False)  # Accumulated symbol bins
    cumulative_score: float = field(compare=False)
    n_valid_hamming: int = field(compare=False)


def beam_search_payload(
    all_spectra: Sequence[np.ndarray],
    all_predicted_phases: Sequence[float],
    all_symbol_candidates: dict[int, SymbolCandidates],
    fixed_bins: dict[int, int],
    sf: int,
    cr: int,
    payload_symbol_count: int,
    beam_width: int = 10,
    max_block_candidates: int = 100,
    phase_weight: float = 0.85,
) -> list[BeamState]:
    """Beam search across all interleaver blocks.

    Returns:
        List of final beam states, sorted by cumulative_score (descending)
    """
    # Determine block boundaries
    # For CR=1: each block outputs CR+4=5 symbols
    symbols_per_block = cr + 4
    n_blocks = (payload_symbol_count + symbols_per_block - 1) // symbols_per_block

    # Initialize beam with empty state
    beam = [BeamState(
        priority=0.0,
        block_index=0,
        nibbles=(),
        symbol_bins=(),
        cumulative_score=0.0,
        n_valid_hamming=0,
    )]

    # Process each block
    for block_idx in range(n_blocks):
        start_sym = block_idx * symbols_per_block
        end_sym = min(start_sym + symbols_per_block, payload_symbol_count)
        block_symbol_indices = list(range(start_sym, end_sym))

        block_spectra = [all_spectra[i] for i in block_symbol_indices]
        block_phases = [all_predicted_phases[i] for i in block_symbol_indices]

        # Expand each beam state
        new_beam = []
        for state in beam:
            # Generate block candidates
            block_candidates = enumerate_block_candidates(
                block_symbol_indices,
                all_symbol_candidates,
                fixed_bins,
                block_spectra,
                block_phases,
                sf,
                cr,
                max_candidates=max_block_candidates,
                phase_weight=phase_weight,
            )

            # Extend state with each block candidate
            for bc in block_candidates:
                new_nibbles = state.nibbles + bc.nibbles
                new_symbol_bins = state.symbol_bins + bc.symbol_bins
                new_score = state.cumulative_score + bc.total_score
                new_n_valid = state.n_valid_hamming + (1 if bc.hamming_valid else 0)

                new_state = BeamState(
                    priority=-new_score,  # Min-heap: negate for max
                    block_index=block_idx + 1,
                    nibbles=new_nibbles,
                    symbol_bins=new_symbol_bins,
                    cumulative_score=new_score,
                    n_valid_hamming=new_n_valid,
                )
                new_beam.append(new_state)

        # Prune to beam_width
        new_beam.sort(key=lambda s: -s.cumulative_score)  # Descending
        beam = new_beam[:beam_width]

    # Return final beam, sorted by score
    beam.sort(key=lambda s: -s.cumulative_score)
    return beam


# ── Payload Reconstruction ──────────────────────────────────────────────────

def nibbles_to_payload(
    nibbles: Sequence[int],
    payload_len: int,
    has_crc: bool,
) -> Tuple[bytes, bytes, bool]:
    """Convert nibbles to payload bytes + CRC validation.

    Returns:
        (payload_bytes, crc_bytes, crc_valid)
    """
    # Nibbles to bytes
    byte_list = []
    for i in range(0, len(nibbles), 2):
        if i + 1 < len(nibbles):
            byte_val = (nibbles[i] << 4) | nibbles[i + 1]
        else:
            byte_val = nibbles[i] << 4  # Padding
        byte_list.append(byte_val)

    raw_bytes = bytes(byte_list)

    # Separate payload and CRC
    crc_len = 2 if has_crc else 0
    if len(raw_bytes) < payload_len + crc_len:
        # Insufficient bytes
        return raw_bytes[:payload_len], b"", False

    whiten_payload = raw_bytes[:payload_len]
    crc_bytes = raw_bytes[payload_len:payload_len + crc_len] if has_crc else b""

    # Dewhiten payload
    payload_bytes = bytes(
        whiten_payload[i] ^ WHITENING_SEQ[i % len(WHITENING_SEQ)]
        for i in range(len(whiten_payload))
    )

    # CRC validation
    crc_valid = False
    if has_crc and len(crc_bytes) == 2:
        # Compute CRC over whitened payload (before dewhitening)
        crc_received = (crc_bytes[0] << 8) | crc_bytes[1]
        # LoRa CRC-16 (poly=0x1021, init=0)
        crc_computed = crc16(whiten_payload)
        crc_valid = (crc_computed == crc_received)

    return payload_bytes, crc_bytes, crc_valid


@dataclass
class BlindDecodeResult:
    """Result of blind payload decoding."""
    success: bool
    payload_bytes: bytes
    crc_valid: bool
    symbol_bins: tuple[int, ...]
    nibbles: tuple[int, ...]

    beam_rank: int  # Which beam state was selected (0 = best)
    cumulative_score: float
    n_valid_hamming: int

    # Statistics
    n_high_confidence: int
    n_low_confidence: int
    avg_candidates_per_symbol: float
    beam_width: int
    n_blocks: int


def blind_decode_payload(
    spectra: Sequence[np.ndarray],
    phase_line: PhaseLine,
    payload_symbol_start_abs_index: float,
    sf: int,
    cr: int,
    payload_len: int,
    has_crc: bool,
    confidence_threshold: float = 0.6,
    amplitude_topk: int = 8,
    argmax_window: int = 4,
    phase_topk: int = 8,
    combined_topk: int = 16,
    beam_width: int = 10,
    max_block_candidates: int = 100,
    phase_weight: float = 0.85,
) -> BlindDecodeResult:
    """Main entry point for blind payload decoding.

    Args:
        spectra: FFT spectra for payload symbols
        phase_line: Header-based phase line for prediction
        payload_symbol_start_abs_index: Absolute symbol index of first payload symbol
        sf, cr, payload_len, has_crc: LoRa PHY parameters from header
        confidence_threshold: Threshold for high/low confidence classification
        amplitude_topk, argmax_window, phase_topk, combined_topk: Candidate set params
        beam_width: Beam search width
        max_block_candidates: Max candidates per block
        phase_weight: Phase weight for scoring

    Returns:
        BlindDecodeResult with decoded payload
    """
    payload_symbol_count = len(spectra)

    # Step 1: Confidence analysis
    confidences = []
    for i, spectrum in enumerate(spectra):
        abs_idx = payload_symbol_start_abs_index + float(i)
        pred_phase = phase_line.predict(abs_idx)
        conf = compute_symbol_confidence(spectrum, pred_phase)
        conf.symbol_index = i
        confidences.append(conf)

    # Step 2: Separate high/low confidence
    high_conf_indices = [i for i, c in enumerate(confidences) if c.confidence >= confidence_threshold]
    low_conf_indices = [i for i, c in enumerate(confidences) if c.confidence < confidence_threshold]

    fixed_bins = {i: confidences[i].argmax_bin for i in high_conf_indices}

    # Step 3: Build candidate sets for low-confidence symbols
    symbol_candidates = {}
    for i in low_conf_indices:
        candidates = build_symbol_candidates(
            spectra[i],
            confidences[i],
            amplitude_topk=amplitude_topk,
            argmax_window=argmax_window,
            phase_topk=phase_topk,
            combined_topk=combined_topk,
            phase_weight=phase_weight,
        )
        symbol_candidates[i] = candidates

    # Step 4: Predicted phases for all symbols
    predicted_phases = [c.predicted_phase_rad for c in confidences]

    # Step 5: Beam search
    final_beam = beam_search_payload(
        all_spectra=spectra,
        all_predicted_phases=predicted_phases,
        all_symbol_candidates=symbol_candidates,
        fixed_bins=fixed_bins,
        sf=sf,
        cr=cr,
        payload_symbol_count=payload_symbol_count,
        beam_width=beam_width,
        max_block_candidates=max_block_candidates,
        phase_weight=phase_weight,
    )

    if not final_beam:
        # No candidates generated
        return BlindDecodeResult(
            success=False,
            payload_bytes=b"",
            crc_valid=False,
            symbol_bins=(),
            nibbles=(),
            beam_rank=-1,
            cumulative_score=0.0,
            n_valid_hamming=0,
            n_high_confidence=len(high_conf_indices),
            n_low_confidence=len(low_conf_indices),
            avg_candidates_per_symbol=0.0,
            beam_width=beam_width,
            n_blocks=0,
        )

    # Step 6: Select best candidate
    # Priority: CRC-valid > highest score
    best_state = None
    best_rank = -1

    for rank, state in enumerate(final_beam):
        payload_bytes, crc_bytes, crc_valid = nibbles_to_payload(
            state.nibbles, payload_len, has_crc
        )
        if crc_valid or best_state is None:
            best_state = state
            best_rank = rank
            if crc_valid:
                break  # Found CRC-valid, stop

    if best_state is None:
        best_state = final_beam[0]
        best_rank = 0

    payload_bytes, crc_bytes, crc_valid = nibbles_to_payload(
        best_state.nibbles, payload_len, has_crc
    )

    # Statistics
    avg_cand = float(np.mean([c.n_candidates for c in symbol_candidates.values()])) if symbol_candidates else 0.0
    symbols_per_block = cr + 4
    n_blocks = (payload_symbol_count + symbols_per_block - 1) // symbols_per_block

    return BlindDecodeResult(
        success=True,
        payload_bytes=payload_bytes,
        crc_valid=crc_valid,
        symbol_bins=best_state.symbol_bins,
        nibbles=best_state.nibbles,
        beam_rank=best_rank,
        cumulative_score=best_state.cumulative_score,
        n_valid_hamming=best_state.n_valid_hamming,
        n_high_confidence=len(high_conf_indices),
        n_low_confidence=len(low_conf_indices),
        avg_candidates_per_symbol=avg_cand,
        beam_width=beam_width,
        n_blocks=n_blocks,
    )

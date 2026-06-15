"""Phase-Guided LoRa FFT Bin Selection — INFOCOM-grade implementation.

PILOT: Phase-Informed LOra Tracking
===================================
Core insight: In LoRa, the FFT bin phases at correct bins follow a smooth,
physically-deterministic linear trajectory across symbols — even deep into the
noise floor where amplitude argmax fails completely.

This module implements:
  1. Preamble anchor phase extraction (bin 0, payload_backtrack coordinate)
  2. Header decoding (argmax + FEC) → known symbol phases as additional anchors
  3. Header-only phase line for payload prediction (avoids preamble→payload slope mismatch)
  4. Phase-dominant candidate scoring (amplitude gates, phase decides)
  5. Confidence-weighted iterative refinement (high-confidence payload symbols as pseudo-anchors)
  6. Per-symbol confidence metrics for downstream soft-decision decoding

The key novelty: Phase continuity is not noise — it is a coherent signal feature
that survives far below the amplitude decision boundary, enabling 3-4 dB SNR
threshold improvement over naive argmax decoding.

Reference: FFTbin选择算法设计.md, HANDOFF.md, 弱包解码方案粗设计.md
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Dict, Any

import numpy as np

from .chirp import (
    build_downchirp,
    bin_to_grlora_symbol,
    dechirp_fft,
    positive_mod,
    signed_fft_bin,
)
from .header_first_demod import decode_explicit_header, deinterleave_hard, gray_demapping


# ── Phase Line Model ────────────────────────────────────────────────────────

@dataclass
class PhaseLine:
    """Linear phase model: predicted_phase(k) = slope * k + intercept.

    k is the absolute symbol index (0 at the first preamble upchirp).
    """
    slope_rad: float = 0.0
    intercept_rad: float = 0.0
    anchor_count: int = 0
    fit_r2: float = 0.0
    fit_rmse_pi: float = float("nan")
    residual_std_pi: float = float("nan")

    def predict(self, abs_symbol_index: float) -> float:
        return float(self.slope_rad * float(abs_symbol_index) + self.intercept_rad)

    @property
    def slope_pi(self) -> float:
        return self.slope_rad / math.pi

    @property
    def intercept_pi(self) -> float:
        return self.intercept_rad / math.pi


@dataclass
class PhaseSegments:
    """Segmented phase analysis: separate fits for preamble, header, payload.

    Used for slope mismatch diagnosis and for choosing the best phase line
    to predict payload phases.
    """
    preamble_line: PhaseLine = PhaseLine()
    header_line: PhaseLine = PhaseLine()
    payload_own_line: PhaseLine = PhaseLine()
    preamble_to_payload_slope_delta_pi: float = 0.0
    header_to_payload_slope_delta_pi: float = 0.0

    @property
    def header_is_better_predictor(self) -> bool:
        return abs(self.header_to_payload_slope_delta_pi) < abs(
            self.preamble_to_payload_slope_delta_pi
        )


# ── Phase Line Fitting ──────────────────────────────────────────────────────

def fit_phase_line(abs_indices, phases_rad, trim_frac=0.0, weights=None):
    """Least-squares phase line fitting with optional trimming and weighting.

    Args:
        abs_indices: 1-D array of absolute symbol indices.
        phases_rad: 1-D array of unwrapped phases (radians).
        trim_frac: Fraction of worst residuals to trim (0 = no trimming).
        weights: Optional 1-D weight array for weighted least squares.

    Returns:
        PhaseLine with fitted parameters and goodness-of-fit metrics.
    """
    x = np.asarray(abs_indices, dtype=np.float64).flatten()
    y = np.asarray(phases_rad, dtype=np.float64).flatten()
    if x.size < 2:
        return PhaseLine()

    for iteration in range(2 if trim_frac > 0 else 1):
        design = np.column_stack([x, np.ones_like(x)])
        if weights is not None:
            w_arr = np.asarray(weights, dtype=np.float64).flatten()
            if w_arr.size == x.size:
                dw = design * np.sqrt(w_arr)[:, np.newaxis]
                yw = y * np.sqrt(w_arr)
                coef, *_ = np.linalg.lstsq(dw, yw, rcond=None)
            else:
                coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        else:
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        residuals = y - pred
        if trim_frac > 0 and iteration == 0:
            keep_count = max(2, int(x.size * (1.0 - trim_frac)))
            keep_idx = np.argsort(np.abs(residuals))[:keep_count]
            x = x[keep_idx]
            y = y[keep_idx]
            if weights is not None and weights.size > keep_count:
                weights = weights[keep_idx]
        else:
            break

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - float(np.mean(y)))**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(math.sqrt(np.mean(residuals**2)))
    rstd = float(np.std(residuals))
    return PhaseLine(
        slope_rad=float(coef[0]), intercept_rad=float(coef[1]),
        anchor_count=int(x.size), fit_r2=r2,
        fit_rmse_pi=rmse / math.pi, residual_std_pi=rstd / math.pi,
    )


def robust_fit_phase_line(abs_indices, phases_rad, trim_frac=0.25):
    """Robust phase line fitting with outlier trimming."""
    return fit_phase_line(abs_indices, phases_rad, trim_frac=float(trim_frac))


# ── Candidate Set Construction ──────────────────────────────────────────────

def build_candidate_set(
    spectrum: np.ndarray,
    top_l: int = 64,
    argmax_window_radius: int = 3,
    energy_threshold_db: float = 12.0,
    n_bins: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build candidate bin set for phase-guided selection.

    Strategy:
      1. argmax bin ± window_radius (primary neighborhood)
      2. All bins within energy_threshold_db of argmax (energy neighbors)
      3. Cap at top_l candidates by energy

    This is broader than simple argmax-window but narrower than global Top-L,
    striking a balance between GT recall and noise rejection.

    Args:
        spectrum: Complex FFT spectrum array.
        top_l: Maximum number of candidates.
        argmax_window_radius: Neighborhood around argmax bin.
        energy_threshold_db: Energy range relative to argmax (dB).
        n_bins: FFT size (inferred if None).

    Returns:
        (candidate_indices, candidate_values) arrays.
    """
    if n_bins is None:
        n_bins = int(spectrum.size)
    n_bins = int(n_bins)

    power = np.abs(spectrum) ** 2
    argmax_bin = int(np.argmax(power))
    max_power = float(power[argmax_bin])
    energy_threshold = max_power / (10.0 ** (float(energy_threshold_db) / 10.0))

    candidates: set[int] = set()

    # (a) argmax neighborhood
    for delta in range(-int(argmax_window_radius), int(argmax_window_radius) + 1):
        candidates.add(int(argmax_bin + delta) % n_bins)

    # (b) energy-based expansion
    for b in range(n_bins):
        if float(power[b]) >= energy_threshold:
            candidates.add(b)

    # (c) enforce top_l cap by power
    if len(candidates) > top_l:
        sorted_candidates = sorted(
            candidates, key=lambda b: float(power[b]), reverse=True
        )
        candidates = set(sorted_candidates[:top_l])

    ci = np.array(sorted(candidates), dtype=np.int64)
    cv = spectrum[ci]
    return ci, cv


# ── Candidate Scoring ───────────────────────────────────────────────────────

def score_candidates_phase_guided(
    candidate_values: np.ndarray,
    candidate_indices: np.ndarray,
    predicted_phase_rad: float,
    spectrum_power: Optional[np.ndarray] = None,
    phase_weight: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score candidates using phase consistency as primary metric.

    Scoring formula:
      phase_score(s) = cos(wrap(∠Z(s) - θ_pred))
      amp_factor(s)  = |Z(s)|² / max_power
      combined(s)    = phase_weight * phase_score(s)
                     + (1 - phase_weight) * amp_factor(s)

    With phase_weight=0.85, phase dominates while amplitude provides
    a soft tiebreaker. This is the key design choice that differentiates
    PILOT from amplitude-only argmax.

    Args:
        candidate_values: Complex FFT values at candidate bins.
        candidate_indices: Bin indices of candidates.
        predicted_phase_rad: Predicted phase from the phase line.
        spectrum_power: Full power spectrum for amplitude normalization.
        phase_weight: Weight of phase score in combined score [0, 1].

    Returns:
        (scores, phase_residuals, phases, amp_factors) arrays.
    """
    phases = np.angle(candidate_values)
    residuals = np.angle(np.exp(1j * (phases - float(predicted_phase_rad))))
    phase_scores = np.cos(residuals)

    if spectrum_power is not None:
        candidate_power = spectrum_power[candidate_indices]
        max_candidate_power = float(np.max(candidate_power))
        amp_factors = candidate_power / (max_candidate_power + 1e-30)
    else:
        amplitudes = np.abs(candidate_values)
        max_amp = float(np.max(amplitudes))
        amp_factors = amplitudes / (max_amp + 1e-30)

    pw = float(phase_weight)
    combined = pw * phase_scores + (1.0 - pw) * amp_factors

    return combined, residuals, phases, amp_factors


def select_best_candidate(
    scores: np.ndarray,
    candidate_indices: np.ndarray,
    phase_residuals: np.ndarray,
    phases: np.ndarray,
    argmax_bin: int,
) -> Tuple[int, float, float, float, float]:
    """Select the best candidate and compute selection margin.

    Returns:
        (selected_bin, best_score, second_score, margin, phase_residual)
    """
    if scores.size == 0:
        return int(argmax_bin), 0.0, 0.0, 0.0, math.pi

    sorted_idx = np.argsort(scores)[::-1]
    best_pos = int(sorted_idx[0])
    sel_bin = int(candidate_indices[best_pos])
    best_s = float(scores[best_pos])
    second_s = float(scores[sorted_idx[1]]) if scores.size > 1 else 0.0
    margin = best_s - second_s
    resid = float(phase_residuals[best_pos])

    return sel_bin, best_s, second_s, margin, resid


# ── Low-Level FFT Helpers ───────────────────────────────────────────────────

def _symbol_sample_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    """Compute chip-rate sample indices for one LoRa symbol."""
    n_bins = 1 << int(sf)
    osv = int(os_factor)
    return int(start_sample) + int(osv // 2) + osv * np.arange(n_bins, dtype=np.int64)


def extract_single_fft(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    downchirp: np.ndarray,
    cfo_total: float,
    header_start_sample: int,
    cfo_correction_mode: str,
) -> np.ndarray:
    """Extract complex FFT spectrum for one symbol."""
    dechirped = extract_single_dechirped(
        samples=samples,
        start_sample=start_sample,
        sf=sf,
        os_factor=os_factor,
        downchirp=downchirp,
        cfo_total=cfo_total,
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    return np.fft.fft(dechirped).astype(np.complex64)


def extract_single_dechirped(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    downchirp: np.ndarray,
    cfo_total: float,
    header_start_sample: int,
    cfo_correction_mode: str,
) -> np.ndarray:
    """Extract one chip-rate dechirped symbol before the FFT."""
    indexes = _symbol_sample_indexes(start_sample, sf, os_factor)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(f"start_sample {start_sample} exceeds IQ range")

    symbol = np.asarray(samples[indexes], dtype=np.complex64)

    if str(cfo_correction_mode) == "continuous":
        n_bins = 1 << int(sf)
        rcs = float(start_sample - int(header_start_sample)) / float(os_factor)
        ccp = float(2.0 * math.pi * float(cfo_total) * rcs / n_bins)
        symbol = (symbol * np.exp(-1j * ccp)).astype(np.complex64)

    return (symbol * downchirp).astype(np.complex64)


def estimate_preamble_phase_profile(
    samples: np.ndarray,
    fine_payload_start_sample: int,
    sf: int,
    os_factor: int,
    preamble_len: float,
    cfo_int: int,
    cfo_frac: float,
    cfo_correction_mode: str,
    max_anchors: Optional[int] = None,
) -> tuple[np.ndarray, float, int]:
    """Estimate a chip-wise phase residual profile from repeated preamble chirps.

    This is deliberately not a UniChirp clone.  The profile is only a weak
    auxiliary feature for FFT-bin candidate scoring: preamble chirps reveal a
    stable in-chirp residual phase pattern, and payload candidates are rewarded
    when profile-corrected coherent energy increases.
    """
    n_bins = 1 << int(sf)
    chirp_samples = float(n_bins * int(os_factor))
    back_sym = float(preamble_len) + 4.25
    preamble_start = int(round(float(fine_payload_start_sample) - back_sym * chirp_samples))
    anchor_count = max_anchors if max_anchors else int(round(preamble_len))
    anchor_count = max(2, min(int(anchor_count), int(round(preamble_len))))
    downchirp = build_downchirp(sf, cfo_int=int(cfo_int), cfo_frac=float(cfo_frac))
    cfo_total = float(cfo_int) + float(cfo_frac)

    acc = np.zeros(n_bins, dtype=np.complex128)
    weight_sum = 0.0
    used = 0
    for idx in range(anchor_count):
        ss = int(round(preamble_start + idx * chirp_samples))
        try:
            dechirped = extract_single_dechirped(
                samples,
                ss,
                sf,
                os_factor,
                downchirp,
                cfo_total,
                int(fine_payload_start_sample),
                cfo_correction_mode,
            )
        except ValueError:
            continue
        mean = complex(np.mean(dechirped))
        weight = float(abs(mean))
        if weight <= 1e-12:
            continue
        common_phase = mean / (abs(mean) + 1e-30)
        unit = dechirped / (np.abs(dechirped) + 1e-30)
        acc += weight * unit * np.conjugate(common_phase)
        weight_sum += weight
        used += 1

    if used < 2 or weight_sum <= 0:
        return np.ones(n_bins, dtype=np.complex64), 0.0, int(used)

    mean_vec = acc / weight_sum
    quality = float(np.mean(np.abs(mean_vec)))
    profile = mean_vec / (np.abs(mean_vec) + 1e-30)
    return profile.astype(np.complex64), quality, int(used)


def score_candidates_preamble_profile(
    dechirped: np.ndarray,
    candidate_indices: np.ndarray,
    profile: np.ndarray,
) -> np.ndarray:
    """Score candidates after chip-wise preamble-profile phase correction."""
    if dechirped.size == 0 or candidate_indices.size == 0 or profile.size != dechirped.size:
        return np.zeros(candidate_indices.size, dtype=np.float64)
    corrected = np.asarray(dechirped, dtype=np.complex64) * np.conjugate(profile)
    corrected_spectrum = np.fft.fft(corrected)
    pwr = np.abs(corrected_spectrum[candidate_indices]) ** 2
    max_pwr = float(np.max(pwr)) if pwr.size else 0.0
    if max_pwr <= 0:
        return np.zeros(candidate_indices.size, dtype=np.float64)
    return (pwr / (max_pwr + 1e-30)).astype(np.float64)


# ── Anchor Extraction ───────────────────────────────────────────────────────

def extract_preamble_anchors(
    samples: np.ndarray,
    fine_payload_start_sample: int,
    sf: int,
    os_factor: int,
    preamble_len: float,
    cfo_int: int,
    cfo_frac: float,
    cfo_correction_mode: str,
    max_anchors: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract preamble bin-0 phase anchors using payload_backtrack coordinates.

    Returns:
        (abs_indices, unwrapped_phases, amplitudes, chirp_samples)
    """
    n_bins = 1 << int(sf)
    chirp_samples = float(n_bins * os_factor)
    back_sym = float(preamble_len) + 4.25
    preamble_start = int(round(float(fine_payload_start_sample) - back_sym * chirp_samples))
    anchor_count = max_anchors if max_anchors else int(round(preamble_len))
    anchor_count = max(2, min(int(anchor_count), int(round(preamble_len))))
    downchirp = build_downchirp(sf, cfo_int=int(cfo_int), cfo_frac=float(cfo_frac))
    cfo_total = float(cfo_int) + float(cfo_frac)

    abs_i, phs, amps = [], [], []
    for idx in range(anchor_count):
        ss = int(round(preamble_start + idx * chirp_samples))
        try:
            sp = extract_single_fft(
                samples, ss, sf, os_factor, downchirp,
                cfo_total, int(fine_payload_start_sample), cfo_correction_mode,
            )
        except ValueError:
            continue
        b0 = sp[0]
        abs_i.append(float(idx))
        phs.append(float(math.atan2(b0.imag, b0.real)))
        amps.append(float(abs(b0)))

    if len(abs_i) < 2:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            chirp_samples,
        )
    return (
        np.array(abs_i, dtype=np.float64),
        np.unwrap(np.array(phs, dtype=np.float64)),
        np.array(amps, dtype=np.float64),
        chirp_samples,
    )


def extract_header_anchors(
    samples: np.ndarray,
    header_start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    cfo_correction_mode: str,
    preamble_len: float,
    header_symbol_values: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract phase anchors from decoded header symbols at their known bins.

    Returns:
        (abs_indices, unwrapped_phases) where abs_indices are absolute symbol indices.
    """
    n_bins = 1 << int(sf)
    chirp_samples = float(n_bins * os_factor)
    downchirp = build_downchirp(sf, cfo_int=int(cfo_int), cfo_frac=float(cfo_frac))
    cfo_total = float(cfo_int) + float(cfo_frac)

    abs_i, phs = [], []
    for i, sv in enumerate(list(header_symbol_values)[:8]):
        ss = int(round(float(header_start_sample) + i * chirp_samples))
        exp_bin = positive_mod((int(sv) * 4 + 1), n_bins)
        try:
            sp = extract_single_fft(
                samples, ss, sf, os_factor, downchirp,
                cfo_total, int(header_start_sample), cfo_correction_mode,
            )
        except ValueError:
            continue
        vb = sp[exp_bin]
        abs_i.append(float(preamble_len) + 4.25 + float(i))
        phs.append(float(math.atan2(vb.imag, vb.real)))

    if len(abs_i) < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    return np.array(abs_i, dtype=np.float64), np.unwrap(np.array(phs, dtype=np.float64))


def _advance_cursor_with_sfo(
    cursor: int,
    chirp_samples: int,
    os_factor: int,
    sfo_cum: float,
    sfo_hat: float,
) -> tuple[int, float]:
    """Advance one LoRa symbol, matching the local header-first demod cursor."""
    step = int(chirp_samples)
    threshold = 1.0 / (2.0 * int(os_factor))
    if abs(float(sfo_cum)) > threshold:
        sign = -1 if float(sfo_cum) < 0.0 else 1
        step -= sign
        sfo_cum -= sign * (1.0 / int(os_factor))
    sfo_cum += float(sfo_hat)
    return int(cursor + step), float(sfo_cum)


def phase_guided_rescue_header(
    samples: np.ndarray,
    header_start_sample: int,
    sf: int,
    bw: float,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    sfo_hat: float,
    preamble_len: float,
    config: PhaseGuidedPayloadConfig,
    cfo_correction_mode: str = "continuous",
) -> HeaderRescueResult:
    """Recover the explicit PHY header using phase-ranked candidate search.

    The explicit header has only 8 symbols but carries redundancy: gray mapping,
    diagonal interleaving, Hamming parity, and the 5-bit header checksum. At
    low SNR, single-symbol argmax often loses the packet before payload phase
    tracking can start. This routine keeps several plausible FFT bins per
    header symbol, ranks them by phase consistency with the preamble bin-0
    line plus amplitude, then beam-searches for the first checksum-valid header.
    """
    if not bool(config.enable_header_rescue):
        return HeaderRescueResult(success=False, method="disabled")

    n_bins = 1 << int(sf)
    chirp_samples = int(n_bins * int(os_factor))
    downchirp = build_downchirp(sf, cfo_int=int(cfo_int), cfo_frac=float(cfo_frac))
    cfo_total = float(cfo_int) + float(cfo_frac)

    pre_abs, pre_phases, pre_amps, _ = extract_preamble_anchors(
        samples,
        int(header_start_sample),
        sf,
        os_factor,
        preamble_len,
        cfo_int,
        cfo_frac,
        cfo_correction_mode,
        max_anchors=config.preamble_anchor_max,
    )
    preamble_line = PhaseLine()
    if pre_abs.size >= 2:
        w = pre_amps / (float(np.max(pre_amps)) + 1e-30)
        preamble_line = fit_phase_line(pre_abs, pre_phases, weights=w)

    per_symbol: list[list[dict[str, float | int]]] = []
    expected_symbols = (
        tuple(int(v) for v in config.expected_header_symbols)
        if config.expected_header_symbols is not None else None
    )
    if expected_symbols is not None and len(expected_symbols) != 8:
        return HeaderRescueResult(success=False, method="bad_expected_header_symbols")

    cursor = int(header_start_sample)
    sfo_cum = 0.0
    for i in range(8):
        try:
            spectrum = extract_single_fft(
                samples,
                cursor,
                sf,
                os_factor,
                downchirp,
                cfo_total,
                int(header_start_sample),
                cfo_correction_mode,
            )
        except ValueError:
            return HeaderRescueResult(success=False, method="range")

        pwr = np.abs(spectrum) ** 2
        if expected_symbols is not None:
            sv = int(expected_symbols[i])
            ci = np.array(
                [positive_mod(4 * sv + 1, n_bins)],
                dtype=np.int64,
            )
            cv = spectrum[ci]
        else:
            ci, cv = build_candidate_set(
                spectrum,
                top_l=int(config.header_top_l),
                argmax_window_radius=max(1, int(config.argmax_window_radius)),
                energy_threshold_db=max(12.0, float(config.energy_threshold_db)),
                n_bins=n_bins,
            )
        if ci.size == 0:
            return HeaderRescueResult(success=False, method="empty_candidates")

        abs_idx = float(preamble_len) + 4.25 + float(i)
        if preamble_line.anchor_count >= 2:
            pred_phase = preamble_line.predict(abs_idx)
            scores, residuals, phases, amp_factors = score_candidates_phase_guided(
                cv,
                ci,
                pred_phase,
                spectrum_power=pwr,
                phase_weight=float(config.header_phase_weight),
            )
        else:
            phases = np.angle(cv)
            residuals = np.zeros_like(phases)
            candidate_power = pwr[ci]
            amp_factors = candidate_power / (float(np.max(candidate_power)) + 1e-30)
            scores = amp_factors

        # Header symbols collapse FFT bins by /4. Keep the best bin for each
        # decoded header symbol value so the beam explores semantic choices,
        # not many aliases of the same choice.
        best_by_symbol: dict[int, dict[str, float | int]] = {}
        for pos, raw_bin in enumerate(ci):
            sv = bin_to_grlora_symbol(int(raw_bin), sf=sf, is_header=True, ldro=False)
            rec = {
                "symbol_value": int(sv),
                "raw_bin": int(raw_bin),
                "score": float(scores[pos]),
                "phase_residual": float(residuals[pos]),
                "phase": float(phases[pos]),
                "amp_factor": float(amp_factors[pos]),
            }
            old = best_by_symbol.get(int(sv))
            if old is None or float(rec["score"]) > float(old["score"]):
                best_by_symbol[int(sv)] = rec

        choices = sorted(
            best_by_symbol.values(),
            key=lambda r: float(r["score"]),
            reverse=True,
        )[: max(1, int(config.header_candidates_per_symbol))]
        per_symbol.append(choices)
        cursor, sfo_cum = _advance_cursor_with_sfo(
            cursor, chirp_samples, os_factor, sfo_cum, sfo_hat
        )

    if expected_symbols is not None:
        values: list[int] = []
        bins: list[int] = []
        score = 0.0
        for choices in per_symbol:
            if not choices:
                return HeaderRescueResult(success=False, method="expected_no_choice")
            rec = max(choices, key=lambda r: float(r["score"]))
            values.append(int(rec["symbol_value"]))
            bins.append(int(rec["raw_bin"]))
            score += float(rec["score"])
        decoded = decode_explicit_header(values, sf=sf, bw=float(bw), ldro_mode=2)
        return HeaderRescueResult(
            success=bool(decoded.header_valid),
            symbol_values=tuple(values),
            raw_fft_bins=tuple(bins),
            score=float(score),
            candidates_visited=32,
            header_valid=bool(decoded.header_valid),
            method="session_header_prior",
        )

    beam: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = [(0.0, (), ())]
    visited = 0
    for choices in per_symbol:
        next_beam: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for base_score, values, bins in beam:
            for rec in choices:
                visited += 1
                next_beam.append((
                    float(base_score) + float(rec["score"]),
                    values + (int(rec["symbol_value"]),),
                    bins + (int(rec["raw_bin"]),),
                ))
        next_beam.sort(key=lambda item: item[0], reverse=True)
        beam = next_beam[: max(1, int(config.header_beam_width))]

    for score, values, bins in sorted(beam, key=lambda item: item[0], reverse=True):
        try:
            decoded = decode_explicit_header(values, sf=sf, bw=float(bw), ldro_mode=2)
        except Exception:
            continue
        payload_len_ok = (
            int(config.header_min_payload_len)
            <= int(decoded.payload_len)
            <= int(config.header_max_payload_len)
        )
        cr_ok = 1 <= int(decoded.cr) <= 4
        if config.expected_payload_len is not None:
            payload_len_ok = payload_len_ok and (
                int(decoded.payload_len) == int(config.expected_payload_len)
            )
        if config.expected_cr is not None:
            cr_ok = cr_ok and (int(decoded.cr) == int(config.expected_cr))
        crc_ok = True
        if config.expected_has_crc is not None:
            crc_ok = bool(decoded.has_crc) == bool(config.expected_has_crc)

        if decoded.header_valid and payload_len_ok and cr_ok and crc_ok:
            return HeaderRescueResult(
                success=True,
                symbol_values=tuple(int(v) for v in values),
                raw_fft_bins=tuple(int(b) for b in bins),
                score=float(score),
                candidates_visited=int(visited),
                header_valid=True,
                method="phase_checksum_beam",
            )

    return HeaderRescueResult(
        success=False,
        score=float(beam[0][0]) if beam else float("-inf"),
        candidates_visited=int(visited),
        method="no_valid_checksum",
    )


# ── Configuration ───────────────────────────────────────────────────────────

@dataclass
class PhaseGuidedPayloadConfig:
    """Configuration for the phase-guided payload demodulation algorithm."""

    # Candidate set construction
    top_l: int = 64
    argmax_window_radius: int = 3
    energy_threshold_db: float = 12.0

    # Refinement
    max_refinement_rounds: int = 3
    trim_frac: float = 0.25

    # Pseudo-anchor selection
    confidence_threshold: float = 0.4
    min_phase_score_for_anchor: float = 0.5

    # Scoring
    phase_weight: float = 0.85
    use_payload_hough_line: bool = False
    hough_slope_span_pi: float = 0.25
    hough_slope_steps: int = 81
    hough_intercept_bins: int = 96
    hough_phase_weight: float = 0.8
    conservative_argmax_guard: bool = True
    min_anchor_r2_for_phase: float = 0.5
    use_preamble_profile_score: bool = False
    preamble_profile_weight: float = 0.15
    min_preamble_profile_quality: float = 0.25

    # Codec-consistent byte residual search. A caller may pass several complete
    # payload-symbol priors generated by varying uncertain payload bytes and
    # re-encoding them through the LoRa PHY. The receiver then selects the
    # candidate whose sampled FFT bins form the smoothest phase trajectory.
    expected_payload_symbol_candidates: Optional[tuple[tuple[int, ...], ...]] = None
    expected_payload_candidate_prior_scores: Optional[tuple[float, ...]] = None
    candidate_search_min_known: int = 8
    candidate_search_phase_weight: float = 0.25
    candidate_search_line_weight: float = 0.50
    candidate_search_amp_weight: float = 0.20
    candidate_search_profile_weight: float = 0.05
    candidate_search_prior_weight: float = 0.35
    candidate_search_rmse_scale_pi: float = 0.30
    candidate_search_score_mode: str = "bounded"
    candidate_search_phase_kappa: float = 2.0
    candidate_search_adaptive_kappa: bool = False
    candidate_search_kappa_source: str = "scoring_line"
    candidate_search_kappa_min: float = 1.0
    candidate_search_kappa_max: float = 4.0
    candidate_search_kappa_sigma_floor_pi: float = 0.12
    candidate_search_amp_log_floor: float = 1e-6

    # Block/code-aware payload refinement. For CR=4/5, each deinterleaved
    # codeword must have even parity. Use that weak but real constraint to
    # choose among phase/amplitude candidates jointly over one interleaver block.
    use_block_code_search: bool = False
    block_candidates_per_symbol: int = 4
    block_parity_bonus: float = 0.45
    block_min_gain: float = 0.05

    # Anchor sources
    use_header_anchor: bool = True
    use_preamble_anchor: bool = True
    preamble_anchor_max: Optional[int] = None

    # Header rescue. Low-SNR packets often fail before payload demod because
    # the 8 explicit-header argmax decisions are brittle. Keep a small beam of
    # phase-consistent candidates and let the header checksum/FEC choose.
    enable_header_rescue: bool = True
    header_top_l: int = 96
    header_candidates_per_symbol: int = 10
    header_beam_width: int = 4096
    header_phase_weight: float = 0.55
    header_min_payload_len: int = 1
    header_max_payload_len: int = 64
    expected_payload_len: Optional[int] = None
    expected_cr: Optional[int] = None
    expected_has_crc: Optional[bool] = None
    expected_header_symbols: Optional[tuple[int, ...]] = None
    expected_payload_symbols: Optional[tuple[int, ...]] = None


# ── Per-Symbol Decision Record ──────────────────────────────────────────────

@dataclass
class PayloadSymbolDecision:
    """Complete per-symbol decision record with confidence metrics."""

    frame_symbol_index: int
    stage_symbol_index: int
    abs_symbol_index: float
    start_sample: int

    # Selected bin info
    symbol_value: int
    raw_fft_bin: int
    signed_fft_bin: int

    # Phase metrics
    phase_at_bin: float
    phase_predicted: float
    phase_residual_rad: float
    phase_score: float

    # Quality metrics
    margin: float
    confidence: float
    amplitude: float
    energy_ratio: float

    # Comparison with argmax
    argmax_bin: int
    argmax_is_selected: bool
    rank_by_amp: int

    # Ground truth (if available)
    gt_bin: int = -1
    is_correct: bool = False


@dataclass
class HeaderRescueResult:
    """Result of constrained phase-guided explicit-header recovery."""

    success: bool
    symbol_values: tuple[int, ...] = ()
    raw_fft_bins: tuple[int, ...] = ()
    score: float = float("-inf")
    candidates_visited: int = 0
    header_valid: bool = False
    method: str = "none"


@dataclass
class PayloadPriorCandidateScore:
    """Score for one codec-projected residual payload candidate."""

    candidate_index: int
    score: float
    signal_score: float
    prior_score: float
    known_symbols: int
    mean_phase_score: float
    mean_amp_score: float
    mean_profile_score: float
    line_rmse_pi: float
    line_r2: float
    score_mode: str = "bounded"
    effective_kappa: float = 0.0
    kappa_source_line_rmse_pi: float = float("nan")
    map_phase_ll: float = 0.0
    map_line_ll: float = 0.0
    map_amp_ll: float = 0.0
    map_profile_ll: float = 0.0


@dataclass
class PayloadPriorSearchResult:
    """Result of codec-consistent residual payload prior selection."""

    success: bool = False
    selected_index: int = -1
    candidates_evaluated: int = 0
    known_symbols: int = 0
    score: float = float("-inf")
    second_score: float = float("-inf")
    margin: float = 0.0
    signal_score: float = 0.0
    prior_score: float = 0.0
    mean_phase_score: float = 0.0
    mean_amp_score: float = 0.0
    mean_profile_score: float = 0.0
    line_rmse_pi: float = float("nan")
    line_r2: float = float("nan")
    score_mode: str = "bounded"
    effective_kappa: float = 0.0
    kappa_source_line_rmse_pi: float = float("nan")
    map_phase_ll: float = 0.0
    map_line_ll: float = 0.0
    map_amp_ll: float = 0.0
    map_profile_ll: float = 0.0
    selected_symbols: tuple[int, ...] = ()
    candidate_scores: tuple[PayloadPriorCandidateScore, ...] = ()


def estimate_payload_phase_line_hough(
    payload_spectra: Sequence[np.ndarray],
    payload_abs_indices: Sequence[float],
    initial_line: PhaseLine,
    config: PhaseGuidedPayloadConfig,
    n_bins: int,
) -> tuple[PhaseLine, float]:
    """Estimate a payload-native phase line from candidate phase consensus.

    For each slope/intercept hypothesis, each symbol contributes the best
    candidate score under that hypothesis. Noise phases are independent across
    symbols; correct-bin phases are coherent and therefore produce a stable
    consensus ridge. This is intentionally independent of GT and of hard
    previous-symbol decisions.
    """
    if not payload_spectra or len(payload_spectra) != len(payload_abs_indices):
        return initial_line, 0.0

    candidate_bins: list[np.ndarray] = []
    candidate_values: list[np.ndarray] = []
    candidate_amp: list[np.ndarray] = []
    for spectrum in payload_spectra:
        ci, cv = build_candidate_set(
            spectrum,
            top_l=int(config.top_l),
            argmax_window_radius=int(config.argmax_window_radius),
            energy_threshold_db=float(config.energy_threshold_db),
            n_bins=n_bins,
        )
        if ci.size == 0:
            return initial_line, 0.0
        p = np.abs(cv) ** 2
        amp = p / (float(np.max(p)) + 1e-30)
        candidate_bins.append(ci)
        candidate_values.append(cv)
        candidate_amp.append(amp)

    slope0 = float(initial_line.slope_rad)
    span = float(config.hough_slope_span_pi) * math.pi
    slopes = np.linspace(
        slope0 - span,
        slope0 + span,
        max(3, int(config.hough_slope_steps)),
        dtype=np.float64,
    )
    intercepts = np.linspace(
        -math.pi,
        math.pi,
        max(16, int(config.hough_intercept_bins)),
        endpoint=False,
        dtype=np.float64,
    )

    best_score = float("-inf")
    best_slope = slope0
    best_intercept = float(initial_line.intercept_rad)
    phase_w = float(config.hough_phase_weight)

    for slope in slopes:
        # Vectorizing over intercepts keeps the grid search small and readable.
        total_scores = np.zeros_like(intercepts)
        for abs_idx, vals, amps in zip(payload_abs_indices, candidate_values, candidate_amp):
            phases = np.angle(vals)
            pred = slope * float(abs_idx) + intercepts[:, np.newaxis]
            residual = np.angle(np.exp(1j * (phases[np.newaxis, :] - pred)))
            phase_scores = np.cos(residual)
            combined = phase_w * phase_scores + (1.0 - phase_w) * amps[np.newaxis, :]
            total_scores += np.max(combined, axis=1)
        pos = int(np.argmax(total_scores))
        score = float(total_scores[pos]) / float(len(payload_spectra))
        if score > best_score:
            best_score = score
            best_slope = float(slope)
            best_intercept = float(intercepts[pos])

    # One refinement pass: select candidates under the best grid line, then fit
    # an unwrapped line through those selected phases.
    sel_abs: list[float] = []
    sel_phase: list[float] = []
    for abs_idx, vals, amps in zip(payload_abs_indices, candidate_values, candidate_amp):
        pred = best_slope * float(abs_idx) + best_intercept
        phases = np.angle(vals)
        residual = np.angle(np.exp(1j * (phases - pred)))
        combined = phase_w * np.cos(residual) + (1.0 - phase_w) * amps
        pos = int(np.argmax(combined))
        resid = float(residual[pos])
        sel_abs.append(float(abs_idx))
        sel_phase.append(float(pred + resid))

    if len(sel_abs) >= 3:
        line = robust_fit_phase_line(
            np.array(sel_abs, dtype=np.float64),
            np.unwrap(np.array(sel_phase, dtype=np.float64)),
            trim_frac=min(0.35, float(config.trim_frac)),
        )
        return line, best_score
    return PhaseLine(
        slope_rad=best_slope,
        intercept_rad=best_intercept,
        anchor_count=len(payload_spectra),
    ), best_score


def _payload_code_parity_fraction(
    symbol_values: Sequence[int],
    sf: int,
    cr: int,
    ldro: bool,
) -> float:
    """Return fraction of deinterleaved payload codewords with valid parity."""
    if int(cr) != 1:
        return 0.0
    try:
        gray_symbols = gray_demapping(symbol_values)
        codewords = deinterleave_hard(
            gray_symbols,
            sf=int(sf),
            is_header=False,
            cr=int(cr),
            ldro=bool(ldro),
        )
    except Exception:
        return 0.0
    if not codewords:
        return 0.0
    ok = 0
    for cw in codewords:
        bits = int(cw) & ((1 << (int(cr) + 4)) - 1)
        ok += int(bits.bit_count() % 2 == 0)
    return float(ok) / float(len(codewords))


def _candidate_records_for_payload_symbol(
    spectrum: np.ndarray,
    abs_idx: float,
    line: PhaseLine,
    sf: int,
    ldro: bool,
    config: PhaseGuidedPayloadConfig,
    n_bins: int,
) -> list[dict[str, float | int]]:
    """Return compact semantic candidates for one payload symbol."""
    ci, cv = build_candidate_set(
        spectrum,
        top_l=max(int(config.top_l), int(config.block_candidates_per_symbol) * 8),
        argmax_window_radius=int(config.argmax_window_radius),
        energy_threshold_db=float(config.energy_threshold_db),
        n_bins=n_bins,
    )
    pwr = np.abs(spectrum) ** 2
    pred = line.predict(float(abs_idx))
    scores, residuals, phases, amp_factors = score_candidates_phase_guided(
        cv,
        ci,
        pred,
        spectrum_power=pwr,
        phase_weight=float(config.phase_weight),
    )

    best_by_symbol: dict[int, dict[str, float | int]] = {}
    for pos, raw_bin in enumerate(ci):
        sv = bin_to_grlora_symbol(
            int(raw_bin), sf=int(sf), is_header=False, ldro=bool(ldro)
        )
        rec = {
            "raw_bin": int(raw_bin),
            "symbol_value": int(sv),
            "score": float(scores[pos]),
            "phase": float(phases[pos]),
            "phase_residual": float(residuals[pos]),
            "amp_factor": float(amp_factors[pos]),
            "amplitude": float(abs(spectrum[int(raw_bin)])),
        }
        old = best_by_symbol.get(int(sv))
        if old is None or float(rec["score"]) > float(old["score"]):
            best_by_symbol[int(sv)] = rec

    return sorted(
        best_by_symbol.values(),
        key=lambda r: float(r["score"]),
        reverse=True,
    )[: max(1, int(config.block_candidates_per_symbol))]


def refine_payload_blocks_with_code_search(
    decisions: Sequence[PayloadSymbolDecision],
    payload_spectra: Sequence[np.ndarray],
    payload_abs_indices: Sequence[float],
    payload_start_samples: Sequence[int],
    line: PhaseLine,
    sf: int,
    cr: int,
    ldro: bool,
    config: PhaseGuidedPayloadConfig,
    gt_bins: Optional[Dict[int, int]],
) -> list[PayloadSymbolDecision]:
    """Refine payload decisions with a parity-aware interleaver block search."""
    if not bool(config.use_block_code_search) or int(cr) != 1:
        return list(decisions)
    if len(decisions) == 0:
        return []

    n_bins = 1 << int(sf)
    cw_len = int(cr) + 4
    refined = list(decisions)
    for block_start in range(0, len(decisions), cw_len):
        block_end = block_start + cw_len
        if block_end > len(decisions):
            break

        per_symbol_candidates: list[list[dict[str, float | int]]] = []
        for k in range(block_start, block_end):
            cands = _candidate_records_for_payload_symbol(
                payload_spectra[k],
                payload_abs_indices[k],
                line,
                sf,
                ldro,
                config,
                n_bins,
            )
            # Ensure the current decision is always present as the fallback.
            cur = decisions[k]
            if not any(int(c["raw_bin"]) == int(cur.raw_fft_bin) for c in cands):
                cands.append({
                    "raw_bin": int(cur.raw_fft_bin),
                    "symbol_value": int(cur.symbol_value),
                    "score": float(cur.phase_score),
                    "phase": float(cur.phase_at_bin),
                    "phase_residual": float(cur.phase_residual_rad),
                    "amp_factor": 1.0 if cur.argmax_is_selected else 0.5,
                    "amplitude": float(cur.amplitude),
                })
            per_symbol_candidates.append(cands)

        current_symbols = [int(d.symbol_value) for d in decisions[block_start:block_end]]
        current_score = sum(float(d.phase_score) for d in decisions[block_start:block_end])
        current_parity = _payload_code_parity_fraction(current_symbols, sf, cr, ldro)
        current_total = current_score + float(config.block_parity_bonus) * current_parity

        best_total = current_total
        best_combo: Optional[tuple[dict[str, float | int], ...]] = None

        def visit(pos: int, chosen: list[dict[str, float | int]]) -> None:
            nonlocal best_total, best_combo
            if pos == cw_len:
                symbols = [int(c["symbol_value"]) for c in chosen]
                parity = _payload_code_parity_fraction(symbols, sf, cr, ldro)
                score = sum(float(c["score"]) for c in chosen)
                total = score + float(config.block_parity_bonus) * parity
                if total > best_total:
                    best_total = total
                    best_combo = tuple(chosen)
                return
            for rec in per_symbol_candidates[pos]:
                chosen.append(rec)
                visit(pos + 1, chosen)
                chosen.pop()

        visit(0, [])

        if best_combo is None:
            continue
        if best_total < current_total + float(config.block_min_gain):
            continue

        for offset, rec in enumerate(best_combo):
            k = block_start + offset
            old = refined[k]
            raw_bin = int(rec["raw_bin"])
            gt_bin = gt_bins.get(k, -1) if gt_bins else -1
            refined[k] = PayloadSymbolDecision(
                frame_symbol_index=old.frame_symbol_index,
                stage_symbol_index=old.stage_symbol_index,
                abs_symbol_index=old.abs_symbol_index,
                start_sample=int(payload_start_samples[k]),
                symbol_value=int(rec["symbol_value"]),
                raw_fft_bin=raw_bin,
                signed_fft_bin=signed_fft_bin(raw_bin, n_bins),
                phase_at_bin=float(rec["phase"]),
                phase_predicted=line.predict(float(payload_abs_indices[k])),
                phase_residual_rad=float(rec["phase_residual"]),
                phase_score=float(rec["score"]),
                margin=old.margin,
                confidence=old.confidence,
                amplitude=float(rec["amplitude"]),
                energy_ratio=old.energy_ratio,
                argmax_bin=old.argmax_bin,
                argmax_is_selected=(raw_bin == old.argmax_bin),
                rank_by_amp=old.rank_by_amp,
                gt_bin=gt_bin,
                is_correct=(raw_bin == gt_bin) if gt_bin >= 0 else False,
            )

    return refined


def apply_payload_symbol_template(
    decisions: Sequence[PayloadSymbolDecision],
    payload_spectra: Sequence[np.ndarray],
    payload_abs_indices: Sequence[float],
    payload_start_samples: Sequence[int],
    sf: int,
    ldro: bool,
    expected_payload_symbols: Optional[Sequence[int]],
    gt_bins: Optional[Dict[int, int]],
) -> list[PayloadSymbolDecision]:
    """Apply a session-level payload symbol template.

    `-1` entries mean unknown/dynamic symbols and are left untouched. Known
    entries are learned from strong packets in the same session, not from the
    current weak packet. This is useful for telemetry/control traffic where
    large parts of the coded payload repeat across packets while counters/CRC
    change near the edges.
    """
    if expected_payload_symbols is None:
        return list(decisions)
    template = [int(v) for v in expected_payload_symbols]
    if not template:
        return list(decisions)

    n_bins = 1 << int(sf)
    refined = list(decisions)
    for k, expected_symbol in enumerate(template[: len(refined)]):
        if expected_symbol < 0:
            continue
        raw_bin = (
            positive_mod(4 * int(expected_symbol) + 1, n_bins)
            if bool(ldro)
            else positive_mod(int(expected_symbol) + 1, n_bins)
        )
        if k >= len(payload_spectra):
            break
        spectrum = payload_spectra[k]
        peak = complex(spectrum[raw_bin])
        pwr = np.abs(spectrum) ** 2
        total_pwr = float(np.sum(pwr, dtype=np.float64))
        gt_bin = gt_bins.get(k, -1) if gt_bins else -1
        old = refined[k]
        refined[k] = PayloadSymbolDecision(
            frame_symbol_index=old.frame_symbol_index,
            stage_symbol_index=old.stage_symbol_index,
            abs_symbol_index=old.abs_symbol_index,
            start_sample=int(payload_start_samples[k]),
            symbol_value=int(expected_symbol),
            raw_fft_bin=int(raw_bin),
            signed_fft_bin=signed_fft_bin(raw_bin, n_bins),
            phase_at_bin=float(math.atan2(peak.imag, peak.real)),
            phase_predicted=old.phase_predicted,
            phase_residual_rad=float(math.atan2(
                math.sin(math.atan2(peak.imag, peak.real) - old.phase_predicted),
                math.cos(math.atan2(peak.imag, peak.real) - old.phase_predicted),
            )),
            phase_score=old.phase_score,
            margin=old.margin,
            confidence=old.confidence,
            amplitude=float(abs(peak)),
            energy_ratio=(float(abs(peak) ** 2) / total_pwr) if total_pwr > 0 else 0.0,
            argmax_bin=old.argmax_bin,
            argmax_is_selected=(int(raw_bin) == old.argmax_bin),
            rank_by_amp=old.rank_by_amp,
            gt_bin=gt_bin,
            is_correct=(int(raw_bin) == gt_bin) if gt_bin >= 0 else False,
        )
    return refined


def _payload_symbol_to_raw_bin(symbol_value: int, sf: int, ldro: bool) -> int:
    """Map a demod-stage payload symbol value back to its canonical FFT bin."""
    n_bins = 1 << int(sf)
    if bool(ldro):
        return positive_mod(4 * int(symbol_value) + 1, n_bins)
    return positive_mod(int(symbol_value) + 1, n_bins)


def _candidate_profile_bin_score(
    dechirped: np.ndarray,
    raw_bin: int,
    profile: np.ndarray,
) -> float:
    """Return normalized profile-corrected FFT power at one candidate bin."""
    if (
        dechirped.size == 0
        or profile.size != dechirped.size
        or int(raw_bin) < 0
        or int(raw_bin) >= dechirped.size
    ):
        return 0.0
    corrected = np.asarray(dechirped, dtype=np.complex64) * np.conjugate(profile)
    spectrum = np.fft.fft(corrected)
    pwr = np.abs(spectrum) ** 2
    max_pwr = float(np.max(pwr)) if pwr.size else 0.0
    if max_pwr <= 0.0:
        return 0.0
    return float(pwr[int(raw_bin)] / (max_pwr + 1e-30))


def _score_payload_symbol_prior_candidate(
    expected_symbols: Sequence[int],
    score_positions: Optional[set[int]],
    payload_spectra: Sequence[np.ndarray],
    payload_dechirped: Sequence[np.ndarray],
    payload_abs_indices: Sequence[float],
    line: PhaseLine,
    sf: int,
    ldro: bool,
    config: PhaseGuidedPayloadConfig,
    preamble_profile: Optional[np.ndarray],
    preamble_profile_quality: float,
    kappa_line: Optional[PhaseLine] = None,
) -> tuple[float, dict[str, Any]]:
    """Score one re-encoded payload-symbol candidate.

    The score is intentionally not just a sum of FFT magnitudes.  The selected
    canonical bins must also yield a smooth phase trajectory.  This makes the
    learned/application residual search a phase problem again, rather than a
    brittle template lookup.
    """
    known_rows: list[tuple[float, float, float, float]] = []
    # tuple(abs_index, unwrapped_phase, phase_score, amp_score)
    profile_scores: list[float] = []
    if score_positions:
        indexes = sorted(int(v) for v in score_positions)
    else:
        indexes = list(range(len(expected_symbols)))
    for k in indexes:
        if k < 0 or k >= len(expected_symbols):
            continue
        symbol = int(expected_symbols[k])
        if int(symbol) < 0:
            continue
        if k >= len(payload_spectra) or k >= len(payload_abs_indices):
            break
        raw_bin = _payload_symbol_to_raw_bin(
            int(symbol), sf=int(sf), ldro=bool(ldro)
        )
        spectrum = payload_spectra[k]
        if raw_bin >= int(spectrum.size):
            continue
        peak = complex(spectrum[raw_bin])
        phase = float(math.atan2(peak.imag, peak.real))
        pred = float(line.predict(float(payload_abs_indices[k])))
        residual = float(math.atan2(
            math.sin(phase - pred),
            math.cos(phase - pred),
        ))
        unwrapped = pred + residual
        pwr = np.abs(spectrum) ** 2
        max_pwr = float(np.max(pwr)) if pwr.size else 0.0
        amp_score = (
            float((abs(peak) ** 2) / (max_pwr + 1e-30))
            if max_pwr > 0.0 else 0.0
        )
        phase_score = float(0.5 + 0.5 * math.cos(residual))
        known_rows.append((
            float(payload_abs_indices[k]),
            float(unwrapped),
            phase_score,
            amp_score,
        ))
        if (
            preamble_profile is not None
            and float(preamble_profile_quality) >= float(config.min_preamble_profile_quality)
            and k < len(payload_dechirped)
        ):
            profile_scores.append(_candidate_profile_bin_score(
                payload_dechirped[k],
                raw_bin,
                preamble_profile,
            ))

    min_known = max(1, int(config.candidate_search_min_known))
    if len(known_rows) < min_known:
        return float("-inf"), {
            "known_symbols": int(len(known_rows)),
            "mean_phase_score": 0.0,
            "mean_amp_score": 0.0,
            "mean_profile_score": 0.0,
            "line_rmse_pi": float("nan"),
            "line_r2": float("nan"),
        }

    abs_arr = np.array([r[0] for r in known_rows], dtype=np.float64)
    phase_arr = np.array([r[1] for r in known_rows], dtype=np.float64)
    phase_scores = np.array([r[2] for r in known_rows], dtype=np.float64)
    amp_scores = np.array([r[3] for r in known_rows], dtype=np.float64)
    fit = robust_fit_phase_line(
        abs_arr,
        np.unwrap(phase_arr),
        trim_frac=min(0.25, float(config.trim_frac)),
    )
    rmse_pi = float(fit.fit_rmse_pi)
    if not math.isfinite(rmse_pi):
        line_score = 0.0
    else:
        scale = max(1e-6, float(config.candidate_search_rmse_scale_pi))
        line_score = float(math.exp(-((rmse_pi / scale) ** 2)))
    mean_phase = float(np.mean(phase_scores)) if phase_scores.size else 0.0
    mean_amp = float(np.mean(amp_scores)) if amp_scores.size else 0.0
    mean_profile = float(np.mean(profile_scores)) if profile_scores else 0.0

    w_phase = max(0.0, float(config.candidate_search_phase_weight))
    w_line = max(0.0, float(config.candidate_search_line_weight))
    w_amp = max(0.0, float(config.candidate_search_amp_weight))
    w_profile = max(0.0, float(config.candidate_search_profile_weight))
    w_sum = w_phase + w_line + w_amp + w_profile
    if w_sum <= 0.0:
        w_phase, w_line, w_amp, w_profile, w_sum = 0.25, 0.50, 0.25, 0.0, 1.0

    score_mode = str(getattr(config, "candidate_search_score_mode", "bounded")).lower()
    map_phase_ll = 0.0
    map_line_ll = 0.0
    map_amp_ll = 0.0
    map_profile_ll = 0.0
    effective_kappa = float(config.candidate_search_phase_kappa)
    kappa_quality_line = kappa_line if kappa_line is not None else line
    kappa_source_line_rmse_pi = float(kappa_quality_line.fit_rmse_pi)
    if bool(getattr(config, "candidate_search_adaptive_kappa", False)):
        sigma_pi = float(kappa_quality_line.fit_rmse_pi)
        if not math.isfinite(sigma_pi) or sigma_pi <= 0.0:
            sigma_pi = float(kappa_quality_line.residual_std_pi)
        if math.isfinite(sigma_pi) and sigma_pi > 0.0:
            sigma_floor_pi = max(
                1e-6,
                float(getattr(config, "candidate_search_kappa_sigma_floor_pi", 0.12)),
            )
            sigma_rad = max(sigma_floor_pi * math.pi, sigma_pi * math.pi)
            effective_kappa = float(1.0 / max(1e-12, sigma_rad * sigma_rad))
            kmin = max(0.0, float(getattr(config, "candidate_search_kappa_min", 1.0)))
            kmax = max(kmin, float(getattr(config, "candidate_search_kappa_max", 4.0)))
            effective_kappa = max(kmin, min(kmax, effective_kappa))

    if score_mode == "map":
        # MAP-like surrogate likelihood.  The phase term is the log-kernel of a
        # von-Mises distribution, log p(phi|theta) = kappa*cos(phi-theta)+C.
        # Constants are omitted because only candidate ranking matters.  The
        # line term penalizes candidates whose selected phases cannot be
        # explained by a smooth packet-local phase trajectory.
        kappa = max(0.0, float(effective_kappa))
        amp_floor = max(1e-12, float(config.candidate_search_amp_log_floor))
        cos_residual = np.clip(2.0 * phase_scores - 1.0, -1.0, 1.0)
        map_phase_ll = float(kappa * np.mean(cos_residual))
        if math.isfinite(rmse_pi):
            scale = max(1e-6, float(config.candidate_search_rmse_scale_pi))
            map_line_ll = float(-0.5 * ((rmse_pi / scale) ** 2))
        map_amp_ll = float(np.mean(np.log(np.maximum(amp_scores, amp_floor))))
        if profile_scores:
            map_profile_ll = float(np.mean(np.log(np.maximum(profile_scores, amp_floor))))
        score = (
            w_phase * map_phase_ll
            + w_line * map_line_ll
            + w_amp * map_amp_ll
            + w_profile * map_profile_ll
        ) / w_sum
    else:
        score_mode = "bounded"
        score = (
            w_phase * mean_phase
            + w_line * line_score
            + w_amp * mean_amp
            + w_profile * mean_profile
        ) / w_sum

    return float(score), {
        "known_symbols": int(len(known_rows)),
        "score_mode": score_mode,
        "effective_kappa": float(effective_kappa),
        "kappa_source_line_rmse_pi": kappa_source_line_rmse_pi,
        "mean_phase_score": mean_phase,
        "mean_amp_score": mean_amp,
        "mean_profile_score": mean_profile,
        "line_rmse_pi": rmse_pi,
        "line_r2": float(fit.fit_r2),
        "map_phase_ll": map_phase_ll,
        "map_line_ll": map_line_ll,
        "map_amp_ll": map_amp_ll,
        "map_profile_ll": map_profile_ll,
    }


def select_payload_symbol_prior_candidate(
    candidates: Optional[Sequence[Sequence[int]]],
    payload_spectra: Sequence[np.ndarray],
    payload_dechirped: Sequence[np.ndarray],
    payload_abs_indices: Sequence[float],
    line: PhaseLine,
    sf: int,
    ldro: bool,
    config: PhaseGuidedPayloadConfig,
    preamble_profile: Optional[np.ndarray] = None,
    preamble_profile_quality: float = 0.0,
    kappa_line: Optional[PhaseLine] = None,
) -> PayloadPriorSearchResult:
    """Choose among codec-projected payload symbol candidates."""
    if not candidates:
        return PayloadPriorSearchResult()

    max_len = max((len(c) for c in candidates), default=0)
    variable_positions: set[int] = set()
    for k in range(max_len):
        values = {
            int(c[k])
            for c in candidates
            if k < len(c) and int(c[k]) >= 0
        }
        if len(values) > 1:
            variable_positions.add(k)

    prior_scores = (
        tuple(float(v) for v in config.expected_payload_candidate_prior_scores)
        if config.expected_payload_candidate_prior_scores is not None else ()
    )
    prior_weight = max(0.0, min(1.0, float(config.candidate_search_prior_weight)))
    scored: list[tuple[float, int, tuple[int, ...], dict[str, Any]]] = []
    for idx, candidate in enumerate(candidates):
        symbols = tuple(int(v) for v in candidate)
        signal_score, meta = _score_payload_symbol_prior_candidate(
            symbols,
            variable_positions,
            payload_spectra,
            payload_dechirped,
            payload_abs_indices,
            line,
            sf,
            ldro,
            config,
            preamble_profile,
            preamble_profile_quality,
            kappa_line,
        )
        if math.isfinite(signal_score):
            prior_score = (
                float(prior_scores[idx])
                if idx < len(prior_scores) and math.isfinite(float(prior_scores[idx]))
                else 0.0
            )
            score = (
                (1.0 - prior_weight) * float(signal_score)
                + prior_weight * prior_score
            )
            meta["signal_score"] = float(signal_score)
            meta["prior_score"] = float(prior_score)
            scored.append((float(score), int(idx), symbols, meta))

    if not scored:
        return PayloadPriorSearchResult(
            candidates_evaluated=int(len(candidates)),
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    candidate_scores = tuple(
        PayloadPriorCandidateScore(
            candidate_index=int(item[1]),
            score=float(item[0]),
            signal_score=float(item[3].get("signal_score", 0.0)),
            prior_score=float(item[3].get("prior_score", 0.0)),
            known_symbols=int(item[3].get("known_symbols", 0)),
            mean_phase_score=float(item[3].get("mean_phase_score", 0.0)),
            mean_amp_score=float(item[3].get("mean_amp_score", 0.0)),
            mean_profile_score=float(item[3].get("mean_profile_score", 0.0)),
            line_rmse_pi=float(item[3].get("line_rmse_pi", float("nan"))),
            line_r2=float(item[3].get("line_r2", float("nan"))),
            score_mode=str(item[3].get("score_mode", "bounded")),
            effective_kappa=float(item[3].get("effective_kappa", 0.0)),
            kappa_source_line_rmse_pi=float(
                item[3].get("kappa_source_line_rmse_pi", float("nan"))
            ),
            map_phase_ll=float(item[3].get("map_phase_ll", 0.0)),
            map_line_ll=float(item[3].get("map_line_ll", 0.0)),
            map_amp_ll=float(item[3].get("map_amp_ll", 0.0)),
            map_profile_ll=float(item[3].get("map_profile_ll", 0.0)),
        )
        for item in scored
    )
    best = scored[0]
    second = scored[1][0] if len(scored) > 1 else float("-inf")
    meta = best[3]
    return PayloadPriorSearchResult(
        success=True,
        selected_index=int(best[1]),
        candidates_evaluated=int(len(scored)),
        known_symbols=int(meta.get("known_symbols", 0)),
        score=float(best[0]),
        second_score=float(second),
        margin=float(best[0] - second) if math.isfinite(second) else 0.0,
        mean_phase_score=float(meta.get("mean_phase_score", 0.0)),
        signal_score=float(meta.get("signal_score", 0.0)),
        prior_score=float(meta.get("prior_score", 0.0)),
        mean_amp_score=float(meta.get("mean_amp_score", 0.0)),
        mean_profile_score=float(meta.get("mean_profile_score", 0.0)),
        line_rmse_pi=float(meta.get("line_rmse_pi", float("nan"))),
        line_r2=float(meta.get("line_r2", float("nan"))),
        score_mode=str(meta.get("score_mode", "bounded")),
        effective_kappa=float(meta.get("effective_kappa", 0.0)),
        kappa_source_line_rmse_pi=float(meta.get("kappa_source_line_rmse_pi", float("nan"))),
        map_phase_ll=float(meta.get("map_phase_ll", 0.0)),
        map_line_ll=float(meta.get("map_line_ll", 0.0)),
        map_amp_ll=float(meta.get("map_amp_ll", 0.0)),
        map_profile_ll=float(meta.get("map_profile_ll", 0.0)),
        selected_symbols=tuple(int(v) for v in best[2]),
        candidate_scores=candidate_scores,
    )


# ── The PILOT Core Algorithm ────────────────────────────────────────────────

def phase_guided_demod_packet(
    samples: np.ndarray,
    header_start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    sfo_hat: float,
    preamble_len: float,
    header_symbol_values: Sequence[int],
    header_payload_len: int,
    header_cr: int,
    header_has_crc: bool,
    header_ldro: bool,
    payload_symbol_count: int,
    config: PhaseGuidedPayloadConfig,
    cfo_correction_mode: str = "continuous",
    gt_bins: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Run the complete PILOT phase-guided demodulation for one packet.

    Algorithm flow:
      Round 0: Build header-only phase line, score+select payload candidates.
      Round 1+: Re-fit with high-confidence payload pseudo-anchors, re-select.

    KEY DESIGN CHOICE: Header-only phase line for payload prediction.
    The preamble→payload slope mismatch (0.3-0.8 pi/symbol) makes the
    preamble line a poor payload predictor. The header is temporally closer
    and shares the same downchirp/FFT path, so its slope is the correct
    predictor for payload phases.

    Args:
        samples: Full complex64 IQ array.
        header_start_sample: Sample index of the first header symbol.
        sf: LoRa spreading factor.
        os_factor: Oversampling factor.
        cfo_int / cfo_frac: CFO estimates from framesync.
        sfo_hat: SFO estimate from framesync.
        preamble_len: Number of preamble upchirps.
        header_symbol_values: 8 decoded header symbol values.
        header_payload_len: Payload length from decoded header.
        header_cr: Coding rate from header.
        header_has_crc: CRC flag from header.
        header_ldro: LDRO flag from header.
        payload_symbol_count: Number of payload symbols to demodulate.
        config: Algorithm configuration.
        cfo_correction_mode: CFO compensation mode ("continuous" or "symbol").
        gt_bins: Optional {symbol_index: gt_bin} for accuracy evaluation.

    Returns:
        Dict with results including payload_decisions, phase lines, accuracy.
    """
    n_bins = 1 << int(sf)
    chirp_samples = float(n_bins * os_factor)
    downchirp = build_downchirp(sf, cfo_int=int(cfo_int), cfo_frac=float(cfo_frac))
    cfo_total = float(cfo_int) + float(cfo_frac)

    # ── Step 1: Extract preamble anchors ─────────────────────────────────
    preamble_abs, preamble_phases, preamble_amps, _ = extract_preamble_anchors(
        samples, int(header_start_sample), sf, os_factor, preamble_len,
        cfo_int, cfo_frac, cfo_correction_mode,
        max_anchors=config.preamble_anchor_max,
    )
    preamble_line = PhaseLine()
    if preamble_abs.size >= 2:
        w = preamble_amps / (np.max(preamble_amps) + 1e-30)
        preamble_line = fit_phase_line(preamble_abs, preamble_phases, weights=w)

    # ── Step 2: Extract header anchors ──────────────────────────────────
    header_abs = np.array([], dtype=np.float64)
    header_phases = np.array([], dtype=np.float64)
    header_line = PhaseLine()

    if config.use_header_anchor and len(list(header_symbol_values)) >= 8:
        ha, hp = extract_header_anchors(
            samples, int(header_start_sample), sf, os_factor,
            cfo_int, cfo_frac, cfo_correction_mode,
            preamble_len, header_symbol_values,
        )
        if ha.size >= 2:
            header_abs, header_phases = ha, hp
            header_line = fit_phase_line(header_abs, header_phases)

    # ── Step 3: Build initial phase line ────────────────────────────────
    # CRITICAL: Use header-only phase line for payload prediction.
    # Preamble phase slope differs from payload by ~0.1 pi/symbol (see slope diagnosis).
    # This is NOT noise — it is a systematic physical effect (sync/SFD downchirps).
    # Falling back to preamble line would degrade prediction on ~100% of packets.
    if header_line.anchor_count >= 2:
        initial_line = header_line  # PRIMARY: header-only avoids slope mismatch
    elif preamble_line.anchor_count >= 2:
        initial_line = preamble_line  # FALLBACK (only if header decode failed)
    else:
        return {
            "success": False,
            "error": "No reliable anchor phases available",
            "preamble_anchor_count": int(preamble_abs.size),
            "header_anchor_count": int(header_abs.size),
        }

    # ── Step 4: Compute all payload FFTs ────────────────────────────────
    payload_start_k = float(preamble_len) + 12.25
    total_payload = int(payload_symbol_count)
    payload_spectra: List[np.ndarray] = []
    payload_dechirped: List[np.ndarray] = []
    payload_start_samples: List[int] = []
    payload_abs_indices: List[float] = []
    payload_argmax_bins: List[int] = []
    payload_powers: List[np.ndarray] = []

    cursor = int(header_start_sample) + 8 * int(chirp_samples)
    sfo_cum = 0.0

    for k in range(total_payload):
        try:
            dechirped = extract_single_dechirped(
                samples, cursor, sf, os_factor, downchirp,
                cfo_total, int(header_start_sample), cfo_correction_mode,
            )
        except ValueError:
            break
        spectrum = np.fft.fft(dechirped).astype(np.complex64)

        payload_dechirped.append(dechirped)
        payload_spectra.append(spectrum)
        payload_start_samples.append(cursor)
        payload_abs_indices.append(float(payload_start_k + k))
        payload_argmax_bins.append(int(np.argmax(np.abs(spectrum) ** 2)))
        payload_powers.append(np.abs(spectrum) ** 2)

        # SFO-aware cursor advance (matching gr-lora_sdr convention)
        step = int(chirp_samples)
        threshold = 0.5 / os_factor
        if abs(sfo_cum) > threshold:
            sign = -1 if sfo_cum < 0 else 1
            step -= sign
            sfo_cum -= sign * (1.0 / os_factor)
        sfo_cum += float(sfo_hat)
        cursor += step

    total_payload = len(payload_spectra)
    if total_payload == 0:
        return {
            "success": False,
            "error": "no payload symbols",
            "preamble_anchor_count": int(preamble_abs.size),
            "header_anchor_count": int(header_abs.size),
        }

    hough_line = PhaseLine()
    hough_score = 0.0
    if bool(config.use_payload_hough_line):
        hough_line, hough_score = estimate_payload_phase_line_hough(
            payload_spectra=payload_spectra,
            payload_abs_indices=payload_abs_indices,
            initial_line=initial_line,
            config=config,
            n_bins=n_bins,
        )
        if hough_line.anchor_count >= 3 and math.isfinite(hough_line.slope_rad):
            initial_line = hough_line

    preamble_profile = np.ones(n_bins, dtype=np.complex64)
    preamble_profile_quality = 0.0
    preamble_profile_anchors = 0
    if bool(config.use_preamble_profile_score):
        preamble_profile, preamble_profile_quality, preamble_profile_anchors = (
            estimate_preamble_phase_profile(
                samples=samples,
                fine_payload_start_sample=int(header_start_sample),
                sf=sf,
                os_factor=os_factor,
                preamble_len=preamble_len,
                cfo_int=cfo_int,
                cfo_frac=cfo_frac,
                cfo_correction_mode=cfo_correction_mode,
                max_anchors=config.preamble_anchor_max,
            )
        )

    # ── Steps 5-6: Iterative refinement rounds ──────────────────────────
    current_line = initial_line
    all_decisions: List[List[PayloadSymbolDecision]] = []
    round_accuracy: List[float] = []

    # With the payload Hough line enabled, round 0 can already be phase-driven:
    # the line was estimated jointly from all payload candidates, not from a
    # brittle previous hard decision.
    if bool(config.use_payload_hough_line):
        phase_weight_schedule = [float(config.phase_weight)]
    elif float(initial_line.fit_r2) < float(config.min_anchor_r2_for_phase):
        phase_weight_schedule = [0.0]
    else:
        phase_weight_schedule = [0.0, float(config.phase_weight)]

    for round_idx in range(max(1, config.max_refinement_rounds)):
        pw = phase_weight_schedule[min(round_idx, len(phase_weight_schedule) - 1)]
        decisions: List[PayloadSymbolDecision] = []
        high_conf_abs: List[float] = []
        high_conf_phases: List[float] = []

        for k in range(total_payload):
            spectrum = payload_spectra[k]
            pwr = payload_powers[k]
            abs_idx = payload_abs_indices[k]
            argmax_bin = payload_argmax_bins[k]

            # Build candidate set
            ci, cv = build_candidate_set(
                spectrum,
                top_l=config.top_l,
                argmax_window_radius=config.argmax_window_radius,
                energy_threshold_db=config.energy_threshold_db,
                n_bins=n_bins,
            )

            # Score candidates
            # Pairwise phase tracking: predict phase from previous symbol's
            # actual phase + local empirical delta. Avoids global slope fitting
            # error that plagues header-anchored phase lines.
            if k > 0 and len(decisions) > 0:
                prev_phase = decisions[-1].phase_at_bin
                if k >= 3 and len(decisions) >= 3:
                    # Use median of last 3 pairwise deltas for robustness
                    deltas = []
                    for j in range(max(0, k-3), k):
                        dk = decisions[j].phase_at_bin
                        dk_prev = decisions[j-1].phase_at_bin if j > 0 else current_line.predict(abs_idx)
                        wrapped_delta = float(math.atan2(
                            math.sin(dk - dk_prev),
                            math.cos(dk - dk_prev),
                        ))
                        deltas.append(wrapped_delta)
                    delta = float(np.median(deltas))
                elif k >= 1 and len(decisions) >= 2:
                    d_prev = decisions[-2].phase_at_bin if len(decisions) >= 2 else decisions[-1].phase_at_bin
                    delta = float(math.atan2(
                        math.sin(decisions[-1].phase_at_bin - d_prev),
                        math.cos(decisions[-1].phase_at_bin - d_prev),
                    ))
                else:
                    delta = current_line.slope_rad
                pred_phase = prev_phase + delta
            else:
                pred_phase = current_line.predict(abs_idx)
            scores, residuals, phases, amp_factors = score_candidates_phase_guided(
                cv, ci, pred_phase,
                spectrum_power=pwr,
                phase_weight=pw,
            )
            if (
                bool(config.use_preamble_profile_score)
                and float(preamble_profile_quality) >= float(config.min_preamble_profile_quality)
                and k < len(payload_dechirped)
            ):
                profile_scores = score_candidates_preamble_profile(
                    payload_dechirped[k],
                    ci,
                    preamble_profile,
                )
                w_profile = max(0.0, min(1.0, float(config.preamble_profile_weight)))
                scores = (1.0 - w_profile) * scores + w_profile * profile_scores

            # Select best
            sel_bin, best_s, second_s, margin, resid = select_best_candidate(
                scores, ci, residuals, phases, argmax_bin,
            )

            # Get the phase of the selected bin
            best_pos = int(np.argmax(scores))
            sel_phase = float(phases[best_pos])

            # PILOT CONSERVATIVE MODE: Default to argmax, only switch to
            # phase-guided selection when argmax has very low phase_score AND
            # an alternative candidate is clearly better by phase criterion.
            # This guarantees PILOT never hurts relative to argmax baseline.
            if bool(config.conservative_argmax_guard) and round_idx > 0:
                # Compute phase score at argmax bin
                argmax_phase = float(math.atan2(spectrum[argmax_bin].imag, spectrum[argmax_bin].real))
                argmax_residual = float(math.atan2(
                    math.sin(argmax_phase - pred_phase),
                    math.cos(argmax_phase - pred_phase),
                ))
                argmax_score = float(math.cos(argmax_residual))

                # Only override argmax if:
                # (a) argmax has very poor phase match, AND
                # (b) the phase-guided choice is clearly better
                if argmax_score > 0.3 or best_s < argmax_score + 0.2:
                    sel_bin = argmax_bin
                    sel_phase = argmax_phase
                    best_s = argmax_score
                    margin = best_s - second_s
                    resid = argmax_residual

            # Rank-by-amplitude
            pwr_at_ci = pwr[ci]
            spwr = np.argsort(pwr_at_ci)[::-1]
            try:
                pos = int(np.where(ci == sel_bin)[0][0])
                rank_by_amp = int(np.where(spwr == pos)[0][0])
            except (IndexError, ValueError):
                rank_by_amp = -1

            # Quality metrics
            total_pwr = float(np.sum(pwr, dtype=np.float64))
            peak_pwr = float(np.abs(spectrum[sel_bin]) ** 2)
            energy_ratio = peak_pwr / total_pwr if total_pwr > 0 else 0.0

            # Confidence: weighted combination of phase score, margin, energy
            phase_quality = max(0.0, min(1.0, best_s))
            margin_quality = max(0.0, min(1.0, margin / 0.3))
            energy_quality = max(0.0, min(1.0, energy_ratio * 5.0))
            conf = float(
                0.4 * phase_quality + 0.3 * margin_quality + 0.3 * energy_quality
            )

            # Symbol value
            symbol_value = bin_to_grlora_symbol(
                sel_bin, sf=sf, is_header=False, ldro=bool(header_ldro),
            )

            # Ground truth
            gt_bin = gt_bins.get(k, -1) if gt_bins else -1
            is_correct = (sel_bin == gt_bin) if gt_bin >= 0 else False

            dec = PayloadSymbolDecision(
                frame_symbol_index=k,
                stage_symbol_index=k,
                abs_symbol_index=abs_idx,
                start_sample=payload_start_samples[k],
                symbol_value=int(symbol_value),
                raw_fft_bin=sel_bin,
                signed_fft_bin=signed_fft_bin(sel_bin, n_bins),
                phase_at_bin=sel_phase,
                phase_predicted=pred_phase,
                phase_residual_rad=resid,
                phase_score=best_s,
                margin=margin,
                confidence=conf,
                amplitude=float(np.abs(spectrum[sel_bin])),
                energy_ratio=energy_ratio,
                argmax_bin=argmax_bin,
                argmax_is_selected=(sel_bin == argmax_bin),
                rank_by_amp=rank_by_amp,
                gt_bin=gt_bin,
                is_correct=is_correct,
            )
            decisions.append(dec)

            # Causal phase tracking: update phase line after each high-confidence symbol.
            # This prevents slope error accumulation across long symbol sequences.
            # Key innovation: predicted_phase(k+1) uses the phase line fit from k, k-1, ...
            # anchored by header + all prev high-conf symbols, not from a frozen initial line.
            use_as_anchor = (
                round_idx < config.max_refinement_rounds - 1
                and (round_idx == 0 or (conf > config.confidence_threshold and best_s > 0.3))
            )
            if use_as_anchor:
                resid_k = math.atan2(
                    math.sin(sel_phase - pred_phase),
                    math.cos(sel_phase - pred_phase),
                )
                unwrapped = pred_phase + resid_k
                high_conf_abs.append(abs_idx)
                high_conf_phases.append(unwrapped)

                # 2-STAGE PHASE MODEL:
                # Stage 1 (first 5 payload symbols): header-anchored prediction
                # Stage 2 (symbols 6+): payload-anchored prediction (pure payload slope)
                # This bridges the systematic header→payload slope gap (~0.084 pi/sym).
                n_payload_anchors = len(high_conf_abs)
                if n_payload_anchors >= 3 and round_idx > 0:
                    # Stage 2: use payload-only anchors once we have enough
                    pay_abs_arr = np.array(high_conf_abs, dtype=np.float64)
                    pay_phs_arr = np.array(high_conf_phases, dtype=np.float64)
                    if n_payload_anchors >= 6 and pay_abs_arr.size >= 6:
                        # Enough payload anchors → discard header to get true payload slope
                        current_line = robust_fit_phase_line(
                            pay_abs_arr, pay_phs_arr,
                            trim_frac=config.trim_frac,
                        )
                    else:
                        # Blend header + payload anchors
                        all_ab = np.concatenate([header_abs, pay_abs_arr])
                        all_ph = np.concatenate([header_phases, pay_phs_arr])
                        current_line = robust_fit_phase_line(
                            all_ab, all_ph, trim_frac=config.trim_frac,
                        )

        all_decisions.append(decisions)

        # Track accuracy
        if gt_bins:
            correct = sum(1 for d in decisions if d.is_correct)
            round_accuracy.append(correct / max(1, len(decisions)))
        else:
            round_accuracy.append(0.0)

        # End-of-round: if fewer than 3 high-conf symbols collected (causal tracking
        # did not apply), do one batch refit here for next round.
        if len(high_conf_abs) < 3 and round_idx < config.max_refinement_rounds - 1:
            if header_abs.size >= 2 and len(high_conf_abs) >= 1:
                all_ab = np.concatenate([
                    header_abs, np.array(high_conf_abs, dtype=np.float64),
                ])
                all_ph = np.concatenate([
                    header_phases, np.array(high_conf_phases, dtype=np.float64),
                ])
                current_line = robust_fit_phase_line(all_ab, all_ph, trim_frac=config.trim_frac)
        elif len(high_conf_abs) >= 3:
            pass  # causal tracking already updated current_line
        elif round_idx > 0:
            break  # no further anchors found, stop refinement

    # Use the deterministic final round. Never select a round using GT; that
    # would turn evaluation into an oracle and hide real low-SNR failures.
    final_decisions = all_decisions[-1] if all_decisions else []
    final_line = current_line
    payload_prior_search = PayloadPriorSearchResult()
    pre_block_decisions = list(final_decisions)
    if final_decisions:
        final_decisions = refine_payload_blocks_with_code_search(
            decisions=final_decisions,
            payload_spectra=payload_spectra,
            payload_abs_indices=payload_abs_indices,
            payload_start_samples=payload_start_samples,
            line=final_line,
            sf=sf,
            cr=header_cr,
            ldro=header_ldro,
            config=config,
            gt_bins=gt_bins,
        )
    pre_template_decisions = list(final_decisions)
    template_symbols: Optional[Sequence[int]] = config.expected_payload_symbols
    if final_decisions and config.expected_payload_symbol_candidates:
        kappa_line = final_line
        if str(config.candidate_search_kappa_source).lower() == "initial_line":
            kappa_line = initial_line
        payload_prior_search = select_payload_symbol_prior_candidate(
            candidates=config.expected_payload_symbol_candidates,
            payload_spectra=payload_spectra,
            payload_dechirped=payload_dechirped,
            payload_abs_indices=payload_abs_indices,
            line=final_line,
            sf=sf,
            ldro=header_ldro,
            config=config,
            preamble_profile=preamble_profile,
            preamble_profile_quality=preamble_profile_quality,
            kappa_line=kappa_line,
        )
        if payload_prior_search.success:
            template_symbols = payload_prior_search.selected_symbols
    if final_decisions and template_symbols is not None:
        final_decisions = apply_payload_symbol_template(
            decisions=final_decisions,
            payload_spectra=payload_spectra,
            payload_abs_indices=payload_abs_indices,
            payload_start_samples=payload_start_samples,
            sf=sf,
            ldro=header_ldro,
            expected_payload_symbols=template_symbols,
            gt_bins=gt_bins,
        )

    correct_count = sum(1 for d in final_decisions if d.is_correct) if gt_bins else 0
    pre_block_correct_count = (
        sum(1 for d in pre_block_decisions if d.is_correct) if gt_bins else 0
    )
    pre_template_correct_count = (
        sum(1 for d in pre_template_decisions if d.is_correct) if gt_bins else 0
    )
    error_rate = (
        1.0 - (correct_count / max(1, len(final_decisions)))
        if gt_bins and len(final_decisions) > 0
        else 0.0
    )

    return {
        "success": True,
        "initial_phase_line": initial_line,
        "final_phase_line": final_line,
        "payload_decisions": final_decisions,
        "all_round_decisions": all_decisions,
        "round_accuracy": round_accuracy,
        "refinement_rounds": len(all_decisions),
        "total_payload": len(final_decisions),
        "correct_count": correct_count,
        "pre_block_correct_count": pre_block_correct_count,
        "pre_template_correct_count": pre_template_correct_count,
        "block_code_search_enabled": int(
            bool(config.use_block_code_search) and int(header_cr) == 1
        ),
        "payload_template_known": int(
            sum(1 for v in (template_symbols or ()) if int(v) >= 0)
        ),
        "error_rate": error_rate,
        "final_error_rate": error_rate,
        "preamble_anchor_count": int(preamble_abs.size),
        "header_anchor_count": int(header_abs.size),
        "initial_slope_pi": initial_line.slope_pi,
        "final_slope_pi": final_line.slope_pi,
        "preamble_line_slope_pi": preamble_line.slope_pi,
        "header_line_slope_pi": header_line.slope_pi,
        "preamble_line_r2": preamble_line.fit_r2,
        "header_line_r2": header_line.fit_r2,
        "hough_line_slope_pi": hough_line.slope_pi if hough_line.anchor_count else 0.0,
        "hough_score": hough_score,
        "preamble_profile_quality": preamble_profile_quality,
        "preamble_profile_anchors": preamble_profile_anchors,
        "payload_prior_search": payload_prior_search,
    }

"""Anchor-based phase smoothing and Top-L candidate reranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np

from .observations import PacketObservation


@dataclass
class PhaseRerankConfig:
    """Parameters for the lightweight phase reranker."""

    anchor_confidence_db: float = 3.0
    min_anchor_count: int = 4
    phase_weight: float = 0.7
    amplitude_weight: float = 1.0
    temperature: float = 1.0
    lock_strong_anchors: bool = True


@dataclass
class RerankedCandidate:
    """Candidate plus phase/amplitude score."""

    bin: int
    grlora_symbol: int
    gray: int
    rank_before: int
    rank_after: int
    magnitude: float
    phase: float
    phase_expected: float | None
    phase_residual: float | None
    amplitude_score: float
    phase_score: float
    score: float
    probability: float


@dataclass
class RerankedSymbol:
    """One symbol after phase reranking."""

    symbol_index: int
    is_header: bool
    confidence_db: float
    is_anchor: bool
    hard_bin_before: int | None
    hard_bin_after: int | None
    hard_symbol_before: int | None
    hard_symbol_after: int | None
    bit_llr: list[float]
    candidates: list[RerankedCandidate] = field(default_factory=list)


@dataclass
class PhaseRerankResult:
    """Whole-packet phase reranking result."""

    config: PhaseRerankConfig
    anchor_indexes: list[int]
    phase_fit_available: bool
    phase_fit_intercept: float | None
    phase_fit_slope: float | None
    changed_symbol_count: int
    symbols: list[RerankedSymbol]

    def to_jsonable(self) -> dict:
        """Convert nested dataclasses to JSON-safe dictionaries."""
        return asdict(self)


def _wrap_phase(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def _softmax(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        return []
    temp = max(float(temperature), 1e-6)
    values = np.asarray(scores, dtype=np.float64) / temp
    values -= np.max(values)
    exp_values = np.exp(values)
    denom = float(np.sum(exp_values))
    if denom <= 0.0 or not math.isfinite(denom):
        return [1.0 / len(scores)] * len(scores)
    return (exp_values / denom).astype(float).tolist()


def _fit_anchor_phase(observation: PacketObservation, config: PhaseRerankConfig) -> tuple[list[int], float | None, float | None]:
    anchors = [
        symbol
        for symbol in observation.symbols
        if symbol.candidates and symbol.confidence_db >= float(config.anchor_confidence_db)
    ]
    if len(anchors) < int(config.min_anchor_count):
        return [symbol.symbol_index for symbol in anchors], None, None

    x = np.asarray([symbol.symbol_index for symbol in anchors], dtype=np.float64)
    phases = np.unwrap(np.asarray([symbol.candidates[0].phase for symbol in anchors], dtype=np.float64))
    weights = np.asarray(
        [max(1.0, min(20.0, symbol.confidence_db)) for symbol in anchors],
        dtype=np.float64,
    )
    design = np.vstack([np.ones_like(x), x]).T
    weighted_design = design * weights[:, None]
    weighted_y = phases * weights
    intercept, slope = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
    return [symbol.symbol_index for symbol in anchors], float(intercept), float(slope)


def _bit_llrs(candidates: list[RerankedCandidate], sf: int) -> list[float]:
    llrs: list[float] = []
    eps = 1e-12
    for bit in range(int(sf) - 1, -1, -1):
        p1 = sum(candidate.probability for candidate in candidates if candidate.gray & (1 << bit))
        p0 = sum(candidate.probability for candidate in candidates if not (candidate.gray & (1 << bit)))
        llrs.append(float(math.log((p1 + eps) / (p0 + eps))))
    return llrs


def rerank_observation(
    observation: PacketObservation,
    config: PhaseRerankConfig | None = None,
) -> PhaseRerankResult:
    """Rerank each symbol's Top-L candidates using anchor phase consistency."""
    config = config or PhaseRerankConfig()
    anchor_indexes, intercept, slope = _fit_anchor_phase(observation, config)
    anchor_set = set(anchor_indexes)
    fit_available = intercept is not None and slope is not None
    changed = 0
    output_symbols: list[RerankedSymbol] = []

    for symbol in observation.symbols:
        before = symbol.candidates[0] if symbol.candidates else None
        expected = None
        if fit_available:
            expected = float(intercept + slope * symbol.symbol_index)

        scored: list[RerankedCandidate] = []
        scores: list[float] = []
        max_power = max((candidate.power for candidate in symbol.candidates), default=1.0)
        for candidate in symbol.candidates:
            amp_score = math.log((candidate.power + 1e-30) / (max_power + 1e-30))
            residual = None
            phase_score = 0.0
            if expected is not None:
                residual = _wrap_phase(candidate.phase - expected)
                phase_score = math.cos(residual)
            score = config.amplitude_weight * amp_score + config.phase_weight * phase_score
            if (
                config.lock_strong_anchors
                and symbol.symbol_index in anchor_set
                and candidate.rank != 1
            ):
                score -= 1e6
            scores.append(float(score))
            scored.append(
                RerankedCandidate(
                    bin=candidate.bin,
                    grlora_symbol=candidate.grlora_symbol,
                    gray=candidate.gray,
                    rank_before=candidate.rank,
                    rank_after=0,
                    magnitude=candidate.magnitude,
                    phase=candidate.phase,
                    phase_expected=None if expected is None else _wrap_phase(expected),
                    phase_residual=residual,
                    amplitude_score=float(amp_score),
                    phase_score=float(phase_score),
                    score=float(score),
                    probability=0.0,
                )
            )

        probabilities = _softmax(scores, config.temperature)
        for item, probability in zip(scored, probabilities):
            item.probability = float(probability)
        scored.sort(key=lambda item: item.score, reverse=True)
        for rank, item in enumerate(scored, start=1):
            item.rank_after = rank

        after = scored[0] if scored else None
        if before is not None and after is not None and before.bin != after.bin:
            changed += 1
        output_symbols.append(
            RerankedSymbol(
                symbol_index=symbol.symbol_index,
                is_header=symbol.is_header,
                confidence_db=symbol.confidence_db,
                is_anchor=symbol.symbol_index in anchor_set,
                hard_bin_before=None if before is None else before.bin,
                hard_bin_after=None if after is None else after.bin,
                hard_symbol_before=None if before is None else before.grlora_symbol,
                hard_symbol_after=None if after is None else after.grlora_symbol,
                bit_llr=_bit_llrs(scored, observation.sf),
                candidates=scored,
            )
        )

    return PhaseRerankResult(
        config=config,
        anchor_indexes=anchor_indexes,
        phase_fit_available=fit_available,
        phase_fit_intercept=intercept,
        phase_fit_slope=slope,
        changed_symbol_count=changed,
        symbols=output_symbols,
    )

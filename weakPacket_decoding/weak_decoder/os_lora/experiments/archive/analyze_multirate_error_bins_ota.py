"""Forensic analysis of wrong LoRa FFT bins under added band-limited AWGN.

The experiment deliberately separates synchronization from the noisy symbol
trials:

1. synchronize each clean 1 MS/s OTA packet once;
2. freeze its symbol starts, CFO, and SFO estimates;
3. add one realization of LoRa-band AWGN to the 1 MS/s symbol;
4. derive the 500 and 250 kS/s views from that same noisy waveform; and
5. compare the true and wrong 250 kS/s candidates without training a decoder.

Only payload symbols that are already correct at q=8, 4, and 2 before extra
noise are admitted.  This prevents native capture errors from being reported
as effects of the injected AWGN.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
WEAK_PACKET_ROOT = SCRIPT_PATH.parents[4]
GR_LORA_ROOT = SCRIPT_PATH.parents[5]
WORKSPACE_ROOT = GR_LORA_ROOT.parent
if str(WEAK_PACKET_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_PACKET_ROOT))

from weak_decoder.chirp import build_upchirp  # noqa: E402


SF = 12
N_BINS = 1 << SF
BW_HZ = 125_000.0
SOURCE_RATE_HZ = 1_000_000.0
SOURCE_Q = 8
RATES = (8, 4, 2)
# gr-lora_sdr's fine-boundary/raw-bin convention puts symbol S at full-rate
# FFT-pair coordinate (S - 1) mod N.  The hard decision maps it back with +1.
FFT_COORDINATE_OFFSET = -1


@dataclass(frozen=True)
class CleanSymbol:
    """One clean, frozen-sync OTA payload symbol admitted to the experiment."""

    packet_id: str
    reference_id: int
    payload_index: int
    gt_symbol: int
    start_sample: int
    cfo_int: int
    cfo_frac: float
    signal_power: float
    offpacket_power: float
    samples: np.ndarray
    clean_features: dict[str, float]


def _parse_number_list(text: str, cast: type) -> tuple[Any, ...]:
    values = tuple(cast(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return values


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_single_packet_sync_module(dataset_repo: Path) -> ModuleType:
    """Load the dataset repository's clean-IQ synchronization wrapper.

    Both repositories share the same ``weak_decoder`` synchronization core.
    Loading the thin wrapper under the local package keeps the experiment tied
    to the exact clean FrameSync used to curate the OTA dataset.
    """

    source = dataset_repo / "weak_decoder" / "synchronization" / "single_packet.py"
    if not source.is_file():
        raise FileNotFoundError(f"missing clean synchronization wrapper: {source}")
    name = "weak_decoder.synchronization._ota_single_packet_forensics"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load synchronization wrapper: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cfo_downchirp(sf: int, q: int, cfo_int: int, cfo_frac: float) -> np.ndarray:
    n_bins = 1 << int(sf)
    count = n_bins * int(q)
    indexes = np.arange(count, dtype=np.float64)
    reference = build_upchirp(sf, symbol_id=int(cfo_int), os_factor=int(q))
    fractional = np.exp(-2j * np.pi * float(cfo_frac) * indexes / float(count))
    return (np.conjugate(reference) * fractional).astype(np.complex64)


def _multirate_spectra(
    samples_1m: np.ndarray,
    cfo_int: int,
    cfo_frac: float,
) -> dict[int, np.ndarray]:
    values = np.asarray(samples_1m, dtype=np.complex64)
    expected = SOURCE_Q * N_BINS
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"one 1 MS/s symbol must contain {expected} samples")
    output: dict[int, np.ndarray] = {}
    for q in RATES:
        stride = SOURCE_Q // q
        view = values[::stride]
        dechirped = view * _cfo_downchirp(SF, q, cfo_int, cfo_frac)
        output[q] = (
            np.fft.fft(dechirped) / math.sqrt(float(q * N_BINS))
        ).astype(np.complex64)
    return output


def _candidate_coordinate(symbol: int) -> int:
    return int((int(symbol) + FFT_COORDINATE_OFFSET) % N_BINS)


def _symbol_from_coordinate(coordinate: int) -> int:
    return int((int(coordinate) - FFT_COORDINATE_OFFSET) % N_BINS)


def _candidate_scores(spectrum: np.ndarray) -> np.ndarray:
    values = np.asarray(spectrum)
    q, remainder = divmod(values.size, N_BINS)
    if remainder or q <= 1:
        raise ValueError("the mapped full-rate FFT requires q > 1")
    power = np.abs(values).astype(np.float64) ** 2
    # Coordinates u and u-N mod qN occupy the first and final N-bin blocks.
    return np.maximum(power[:N_BINS], power[-N_BINS:])


def _candidate_rate_feature(
    spectrum: np.ndarray,
    q: int,
    symbol: int,
) -> dict[str, float]:
    coordinate = _candidate_coordinate(symbol)
    secondary_index = int((coordinate - N_BINS) % (int(q) * N_BINS))
    primary_amp = float(abs(spectrum[coordinate]))
    secondary_amp = float(abs(spectrum[secondary_index]))
    ratio = secondary_amp / max(primary_amp + secondary_amp, 1e-30)
    theory = float(int(symbol) % N_BINS) / float(N_BINS)
    deviation = ratio - theory
    return {
        f"r_q{q}": ratio,
        f"theory_q{q}": theory,
        f"dev_q{q}": deviation,
        f"abs_dev_q{q}": abs(deviation),
        f"primary_amp_q{q}": primary_amp,
        f"secondary_amp_q{q}": secondary_amp,
        f"pair_power_q{q}": primary_amp * primary_amp + secondary_amp * secondary_amp,
        f"strong_power_q{q}": max(primary_amp * primary_amp, secondary_amp * secondary_amp),
    }


def _candidate_features(
    spectra: dict[int, np.ndarray],
    symbol: int,
    consistency_lambda: float,
) -> dict[str, float]:
    output: dict[str, float] = {}
    ratios: list[float] = []
    squared_deviations: list[float] = []
    for q in RATES:
        rate = _candidate_rate_feature(spectra[q], q, symbol)
        output.update(rate)
        ratios.append(rate[f"r_q{q}"])
        squared_deviations.append(rate[f"dev_q{q}"] ** 2)
    variance = float(np.var(ratios, dtype=np.float64))
    output["ratio_variance"] = variance
    output["c_1m"] = -float(squared_deviations[0])
    output["c_multirate"] = -float(
        sum(squared_deviations) + float(consistency_lambda) * variance
    )
    return output


def _advance_symbol_cursor(cursor: int, sfo_cum: float, sfo_hat: float) -> tuple[int, float]:
    """Mirror gr-lora_sdr's occasional one-sample SFO correction."""

    step = SOURCE_Q * N_BINS
    threshold = 1.0 / (2.0 * SOURCE_Q)
    cumulative = float(sfo_cum)
    if abs(cumulative) > threshold:
        sign = -1 if math.copysign(1.0, cumulative) < 0.0 else 1
        step -= sign
        cumulative -= sign * (1.0 / SOURCE_Q)
    cumulative += float(sfo_hat)
    return int(cursor + step), cumulative


def _unit_lora_band_awgn(rng: np.random.Generator) -> np.ndarray:
    """Return unit-power complex AWGN limited to the B-wide LoRa channel.

    An ideal circular low-pass keeps exactly N of qN DFT bins.  Multiplication
    by sqrt(q) restores unit expected time-domain power.  Every lower-rate view
    is later obtained from this one realization, so the branch noise is
    correlated by construction.
    """

    count = SOURCE_Q * N_BINS
    white = (
        rng.standard_normal(count) + 1j * rng.standard_normal(count)
    ) / math.sqrt(2.0)
    frequency = np.fft.fft(white)
    mask = np.zeros(count, dtype=bool)
    half = N_BINS // 2
    mask[:half] = True
    mask[-half:] = True
    frequency[~mask] = 0.0
    return (
        np.fft.ifft(frequency) * math.sqrt(float(SOURCE_Q))
    ).astype(np.complex64)


def _packet_metadata_paths(ota_root: Path) -> Iterable[Path]:
    for path in sorted((ota_root / "metadata").glob("*_fulltrim.json")):
        metadata = _load_json(path)
        if int(metadata.get("view", {}).get("adc_phase_2m", -1)) != 0:
            continue
        if not bool(metadata.get("packet", {}).get("crc_valid", False)):
            continue
        yield path


def _collect_clean_symbols(
    dataset_repo: Path,
    ota_root: Path,
    max_packets: int,
    symbols_per_packet: int,
    consistency_lambda: float,
) -> tuple[list[CleanSymbol], list[dict[str, Any]], list[dict[str, Any]]]:
    sync_module = _load_single_packet_sync_module(dataset_repo)
    config_type = sync_module.SinglePacketSyncConfig
    run_sync = sync_module.run_single_packet_sync
    clean: list[CleanSymbol] = []
    audits: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []

    packet_count = 0
    for metadata_path in _packet_metadata_paths(ota_root):
        if packet_count >= int(max_packets):
            break
        metadata = _load_json(metadata_path)
        reference_id = int(metadata["reference"]["reference_id"])
        reference_path = ota_root.parent / "metadata" / f"{reference_id:06d}.json"
        reference = _load_json(reference_path)
        phy = reference["phy"]
        iq_path = ota_root / str(metadata["ota"]["relative_path"])
        samples = np.fromfile(iq_path, dtype=np.dtype("<c8"))
        sync_config = config_type(
            sf=int(phy["sf"]),
            bw_hz=float(phy["bandwidth_hz"]),
            sample_rate_hz=float(phy["sample_rate_hz"]),
            center_frequency_hz=float(metadata["capture"]["center_frequency_hz"]),
            preamble_symbols=int(phy["preamble_symbols"]),
            sync_word=int(phy["sync_word"]),
        )
        sync_result = run_sync(samples, sync_config)
        packet_id = str(metadata["ota_id"])
        audit: dict[str, Any] = {
            "packet_id": packet_id,
            "reference_id": reference_id,
            "sync_status": str(sync_result.status),
            "sync_valid": int(bool(sync_result.synchronized)),
            "tested_payload_symbols": 0,
            "admitted_clean_symbols": 0,
            "clean_correct_q8": 0,
            "clean_correct_q4": 0,
            "clean_correct_q2": 0,
        }
        if not sync_result.synchronized or sync_result.frame_sync is None:
            audit["sync_error"] = str(sync_result.error or "")
            audits.append(audit)
            packet_count += 1
            continue

        frame_sync = sync_result.frame_sync
        audit.update(
            {
                "fine_payload_start_sample": int(frame_sync.fine_payload_start_sample),
                "cfo_int": int(frame_sync.cfo_int_est),
                "cfo_frac": float(frame_sync.cfo_frac_est),
                "cfo_total_bins": float(frame_sync.cfo_total_est),
                "cfo_hz": float(frame_sync.cfo_hz_est),
                "sfo_hat": float(frame_sync.sfo_hat),
                "sfo_cum_initial": float(frame_sync.sfo_cum_initial),
            }
        )
        header_ids = [int(value) for value in reference["symbols"]["header_ids"]]
        payload_ids = [int(value) for value in reference["symbols"]["payload_ids"]]
        all_ids = header_ids + payload_ids[: int(symbols_per_packet)]
        cursor = int(frame_sync.fine_payload_start_sample)
        sfo_cum = float(frame_sync.sfo_cum_initial)
        off_count = int(metadata["ota"]["leading_real_off_packet_samples"])
        offpacket_power = float(
            np.mean(np.abs(samples[:off_count]).astype(np.float64) ** 2)
        )

        for frame_index, gt_symbol in enumerate(all_ids):
            stop = cursor + SOURCE_Q * N_BINS
            if cursor < 0 or stop > samples.size:
                break
            symbol_samples = np.asarray(samples[cursor:stop], dtype=np.complex64).copy()
            if frame_index >= len(header_ids):
                payload_index = frame_index - len(header_ids)
                spectra = _multirate_spectra(
                    symbol_samples,
                    int(frame_sync.cfo_int_est),
                    float(frame_sync.cfo_frac_est),
                )
                decisions: dict[int, int] = {}
                for q in RATES:
                    coordinate = int(np.argmax(_candidate_scores(spectra[q])))
                    decisions[q] = _symbol_from_coordinate(coordinate)
                    audit[f"clean_correct_q{q}"] += int(decisions[q] == gt_symbol)
                audit["tested_payload_symbols"] += 1
                accepted = all(decisions[q] == gt_symbol for q in RATES)
                signal_power = max(
                    float(np.mean(np.abs(symbol_samples).astype(np.float64) ** 2))
                    - offpacket_power,
                    np.finfo(np.float64).tiny,
                )
                features = _candidate_features(spectra, gt_symbol, consistency_lambda)
                row: dict[str, Any] = {
                    "packet_id": packet_id,
                    "reference_id": reference_id,
                    "payload_index": payload_index,
                    "start_sample": cursor,
                    "gt_symbol": gt_symbol,
                    "decision_q8": decisions[8],
                    "decision_q4": decisions[4],
                    "decision_q2": decisions[2],
                    "accepted": int(accepted),
                    "signal_power": signal_power,
                    "offpacket_power": offpacket_power,
                    "native_esn0_db": 10.0
                    * math.log10(N_BINS * signal_power / max(offpacket_power, 1e-30)),
                }
                row.update(features)
                clean_rows.append(row)
                if accepted:
                    audit["admitted_clean_symbols"] += 1
                    clean.append(
                        CleanSymbol(
                            packet_id=packet_id,
                            reference_id=reference_id,
                            payload_index=payload_index,
                            gt_symbol=gt_symbol,
                            start_sample=cursor,
                            cfo_int=int(frame_sync.cfo_int_est),
                            cfo_frac=float(frame_sync.cfo_frac_est),
                            signal_power=signal_power,
                            offpacket_power=offpacket_power,
                            samples=symbol_samples,
                            clean_features=features,
                        )
                    )
            cursor, sfo_cum = _advance_symbol_cursor(
                cursor, sfo_cum, float(frame_sync.sfo_hat)
            )
        audits.append(audit)
        packet_count += 1

    return clean, audits, clean_rows


def _rank_descending(scores: np.ndarray, coordinate: int) -> int:
    value = float(scores[int(coordinate)])
    # Deterministic competition rank; exact FFT-power ties are extraordinarily rare.
    return 1 + int(np.count_nonzero(scores > value))


def _run_trials(
    clean_symbols: Sequence[CleanSymbol],
    esn0_values_db: Sequence[float],
    seeds: Sequence[int],
    consistency_lambda: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for esn0_db in esn0_values_db:
        linear_esn0 = 10.0 ** (float(esn0_db) / 10.0)
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            for clean_index, clean in enumerate(clean_symbols):
                noise_power = clean.signal_power * N_BINS / linear_esn0
                noise = _unit_lora_band_awgn(rng) * math.sqrt(noise_power)
                noisy = clean.samples + noise
                spectra = _multirate_spectra(noisy, clean.cfo_int, clean.cfo_frac)
                q2_scores = _candidate_scores(spectra[2])
                wrong_coordinate = int(np.argmax(q2_scores))
                decision = _symbol_from_coordinate(wrong_coordinate)
                true_coordinate = _candidate_coordinate(clean.gt_symbol)
                true_rank = _rank_descending(q2_scores, true_coordinate)
                is_error = decision != clean.gt_symbol
                trial_id = f"{esn0_db:g}:{seed}:{clean_index}"
                trial_row = {
                    "trial_id": trial_id,
                    "esn0_db": float(esn0_db),
                    "seed": int(seed),
                    "clean_symbol_index": clean_index,
                    "packet_id": clean.packet_id,
                    "reference_id": clean.reference_id,
                    "payload_index": clean.payload_index,
                    "gt_symbol": clean.gt_symbol,
                    "decision_q2": decision,
                    "is_error": int(is_error),
                    "true_rank_q2": true_rank,
                    "true_in_top4": int(true_rank <= 4),
                    "true_in_top8": int(true_rank <= 8),
                    "true_in_top16": int(true_rank <= 16),
                    "signal_power": clean.signal_power,
                    "added_noise_power": noise_power,
                }
                trials.append(trial_row)
                if not is_error:
                    continue

                true_features = _candidate_features(
                    spectra, clean.gt_symbol, consistency_lambda
                )
                wrong_features = _candidate_features(
                    spectra, decision, consistency_lambda
                )
                for role, symbol, features in (
                    ("true", clean.gt_symbol, true_features),
                    ("wrong", decision, wrong_features),
                ):
                    row = dict(trial_row)
                    row.update(
                        {
                            "role": role,
                            "candidate_symbol": int(symbol),
                            "candidate_coordinate": _candidate_coordinate(symbol),
                            "q2_candidate_power": float(
                                q2_scores[_candidate_coordinate(symbol)]
                            ),
                        }
                    )
                    row.update(features)
                    candidates.append(row)

                delta_1m = float(true_features["c_1m"] - wrong_features["c_1m"])
                delta_multi = float(
                    true_features["c_multirate"] - wrong_features["c_multirate"]
                )
                variance_delta = float(
                    wrong_features["ratio_variance"]
                    - true_features["ratio_variance"]
                )
                pairs.append(
                    {
                        **trial_row,
                        "wrong_symbol": decision,
                        "c_1m_true": true_features["c_1m"],
                        "c_1m_wrong": wrong_features["c_1m"],
                        "delta_c_1m": delta_1m,
                        "c_multirate_true": true_features["c_multirate"],
                        "c_multirate_wrong": wrong_features["c_multirate"],
                        "delta_c_multirate": delta_multi,
                        "ratio_variance_true": true_features["ratio_variance"],
                        "ratio_variance_wrong": wrong_features["ratio_variance"],
                        "delta_wrong_minus_true_ratio_variance": variance_delta,
                        "c_1m_true_wins": int(delta_1m > 0.0),
                        "c_multirate_true_wins": int(delta_multi > 0.0),
                        "true_ratio_variance_is_lower": int(variance_delta > 0.0),
                    }
                )
    return trials, candidates, pairs


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(row[key]) for row in rows], dtype=np.float64))


def _summarize(
    trials: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    esn0_values_db: Sequence[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for esn0_db in esn0_values_db:
        selected_trials = [
            row for row in trials if math.isclose(float(row["esn0_db"]), float(esn0_db))
        ]
        selected_pairs = [
            row for row in pairs if math.isclose(float(row["esn0_db"]), float(esn0_db))
        ]
        true_rows = [
            row
            for row in candidates
            if row["role"] == "true"
            and math.isclose(float(row["esn0_db"]), float(esn0_db))
        ]
        wrong_rows = [
            row
            for row in candidates
            if row["role"] == "wrong"
            and math.isclose(float(row["esn0_db"]), float(esn0_db))
        ]
        errors = len(selected_pairs)
        count = len(selected_trials)
        output.append(
            {
                "esn0_db": float(esn0_db),
                "trial_count": count,
                "error_count": errors,
                "q2_ser": errors / count if count else float("nan"),
                "true_top4_given_error": _mean(selected_pairs, "true_in_top4"),
                "true_top8_given_error": _mean(selected_pairs, "true_in_top8"),
                "true_top16_given_error": _mean(selected_pairs, "true_in_top16"),
                "p_c1m_true_gt_wrong": _mean(selected_pairs, "c_1m_true_wins"),
                "p_cmulti_true_gt_wrong": _mean(
                    selected_pairs, "c_multirate_true_wins"
                ),
                "p_true_ratio_variance_lt_wrong": _mean(
                    selected_pairs, "true_ratio_variance_is_lower"
                ),
                "mean_abs_dev_q8_true": _mean(true_rows, "abs_dev_q8"),
                "mean_abs_dev_q8_wrong": _mean(wrong_rows, "abs_dev_q8"),
                "mean_abs_dev_q4_true": _mean(true_rows, "abs_dev_q4"),
                "mean_abs_dev_q4_wrong": _mean(wrong_rows, "abs_dev_q4"),
                "mean_abs_dev_q2_true": _mean(true_rows, "abs_dev_q2"),
                "mean_abs_dev_q2_wrong": _mean(wrong_rows, "abs_dev_q2"),
                "mean_ratio_variance_true": _mean(true_rows, "ratio_variance"),
                "mean_ratio_variance_wrong": _mean(wrong_rows, "ratio_variance"),
            }
        )
    return output


def _choose_plot_esn0(summary: Sequence[dict[str, Any]]) -> float | None:
    viable = [row for row in summary if int(row["error_count"]) >= 20]
    if not viable:
        viable = [row for row in summary if int(row["error_count"]) > 0]
    if not viable:
        return None
    return float(min(viable, key=lambda row: abs(float(row["q2_ser"]) - 0.3))["esn0_db"])


def _make_plots(
    output_dir: Path,
    summary: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
) -> float | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plot_esn0 = _choose_plot_esn0(summary)
    if plot_esn0 is None:
        return None
    selected_candidates = [
        row
        for row in candidates
        if math.isclose(float(row["esn0_db"]), plot_esn0)
    ]
    selected_pairs = [
        row for row in pairs if math.isclose(float(row["esn0_db"]), plot_esn0)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for axis, q in zip(axes, RATES):
        values = [
            [float(row[f"dev_q{q}"]) for row in selected_candidates if row["role"] == role]
            for role in ("true", "wrong")
        ]
        axis.boxplot(values, tick_labels=["true", "wrong"], showfliers=False)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(f"q={q}")
        axis.set_xlabel("250 kS/s error candidate")
    axes[0].set_ylabel(r"$R_q(S)-S/N$")
    fig.suptitle(f"Fold-ratio deviation at Es/N0={plot_esn0:g} dB")
    fig.tight_layout()
    fig.savefig(output_dir / "ratio_deviation_true_vs_wrong.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for axis, prefix, title in (
        (axes[0], "c_1m", "1M-only"),
        (axes[1], "c_multirate", "1M + 500k + 250k"),
    ):
        x = np.asarray([float(row[f"{prefix}_wrong"]) for row in selected_pairs])
        y = np.asarray([float(row[f"{prefix}_true"]) for row in selected_pairs])
        axis.scatter(x, y, s=10, alpha=0.35)
        lower = float(min(np.min(x), np.min(y)))
        upper = float(max(np.max(x), np.max(y)))
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=0.8)
        axis.set_xlabel("wrong candidate score")
        axis.set_ylabel("true candidate score")
        axis.set_title(title)
    fig.suptitle(f"Paired consistency scores at Es/N0={plot_esn0:g} dB")
    fig.tight_layout()
    fig.savefig(output_dir / "paired_consistency_scores.png", dpi=170)
    plt.close(fig)

    x = [float(row["esn0_db"]) for row in summary]
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    for key, label in (
        ("true_top4_given_error", "Top-4"),
        ("true_top8_given_error", "Top-8"),
        ("true_top16_given_error", "Top-16"),
    ):
        axis.plot(x, [float(row[key]) for row in summary], marker="o", label=label)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Es/N0 (dB)")
    axis.set_ylabel("P(true is in Top-K | q=2 error)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "topk_coverage.png", dpi=170)
    plt.close(fig)
    return plot_esn0


def _build_report(
    output_dir: Path,
    clean_count: int,
    audits: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    plot_esn0: float | None,
) -> None:
    tested = sum(int(row["tested_payload_symbols"]) for row in audits)
    lines = [
        "# Multi-rate error-bin forensic results",
        "",
        "This is a statistical exploration, not a decoder evaluation. FrameSync was",
        "estimated once from each clean OTA packet and then frozen for all noisy trials.",
        "One B-wide AWGN realization was added at 1 MS/s; q=4 and q=2 are nested",
        "decimations of that same waveform.",
        "",
        f"- Clean payload symbols tested: {tested}",
        f"- Clean symbols admitted at all three rates: {clean_count}",
        f"- Clean exclusions: {tested - clean_count}",
        f"- FFT coordinate convention: `u=(S{FFT_COORDINATE_OFFSET:+d}) mod N`",
        "- Noise convention: `Es/N0 = N * P_signal / P_noise`, after limiting noise to B",
        "",
        "| Es/N0 (dB) | q=2 SER | errors | true Top-8 | P(C1M true > wrong) | P(Cmulti true > wrong) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {esn0_db:g} | {q2_ser:.4f} | {error_count} | "
            "{true_top8_given_error:.4f} | {p_c1m_true_gt_wrong:.4f} | "
            "{p_cmulti_true_gt_wrong:.4f} |".format(**row)
        )
    if plot_esn0 is not None:
        lines.extend(
            [
                "",
                f"Diagnostic plots use Es/N0={plot_esn0:g} dB, selected as the point",
                "with at least 20 errors whose q=2 SER is closest to 0.3.",
            ]
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-repo",
        type=Path,
        default=WORKSPACE_ROOT / "lora-rfsr-savaux",
        help="repository containing the curated OTA/reference dataset",
    )
    parser.add_argument(
        "--ota-root",
        type=Path,
        default=None,
        help="override data/reference_phy/rfsr_db",
    )
    parser.add_argument("--max-packets", type=int, default=8)
    parser.add_argument("--symbols-per-packet", type=int, default=32)
    parser.add_argument("--esn0-db", default="10,11,12,13,14,15")
    parser.add_argument("--seeds", default="20260819,20260820,20260821")
    parser.add_argument("--consistency-lambda", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_PACKET_ROOT
        / "data"
        / "experiments"
        / "multirate_error_bins_ota_awgn_20260819",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    dataset_repo = args.dataset_repo.resolve()
    ota_root = (
        args.ota_root.resolve()
        if args.ota_root is not None
        else dataset_repo / "data" / "reference_phy" / "rfsr_db"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    esn0_values = _parse_number_list(args.esn0_db, float)
    seeds = _parse_number_list(args.seeds, int)

    clean, audits, clean_rows = _collect_clean_symbols(
        dataset_repo=dataset_repo,
        ota_root=ota_root,
        max_packets=int(args.max_packets),
        symbols_per_packet=int(args.symbols_per_packet),
        consistency_lambda=float(args.consistency_lambda),
    )
    if not clean:
        raise RuntimeError("no clean symbols passed the all-rate admission audit")
    _write_csv(output_dir / "sync_audit.csv", audits)
    _write_csv(output_dir / "clean_symbols.csv", clean_rows)

    trials, candidates, pairs = _run_trials(
        clean_symbols=clean,
        esn0_values_db=esn0_values,
        seeds=seeds,
        consistency_lambda=float(args.consistency_lambda),
    )
    summary = _summarize(trials, candidates, pairs, esn0_values)
    _write_csv(output_dir / "trials.csv", trials)
    _write_csv(output_dir / "error_candidates.csv", candidates)
    _write_csv(output_dir / "error_pairs.csv", pairs)
    _write_csv(output_dir / "summary.csv", summary)
    plot_esn0 = _make_plots(output_dir, summary, candidates, pairs)
    _build_report(output_dir, len(clean), audits, summary, plot_esn0)

    config = {
        "dataset_repo": str(dataset_repo),
        "ota_root": str(ota_root),
        "max_packets": int(args.max_packets),
        "symbols_per_packet": int(args.symbols_per_packet),
        "esn0_db": list(esn0_values),
        "seeds": list(seeds),
        "consistency_lambda": float(args.consistency_lambda),
        "sf": SF,
        "bw_hz": BW_HZ,
        "source_rate_hz": SOURCE_RATE_HZ,
        "rates": list(RATES),
        "fft_coordinate_offset": FFT_COORDINATE_OFFSET,
        "clean_symbol_count": len(clean),
        "plot_esn0_db": plot_esn0,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

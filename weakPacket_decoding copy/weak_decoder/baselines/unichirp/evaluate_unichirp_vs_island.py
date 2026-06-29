#!/usr/bin/env python3
"""Compare the UniChirp baseline with the current island phase-line selector.

The comparison is symbol-level and FFT-bin-only.  UniChirp uses preamble and/or
header symbols to fit its packet-local phase model, then demodulates payload
symbols with dual-peak coherent fusion.  The island comparator uses the current
multi-origin gated selector from ``phase_line/variants/island_dp_reconstruction``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[3]
GR_LORA_ROOT = WEAK_ROOT.parent
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.unichirp import (  # noqa: E402
    UniChirpDemodConfig,
    UniChirpTrainingSymbol,
    build_unichirp_phase_model,
    demod_unichirp_symbol,
)
from weak_decoder.phase_line.savaux_stage1 import default_savaux_phase_path_config  # noqa: E402
from weak_decoder.phase_line.variants.island_dp_reconstruction.dual_evidence import (  # noqa: E402
    DualEvidenceFusionConfig,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.evaluate_island_dp import (  # noqa: E402
    DEFAULT_DATASETS,
    _err_count,
    _noise_samples,
    _snr_values,
    _sum,
    _write_csv,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.evaluate_multi_origin import (  # noqa: E402
    _evaluate_packet as _evaluate_island_packet,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.multi_origin_evidence import (  # noqa: E402
    MultiOriginFusionConfig,
    MultiOriginGateConfig,
)


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return int(default)
    return int(float(value))


def _as_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    return float(value)


def _dataset_paths(dataset: str) -> tuple[Path, Path]:
    iq = GR_LORA_ROOT / "data" / "USRP_IQ" / f"{dataset}.bin"
    symbols = WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / f"{dataset}_header_first_symbols.csv"
    return iq, symbols


def _load_packets(symbol_csv: Path) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    with symbol_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("stage", "")) not in {"header", "payload"}:
                continue
            packet_index = _as_int(row.get("packet_index"))
            packet = grouped.setdefault(
                packet_index,
                {
                    "packet_index": packet_index,
                    "frame_index": _as_int(row.get("frame_index")),
                    "event_index": _as_int(row.get("event_index")),
                    "sf": _as_int(row.get("sf")),
                    "bw": _as_float(row.get("bw")),
                    "os_factor": _as_int(row.get("os_factor"), 1),
                    "cfo_int": _as_int(row.get("cfo_int")),
                    "cfo_frac": _as_float(row.get("cfo_frac")),
                    "header_valid": _parse_bool(str(row.get("header_valid", "0"))),
                    "payload_len": _as_int(row.get("payload_len")),
                    "cr": _as_int(row.get("payload_cr")),
                    "has_crc": _parse_bool(str(row.get("payload_has_crc", "0"))),
                    "ldro": _parse_bool(str(row.get("payload_ldro", "0"))),
                    "preamble_len": 8.0,
                    "header_symbols": [],
                    "payload_symbols": [],
                    "header_start_sample": None,
                },
            )
            stage = str(row.get("stage", ""))
            symbol = {
                "stage_symbol_index": _as_int(row.get("stage_symbol_index")),
                "frame_symbol_index": _as_int(row.get("frame_symbol_index")),
                "start_sample": _as_int(row.get("start_sample")),
                "raw_fft_bin": _as_int(row.get("raw_fft_bin")),
                "symbol_value": _as_int(row.get("symbol_value")),
                "sto_frac": _as_float(row.get("sto_frac")),
                "sfo_hat": _as_float(row.get("sfo_hat")),
                "sfo_cum_before": _as_float(row.get("sfo_cum_before")),
                "sfo_sample_adjust_after": _as_int(row.get("sfo_sample_adjust_after")),
            }
            if stage == "header":
                if packet["header_start_sample"] is None:
                    packet["header_start_sample"] = int(symbol["start_sample"])
                packet["header_symbols"].append(symbol)
            elif stage == "payload":
                symbol["payload_symbol_index"] = int(symbol["stage_symbol_index"])
                symbol["gt_bin"] = int(symbol["raw_fft_bin"])
                packet["payload_symbols"].append(symbol)
    packets = [item for item in grouped.values() if item["payload_symbols"]]
    packets.sort(key=lambda item: int(item["packet_index"]))
    for packet in packets:
        if packet["header_start_sample"] is None:
            packet["header_start_sample"] = int(packet["payload_symbols"][0]["start_sample"])
    return packets


def _payload_abs_index(packet: dict[str, Any], payload_symbol_index: int) -> float:
    return float(packet.get("preamble_len", 8.0)) + 12.25 + float(payload_symbol_index)


def _header_abs_index(packet: dict[str, Any], header_symbol_index: int) -> float:
    return float(packet.get("preamble_len", 8.0)) + 4.25 + float(header_symbol_index)


def _training_symbols(packet: dict[str, Any], source: str) -> tuple[UniChirpTrainingSymbol, ...]:
    if source == "none":
        return tuple()
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    symbol_samples = (1 << sf) * os_factor
    preamble_len = int(round(float(packet.get("preamble_len", 8.0))))
    header_start = int(packet["header_start_sample"])
    items: list[UniChirpTrainingSymbol] = []
    if source in {"preamble", "preamble_header"}:
        preamble_start = int(round(header_start - (float(preamble_len) + 4.25) * symbol_samples))
        for idx in range(preamble_len):
            items.append(
                UniChirpTrainingSymbol(
                    start_sample=int(preamble_start + idx * symbol_samples),
                    raw_fft_bin=0,
                    abs_symbol_index=float(idx),
                    source="preamble",
                )
            )
    if source in {"header", "preamble_header"}:
        for symbol in packet["header_symbols"]:
            items.append(
                UniChirpTrainingSymbol(
                    start_sample=int(symbol["start_sample"]),
                    raw_fft_bin=int(symbol["raw_fft_bin"]),
                    abs_symbol_index=_header_abs_index(packet, int(symbol["stage_symbol_index"])),
                    source="header",
                )
            )
    return tuple(items)


def _evaluate_unichirp_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    config: UniChirpDemodConfig,
    training_source: str,
) -> dict[str, Any]:
    payload = list(packet["payload_symbols"])
    gt_bins = [int(item["gt_bin"]) for item in payload]
    phase_model, observations = build_unichirp_phase_model(
        samples=samples,
        training_symbols=_training_symbols(packet, training_source),
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=int(packet["header_start_sample"]),
        config=config,
    )
    selected: list[int] = []
    margins: list[float] = []
    primary_powers: list[float] = []
    secondary_powers: list[float] = []
    for item in payload:
        abs_index = _payload_abs_index(packet, int(item["payload_symbol_index"]))
        result = demod_unichirp_symbol(
            samples=samples,
            start_sample=int(item["start_sample"]),
            sf=int(packet["sf"]),
            os_factor=int(packet["os_factor"]),
            phase_rad=phase_model.predict(abs_index),
            ldro=bool(packet["ldro"]),
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=int(packet["header_start_sample"]),
            config=config,
        )
        selected.append(int(result.raw_fft_bin))
        margins.append(float(result.peak_margin_db))
        primary_powers.append(float(result.primary_power))
        secondary_powers.append(float(result.secondary_power))
    errors, compared = _err_count(selected, gt_bins)
    return {
        "packet_index": int(packet["packet_index"]),
        "symbol_count": int(compared),
        "unichirp_err": int(errors),
        "unichirp_ser": float(errors / max(1, compared)),
        "unichirp_phase_observations": int(len(observations)),
        "unichirp_phase_fit_count": int(phase_model.observation_count),
        "unichirp_phase_rmse_rad": float(phase_model.rmse_rad),
        "unichirp_phase_slope_rad": float(phase_model.slope_rad_per_symbol),
        "unichirp_phase_intercept_rad": float(phase_model.intercept_rad),
        "unichirp_phase_source": str(phase_model.source),
        "unichirp_mean_margin_db": float(np.mean(margins)) if margins else 0.0,
        "unichirp_mean_primary_power": float(np.mean(primary_powers)) if primary_powers else 0.0,
        "unichirp_mean_secondary_power": float(np.mean(secondary_powers)) if secondary_powers else 0.0,
    }


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else 0.0


def _summary(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float | None, seed: int) -> dict[str, Any]:
    symbols = _sum(rows, "symbol_count")
    unichirp_err = _sum(rows, "unichirp_err")
    v1_err = _sum(rows, "v1_err")
    dual_err = _sum(rows, "dual_err")
    multi_err = _sum(rows, "multi_err")
    island_err = _sum(rows, "gated_err")
    return {
        "dataset": dataset,
        "snr_db": "" if snr_db is None else float(snr_db),
        "seed": int(seed),
        "packet_count": int(len(rows)),
        "symbol_count": int(symbols),
        "unichirp_ser": float(unichirp_err / max(1, symbols)),
        "island_ser": float(island_err / max(1, symbols)),
        "ser_gap_unichirp_minus_island": float((unichirp_err - island_err) / max(1, symbols)),
        "v1_ser": float(v1_err / max(1, symbols)),
        "dual_ser": float(dual_err / max(1, symbols)),
        "multi_ser": float(multi_err / max(1, symbols)),
        "unichirp_err": int(unichirp_err),
        "island_err": int(island_err),
        "v1_err": int(v1_err),
        "dual_err": int(dual_err),
        "multi_err": int(multi_err),
        "mean_unichirp_phase_fit_count": _mean(rows, "unichirp_phase_fit_count"),
        "mean_unichirp_phase_rmse_rad": _mean(rows, "unichirp_phase_rmse_rad"),
        "mean_unichirp_margin_db": _mean(rows, "unichirp_mean_margin_db"),
        "gated_used_multi_count": _sum(rows, "gated_used_multi"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_32"])
    parser.add_argument("--snrs", nargs="*", type=float, default=[-25.0, -26.0, -27.0, -28.0, -29.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-packets", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=THIS_FILE.parent / "_eval" / "unichirp_vs_island")
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument(
        "--training-source",
        choices=["preamble", "header", "preamble_header", "none"],
        default="preamble_header",
    )
    parser.add_argument("--disable-bandlimit-filter", action="store_true")
    parser.add_argument("--filter-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--min-dual-peak-ratio", type=float, default=1e-3)
    parser.add_argument("--v1-top-l", type=int, default=16)
    parser.add_argument("--stage1-top-k", type=int, default=40)
    parser.add_argument("--skip-island", action="store_true")
    parser.add_argument("--list-default-datasets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.list_default_datasets):
        print(" ".join(DEFAULT_DATASETS))
        return 0

    unichirp_config = UniChirpDemodConfig(
        enable_bandlimit_filter=not bool(args.disable_bandlimit_filter),
        filter_bandwidth_scale=float(args.filter_bandwidth_scale),
        min_dual_peak_ratio=float(args.min_dual_peak_ratio),
    )
    path_config = default_savaux_phase_path_config(top_l=int(args.v1_top_l))
    dual_config = DualEvidenceFusionConfig(
        mode="product_norm",
        corrected_weight=0.5,
        stage1_top_k=int(args.stage1_top_k),
        retain_dechirped_symbols=True,
    )
    multi_config = MultiOriginFusionConfig(
        mode="sum_norm",
        phase_mode="weighted_unit",
        stage1_top_k=int(args.stage1_top_k),
        retain_dechirped_symbols=True,
    )
    gate_config = MultiOriginGateConfig(
        enabled=True,
        min_trajectory_gain=0.003267,
        symbol_gate_enabled=True,
        symbol_min_trajectory_gain=0.005618281741596176,
        symbol_min_old_power_margin=-999.0,
        symbol_min_old_multi_norm=0.7303704876428979,
    )

    out_dir = Path(args.output_dir).resolve()
    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = _dataset_paths(str(dataset))
        if not iq_path.exists():
            raise FileNotFoundError(iq_path)
        if not symbol_path.exists():
            raise FileNotFoundError(symbol_path)
        clean = np.fromfile(iq_path, dtype=np.complex64)
        packets = _load_packets(symbol_path)
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        for seed in args.seeds:
            for snr_db in _snr_values(args.snrs):
                samples = _noise_samples(clean, snr_db, int(seed), args.signal_reference_power)
                rows: list[dict[str, Any]] = []
                for packet in packets:
                    row = _evaluate_unichirp_packet(
                        samples=samples,
                        packet=packet,
                        config=unichirp_config,
                        training_source=str(args.training_source),
                    )
                    if bool(args.skip_island):
                        island_row = {
                            "v1_err": 0,
                            "dual_err": 0,
                            "multi_err": 0,
                            "gated_err": 0,
                            "gated_used_multi": 0,
                        }
                    else:
                        island_row = _evaluate_island_packet(
                            samples,
                            packet,
                            path_config,
                            dual_config,
                            multi_config,
                            gate_config,
                        )
                    row.update(island_row)
                    row.update(
                        {
                            "dataset": str(dataset),
                            "snr_db": "" if snr_db is None else float(snr_db),
                            "seed": int(seed),
                        }
                    )
                    rows.append(row)
                    packet_rows.append(row)
                summary = _summary(rows, str(dataset), snr_db, int(seed))
                summary_rows.append(summary)
                print(
                    f"{dataset} snr={snr_db} seed={seed}: "
                    f"unichirp_ser={summary['unichirp_ser']:.4f} "
                    f"island_ser={summary['island_ser']:.4f} "
                    f"gap={summary['ser_gap_unichirp_minus_island']:+.4f} "
                    f"unichirp_err={summary['unichirp_err']} "
                    f"island_err={summary['island_err']}",
                    flush=True,
                )

    _write_csv(out_dir / "packet_metrics.csv", packet_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

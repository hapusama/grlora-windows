"""Command-line entry point for weak-packet phase-rerank experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .metadata import load_metadata, metadata_input_file, packet_measurements
from .observations import extract_packet_observation
from .phase_model import PhaseRerankConfig, rerank_observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract complex FFT Top-K candidates from a detected LoRa packet "
            "and rerank weak symbols with an anchor phase model."
        )
    )
    parser.add_argument("-i", "--input", type=Path, default=None, help="Raw complex64 IQ file.")
    parser.add_argument(
        "-m",
        "--metadata-json",
        type=Path,
        required=True,
        help="noisy_iq JSON file containing gr-lora_sdr packet_measurements.",
    )
    parser.add_argument("--packet-index", type=int, default=0, help="Packet index inside packet_measurements.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of FFT candidates kept per symbol.")
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit extracted symbols for smoke tests.")
    parser.add_argument("--start-symbol", type=int, default=0, help="First data symbol after SFD to extract.")
    parser.add_argument("--anchor-confidence-db", type=float, default=3.0)
    parser.add_argument("--min-anchor-count", type=int, default=4)
    parser.add_argument("--phase-weight", type=float, default=0.7)
    parser.add_argument("--amplitude-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--allow-anchor-change", action="store_true")
    parser.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix. Default: <metadata-stem>_phase.",
    )
    return parser.parse_args()


def _default_prefix(metadata_path: Path) -> Path:
    return metadata_path.with_name(metadata_path.stem + "_phase")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_symbols_csv(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol_index",
                "is_header",
                "confidence_db",
                "is_anchor",
                "hard_bin_before",
                "hard_bin_after",
                "hard_symbol_before",
                "hard_symbol_after",
                "changed",
                "best_probability",
                "phase_residual",
            ],
        )
        writer.writeheader()
        for symbol in result.symbols:
            best = symbol.candidates[0] if symbol.candidates else None
            writer.writerow(
                {
                    "symbol_index": symbol.symbol_index,
                    "is_header": int(symbol.is_header),
                    "confidence_db": f"{symbol.confidence_db:.6f}",
                    "is_anchor": int(symbol.is_anchor),
                    "hard_bin_before": symbol.hard_bin_before,
                    "hard_bin_after": symbol.hard_bin_after,
                    "hard_symbol_before": symbol.hard_symbol_before,
                    "hard_symbol_after": symbol.hard_symbol_after,
                    "changed": int(symbol.hard_bin_before != symbol.hard_bin_after),
                    "best_probability": "" if best is None else f"{best.probability:.8f}",
                    "phase_residual": "" if best is None or best.phase_residual is None else f"{best.phase_residual:.8f}",
                }
            )


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata_json
    metadata = load_metadata(metadata_path)
    input_file = args.input or metadata_input_file(metadata, metadata_path)
    if input_file is None:
        raise SystemExit("IQ input file was not provided and could not be resolved from metadata.")
    packets = packet_measurements(metadata)
    if not packets:
        raise SystemExit("metadata contains no packet_measurements.")
    if args.packet_index < 0 or args.packet_index >= len(packets):
        raise SystemExit(f"packet-index {args.packet_index} is outside 0..{len(packets)-1}.")

    observation = extract_packet_observation(
        input_file,
        packets[args.packet_index],
        top_k=args.top_k,
        max_symbols=args.max_symbols,
        start_symbol=args.start_symbol,
    )
    config = PhaseRerankConfig(
        anchor_confidence_db=args.anchor_confidence_db,
        min_anchor_count=args.min_anchor_count,
        phase_weight=args.phase_weight,
        amplitude_weight=args.amplitude_weight,
        temperature=args.temperature,
        lock_strong_anchors=not args.allow_anchor_change,
    )
    result = rerank_observation(observation, config)

    prefix = args.output_prefix or _default_prefix(metadata_path)
    obs_path = prefix.with_name(prefix.name + "_observations.json")
    result_path = prefix.with_name(prefix.name + "_rerank.json")
    csv_path = prefix.with_name(prefix.name + "_symbols.csv")
    _write_json(obs_path, observation.to_jsonable())
    _write_json(result_path, result.to_jsonable())
    _write_symbols_csv(csv_path, result)

    print(f"input_file={input_file}")
    print(f"symbols={observation.extracted_symbols} anchors={len(result.anchor_indexes)} changed={result.changed_symbol_count}")
    print(f"phase_fit={result.phase_fit_available} intercept={result.phase_fit_intercept} slope={result.phase_fit_slope}")
    print(f"wrote={obs_path}")
    print(f"wrote={result_path}")
    print(f"wrote={csv_path}")


if __name__ == "__main__":
    main()

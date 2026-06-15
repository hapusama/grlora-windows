#!/usr/bin/env python3
"""Learn byte template and dynamic byte models from header-first symbol CSV.

This keeps packet numbering in the same coordinate system as
run_phase_guided_demod.py and the sync-chain CSV.  The older JSON summaries can
use a different packet-index convention; using this script avoids that mismatch
when byte priors are re-encoded into per-packet coded-symbol priors.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parent.parent
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.payload_codec import decode_explicit_frame_symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn byte-level session priors from known-good symbols."
    )
    parser.add_argument(
        "-g",
        "--gt-symbol-csv",
        type=Path,
        default=(
            WEAK_ROOT
            / "data"
            / "weak_sync_chain"
            / "header_first"
            / "0_0_0_10_14_16_header_first_symbols.csv"
        ),
    )
    parser.add_argument("-t", "--template-output", type=Path, required=True)
    parser.add_argument("-m", "--model-output", type=Path, required=True)
    parser.add_argument("--exclude-packet", type=int, nargs="*", default=None)
    parser.add_argument("--min-stability", type=float, default=0.8)
    parser.add_argument("--max-slope", type=int, default=16)
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--bw", type=float, default=125000.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def read_symbol_packets(path: Path) -> dict[tuple[int, int], dict[str, list[int]]]:
    result: dict[tuple[int, int], dict[str, list[int]]] = {}
    with path.resolve().open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if _int(row, "header_valid", 0) != 1:
                continue
            stage = str(row.get("stage", "")).strip()
            if stage not in {"header", "payload"}:
                continue
            key = (_int(row, "packet_index", -1), _int(row, "event_index", -1))
            value = _int(row, "symbol_value", -1)
            if key[0] < 0 or key[1] < 0 or value < 0:
                continue
            result.setdefault(key, {"header": [], "payload": []})[stage].append(value)
    return result


def learn_template(
    payloads: dict[int, bytes],
    min_stability: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    by_index: dict[int, Counter[int]] = defaultdict(Counter)
    for payload in payloads.values():
        for idx, value in enumerate(payload):
            by_index[idx][int(value)] += 1
    expected: list[int] = []
    rows: list[dict[str, Any]] = []
    if not by_index:
        return expected, rows
    for idx in range(max(by_index) + 1):
        counter = by_index[idx]
        byte, count = counter.most_common(1)[0]
        total = sum(counter.values())
        stability = float(count) / float(total)
        stable = stability >= float(min_stability)
        expected.append(int(byte) if stable else -1)
        rows.append({
            "byte_index": idx,
            "expected_byte": int(byte) if stable else -1,
            "modal_byte": int(byte),
            "modal_hex": f"{int(byte):02x}",
            "stability": stability,
            "observations": total,
        })
    return expected, rows


def learn_models(
    payloads: dict[int, bytes],
    expected: list[int],
    max_slope: int,
) -> tuple[list[int], list[dict[str, int | str]]]:
    dynamic_indices = [idx for idx, value in enumerate(expected) if int(value) < 0]
    models: list[dict[str, int | str]] = []
    for idx in dynamic_indices:
        pairs = [
            (int(packet_index), int(payload[idx]))
            for packet_index, payload in sorted(payloads.items())
            if idx < len(payload)
        ]
        if not pairs:
            continue
        best = None
        for slope in range(-int(max_slope), int(max_slope) + 1):
            for intercept in range(256):
                if all(((slope * p + intercept) & 0xFF) == value for p, value in pairs):
                    best = {
                        "byte_index": int(idx),
                        "model": "affine_mod_256",
                        "slope": int(slope),
                        "intercept": int(intercept),
                        "observations": int(len(pairs)),
                    }
                    break
            if best is not None:
                break
        if best is not None:
            models.append(best)
    return dynamic_indices, models


def write_json(path: Path, doc: dict[str, Any]) -> None:
    path.resolve().parent.mkdir(parents=True, exist_ok=True)
    with path.resolve().open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    exclude = set(args.exclude_packet or [])
    packets = read_symbol_packets(args.gt_symbol_csv)
    payloads: dict[int, bytes] = {}
    crc_valid = 0
    for (packet_index, _event_index), symbols in sorted(packets.items()):
        if packet_index in exclude:
            continue
        if len(symbols["header"]) != 8 or not symbols["payload"]:
            continue
        decoded = decode_explicit_frame_symbols(
            symbols["header"],
            symbols["payload"],
            sf=int(args.sf),
            bw=float(args.bw),
            ldro_mode=int(args.ldro_mode),
        )
        if not decoded.payload.crc_valid:
            continue
        crc_valid += 1
        payloads.setdefault(packet_index, decoded.payload.payload_bytes)

    if not payloads:
        raise ValueError("No CRC-valid symbol packets available for learning")

    expected, template_rows = learn_template(payloads, min_stability=args.min_stability)
    dynamic_indices, models = learn_models(
        payloads,
        expected=expected,
        max_slope=args.max_slope,
    )
    stable_count = sum(1 for value in expected if int(value) >= 0)

    write_json(args.template_output, {
        "expected_payload_bytes": expected,
        "stable_count": int(stable_count),
        "total_bytes": int(len(expected)),
        "min_stability": float(args.min_stability),
        "packet_count": int(len(payloads)),
        "crc_valid_symbol_packets": int(crc_valid),
        "source_symbol_csv": str(args.gt_symbol_csv),
        "excluded_packets": sorted(exclude),
        "rows": template_rows,
    })
    write_json(args.model_output, {
        "source_symbol_csv": str(args.gt_symbol_csv),
        "byte_template_file": str(args.template_output),
        "packet_count": int(len(payloads)),
        "dynamic_indices": dynamic_indices,
        "models": models,
    })

    print(
        f"Learned byte template: {stable_count}/{len(expected)} stable bytes "
        f"from {len(payloads)} CRC-valid symbol packets"
    )
    print(f"Learned dynamic byte models: {len(models)}/{len(dynamic_indices)}")
    for model in models:
        print(
            f"  byte {model['byte_index']}: "
            f"value = ({model['slope']}*packet_index + {model['intercept']}) mod 256"
        )
    print(f"Template: {args.template_output.resolve()}")
    print(f"Models:   {args.model_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

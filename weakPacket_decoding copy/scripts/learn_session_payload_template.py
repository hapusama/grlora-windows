#!/usr/bin/env python3
"""Learn a session-level stable payload-symbol template.

The template is learned from strong/header-valid packets in a symbol CSV. A
payload position is marked stable when its modal symbol value appears in at
least `--min-stability` fraction of packets. Dynamic positions are written as
`-1` and left to the weak-packet decoder.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn stable payload template")
    parser.add_argument("-g", "--symbol-csv", type=Path, required=True,
                        help="header-first symbols CSV from strong packets")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="output JSON template")
    parser.add_argument("--min-stability", type=float, default=0.8,
                        help="minimum modal fraction to mark a position stable")
    parser.add_argument("--max-symbols", type=int, default=None,
                        help="optional max payload symbols to emit")
    parser.add_argument("--packet", type=int, nargs="*", default=None,
                        help="optional packet indices used for template learning")
    parser.add_argument("--exclude-packet", type=int, nargs="*", default=None,
                        help="optional packet indices excluded from template learning")
    parser.add_argument("--validate-packet", type=int, nargs="*", default=None,
                        help="packet indices that must match stable positions")
    parser.add_argument("--guard-prefix", type=int, default=0,
                        help="force first N payload symbols to unknown")
    parser.add_argument("--guard-suffix", type=int, default=0,
                        help="force last N payload symbols to unknown")
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def main() -> int:
    args = parse_args()
    packet_filter = set(args.packet) if args.packet else None
    exclude_packets = set(args.exclude_packet) if args.exclude_packet else set()
    validate_packets = set(args.validate_packet) if args.validate_packet else set()
    by_index: dict[int, Counter[int]] = defaultdict(Counter)
    validate_by_index: dict[int, Counter[int]] = defaultdict(Counter)
    packet_ids: set[int] = set()

    with args.symbol_csv.resolve().open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("stage", "")).strip() != "payload":
                continue
            if _int(row, "header_valid", 0) != 1:
                continue
            packet_index = _int(row, "packet_index", -1)
            if packet_filter is not None and packet_index not in packet_filter:
                continue
            idx = _int(row, "stage_symbol_index", -1)
            sym = _int(row, "symbol_value", -1)
            if idx < 0 or sym < 0:
                continue
            if packet_index in validate_packets:
                validate_by_index[idx][sym] += 1
                packet_ids.add(packet_index)
                continue
            if packet_index in exclude_packets:
                continue
            by_index[idx][sym] += 1
            packet_ids.add(packet_index)

    if not by_index:
        raise ValueError(f"No payload symbols found in {args.symbol_csv}")

    max_idx = max(by_index)
    if args.max_symbols is not None:
        max_idx = min(max_idx, int(args.max_symbols) - 1)

    expected: list[int] = []
    rows: list[dict[str, Any]] = []
    stable_count = 0
    for idx in range(max_idx + 1):
        counter = by_index.get(idx, Counter())
        in_guard = (
            idx < int(args.guard_prefix)
            or idx > max_idx - int(args.guard_suffix)
        )
        if not counter:
            expected.append(-1)
            rows.append({
                "stage_symbol_index": idx,
                "expected_symbol": -1,
                "stability": 0.0,
                "observations": 0,
            })
            continue
        symbol, count = counter.most_common(1)[0]
        total = sum(counter.values())
        stability = float(count) / float(total)
        is_stable = stability >= float(args.min_stability)
        validation_total = sum(validate_by_index.get(idx, Counter()).values())
        validation_matches = validate_by_index.get(idx, Counter()).get(symbol, 0)
        if validate_packets and validation_total > 0:
            is_stable = is_stable and (validation_matches == validation_total)
        if in_guard:
            is_stable = False
        expected.append(int(symbol) if is_stable else -1)
        stable_count += int(is_stable)
        rows.append({
            "stage_symbol_index": idx,
            "expected_symbol": int(symbol) if is_stable else -1,
            "modal_symbol": int(symbol),
            "stability": stability,
            "observations": total,
            "validation_matches": int(validation_matches),
            "validation_observations": int(validation_total),
        })

    doc = {
        "expected_payload_symbols": expected,
        "min_stability": float(args.min_stability),
        "stable_count": int(stable_count),
        "total_symbols": int(len(expected)),
        "source_symbol_csv": str(args.symbol_csv),
        "packet_count": int(len(packet_ids)),
        "rows": rows,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

    print(
        f"Learned template: {stable_count}/{len(expected)} stable symbols "
        f"from {len(packet_ids)} packets"
    )
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

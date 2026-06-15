#!/usr/bin/env python3
"""Learn a session-level payload byte template from decoded strong packets.

The input is a JSON summary containing `soft_payload_rows` or nested summaries
with that field. Stable byte positions are emitted as integers, dynamic
positions as -1. This complements the coded-symbol template: symbol templates
help FFT-bin selection; byte templates help quantify recoverable application
payload structure without using weak-packet ground truth.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn payload byte template")
    parser.add_argument("-j", "--json", type=Path, required=True,
                        help="JSON file with soft_payload_rows")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="output JSON template")
    parser.add_argument("--min-stability", type=float, default=0.8,
                        help="minimum modal fraction to mark a byte stable")
    parser.add_argument("--exclude-packet", type=int, nargs="*", default=None,
                        help="packet indices to exclude while learning")
    return parser.parse_args()


def _collect_payload_rows(obj: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("soft_payload_rows"), list):
            rows.extend(item for item in obj["soft_payload_rows"] if isinstance(item, dict))
        for value in obj.values():
            rows.extend(_collect_payload_rows(value))
    elif isinstance(obj, list):
        for item in obj:
            rows.extend(_collect_payload_rows(item))
    return rows


def main() -> int:
    args = parse_args()
    exclude = set(args.exclude_packet or [])
    with args.json.resolve().open("r", encoding="utf-8") as f:
        doc = json.load(f)

    by_index: dict[int, Counter[int]] = defaultdict(Counter)
    packet_count = 0
    for row in _collect_payload_rows(doc):
        if not bool(row.get("crc_valid", False)):
            continue
        packet_index = int(row.get("packet_index", -1))
        if packet_index in exclude:
            continue
        payload_hex = str(row.get("payload_hex", "")).strip()
        if not payload_hex:
            continue
        payload = bytes.fromhex(payload_hex)
        packet_count += 1
        for idx, value in enumerate(payload):
            by_index[idx][int(value)] += 1

    if not by_index:
        raise ValueError(f"No CRC-valid payload rows found in {args.json}")

    expected: list[int] = []
    rows: list[dict[str, Any]] = []
    stable_count = 0
    for idx in range(max(by_index) + 1):
        counter = by_index[idx]
        byte, count = counter.most_common(1)[0]
        total = sum(counter.values())
        stability = float(count) / float(total)
        is_stable = stability >= float(args.min_stability)
        expected.append(int(byte) if is_stable else -1)
        stable_count += int(is_stable)
        rows.append({
            "byte_index": idx,
            "expected_byte": int(byte) if is_stable else -1,
            "modal_byte": int(byte),
            "modal_hex": f"{int(byte):02x}",
            "stability": stability,
            "observations": total,
        })

    out = {
        "expected_payload_bytes": expected,
        "stable_count": int(stable_count),
        "total_bytes": int(len(expected)),
        "min_stability": float(args.min_stability),
        "packet_count": int(packet_count),
        "source_json": str(args.json),
        "rows": rows,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"Learned byte template: {stable_count}/{len(expected)} stable bytes from {packet_count} packets")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

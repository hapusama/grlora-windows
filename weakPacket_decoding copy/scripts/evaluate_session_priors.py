#!/usr/bin/env python3
"""Evaluate learned session priors without running the demodulator.

This script reports how much of a LoRa session is structurally recoverable from
learned stable coded-symbol and payload-byte templates. It is a sanity check for
the phase-guided weak-packet design: phase/sync keeps packet alignment, while
session priors shrink the residual decoding space.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate session priors")
    parser.add_argument("-g", "--gt-symbol-csv", type=Path, required=True)
    parser.add_argument("--payload-template-file", type=Path, required=True)
    parser.add_argument("--byte-template-file", type=Path, default=None)
    parser.add_argument("--packet", type=int, nargs="*", default=None)
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def _load_template(path: Path, key: str) -> list[int]:
    with path.resolve().open("r", encoding="utf-8") as f:
        doc = json.load(f)
    values = doc.get(key, doc if isinstance(doc, list) else [])
    return [int(v) for v in values]


def main() -> int:
    args = parse_args()
    packet_filter = set(args.packet) if args.packet else None
    symbol_template = _load_template(args.payload_template_file, "expected_payload_symbols")
    byte_template = (
        _load_template(args.byte_template_file, "expected_payload_bytes")
        if args.byte_template_file else []
    )

    by_packet: dict[int, dict[int, int]] = {}
    with args.gt_symbol_csv.resolve().open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("stage", "")).strip() != "payload":
                continue
            if _int(row, "header_valid", 0) != 1:
                continue
            packet = _int(row, "packet_index", -1)
            if packet_filter is not None and packet not in packet_filter:
                continue
            idx = _int(row, "stage_symbol_index", -1)
            value = _int(row, "symbol_value", -1)
            if packet < 0 or idx < 0 or value < 0:
                continue
            by_packet.setdefault(packet, {})[idx] = value

    rows: list[dict[str, Any]] = []
    total_known = sum(1 for v in symbol_template if v >= 0)
    total_dynamic = len(symbol_template) - total_known
    for packet, symbols in sorted(by_packet.items()):
        matched = 0
        checked = 0
        for idx, expected in enumerate(symbol_template):
            if expected < 0:
                continue
            if idx not in symbols:
                continue
            checked += 1
            matched += int(symbols[idx] == expected)
        rows.append({
            "packet_index": packet,
            "known_symbol_positions": checked,
            "known_symbol_matches": matched,
            "known_symbol_match_rate": matched / checked if checked else 0.0,
        })

    total_checked = sum(int(r["known_symbol_positions"]) for r in rows)
    total_matched = sum(int(r["known_symbol_matches"]) for r in rows)
    print(f"Payload symbol template: {total_known}/{len(symbol_template)} known, {total_dynamic} dynamic")
    print(
        f"Template matches GT stable positions: {total_matched}/{total_checked} "
        f"({(total_matched / total_checked if total_checked else 0.0):.4f})"
    )
    if byte_template:
        byte_known = sum(1 for v in byte_template if v >= 0)
        print(f"Payload byte template: {byte_known}/{len(byte_template)} known")
    print("Per packet:")
    for row in rows:
        print(
            f"  pkt {row['packet_index']}: "
            f"{row['known_symbol_matches']}/{row['known_symbol_positions']} "
            f"({row['known_symbol_match_rate']:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

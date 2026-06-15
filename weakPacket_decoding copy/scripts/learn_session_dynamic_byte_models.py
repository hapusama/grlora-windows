#!/usr/bin/env python3
"""Learn affine models for dynamic payload bytes.

Stable bytes are handled by `learn_session_payload_byte_template.py`. For the
remaining dynamic bytes, this script tries simple session counter models:

    byte_value = (a * packet_index + b) mod 256

Only models that exactly match the training rows are emitted. The model is
deliberately conservative: dynamic bytes without a perfect affine fit remain
unknown rather than being guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn dynamic byte models")
    parser.add_argument("-j", "--json", type=Path, required=True,
                        help="JSON file with soft_payload_rows")
    parser.add_argument("-t", "--byte-template-file", type=Path, required=True,
                        help="byte template JSON with -1 dynamic positions")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="output JSON model file")
    parser.add_argument("--exclude-packet", type=int, nargs="*", default=None,
                        help="packet indices excluded from model learning")
    parser.add_argument("--max-slope", type=int, default=16,
                        help="try slopes in [-max_slope, max_slope]")
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


def _load_expected_bytes(path: Path) -> list[int]:
    with path.resolve().open("r", encoding="utf-8") as f:
        doc = json.load(f)
    return [int(v) for v in doc.get("expected_payload_bytes", [])]


def main() -> int:
    args = parse_args()
    exclude = set(args.exclude_packet or [])
    expected = _load_expected_bytes(args.byte_template_file)
    dynamic_indices = [idx for idx, value in enumerate(expected) if int(value) < 0]
    with args.json.resolve().open("r", encoding="utf-8") as f:
        doc = json.load(f)

    observations: dict[int, dict[int, list[int]]] = {idx: {} for idx in dynamic_indices}
    packet_count = 0
    for row in _collect_payload_rows(doc):
        if not bool(row.get("crc_valid", False)):
            continue
        packet_index = int(row.get("packet_index", -1))
        if packet_index < 0 or packet_index in exclude:
            continue
        payload_hex = str(row.get("payload_hex", "")).strip()
        if not payload_hex:
            continue
        payload = bytes.fromhex(payload_hex)
        packet_count += 1
        for idx in dynamic_indices:
            if idx < len(payload):
                observations[idx].setdefault(packet_index, []).append(int(payload[idx]))

    models: list[dict[str, int | str]] = []
    for idx, pairs in observations.items():
        # Collapse repeated observations from multiple noise summaries by modal
        # value per packet. This keeps one vote per packet in the affine fit.
        unique = []
        for packet_index, values in sorted(pairs.items()):
            counts: dict[int, int] = {}
            for value in values:
                counts[int(value)] = counts.get(int(value), 0) + 1
            modal = max(counts.items(), key=lambda item: (item[1], -item[0]))[0]
            unique.append((int(packet_index), int(modal)))
        if not unique:
            continue
        best = None
        for slope in range(-int(args.max_slope), int(args.max_slope) + 1):
            for intercept in range(256):
                if all(((slope * p + intercept) & 0xFF) == value for p, value in unique):
                    best = {
                        "byte_index": int(idx),
                        "model": "affine_mod_256",
                        "slope": int(slope),
                        "intercept": int(intercept),
                        "observations": int(len(unique)),
                    }
                    break
            if best is not None:
                break
        if best is not None:
            models.append(best)

    out = {
        "source_json": str(args.json),
        "byte_template_file": str(args.byte_template_file),
        "packet_count": int(packet_count),
        "dynamic_indices": dynamic_indices,
        "models": models,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"Learned dynamic byte models: {len(models)}/{len(dynamic_indices)}")
    for model in models:
        print(
            f"  byte {model['byte_index']}: "
            f"value = ({model['slope']}*packet_index + {model['intercept']}) mod 256"
        )
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

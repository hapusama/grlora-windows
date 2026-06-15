#!/usr/bin/env python3
"""Reconstruct session payload bytes from stable template + dynamic models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct session payloads")
    parser.add_argument("-t", "--byte-template-file", type=Path, required=True)
    parser.add_argument("-m", "--dynamic-model-file", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--packet", type=int, nargs="*", required=True,
                        help="packet indices to reconstruct")
    parser.add_argument("--expected-json", type=Path, default=None,
                        help="optional JSON with soft_payload_rows for evaluation")
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


def _load_expected_map(path: Path | None) -> dict[int, bytes]:
    if path is None:
        return {}
    with path.resolve().open("r", encoding="utf-8") as f:
        doc = json.load(f)
    result: dict[int, bytes] = {}
    for row in _collect_payload_rows(doc):
        if not bool(row.get("crc_valid", False)):
            continue
        packet_index = int(row.get("packet_index", -1))
        payload_hex = str(row.get("payload_hex", "")).strip()
        if packet_index >= 0 and payload_hex:
            result.setdefault(packet_index, bytes.fromhex(payload_hex))
    return result


def main() -> int:
    args = parse_args()
    with args.byte_template_file.resolve().open("r", encoding="utf-8") as f:
        template_doc = json.load(f)
    expected_bytes = [int(v) for v in template_doc.get("expected_payload_bytes", [])]
    with args.dynamic_model_file.resolve().open("r", encoding="utf-8") as f:
        model_doc = json.load(f)
    models = {int(m["byte_index"]): m for m in model_doc.get("models", [])}
    expected_map = _load_expected_map(args.expected_json)

    rows: list[dict[str, Any]] = []
    for packet_index in args.packet:
        payload = []
        unknown = []
        for idx, value in enumerate(expected_bytes):
            if value >= 0:
                payload.append(int(value) & 0xFF)
                continue
            model = models.get(idx)
            if model and model.get("model") == "affine_mod_256":
                payload.append((int(model["slope"]) * int(packet_index) + int(model["intercept"])) & 0xFF)
            else:
                payload.append(0)
                unknown.append(idx)
        payload_bytes = bytes(payload)
        expected = expected_map.get(int(packet_index))
        byte_errors = None
        exact = None
        if expected is not None:
            n = min(len(expected), len(payload_bytes))
            byte_errors = sum(1 for i in range(n) if expected[i] != payload_bytes[i]) + abs(len(expected) - len(payload_bytes))
            exact = byte_errors == 0
        rows.append({
            "packet_index": int(packet_index),
            "payload_hex": payload_bytes.hex(),
            "unknown_indices": unknown,
            "expected_available": expected is not None,
            "byte_errors": byte_errors,
            "exact": exact,
        })

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
        f.write("\n")

    exact_count = sum(1 for row in rows if row["exact"] is True)
    compared = sum(1 for row in rows if row["expected_available"])
    print(f"Reconstructed {len(rows)} payloads")
    if compared:
        print(f"Exact matches: {exact_count}/{compared}")
    for row in rows:
        suffix = ""
        if row["expected_available"]:
            suffix = f" byte_errors={row['byte_errors']} exact={row['exact']}"
        print(f"  pkt {row['packet_index']}: {row['payload_hex']}{suffix}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

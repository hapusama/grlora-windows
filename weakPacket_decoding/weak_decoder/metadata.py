"""Metadata loading helpers for noisy_iq JSON outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def load_metadata(path: Path) -> dict[str, Any]:
    """Load a noisy_iq JSON metadata file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def packet_measurements(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return packet measurements from noisy_iq metadata."""
    measurement = metadata.get("grlora_snr_measurement", {})
    packets = measurement.get("packet_measurements", [])
    if not isinstance(packets, list):
        return []
    return [packet for packet in packets if isinstance(packet, dict)]


def metadata_input_file(metadata: dict[str, Any], metadata_path: Path | None = None) -> Path | None:
    """Resolve the IQ file referenced by noisy_iq metadata."""
    for key in ("output_file", "file", "input_file"):
        value = metadata.get(key)
        if value:
            path = Path(value)
            if path.exists():
                return path
            if metadata_path is not None:
                candidate = metadata_path.parent / path.name
                if candidate.exists():
                    return candidate
    return None

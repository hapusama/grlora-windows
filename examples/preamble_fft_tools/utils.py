# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""Small generic helpers for paths, formatting, CSV, and JSON safety."""

import csv
import hashlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def suppress_native_output(enabled=True):
    """Temporarily hide GNU Radio/C++ stdout/stderr noise during one detector run."""
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def prepare_file_source_path(input_file):
    """Return an ASCII-only hardlink path for GNU Radio file_source on Windows.

    GNU Radio's C++ file_source uses fopen on Windows, which can fail on paths
    containing Chinese characters. Python can still read those paths, so only
    the GNU Radio input side is redirected through a temporary hardlink.
    """
    source_path = Path(input_file).resolve()
    staging_dir = Path(__file__).resolve().parents[1] / "_file_source_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()
    staged_path = staging_dir / f"{digest}{source_path.suffix.lower()}"
    if staged_path.exists():
        try:
            if staged_path.samefile(source_path):
                return str(staged_path)
        except OSError:
            pass
        staged_path.unlink()

    try:
        os.link(source_path, staged_path)
    except OSError as exc:
        raise RuntimeError(
            f"failed to create ASCII hardlink for GNU Radio file_source: {source_path}"
        ) from exc
    return str(staged_path)


def cleanup_file_source_path(capture_args):
    staged_path = getattr(capture_args, "file_source_path", "")
    if not staged_path:
        return
    try:
        Path(staged_path).unlink(missing_ok=True)
    except OSError:
        pass


def int_or_default(value, default=-1):
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_position_ids(value):
    """Parse a comma-separated location/position filter."""
    if value is None or value == [] or str(value).strip() == "":
        return None
    position_ids = set()
    values = value if isinstance(value, (list, tuple)) else [value]
    for raw_value in values:
        for item in str(raw_value).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                position_ids.add(int(item))
            except ValueError as exc:
                raise ValueError(f"invalid position id: {item!r}") from exc
    return position_ids or None


def fmt_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(value):
        return ""
    return f"{value:.9g}"


def write_dict_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_text_file(path):
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return Path(path).read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(errors="ignore").strip()


def json_safe(value):
    """Convert numpy/Path values into plain JSON-compatible Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def tail_lines(text, limit=40):
    lines = text.splitlines()
    return "\n".join(lines[-limit:])

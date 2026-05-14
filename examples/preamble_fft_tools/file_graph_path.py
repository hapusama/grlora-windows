# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""Stable paths shared by subprocess helpers."""

from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "lora_file_preamble_fft.py"

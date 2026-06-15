#!/usr/bin/env python3
"""Small SNR sweep for phase-stabilized session decoding.

This script intentionally runs only a small number of packets by default.  It
wraps run_phase_guided_demod.py, then summarizes symbol SER and optional
application-payload reconstruction accuracy into one CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parent.parent
RUNNER = WEAK_ROOT / "scripts" / "run_phase_guided_demod.py"
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.payload_codec import decode_explicit_frame_symbols

DEFAULT_HEADER = "75,163,15,20,211,206,182,86"
DEFAULT_SYNC = WEAK_ROOT / "data" / "weak_sync_chain" / "sync_chain" / "0_0_0_10_14_16_sync_chain.csv"
DEFAULT_GT = WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / "0_0_0_10_14_16_header_first_symbols.csv"
DEFAULT_SYMBOL_TEMPLATE = WEAK_ROOT / "data" / "phase_guided" / "session_payload_template_excl_1_2_3_5_6_guarded.json"
DEFAULT_SYMBOL_BYTE_TEMPLATE = WEAK_ROOT / "data" / "phase_guided" / "session_payload_byte_template_from_symbols_excl_1_2_3_5_6.json"
DEFAULT_SYMBOL_DYNAMIC_MODELS = WEAK_ROOT / "data" / "phase_guided" / "session_dynamic_byte_models_from_symbols_excl_1_2_3_5_6.json"
DEFAULT_BYTE_TEMPLATE = DEFAULT_SYMBOL_BYTE_TEMPLATE
DEFAULT_DYNAMIC_MODELS = DEFAULT_SYMBOL_DYNAMIC_MODELS
DEFAULT_EXPECTED_JSON = WEAK_ROOT / "data" / "noisy_iq" / "_phase_symbol_eval_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact SNR sweep for session-level weak decoding."
    )
    parser.add_argument("--snr", type=int, nargs="*", default=[-20, -23, -25, -27],
                        help="SNR points in dB, e.g. --snr -23 -25 -27")
    parser.add_argument("--mode", nargs="*", default=["no_prior", "header_prior", "session_full"],
                        choices=(
                            "no_prior",
                            "header_prior",
                            "session_full",
                            "session_residual",
                            "session_residual_profile",
                            "session_residual_search",
                            "session_residual_search_profile",
                        ),
                        help="decoder variants to evaluate")
    parser.add_argument("--max-packets", type=int, default=5,
                        help="framesync-valid packets per run")
    parser.add_argument("--output-root", type=Path,
                        default=WEAK_ROOT / "data" / "phase_guided" / "session_sweep",
                        help="directory for per-run outputs and summary CSV")
    parser.add_argument("--summary-csv", type=Path, default=None,
                        help="optional explicit summary CSV path")
    parser.add_argument("--sync-chain-csv", type=Path, default=DEFAULT_SYNC)
    parser.add_argument("--gt-symbol-csv", type=Path, default=DEFAULT_GT)
    parser.add_argument("--expected-header-symbols", type=str, default=DEFAULT_HEADER)
    parser.add_argument("--payload-template-file", type=Path, default=DEFAULT_SYMBOL_TEMPLATE)
    parser.add_argument("--byte-template-file", type=Path, default=DEFAULT_BYTE_TEMPLATE)
    parser.add_argument("--dynamic-byte-model-file", type=Path, default=DEFAULT_DYNAMIC_MODELS)
    parser.add_argument("--symbol-byte-template-file", type=Path, default=DEFAULT_SYMBOL_BYTE_TEMPLATE)
    parser.add_argument("--symbol-dynamic-byte-model-file", type=Path, default=DEFAULT_SYMBOL_DYNAMIC_MODELS)
    parser.add_argument("--expected-json", type=Path, default=DEFAULT_EXPECTED_JSON)
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--bw", type=float, default=125000.0)
    parser.add_argument("--samp-rate", type=float, default=500000.0)
    parser.add_argument("--preamble-len", type=float, default=16.0)
    parser.add_argument("--refinement-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without running them")
    return parser.parse_args()


def snr_input_path(snr_db: int) -> Path:
    stem = f"0_0_0_10_14_16_snr_m{abs(int(snr_db))}dB.bin"
    if int(snr_db) in (-10, -15, -20):
        return WEAK_ROOT / "data" / "low_snr_gt_bin" / "0_0_0_10_14_16" / stem
    return WEAK_ROOT / "data" / "low_snr_gt_bin" / "0_0_0_10_14_16_extreme_snr" / stem


def collect_payload_rows(obj: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        value = obj.get("soft_payload_rows")
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
        for child in obj.values():
            rows.extend(collect_payload_rows(child))
    elif isinstance(obj, list):
        for child in obj:
            rows.extend(collect_payload_rows(child))
    return rows


def load_expected_payloads(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    result: dict[int, str] = {}
    for row in collect_payload_rows(doc):
        if not bool(row.get("crc_valid", False)):
            continue
        packet_index = int(row.get("packet_index", -1))
        payload_hex = str(row.get("payload_hex", "")).strip().lower()
        if packet_index >= 0 and payload_hex:
            result.setdefault(packet_index, payload_hex)
    return result


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def load_expected_payloads_from_symbols(
    path: Path,
    sf: int,
    bw: float,
    ldro_mode: int = 2,
) -> dict[int, str]:
    """Build expected payload map in sync-chain packet-index coordinates."""
    if not path.exists():
        return {}
    grouped: dict[tuple[int, int], dict[str, list[int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
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
            grouped.setdefault(key, {"header": [], "payload": []})[stage].append(value)
    result: dict[int, str] = {}
    for (packet_index, _event_index), symbols in sorted(grouped.items()):
        if len(symbols["header"]) != 8 or not symbols["payload"]:
            continue
        decoded = decode_explicit_frame_symbols(
            symbols["header"],
            symbols["payload"],
            sf=int(sf),
            bw=float(bw),
            ldro_mode=int(ldro_mode),
        )
        if decoded.payload.crc_valid:
            result.setdefault(int(packet_index), decoded.payload.payload_bytes.hex())
    return result


def run_one(args: argparse.Namespace, snr_db: int, mode: str) -> dict[str, Any]:
    input_path = snr_input_path(snr_db)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    run_dir = args.output_root / f"snr_m{abs(int(snr_db))}dB" / mode

    cmd = [
        sys.executable,
        str(RUNNER),
        "-i", str(input_path),
        "-s", str(args.sync_chain_csv),
        "-g", str(args.gt_symbol_csv),
        "-o", str(run_dir),
        "--sf", str(args.sf),
        "--bw", str(args.bw),
        "--samp-rate", str(args.samp_rate),
        "--preamble-len", str(args.preamble_len),
        "--max-packets", str(args.max_packets),
        "--seed", str(args.seed),
        "--refinement-rounds", str(args.refinement_rounds),
    ]

    if mode in (
        "header_prior",
        "session_full",
        "session_residual",
        "session_residual_profile",
        "session_residual_search",
        "session_residual_search_profile",
    ):
        cmd.extend(["--expected-header-symbols", str(args.expected_header_symbols)])
    if mode in (
        "session_full",
        "session_residual",
        "session_residual_profile",
        "session_residual_search",
        "session_residual_search_profile",
    ):
        byte_template_file = args.byte_template_file
        dynamic_model_file = args.dynamic_byte_model_file
        if mode in (
            "session_residual",
            "session_residual_profile",
            "session_residual_search",
            "session_residual_search_profile",
        ):
            byte_template_file = args.symbol_byte_template_file
            dynamic_model_file = args.symbol_dynamic_byte_model_file
        cmd.extend([
            "--payload-template-file", str(args.payload_template_file),
            "--byte-template-file", str(byte_template_file),
            "--dynamic-byte-model-file", str(dynamic_model_file),
        ])
        if mode in ("session_residual", "session_residual_profile"):
            cmd.append("--enable-byte-prior-symbols")
        if mode in ("session_residual_search", "session_residual_search_profile"):
            cmd.extend([
                "--enable-byte-residual-search",
                "--residual-byte-index", "6",
                "--residual-byte-values", "0-255",
                "--residual-max-unknown-bytes", "1",
                "--residual-max-candidates", "256",
                "--candidate-search-min-known", "3",
                "--candidate-search-prior-weight", "0.35",
                "--residual-model-prior-sigma", "0.5",
            ])
        if mode in ("session_residual_profile", "session_residual_search_profile"):
            cmd.extend([
                "--enable-preamble-profile-score",
                "--preamble-profile-weight", "0.15",
                "--candidate-search-profile-weight", "0.10",
            ])

    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return {
            "snr_db": int(snr_db),
            "mode": mode,
            "packets": 0,
            "total_symbols": 0,
            "correct_symbols": 0,
            "overall_ser": "",
            "payload_template_known_total": 0,
            "reconstructed_exact": 0,
            "reconstructed_compared": 0,
            "reconstructed_unknown_bytes_total": 0,
            "header_methods": "{}",
            "summary_path": str(run_dir / f"{input_path.stem}_phase_guided_summary.csv"),
        }
    subprocess.run(cmd, check=True)

    summary_path = run_dir / f"{input_path.stem}_phase_guided_summary.csv"
    expected_payloads = load_expected_payloads_from_symbols(
        args.gt_symbol_csv,
        sf=int(args.sf),
        bw=float(args.bw),
    )
    if not expected_payloads:
        expected_payloads = load_expected_payloads(args.expected_json)
    return summarize_run(summary_path, snr_db, mode, expected_payloads)


def summarize_run(
    summary_path: Path,
    snr_db: int,
    mode: str,
    expected_payloads: dict[int, str],
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    total_symbols = sum(int(float(row.get("total_payload", 0) or 0)) for row in rows)
    correct = sum(int(float(row.get("correct_count", 0) or 0)) for row in rows)
    overall_ser = 1.0 - correct / total_symbols if total_symbols else 1.0
    symbol_path = Path(str(summary_path).replace(
        "_phase_guided_summary.csv",
        "_phase_guided_symbols.csv",
    ))
    argmax_total = 0
    argmax_correct = 0
    if symbol_path.exists():
        with symbol_path.open("r", encoding="utf-8", newline="") as f:
            for sym in csv.DictReader(f):
                gt = int(float(sym.get("gt_bin", -1) or -1))
                argmax_bin = int(float(sym.get("argmax_bin", -1) or -1))
                if gt >= 0:
                    argmax_total += 1
                    argmax_correct += int(argmax_bin == gt)
    argmax_ser = 1.0 - argmax_correct / argmax_total if argmax_total else ""
    exact = 0
    compared = 0
    unknown_sum = 0
    known_template = 0
    byte_symbol_prior_known = 0
    byte_residual_candidates = 0
    prior_search_success = 0
    prior_search_margin_sum = 0.0
    prior_search_rmse_sum = 0.0
    header_methods: dict[str, int] = {}

    for row in rows:
        packet_index = int(float(row.get("packet_index", -1) or -1))
        payload_hex = str(row.get("reconstructed_payload_hex", "")).strip().lower()
        unknown_sum += int(float(row.get("reconstructed_unknown_bytes", 0) or 0))
        known_template += int(float(row.get("payload_template_known", 0) or 0))
        byte_symbol_prior_known += int(float(row.get("byte_symbol_prior_known", 0) or 0))
        byte_residual_candidates += int(float(row.get("byte_residual_candidates", 0) or 0))
        if int(float(row.get("prior_search_success", 0) or 0)):
            prior_search_success += 1
            prior_search_margin_sum += float(row.get("prior_search_margin", 0) or 0)
            prior_search_rmse_sum += float(row.get("prior_search_line_rmse_pi", 0) or 0)
        method = str(row.get("header_method", "")).strip()
        header_methods[method] = header_methods.get(method, 0) + 1
        expected = expected_payloads.get(packet_index)
        if payload_hex and expected:
            compared += 1
            exact += int(payload_hex == expected)

    return {
        "snr_db": int(snr_db),
        "mode": mode,
        "packets": len(rows),
        "total_symbols": total_symbols,
        "correct_symbols": correct,
        "overall_ser": f"{overall_ser:.4f}",
        "argmax_total_symbols": argmax_total,
        "argmax_correct_symbols": argmax_correct,
        "argmax_ser": f"{argmax_ser:.4f}" if argmax_total else "",
        "correct_gain_vs_argmax": correct - argmax_correct if argmax_total else "",
        "payload_template_known_total": known_template,
        "byte_symbol_prior_known_total": byte_symbol_prior_known,
        "byte_residual_candidates_total": byte_residual_candidates,
        "prior_search_success": prior_search_success,
        "prior_search_mean_margin": (
            f"{prior_search_margin_sum / prior_search_success:.6f}"
            if prior_search_success else ""
        ),
        "prior_search_mean_rmse_pi": (
            f"{prior_search_rmse_sum / prior_search_success:.6f}"
            if prior_search_success else ""
        ),
        "reconstructed_exact": exact,
        "reconstructed_compared": compared,
        "reconstructed_unknown_bytes_total": unknown_sum,
        "header_methods": json.dumps(header_methods, sort_keys=True),
        "summary_path": str(summary_path),
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snr_db",
        "mode",
        "packets",
        "total_symbols",
        "correct_symbols",
        "overall_ser",
        "argmax_total_symbols",
        "argmax_correct_symbols",
        "argmax_ser",
        "correct_gain_vs_argmax",
        "payload_template_known_total",
        "byte_symbol_prior_known_total",
        "byte_residual_candidates_total",
        "prior_search_success",
        "prior_search_mean_margin",
        "prior_search_mean_rmse_pi",
        "reconstructed_exact",
        "reconstructed_compared",
        "reconstructed_unknown_bytes_total",
        "header_methods",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    summary_csv = (
        args.summary_csv.resolve()
        if args.summary_csv else args.output_root / "session_sweep_summary.csv"
    )
    rows = []
    for snr_db in args.snr:
        for mode in args.mode:
            rows.append(run_one(args, int(snr_db), str(mode)))
            write_summary(summary_csv, rows)
    print(f"\nSummary: {summary_csv}")
    for row in rows:
        recon = ""
        if int(row["reconstructed_compared"]):
            recon = (
                f", app_exact={row['reconstructed_exact']}/"
                f"{row['reconstructed_compared']}"
            )
        print(
            f"  {row['snr_db']:>4} dB {row['mode']:<13} "
            f"SER={row['overall_ser']} "
            f"argmax_SER={row.get('argmax_ser', '')} "
            f"symbols={row['correct_symbols']}/{row['total_symbols']}{recon}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

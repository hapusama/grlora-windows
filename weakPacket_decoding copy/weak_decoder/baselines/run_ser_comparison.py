#!/usr/bin/env python3
"""Run SER comparison for local baselines and phase-line selectors.

The runner compares payload raw FFT-bin SER on the same noisy IQ realization.
It intentionally keeps every method FFT-bin/symbol-level only: no payload
templates, cross-packet priors, or CRC-guided symbol selection are used.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[2]
GR_LORA_ROOT = WEAK_ROOT.parent
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.loratrimmer import demod_loratrimmer_symbol  # noqa: E402
from weak_decoder.baselines.symfec import SymFECConfig, decode_symfec_payload_from_spectra  # noqa: E402
from weak_decoder.baselines.symfec.run_symfec_baseline import extract_multi_offset_spectrum  # noqa: E402
from weak_decoder.baselines.unichirp.evaluate_unichirp_vs_island import (  # noqa: E402
    _dataset_paths,
    _evaluate_unichirp_packet,
    _load_packets,
)
from weak_decoder.baselines.unichirp.paper_unichirp_demod import UniChirpDemodConfig  # noqa: E402
from weak_decoder.phase_line.savaux_stage1 import default_savaux_phase_path_config  # noqa: E402
from weak_decoder.phase_line.variants.island_dp_reconstruction.dual_evidence import (  # noqa: E402
    DualEvidenceFusionConfig,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.evaluate_island_dp import (  # noqa: E402
    _evaluate_packet as _evaluate_island_packet,
    _err_count,
    _noise_samples,
    _snr_values,
    _write_csv,
)
from weak_decoder.phase_line.variants.island_dp_reconstruction.selector import (  # noqa: E402
    IslandReconstructionConfig,
)


METHOD_ORDER = (
    "savaux_oversampled",
    "loratrimmer",
    "symfec",
    "unichirp",
    "phaseline_one_order_dp",
    "phaseline_island_dp",
)

METHOD_LABELS = {
    "savaux_oversampled": "Savaux oversampled",
    "loratrimmer": "LoRaTrimmer",
    "symfec": "Sym-FEC",
    "unichirp": "UniChirp",
    "phaseline_one_order_dp": "Phase-line one-order DP",
    "phaseline_island_dp": "Phase-line island DP",
}


def _parse_snr_grid(start: float, stop: float, step: float) -> list[float]:
    if step == 0.0:
        raise ValueError("snr step must be non-zero")
    values: list[float] = []
    cur = float(start)
    if step < 0:
        while cur >= float(stop) - 1e-9:
            values.append(round(cur, 6))
            cur += float(step)
    else:
        while cur <= float(stop) + 1e-9:
            values.append(round(cur, 6))
            cur += float(step)
    return values


def _payload_gt_bins(packet: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(item["gt_bin"]) for item in packet["payload_symbols"])


def _evaluate_loratrimmer_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
) -> dict[str, Any]:
    selected: list[int] = []
    origin_shift = int(packet["os_factor"]) // 2
    for item in packet["payload_symbols"]:
        result = demod_loratrimmer_symbol(
            samples=samples,
            start_sample=int(item["start_sample"]) + origin_shift,
            sf=int(packet["sf"]),
            os_factor=int(packet["os_factor"]),
            ldro=bool(packet["ldro"]),
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=int(packet["header_start_sample"]) + origin_shift,
            cfo_correction_mode="continuous",
        )
        selected.append(int(result.raw_fft_bin))
    errors, compared = _err_count(selected, _payload_gt_bins(packet))
    return {"loratrimmer_err": int(errors), "symbol_count": int(compared)}


def _evaluate_symfec_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    config: SymFECConfig,
) -> dict[str, Any]:
    spectra: list[np.ndarray] = []
    for item in packet["payload_symbols"]:
        try:
            spectra.append(
                extract_multi_offset_spectrum(
                    samples=samples,
                    packet=packet,
                    start_sample=int(item["start_sample"]),
                    cfo_correction_mode="continuous",
                )
            )
        except ValueError:
            continue
    if not spectra:
        return {"symfec_err": 0, "symfec_crc_valid": 0, "symbol_count": 0}
    result = decode_symfec_payload_from_spectra(
        spectra_or_powers=spectra,
        sf=int(packet["sf"]),
        cr=int(packet["cr"]),
        ldro=bool(packet["ldro"]),
        config=config,
        header_symbol_values=tuple(int(item["symbol_value"]) for item in packet["header_symbols"]),
        bw=float(packet["bw"]),
        ldro_mode=2,
        payload_len=int(packet["payload_len"]),
        has_crc=bool(packet["has_crc"]),
        crc_mode="grlora",
    )
    errors, compared = _err_count(result.selected_raw_fft_bins, _payload_gt_bins(packet))
    return {
        "symfec_err": int(errors),
        "symfec_crc_valid": int(result.crc_valid),
        "symbol_count": int(compared),
    }


def _evaluate_group(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    include_loratrimmer: bool,
) -> dict[str, int]:
    totals = {f"{name}_err": 0 for name in METHOD_ORDER}
    totals["symbol_count"] = 0
    totals["packet_count"] = 0
    totals["symfec_crc_valid_count"] = 0

    path_config = default_savaux_phase_path_config(top_l=16)
    island_config = IslandReconstructionConfig()
    fusion_config = DualEvidenceFusionConfig(
        mode="product_norm",
        corrected_weight=0.5,
        stage1_top_k=40,
        retain_dechirped_symbols=True,
    )
    unichirp_config = UniChirpDemodConfig()
    symfec_config = SymFECConfig()

    for packet in packets:
        symbol_count = len(packet["payload_symbols"])
        totals["packet_count"] += 1
        totals["symbol_count"] += int(symbol_count)

        phase = _evaluate_island_packet(
            samples=samples,
            packet=packet,
            path_config=path_config,
            island_config=island_config,
            fusion_config=fusion_config,
            use_aux_evidence=False,
        )
        totals["savaux_oversampled_err"] += int(phase["hard_err"])
        totals["phaseline_one_order_dp_err"] += int(phase["v1_err"])
        totals["phaseline_island_dp_err"] += int(phase["fusion_island_err"])

        unichirp = _evaluate_unichirp_packet(
            samples=samples,
            packet=packet,
            config=unichirp_config,
            training_source="preamble_header",
        )
        totals["unichirp_err"] += int(unichirp["unichirp_err"])

        symfec = _evaluate_symfec_packet(samples=samples, packet=packet, config=symfec_config)
        totals["symfec_err"] += int(symfec["symfec_err"])
        totals["symfec_crc_valid_count"] += int(symfec["symfec_crc_valid"])

        if include_loratrimmer:
            loratrimmer = _evaluate_loratrimmer_packet(samples=samples, packet=packet)
            totals["loratrimmer_err"] += int(loratrimmer["loratrimmer_err"])
        else:
            totals["loratrimmer_err"] = -1

    return totals


def _summary_rows(group_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for row in group_rows:
        key = (str(row["dataset"]), float(row["snr_db"]))
        out = by_key.setdefault(
            key,
            {
                "dataset": str(row["dataset"]),
                "snr_db": float(row["snr_db"]),
                "packet_count": 0,
                "symbol_count": 0,
                "symfec_crc_valid_count": 0,
            },
        )
        out["packet_count"] += int(row["packet_count"])
        out["symbol_count"] += int(row["symbol_count"])
        out["symfec_crc_valid_count"] += int(row.get("symfec_crc_valid_count", 0))
        for method in METHOD_ORDER:
            err_key = f"{method}_err"
            out[err_key] = int(out.get(err_key, 0)) + int(row.get(err_key, 0))
    rows: list[dict[str, Any]] = []
    for out in by_key.values():
        symbols = int(out["symbol_count"])
        for method in METHOD_ORDER:
            err_key = f"{method}_err"
            ser_key = f"{method}_ser"
            err = int(out.get(err_key, 0))
            out[ser_key] = "" if err < 0 else float(err / max(1, symbols))
        out["symfec_crc_valid_rate"] = float(out["symfec_crc_valid_count"] / max(1, int(out["packet_count"])))
        rows.append(out)
    rows.sort(key=lambda item: (str(item["dataset"]), float(item["snr_db"])), reverse=True)
    return rows


def _overall_rows(group_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for row in group_rows:
        dataset = str(row["dataset"])
        out = by_dataset.setdefault(
            dataset,
            {
                "dataset": dataset,
                "packet_count": 0,
                "symbol_count": 0,
            },
        )
        out["packet_count"] += int(row["packet_count"])
        out["symbol_count"] += int(row["symbol_count"])
        for method in METHOD_ORDER:
            err_key = f"{method}_err"
            out[err_key] = int(out.get(err_key, 0)) + int(row.get(err_key, 0))
    rows: list[dict[str, Any]] = []
    for out in by_dataset.values():
        symbols = int(out["symbol_count"])
        for method in METHOD_ORDER:
            err = int(out.get(f"{method}_err", 0))
            out[f"{method}_ser"] = "" if err < 0 else float(err / max(1, symbols))
        rows.append(out)
    rows.sort(key=lambda item: str(item["dataset"]))
    return rows


def _nice_y_max(values: Sequence[float]) -> float:
    vmax = max([0.01, *[float(v) for v in values if math.isfinite(float(v))]])
    return math.ceil(vmax * 20.0) / 20.0


def _svg_line_chart(
    path: Path,
    rows: Sequence[dict[str, Any]],
    dataset: str,
    methods: Sequence[str],
) -> None:
    snrs = sorted({float(row["snr_db"]) for row in rows if str(row["dataset"]) == str(dataset)}, reverse=True)
    points_by_method: dict[str, list[tuple[float, float]]] = {method: [] for method in methods}
    for method in methods:
        key = f"{method}_ser"
        for snr in snrs:
            match = next(
                (row for row in rows if str(row["dataset"]) == str(dataset) and abs(float(row["snr_db"]) - snr) < 1e-9),
                None,
            )
            if match is None or match.get(key, "") == "":
                continue
            points_by_method[method].append((snr, float(match[key])))
    all_ser = [ser for points in points_by_method.values() for _snr, ser in points]
    ymax = _nice_y_max(all_ser)

    width = 1100
    height = 680
    left = 90
    right = 260
    top = 70
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    denom = max(1, len(snrs) - 1)

    def x_pos(snr: float) -> float:
        idx = snrs.index(float(snr))
        return left + plot_w * idx / denom

    def y_pos(ser: float) -> float:
        return top + plot_h * (1.0 - float(ser) / ymax)

    colors = {
        "savaux_oversampled": "#4C78A8",
        "loratrimmer": "#F58518",
        "symfec": "#54A24B",
        "unichirp": "#E45756",
        "phaseline_one_order_dp": "#B279A2",
        "phaseline_island_dp": "#72B7B2",
    }
    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    parts.append(f'<text x="{left}" y="35" font-family="Arial" font-size="24" font-weight="700">SER comparison on {html.escape(dataset)}</text>')
    parts.append(f'<text x="{left}" y="58" font-family="Arial" font-size="13" fill="#555">Payload raw FFT-bin SER, lower is better</text>')
    for tick in range(6):
        yv = ymax * tick / 5.0
        y = y_pos(yv)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{yv:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.2"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.2"/>')
    for snr in snrs:
        x = x_pos(snr)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#333" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 26}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{snr:g}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">SNR (dB)</text>')
    parts.append(f'<text x="24" y="{top + plot_h / 2:.2f}" transform="rotate(-90 24 {top + plot_h / 2:.2f})" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">SER</text>')

    for method in methods:
        points = points_by_method[method]
        if not points:
            continue
        coords = " ".join(f"{x_pos(snr):.2f},{y_pos(ser):.2f}" for snr, ser in points)
        color = colors.get(method, "#333333")
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.7" stroke-linejoin="round" stroke-linecap="round"/>')
        for snr, ser in points:
            parts.append(f'<circle cx="{x_pos(snr):.2f}" cy="{y_pos(ser):.2f}" r="3.5" fill="{color}"/>')
    legend_x = left + plot_w + 35
    legend_y = top + 15
    parts.append(f'<text x="{legend_x}" y="{legend_y - 18}" font-family="Arial" font-size="14" font-weight="700" fill="#333">Methods</text>')
    for idx, method in enumerate(methods):
        y = legend_y + idx * 28
        color = colors.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x + 13}" cy="{y}" r="3.5" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{y + 4}" font-family="Arial" font-size="13" fill="#333">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_32"])
    parser.add_argument("--snr-start", type=float, default=-25.0)
    parser.add_argument("--snr-stop", type=float, default=-29.0)
    parser.add_argument("--snr-step", type=float, default=-0.5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-packets", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "ser_comparison_baselines_phase_line")
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument("--skip-loratrimmer", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snrs = _parse_snr_grid(args.snr_start, args.snr_stop, args.snr_step)
    out_dir = Path(args.output_dir).resolve()
    group_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = _dataset_paths(str(dataset))
        if not iq_path.exists():
            raise FileNotFoundError(iq_path)
        if not symbol_path.exists():
            raise FileNotFoundError(symbol_path)
        clean = np.fromfile(iq_path, dtype=np.complex64)
        packets = _load_packets(symbol_path)
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        for seed in args.seeds:
            for snr_db in _snr_values(snrs):
                samples = _noise_samples(clean, snr_db, int(seed), args.signal_reference_power)
                totals = _evaluate_group(
                    samples=samples,
                    packets=packets,
                    include_loratrimmer=not bool(args.skip_loratrimmer),
                )
                row: dict[str, Any] = {
                    "dataset": str(dataset),
                    "snr_db": float(snr_db),
                    "seed": int(seed),
                    **totals,
                }
                for method in METHOD_ORDER:
                    err = int(row.get(f"{method}_err", 0))
                    row[f"{method}_ser"] = "" if err < 0 else float(err / max(1, int(row["symbol_count"])))
                group_rows.append(row)
                print(
                    f"{dataset} snr={snr_db:g} seed={seed}: "
                    + " ".join(
                        f"{method}={row[f'{method}_ser']:.4f}"
                        for method in METHOD_ORDER
                        if row.get(f"{method}_ser", "") != ""
                    ),
                    flush=True,
                )

    by_snr = _summary_rows(group_rows)
    overall = _overall_rows(group_rows)
    _write_csv(out_dir / "summary_by_seed.csv", group_rows)
    _write_csv(out_dir / "ser_by_snr.csv", by_snr)
    _write_csv(out_dir / "overall.csv", overall)
    (out_dir / "summary_by_seed.json").write_text(json.dumps(group_rows, indent=2), encoding="utf-8")
    for dataset in args.datasets:
        _svg_line_chart(
            out_dir / f"{dataset}_ser_comparison.svg",
            rows=by_snr,
            dataset=str(dataset),
            methods=tuple(method for method in METHOD_ORDER if not (method == "loratrimmer" and bool(args.skip_loratrimmer))),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""PNG plotting for raw preamble dechirp FFT spectra.

The plots keep raw FFT bin positions, draw the peak trace in blue, and annotate
each packet plot with the absolute linear peak magnitude.
"""

import re
from pathlib import Path

import numpy as np

from .signal_analysis import packet_preamble_dechirp_fft_amplitude
from .utils import int_or_default


PEAK_BLUE = "#1f77b4"
LIGHT_BLUE = "#9ecae1"


def resolve_plot_output_dir(args):
    if getattr(args, "plot_output_dir", None):
        return Path(args.plot_output_dir)
    return Path(args.output_dir) / "preamble_fft_plots"


def safe_plot_name(value):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return name[:120] if name else "unknown"


def import_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def collect_dechirp_fft_plot_groups(args, capture_results):
    groups = {}
    for capture_args, frames in capture_results:
        iq_path = Path(capture_args.input_file)
        iq = np.memmap(iq_path, dtype=np.complex64, mode="r")
        for frame in frames:
            plot_data = packet_preamble_dechirp_fft_amplitude(iq, frame, capture_args)
            if plot_data is None:
                continue

            position_id = int_or_default(frame.get("position_id", ""), -1)
            group_key = position_id if position_id >= 0 else frame.get("file_stem", iq_path.stem)
            group = groups.setdefault(
                group_key,
                {
                    "position_id": position_id,
                    "label": f"location {position_id}" if position_id >= 0 else str(group_key),
                    "x_offsets": plot_data["x_offsets"],
                    "n_bins": plot_data["n_bins"],
                    "sf": plot_data["sf"],
                    "spectra": [],
                    "file_names": set(),
                    "packet_labels": [],
                    "symbol_counts": [],
                },
            )
            if int(group["n_bins"]) != int(plot_data["n_bins"]):
                print(
                    f"[preamble_fft] skip plot packet from {frame.get('file_name', iq_path.name)}: "
                    f"FFT length {plot_data['n_bins']} does not match group {group['n_bins']}"
                )
                continue

            group["spectra"].append(plot_data["spectrum"])
            group["file_names"].add(frame.get("file_name", iq_path.name))
            group["packet_labels"].append(
                f"{frame.get('file_stem', iq_path.stem)}#{frame.get('packet_index_in_file', len(group['spectra']) - 1)}"
            )
            group["symbol_counts"].append(int(plot_data["symbol_count"]))
            group.setdefault("packet_peak_offsets", []).append(int(plot_data["packet_peak_offset"]))
            group.setdefault("packet_peak_abs", []).append(float(plot_data["packet_peak_abs"]))
            group.setdefault("packets", []).append(
                {
                    "frame": dict(frame),
                    "x_offsets": plot_data["x_offsets"],
                    "spectrum": plot_data["spectrum"],
                    "n_bins": int(plot_data["n_bins"]),
                    "sf": int(plot_data["sf"]),
                    "symbol_count": int(plot_data["symbol_count"]),
                    "file_name": frame.get("file_name", iq_path.name),
                    "file_stem": frame.get("file_stem", iq_path.stem),
                    "position_id": position_id,
                    "packet_index_in_file": int(frame.get("packet_index_in_file", len(group["spectra"]) - 1)),
                    "packet_peak_bin": int(plot_data["packet_peak_bin"]),
                    "packet_peak_offset": int(plot_data["packet_peak_offset"]),
                    "packet_peak_abs": float(plot_data["packet_peak_abs"]),
                    "peak_bins": plot_data["peak_bins"],
                }
            )

    return groups


def finite_amplitude(values):
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def magnitude_to_peak_normalized(magnitude, reference=None):
    magnitude = finite_amplitude(magnitude)
    if reference is None:
        reference = float(np.max(magnitude)) if magnitude.size else 0.0
    reference = float(reference)
    if reference <= 0.0 or not np.isfinite(reference):
        return np.zeros_like(magnitude, dtype=np.float32)
    normalized = magnitude / reference
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def spectra_to_peak_normalized(spectra):
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim == 1:
        return magnitude_to_peak_normalized(spectra)
    peak_refs = np.max(finite_amplitude(spectra), axis=1)
    rows = [
        magnitude_to_peak_normalized(row, reference=peak_refs[index])
        for index, row in enumerate(spectra)
    ]
    return np.vstack(rows).astype(np.float32, copy=False)


def set_normalized_ylim(ax):
    ax.set_ylim(0.0, 1.05)


def add_location_fft_curves(ax, x_offsets, spectra_norm, mean_norm, std_norm, peak_offset, packet_color=LIGHT_BLUE):
    for spectrum_norm in spectra_norm:
        ax.plot(x_offsets, spectrum_norm, color=packet_color, alpha=0.24, linewidth=0.8)
    lower = np.maximum(mean_norm - std_norm, 0.0)
    upper = np.minimum(mean_norm + std_norm, 1.0)
    ax.fill_between(
        x_offsets,
        lower,
        upper,
        color=LIGHT_BLUE,
        alpha=0.26,
        linewidth=0,
        label="mean +/- std",
    )
    ax.plot(x_offsets, mean_norm, color=PEAK_BLUE, linewidth=1.9, label="mean")
    ax.axvline(int(peak_offset), color=PEAK_BLUE, linestyle=":", linewidth=1.1, label="mean peak")
    set_normalized_ylim(ax)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_ylabel("Peak-normalized magnitude")


def padded_id(value, width=2, prefix=""):
    value = int_or_default(value, None)
    if value is None:
        return f"{prefix}xx"
    sign = "m" if int(value) < 0 else ""
    return f"{prefix}{sign}{abs(int(value)):0{width}d}"


def packet_plot_filename(packet):
    frame = packet["frame"]
    packet_id = int(packet["packet_index_in_file"]) + 1
    return (
        f"{padded_id(frame.get('experiment_id', ''), 2, 'lab')}_"
        f"{padded_id(packet.get('sf', frame.get('sf', '')), 2, 'sf')}_"
        f"{padded_id(frame.get('tx_power_dbm', frame.get('filename_tx_power_dbm', '')), 2, 'tp')}_"
        f"{padded_id(frame.get('preamble_len', frame.get('filename_preamble_len', '')), 2, 'preamble')}_"
        f"{padded_id(packet.get('position_id', frame.get('position_id', '')), 2, 'locationid')}_"
        f"{padded_id(packet_id, 3, 'packet')}.png"
    )


def packet_location_dir(output_dir, packet):
    position_id = int_or_default(packet.get("position_id", packet["frame"].get("position_id", "")), -1)
    if position_id >= 0:
        return output_dir / f"locationid{position_id:02d}"
    return output_dir / safe_plot_name(packet.get("file_stem", "unknown_location"))


def save_packet_dechirp_fft_plot(packet, output_dir, args, plt):
    x_offsets = packet["x_offsets"]
    spectrum = packet["spectrum"]
    peak_offset = int(packet["packet_peak_offset"])
    peak_abs = float(packet["packet_peak_abs"])
    spectrum_norm = magnitude_to_peak_normalized(spectrum, reference=peak_abs)
    zoom_bins = max(0, int(args.plot_zoom_bins))
    zoom_mask = np.abs(x_offsets - peak_offset) <= zoom_bins

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 8.5),
        sharey=False,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        constrained_layout=True,
    )

    for ax, mask, title in (
        (axes[0], slice(None), "Complete raw FFT magnitude (peak-normalized)"),
        (axes[1], zoom_mask, f"Main-peak zoom (+/-{zoom_bins} bins)"),
    ):
        ax.plot(x_offsets[mask], spectrum_norm[mask], color=PEAK_BLUE, linewidth=1.3)
        ax.axvline(peak_offset, color=PEAK_BLUE, linestyle=":", linewidth=1.1)
        set_normalized_ylim(ax)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_ylabel("Peak-normalized magnitude")
        ax.set_title(title)
    axes[0].text(
        0.012,
        0.96,
        f"abs peak magnitude = {peak_abs:.6g}\nraw peak bin = {packet['packet_peak_bin']} ({peak_offset:+d})",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color=PEAK_BLUE,
        bbox={"facecolor": "white", "edgecolor": PEAK_BLUE, "alpha": 0.86, "boxstyle": "round,pad=0.3"},
    )
    axes[0].set_xlim(int(x_offsets[0]), int(x_offsets[-1]))
    axes[1].set_xlim(peak_offset - zoom_bins, peak_offset + zoom_bins)
    axes[1].set_xlabel("Raw FFT bin offset")

    frame = packet["frame"]
    packet_id = int(packet["packet_index_in_file"]) + 1
    fig.suptitle(
        f"{frame.get('lab_name', '')} | {packet['file_name']} | packet {packet_id} | "
        f"location {packet.get('position_id', '')} | SF{packet['sf']} | "
        f"{packet['symbol_count']} raw preamble FFTs averaged | abs peak {peak_abs:.4g}",
        fontsize=12,
    )

    location_dir = packet_location_dir(output_dir, packet)
    location_dir.mkdir(parents=True, exist_ok=True)
    path = location_dir / packet_plot_filename(packet)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_location_dechirp_fft_plot(group, output_dir, args, plt):
    x_offsets = group["x_offsets"]
    spectra = np.vstack(group["spectra"]).astype(np.float32, copy=False)
    spectra_norm = spectra_to_peak_normalized(spectra)
    mean_norm = np.nanmean(spectra_norm, axis=0)
    std_norm = np.nanstd(spectra_norm, axis=0)
    linear_mean = np.nanmean(spectra, axis=0)
    peak_offset = int(x_offsets[int(np.argmax(linear_mean))])
    packet_peaks = np.asarray(group.get("packet_peak_abs", []), dtype=np.float32)
    zoom_bins = max(0, int(args.plot_zoom_bins))
    zoom_mask = np.abs(x_offsets - peak_offset) <= zoom_bins

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 8.5),
        sharey=False,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        constrained_layout=True,
    )

    add_location_fft_curves(axes[0], x_offsets, spectra_norm, mean_norm, std_norm, peak_offset)
    axes[0].set_xlim(int(x_offsets[0]), int(x_offsets[-1]))
    file_names = ", ".join(sorted(group["file_names"]))
    axes[0].set_title(
        f"{group['label']} raw preamble dechirp FFT magnitude (peak-normalized) | {spectra.shape[0]} packets | "
        f"SF{group['sf']} | {group['n_bins']} bins | {file_names}"
    )
    if packet_peaks.size:
        axes[0].text(
            0.012,
            0.96,
            f"abs peak magnitude per packet\nmin/mean/max = "
            f"{float(np.min(packet_peaks)):.4g} / {float(np.mean(packet_peaks)):.4g} / {float(np.max(packet_peaks)):.4g}",
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color=PEAK_BLUE,
            bbox={"facecolor": "white", "edgecolor": PEAK_BLUE, "alpha": 0.86, "boxstyle": "round,pad=0.3"},
        )
    axes[0].legend(loc="lower right", frameon=False)

    add_location_fft_curves(
        axes[1],
        x_offsets[zoom_mask],
        spectra_norm[:, zoom_mask],
        mean_norm[zoom_mask],
        std_norm[zoom_mask],
        peak_offset,
    )
    axes[1].set_xlim(peak_offset - zoom_bins, peak_offset + zoom_bins)
    axes[1].set_xlabel("Raw FFT bin offset")
    axes[1].set_title(f"Main-peak zoom (+/-{zoom_bins} bins)")

    position_id = group["position_id"]
    if int(position_id) >= 0:
        name = f"location_{int(position_id):02d}_preamble_dechirp_fft.png"
    else:
        name = f"{safe_plot_name(group['label'])}_preamble_dechirp_fft.png"
    path = output_dir / name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_mean_comparison_plot(groups, output_dir, args, plt):
    compatible_groups = []
    first_n_bins = None
    for _, group in sorted(groups.items(), key=lambda item: (isinstance(item[0], str), item[0])):
        if not group["spectra"]:
            continue
        if first_n_bins is None:
            first_n_bins = int(group["n_bins"])
        if int(group["n_bins"]) != first_n_bins:
            print(
                f"[preamble_fft] skip {group['label']} in comparison plot: "
                f"FFT length {group['n_bins']} does not match {first_n_bins}"
            )
            continue
        compatible_groups.append(group)

    if len(compatible_groups) < 2:
        return None

    x_offsets = compatible_groups[0]["x_offsets"]
    zoom_bins = max(0, int(args.plot_zoom_bins))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.5, 8.5),
        sharey=False,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        constrained_layout=True,
    )
    color_map = plt.get_cmap("tab10")
    full_means_norm = []

    for index, group in enumerate(compatible_groups):
        spectra = np.vstack(group["spectra"]).astype(np.float32, copy=False)
        spectra_norm = spectra_to_peak_normalized(spectra)
        mean_norm = np.nanmean(spectra_norm, axis=0)
        peak_abs = float(np.mean(group.get("packet_peak_abs", [np.max(spectra)])))
        full_means_norm.append(mean_norm)
        label = f"{group['label']} (n={spectra.shape[0]}, peak={peak_abs:.4g})"
        color = color_map(index % 10)
        axes[0].plot(x_offsets, mean_norm, linewidth=1.6, color=color, label=label)

    combined_mean = np.nanmean(np.vstack(full_means_norm), axis=0)
    peak_offset = int(x_offsets[int(np.argmax(combined_mean))])
    zoom_mask = np.abs(x_offsets - peak_offset) <= zoom_bins
    for index, mean_norm in enumerate(full_means_norm):
        color = color_map(index % 10)
        axes[1].plot(
            x_offsets[zoom_mask],
            mean_norm[zoom_mask],
            linewidth=1.6,
            color=color,
            label=compatible_groups[index]["label"],
        )

    for ax in axes:
        ax.axvline(peak_offset, color=PEAK_BLUE, linestyle=":", linewidth=1.1)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_ylabel("Peak-normalized magnitude")
        set_normalized_ylim(ax)
    axes[0].set_xlim(int(x_offsets[0]), int(x_offsets[-1]))
    axes[0].set_title(f"Location mean raw preamble dechirp FFT comparison (peak-normalized) | {first_n_bins} bins")
    axes[0].legend(loc="lower right", frameon=False, ncol=2)
    axes[1].set_xlim(peak_offset - zoom_bins, peak_offset + zoom_bins)
    axes[1].set_xlabel("Raw FFT bin offset")
    axes[1].set_title(f"Main-peak zoom (+/-{zoom_bins} bins)")

    path = output_dir / "location_mean_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_dechirp_fft_plots(args, capture_results):
    output_dir = resolve_plot_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = collect_dechirp_fft_plot_groups(args, capture_results)
    if not groups:
        print("[preamble_fft] no packet spectra available for plotting")
        return {
            "packet_count": 0,
            "plot_count": 0,
            "plots": [],
            "output_dir": output_dir,
        }

    plt = import_matplotlib_pyplot()
    paths = []
    for _, group in sorted(groups.items(), key=lambda item: (isinstance(item[0], str), item[0])):
        if not group["spectra"]:
            continue
        for packet in group.get("packets", []):
            paths.append(save_packet_dechirp_fft_plot(packet, output_dir, args, plt))
        paths.append(save_location_dechirp_fft_plot(group, output_dir, args, plt))

    comparison_path = save_mean_comparison_plot(groups, output_dir, args, plt)
    if comparison_path is not None:
        paths.append(comparison_path)

    return {
        "packet_count": sum(len(group["spectra"]) for group in groups.values()),
        "plot_count": len(paths),
        "plots": paths,
        "output_dir": output_dir,
    }

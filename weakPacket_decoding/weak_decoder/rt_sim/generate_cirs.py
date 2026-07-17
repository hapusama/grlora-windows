#!/usr/bin/env python3
"""Generate reproducible Sionna RT CIRs for the aligned LoRa SER study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mitsuba as mi
import numpy as np
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene


DEFAULT_SCENE = (
    Path(__file__).resolve().parents[2]
    / "Buildings"
    / "B"
    / "sionna_scene"
    / "hospital.xml"
)
DEFAULT_OUTPUT = DEFAULT_SCENE.parent / "hospital_cirs.npz"

DEFAULT_RX = (3.30, 9.50, 0.20)
DEFAULT_TRANSMITTERS = {
    "target": (-0.20, 5.00, 0.20),
    "interferer_a": (6.00, 15.00, 0.20),
    "interferer_b": (-0.20, 15.00, 0.20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frequency", type=float, default=487.7e6)
    parser.add_argument("--bandwidth", type=float, default=125e3)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--samples-per-source", type=int, default=200_000)
    parser.add_argument("--max-paths", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def _single_link(
    scene_path: Path,
    tx_position: tuple[float, float, float],
    rx_position: tuple[float, float, float],
    frequency: float,
    bandwidth: float,
    max_depth: int,
    samples_per_source: int,
    max_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    scene = load_scene(str(scene_path))
    scene.frequency = float(frequency)
    scene.bandwidth = float(bandwidth)
    array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = array
    scene.rx_array = array
    transmitter = Transmitter(name="tx", position=mi.Point3f(*tx_position))
    receiver = Receiver(name="rx", position=mi.Point3f(*rx_position))
    scene.add(transmitter)
    scene.add(receiver)
    transmitter.look_at(receiver)

    paths = PathSolver()(
        scene,
        max_depth=max(0, int(max_depth)),
        max_num_paths_per_src=max(1, int(max_paths)),
        samples_per_src=max(1, int(samples_per_source)),
        synthetic_array=True,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=False,
        edge_diffraction=False,
        seed=int(seed),
    )
    coefficients, delays = paths.cir(normalize_delays=False, out_type="numpy")
    flat_delays = np.asarray(delays, dtype=np.float64).reshape(-1)
    flat_coefficients = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    if flat_coefficients.size != flat_delays.size:
        raise RuntimeError(
            f"unexpected CIR shapes: coefficients={coefficients.shape}, delays={delays.shape}"
        )
    valid = np.isfinite(flat_delays) & (flat_delays >= 0.0) & np.isfinite(flat_coefficients)
    flat_delays = flat_delays[valid]
    flat_coefficients = flat_coefficients[valid]
    order = np.argsort(flat_delays)
    return flat_coefficients[order], flat_delays[order]


def generate(args: argparse.Namespace) -> dict[str, object]:
    scene_path = args.scene.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    links: dict[str, dict[str, object]] = {}
    for link_index, (name, tx_position) in enumerate(DEFAULT_TRANSMITTERS.items()):
        coefficients, delays = _single_link(
            scene_path=scene_path,
            tx_position=tx_position,
            rx_position=DEFAULT_RX,
            frequency=float(args.frequency),
            bandwidth=float(args.bandwidth),
            max_depth=int(args.max_depth),
            samples_per_source=int(args.samples_per_source),
            max_paths=int(args.max_paths),
            seed=int(args.seed) + link_index,
        )
        if coefficients.size == 0:
            raise RuntimeError(f"Sionna RT found no path for link {name}")
        arrays[f"{name}_coefficients"] = coefficients
        arrays[f"{name}_delays_s"] = delays
        coherent_gain = complex(np.sum(coefficients))
        links[name] = {
            "tx_position": list(tx_position),
            "rx_position": list(DEFAULT_RX),
            "path_count": int(coefficients.size),
            "first_delay_s": float(delays.min()),
            "last_delay_s": float(delays.max()),
            "rms_path_gain": float(np.sqrt(np.sum(np.abs(coefficients) ** 2))),
            "coherent_gain_real": float(coherent_gain.real),
            "coherent_gain_imag": float(coherent_gain.imag),
            "incoherent_path_power_db": float(
                10.0 * np.log10(max(float(np.sum(np.abs(coefficients) ** 2)), 1e-300))
            ),
        }

    metadata = {
        "scene": str(scene_path),
        "frequency_hz": float(args.frequency),
        "bandwidth_hz": float(args.bandwidth),
        "max_depth": int(args.max_depth),
        "samples_per_source": int(args.samples_per_source),
        "max_paths": int(args.max_paths),
        "seed": int(args.seed),
        "mitsuba_variant": str(mi.variant()),
        "links": links,
        "note": (
            "CIRs contain deterministic propagation only. Thermal receiver noise "
            "and interfering waveforms are generated by the SER experiment."
        ),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=True))
    np.savez_compressed(output_path, **arrays)
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    args = parse_args()
    print(json.dumps(generate(args), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

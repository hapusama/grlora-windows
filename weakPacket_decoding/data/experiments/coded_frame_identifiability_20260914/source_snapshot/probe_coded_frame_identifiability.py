"""Audit full-frame alias identifiability, including CRC and fixed padding.

This is a finite-code calculation for the repository PHY encoder, known frame
configuration, ideal synchronized chirps and residue-only observations. It is
not a universal analog sampling lower bound. Hardware waveform distortions or
cross-symbol phase information can supply information absent from this model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import probe_coded_undersampling as gate


def observe(payload, sf, downclock, crc_mode):
    header, body = gate.encode_explicit_frame_symbols(payload, sf, 1, True, sf == 12, crc_mode)
    lost = downclock.bit_length()-1
    return np.array([(symbol >> bit) & 1
                     for values, sf_app in ((header, sf-2), (body, sf-2 if sf == 12 else sf))
                     for symbol in values for bit in range(sf_app-lost)], dtype=np.uint8)


def build_system(length, sf, downclock, crc_mode):
    origin = observe(bytes(length), sf, downclock, crc_mode)
    columns = []
    for bit in range(8*length):
        payload = bytearray(length)
        payload[bit//8] = 1 << (bit % 8)
        columns.append(observe(payload, sf, downclock, crc_mode) ^ origin)
    matrix = np.array(columns, dtype=np.uint8).T
    reduced = matrix.copy()
    transform = np.eye(len(matrix), dtype=np.uint8)
    pivots = []
    for column in range(matrix.shape[1]):
        row = len(pivots)
        candidates = np.flatnonzero(reduced[row:, column])
        if len(candidates) == 0:
            continue
        pivot = row + int(candidates[0])
        reduced[[row, pivot]] = reduced[[pivot, row]]
        transform[[row, pivot]] = transform[[pivot, row]]
        others = np.flatnonzero(reduced[:, column])
        others = others[others != row]
        reduced[others] ^= reduced[row]
        transform[others] ^= transform[row]
        pivots.append(column)
        if len(pivots) == len(matrix):
            break
    return origin, matrix, reduced, transform, pivots


def bits_to_bytes(bits):
    return bytes(np.sum(bits.reshape(-1, 8) * (1 << np.arange(8)), axis=1).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output directory")
    args.output.mkdir(parents=True)
    configurations = [(sf, length, 4, mode) for sf in (7, 10, 12)
                      for length in (4, 8, 12, 16, 20, 24, 28, 33)
                      for mode in ("grlora", "sx1276")]
    configurations += [(10, 33, d, "grlora") for d in (1, 2, 8)]
    rng = np.random.default_rng(914031)
    rows = []
    collisions = []
    for sf, length, downclock, crc_mode in configurations:
        origin, matrix, reduced, transform, pivots = build_system(length, sf, downclock, crc_mode)
        unique = len(pivots) == 8*length
        # Random complete frames verify the affine map; recover payload when unique.
        for _ in range(16):
            payload = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
            bits = ((np.frombuffer(payload, dtype=np.uint8)[:, None] >> np.arange(8)) & 1).reshape(-1).astype(np.uint8)
            observed = observe(payload, sf, downclock, crc_mode)
            if np.any(observed != (((matrix @ bits) & 1) ^ origin)):
                raise AssertionError("full PHY does not match affine observation map")
            if unique:
                rhs = (transform @ (observed ^ origin)) & 1
                decoded = np.zeros(8*length, dtype=np.uint8)
                decoded[pivots] = rhs[:len(pivots)]
                if bits_to_bytes(decoded) != payload:
                    raise AssertionError("unique full-frame recovery failed")
        if not unique:
            free = next(c for c in range(8*length) if c not in pivots)
            vector = np.zeros(8*length, dtype=np.uint8)
            vector[free] = 1
            vector[pivots] = reduced[:len(pivots), free]
            first, second = bytes(length), bits_to_bytes(vector)
            if first == second or np.any(observe(first, sf, downclock, crc_mode) != observe(second, sf, downclock, crc_mode)):
                raise AssertionError("invalid alias-collision witness")
            if (sf, length, downclock, crc_mode) == (10, 33, 4, "grlora"):
                collisions.append(dict(sf=sf, payload_length=length, downclock=downclock, crc_mode=crc_mode,
                                       payload_a_hex=first.hex(), payload_b_hex=second.hex(),
                                       statement="different payloads reencoded with valid CRC have identical observed residues"))
        rows.append(dict(sf=sf, ldro=int(sf==12), cr=1, payload_length=length, downclock=downclock,
                         crc_mode=crc_mode, payload_bits=8*length, rank=len(pivots),
                         ambiguity_bits=8*length-len(pivots), random_affine_checks=16,
                         random_unique_payload_recoveries=16 if unique else 0))
    gate.common.write_csv(args.output / "frame_ranks.csv", rows)
    (args.output / "collision_witnesses.json").write_text(json.dumps(collisions, indent=2), encoding="utf-8")
    sources = [Path(__file__), Path(gate.__file__)]
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        (snapshot / source.name).write_bytes(source.read_bytes())
    (args.output / "config.json").write_text(json.dumps(dict(
        scope=__doc__, cr=1, has_crc=True, seed=914031,
        crc_modes="repository grlora and sx1276 conventions audited separately; not a new hardware verification of either convention",
        known="payload length, CR, LDRO, CRC convention, symbol boundaries and CFO; zero padding as generated by local encoder",
        crc_usage="CRC is an inner code constraint; it is NOT an independent final error detector in this experiment",
        evidence="rank is exact for this binary map; correctness checked against random payload bytes; no RF/noise experiment here",
        random_affine_checks=sum(r["random_affine_checks"] for r in rows),
        source_sha256={s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
    ), indent=2), encoding="utf-8")
    print(json.dumps([r for r in rows if r["sf"] == 10 and r["crc_mode"] == "grlora"], indent=2))


if __name__ == "__main__":
    main()

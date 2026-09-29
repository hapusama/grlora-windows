"""Audit full-frame alias identifiability, including CRC and unknown padding.

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
from weak_decoder.decoding import payload_codec as codec


def padding_count(length, sf):
    return (-(5 + 2*length + 4 - (sf-2))) % (sf-2 if sf == 12 else sf)


def observe(payload, sf, downclock, crc_mode, padding=None):
    if padding is None:
        header, body = gate.encode_explicit_frame_symbols(payload, sf, 1, True, sf == 12, crc_mode)
    else:
        nibbles = codec.compute_header_nibbles(len(payload), 1, True)
        nibbles.extend(codec.payload_bytes_to_whitened_nibbles(payload))
        nibbles.extend(codec.grlora_crc_nibbles(payload) if crc_mode == "grlora" else codec.sx1276_crc_nibbles(payload))
        nibbles.extend(padding)
        words = codec.hamming_encode_nibbles(nibbles, sf, 1)
        interleaved = codec.interleave_codewords(words, sf, 1, sf == 12)
        symbols = codec.interleaved_to_fft_demod_symbols(interleaved, sf, sf == 12)
        header, body = symbols[:8], symbols[8:]
    lost = downclock.bit_length()-1
    return np.array([(symbol >> bit) & 1
                     for values, sf_app in ((header, sf-2), (body, sf-2 if sf == 12 else sf))
                     for symbol in values for bit in range(sf_app-lost)], dtype=np.uint8)


def binary_rank(matrix):
    a = matrix.copy()
    row = 0
    for column in range(a.shape[1]):
        possible = np.flatnonzero(a[row:, column])
        if len(possible) == 0:
            continue
        pivot = row+int(possible[0])
        a[[row, pivot]] = a[[pivot, row]]
        other = np.flatnonzero(a[row+1:, column])+row+1
        a[other] ^= a[row]
        row += 1
        if row == len(a):
            break
    return row


def build_system(length, sf, downclock, crc_mode, unknown_padding=False):
    count = padding_count(length, sf) if unknown_padding else 0
    origin = observe(bytes(length), sf, downclock, crc_mode, [0]*count if unknown_padding else None)
    columns = []
    for bit in range(8*length + 4*count):
        payload = bytearray(length)
        padding = [0]*count if unknown_padding else None
        if bit < 8*length:
            payload[bit//8] = 1 << (bit % 8)
        else:
            padding[(bit-8*length)//4] = 1 << ((bit-8*length) % 4)
        columns.append(observe(payload, sf, downclock, crc_mode, padding) ^ origin)
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
    configurations = [(*c, unknown) for c in configurations for unknown in (False, True)]
    rng = np.random.default_rng(914031)
    rows = []
    collisions = []
    for sf, length, downclock, crc_mode, unknown in configurations:
        origin, matrix, reduced, transform, pivots = build_system(length, sf, downclock, crc_mode, unknown)
        nuisance_rank = binary_rank(matrix[:, 8*length:])
        effective_rank = len(pivots)-nuisance_rank
        unique = effective_rank == 8*length
        # Random complete frames verify the affine map; recover payload when unique.
        for _ in range(16):
            payload = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
            bits = ((np.frombuffer(payload, dtype=np.uint8)[:, None] >> np.arange(8)) & 1).reshape(-1).astype(np.uint8)
            padding = rng.integers(0, 16, padding_count(length, sf), dtype=np.uint8) if unknown else None
            if padding is not None:
                bits = np.concatenate((bits, ((padding[:, None] >> np.arange(4)) & 1).reshape(-1)))
            observed = observe(payload, sf, downclock, crc_mode, padding)
            if np.any(observed != (((matrix @ bits) & 1) ^ origin)):
                raise AssertionError("full PHY does not match affine observation map")
            if unique:
                rhs = (transform @ (observed ^ origin)) & 1
                decoded = np.zeros(matrix.shape[1], dtype=np.uint8)
                decoded[pivots] = rhs[:len(pivots)]
                if bits_to_bytes(decoded[:8*length]) != payload:
                    raise AssertionError("unique full-frame recovery failed")
        if not unique:
            for free in (c for c in range(matrix.shape[1]) if c not in pivots):
                vector = np.zeros(matrix.shape[1], dtype=np.uint8)
                vector[free] = 1
                vector[pivots] = reduced[:len(pivots), free]
                if np.any(vector[:8*length]):
                    break
            first, second = bytes(length), bits_to_bytes(vector[:8*length])
            first_pad = [0]*padding_count(length, sf) if unknown else None
            second_pad = np.sum(vector[8*length:].reshape(-1, 4)*(1 << np.arange(4)), axis=1).tolist() if unknown else None
            if first == second or np.any(observe(first, sf, downclock, crc_mode, first_pad) != observe(second, sf, downclock, crc_mode, second_pad)):
                raise AssertionError("invalid alias-collision witness")
            if (sf, length, downclock, crc_mode) == (10, 33, 4, "grlora"):
                collisions.append(dict(sf=sf, payload_length=length, downclock=downclock, crc_mode=crc_mode,
                                       payload_a_hex=first.hex(), payload_b_hex=second.hex(),
                                       unknown_padding=unknown, padding_a=first_pad, padding_b=second_pad,
                                       statement="different payloads reencoded with valid CRC have identical observed residues"))
        rows.append(dict(sf=sf, ldro=int(sf==12), cr=1, payload_length=length, downclock=downclock,
                         crc_mode=crc_mode, unknown_padding=int(unknown), payload_bits=8*length,
                         rank=len(pivots), nuisance_rank=nuisance_rank, effective_payload_rank=effective_rank,
                         ambiguity_bits=8*length-effective_rank, random_affine_checks=16,
                         random_unique_payload_recoveries=16 if unique else 0))
    gate.common.write_csv(args.output / "frame_ranks.csv", rows)
    ota_checks = []
    for dataset in gate.common.DEFAULT_DATASETS:
        _, csv_path = gate.common.dataset_paths(dataset)
        for packet in gate.common.load_packets(csv_path):
            sf = packet["sf"]
            header = [s["symbol_value"] for s in packet["header_symbols"]]
            body = [s["symbol_value"] for s in packet["payload_symbols"]]
            frame = gate.decode_explicit_frame_symbols(header, body, sf, 125000, int(packet["ldro"]), "grlora")
            if not frame.payload.crc_valid:
                raise AssertionError("OTA reference CRC failed")
            payload = frame.payload.payload_bytes
            count = padding_count(len(payload), sf)
            padding = frame.frame_nibbles[-count:] if count else []
            actual = np.array([(s >> bit) & 1 for values, width in ((header, sf-2), (body, sf))
                               for s in values for bit in range(width)], dtype=np.uint8)
            predicted = observe(payload, sf, 1, "grlora", padding)
            if np.any(actual != predicted):
                raise AssertionError("unknown-padding model does not reproduce OTA symbols")
            zero_header, zero_body = gate.encode_explicit_frame_symbols(payload, sf, 1, True, False, "grlora")
            ota_checks.append(dict(dataset=dataset, packet_index=packet["packet_index"],
                                   default_zero_padding_symbol_mismatch=sum(a != b for a, b in zip(header+body, zero_header+zero_body)),
                                   unknown_padding_observation_mismatch=int(np.sum(actual != predicted))))
    gate.common.write_csv(args.output / "ota_padding_checks.csv", ota_checks)
    (args.output / "collision_witnesses.json").write_text(json.dumps(collisions, indent=2), encoding="utf-8")
    sources = [Path(__file__), Path(gate.__file__)]
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        (snapshot / source.name).write_bytes(source.read_bytes())
    (args.output / "config.json").write_text(json.dumps(dict(
        scope=__doc__, cr=1, has_crc=True, seed=914031,
        crc_modes="repository grlora and sx1276 conventions audited separately; not a new hardware verification of either convention",
        known="payload length, CR, LDRO, CRC convention, symbol boundaries and CFO",
        padding="fixed zero padding and unknown Hamming-coded padding nibbles analyzed separately; OTA padding differs from the zero-padding encoder",
        crc_usage="CRC is an inner code constraint; it is NOT an independent final error detector in this experiment",
        evidence="rank is exact for this binary map; correctness checked against random payload bytes; no RF/noise experiment here",
        random_affine_checks=sum(r["random_affine_checks"] for r in rows),
        ota_padding_checks=len(ota_checks),
        source_sha256={s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
    ), indent=2), encoding="utf-8")
    print(json.dumps([r for r in rows if r["sf"] == 10 and r["crc_mode"] == "grlora"], indent=2))


if __name__ == "__main__":
    main()

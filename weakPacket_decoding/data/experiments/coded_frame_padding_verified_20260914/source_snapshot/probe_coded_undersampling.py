"""Identifiability gate for coded uniform undersampling of LoRa.

This is an oracle-timing experiment. It decodes both header and payload from
retained observations, but uses clean symbol boundaries and CFO. It is not an
end-to-end low-rate receiver or a sensitivity claim.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from weak_decoder.baselines import common
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import _oversampled_downchirp
from weak_decoder.decoding.header_first_demod import decode_explicit_header
from weak_decoder.decoding.payload_codec import (
    decode_explicit_frame_symbols, encode_explicit_frame_symbols, encode_hamming_nibble,
)


def encode_block(bits, sf_app, cr):
    """Hamming, diagonal interleaver and inverse Gray map, without +1 bin."""
    width = int(cr) + 4
    words = [encode_hamming_nibble(sum(int(bits[4*r+j]) << (3-j) for j in range(4)), cr)
             for r in range(sf_app)]
    symbols = []
    for column in range(width):
        gray = sum(((words[(column-j-1) % sf_app] >> (width-1-column)) & 1)
                   << (sf_app-1-j) for j in range(sf_app))
        symbol = gray
        for shift in range(1, sf_app):
            symbol ^= gray >> shift
        symbols.append(symbol)
    return symbols


@lru_cache(maxsize=None)
def observation_system(sf_app, cr, downclock):
    """Return the binary linear map from data nibbles to observable residues."""
    lost = int(downclock).bit_length() - 1
    if downclock != 1 << lost or lost >= sf_app:
        raise ValueError("downclock must be a supported power of two")
    kept = sf_app - lost
    columns = []
    for index in range(4 * sf_app):
        bits = np.zeros(4 * sf_app, dtype=np.uint8)
        bits[index] = 1
        symbols = encode_block(bits, sf_app, cr)
        columns.append([(symbol >> shift) & 1 for symbol in symbols for shift in range(kept)])
    matrix = np.array(columns, dtype=np.uint8).T
    reduced = matrix.copy()
    transform = np.eye(len(matrix), dtype=np.uint8)
    pivots = []
    for column in range(matrix.shape[1]):
        row = len(pivots)
        possible = np.flatnonzero(reduced[row:, column])
        if len(possible) == 0:
            continue
        pivot = row + int(possible[0])
        reduced[[row, pivot]] = reduced[[pivot, row]]
        transform[[row, pivot]] = transform[[pivot, row]]
        for other in range(len(matrix)):
            if other != row and reduced[other, column]:
                reduced[other] ^= reduced[row]
                transform[other] ^= transform[row]
        pivots.append(column)
        if len(pivots) == len(matrix):
            break
    return matrix, transform, tuple(pivots), kept


def recover_block(residues, sf_app, cr, downclock):
    """Solve hard residue constraints; reject nonunique or inconsistent blocks."""
    matrix, transform, pivots, kept = observation_system(sf_app, cr, downclock)
    if len(residues) != cr + 4:
        return None, "incomplete_block"
    if len(pivots) != 4 * sf_app:
        return None, "nonidentifiable"
    observations = np.array([(int(symbol) >> shift) & 1 for symbol in residues
                             for shift in range(kept)], dtype=np.uint8)
    rhs = (transform.astype(np.int64) @ observations) % 2
    if np.any(rhs[len(pivots):]):
        return None, "inconsistent"
    bits = np.zeros(4 * sf_app, dtype=np.uint8)
    bits[list(pivots)] = rhs[:len(pivots)]
    if np.any((matrix.astype(np.int64) @ bits) % 2 != observations):
        raise AssertionError("GF(2) solution does not reproduce observations")
    return encode_block(bits, sf_app, cr), "unique"


def self_check():
    """Compare against the repository's separate full PHY encoder and decoder."""
    rng = np.random.default_rng(913431)
    results = []
    for sf in (7, 10, 12):
        for cr in (1, 2, 3, 4):
            for downclock in (1, 2, 4, 8):
                correct = 0
                for _ in range(8):
                    payload = rng.integers(0, 256, 33, dtype=np.uint8).tobytes()
                    ldro = sf == 12
                    head, body = encode_explicit_frame_symbols(payload, sf, cr, True, ldro, "grlora")
                    head_residues = [s % (1 << (sf-2)) // 1 % ((1 << (sf-2)) // downclock) for s in head]
                    h, status = recover_block(head_residues, sf-2, 4, downclock)
                    p = []
                    for offset in range(0, len(body), cr+4):
                        sf_app = sf-2 if ldro else sf
                        residues = [s % ((1 << sf_app) // downclock) for s in body[offset:offset+cr+4]]
                        values, state = recover_block(residues, sf_app, cr, downclock)
                        if values is None:
                            p = None
                            break
                        p.extend(values)
                    if h is not None and p is not None:
                        frame = decode_explicit_frame_symbols(h, p, sf, 125000, 1 if ldro else 0, "grlora")
                        if frame.payload.payload_bytes != payload or not frame.payload.crc_valid:
                            raise AssertionError("unique alias solution disagrees with independent PHY codec")
                        correct += 1
                results.append(dict(sf=sf, cr=cr, downclock=downclock, trials=8, exact=correct))
    return results


def symbol_residue(iq, item, packet, downclock, is_header):
    """FFT of retained q=0 samples only; no hardware phase fingerprint."""
    sf, oversampling = int(packet["sf"]), int(packet["os_factor"])
    n_bins = 1 << sf
    indices = np.arange(0, n_bins * oversampling, oversampling * downclock)
    start = int(item["start_sample"]) + oversampling // 2
    reference = _oversampled_downchirp(sf, oversampling, packet["cfo_int"], packet["cfo_frac"])
    observed = np.asarray(iq[start + indices], dtype=np.complex64) * reference[indices]
    power = np.abs(np.fft.fft(observed)) ** 2
    alias_bin = int(np.argmax(power))
    divisor = 4 if is_header or packet["ldro"] else 1
    return ((alias_bin - 1) % len(power)) // divisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output directory")
    args.output.mkdir(parents=True)
    ranks = []
    for sf_app in range(5, 13):
        for cr in (1, 2, 3, 4):
            for downclock in (1, 2, 4, 8):
                matrix, _, pivots, _ = observation_system(sf_app, cr, downclock)
                ranks.append(dict(sf_app=sf_app, cr=cr, downclock=downclock,
                                  message_bits=4*sf_app, observed_bits=matrix.shape[0],
                                  rank=len(pivots), ambiguity_bits=4*sf_app-len(pivots)))
    synthetic = self_check()
    rows = []
    for dataset in common.DEFAULT_DATASETS:
        iq_path, csv_path = common.dataset_paths(dataset)
        iq = np.memmap(iq_path, dtype=np.complex64, mode="r")
        for packet in common.load_packets(csv_path):
            reference = decode_explicit_frame_symbols(
                [s["symbol_value"] for s in packet["header_symbols"]],
                [s["symbol_value"] for s in packet["payload_symbols"]],
                packet["sf"], 125000, int(packet["ldro"]), "grlora",
            )
            if not reference.payload.crc_valid:
                raise ValueError("clean reference CRC failure")
            for downclock in (1, 2, 4, 8):
                head_residues = [symbol_residue(iq, s, packet, downclock, True) for s in packet["header_symbols"]]
                h, status = recover_block(head_residues, packet["sf"]-2, 4, downclock)
                body = []
                header_valid = False
                if h is not None:
                    decoded_header = decode_explicit_header(h, packet["sf"], 125000, int(packet["ldro"]))
                    header_valid = decoded_header.header_valid and 1 <= decoded_header.cr <= 4
                    if header_valid:
                        cr = int(decoded_header.cr)
                        # Timing and symbol windows remain supplied by the clean locator.
                        for offset in range(0, len(packet["payload_symbols"]), cr+4):
                            items = packet["payload_symbols"][offset:offset+cr+4]
                            residues = [symbol_residue(iq, s, packet, downclock, False) for s in items]
                            values, status = recover_block(residues, packet["sf"]-2 if decoded_header.ldro else packet["sf"], cr, downclock)
                            if values is None:
                                body = None
                                break
                            body.extend(values)
                accepted = exact = False
                if header_valid and body is not None:
                    decoded = decode_explicit_frame_symbols(h, body, packet["sf"], 125000, int(packet["ldro"]), "grlora")
                    accepted = decoded.header.header_valid and decoded.header.has_crc and decoded.payload.crc_valid
                    exact = accepted and decoded.payload.payload_bytes == reference.payload.payload_bytes
                rows.append(dict(dataset=dataset, packet_index=packet["packet_index"], downclock=downclock,
                                 samples_per_symbol=(1 << packet["sf"]) // downclock,
                                 header_valid=int(header_valid), status=status, crc_accept=int(accepted), exact=int(exact)))
        print(f"completed {dataset}", flush=True)
    for name, values in (("rank_audit", ranks), ("synthetic_checks", synthetic), ("ota_clean", rows)):
        with (args.output / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = [dict(downclock=d, trials=sum(r["downclock"] == d for r in rows),
                    exact=sum(r["exact"] for r in rows if r["downclock"] == d)) for d in (1, 2, 4, 8)]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "config.json").write_text(json.dumps(dict(
        scope="oracle timing/CFO and known clean frame extents; header and payload decoded from retained samples; no added noise",
        sampling="q=0 uniform every R*D samples; complex ADC average rate B/D; analogue bandwidth must still pass LoRa band",
        decoder="joint GF(2) Hamming/interleaver/inverse-Gray constraints; rejects inconsistent or rank-deficient blocks",
        datasets=list(common.DEFAULT_DATASETS), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        synthetic_independent_codec_checks=sum(r["trials"] for r in synthetic),
    ), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

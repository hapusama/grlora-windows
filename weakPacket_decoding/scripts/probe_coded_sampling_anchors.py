"""Gate a code-constrained alias shortlist against equal-budget full searches.

SF10/CR1 payload only. Header uses B-rate samples and is actually decoded.
Clean boundaries, CFO, LDRO and frame extents are supplied. This tests sample
and computation tradeoffs, not RF sensitivity, ADC power, or blind acquisition.
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import probe_coded_undersampling as gate
import probe_coded_undersampling_noise as noise_gate


@lru_cache(maxsize=None)
def solution_system(sf_app, cr, downclock):
    matrix, transform, pivots, kept = gate.observation_system(sf_app, cr, downclock)
    reduced = (transform @ matrix) & 1
    free = [c for c in range(matrix.shape[1]) if c not in pivots]
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    for index, column in enumerate(free):
        basis[index, column] = 1
        basis[index, list(pivots)] = reduced[:len(pivots), column]
    choices = noise_gate.candidate_choices(len(free)).astype(np.uint8)
    nullspace = (choices @ basis) & 1
    return transform, pivots, kept, nullspace


def block_candidates(powers, sf_app=10, cr=1, downclock=4):
    """Keep every valid data vector under the top-two residue combinations."""
    transform, pivots, kept, nullspace = solution_system(sf_app, cr, downclock)
    spectra = np.asarray(powers)
    top = np.argpartition(spectra, -2, axis=1)[:, -2:]
    residues = top[np.arange(cr+4)[None, :], noise_gate.candidate_choices(cr+4)]
    observed = ((residues[:, :, None] >> np.arange(kept)) & 1).reshape(len(residues), -1).astype(np.uint8)
    rhs = (observed @ transform.T) & 1
    valid = np.flatnonzero(~np.any(rhs[:, len(pivots):], axis=1))
    if len(valid) == 0:
        return np.empty((0, cr+4), dtype=np.int64)
    particular = np.zeros((len(valid), 4*sf_app), dtype=np.uint8)
    particular[:, list(pivots)] = rhs[valid, :len(pivots)]
    bits = (particular[:, None, :] ^ nullspace[None, :, :]).reshape(-1, 4*sf_app)
    full_map = gate.observation_system(sf_app, cr, 1)[0]
    symbol_bits = ((bits @ full_map.T) & 1).reshape(len(bits), cr+4, sf_app)
    return np.sum(symbol_bits * (1 << np.arange(sf_app)), axis=2)


def shortlist_decode(dechirped, anchors, anchor_bank):
    main = np.fft.fft(dechirped[:, ::4], axis=1)
    residue_power = np.roll(np.abs(main)**2, -1, axis=1)
    body = []
    largest_list = 0
    for offset in range(0, len(dechirped), 5):
        candidates = block_candidates(residue_power[offset:offset+5])
        largest_list = max(largest_list, len(candidates))
        if len(candidates) == 0:
            return None, largest_list
        raw = (candidates + 1) % 1024
        scores = np.zeros(len(candidates))
        for column in range(5):
            bins = raw[:, column]
            anchor_projection = anchor_bank[bins] @ dechirped[offset+column, anchors]
            total = main[offset+column, bins % 256] + anchor_projection
            scores += np.abs(total)**2
        body.extend(candidates[int(np.argmax(scores))].tolist())
    return body, largest_list


def full_search_decode(dechirped, positions, bank):
    raw_power = np.abs(dechirped[:, positions] @ bank.T)**2
    semantic_power = np.roll(raw_power, -1, axis=1)
    body = []
    for offset in range(0, len(dechirped), 5):
        symbols = noise_gate.list_block(semantic_power[offset:offset+5], 10, 1, 1)
        if symbols is None:
            return None
        body.extend(symbols)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output directory")
    args.output.mkdir(parents=True)
    n = 1024
    # Fixed independently of captures, labels, and noise. Anchors all use coset 1.
    anchors = np.arange(16) * 64 + 1
    hybrid = np.sort(np.concatenate((np.arange(0, n, 4), anchors)))
    random_pattern = np.sort(np.random.default_rng(913461).choice(n, len(hybrid), replace=False))
    tones = np.exp(-2j*np.pi*np.arange(n)[:, None]*np.arange(n)[None, :]/n)
    anchor_bank = tones[:, anchors]
    hybrid_bank = tones[:, hybrid]
    random_bank = tones[:, random_pattern]
    # Algebra and shortlist sanity on independent random messages.
    rng = np.random.default_rng(913462)
    for _ in range(100):
        bits = rng.integers(0, 2, 40, dtype=np.uint8)
        symbols = np.array(gate.encode_block(bits, 10, 1))
        ideal = np.conj(tones[(symbols+1) % n])
        decoded, count = shortlist_decode(ideal, anchors, anchor_bank)
        if decoded != symbols.tolist():
            raise AssertionError("ideal random block not recovered")
    seeds = (913471, 913472, 913473)
    levels = (None, -7, -10, -13, -16)
    rows = []
    start_time = time.perf_counter()
    for dataset_index, dataset in enumerate(gate.common.DEFAULT_DATASETS):
        iq_path, csv_path = gate.common.dataset_paths(dataset)
        iq = np.memmap(iq_path, dtype=np.complex64, mode="r")
        for packet in gate.common.load_packets(csv_path):
            if (packet["sf"], packet["cr"], packet["ldro"]) != (10, 1, False):
                raise ValueError("this gate is for SF10 CR1 without LDRO")
            items = packet["header_symbols"] + packet["payload_symbols"]
            rate = packet["os_factor"]
            indices = np.arange(n)*rate + rate//2
            clean = np.stack([np.asarray(iq[int(s["start_sample"])+indices]) for s in items])
            reference = gate._oversampled_downchirp(10, rate, packet["cfo_int"], packet["cfo_frac"])[::rate]
            target = gate.decode_explicit_frame_symbols(
                [s["symbol_value"] for s in packet["header_symbols"]],
                [s["symbol_value"] for s in packet["payload_symbols"]], 10, 125000, 0, "grlora",
            ).payload.payload_bytes
            power = float(np.mean(np.abs(clean.astype(np.complex128))**2))
            for seed in seeds:
                rng = np.random.default_rng(np.random.SeedSequence([seed, dataset_index, packet["packet_index"]]))
                unit_noise = (rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape))/np.sqrt(2)
                for rgr in levels:
                    if rgr is None and seed != seeds[0]:
                        continue
                    samples = clean if rgr is None else clean + unit_noise*np.sqrt(power*10**(-rgr/10))
                    dechirped = samples*reference[None, :]
                    head_power = noise_gate.demodulate(samples[:8], reference, 1, 10, False)
                    header = noise_gate.list_block(head_power, 8, 4, 1)
                    header_valid = False
                    if header is not None:
                        fields = gate.decode_explicit_header(header, 10, 125000, 0)
                        header_valid = fields.header_valid and fields.cr == 1
                    for method in ("uniform512_joint", "random272_full_joint", "hybrid272_full_joint", "hybrid272_shortlist"):
                        begin = time.perf_counter()
                        largest_list = 0
                        body = None
                        if header_valid:
                            if method == "uniform512_joint":
                                powers = np.roll(np.abs(np.fft.fft(dechirped[8:, ::2], axis=1))**2, -1, axis=1)
                                body = []
                                for offset in range(0, len(powers), 5):
                                    block = noise_gate.list_block(powers[offset:offset+5], 10, 1, 2)
                                    if block is None:
                                        body = None
                                        break
                                    body.extend(block)
                            elif method == "random272_full_joint":
                                body = full_search_decode(dechirped[8:], random_pattern, random_bank)
                            elif method == "hybrid272_full_joint":
                                body = full_search_decode(dechirped[8:], hybrid, hybrid_bank)
                            else:
                                body, largest_list = shortlist_decode(dechirped[8:], anchors, anchor_bank)
                        elapsed = time.perf_counter() - begin
                        accepted = exact = False
                        if body is not None:
                            frame = gate.decode_explicit_frame_symbols(header, body, 10, 125000, 0, "grlora")
                            accepted = frame.header.header_valid and frame.header.has_crc and frame.payload.crc_valid
                            exact = accepted and frame.payload.payload_bytes == target
                        rows.append(dict(dataset=dataset, packet_index=packet["packet_index"], seed=seed,
                                         rgr_db="clean" if rgr is None else rgr, method=method,
                                         header_valid=int(header_valid), exact=int(exact),
                                         false_accept=int(accepted and not exact),
                                         payload_demod_decode_seconds=elapsed, max_block_candidates=largest_list))
        print(f"completed {dataset}: {time.perf_counter()-start_time:.1f}s", flush=True)
    summary = []
    for rgr in ("clean", -7, -10, -13, -16):
        for method in ("uniform512_joint", "random272_full_joint", "hybrid272_full_joint", "hybrid272_shortlist"):
            group = [r for r in rows if (r["rgr_db"], r["method"]) == (rgr, method)]
            summary.append(dict(rgr_db=rgr, method=method, trials=len(group), exact=sum(r["exact"] for r in group),
                                false_accept=sum(r["false_accept"] for r in group),
                                median_payload_seconds=float(np.median([r["payload_demod_decode_seconds"] for r in group]))))
    gate.common.write_csv(args.output / "packet_trials.csv", rows)
    gate.common.write_csv(args.output / "summary.csv", summary)
    sources = [Path(__file__), Path(gate.__file__), Path(noise_gate.__file__)]
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        (snapshot / source.name).write_bytes(source.read_bytes())
    (args.output / "config.json").write_text(json.dumps(dict(
        scope="oracle timing/CFO, LDRO and frame extent; B-rate decoded header common to all methods",
        noise="paired iid circular Gaussian at B-rate complex grid; RGR includes original noise",
        sampling="payload patterns on B-rate grid, header 1024 complex samples per symbol for all methods",
        hybrid=hybrid.tolist(), anchors=anchors.tolist(), random_pattern=random_pattern.tolist(),
        shortlist="top 2 of 256 residues per symbol; all GF(2)-valid CR1 blocks including 16-way nullspace; retained-sample energy score",
        full_search="all 1024 symbol correlations, then top 2 full-symbol exact block decoder; finite list, not full ML",
        crc="final acceptance only; reference payload only scores exact recovery",
        independent_synthetic_blocks=100, seeds=seeds, rgr_db=levels,
        timing="single run, cached dictionaries; payload demodulation/selection only, excludes preprocessing and final codec; not end-to-end latency",
        source_sha256={s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
    ), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

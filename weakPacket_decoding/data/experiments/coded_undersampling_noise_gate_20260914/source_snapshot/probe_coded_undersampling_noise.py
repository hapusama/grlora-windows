"""Paired noise gate for coded undersampling, with clean timing and CFO.

Noise is iid circular Gaussian on the B-rate complex sample grid; each reduced
rate sees a subset of the same noisy observation. RGR includes noise already in
the real capture. This is not a calibrated RF SNR or an acquisition experiment.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import probe_coded_undersampling as gate


@lru_cache(maxsize=None)
def candidate_choices(width):
    return ((np.arange(1 << width)[:, None] >> np.arange(width)) & 1)


def list_block(powers, sf_app, cr, downclock):
    """Top two residues per symbol, exact block constraints, then energy score.

    CRC and transmitted symbols do not participate in candidate selection.
    Both sample rates use the same list budget (2**block_length).
    """
    matrix, transform, pivots, kept = gate.observation_system(sf_app, cr, downclock)
    if len(pivots) != 4 * sf_app or len(powers) != cr + 4:
        return None
    spectra = np.asarray(powers)
    top = np.argpartition(spectra, -2, axis=1)[:, -2:]
    choices = candidate_choices(cr + 4)
    residues = top[np.arange(cr + 4)[None, :], choices]
    observed = ((residues[:, :, None] >> np.arange(kept)) & 1).reshape(len(choices), -1).astype(np.uint8)
    # uint8 overflow is harmless because the final arithmetic is modulo two.
    syndrome = (observed @ transform[len(pivots):].T) & 1
    valid = np.flatnonzero(~np.any(syndrome, axis=1))
    if len(valid) == 0:
        return None
    scores = np.sum(spectra[np.arange(cr + 4)[None, :], residues[valid]], axis=1)
    winner = valid[int(np.argmax(scores))]
    rhs = (transform[:len(pivots)] @ observed[winner]) & 1
    bits = np.zeros(4 * sf_app, dtype=np.uint8)
    bits[list(pivots)] = rhs
    if np.any(((matrix @ bits) & 1) != observed[winner]):
        raise AssertionError("candidate violates observable constraints")
    return gate.encode_block(bits, sf_app, cr)


def demodulate(samples, reference, downclock, sf, ldro):
    raw = np.abs(np.fft.fft(samples[:, ::downclock] * reference[None, ::downclock], axis=1)) ** 2
    result = []
    for index, spectrum in enumerate(raw):
        divisor = 4 if index < 8 or ldro else 1
        # Undo the raw-bin +1 convention, retaining nuisance low bits by max.
        result.append(np.max(np.roll(spectrum, -1).reshape(-1, divisor), axis=1))
    return result


def decode(powers, sf, ldro, downclock, method):
    if method == "hard":
        head = [int(np.argmax(s)) for s in powers[:8]]
    else:
        head = list_block(powers[:8], sf-2, 4, downclock)
    if head is None:
        return None
    header = gate.decode_explicit_header(head, sf, 125000, int(ldro))
    if not header.header_valid or not 1 <= header.cr <= 4:
        return None
    body = []
    for offset in range(8, len(powers), header.cr+4):
        block = powers[offset:offset+header.cr+4]
        if method == "hard":
            symbols = [int(np.argmax(s)) for s in block]
        else:
            symbols = list_block(block, sf-2 if header.ldro else sf, header.cr, downclock)
        if symbols is None:
            return None
        body.extend(symbols)
    return gate.decode_explicit_frame_symbols(head, body, sf, 125000, int(ldro), "grlora")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output directory")
    args.output.mkdir(parents=True)
    rows = []
    started = time.perf_counter()
    seeds = (913441, 913442, 913443)
    levels = (None, -7, -10, -13, -16)
    for dataset_index, dataset in enumerate(gate.common.DEFAULT_DATASETS):
        iq_path, csv_path = gate.common.dataset_paths(dataset)
        iq = np.memmap(iq_path, dtype=np.complex64, mode="r")
        for packet in gate.common.load_packets(csv_path):
            sf, rate = packet["sf"], packet["os_factor"]
            items = packet["header_symbols"] + packet["payload_symbols"]
            indices = np.arange(1 << sf) * rate + rate // 2
            clean = np.stack([np.asarray(iq[int(item["start_sample"]) + indices]) for item in items])
            reference = gate._oversampled_downchirp(sf, rate, packet["cfo_int"], packet["cfo_frac"])[::rate]
            target = gate.decode_explicit_frame_symbols(
                [s["symbol_value"] for s in packet["header_symbols"]],
                [s["symbol_value"] for s in packet["payload_symbols"]],
                sf, 125000, int(packet["ldro"]), "grlora",
            ).payload.payload_bytes
            power = float(np.mean(np.abs(clean.astype(np.complex128)) ** 2))
            for seed in seeds:
                rng = np.random.default_rng(np.random.SeedSequence([seed, dataset_index, packet["packet_index"]]))
                unit_noise = (rng.standard_normal(clean.shape) + 1j*rng.standard_normal(clean.shape)) / np.sqrt(2)
                for rgr in levels:
                    if rgr is None and seed != seeds[0]:
                        continue
                    noisy = clean if rgr is None else clean + unit_noise * np.sqrt(power * 10**(-rgr/10))
                    for downclock in (1, 2):
                        powers = demodulate(noisy, reference, downclock, sf, packet["ldro"])
                        for method in ("hard", "joint_top2"):
                            begin = time.perf_counter()
                            frame = decode(powers, sf, packet["ldro"], downclock, method)
                            elapsed = time.perf_counter() - begin
                            accepted = frame is not None and frame.header.header_valid and frame.header.has_crc and frame.payload.crc_valid
                            exact = accepted and frame.payload.payload_bytes == target
                            rows.append(dict(dataset=dataset, packet_index=packet["packet_index"], seed=seed,
                                             rgr_db="clean" if rgr is None else rgr, downclock=downclock,
                                             method=method, crc_accept=int(accepted), exact=int(exact),
                                             false_accept=int(accepted and not exact), decode_seconds=elapsed))
        print(f"completed {dataset}: {time.perf_counter()-started:.1f}s", flush=True)
    summary = []
    for rgr in ("clean", -7, -10, -13, -16):
        for downclock in (1, 2):
            for method in ("hard", "joint_top2"):
                group = [r for r in rows if (r["rgr_db"], r["downclock"], r["method"]) == (rgr, downclock, method)]
                summary.append(dict(rgr_db=rgr, downclock=downclock, method=method, trials=len(group),
                                    exact=sum(r["exact"] for r in group),
                                    false_accept=sum(r["false_accept"] for r in group)))
    gate.common.write_csv(args.output / "packet_trials.csv", rows)
    gate.common.write_csv(args.output / "summary.csv", summary)
    sources = [Path(__file__), Path(gate.__file__)]
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        (snapshot / source.name).write_bytes(source.read_bytes())
    (args.output / "config.json").write_text(json.dumps(dict(
        scope="clean timing/CFO, frame extents and LDRO supplied; actual header/CR and payload decoded",
        noise="iid circular Gaussian at B-rate complex grid; D=2 sees exact subset of D=1 observations",
        rgr="mean measured packet-active raw IQ power / added noise variance; includes original noise",
        seeds=seeds, rgr_db=levels, list_budget="top 2 residues per symbol; 256 header / 32 CR1 payload candidates",
        selection="block-valid maximum sum spectral power; CRC only final check; ground truth only scoring",
        limitations="no detection, no independent capture sessions, no RF front-end or ADC power measurement; finite candidate list is not ML",
        source_sha256={s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
    ), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

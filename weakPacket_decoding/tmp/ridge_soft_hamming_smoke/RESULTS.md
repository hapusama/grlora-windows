# Decoder-aware FrameSync: full FEC/CRC OTA AWGN evaluation

## Protocol

- Add B-wide complex AWGN once to the complete 1 MS/s OTA packet before FrameSync.
- Decode the noisy estimate with Savaux, explicit-header FEC, payload FEC, dewhitening, and PHY CRC.
- Strict and decoder-aware policies reuse the identical synchronization estimate and decoded candidate; only admission differs.
- The oracle branch reuses clean synchronization on the same noisy IQ.
- Expected bytes are consulted only after CRC to detect collisions and score exact PDR.

## Clean calibration

- Exact full-frame round trips: **1/1**.
- CRC mode: `grlora`; demod tail: 256 noise-only samples.

## Packet-level results

| Es/N0 (dB) | trials | strict PDR | decoder-aware PDR | ridge K=4 PDR | ridge + soft Hamming PDR | oracle PDR | extra calls | rescues | CRC rejects | CRC false deliveries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0 |

## Decoder arbitration audit

- Extra candidates admitted beyond strict gating: **0**.
- Extra candidates producing exact CRC-valid packets: **0**.
- Extra candidates rejected by PHY CRC: **0**.
- CRC-valid but wrong payload deliveries: **0**.

The CSV files retain per-packet header/payload symbol errors and modeled decoder runtime for complexity analysis.

## Remaining failure decomposition

| Es/N0 (dB) | oracle itself fails | oracle succeeds, decoder-aware gate rejects | gate accepts, noisy candidate fails | decoder-aware beats oracle |
|---:|---:|---:|---:|---:|

## Ambiguity-ridge list synchronization

- Ridge K=4: 0 exact packets; soft gate: 0; oracle: 0.
- Operational mean decoder attempts per input trial: 4.000.
- CRC-valid wrong deliveries: 0.

## Soft-Hamming decoder linkage

- Hard ridge: 0 exact packets; ridge + soft Hamming: 0.
- Soft-Hamming oracle sync: 0.
- CRC-valid wrong deliveries: 0.
| 13 | 1 | 0 | 0 | 0 |

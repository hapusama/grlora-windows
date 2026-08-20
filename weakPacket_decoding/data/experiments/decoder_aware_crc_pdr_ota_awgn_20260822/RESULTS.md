# Decoder-aware FrameSync: full FEC/CRC OTA AWGN evaluation

## Protocol

- Add B-wide complex AWGN once to the complete 1 MS/s OTA packet before FrameSync.
- Decode the noisy estimate with Savaux, explicit-header FEC, payload FEC, dewhitening, and PHY CRC.
- Strict and decoder-aware policies reuse the identical synchronization estimate and decoded candidate; only admission differs.
- The oracle branch reuses clean synchronization on the same noisy IQ.
- Expected bytes are consulted only after CRC to detect collisions and score exact PDR.

## Clean calibration

- Exact full-frame round trips: **8/8**.
- CRC mode: `grlora`; demod tail: 256 noise-only samples.

## Packet-level results

| Es/N0 (dB) | trials | strict PDR | decoder-aware PDR | oracle PDR | extra calls | rescues | CRC rejects | CRC false deliveries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 16 | 0.062 | 0.250 | 0.500 | 5 | 3 | 2 | 0 |
| 13 | 16 | 0.375 | 0.625 | 0.875 | 4 | 4 | 0 | 0 |
| 14 | 16 | 0.500 | 0.750 | 1.000 | 4 | 4 | 0 | 0 |
| 15 | 16 | 0.688 | 0.938 | 1.000 | 4 | 4 | 0 | 0 |
| 16 | 16 | 0.938 | 1.000 | 1.000 | 1 | 1 | 0 | 0 |

## Decoder arbitration audit

- Extra candidates admitted beyond strict gating: **18**.
- Extra candidates producing exact CRC-valid packets: **16**.
- Extra candidates rejected by PHY CRC: **2**.
- CRC-valid but wrong payload deliveries: **0**.

The CSV files retain per-packet header/payload symbol errors and modeled decoder runtime for complexity analysis.

## Remaining failure decomposition

| Es/N0 (dB) | oracle itself fails | oracle succeeds, decoder-aware gate rejects | gate accepts, noisy candidate fails | decoder-aware beats oracle |
|---:|---:|---:|---:|---:|
| 12 | 8 | 4 | 0 | 0 |
| 13 | 2 | 3 | 1 | 0 |
| 14 | 0 | 2 | 2 | 0 |
| 15 | 0 | 0 | 1 | 0 |
| 16 | 0 | 0 | 0 | 0 |

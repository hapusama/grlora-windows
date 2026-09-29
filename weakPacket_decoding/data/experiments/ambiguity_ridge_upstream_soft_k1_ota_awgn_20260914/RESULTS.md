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

| Es/N0 (dB) | trials | strict PDR | decoder-aware PDR | ridge K=1 PDR | ridge + soft Hamming PDR | hard oracle PDR | soft oracle PDR | extra calls | rescues | CRC rejects | gate-only CRC false deliveries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 40 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.025 | 4 | 0 | 4 | 0 |
| 11 | 40 | 0.000 | 0.000 | 0.000 | 0.025 | 0.000 | 0.350 | 4 | 0 | 4 | 0 |
| 12 | 40 | 0.025 | 0.150 | 0.200 | 0.425 | 0.425 | 0.800 | 13 | 5 | 8 | 0 |
| 13 | 40 | 0.350 | 0.525 | 0.550 | 0.575 | 0.850 | 0.975 | 9 | 7 | 2 | 0 |
| 14 | 40 | 0.625 | 0.775 | 0.775 | 0.800 | 0.950 | 1.000 | 7 | 6 | 1 | 0 |
| 15 | 40 | 0.750 | 0.925 | 0.925 | 0.925 | 1.000 | 1.000 | 7 | 7 | 0 | 0 |
| 16 | 40 | 0.925 | 0.975 | 0.975 | 0.975 | 1.000 | 1.000 | 2 | 2 | 0 | 0 |

## Decoder arbitration audit

- Extra candidates admitted beyond strict gating: **46**.
- Extra candidates producing exact CRC-valid packets: **27**.
- Extra candidates rejected by PHY CRC: **19**.
- CRC-valid but wrong payload deliveries: **0**.

The CSV files retain per-packet header/payload symbol errors and modeled decoder runtime for complexity analysis.

## Remaining failure decomposition

| Es/N0 (dB) | oracle itself fails | oracle succeeds, decoder-aware gate rejects | gate accepts, noisy candidate fails | decoder-aware beats oracle |
|---:|---:|---:|---:|---:|
| 10 | 40 | 0 | 0 | 0 |
| 11 | 40 | 0 | 0 | 0 |
| 12 | 23 | 9 | 2 | 0 |
| 13 | 6 | 7 | 7 | 1 |
| 14 | 2 | 3 | 4 | 0 |
| 15 | 0 | 0 | 3 | 0 |
| 16 | 0 | 0 | 1 | 0 |

## Ambiguity-ridge list synchronization

- Ridge K=1: 137 exact packets; soft gate: 134; oracle: 169.
- Operational mean decoder attempts per input trial: 0.889.
- CRC-valid wrong deliveries: 0.

## Soft-Hamming decoder linkage

- Hard ridge: 137 exact packets; ridge + soft Hamming: 149.
- Soft-Hamming oracle sync: 206.
- Operational mean soft decoder attempts per input trial: 0.889.
- CRC-valid wrong deliveries: 1.

## Interpolated sensitivity thresholds

Linear interpolation is applied only between adjacent measured SNR points.

| receiver | Es/N0 at PDR50 | Es/N0 at PDR80 |
|---|---:|---:|
| strict gate | 13.545 dB | 15.286 dB |
| decoder-aware gate | 12.933 dB | 14.167 dB |
| ridge K=1 | 12.857 dB | 14.167 dB |
| ridge K=1 + soft Hamming | 12.500 dB | 14.000 dB |

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

| Es/N0 (dB) | trials | strict PDR | decoder-aware PDR | ridge K=4 PDR | ridge + soft Hamming PDR | hard oracle PDR | soft oracle PDR | extra calls | rescues | CRC rejects | gate-only CRC false deliveries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 80 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.013 | 4 | 0 | 4 | 0 |
| 11 | 80 | 0.000 | 0.000 | 0.000 | 0.188 | 0.000 | 0.362 | 11 | 0 | 11 | 0 |
| 12 | 80 | 0.025 | 0.138 | 0.263 | 0.650 | 0.350 | 0.787 | 25 | 9 | 16 | 0 |
| 13 | 80 | 0.312 | 0.500 | 0.738 | 0.887 | 0.787 | 0.975 | 19 | 15 | 4 | 0 |
| 14 | 80 | 0.575 | 0.750 | 0.887 | 0.925 | 0.950 | 1.000 | 15 | 14 | 1 | 0 |
| 15 | 80 | 0.775 | 0.963 | 0.988 | 0.988 | 1.000 | 1.000 | 15 | 15 | 0 | 0 |
| 16 | 80 | 0.912 | 0.988 | 0.988 | 0.988 | 1.000 | 1.000 | 6 | 6 | 0 | 0 |

## Decoder arbitration audit

- Extra candidates admitted beyond strict gating: **95**.
- Extra candidates producing exact CRC-valid packets: **59**.
- Extra candidates rejected by PHY CRC: **36**.
- CRC-valid but wrong payload deliveries: **0**.

The CSV files retain per-packet header/payload symbol errors and modeled decoder runtime for complexity analysis.

## Remaining failure decomposition

| Es/N0 (dB) | oracle itself fails | oracle succeeds, decoder-aware gate rejects | gate accepts, noisy candidate fails | decoder-aware beats oracle |
|---:|---:|---:|---:|---:|
| 10 | 80 | 0 | 0 | 0 |
| 11 | 80 | 0 | 0 | 0 |
| 12 | 52 | 14 | 3 | 0 |
| 13 | 17 | 16 | 8 | 1 |
| 14 | 4 | 9 | 7 | 0 |
| 15 | 0 | 0 | 3 | 0 |
| 16 | 0 | 0 | 1 | 0 |

## Ambiguity-ridge list synchronization

- Ridge K=4: 309 exact packets; soft gate: 267; oracle: 327.
- Operational mean decoder attempts per input trial: 1.964.
- CRC-valid wrong deliveries: 1.

## Soft-Hamming decoder linkage

- Hard ridge: 309 exact packets; ridge + soft Hamming: 370.
- Soft-Hamming oracle sync: 411.
- Operational mean soft decoder attempts per input trial: 1.698.
- CRC-valid wrong deliveries: 1.

## Interpolated sensitivity thresholds

Linear interpolation is applied only between adjacent measured SNR points.

| receiver | Es/N0 at PDR50 | Es/N0 at PDR80 |
|---|---:|---:|
| strict gate | 13.714 dB | 15.182 dB |
| decoder-aware gate | 13.000 dB | 14.235 dB |
| ridge K=4 | 12.500 dB | 13.417 dB |
| ridge K=4 + soft Hamming | 11.676 dB | 12.632 dB |

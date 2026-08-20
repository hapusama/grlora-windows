# Multi-rate single-symbol AWGN experiment

All lower-rate views are nested phase-zero decimations of the same noisy input. 
No anti-alias filter, resynchronization, learned noise model, or independent noise draw is used.

- SF: 7
- Bandwidth: 125000 Hz
- Source rate: q=8 (1e+06 sample/s)
- Views: q=8, q=4, q=2, q=1
- Trials per SNR: 40
- SNR convention: per-source-sample signal power divided by complex AWGN power.
- Synchronization/CFO/SFO: perfect; unknown common symbol phase is randomized.

## Symbol error rate

| SNR (dB) | fft_argmax_q8 | fft_argmax_q4 | fft_argmax_q2 | fft_argmax_q1 | pair_energy_q8 | fold_profile_q8 | multirate_structure | coherent_ml_q8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -24 | 0.9250 | 1.0000 | 1.0000 | 1.0000 | 0.9250 | 0.8750 | 0.9250 | 0.7000 |
| -20 | 0.4250 | 0.8250 | 0.9750 | 0.9500 | 0.4250 | 0.2000 | 0.4250 | 0.0000 |

## Method boundary

- `fft_argmax_q8` maps the stronger legal full-rate peak back modulo N; it does not combine the two segments.
- `pair_energy_q8` uses the N-bin spacing but not the expected wrap ratio.
- `fold_profile_q8` additionally projects onto the expected `(N-m):m` fold profile.
- `multirate_structure` sums the fold-profile scores from all requested rates. Its inputs are correlated, so the sum is not described as MRC or an SNR gain.
- `coherent_ml_q8` correlates against the complete source-rate LoRa waveform and is the AWGN upper control.

The invariant audit is in `invariants.csv`, and paired trial decisions are in `trials.csv`.

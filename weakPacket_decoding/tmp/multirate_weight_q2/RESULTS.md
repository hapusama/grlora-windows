# Multi-rate single-symbol AWGN experiment

All lower-rate views are nested phase-zero decimations of the same noisy input. 
No anti-alias filter, resynchronization, learned noise model, or independent noise draw is used.

- SF: 7
- Bandwidth: 125000 Hz
- Source rate: q=8 (1e+06 sample/s)
- Views: q=8, q=4, q=2, q=1
- Trials per SNR: 500
- SNR convention: per-source-sample signal power divided by complex AWGN power.
- Synchronization/CFO/SFO: perfect; unknown common symbol phase is randomized.

## Symbol error rate

| SNR (dB) | fft_argmax_q8 | fft_argmax_q4 | fft_argmax_q2 | fft_argmax_q1 | pair_energy_q8 | fold_profile_q8 | multirate_structure | coherent_ml_q8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -24 | 0.8360 | 0.9320 | 0.9700 | 0.9560 | 0.8400 | 0.7540 | 0.7440 | 0.6160 |
| -22 | 0.7180 | 0.8960 | 0.9640 | 0.9640 | 0.6680 | 0.5800 | 0.5940 | 0.3240 |
| -20 | 0.4260 | 0.7520 | 0.9280 | 0.9140 | 0.4020 | 0.3180 | 0.3200 | 0.1140 |
| -18 | 0.2100 | 0.5940 | 0.8420 | 0.8520 | 0.1800 | 0.0880 | 0.0960 | 0.0080 |

## Method boundary

- `fft_argmax_q8` maps the stronger legal full-rate peak back modulo N; it does not combine the two segments.
- `pair_energy_q8` uses the N-bin spacing but not the expected wrap ratio.
- `fold_profile_q8` additionally projects onto the expected `(N-m):m` fold profile.
- `multirate_structure` sums the fold-profile scores from all requested rates. Its inputs are correlated, so the sum is not described as MRC or an SNR gain.
- `coherent_ml_q8` correlates against the complete source-rate LoRa waveform and is the AWGN upper control.

The invariant audit is in `invariants.csv`, and paired trial decisions are in `trials.csv`.

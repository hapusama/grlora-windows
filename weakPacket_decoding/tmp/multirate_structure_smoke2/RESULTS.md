# Multi-rate single-symbol AWGN experiment

All lower-rate views are nested phase-zero decimations of the same noisy input. 
No anti-alias filter, resynchronization, learned noise model, or independent noise draw is used.

- SF: 7
- Bandwidth: 125000 Hz
- Source rate: q=8 (1e+06 sample/s)
- Views: q=8, q=4, q=2, q=1
- Trials per SNR: 50
- SNR convention: per-source-sample signal power divided by complex AWGN power.
- Synchronization/CFO/SFO: perfect; unknown common symbol phase is randomized.

## Symbol error rate

| SNR (dB) | fft_argmax_q8 | fft_argmax_q4 | fft_argmax_q2 | fft_argmax_q1 | pair_energy_q8 | fold_profile_q8 | multirate_equal_sum | multirate_awgn_glrt | coherent_ml_q8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -22 | 0.8000 | 0.9200 | 0.9800 | 0.9800 | 0.8400 | 0.6200 | 0.7600 | 0.6200 | 0.3400 |

## Method boundary

- `fft_argmax_q8` maps the stronger legal full-rate peak back modulo N; it does not combine the two segments.
- `pair_energy_q8` uses the N-bin spacing but not the expected wrap ratio.
- `fold_profile_q8` additionally projects onto the expected `(N-m):m` fold profile.
- `multirate_equal_sum` directly sums the fold-profile scores from all requested rates. It is the simple proposed heuristic, not an independent-view likelihood.
- `multirate_awgn_glrt` whitens the nested-view correlation. Under source-rate white AWGN it algebraically collapses to the source-rate fold profile because the conditional lower-rate residual contains no candidate signal.
- `coherent_ml_q8` correlates against the complete source-rate LoRa waveform and is the AWGN upper control.

The invariant audit is in `invariants.csv`, and paired trial decisions are in `trials.csv`.

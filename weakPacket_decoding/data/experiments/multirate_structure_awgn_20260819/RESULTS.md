# Multi-rate single-symbol AWGN experiment

All lower-rate views are nested phase-zero decimations of the same noisy input. 
No anti-alias filter, resynchronization, learned noise model, or independent noise draw is used.

- SF: 10
- Bandwidth: 125000 Hz
- Source rate: q=8 (1e+06 sample/s)
- Views: q=8, q=4, q=2, q=1
- Trials per SNR: 2000
- SNR convention: per-source-sample signal power divided by complex AWGN power.
- Synchronization/CFO/SFO: perfect; unknown common symbol phase is randomized.

## Symbol error rate

| SNR (dB) | fft_argmax_q8 | fft_argmax_q4 | fft_argmax_q2 | fft_argmax_q1 | pair_energy_q8 | fold_profile_q8 | multirate_equal_sum | multirate_awgn_glrt | coherent_ml_q8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -34 | 0.9590 | 0.9880 | 0.9955 | 0.9960 | 0.9575 | 0.9285 | 0.9710 | 0.9285 | 0.8650 |
| -32 | 0.8960 | 0.9740 | 0.9870 | 0.9945 | 0.8950 | 0.8460 | 0.9350 | 0.8460 | 0.6875 |
| -30 | 0.7340 | 0.9205 | 0.9820 | 0.9795 | 0.7370 | 0.6335 | 0.8250 | 0.6335 | 0.3690 |
| -28 | 0.5080 | 0.8450 | 0.9655 | 0.9630 | 0.4865 | 0.3525 | 0.6245 | 0.3525 | 0.0895 |
| -26 | 0.2575 | 0.6545 | 0.8985 | 0.9130 | 0.1945 | 0.1140 | 0.3215 | 0.1140 | 0.0055 |
| -24 | 0.0775 | 0.3795 | 0.7560 | 0.7945 | 0.0525 | 0.0105 | 0.0925 | 0.0105 | 0.0000 |
| -22 | 0.0250 | 0.1405 | 0.5165 | 0.5570 | 0.0140 | 0.0005 | 0.0060 | 0.0005 | 0.0000 |
| -21 | 0.0140 | 0.0895 | 0.3985 | 0.4040 | 0.0075 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| -20 | 0.0080 | 0.0420 | 0.2455 | 0.2300 | 0.0055 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Method boundary

- `fft_argmax_q8` maps the stronger legal full-rate peak back modulo N; it does not combine the two segments.
- `pair_energy_q8` uses the N-bin spacing but not the expected wrap ratio.
- `fold_profile_q8` additionally projects onto the expected `(N-m):m` fold profile.
- `multirate_equal_sum` directly sums the fold-profile scores from all requested rates. It is the simple proposed heuristic, not an independent-view likelihood.
- `multirate_awgn_glrt` whitens the nested-view correlation. Under source-rate white AWGN it algebraically collapses to the source-rate fold profile because the conditional lower-rate residual contains no candidate signal.
- `coherent_ml_q8` correlates against the complete source-rate LoRa waveform and is the AWGN upper control.

The invariant audit is in `invariants.csv`, and paired trial decisions are in `trials.csv`.

# Threshold Gain Evaluation

Date: 2026-06-16

Scope:

```text
gr-lora_sdr/weakPacket_decoding copy
```

## Goal

Evaluate the current symbol-level two-stage weak decoder against the standard
single-center-offset FFT argmax LoRa PHY baseline using SNR thresholds:

```text
SER <= 10%
symbol accuracy >= 90%
CRC-valid / PRR >= 90%, 80%, 50%
```

Target:

```text
SER-threshold gain >= 5 dB
CRC/PRR-threshold gain >= 3 dB
prefer CRC-valid > 0.9 where possible
```

## Artifacts

Linear default selector:

```text
data/symbol_phase_threshold_sweep/per_packet_metrics.csv
data/symbol_phase_threshold_sweep/snr_curve_summary.csv
data/symbol_phase_threshold_sweep/threshold_table.csv
data/symbol_phase_threshold_sweep/gain_table.csv
data/symbol_phase_threshold_sweep/summary.json
```

Quadratic phase selector trial:

```text
data/symbol_phase_threshold_sweep_quadratic/
```

Aggressive phase-gate trial:

```text
data/symbol_phase_threshold_sweep_balanced_linear/
```

GT-only phase model diagnostic:

```text
data/symbol_phase_model_diagnostics/phase_model_detail.csv
data/symbol_phase_model_diagnostics/phase_model_summary.csv
```

## Current Mean Thresholds

Mean across:

```text
0_0_0_10_14_8
0_0_0_10_14_16
0_0_0_10_14_32
```

| Method | SER<=10% | CRC/PRR>=90% | CRC/PRR>=80% | CRC/PRR>=50% |
| --- | ---: | ---: | ---: | ---: |
| center FFT argmax | -17.23 dB | -15.17 dB | -15.94 dB | -17.11 dB |
| multi-offset argmax | -20.61 dB | -17.82 dB | -19.27 dB | -20.60 dB |
| two-stage selected | -20.71 dB | -18.30 dB | -19.27 dB | -20.74 dB |

Current gain of two-stage selected vs center FFT argmax:

| Metric | Gain |
| --- | ---: |
| SER<=10% | 3.48 dB |
| accuracy>=90% | 3.48 dB |
| CRC/PRR>=90% | 3.13 dB |
| CRC/PRR>=80% | 3.33 dB |
| CRC/PRR>=50% | 3.63 dB |

Status:

```text
CRC/PRR gain >= 3 dB: currently achieved on the mean curve.
SER gain >= 5 dB: not achieved; current gap is about 1.5 dB.
```

## Phase Model Trials

### Quadratic selector

Adding a quadratic payload phase model made almost no difference:

```text
SER<=10% gain: 3.48 dB -> 3.48 dB
CRC/PRR>=90% gain: 3.13 dB -> 3.13 dB
```

It slightly changes some individual SNR points but does not solve the threshold
gap.

### Aggressive phase-gate trial

Tried a less conservative gate:

```text
phase_weight = 0.45
amp_weight = 0.55
phase_override_min_gain = 0.05
phase_override_max_drop_db = 1.5
phase_override_score_margin = 0.0
phase_override_min_line_anchors = 4
phase_override_max_line_rmse_pi = 0.45
```

This degraded SER and PRR.  The failure mode is consistent with false phase
overrides: wrong candidates can form a smooth-looking trajectory, so merely
relaxing gates is not enough.

## GT-Only Phase Diagnostic

Diagnostic question:

```text
When multi-offset Top-1 is wrong but GT is inside Top-8,
does phase ranking put GT near the top?
```

Across `-18` to `-22 dB`, there were 226 ambiguous symbols.

Summary:

| Model | GT rank-1 rate | GT rank<=2 rate | Mean GT phase rank |
| --- | ---: | ---: | ---: |
| linear locked-line | 0.157 | 0.396 | 2.88 |
| quadratic locked-line | 0.185 | 0.485 | 2.65 |
| local linear | 0.201 | 0.249 | 3.56 |
| oracle GT linear leave-one-out | 0.181 | 0.437 | 3.64 |
| oracle GT quadratic leave-one-out | 0.226 | 0.453 | 3.10 |

Interpretation:

```text
Phase contains useful information, but the current predictor is not strong
enough to reliably make GT the Top-1 candidate inside Top-8.
Quadratic is directionally better than linear in the diagnostic, but the gain
is too small to move the end-to-end threshold with the current conservative
decision rule.
```

## Current Conclusion

The current system is materially stronger than standard center FFT argmax:

```text
multi-offset evidence is the dominant gain source.
phase consistency provides small extra gains and improves CRC/PRR threshold,
but it is not yet strong enough to reach SER gain >= 5 dB.
```

The current objective is therefore not complete.

## 2026-06-16 Update - Offset-Phase Coherence Selector

The useful phase signal turned out to be stronger inside the multi-offset FFT
evidence than in a global payload phase-line gate.

New default selector:

```text
selection_mode = coherence
top_l_low_confidence = 24
smooth_phase_weight = 0.05
smooth_amp_weight = 0.50
smooth_coherence_weight = 0.90
smooth_max_energy_drop_db = 20
```

Per symbol, the decoder still locks high-confidence Top-1 bins first.  For
uncertain symbols it keeps energy Top-24 candidates and scores each candidate
with:

```text
score =
  0.50 * normalized multi-offset energy
+ 0.90 * offset-phase coherence
+ 0.05 * packet-line phase score
```

The new phase feature is:

```text
offset_coherence_k[b] =
  |sum_o Z_o[k,b]| / (sum_o |Z_o[k,b]| + eps)
```

This is a bounded phase-consistency score across oversampling offsets.  It is
computed per symbol and per bin, with no payload byte prior, no CRC feedback,
and no cross-packet information.

Formal sweep output:

```text
data/symbol_phase_threshold_sweep_coherence_default/
```

Mean thresholds across the three validation datasets:

| Method | SER<=10% | CRC/PRR>=90% | CRC/PRR>=80% | CRC/PRR>=50% |
| --- | ---: | ---: | ---: | ---: |
| center FFT argmax | -17.23 dB | -15.17 dB | -15.94 dB | -17.11 dB |
| multi-offset argmax | -20.61 dB | -17.82 dB | -19.27 dB | -20.60 dB |
| offset-coherence selected | -21.97 dB | -19.31 dB | -20.31 dB | -21.62 dB |

Mean gain of offset-coherence selected vs center FFT argmax:

| Metric | Gain |
| --- | ---: |
| SER<=10% | 4.74 dB |
| accuracy>=90% | 4.74 dB |
| CRC/PRR>=90% | 4.14 dB |
| CRC/PRR>=80% | 4.37 dB |
| CRC/PRR>=50% | 4.51 dB |

Mean `-20` to `-25 dB` curve:

| SNR | Center SER | Multi SER | Selected SER | Selected accuracy | Selected CRC/PRR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -20 dB | 0.373 | 0.068 | 0.044 | 0.956 | 0.858 |
| -21 dB | 0.519 | 0.120 | 0.060 | 0.940 | 0.668 |
| -22 dB | 0.646 | 0.204 | 0.101 | 0.899 | 0.397 |
| -23 dB | 0.767 | 0.332 | 0.184 | 0.816 | 0.114 |
| -24 dB | 0.849 | 0.506 | 0.314 | 0.686 | 0.000 |
| -25 dB | 0.910 | 0.649 | 0.446 | 0.554 | 0.000 |

Status after this update:

```text
CRC/PRR threshold gain target is achieved with margin.
SER threshold gain is close to the original 5 dB target: 4.74 dB mean gain.
The remaining gap is about 0.26 dB, and additional smooth-trajectory beam
attempts increased complexity while degrading SER at -22 dB.
```

Recommended stopping point:

```text
Keep the low-complexity offset-coherence selector as the current default.
Do not add the smooth beam path to the main claim unless a future diagnostic
shows a clear gain; current probes made the -22 dB SER worse.
```

## Next Optimization Direction

The next work should avoid simply relaxing phase gates.  More promising:

```text
1. Build a smooth trajectory decoder over Top-8 candidates, not independent
   per-symbol overrides.
2. Penalize second difference of unwrapped selected phases instead of forcing
   every point onto one global line.
3. Use dynamic programming / beam over Top-8 with:
     local multi-offset amplitude
     phase residual to a smooth latent trajectory
     pairwise phase increment smoothness
     curvature penalty
4. Add a no-GT phase-model selector:
     use locked symbols to estimate allowed slope/curvature range
     choose Top-8 path minimizing robust trajectory energy
5. Keep CRC out of candidate choice; use it only for PRR reporting.
```

Expected benefit:

```text
The existing per-symbol phase gate only uses phase as a local bonus.
A smooth path model can exploit the observed packet-level phase structure more
directly and may recover the missing ~1.5 dB SER-threshold gap.
```

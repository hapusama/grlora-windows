# Branch4 real low-SNR decoding and GLS check

Date: 2026-07-17 (Asia/Shanghai)

## Evaluation boundary

- Each low-SNR capture was detected and synchronized from its own IQ.  No
  high-SNR frame-sync intermediate (CFO, STO, SFO, or sample start) was reused.
- The high-SNR consensus table was used only as the 49-position payload
  FFT-bin ground truth after synchronization.
- The primary SER set contains only `grlora_framesync_valid=1` frames with a
  complete 49-symbol payload.
- GLS pattern selection and fixed covariance estimation used off-packet
  windows from a different capture from the test capture.

## Detection and ordinary FFT results

| capture | duration (s) | scheduled packets | detections | frame valid | strict frame-sync | complete payload frames | payload FFT-bin errors | SER | FER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low1 | 44.753 | 15 | 15 | 11 | 11 | 11 | 0 / 539 | 0 | 0 / 11 |
| low2 | 55.337 | 18 | 18 | 15 | 15 | 15 | 98 / 735 | 13.3333% | 2 / 15 |
| low3 | 59.653 | 20 (inferred) | 19 | 18 | 13 | 12 | 0 / 588 | 0 | 0 / 12 |

The low3 scheduled count is inferred from the 3-second event grid: the first
and last detections span 20 scheduled positions and one two-period gap exposes
one missed event.  The last strict-sync packet is too close to the end of the
file to contain a complete payload, so 13 strict frames yield 12 SER frames.

`low1` and `low2` used `win_chirps=4`.  For `low3`, `win_chirps=4` found 16
events and 10 strict frames.  `win_chirps=8` found 19 regularly spaced events,
but the longer window advanced the coarse event start by about 5.4 chirps.
The wrapper's default +/-1-chirp alignment radius then produced 0 valid
frames.  Running the existing lower-level chain with
`align_search_chirps=8, align_step_samples=8` recovered 18 frame-locator-valid
and 13 strict-sync frames.  SFD localization remained at one-sample steps.

The 98 low2 errors are not random symbol errors.  Two frames have every one of
their 49 payload bins exactly one bin below GT.  Their low-capture sync/NetID
estimate has a systematic one-bin offset.  No GT-driven CFO correction was
applied.

## Savaux versus GLS on strict-sync frames

The primary aggregate contains 38 complete frames and 1862 payload symbols.

| method | errors / symbols | SER | frame errors / frames | FER | fixes vs Savaux | breaks vs Savaux |
|---|---:|---:|---:|---:|---:|---:|
| ordinary center FFT | 98 / 1862 | 5.2632% | 2 / 38 | 5.2632% | 0 | 0 |
| Savaux OSR=4 | 98 / 1862 | 5.2632% | 2 / 38 | 5.2632% | - | - |
| matrix-free cross-fit GLS, 8 patterns | 98 / 1862 | 5.2632% | 2 / 38 | 5.2632% | 0 | 0 |
| fixed off-packet-covariance GLS, 8 patterns | 98 / 1862 | 5.2632% | 2 / 38 | 5.2632% | 0 | 0 |

Therefore GLS does not exceed Savaux on the trustworthy strict-sync set.
Most frames already have zero Savaux errors, while the two low2 failures are a
frame-sync integer-bin error shared by all symbol demodulators.

## Diagnostic frame-valid set

To expose difficult symbol decisions, a secondary diagnostic includes 17
complete low3 frames that pass the frame locator, including five that fail the
strict gr-lora frame-sync gate.  This set is not counted as successful decoding.

| method | errors / symbols | SER | frame errors / frames | FER |
|---|---:|---:|---:|---:|
| ordinary center FFT | 209 / 833 | 25.0900% | 5 / 17 | 29.4118% |
| Savaux OSR=4 | 141 / 833 | 16.9268% | 5 / 17 | 29.4118% |
| matrix-free cross-fit GLS | 139 / 833 | 16.6867% | 5 / 17 | 29.4118% |
| fixed off-packet-covariance GLS | 141 / 833 | 16.9268% | 5 / 17 | 29.4118% |

Cross-fit GLS makes three Savaux-error-to-correct changes and one
Savaux-correct-to-error change, for a net two-symbol reduction (0.2401
percentage point absolute).  The result repeats when pattern selection is
trained on low1 or low2, but it does not improve FER.  With only four paired
correctness changes, the two-sided exact McNemar/binomial p-value is 0.625.
This is not sufficient evidence to claim a reliable GLS improvement.

## Off-packet noise color

Each result below uses 256 off-packet windows of 4096 complex samples.  All
detected events, including sync failures, are excluded from the training
windows with a conservative packet guard.

| capture | lag-1 | max abs autocorrelation, lags 1..64 | significant lags / 64 | PSD flatness | PSD P95/P05 | selected-pattern covariance mismatch vs white |
|---|---:|---:|---:|---:|---:|---:|
| low1 | 0.1299 | 0.1299 | 64 / 64 | 0.9414 | 5.08 dB | 0.1508 |
| low2 | 0.1336 | 0.1336 | 64 / 64 | 0.9419 | 4.81 dB | 0.1604 |
| low3 | 0.1398 | 0.1398 | 64 / 64 | 0.9382 | 5.00 dB | 0.2004 |

The white-noise 3-sigma autocorrelation reference for this sample count is
0.00293.  A same-size complex AWGN control produced max-lag correlation
0.00181, PSD flatness 0.9899, and P95/P05 0.90 dB.  The real off-packet IQ is
therefore decisively non-white/colored at the sample and pattern-output levels.

This does not yet prove that the color is entirely stationary environmental
noise.  Without an independent transmitter-off `noise_only` capture, the
observed shape can combine receiver/frontend filtering, environmental
interference, and any missed on-air packet not covered by the event mask.

## Artifacts

- `low1_win4/`, `low2_win4/`: detection, sync, and ordinary FFT outputs.
- `low3_win4/`: original low3 `win_chirps=4` result.
- `low3_win8/`: demonstrates the insufficient default alignment radius.
- `low3_win8_align8_step8/`: selected low3 result.
- `gls_low{N}_train_low{M}/`: strict cross-capture GLS summaries, per-symbol
  decisions, pattern selections, and noise summaries.
- `gls_low3_framevalid_train_low{1,2}/`: diagnostic weak-frame comparison.
- `weak_decoder/os_lora/experiments/evaluate_real_capture_gls.py`: reusable real-capture
  evaluator.  It does not add synthetic noise.

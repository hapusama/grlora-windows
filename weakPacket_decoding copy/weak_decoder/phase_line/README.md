# Phase-Line FFT-Bin Selection

This directory contains the current weak-packet raw FFT-bin selector.  The
active design is no longer a phase-only replacement for the baseline.  It is a
guarded selector:

```text
Savaux synchronized oversampled hard bin
+ small, runtime-guarded phase-line corrections
```

The scope is intentionally narrow.  This module selects raw payload FFT bins.
It does not use CRC-guided search, LoRa codec enumeration, payload templates,
cross-packet priors, or ground-truth bins during selection.

## Current Status

Final validated configuration:

```text
Stage 1: Savaux-style synchronized oversampled branch DFT evidence
Stage 2: conservative first-order phase Viterbi over Stage-1 candidates
Guard: accept phase correction only when it is small and self-consistent
Fallback: Savaux hard argmax
```

The final report is:

```text
SAVAUX_STAGE1_SELECTOR_REPORT_2026-06-20.md
```

Across the final 75 dataset/SNR/seed groups and 700 paired packet comparisons:

| Selector | Mean SER |
|---|---:|
| Legacy phase-line | 0.3235 |
| Savaux hard | 0.1980 |
| Unconditional Savaux + phase | 0.1878 |
| Guarded Savaux + phase | 0.1951 |

The guarded selector had:

```text
negative groups vs Savaux hard = 0
packet wins / ties / losses    = 63 / 637 / 0
```

So the current default prioritizes stable improvement over maximum average
gain.

## File Map

```text
configs.py
  PhaseLineSelectorConfig and PhasePathSelectorConfig.  These keep the older
  phase-smooth and Viterbi selector settings.

trajectory.py
  Circular phase unwrap, local prediction, selected-bin line fitting, and phase
  likelihood helpers.

selector.py
  Low-level phase selectors.  The important active primitive is
  select_phase_viterbi_path(...).  Older smooth/local-anchor machinery remains
  here for diagnostics and ablations.

savaux_stage1.py
  Current active Stage-1 and guarded selector support:
    SavauxStage1Config
    SavauxPhaseGuardConfig
    build_savaux_stage1_packet_evidence(...)
    default_savaux_phase_path_config(...)
    evaluate_savaux_phase_guard(...)
    select_savaux_phase_viterbi_path(...)

_eval/diagnose_savaux_stage1_selector.py
  Reproducible diagnostic runner.  It reports center argmax, multi-offset
  argmax, legacy phase-line, Savaux hard, unconditional phase, and guarded phase
  SER side by side.

docs/history/
  Archived reports from earlier phase-line and DP attempts.
```

## Guard Rule

The default guard is defined by `SavauxPhaseGuardConfig`:

```text
1 <= changed_symbols <= 2
phase-path eval RMSE improves over Savaux hard
mean_phase_score >= 0.990
changed-bin mean energy drop >= -2.0 dB
```

If any check fails, the selector keeps the Savaux hard bins.  This matters
because unconditional phase Viterbi has better mean SER but causes regressions
on some packet lengths and noise seeds.

## Reproduce Final Experiments

Tuning/stability matrix:

```powershell
python weak_decoder/phase_line/_eval/diagnose_savaux_stage1_selector.py `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snrs -22 -23 -24 -25 -26 `
  --seeds 42 43 44 `
  --output-dir weak_decoder/phase_line/_eval/savaux_stage1_guard099_stability_3datasets_5snr_3seed
```

Holdout validation:

```powershell
python weak_decoder/phase_line/_eval/diagnose_savaux_stage1_selector.py `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snrs -22 -23 -24 -25 -26 `
  --seeds 45 46 `
  --output-dir weak_decoder/phase_line/_eval/savaux_stage1_guard099_holdout_3datasets_5snr_2seed
```

The two final output directories are:

```text
_eval/savaux_stage1_guard099_stability_3datasets_5snr_3seed/
_eval/savaux_stage1_guard099_holdout_3datasets_5snr_2seed/
```

## System Summary

The earlier phase-line work found that low-SNR failures were mostly caused by
Stage-1 candidate recall: if the correct raw FFT bin was absent from Top-L,
no phase DP could recover it.  The current system fixes that first by reusing
the Savaux oversampled spectrum primitive, building a stronger synchronized
payload observation before any phase path selection.

After Stage 1, the phase path is deliberately conservative.  It may correct a
small number of ambiguous symbols, but it is not allowed to rewrite the packet
when its own runtime metrics are marginal.  This gives up some oracle-like mean
SER from the unguarded phase path, but it avoids the instability seen across
packet lengths 8/16/32 and new noise seeds.

In short:

```text
The active system is Savaux-first, phase-guarded, FFT-bin-only.
```

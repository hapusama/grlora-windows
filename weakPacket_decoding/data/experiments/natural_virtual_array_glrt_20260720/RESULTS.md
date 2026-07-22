# Natural Virtual-Array GLRT Results

This output is generated from frozen real IQ and frozen source-demod CSVs. No
synchronization or GT regeneration is performed.

## Final observations

- `0_0_0_10_14_8` has only 12 non-zero off-packet windows after sparse-file
  zero regions and packet guards are removed. That is insufficient for a
  16-dimensional four-segment covariance; `segmented_gls` is therefore not a
  valid result for this capture.
- With the final `noise_windows=128` run, the audited clean symbols have
  `Savaux SER=0` on all three captures. `colored_ml SER=0` on all three and
  therefore ties the strongest available baseline.
- Fixed four-branch GLS and four-segment GLS also tie Savaux on capture
  `0_0_0_10_14_16`. On `0_0_0_10_14_32`, they produce respectively 5 and 6 breaks
  (`SER=2.38%` and `2.86%`), with no fixes. This is a covariance/steering
  mismatch, not evidence of a natural colored-noise gain.
- The full model-derived pattern run on `0_0_0_10_14_32` uses 12 patterns and
  all 1024 true/candidate bins. Its whitened leakage diagnostic is median
  `0.3253`, maximum `0.8956`, but `model_gls` still has 5 breaks. Lower
  cross-hypothesis leakage alone is therefore not sufficient.
- The natural-data result is a regression and upper-bound check, not evidence
  of a SER improvement. A stable gain requires an independent natural test
  capture with enough Savaux errors and enough off-packet covariance samples.

Per-symbol decisions and the exact paired counts are stored in `symbols.csv`;
per-capture counts are stored in `summary.csv`. The model-derived full-bin
diagnostic is recorded in `summary_model_0_0_0_10_14_32.csv` and
`pattern_design_0_0_0_10_14_32.json`.

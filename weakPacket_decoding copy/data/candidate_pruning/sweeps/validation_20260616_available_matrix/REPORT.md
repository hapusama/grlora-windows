# Candidate Pruning Recall Validation - 2026-06-16

Working root:

```text
d:\Desktop\proj\gr-lora_sdr\weakPacket_decoding copy
```

## Scope

This sweep validates the first-stage phase-aware candidate-pruning metric only.
It measures whether the clean GT payload bin is retained in Top-L candidates.
It does not run dewhitening, CRC, beam search, or the two-stage weak decoder.

All generated files were written under `weakPacket_decoding copy`.  The original
USRP IQ files in `gr-lora_sdr\data\USRP_IQ` were read only to synthesize the
missing -25/-27 dB noisy IQ cases for preamble 8 and 32.

## Commands

Syntax checks:

```powershell
python -m py_compile `
  weak_decoder/candidate_pruning.py `
  scripts/experiments/evaluate_candidate_pruning_metric.py `
  scripts/experiments/run_candidate_pruning_sweep.py `
  scripts/experiments/run_low_snr_gt_bin_experiment.py
```

Extra noisy-IQ generation for missing preamble 8/32 SNR points:

```powershell
python scripts/experiments/run_low_snr_gt_bin_experiment.py `
  -i "..\data\USRP_IQ\0_0_0_10_14_8.bin" `
  -g "data\weak_sync_chain\header_first\0_0_0_10_14_8_header_first_symbols.csv" `
  -o "data\low_snr_gt_bin\0_0_0_10_14_8_extreme_snr" `
  --target-snr-db -25 -27 --overwrite --no-plots

python scripts/experiments/run_low_snr_gt_bin_experiment.py `
  -i "..\data\USRP_IQ\0_0_0_10_14_32.bin" `
  -g "data\weak_sync_chain\header_first\0_0_0_10_14_32_header_first_symbols.csv" `
  -o "data\low_snr_gt_bin\0_0_0_10_14_32_extreme_snr" `
  --target-snr-db -25 -27 --overwrite --no-plots
```

Main sweep:

```powershell
python scripts/experiments/run_candidate_pruning_sweep.py `
  --snr-db 20,23,25,27 `
  --trend-source early-payload,header-offset `
  --preselect-mode default,allbin `
  --output-dir data/candidate_pruning/sweeps/validation_20260616_available_matrix `
  --resume
```

## Outputs

```text
data/candidate_pruning/sweeps/validation_20260616_available_matrix/
  manifest.json
  merged_summary.csv
  best_by_top_l.csv
  per_packet_recall.csv
  *_summary.json
  *.csv
```

Row counts:

```text
summary JSON files: 48
merged_summary.csv rows: 960
best_by_top_l.csv rows: 336
per_packet_recall.csv rows: 8960
```

## Main Early-Payload / Default Results

This is the closest path to the current proposed design.

| Dataset | SNR | L | Center | Multi-offset | Phase-gated | Gain vs multi | Rescue/Damage | Best config |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0_0_0_10_14_16 | -20 | 1 | 0.6416 | 0.9844 | 0.9818 | -0.0026 | 0/1 | w=0.02, gate=0.166667pi |
| 0_0_0_10_14_16 | -23 | 1 | 0.2675 | 0.6805 | 0.6961 | +0.0156 | 8/2 | w=0.2, gate=0.166667pi |
| 0_0_0_10_14_16 | -25 | 1 | 0.0883 | 0.3558 | 0.3610 | +0.0052 | 2/0 | w=0.02, gate=0.333333pi |
| 0_0_0_10_14_16 | -27 | 1 | 0.0312 | 0.1325 | 0.1351 | +0.0026 | 1/0 | w=0.05, gate=0.5pi |
| 0_0_0_10_14_32 | -20 | 1 | 0.6041 | 0.8571 | 0.8571 | +0.0000 | 0/0 | w=0.02, gate=0.166667pi |
| 0_0_0_10_14_32 | -23 | 1 | 0.2245 | 0.6571 | 0.6612 | +0.0041 | 2/1 | w=0.1, gate=0.5pi |
| 0_0_0_10_14_32 | -25 | 1 | 0.1184 | 0.2980 | 0.3061 | +0.0082 | 2/0 | w=0.02, gate=0.5pi |
| 0_0_0_10_14_32 | -27 | 1 | 0.0612 | 0.1388 | 0.1388 | +0.0000 | 0/0 | w=0.02, gate=0.166667pi |
| 0_0_0_10_14_8 | -20 | 1 | 0.6343 | 0.9543 | 0.9600 | +0.0057 | 2/0 | w=0.2, gate=0.5pi |
| 0_0_0_10_14_8 | -23 | 1 | 0.2057 | 0.6657 | 0.6743 | +0.0086 | 4/1 | w=0.1, gate=0.333333pi |
| 0_0_0_10_14_8 | -25 | 1 | 0.1114 | 0.3800 | 0.3829 | +0.0029 | 1/0 | w=0.02, gate=0.333333pi |
| 0_0_0_10_14_8 | -27 | 1 | 0.0429 | 0.1629 | 0.1657 | +0.0029 | 1/0 | w=0.05, gate=0.166667pi |

| Dataset | SNR | L | Center | Multi-offset | Phase-gated | Gain vs multi | Rescue/Damage | Best config |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0_0_0_10_14_16 | -20 | 8 | 0.8390 | 1.0000 | 1.0000 | +0.0000 | 0/0 | w=0.02, gate=0.166667pi |
| 0_0_0_10_14_16 | -23 | 8 | 0.5273 | 0.8571 | 0.8727 | +0.0156 | 6/0 | w=0.2, gate=0.25pi |
| 0_0_0_10_14_16 | -25 | 8 | 0.3039 | 0.6026 | 0.6026 | +0.0000 | 1/1 | w=0.1, gate=0.166667pi |
| 0_0_0_10_14_16 | -27 | 8 | 0.1636 | 0.3429 | 0.3455 | +0.0026 | 1/0 | w=0.02, gate=0.5pi |
| 0_0_0_10_14_32 | -20 | 8 | 0.7918 | 0.8898 | 0.8980 | +0.0082 | 2/0 | w=0.2, gate=0.25pi |
| 0_0_0_10_14_32 | -23 | 8 | 0.4531 | 0.7918 | 0.8041 | +0.0122 | 3/0 | w=0.1, gate=0.25pi |
| 0_0_0_10_14_32 | -25 | 8 | 0.2939 | 0.5959 | 0.6122 | +0.0163 | 4/0 | w=0.1, gate=0.166667pi |
| 0_0_0_10_14_32 | -27 | 8 | 0.1551 | 0.3224 | 0.3265 | +0.0041 | 1/0 | w=0.05, gate=0.166667pi |
| 0_0_0_10_14_8 | -20 | 8 | 0.8429 | 0.9886 | 0.9886 | +0.0000 | 0/0 | w=0.02, gate=0.166667pi |
| 0_0_0_10_14_8 | -23 | 8 | 0.4657 | 0.8457 | 0.8486 | +0.0029 | 1/0 | w=0.02, gate=0.166667pi |
| 0_0_0_10_14_8 | -25 | 8 | 0.2943 | 0.5857 | 0.5943 | +0.0086 | 5/2 | w=0.2, gate=0.5pi |
| 0_0_0_10_14_8 | -27 | 8 | 0.1514 | 0.3800 | 0.3886 | +0.0086 | 5/2 | w=0.2, gate=0.333333pi |

## Aggregated Main-Line Results

| L | Mean center | Mean multi-offset | Mean phase-gated | Mean gain vs multi | Positive / zero / negative | Total rescue/damage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2526 | 0.5223 | 0.5267 | +0.0044 | 9/2/1 | 23/5 |
| 8 | 0.4402 | 0.6835 | 0.6901 | +0.0066 | 9/3/0 | 29/5 |
| 16 | 0.5131 | 0.7366 | 0.7404 | +0.0038 | 7/4/1 | 19/5 |
| 32 | 0.5929 | 0.7869 | 0.7894 | +0.0026 | 6/6/0 | 14/4 |

## Findings

1. Phase-gated scoring is consistently much better than the center FFT
   argmax/Top-L baseline, but most of that gain comes from the multi-offset
   evidence baseline rather than phase itself.
2. Against the current multi-offset baseline, phase-gated scoring gives small
   average gains.  The strongest main-line result is Top-8, with mean gain
   +0.0066 across the 12 dataset/SNR points.
3. The all-bin phase bonus path matched default energy preselect exactly for
   Recall@1/8/16/32 in this sweep.  The current limitation is not preselect
   excluding GT bins.
4. Early-payload and header-offset are close.  Early-payload is not uniformly
   best across the deeper -25/-27 dB points; header-offset sometimes wins by a
   small amount.  This argues for a phase-quality gate or adaptive trend-source
   selection before wiring this into a real decoder path.
5. Phase damage is controlled but not zero.  The main-line sweep still has
   damage events, especially at Top-1 and in a few weak -27 dB cases.

## Recommendation

Do not connect this phase bonus directly into the two-stage decoder yet.
The Recall@L result is promising as a conservative bonus, especially at Top-8,
but the extra gain over multi-offset is small and trend-source dependent.

Next engineering step: add a phase-quality gate or per-packet adaptive selector:

```text
if trend_quality is weak:
    use pure multi-offset
else:
    use phase-gated score
```

Then re-run this same sweep and compare `phase_damage_count` and
`gain_vs_multi_offset` before integrating into payload search.

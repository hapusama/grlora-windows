# Low-SNR Argmax Neighborhood Experiment

This directory answers a narrow question:

```text
When low-SNR FFT hard decision chooses an argmax bin, is the clean/GT bin
usually near that argmax in circular FFT-bin coordinates?
```

Input data comes from `run_low_snr_gt_bin_experiment.py` feature CSVs.  The
analysis does not regenerate noisy IQ and does not rerun detection/sync.  It
uses the already exported columns:

```text
gt_raw_fft_bin      clean/high-SNR payload bin treated as GT
noisy_argmax_bin   low-SNR FFT hard-decision argmax
gt_bin_rank        GT bin rank by low-SNR FFT power
```

## Command

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe scripts\experiments\analyze_low_snr_argmax_neighborhood.py `
  -i data\low_snr_gt_bin\0_0_0_10_14_16\0_0_0_10_14_16_low_snr_gt_bin_features_all.csv `
     data\low_snr_gt_bin\0_0_0_10_14_16_extreme_snr\0_0_0_10_14_16_low_snr_gt_bin_features_all.csv `
  -o data\low_snr_argmax_neighborhood
```

## Outputs

```text
argmax_neighborhood_symbols.csv
argmax_neighborhood_summary_by_snr.csv
argmax_neighborhood_summary_by_packet.csv
argmax_neighborhood_hit_rates.png
argmax_neighborhood_wrong_argmax_hit_rates.png
argmax_neighborhood_abs_distance_boxplot.png
argmax_neighborhood_signed_delta_histograms.png
```

## Main Result

For SF10 (`n_bins=1024`), the GT bin is near the low-SNR argmax only when the
argmax is already correct.  When argmax is wrong, the GT bin is usually far away
in bin coordinates:

```text
-20 dB: wrong-only median distance = 274.5 bins, within +/-32 bins = 5.8%
-23 dB: wrong-only median distance = 252.5 bins, within +/-32 bins = 3.9%
-25 dB: wrong-only median distance = 251.0 bins, within +/-32 bins = 3.7%
-27 dB: wrong-only median distance = 243.0 bins, within +/-32 bins = 4.6%
```

So the useful low-SNR observation is not "search around argmax locally".
Instead, GT often remains in the high-power candidate set even when it is far
from argmax:

```text
-20 dB: top16 recall = 88.8%, top32 recall = 92.7%
-23 dB: top16 recall = 59.2%, top32 recall = 68.3%
-25 dB: top16 recall = 42.3%, top32 recall = 51.4%
-27 dB: top16 recall = 23.4%, top32 recall = 34.0%
```

This favors a global top-K candidate/rerank strategy over a local neighborhood
search centered on the hard-decision argmax.

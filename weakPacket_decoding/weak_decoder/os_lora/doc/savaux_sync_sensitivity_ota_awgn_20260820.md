# Savaux sensitivity to STO/CFO uncertainty

## Goal

This is the first-stage test for a synchronization-hypothesis Savaux receiver.
It asks only whether the existing Savaux metric is sensitive to synchronization
mismatch. It does not maximize over hypotheses and does not implement a new
decoder.

## Protocol

The experiment uses canonical ADC-phase-zero SF12/BW125 kHz OTA packets from
the sibling `lora-rfsr-savaux` dataset.

1. Run FrameSync once on each clean 1 MS/s packet.
2. Freeze the payload boundary, CFO, and SFO sample schedule.
3. Convert the FrameSync payload cursor to Savaux's full-OSR symbol-origin
   convention with the deterministic `-OSR/2 = -4` ADC-sample offset.
4. Add one complex AWGN realization at 1 MS/s, flat inside the common 125 kHz
   LoRa channel and zero outside it.
5. Reuse the same noisy guarded symbol at every artificial STO/CFO mismatch.
6. Run unchanged Savaux and record SER and true-to-strongest-false margin.

The noise convention is

\[
E_s/N_0 = N P_s/P_n, \qquad N=2^{SF}.
\]

Fractional STO is applied with a 33-tap Kaiser-windowed sinc interpolator to a
guarded slice of the real packet. It therefore imports the true adjacent packet
samples rather than circularly shifting an isolated symbol.

The `-4` sample origin conversion is important. Before adding AWGN, the four
packets' 64 tested symbols are all correct for offsets `-6` through `-1`, with
the maximum mean true margin at `-4` samples:

| FrameSync-relative start | clean correct | mean true margin |
|---:|---:|---:|
| -5 samples | 64/64 | 13.87 dB |
| **-4 samples** | **64/64** | **20.31 dB** |
| -3 samples | 64/64 | 17.98 dB |
| 0 samples | 50/64 | 1.33 dB |

Treating the unconverted FrameSync cursor as Savaux's zero would therefore
confound a fixed interface convention with random synchronization uncertainty.

## Main result

The formal run uses four packets, 16 payload symbols per packet, two seeds, and
`Es/N0=14 dB`. All 64 clean symbols pass admission, giving 128 common-noise
trials per heatmap cell. The oracle-hypothesis cell `(delta_tau,delta_f)=(0,0)`
has SER 0.0078.

At zero residual STO:

| CFO mismatch | SER |
|---:|---:|
| -7.63 Hz (-0.25 bin) | 0.0078 |
| 0 Hz | 0.0078 |
| +7.63 Hz (+0.25 bin) | 0.0312 |
| -15.26 Hz (-0.5 bin) | 0.3750 |
| +15.26 Hz (+0.5 bin) | 0.7578 |
| +/-22.89 Hz (+/-0.75 bin) | 1.0000 |
| +/-125 Hz through +/-2 kHz | 1.0000 |

At zero residual CFO:

| STO mismatch at 1 MS/s | chip mismatch | SER |
|---:|---:|---:|
| -0.5 sample | -0.0625 chip | 0.0000 |
| +0.5 sample | +0.0625 chip | 0.0078 |
| -2 samples | -0.25 chip | 0.0234 |
| +2 samples | +0.25 chip | 0.0078 |
| -4 samples | -0.5 chip | 0.8125 |
| +4 samples | +0.5 chip | 0.3828 |

Thus Savaux is extremely sensitive to residual CFO at the scale of one LoRa
FFT bin (`B/N = 30.5176 Hz`), but it is comparatively tolerant to sub-sample
STO around the calibrated origin. The two errors interact away from the origin,
as expected from the dechirped residual-frequency equivalence, but the low-SER
region is dominated by a narrow CFO strip.

## Decision

This direction passes the sensitivity gate and should proceed to the oracle
multi-hypothesis experiment. The first useful hypothesis bank should emphasize
a fine CFO grid rather than a large STO grid:

- CFO residuals near `{-0.5,-0.25,0,+0.25,+0.5}` LoRa bins;
- a small STO set such as `{-1,0,+1}` 1 MS/s samples;
- unchanged noisy symbols and unchanged Savaux scores for the single-sync,
  max-over-hypotheses, and oracle-sync comparison.

Kilohertz-spaced hypotheses are not useful at this stage: every tested
`|delta_f| >= 125 Hz` condition has SER 1.0. The next experiment must also
measure whether the max over hypotheses creates a look-elsewhere penalty; the
sensitivity heatmap alone shows recoverable mismatch but does not prove that
unweighted maximization improves SER.

## Follow-up outcome

The subsequent whole-packet AWGN experiment reran the real FrameSync estimator
instead of injecting independent CFO/STO perturbations. Across 80 packet trials,
no coordinate-local estimate fell in the raw-CFO 0.25--0.75-bin band. Large CFO
errors were usually coupled to timing displacement along the LoRa chirp ambiguity
ridge, and strict FrameSync validation caused more measured end-to-end loss than
small residual CFO. Therefore the broad per-symbol hypothesis bank proposed
above is no longer the default next step. The evidence, coupled-residual audit,
and narrower packet-shared follow-up are documented in
`noisy_framesync_headroom_ota_awgn_20260821.md`.

## Reproduction

Run from `gr-lora_sdr`:

```powershell
python weakPacket_decoding\weak_decoder\os_lora\experiments\analyze_savaux_sync_sensitivity_ota.py `
  --max-packets 4 --symbols-per-packet 16 `
  --esn0-db 14 --seeds 20260819,20260820
```

The output directory contains the clean-sync audit, all per-hypothesis trials,
the two-dimensional summary, the wide CFO control, and three plots.

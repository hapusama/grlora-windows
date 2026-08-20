# AliasTrim: sampled narrowband trajectory cancellation

## Mechanism

AliasTrim implements the proposed observation without reconstructing a higher
sample rate. Let `d[n]` be the receiver downchirp. After dechirping,

```text
desired LoRa:       a * s_k[n]
narrowband blocker: A * exp(j*w*n) * d[n]
```

The first term is the exact oversampled candidate tone, including its LoRa
wrap phase. The second term is a conjugate-chirp trajectory. A dense STFT mask
could identify that diagonal, but it is unnecessary: multiplying the residual
by `conj(d[n])` is the matched inverse trajectory and turns the blocker back
into a stationary tone at the existing sample rate.

The implemented path is:

1. Run the ordinary Savaux detector.
2. Fit and remove the provisional desired LoRa candidate with a complex median.
3. Apply the inverse trajectory to the residual.
4. Use a 4x zero-padded FFT and log-parabolic peak refinement to estimate at
   most two sampled blocker tones.
5. Accept a tone only when its FFT prominence is at least 18 dB and it explains
   at least 2% of residual power.
6. Least-squares subtract accepted tones, rerun Savaux, and change the hard bin
   only if the cleaned candidate wins by at least 3 dB.

No pre-alias RF frequency is inferred. All frequencies separated by an integer
multiple of the sample rate have the same sampled trajectory, which is enough
for cancellation but not for identifying where the blocker originated.

## Complexity

For `M = OSR * 2^SF`, zero-padding factor `Z = 4`, and at most `J = 2`
blockers:

- closed or rejected gate: `O(Z*M*log(Z*M))` time and `O(Z*M)` memory;
- accepted blocker: the same detector plus one additional Savaux pass;
- no STFT image, high-rate interpolation, neural network, LASSO, or dense
  recovery matrix is used.

On the SF10 Python evaluation, the existing Savaux pass averaged about
1.2--1.3 ms per symbol. AliasTrim overhead averaged 1.45 ms under AWGN and
3.1--3.5 ms with an accepted blocker. These timings are machine-dependent but
remain below the 8.192 ms SF10/BW125 symbol duration in this offline Python
implementation.

## Evaluation protocol

- Desired signals: the three existing real IQ captures
  `data/USRP_IQ/0_0_0_10_14_{8,16,32}.bin`.
- Symbols: 980 frozen payload windows, centered at synchronized start plus
  `OSR/2`.
- Added noise: proper complex AWGN using payload-window power as reference.
- Seeds: 2101, 2202, and 2303.
- Blocker: one phase-continuous sampled CW at normalized frequency `0.05`
  cycles/sample. At 500 ksample/s this is 25 kHz, inside the nominal
  125 kHz LoRa channel.
- Alias interpretation: an out-of-band CW and any frequency differing from it
  by an integer multiple of the sample rate produce this same discrete-time
  tone. The experiment injects that exact post-alias equivalent and never
  generates a higher-rate waveform.
- Ground truth is used only after Savaux and AliasTrim hard decisions.
- Synchronization is frozen, so the result measures demodulation rather than
  packet detection under interference.

The repository's `USRP_collector/data/branch4_fixed/interference` directory
currently contains only a capture README, not interference IQ. Consequently,
the blocker result below is a controlled real-signal/synthetic-interferer
experiment, not a measured OTA blocker claim.

## AWGN safety control

| Added SNR | Symbols | Savaux SER | AliasTrim SER | Gates | Fixes | Breaks |
|---:|---:|---:|---:|---:|---:|---:|
| -22 dB | 2,940 | 4.29% | 4.29% | 0 | 0 | 0 |
| -24 dB | 2,940 | 14.42% | 14.42% | 0 | 0 | 0 |
| -26 dB | 2,940 | 40.61% | 40.61% | 0 | 0 | 0 |

All 8,820 decisions are exactly Savaux. Dense white noise does not form the
matched conjugate-chirp trajectory and is not removable by AliasTrim.

## In-band sampled blocker

All conditions below also contain -24 dB AWGN. ISR is blocker power relative
to the payload reference power.

| Blocker ISR | Symbols | Savaux SER | AliasTrim SER | Fixes | Breaks |
|---:|---:|---:|---:|---:|---:|
| 25 dB | 2,940 | 61.56% | 32.14% | 865 | 0 |
| 30 dB | 2,940 | 70.78% | 29.42% | 1,216 | 0 |
| 35 dB | 2,940 | 80.10% | 21.84% | 1,713 | 0 |

The estimated blocker-frequency RMSE is `1.36e-6`, `7.96e-7`, and `4.89e-7`
cycles/sample for ISR 25, 30, and 35 dB respectively. Every symbol accepted
exactly one trajectory; the second-trajectory gate remained closed.

The residual gap relative to the no-blocker -24 dB SER of 14.42% shows that
cancellation is not perfect, especially at 25 dB ISR. Nevertheless, the paired
fix/break counts support the proposed mechanism under a narrowband blocker.

An additional two-packet diagnostic swept normalized frequencies
`-0.45, -0.25, -0.05, 0.05, 0.17321, 0.35, 0.49`. Harmful in-band/intermediate
tones were corrected with zero observed breaks; tones near regions that did
not disturb Savaux left its hard decision unchanged. This sweep is diagnostic,
not part of the formal aggregate table.

## Reproduction

Run from `weakPacket_decoding`.

AWGN control:

```powershell
python -m weak_decoder.os_lora.experiments.archive.evaluate_alias_trim `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snrs -22 -24 -26 --seeds 2101 2202 2303 `
  --blocker-isrs --include-no-blocker --blocker-frequency 0.05 `
  --output-dir data\experiments\alias_trim_awgn_20260819
```

In-band sampled blocker:

```powershell
python -m weak_decoder.os_lora.experiments.archive.evaluate_alias_trim `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snrs -24 --seeds 2101 2202 2303 `
  --blocker-isrs 25 30 35 --blocker-frequency 0.05 `
  --output-dir data\experiments\alias_trim_blocker_inband_20260819
```

Each run writes `symbols.csv`, `summary.csv`, and `config.json`.

## Validity boundary

AliasTrim is appropriate for continuous-wave blockers, stable spurs, and a
small number of narrowband interferers. It cannot recover information lost to
wideband thermal-noise folding, a blocker occupying much of the LoRa band, or
an arbitrary fast-varying interferer. The next decisive experiment is an OTA
capture with recorded blocker center frequency, bandwidth, power, and duty
cycle. Thresholds should remain frozen for that validation.

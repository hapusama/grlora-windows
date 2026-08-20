# Gated robust sparse-corruption demodulator

## Decision

The implemented receiver does not use a neural upsampler or a general sparse
reconstruction solver. It keeps the existing Savaux oversampled detector as
the first stage and adds an adaptive, bounded-cost robust reranker only when
the sample residual contains enough outliers.

For candidate bin `k`, the dechirped symbol is modeled as

```text
y[n] = a * s_k[n] + w[n] + o[n]
```

where `a` is an unknown common complex amplitude, `w[n]` is dense receiver
noise, and `o[n]` is sparse corruption. The implementation removes the exact
oversampled LoRa phase law for each candidate and estimates `a` with three
Huber IRLS iterations. It then compares the reduction in robust Huber loss.

The default control flow is:

1. Compute the ordinary Savaux spectrum and retain its Top-16 bins.
2. Fit only the Savaux winner and estimate its complex-noise scale from the
   median residual radius.
3. If fewer than 2% of samples exceed `2.5 * scale`, return Savaux exactly.
4. Otherwise fit and score the Top-16 candidates.
5. Change the hard bin only if the best alternative has at least 3 dB robust
   score gain over the Savaux winner.

This is a sparse-outlier receiver, not a claim that LoRa symbols are sparse in
time. It borrows the useful part of the sparse-recovery viewpoint--explicitly
modeling a small corruption support--without paying for LASSO, OMP over a full
dictionary, or a neural network.

## Complexity

Let `N = OSR * 2^SF`, `K = 16`, and `R = 3` IRLS iterations.

- Gate closed: robust overhead is `O(N * R)` time and `O(N)` working memory.
- Gate open: robust overhead is `O(K * N * R)` time and `O(K * N)` memory.
- The first-stage Savaux cost is unchanged.

In the SF10 Python experiment, pure-AWGN windows spent an average of 0.744 ms
in the robust gate, in addition to 1.473 ms in the existing Savaux code. These
timings are machine-dependent, but the adaptive path is comfortably bounded
and does not allocate a `N x N` recovery matrix.

## Evaluation protocol

- IQ files: the three existing `data/USRP_IQ/0_0_0_10_14_{8,16,32}.bin`
  captures.
- Synchronization and ground truth: frozen header-first CSV files.
- Evaluated region: payload symbols only.
- Demodulation origin: the synchronized start plus `OSR/2`, matching the
  centered Savaux convention used by the existing evaluators.
- Added noise: reproducible proper complex AWGN, using average payload-window
  IQ power as the signal reference.
- The Savaux and robust decisions receive the same noisy symbol.
- This is a post-synchronization demodulation test; it does not measure packet
  detection or synchronization failure under added noise.

### Pure-AWGN control

There are 980 frozen payload symbols across the three captures. Three seeds at
each SNR produce 2,940 paired decisions per row.

| Added SNR | Symbols | Savaux SER | Robust SER | Gates | Fixes | Breaks |
|---:|---:|---:|---:|---:|---:|---:|
| -18 dB | 2,940 | 1.73% | 1.73% | 0 | 0 | 0 |
| -20 dB | 2,940 | 2.55% | 2.55% | 0 | 0 | 0 |
| -22 dB | 2,940 | 4.42% | 4.42% | 0 | 0 | 0 |

All 8,820 AWGN decisions are bit-for-bit identical to Savaux. This negative
result is intentional and important: dense Gaussian noise does not create the
sparse structure required by the robust path, so a compressed-sensing prior
does not provide a free SNR gain.

### Sparse-impulse stress test

The held-out stress test adds AWGN at -18 dB and corrupts 4% randomly selected
samples with complex Gaussian impulses. Impulse power at an active sample is
35 dB above the payload reference power. This is a severe synthetic outlier
test, not a model fitted to a measured interferer.

| Dataset suffix | Symbols | Savaux SER | Robust SER | Fixes | Breaks |
|---|---:|---:|---:|---:|---:|
| `8` | 1,050 | 6.67% | 1.24% | 57 | 0 |
| `16` | 1,155 | 3.98% | 0.43% | 41 | 0 |
| `32` | 735 | 13.88% | 12.24% | 12 | 0 |
| **Total** | **2,940** | **7.41%** | **3.67%** | **110** | **0** |

The net reduction is 110 symbol errors, or 3.74 percentage points absolute.
The result supports the intended failure mode: the method helps when a small
sample subset has much larger errors than the dense noise floor. It does not
establish a gain on ordinary AWGN or on every real LoRa interference channel.

## Reproduction

Run from `weakPacket_decoding` with the GNU Radio Conda Python environment.

Pure AWGN:

```powershell
python -m weak_decoder.os_lora.experiments.archive.evaluate_robust_sparse_demod `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snrs -18 -20 -22 --seeds 401 502 603 `
  --output-dir data\experiments\robust_sparse_demod_awgn_20260819
```

Held-out sparse impulses:

```powershell
python -m weak_decoder.os_lora.experiments.archive.evaluate_robust_sparse_demod `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snrs -18 --seeds 1101 1202 1303 `
  --impulse-fraction 0.04 --impulse-isr-db 35 --impulse-layout random `
  --output-dir data\experiments\robust_sparse_demod_impulse35_20260819
```

Each run writes `symbols.csv`, `summary.csv`, and `config.json`. Payload ground
truth is used only after both hard-bin decisions have been made.

## Deployment boundary

The reusable API is
`weak_decoder.decoding.demod_robust_sparse_symbol`. It requires no payload,
CRC, FEC, or ground-truth input. Before enabling it by default in an online GNU
Radio block, the next validation should measure residual outlier fractions on
the project's real interference captures. If those captures do not open the
gate, the correct conclusion is to retain Savaux rather than lowering the gate
until synthetic gains appear.

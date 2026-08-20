# Noisy FrameSync headroom for Savaux under whole-packet AWGN

## Question

The preceding sensitivity experiment showed that Savaux becomes fragile when an
otherwise correct synchronization point is perturbed by roughly 0.25--0.75 LoRa
CFO bin. This experiment tests whether a real weak-packet FrameSync estimator
actually produces that kind of residual error, and whether noisy synchronization
creates a meaningful gap to clean/oracle synchronization.

## Protocol

- Dataset: 8 clean 1 MS/s OTA captures from `lora-rfsr-savaux`.
- PHY: SF12, BW 125 kHz, OSR 8; one LoRa bin is 30.517578 Hz.
- Trials: 5 Es/N0 values, 2 deterministic AWGN seeds, 16 payload symbols per
  packet: 80 packet trials and 1280 symbol decisions.
- Noise: complex AWGN flat inside the 125 kHz LoRa channel, added once to the
  complete packet before FrameSync.
- Oracle branch: Savaux uses clean FrameSync CFO/STO/SFO on the noisy IQ.
- Noisy branch: the real FrameSync implementation is rerun on the noisy IQ and
  its returned CFO/STO/SFO is used by Savaux.
- A returned estimate is evaluated even when strict FrameSync validation rejects
  it. Estimate quality and validation/gating loss are therefore kept separate.

The sample size is deliberately preliminary. Each packet-rate result has a
resolution of 1/16 = 6.25 percentage points.

## Main results

| Es/N0 (dB) | strict sync | estimate available | coordinate-local estimate | local raw CFO in 0.25--0.75 bin | oracle SER | noisy-estimate SER | strict end-to-end SER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.125 | 0.938 | 0.625 | 0.000 | 0.1172 | 0.4000 | 0.8984 |
| 13 | 0.438 | 1.000 | 0.750 | 0.000 | 0.0234 | 0.2070 | 0.5742 |
| 14 | 0.625 | 1.000 | 0.750 | 0.000 | 0.0000 | 0.1875 | 0.4375 |
| 15 | 0.750 | 1.000 | 0.938 | 0.000 | 0.0000 | 0.0000 | 0.2500 |
| 16 | 0.938 | 1.000 | 1.000 | 0.000 | 0.0000 | 0.0000 | 0.0625 |

Coordinate-local means both `|CFO error| <= 2 bins` and
`|payload STO error| <= 32` 1M samples. Within this subset, median absolute raw
CFO error is only 0.0026--0.0061 bin. No coordinate-local estimate at any tested
SNR falls in the previously identified raw-CFO fatal band.

The apparently large errors are usually not independent CFO failures. They are
strongly coupled to timing displacement along the LoRa chirp ambiguity ridge.
For example, a CFO error close to -1 bin occurs with an STO error of -8 samples,
or -1 chip. At 14 dB that joint displacement still decodes 16/16 payload symbols.
A useful first-order diagnostic is

```text
coupled residual proxy = CFO error (bins) - STO error (samples) / OSR.
```

The fraction of available estimates whose absolute proxy lies in 0.25--0.75 bin
is 0.133, 0.0625, 0.125, 0.0625, and 0 at 12--16 dB. This identifies a small
refinement target, but the proxy is not a sufficient decoder statistic. At
14 dB, two trials have a proxy near +0.51 bin: one produces 16/16 noisy-sync
errors while the other produces 0/16. Chirp wrap position and the full
CFO/STO hypothesis still matter.

Conditioning on coordinate-local estimates almost removes the oracle gap:

| Es/N0 (dB) | local oracle SER | local noisy-sync SER | paired gap |
|---:|---:|---:|---:|
| 12 | 0.1000 | 0.1063 | 0.0063 |
| 13 | 0.0156 | 0.0156 | 0.0000 |
| 14 | 0.0000 | 0.0000 | 0.0000 |
| 15 | 0.0000 | 0.0000 | 0.0000 |
| 16 | 0.0000 | 0.0000 | 0.0000 |

At 12 dB, the local paired audit contains two symbols correct only with oracle
sync and one correct only with noisy sync. At 13 dB all three local errors are
shared by both branches. Thus there is essentially no broad local-CFO recovery
headroom in this sample.

## Decision

A five-point CFO bank should not be enabled for every payload symbol. The real
FrameSync estimator does not commonly leave the small independent CFO residual
suggested by the sensitivity sweep. Most of the 12--14 dB aggregate gap comes
from large, coupled synchronization-coordinate jumps; a small CFO-only bank
cannot generally recover those events.

There is still a bounded follow-up worth running:

1. Keep the noisy FrameSync start fixed and evaluate the five packet-shared CFO
   offsets `{-0.5, -0.25, 0, +0.25, +0.5}` bin.
2. Report results separately for coordinate-local estimates, coupled-residual
   0.25--0.75-bin estimates, and coordinate-nonlocal estimates.
3. Select one offset for the complete packet using preamble likelihood; retain
   per-symbol max only as a look-elsewhere ablation.
4. Evaluate strict validation separately. At 15 dB, returned estimates give zero
   Savaux SER while strict gating alone raises end-to-end SER to 0.25. Relaxing or
   reranking the FrameSync candidate/gate has more measured headroom than broad
   CFO-only marginalization.

This follow-up can answer whether the few coupled half-bin events are recoverable
without turning the result into an artificial per-symbol hypothesis gain.

The strict-gate follow-up is now complete. All 15 strict-invalid but fully
decodable trials failed the all-preamble-bin0 condition. A decoder-aware policy
that retains frame-location and netID checks reaches oracle-acceptance PDR at
14--16 dB. See `decoder_aware_framesync_gate_ota_awgn_20260822.md`.

## Reproduction

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python weakPacket_decoding/weak_decoder/os_lora/experiments/analyze_noisy_framesync_headroom_ota.py `
  --max-packets 8 --symbols-per-packet 16 `
  --esn0-db 12,13,14,15,16 --seeds 20260821,20260822 --workers 4 `
  --output-dir weakPacket_decoding/data/experiments/noisy_framesync_headroom_ota_awgn_20260821
```

To regenerate plots and summaries without rerunning FrameSync:

```powershell
python weakPacket_decoding/weak_decoder/os_lora/experiments/analyze_noisy_framesync_headroom_ota.py `
  --summarize-existing --esn0-db 12,13,14,15,16 `
  --output-dir weakPacket_decoding/data/experiments/noisy_framesync_headroom_ota_awgn_20260821
```

# Ambiguity-ridge synchronization with bounded soft-Hamming decoding

## Decision

The decoder linkage is useful under channel-band AWGN.  On 8 clean OTA SF12
captures, 10 independent AWGN seeds per capture, and 7 Es/N0 points from 10 to
16 dB, ambiguity-ridge K=4 plus bounded soft-Hamming decoding improves exact
packet delivery from 309/560 to 370/560 relative to the same ridge list with
hard symbol decisions.  It fixes 61 hard-ridge failures, breaks none of the
309 exact hard-ridge deliveries, and improves every physical capture.

The useful interpretation is a lower required Es/N0, not information created
by resampling.  Linear interpolation of the measured waterfall gives:

| Receiver | Es/N0 at PDR50 | Es/N0 at PDR80 |
|---|---:|---:|
| strict FrameSync + hard decoding | 13.714 dB | 15.182 dB |
| relaxed gate + hard decoding | 13.000 dB | 14.235 dB |
| ambiguity ridge K=4 + hard decoding | 12.500 dB | 13.417 dB |
| ambiguity ridge K=4 + soft Hamming | **11.676 dB** | **12.632 dB** |

Thus soft Hamming contributes about 0.82 dB at PDR50 and 0.79 dB at PDR80
beyond hard ridge synchronization.  The complete receiver improves by about
2.04 dB at PDR50 and 2.55 dB at PDR80 relative to the original strict path.

## Decoder

Savaux already computes a full dechirp/matched-filter spectrum for every LoRa
symbol.  The ordinary path keeps only the Argmax.  The linked decoder instead:

1. converts the complete spectrum to semantic LoRa-symbol log scores;
2. marginalizes those scores to interleaved bit log probabilities;
3. for each deinterleaved codeword, evaluates the 16 valid Hamming codewords;
4. reconstructs the selected LoRa symbols;
5. runs the unchanged explicit header, payload deinterleaving, dewhitening,
   and PHY-CRC path.

The search is bounded at 16 codewords per decoded nibble.  It is not the old
payload beam search, does not enumerate payloads, and never uses expected bytes
for selection.  Expected bytes are consulted only after CRC to measure exact
delivery and identify CRC collisions.

The synchronization list is unchanged: candidate zero is the noisy FrameSync
coordinate, alternatives are selected from the strongest SFD integer-CFO
peaks, and timing moves with CFO along LoRa's ambiguity ridge.  K=4 is tried in
rank order and decoding stops at the first payload-CRC success.

## Experiment protocol

- Physical data: 8 clean OTA packets from the reference PHY dataset.
- PHY: SF12, BW125 kHz, CR4/8, LDRO, explicit header, payload CRC.
- Sampling: 1 MS/s, OSR=8.
- Noise: complex AWGN limited to the LoRa channel bandwidth, added once to the
  complete packet before the real FrameSync.
- Seeds: 20260821 through 20260830.
- Trials: 80 per SNR and 560 total.
- Synchronization policies and decoders see identical noisy IQ in each trial.
- Clean calibration: all 8/8 OTA packets round-trip through the complete PHY
  chain exactly.

The 560 trials are repeated-noise trials over only 8 physical captures.  They
measure conditional AWGN robustness well, but they are not 560 independent
channel/device/location observations.

## Packet delivery results

| Es/N0 | strict | relaxed gate | ridge K=4 hard | ridge K=4 soft | hard oracle | soft oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 10 dB | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 1.25% |
| 11 dB | 0.00% | 0.00% | 0.00% | **18.75%** | 0.00% | 36.25% |
| 12 dB | 2.50% | 13.75% | 26.25% | **65.00%** | 35.00% | 78.75% |
| 13 dB | 31.25% | 50.00% | 73.75% | **88.75%** | 78.75% | 97.50% |
| 14 dB | 57.50% | 75.00% | 88.75% | **92.50%** | 95.00% | 100.00% |
| 15 dB | 77.50% | 96.25% | 98.75% | 98.75% | 100.00% | 100.00% |
| 16 dB | 91.25% | 98.75% | 98.75% | 98.75% | 100.00% | 100.00% |

At 12 dB, where the hard-ridge receiver is below its waterfall midpoint, soft
Hamming recovers 31 additional packets and raises PDR from 26.25% to 65.00%.
At 13 dB it recovers another 12 packets.  There are no soft regressions at any
measured SNR.

The per-capture hard-to-soft exact-delivery counts over 70 trials each are:

```text
25 -> 30, 46 -> 51, 40 -> 51, 37 -> 43,
41 -> 50, 38 -> 47, 40 -> 49, 42 -> 49
```

This rules out a gain caused by only one especially favorable OTA capture.

## Synchronization and decoder headroom

The soft oracle separates decoder failure from synchronization-list failure.
For example:

- at 11 dB, oracle-soft delivers 29/80 while ridge-soft delivers 15/80;
- at 12 dB, oracle-soft delivers 63/80 while ridge-soft delivers 52/80;
- at 13 dB, oracle-soft delivers 78/80 while ridge-soft delivers 71/80.

The decoder therefore creates real low-SNR headroom, but the remaining gap is
mostly candidate generation/ranking.  Increasing Hamming complexity cannot
recover a packet when the useful CFO/STO coordinate is absent from K=4.

## Complexity

CRC early stopping reduces the operational number of packet-decoder attempts:

| Es/N0 | hard ridge | ridge + soft Hamming |
|---:|---:|---:|
| 10 dB | 2.150 | 2.150 |
| 11 dB | 3.200 | 2.738 |
| 12 dB | 2.975 | 2.000 |
| 13 dB | 1.863 | 1.550 |
| 14 dB | 1.463 | 1.350 |
| 15 dB | 1.063 | 1.063 |
| 16 dB | 1.038 | 1.038 |

The overall soft mean is 1.698 attempts per input trial.  The Python prototype
spends about twice as long in soft candidate decoding as in hard candidate
decoding, but noisy FrameSync remains much more expensive in this offline
implementation.  These concurrent Python timings are not GNU Radio C++
latency measurements.

## CRC safety audit

One CRC-valid wrong delivery appears in each decoder family across the 560
trials, at different trials:

- hard ridge: 13 dB, trial
  `exp0_000000_rxg20_0_fulltrim:13:20260827`;
- soft ridge: 11 dB, trial
  `exp0_000005_rxg20_0_fulltrim:11:20260821`.

Soft Hamming corrects the hard decoder's 13 dB CRC collision, but produces its
own collision in the near-random 11 dB region.  No soft CRC-valid wrong packet
occurs from 12 through 16 dB.  This prevents claiming that PHY CRC alone makes
arbitrary list depth risk-free.  A deployed receiver should retain the small
K, reject operation below a calibrated synchronization/reliability floor, or
use stronger existing upper-layer integrity such as a LoRaWAN MIC.  The
experiment reports exact PDR separately from CRC acceptance, so the collision
is not counted as a successful packet.

## Comparison with other local decoder directions

- LiteNap-style undersampling discards full-rate AWGN evidence and was already
  inferior to full-sample Savaux in the pure-AWGN experiments.
- Covariance-correct multi-rate branches are deterministic transforms of the
  same waveform and do not add independent AWGN information.
- The robust sparse detector reduces to Savaux under pure AWGN because its
  structured-interference gate does not open.
- The historical payload/CRC beam can gain sensitivity but has combinatorial
  state growth and false-CRC exposure.  Bounded soft Hamming captures coding
  gain locally with exactly 16 candidates per nibble.

These comparisons make bounded soft Hamming the appropriate low-complexity
decoder to couple to ambiguity-ridge synchronization for the present AWGN
question.

## Limitations and next work

- Expand beyond 8 physical captures, locations, sessions, and devices.  More
  AWGN seeds reduce Monte Carlo noise but do not add RF diversity.
- Repeat at other SF, bandwidth, CR, payload length, and CRC modes.
- Add a general candidate reliability metric or stronger integrity rule for
  the sub-waterfall region instead of relying on estimated SNR alone.
- Improve candidate generation for the remaining oracle-soft gaps before
  increasing K.
- Port the soft interleaver/Hamming metric and ridge list to GNU Radio C++ and
  measure real CPU load and latency.

## Reproduction

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python weakPacket_decoding/weak_decoder/os_lora/experiments/evaluate_decoder_aware_crc_pdr_ota.py `
  --max-packets 8 --esn0-db 10,11,12,13,14,15,16 `
  --seeds 20260821,20260822,20260823,20260824,20260825,20260826,20260827,20260828,20260829,20260830 `
  --workers 8 --ridge-top-k 4 --ridge-sfd-peak-pool 32 `
  --ridge-soft-hamming `
  --output-dir weakPacket_decoding/data/experiments/ambiguity_ridge_soft_hamming_ota_awgn_10seeds_20260820
```

Artifacts:

- decoder: `weak_decoder/os_lora/system/soft_hamming_crc.py`;
- synchronization list and CRC arbiter:
  `weak_decoder/os_lora/system/ambiguity_ridge_list.py`;
- main experiment:
  `weak_decoder/os_lora/experiments/evaluate_decoder_aware_crc_pdr_ota.py`;
- decoder-only oracle experiment:
  `weak_decoder/os_lora/experiments/evaluate_soft_hamming_oracle_ota.py`;
- main output:
  `data/experiments/ambiguity_ridge_soft_hamming_ota_awgn_10seeds_20260820/`;
- independent oracle-soft output:
  `data/experiments/soft_hamming_oracle_ota_awgn_10seeds_20260820/`.

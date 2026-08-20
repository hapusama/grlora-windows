# CRC-aided ambiguity-ridge list synchronization

## Decision

The Top-K synchronization experiment supports the list-synchronization design.
On the same 80 OTA-plus-AWGN trials used by the decoder-aware gate experiment,
a four-coordinate ridge list raises exact CRC-valid packet delivery from 57/80
to 68/80.  Oracle synchronization delivers 70/80.  Thus K=4 recovers 11/13
of the measured soft-gate-to-oracle gap without a CRC false delivery.

This is more than a threshold relaxation: nine of the eleven K=4 gains require
an alternative synchronization coordinate.  The remaining two gains come from
letting CRC judge the original FrameSync coordinate after preamble validation
rejected it.

## Method

The experiment uses 8 clean SF12/BW125 kHz/CR4/8/LDRO OTA packets, 2 fixed
noise seeds, and 5 Es/N0 points from 12 through 16 dB.  Complex AWGN limited to
the 125 kHz LoRa channel is added once to each complete 1 MS/s packet before the
real FrameSync runs.  The trial population and noise realizations exactly match
the preceding full-FEC/CRC gate experiment.

For every available FrameSync estimate, the list builder:

1. reserves candidate zero for the original FrameSync coordinate;
2. computes the SFD2 chip-rate FFT and retains the strongest 32 unique
   integer-CFO hypotheses;
3. ranks alternatives by SFD power and instantiates only the requested Top-K;
4. maps each integer CFO to its coupled payload boundary along
   `delta_payload_samples ~= OSR * delta_CFO_bins`;
5. repeats the candidate-specific fractional-STO and SFO refinement;
6. records net-ID consistency as soft evidence rather than a hard gate.

Each ranked coordinate enters the unchanged Savaux hard-demodulation, explicit
PHY-header FEC/checksum, payload FEC, dewhitening, and PHY-CRC chain.  The
operational implementation stops at the first payload-CRC success.  The formal
audit decodes all K coordinates to detect CRC collisions.  Expected packet
bytes are never used for selection; they are consulted only after CRC to score
exact delivery.

This is a one-dimensional SFD/CFO list with coupled timing, not a Cartesian
CFO-by-STO grid.

## Packet delivery results

| Es/N0 (dB) | strict | soft gate | list K=1 | list K=2 | list K=4 | oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 6.25% | 25.00% | 31.25% | 43.75% | 50.00% | 50.00% |
| 13 | 37.50% | 62.50% | 68.75% | 75.00% | 81.25% | 87.50% |
| 14 | 50.00% | 75.00% | 75.00% | 87.50% | 93.75% | 100.00% |
| 15 | 68.75% | 93.75% | 93.75% | 100.00% | 100.00% | 100.00% |
| 16 | 93.75% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

Across all 80 trials:

| Policy | Exact packets | Gain over soft gate | Soft-to-oracle gap closed |
|---|---:|---:|---:|
| strict | 41 | -16 | - |
| soft gate | 57 | 0 | 0/13 |
| list K=1 | 59 | +2 | 2/13 |
| list K=2 | 65 | +8 | 8/13 |
| list K=4 | 68 | +11 | 11/13 |
| oracle | 70 | +13 | 13/13 |

K=4 has zero regressions relative to the soft gate and zero CRC false
deliveries.  The all-candidate audit contains 95 CRC-valid decodes; every one
reconstructs the expected packet.  Multiple successful coordinates in a trial
therefore correspond to decoding-equivalent points, not conflicting payloads,
in this dataset.

The first successful K=4 candidate is distributed as follows:

| Outcome | Trials |
|---|---:|
| original coordinate (index 0) | 59 |
| first SFD alternative | 6 |
| second SFD alternative | 2 |
| third SFD alternative | 1 |
| no CRC success | 12 |

## Complexity

The expensive candidate-specific synchronization work is O(K), while the SFD
peak pool is one FFT plus sorting.  Candidate zero is tried first and CRC stops
the search early.  Mean packet-decoder attempts per input trial are 0.988,
1.238, and 1.562 for K=1, K=2, and K=4, respectively; the value can be below
one because one trial has no FrameSync estimate.

Formal-run list generation averaged 44--49 ms per trial in Python.  Deliberately
decoding all four candidates for the collision audit averaged 457--630 ms per
trial, but this is not the operational early-stop cost.  These are offline
concurrent Python measurements and are not GNU Radio C++ latency numbers.

## Remaining two oracle gaps

A bounded K=32 diagnostic distinguishes list truncation from candidate-pool
failure:

- the 13 dB gap is recovered by candidate index 8 (SFD peak rank 9), with
  `delta_CFO=-778` bins, `delta_payload=-6225` samples, and ridge residual
  `+0.125` bin;
- the 14 dB gap has no CRC-valid coordinate among the strongest 32 unique SFD
  integer-CFO hypotheses.

Therefore a larger list can close one more oracle gap but is not a good default:
K=4 already obtains 68/70 oracle-deliverable trials at 1.562 mean attempts,
whereas the exceptional rank-9 recovery requires nine decoder attempts.  The
other miss needs a better candidate generator or an independent preamble view,
not merely a larger K.

## Limitations

- The OTA diversity is only 8 physical packets; each per-SNR estimate has
  1/16 resolution.
- Weakness is produced by controlled channel-band AWGN on clean OTA captures,
  not by a large corpus of naturally weak captures.
- Only SF12, BW125 kHz, CR4/8, LDRO, explicit header, and payload CRC are
  evaluated here.
- No CRC collision was observed, but the sample is too small to estimate an
  extremely low collision probability.
- The current implementation is an offline reusable Python system component;
  a GNU Radio C++ block and online latency audit remain future engineering work.

## Reproduction

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python weakPacket_decoding/weak_decoder/os_lora/experiments/evaluate_ambiguity_ridge_list_sync_ota.py `
  --esn0-db 12,13,14,15,16 --top-k-values 1,2,4 `
  --sfd-peak-pool 32 --workers 4 `
  --output-dir weakPacket_decoding/data/experiments/ambiguity_ridge_list_crc_ota_awgn_20260822
```

To regenerate summaries and the plot without rerunning FrameSync:

```powershell
python weakPacket_decoding/weak_decoder/os_lora/experiments/evaluate_ambiguity_ridge_list_sync_ota.py `
  --summarize-existing --esn0-db 12,13,14,15,16 `
  --top-k-values 1,2,4 `
  --output-dir weakPacket_decoding/data/experiments/ambiguity_ridge_list_crc_ota_awgn_20260822
```

Artifacts:

- reusable list builder and CRC arbiter:
  `weak_decoder/os_lora/system/ambiguity_ridge_list.py`;
- formal runner:
  `weak_decoder/os_lora/experiments/evaluate_ambiguity_ridge_list_sync_ota.py`;
- formal outputs:
  `data/experiments/ambiguity_ridge_list_crc_ota_awgn_20260822/`.

# Two-stage weak packet decoder

This implementation starts after weak sync and header-first have already
succeeded. It is PHY-only and does not use a payload template, counter pattern,
or cross-packet application prior.

## Algorithm

1. Keep the full complex FFT evidence `Z[k,b]` for each payload symbol.
   The runner can use the original center-sample FFT evidence or the newer
   `multi-offset` evidence.
2. Build raw-bin and demod-symbol likelihoods, plus Top-K evaluation metrics.
3. Work at the payload interleaver-block level instead of mapping one symbol
   error directly to one byte error.
4. Deinterleave symbol likelihoods into bit soft evidence for each Hamming
   codeword row.
5. Enumerate the 16 legal nibble/codeword candidates per codeword and keep a
   configurable candidate list.
6. Beam-search block candidates, then re-encode each payload candidate through
   the local LoRa PHY codec to project it back to theoretical payload bins.
7. Score projected bins against the original FFT likelihood with trimmed energy.
   A weak phase consistency term is available through `--phase-weight`.
8. Prefer CRC-valid beam candidates. If no beam candidate passes CRC, the
   default behavior is to abstain from destructive repair and return the
   traditional argmax path with `selected_source=argmax_fallback`.

## Main Files

- `weak_decoder/two_stage_weak_decoder.py`
- `scripts/experiments/run_two_stage_weak_decoder.py`
- `scripts/experiments/evaluate_phase_bin_metric.py`
- `scripts/experiments/evaluate_codec_bin_metric.py`

## Example Commands

Clean sanity:

```powershell
python "gr-lora_sdr/weakPacket_decoding copy/scripts/experiments/run_two_stage_weak_decoder.py" `
  -i "gr-lora_sdr/data/USRP_IQ/0_0_0_10_14_16.bin" `
  -s "gr-lora_sdr/weakPacket_decoding copy/data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols_netidvalid.csv" `
  -o "gr-lora_sdr/weakPacket_decoding copy/data/two_stage_weak_decoder/clean_all_results.csv" `
  --summary-json "gr-lora_sdr/weakPacket_decoding copy/data/two_stage_weak_decoder/clean_all_summary.json" `
  --phase-weight 0.0
```

Low-SNR payload-only evaluation:

```powershell
python "gr-lora_sdr/weakPacket_decoding copy/scripts/experiments/run_two_stage_weak_decoder.py" `
  -i "gr-lora_sdr/weakPacket_decoding copy/data/low_snr_gt_bin/0_0_0_10_14_16/0_0_0_10_14_16_snr_m15dB.bin" `
  -s "gr-lora_sdr/weakPacket_decoding copy/data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols_netidvalid.csv" `
  -o "gr-lora_sdr/weakPacket_decoding copy/data/two_stage_weak_decoder/snr_m15_all_fallback_results.csv" `
  --summary-json "gr-lora_sdr/weakPacket_decoding copy/data/two_stage_weak_decoder/snr_m15_all_fallback_summary.json" `
  --fft-evidence-mode multi-offset `
  --phase-weight 0.0
```

`multi-offset` fuses all oversampling phases for each payload symbol:

```text
score[b] = sum_offset |FFT_offset[b]|^2 / max_b |FFT_offset[b]|^2
```

This is the current best bin-candidate metric. It does not assume the true bin
is near the center-offset argmax; it uses the fact that a real LoRa tone remains
stable across the oversampled chip phases while many noise peaks do not.

## Current Validation

Parameters: `phase_weight=0.0`, `nibble_candidates=4`,
`row_beam_width=256`, `block_candidate_limit=64`,
`global_beam_width=64`.

Center FFT evidence:

| Dataset | Packets | Center argmax symbol SER | Two-stage symbol SER | CRC-valid rate | Top-K recall | Fallback rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 11 | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.000 |
| -10 dB | 11 | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.000 |
| -15 dB | 11 | 0.0026 | 0.0000 | 1.000 | 1.000 | 0.000 |
| -20 dB | 11 | 0.3584 | 0.3584 | 0.000 | 0.927 | 1.000 |

Multi-offset FFT evidence:

| Dataset | Packets | Multi-offset argmax symbol SER | Two-stage symbol SER | CRC-valid rate | Top-K recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 11 | 0.0000 | 0.0000 | 1.000 | 1.000 |
| -15 dB | 11 | 0.0000 | 0.0000 | 1.000 | 1.000 |
| -20 dB | 11 | 0.0156 | 0.0000 | 1.000 | 1.000 |
| len8 -23 dB | 10 | 0.3343 | 0.3171 | 0.200 | 0.909 |
| len32 -23 dB | 7 | 0.3429 | 0.3224 | 0.143 | 0.865 |

On the 33-byte -20 dB set, bin-candidate recall changes as follows:

| Bin metric | Top-1 recall | Top-4 recall | Top-32 recall |
| --- | ---: | ---: | ---: |
| center FFT amplitude | 0.642 | 0.777 | 0.927 |
| multi-offset FFT evidence | 0.984 | 1.000 | 1.000 |

Current conclusion: the center FFT argmax assumption is not reliable at low SNR.
Simple absolute phase-line scoring and preamble-profile scoring did not beat
amplitude Top-L on the -20 dB set. The useful metric so far is multi-offset
noncoherent FFT evidence: on the 33-byte -20 dB set it raises true-bin recall
enough for the two-stage codec/CRC decoder to recover every tested packet.

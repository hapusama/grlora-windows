# Decoder-aware FrameSync gating under whole-packet AWGN

## Motivation

The noisy-FrameSync headroom experiment showed that a usable synchronization
estimate is often returned even when strict FrameSync validation rejects the
packet. This experiment audits each hard gate and tests a minimal soft path that
defers final acceptance to the payload decoder and CRC.

## Audit protocol

The same deterministic whole-packet AWGN protocol is replayed with additional
diagnostics:

- 8 clean 1 MS/s OTA captures;
- SF12, BW 125 kHz, OSR 8;
- Es/N0 from 12 through 16 dB;
- 2 AWGN seeds and 16 payload symbols per packet;
- 80 packet trials and 1280 Savaux symbol decisions.

Each selected synchronization candidate records the following gates separately:

1. preamble stability in the frame locator;
2. sync-word symbol 1 distance;
3. sync-word symbol 2 distance;
4. SFD consistency;
5. all coarse corrected preamble peaks in bin zero;
6. common-offset netID validity.

The audit also records coarse/fine preamble counts and all eight ADC sample-phase
branch validity results.

For evaluation only, a returned estimate is called packet-decodable when all 16
tested payload symbols are correct. Ground truth is never used by the proposed
gate itself.

## Root cause

There are 33 strict-invalid trials with a returned estimate:

- 15 are 16/16-symbol decodable;
- 18 are not fully decodable.

Every one of the 15 decodable false rejects fails the
`framesync_preamble_all_bin0` gate. Thirteen of those fifteen still pass netID,
and all fifteen pass the frame locator. The all-bin0 gate therefore has a direct
false-reject rate in this weak-packet sample.

| Gate failure | Decodable false rejects | Non-decodable rejects |
|---|---:|---:|
| all preamble peaks in bin0 | 15 | 13 |
| netID common offset | 2 | 8 |
| locator preamble stability | 0 | 3 |
| locator SFD consistency | 0 | 6 |
| locator sync1 | 0 | 1 |
| locator sync2 | 0 | 1 |

This also shows why simply accepting every returned estimate is too broad at low
SNR. The locator and netID checks still reject several clearly wrong coordinate
jumps.

## Minimal decoder-aware gate

The implemented soft policy is

```text
decoder_candidate_available = frame_location.valid and frame_sync.netid_valid
```

It deliberately ignores only the all-preamble-bin0 hard decision. The original
strict result remains unchanged and is still the default. Callers opt in with:

```python
sync_result.accepted("decoder_aware")
```

The intended receiver flow is:

```text
strict valid --------------------------> decode
strict invalid + decoder-aware valid --> decode --> FEC/CRC final acceptance
structurally invalid ------------------> reject or candidate search
```

CRC must remain mandatory on the soft path. A non-decodable accepted candidate
then costs computation and latency, but cannot be delivered as a valid packet.

## Results

| Es/N0 | strict SER | decoder-aware SER | strict PDR | decoder-aware PDR | oracle-accept PDR |
|---:|---:|---:|---:|---:|---:|
| 12 | 0.8984 | 0.6094 | 0.000 | 0.062 | 0.125 |
| 13 | 0.5742 | 0.3281 | 0.312 | 0.500 | 0.562 |
| 14 | 0.4375 | 0.1875 | 0.562 | 0.812 | 0.812 |
| 15 | 0.2500 | 0.0000 | 0.750 | 1.000 | 1.000 |
| 16 | 0.0625 | 0.0000 | 0.938 | 1.000 | 1.000 |

PDR uses 16/16 raw-symbol correctness as a proxy because FEC/CRC decoding is not
part of this isolated experiment. At 14--16 dB, the decoder-aware gate reaches
the measured oracle-acceptance PDR. At 13 dB it recovers three additional packets
and leaves one decodable netID-failing packet for a future candidate reranker.
At 12 dB, relaxed gating alone cannot solve the underlying decoder and candidate
quality failures.

The soft gate accepts 6 and 3 non-decodable packets at 12 and 13 dB respectively.
With CRC these are rejected after decoding; without CRC the soft path must not be
used as a final validity decision.

## Decision

The all-preamble-bin0 condition should no longer be an unconditional early-drop
gate for CRC-protected weak packets. Keep it as a confidence feature, preserve
the frame locator and netID structural gates, and let decoding plus CRC arbitrate
the soft candidates.

Top-K candidate generation is not needed to explain the 14--16 dB loss. It is a
separate follow-up for the remaining netID failures and coordinate jumps below
14 dB. Any such reranker should operate at packet level and compare candidates
along the joint CFO/STO ambiguity ridge.

## Reproduction

Generate the gate-annotated noisy FrameSync trials:

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
python weakPacket_decoding/weak_decoder/os_lora/experiments/analyze_noisy_framesync_headroom_ota.py `
  --max-packets 8 --symbols-per-packet 16 `
  --esn0-db 16,15,14,13,12 --seeds 20260821,20260822 --workers 4 `
  --output-dir weakPacket_decoding/data/experiments/framesync_gate_audit_ota_awgn_20260822
```

Regenerate the policy tables and plots without rerunning synchronization:

```powershell
python weakPacket_decoding/weak_decoder/os_lora/experiments/analyze_framesync_gate_audit.py `
  --input-dir weakPacket_decoding/data/experiments/framesync_gate_audit_ota_awgn_20260822
```


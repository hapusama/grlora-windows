# Decoder-aware FrameSync with full FEC/CRC

## Question

The preceding 16-symbol audit showed that weak-packet FrameSync can reject an
estimate whose payload is still decodable.  That proxy did not include the
explicit PHY header, the complete 56-symbol payload, Hamming FEC, dewhitening,
or PHY CRC.  This experiment therefore evaluates exact packet delivery.

## Method

For each of 8 clean SF12/BW125 kHz/CR4/8/LDRO OTA packets and 2 fixed noise
seeds, B-wide complex AWGN is added once to the complete 1 MS/s waveform before
running the real FrameSync.  The same noisy estimate is passed through:

```text
Savaux hard decisions
  -> explicit PHY header FEC/checksum
  -> header-derived payload length, CR, CRC flag, and LDRO
  -> complete payload FEC
  -> dewhitening
  -> PHY CRC
```

The two policies differ only in candidate admission:

- `strict`: require the original FrameSync hard validation;
- `decoder-aware`: require a valid locator and net-ID, but allow the
  `preamble_all_bin0` gate to fail and let PHY CRC arbitrate.

The oracle uses clean synchronization parameters on the same noisy IQ.  Ground
truth bytes are used only after decoding to score exact delivery and detect CRC
collisions.  Clean calibration reconstructed all 8/8 complete OTA frames
exactly and passed CRC.

## Results

| Es/N0 (dB) | strict CRC PDR | decoder-aware CRC PDR | oracle CRC PDR | gain |
|---:|---:|---:|---:|---:|
| 12 | 6.25% | 25.00% | 50.00% | +18.75 pp |
| 13 | 37.50% | 62.50% | 87.50% | +25.00 pp |
| 14 | 50.00% | 75.00% | 100.00% | +25.00 pp |
| 15 | 68.75% | 93.75% | 100.00% | +25.00 pp |
| 16 | 93.75% | 100.00% | 100.00% | +6.25 pp |

Across 80 trials, strict delivery was 41/80 and decoder-aware delivery was
57/80.  The relaxed policy admitted 18 additional decoder candidates:

- 16 reconstructed the expected 33-byte frame and passed CRC;
- 2 failed CRC;
- 0 passed CRC with an incorrect frame.

Every additional candidate was rejected by the original
`framesync_preamble_all_bin0` gate.  Thus the full packet experiment confirms
that downstream FEC/CRC can safely recover the same hard-gate false rejects;
it does not yet show that a Top-K synchronization search is necessary.

## Remaining headroom

The decoder-aware-to-oracle gap decomposes as follows:

| Es/N0 (dB) | oracle itself fails | locator/net-ID gate still rejects | admitted noisy estimate fails |
|---:|---:|---:|---:|
| 12 | 8 | 4 | 0 |
| 13 | 2 | 3 | 1 |
| 14 | 0 | 2 | 2 |
| 15 | 0 | 0 | 1 |
| 16 | 0 | 0 | 0 |

The 12--14 dB `locator/net-ID` column is the concrete target for Top-K
synchronization candidates.  The final column is where an ambiguity-ridge
CFO/STO candidate can help.  Ordinary small residual-CFO banks remain a lower
priority.

## Complexity audit

Strict gating would invoke the packet decoder 46 times; decoder-aware gating
invokes it 64 times, an increase of 18 calls.  The measured Python Savaux/FEC
candidate runtime is about 0.20 s per packet.  The additional modeled decode
work is 4.34 s over all 80 trials, or 54 ms per input trial.  These numbers are
offline Python timings and must not be presented as GNU Radio C++ latency.

## Artifacts

- Runner: `weak_decoder/os_lora/experiments/evaluate_decoder_aware_crc_pdr_ota.py`
- Reusable decoder: `weak_decoder/os_lora/system/decoder_aware_crc.py`
- Formal output: `data/experiments/decoder_aware_crc_pdr_ota_awgn_20260822/`
- Main plot: `decoder_aware_crc_pdr.png`
- Per-trial audit: `packet_trials.csv`
- Aggregate table: `summary.csv`


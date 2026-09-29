# Virtual-phase SFD synchronization: preliminary paired OTA result

## Scope

This is deliberately a synchronization-only change.  It does **not** use
payload values, FEC, or CRC to choose a timing/CFO hypothesis.

The existing receiver estimates integer CFO from the second LoRa SFD downchirp
after retaining one chip-rate ADC phase.  At OSR `R=8`, the other seven phases
are discarded for that decision.  The new `sfd_cfo_mode="virtual_phase"` keeps
all phases and treats them as deterministic virtual observations of the *same*
LoRa SFD, not as independent antenna branches.

For the known zero-symbol SFD downchirp `d_0[n]`, dechirping with the full-rate
reference `u_0[n]` produces a residual tone.  For LoRa-bin candidate `k`, the
`q`th polyphase branch has a deterministic relative phase:

```text
Y(k) = sum_q exp(-j 2*pi*k*q/(N*R))
                 sum_p x[pR+q] u_0[pR+q] exp(-j 2*pi*k*p/N)
```

This is exactly the full-rate matched filter evaluated only on valid LoRa-bin
candidates.  It is therefore a LoRa-SFD virtual-SIMO statistic; it is not a
multi-FFT vote, does not claim `R` independent noise observations, and does not
apply payload-symbol wrap logic to the SFD.

The selected SFD bin continues to use the original gr-lora mapping
`cfo_int = floor(signed_sfd_bin / 2)`, so only the evidence used to choose that
bin changes.  Existing STO/SFO refinement, netID checks, frame-location, and
payload Savaux demodulation remain unchanged.

## Paired protocol

- PHY: SF12, BW 125 kHz, 1 MS/s, OSR 8.
- Data: 8 clean OTA captures from `lora-rfsr-savaux`.
- Noise: channel-bandlimited complex AWGN, added once to the whole packet.
- Point tested: Es/N0 = 13 dB; seeds `20260821` and `20260822`.
- Pairing: each condition uses identical clean packet and AWGN realization.
- Evaluation: clean chip-mode synchronization supplies a post-hoc coordinate
  reference only.  It is not available to either noisy receiver.

## Result at 13 dB

| Metric | Chip-phase SFD | Virtual-phase SFD |
|---|---:|---:|
| Packet trials | 16 | 16 |
| Integer CFO matches clean coordinate | 12/16 | 13/16 |
| Coordinate-local (`|CFO|<=2` bins and `|payload STO|<=32` samples) | 12/16 | 15/16 |
| Strict synchronization acceptance | 7/16 | 7/16 |
| Decoder-aware acceptance | 11/16 | 11/16 |
| Savaux payload errors | 53/256 | 20/256 |
| Payload SER | 0.2070 | 0.0781 |
| Paired payload rescues / regressions | -- | 33 / 0 |

The gains are sparse but physically meaningful rather than uniformly small.
For example, at 13 dB/seed `20260821`, packet
`exp0_000003_rxg20_0_fulltrim` has a single-phase SFD CFO slip from clean
`-129` to `-1001`, which moves the payload boundary by `-6977` samples.  The
virtual-phase SFD statistic selects `-129` and restores the payload coordinate.
At seed `20260822`, packet `exp0_000000_rxg20_0_fulltrim` similarly moves from
chip `+652` / `+6247` samples to virtual `-129` / `0` samples.

## What this does and does not establish

It establishes that the discarded OSR phases contain usable LoRa-SFD evidence
which can prevent some catastrophic CFO--timing ambiguity-ridge jumps.  It does
not yet improve strict PDR: rescued packets can still fail the preamble-bin0 or
netID gate, even when their returned payload coordinate demodulates correctly.

The coherent next step is **not** a decoder/CRC rescue.  It is to extend the
same full-rate known-chirp likelihood to the two sync-word chirps and use the
result as a virtual-phase structural gate.  That would keep the entire method
inside known LoRa preamble/sync/SFD physics and is the missing step before
claiming an end-to-end weak-packet receiver improvement.

## Reproduction

```powershell
$env:PYTHONPATH = (Resolve-Path weakPacket_decoding)
python weakPacket_decoding/weak_decoder/os_lora/experiments/evaluate_virtual_phase_sfd_sync_ota.py `
  --max-packets 8 --symbols-per-packet 16 --esn0-db 13 --seeds 20260821 `
  --output-dir weakPacket_decoding/data/experiments/virtual_phase_sfd_sync_ota_awgn_20260826_8pkt_13db

python weakPacket_decoding/weak_decoder/os_lora/experiments/evaluate_virtual_phase_sfd_sync_ota.py `
  --max-packets 8 --symbols-per-packet 16 --esn0-db 13 --seeds 20260822 `
  --output-dir weakPacket_decoding/data/experiments/virtual_phase_sfd_sync_ota_awgn_20260826_8pkt_13db_seed2
```

# Virtual-SIMO LoRa preamble acquisition at -30 dB

## Why the earlier result looked too weak

The noisy FrameSync experiment labels its x-axis as `Es/N0`, not channel-power
SNR.  For SF12,

```text
channel SNR = Es/N0 - 10*log10(4096) = Es/N0 - 36.124 dB.
```

Therefore the earlier 13 dB point corresponds to approximately -23.1 dB
channel SNR.  More importantly, that experiment used the legacy acquisition
path: four separate chirp FFT powers were added noncoherently, and later
integer-CFO selection relied heavily on one SFD symbol.

XCopy's reported -30 dB packet decoding is not a single-copy result.  It first
detects long preambles and then calibrates and coherently combines multiple real
retransmissions; its paper reports that the -30 dB case can require more than 15
transmissions.  This is not an excuse for weak single-packet acquisition, but it
is the correct comparison boundary.

## Virtual-SIMO acquisition statistic

For a window containing `M` repeated LoRa preamble chirps, dechirping converts
the signal into a continuous tone.  Split the OSR=`R` samples into polyphase
branches.  For long-FFT candidate `k`, branch `q` has the known steering phase

```text
a_q(k) = exp(-j*2*pi*k*q/(M*N*R)).
```

The receiver computes a branch DFT, applies `a_q(k)`, and sums across branches.
The result is exactly the full-rate coherent matched filter.  It uses the SIMO
property that every branch observes the same source through a deterministic
LoRa steering manifold.  It does not treat the correlated branches as
independent antennas and does not claim `10*log10(R)` diversity gain.

## Paired OTA+AWGN protocol

- Eight SF12, BW125 kHz, 1 MS/s OTA packets;
- OSR 8 and a 16-chirp preamble;
- two deterministic whole-packet AWGN realizations;
- channel SNR -30 dB, equivalent to `Es/N0 = 6.124 dB`;
- the clean boundary is used only after detection to label whether an event is
  in the real preamble basin;
- all detectors scan the same first 32 chirp durations of the noisy packet.

Four detectors separate window length, temporal coherence, and virtual-SIMO
combining:

| Detector | True preamble basin | False events |
|---|---:|---:|
| legacy: noncoherent, 4 chirps | 0/16 | 0 |
| noncoherent, 12 chirps | 12/16 | 6 |
| coherent, 12 chirps, one ADC phase | 10/16 | 0 |
| coherent, 12 chirps, all virtual-SIMO phases | **15/16** | **1** |

The all-phase result improves over the equally coherent single-phase detector
from 10/16 to 15/16.  Thus the measured difference is not only a longer-window
effect.  The noncoherent 12-chirp result also shows why simply accumulating more
power is insufficient: it misses more packets and creates six times as many
false events as the virtual-SIMO statistic in this small paired sample.

## Current boundary

This result establishes single-packet **acquisition** at -30 dB, not completed
strict synchronization.  The legacy absolute-boundary locator still evaluates
individual preamble, sync-word, and SFD argmax decisions.  At -30 dB it can
receive the correct acquisition basin and nevertheless select the wrong narrow
CFO--STO ridge coordinate.

A post-hoc diagnostic at the real boundary shows that the joint waveform formed
from 16 preamble chirps, two known sync-word chirps, and two SFD downchirps still
has a distinct matched-filter peak at -30 dB.  The implementation follow-up is
therefore a one-dimensional LoRa-manifold search inside the acquired basin:

```text
virtual-SIMO preamble basin
    -> joint 20-known-chirp CFO/STO/boundary likelihood
    -> virtual-SIMO sync-word/SFD structural gate
    -> Savaux payload demodulation
```

CRC, FEC, and payload truth must not participate in that search.

## Reproduction

```powershell
$env:PYTHONPATH = (Resolve-Path weakPacket_decoding)
python weakPacket_decoding/weak_decoder/os_lora/experiments/evaluate_virtual_simo_preamble_acquisition_ota.py `
  --channel-snr-db=-30 --max-packets 8 --seeds 20260821,20260822
```

Results are written to
`weakPacket_decoding/data/experiments/virtual_simo_preamble_acquisition_ota_awgn_20260827`.

# Offset-Coherence Ablation Summary

Scope: `gr-lora_sdr/weakPacket_decoding copy`

Formal full sweeps use `--snr-start -12 --snr-stop -26 --snr-step -1` on the three validation datasets.
Probe rows use only `-20..-23 dB`, so they should be compared by curve values rather than full threshold gains.

## Main Full-Sweep Ablation

| ID | Variant | Multi | Top-L | Lock | Coherence | Line | SER Gain | CRC90 Gain | SER Gain vs Multi | SER@-22 | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | center argmax | no | 1 | no | no | no | 0.00 | 0.00 |  | 0.646 |  |
| A1 | multi-offset argmax | yes | 1 | no | no | no | 3.38 | 2.65 | 0.00 | 0.204 |  |
| A2 | energy-only selected Top-24 | yes | 24 | yes | no | no | 3.38 | 2.65 | 0.00 | 0.204 | 0.924 |
| A3 | offset coherence only Top-24 | yes | 24 | yes | yes | no | 3.20 |  | -0.18 | 0.203 | 0.924 |
| A4 | energy + offset coherence, no line | yes | 24 | yes | yes | no | 4.69 | 4.14 | 1.31 | 0.103 | 0.924 |
| A5 | current default | yes | 24 | yes | yes | small | 4.74 | 4.14 | 1.36 | 0.101 | 0.924 |
| A6 | packet-line phase only Top-24 | yes | 24 | yes | no | yes |  |  |  | 0.576 | 0.924 |
| A8 | no high-confidence lock | yes | 24 | no | yes | small | 4.65 | 4.14 | 1.27 | 0.104 | 0.940 |

## Top-L Full-Sweep Ablation

| ID | Top-L | SER Gain | CRC90 Gain | SER Gain vs Multi | SER@-20 | SER@-21 | SER@-22 | SER@-23 | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A7-L8 | 8 | 4.60 | 4.03 | 1.22 | 0.046 | 0.063 | 0.108 | 0.212 | 0.885 |
| A7-L16 | 16 | 4.73 | 4.14 | 1.34 | 0.044 | 0.063 | 0.102 | 0.188 | 0.911 |
| A7-L24 | 24 | 4.74 | 4.14 | 1.36 | 0.044 | 0.060 | 0.101 | 0.184 | 0.924 |
| A7-L32 | 32 | 4.71 | 4.14 | 1.32 | 0.044 | 0.060 | 0.103 | 0.182 | 0.933 |

## Quick Probes (-20..-23 dB)

| ID | Variant | Mean SER | Mean CRC/PRR | SER Delta vs Default | CRC Delta vs Default |
| --- | --- | --- | --- | --- | --- |
| A5 | current default | 0.097 | 0.509 | 0.000 | 0.000 |
| A9-C32 | coherence candidate Top-32 | 0.099 | 0.493 | 0.002 | -0.016 |
| A9-C64 | coherence candidate Top-64 | 0.099 | 0.493 | 0.002 | -0.016 |
| A9-C128 | coherence candidate Top-128 | 0.099 | 0.493 | 0.002 | -0.016 |
| A10 | smooth trajectory beam probe | 0.123 | 0.369 | 0.026 | -0.140 |

## Interpretation

- Energy-only selected is identical to multi-offset argmax, so the selected-path machinery alone does not create extra gain.
- Offset coherence adds a clear threshold gain over multi-offset energy; amplitude protection is important because coherence-only is weaker.
- Packet-line phase alone is not competitive. The current default uses it only as a small auxiliary term.
- Top-24 is the best full-sweep low-complexity setting in this matrix; Top-32 increases recall but does not improve the formal SER/CRC thresholds.
- Disabling high-confidence locks degrades the threshold, supporting the claim that locks protect already-reliable symbols from over-reranking.
- Coherence-candidate expansion and smooth beam probes do not justify replacing the default on the -20..-23 dB probe range.

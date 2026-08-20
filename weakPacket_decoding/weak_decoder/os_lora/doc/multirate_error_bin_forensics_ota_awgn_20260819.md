# Multi-rate error-bin forensics on OTA LoRa symbols

## Question

When a 250 kS/s LoRa dechirp FFT selects the wrong bin under AWGN, can the
true candidate be distinguished by the legal fold-pair amplitude ratio at
1 MS/s, 500 kS/s, and 250 kS/s?

This experiment does not claim that deterministic rate changes create new
information and does not train or tune a decoder. It only compares the true
and selected-wrong candidates on the same error events.

## Protocol

The input is the curated SF12/BW125 kHz OTA dataset in the sibling
`lora-rfsr-savaux` repository. For every canonical ADC-phase-zero packet:

1. run FrameSync on the clean 1 MS/s IQ once;
2. freeze the fine symbol boundary, CFO, and SFO correction schedule;
3. retain payload symbols whose clean q=8, q=4, and q=2 decisions all match
   the reference symbol;
4. add one B-wide complex AWGN realization to the 1 MS/s symbol; and
5. obtain q=4 and q=2 by nested phase-zero decimation of that same noisy IQ.

Noise power uses

\[
E_s/N_0 = N P_s/P_n, \qquad N=2^{SF},
\]

where the signal power is the active-symbol power after subtracting the
packet's leading off-packet power. The injected noise is flat across the
125 kHz LoRa channel and zero outside it. Thus it is ordinary channel AWGN;
its 1 MS/s samples are correlated only because the common receiver bandwidth
is narrower than the 1 MHz sampling Nyquist band.

For the project's fine-boundary convention, symbol `S` occupies fold-pair FFT
coordinate `u=(S-1) mod N`. For each q, the two amplitudes are read at `u` and
`u-N mod qN`, and

\[
R_q(S)=\frac{|Y_q[u-N]|}{|Y_q[u]|+|Y_q[u-N]|}.
\]

The explored scores are

\[
C_{1M}(S)=-(R_8(S)-S/N)^2
\]

and

\[
C_{multi}(S)=-\sum_{q\in\{8,4,2\}}(R_q(S)-S/N)^2
             -\operatorname{Var}_q R_q(S).
\]

## Result

Eight physical packets supplied 256 payload symbols. All packets synchronized;
249 symbols passed the clean all-rate admission rule. Three common-random-number
seeds produced 747 trials per SNR point.

| Es/N0 (dB) | q=2 SER | Error events | True in Top-8 | C1M true wins | Cmulti true wins |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.8233 | 615 | 0.2748 | 0.3203 | 0.3203 |
| 11 | 0.7229 | 540 | 0.3759 | 0.3278 | 0.3315 |
| 12 | 0.5944 | 444 | 0.5068 | 0.3221 | 0.3288 |
| 13 | 0.4632 | 346 | 0.6879 | 0.3353 | 0.3324 |
| 14 | 0.3333 | 249 | 0.8032 | 0.3414 | 0.3414 |
| 15 | 0.2463 | 184 | 0.9511 | 0.3261 | 0.3043 |

The promising part is Top-K recoverability: at 14 and 15 dB, the true symbol
remains in q=2 Top-8 for 80.3% and 95.1% of q=2 errors. The proposed fold-ratio
test, however, is anti-discriminative on these events. A random choice between
the paired true and wrong candidates would win 50%; both consistency scores
remain near 30--34%. Cross-rate ratio variance alone is also near chance
(51--56% of events have lower variance for the true candidate).

The 1M-only and multi-rate scores have nearly the same paired ordering. This is
consistent with all rate views being deterministic reductions of one
band-limited waveform: rate conversion exposes the same fluctuation but does
not supply an independent check. It also shows a selection effect that matters
for future designs: conditioning on the q=2 argmax selects wrong bins whose
two observed components already look unusually self-consistent. A legal
geometry check is therefore not automatically evidence for the transmitted
symbol.

The redundancy is directly visible in the raw features. At 14 dB, the Pearson
correlations among `R8`, `R4`, and `R2`, measured over both candidates of every
error pair, are 0.9993--0.9998. The multi-rate score changes the 1M-only paired
ordering in only 12 of 249 errors: it rescues six events and harms six, for zero
net gain. The 249 admitted clean symbols show the same greater-than-0.999
cross-rate ratio correlation, so this is not created by conditioning only on
errors.

## Reproduction

Run from `gr-lora_sdr`:

```powershell
python weakPacket_decoding\weak_decoder\os_lora\experiments\archive\analyze_multirate_error_bins_ota.py `
  --max-packets 8 --symbols-per-packet 32 `
  --esn0-db 10,11,12,13,14,15 `
  --seeds 20260819,20260820,20260821
```

The output directory contains the clean sync audit, every noisy trial, paired
true/wrong feature rows, summary tables, and diagnostic plots. The raw event
tables should be used for any later feature proposal so that scoring changes
can be evaluated on exactly the same error events.

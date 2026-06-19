# weak_decoder Historical Version Map

This map makes the research history navigable without deleting old work or
breaking existing scripts. Original modules stay at the top level. The `v1/`,
`v2/`, and `v3/` packages provide stable labels and import aliases.

## v1 - Legacy Phase/Codec Beam Branch

**Research question**

Can low-SNR payload recovery be improved by projecting candidates through LoRa
codec structure, payload-byte enumeration, CRC/beam search, or phase-guided
candidate scoring?

**Main modules**

```text
phase_guided_demod.py
two_stage_weak_decoder.py
blind_payload_search.py
blind_payload_decoder.py
```

**Typical runners**

```text
scripts/run_blind_payload_search.py
scripts/experiments/run_blind_decoder_evaluation.py
scripts/experiments/run_two_stage_weak_decoder.py
scripts/experiments/evaluate_codec_bin_metric.py
```

**How to treat it now**

Keep as historical diagnostic and ablation material. It is not the clean paper
mainline because it can look like codec/CRC-assisted post-processing rather
than a physical-layer demodulation contribution.

## v2 - Symbol-Level Two-Stage Phase Selector

**Research question**

Can the decoder stay at symbol/FFT-bin level by locking high-confidence payload
symbols, keeping Top-L candidates for weak symbols, and using packet-local phase
consistency to rerank only ambiguous symbols?

**Main modules**

```text
candidate_pruning.py
symbol_phase_two_stage.py
phase_guided_demod.py
```

**Typical runners**

```text
scripts/experiments/evaluate_candidate_pruning_metric.py
scripts/experiments/run_candidate_pruning_sweep.py
scripts/experiments/run_symbol_phase_two_stage.py
scripts/experiments/diagnose_symbol_phase_models.py
```

**How to treat it now**

Keep as the first clean PHY-only branch. This version established the right
abstraction: select FFT bins directly, do not enumerate payload bytes, and keep
CRC as a final metric only.

## v3 - Current Phase-Assisted PHY Selector

**Research question**

Given Top-L candidates per payload symbol, can we jointly select a candidate
sequence under a packet-local smooth circular phase trajectory prior, while
using energy and multi-offset coherence as the primary evidence?

**Main modules**

```text
symbol_phase_two_stage.py
candidate_pruning.py
phase_guided_demod.py
```

**Typical runners**

```text
scripts/experiments/run_symbol_phase_threshold_sweep.py
scripts/experiments/make_offset_coherence_ablation_table.py
scripts/experiments/analyze_phase_opportunity_space.py
scripts/experiments/plot_gt_phase_sweep.py
scripts/experiments/plot_payload_phase_traces_by_snr.py
```

**Default selector framing**

```text
header-first timing/header
  -> payload multi-offset FFT evidence
  -> high-confidence Top-1 locking
  -> low-confidence Top-L candidates
  -> energy + offset coherence + small packet-local phase residual rerank
  -> hard symbol bins
  -> normal LoRa PHY decode / CRC metric
```

**Important boundary**

Do not force preamble/sync/SFD and payload onto one absolute phase line. Use
preamble/sync/SFD for timing, CFO/SFO, and sync quality. Use header/payload,
especially payload data-section symbols, for the packet-local phase trajectory.

## Shared Core Modules

```text
chirp.py
preamble_detector.py
frame_locator.py
grlora_frame_sync.py
header_first_demod.py
payload_codec.py
```

These modules are dependencies for several versions, so they remain unversioned
shared infrastructure.

## Baseline / Diagnostic Modules

```text
baselines/savaux_oversampled/
structured_path_demod.py
adaptive_path_demod.py
timing_path_demod.py
```

Use these as baselines, negative results, or future-work probes. They should
not silently become part of the v3 phase-assisted payload trajectory claim.

## Useful Notes

```text
notes/handoffs/HANDOFF_SYMBOL_LEVEL_PHASE_TWO_STAGE_2026-06-16.md
notes/handoffs/HANDOFF_OFFSET_COHERENCE_ABLATION_2026-06-17.md
notes/handoffs/HANDOFF_TWO_STAGE_RETREAT_TO_PHY_OS_ENHANCEMENT_2026-06-18.md
notes/handoffs/HANDOFF_PHASE_ASSISTED_WEAK_DEMOD_2026-06-19.md
notes/design/LOCAL_WINDOW_PHASE_SELECTOR_2026-06-18.md
notes/baselines/PAPER_OVERSAMPLED_DEMOD_BASELINE_2026-06-17.md
```

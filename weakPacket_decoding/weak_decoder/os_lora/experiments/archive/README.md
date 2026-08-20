# Archived experiments

These entry points are intentionally outside the active experiment directory.
They document investigated directions that are not part of the current weak
packet receiver:

- `probe_rfsr_savaux.py`: RF super-resolution integration probe;
- `evaluate_alias_trim.py`: structured blocker/alias trimming;
- `evaluate_robust_sparse_demod.py`: structured-interference sparse gating;
- `evaluate_multirate_structure_awgn.py`: deterministic multi-rate AWGN views;
- `analyze_multirate_error_bins_ota.py`: multi-rate error-bin forensics.

Their algorithms, reports, and result artifacts remain available because the
negative results are useful baselines.  New primary experiments should not
import these modules.  Run an archived entry explicitly, for example:

```powershell
python -m weak_decoder.os_lora.experiments.archive.evaluate_alias_trim --help
```

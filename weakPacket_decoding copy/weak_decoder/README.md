# weak_decoder 版本整理说明

这个目录保留原始实现文件的位置不变，确保现有实验脚本继续可用。
`v1/`、`v2/`、`v3/` 只是叠加在历史代码上的项目管理层，不是破坏性重构。

## 共享核心模块

下面这些文件是多个版本共用的基础设施，不单独归入某一个 decoder 版本：

```text
chirp.py
preamble_detector.py
frame_locator.py
grlora_frame_sync.py
header_first_demod.py
payload_codec.py
```

它们负责 IQ 读取、chirp/FFT 基础操作、弱前导码检测、帧定界、
gr-lora 风格 framesync、header-first demod，以及离线 LoRa PHY codec 检查。

## 版本总览

| 版本 | 核心思路 | 当前定位 | 主要文件 |
| --- | --- | --- | --- |
| `v1` | 早期 phase-guided / codec-beam 弱包解码 | 历史诊断分支 | `phase_guided_demod.py`, `two_stage_weak_decoder.py`, `blind_payload_search.py`, `blind_payload_decoder.py` |
| `v2` | symbol-level two-stage Top-L selector | 历史 PHY-only baseline | `candidate_pruning.py`, `symbol_phase_two_stage.py` |
| `v3` | 当前相位辅助 PHY selector：energy + offset coherence + packet-local phase | 当前研究主线 | `symbol_phase_two_stage.py`, `candidate_pruning.py`, `phase_guided_demod.py` |

当前最值得继续推进的是 `v3`：

```text
每个 payload symbol 保留 Top-L FFT bin candidates；
边选择每个 symbol 的 candidate，边估计或利用未知的 packet-local smooth phase trajectory；
energy 和 multi-offset coherence 是主证据；
packet-local phase residual 是辅助 rerank 项；
CRC 只做最终验证，不参与搜索。
```

## Baseline 和诊断模块

下面这些文件/目录暂时不归入 `v1`-`v3`：

```text
baselines/savaux_oversampled/
structured_path_demod.py
adaptive_path_demod.py
timing_path_demod.py
```

它们适合作为 baseline、负结果、消融实验或 future work 诊断材料，
但不要默认混进当前 `v3` 的 phase-assisted payload trajectory 主线。

更细的版本映射和 handoff 链接见 `VERSION_MAP.md`。

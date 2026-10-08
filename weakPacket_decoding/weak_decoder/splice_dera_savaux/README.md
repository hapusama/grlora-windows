# splice_dera_savaux — DeRa 前端 × Savaux 相干解码拼接研究（2026-10-04 起）

**研究问题**：DeRa 的检测/同步前端 + Savaux 的过采样相干解调内核拼接后，
能否在真实 OTA 弱信号下激发 Savaux 的完整解码潜力？拼接暴露的失效机制
是什么、如何修复、修复后离天花板还差多少？

**地位**：独立研究文件夹，与 `weak_decoder/` 下其他模块（baselines/
decoding/synchronization/os_lora/ 等）的开发**互不干扰**——本文件夹只
新增文件，不改任何既有代码；实现在 `code/`，实验产物（checkpoint/日志/
正式 RESULTS.md）仍在 `data/experiments/dera_savaux_splice_20261004/`。

## 文件地图

| 文件 | 内容 |
|---|---|
| `METHOD.md` | 方法：七链矩阵、亚样本定时混跳机制、fd 修复、协议遵从 |
| `EXPERIMENTS.md` | 实验记录：战表、冒烟/负控、归因链、δ 记账勘误 |
| `NOTES.md` | 思考：瓶颈分层、Savaux 论文精读、与主方案的关系、下一步 |
| `code/splice_runner.py` | 主实验 runner（7 链 × 7 档 × 3 种子，断点续跑，`--smoke` 门禁） |
| `code/debug_f00.py` | 逐帧归因：CFO/时延细扫定位失效残差 |
| `code/debug_delay.py` | 机制确认：时延响应双平台、对抗测试 |
| `code/debug_gap.py` | A/B 归因：δ 记账 vs SNR 窗 vs 种子（.511/.421 之谜） |

## 一句话结论（截至 2026-10-05）

原样拼接**不能**激发（中段 PER 差全先验参照 2~3×）；失效点 = DeRa 前端
缺 −sto_frac·OS 亚样本定时补偿 → Savaux 合并谱峰落 bin 边界 → argmax
逐符号混跳（非 δ 可修）；一行可部署 fd 修复（前导能量 × 17 点时延网格）
后 −20/−22 PER 追平全 DeRa（.024/.107），−24 打平（差 3/84 包），−26 与
深端仍小输；"Savaux −28dB" 不成立（本协议口径下限 ≈−26）。

## 运行

```bash
# conda 环境
PY=/d/mysoft2/miniconda3/envs/gr-lora/python.exe
$PY code/splice_runner.py --smoke   # native 门禁 + 负控（串行 ~15min）
$PY code/splice_runner.py           # 全量 532 单元（6 workers，~70min）
```

checkpoint 在 `data/experiments/dera_savaux_splice_20261004/checkpoint.jsonl`
（断点续跑自动跳过已完成单元；汇总键用 `"%+d" % level`，负号必须带——
protocol §6 警告过的坑）。

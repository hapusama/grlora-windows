# Phase-Line 主导弱包解调开发指南

本目录用于承接新的研究主线：**以 payload data-section 内的 packet-local phase line 为第二轮主判据**，而不是继续把 offset coherence 当成主要增益来源。

当前判断：

```text
multi-offset / oversampling evidence:
  只负责第一轮 Top-L 候选召回，保证正确 bin 尽量留在候选集里。

phase-line selector:
  在第二轮中主导候选序列选择，利用 payload 内正确 bin 的平滑相位轨迹。

CRC / codec:
  只做最终验证和评估，不参与候选搜索。
```

## 1. 研究边界

### 1.1 要解决的问题

每个 payload symbol 有一个候选集合：

```text
C_k = {candidate bin b_1, b_2, ..., b_L}
```

每个 candidate 至少包含：

```text
symbol_index
raw_bin
complex_fft_value
phase = angle(complex_fft_value)
energy_score
rank_in_first_stage
optional: multi_offset_coherence
```

目标是在不知道 ground-truth phase trajectory 的情况下，联合完成：

```text
1. 每个 symbol 选择一个 candidate bin
2. 估计 packet-local smooth circular phase trajectory
```

抽象问题：

```text
Top-L candidate sequence selection under an unknown smooth circular phase trajectory prior.
```

### 1.2 不做什么

当前阶段不要把以下内容放进第二轮主判据：

```text
CRC-valid candidate search
LoRa codec / byte enumeration
payload template / counter prior
cross-packet prior
retransmission combining
offset coherence as the dominant rerank score
preamble/sync/SFD 到 payload 的全包绝对相位线
```

multi-offset coherence 可以保留为：

```text
候选召回质量指标
能量相近时的弱 tie-break
防止 phase-only 跑向极弱噪声峰的 guardrail
```

但论文叙事和算法默认值必须避免“第二轮其实还是 coherence 在赢”。

## 2. 与现有方法的关系

现有 `symbol_phase_two_stage.py` 的默认 selector 更像：

```text
energy + offset coherence + small packet-line phase
```

新方向要改成：

```text
stage 1:
  multi-offset / energy 产生 Top-L candidates

stage 2:
  phase-line trajectory 是主优化目标
  energy 只作为候选可信度下限
  offset coherence 只作为候选召回或 tie-break
```

因此后续实验命名建议明确区分：

```text
coherence_selected     旧/current selector
phase_line_selected    新主线 selector
phase_oracle           用 GT phase line 做上界分析
phase_only_guarded     phase 主导 + energy/coherence guard
```

## 3. 推荐模块拆分

建议在本目录下逐步新增下面这些模块，而不是继续把逻辑塞进一个大文件：

```text
evidence.py
  从现有 multi-offset FFT 输出中组织 Top-L candidate sets。
  注意：这里只做候选召回，不在这里做最终判决。

trajectory.py
  circular phase unwrap、line/polynomial/local-linear fit、RANSAC/robust fit。

selector.py
  phase-line 主导的候选序列选择器。
  可以包含 beam search / Viterbi / EM-like alternating selection。

metrics.py
  phase repair/damage、oracle gap、line RMSE/R2、SER、CRC final metric。

diagnostics.py
  输出每包候选相位图、选中路径图、wrong-bin phase residual 对照。

configs.py
  所有 phase-line selector 参数集中管理。
```

实验脚本建议放在：

```text
scripts/experiments/phase_line/
```

数据输出建议放在：

```text
data/phase_line_selector/
data/phase_line_oracle/
data/phase_line_diagnostics/
```

## 4. 算法路线

### 4.1 P0：固定候选接口

第一步不要急着优化 selector，先固定 candidate 数据结构。

每个 packet 输出一张 candidate 表：

```text
packet_index
symbol_index
candidate_rank
raw_bin
energy_score
energy_drop_db
coherence_score
phase_rad
complex_real
complex_imag
is_gt_bin              # 仅评估用，真实解码不可用
```

候选集生成规则：

```text
Top-L 主要按 multi-offset fused energy 召回。
L 建议先取 24 或 32。
coherence 不作为候选排序主因，避免候选集本身已经被 coherence bias。
```

需要先统计：

```text
GT recall@L
GT candidate rank distribution
energy-only candidate recall
coherence-biased candidate recall
```

如果 GT 经常不在 Top-L 里，phase-line 第二轮再强也救不了。

### 4.2 P1：Phase Oracle 上界

用 clean/GT bin 拟合 payload phase line，只用于评估上界：

```text
theta_k = a * k + c
```

然后在 noisy Top-L candidates 中选 phase residual 最小的 bin：

```text
score_phase_oracle(k, b) = cos(wrap(phi_k,b - theta_k))
```

这一阶段回答：

```text
如果 phase line 已知，Top-L 里能救回多少？
phase 会修复哪些 symbol？
phase 会误伤哪些 symbol？
```

关键指标：

```text
oracle_selected_SER
phase_repair_rate
phase_damage_rate
unrecoverable_rate
oracle_gap_to_current
```

如果 oracle 都没有显著空间，就不要继续复杂化。

### 4.3 P2：Anchor 初始化 + Robust Phase Line

实际解码时 GT phase line 不知道，需要从候选中估计。

可用 anchors：

```text
高能量 Top-1 symbols
energy margin 很大的 symbols
phase residual 自洽的一小段 local window
header/payload data-section 内稳定点
```

不要直接用：

```text
preamble/sync/SFD 的绝对相位
CRC-valid payload candidate
跨包 payload prior
```

robust fit 建议：

```text
1. 从高置信 Top-1 anchors 拟合初始 line
2. 对所有 candidate 计算 circular residual
3. 每个 symbol 暂选 residual 最小且 energy 不太差的 candidate
4. 重新拟合 line
5. 迭代 2-4 步，使用 Huber/Tukey/RANSAC 降低 outlier 权重
```

需要输出 line 质量：

```text
anchor_count
line_rmse_pi
line_r2
inlier_ratio
phase_slope
phase_intercept
```

### 4.4 P3：Phase-Line 主导候选序列搜索

推荐先做两个 baseline。

#### A. Alternating Fit-Select

```text
init line from anchors
repeat:
  select one candidate per symbol by phase residual + energy guard
  refit line from selected candidates
until convergence
```

候选局部分数：

```text
phase_score = cos(wrap(phi_k,b - theta_k))

score(k,b)
  = w_phase * phase_score
  + w_energy * bounded_energy_score
  + w_rank   * bounded_rank_score
  + w_coh    * weak_tiebreak_coherence
```

推荐权重方向：

```text
w_phase >= 0.70
w_energy <= 0.25
w_coh <= 0.05
```

`w_coh` 必须很小，或者只在 phase_score 接近时作为 tie-break。

#### B. Beam / Viterbi Smooth Path

把状态写成：

```text
state_t = selected candidate at t + local phase slope estimate
```

路径代价：

```text
cost =
  - phase_likelihood
  + small energy_guard_penalty
  + slope_smoothness_penalty
  + curvature_penalty
```

一阶/二阶平滑项：

```text
first_order:
  wrap((phi_t - phi_{t-1}) - slope) should be small

second_order:
  wrap(phi_t - 2*phi_{t-1} + phi_{t-2}) should be small
```

Beam search 适合作为第一个工程实现，因为 payload symbol 数不长，Top-L 也有限。

### 4.5 P4：局部窗口 phase-line

如果全包 line 不稳，可以改成局部窗口：

```text
for symbol k:
  use nearby reliable symbols in [k-W, k+W]
  fit local phase line
  rerank candidates of k
```

这条线适合处理：

```text
phase slope 漂移
局部 timing/SFO residual
部分 symbol 被噪声污染
```

但要注意不要把它退化成“局部 coherence selector”。窗口内仍然应以 payload phase trajectory 为主。

## 5. Phase Wrapping 处理

不要直接对 wrapped phase 做普通线性回归。

必须统一使用：

```text
wrap(x) = angle(exp(1j*x))  -> [-pi, pi]
```

两个可选策略：

```text
1. 先根据候选路径做 unwrap，再拟合 line。
2. 直接最小化 circular residual:
   residual = wrap(phi_k,b - theta_k)
   loss = 1 - cos(residual)
```

低 SNR 下推荐优先用 circular residual，因为错误 unwrap 很容易把路径带偏。

## 6. Guardrails

phase-line 主导不等于完全相信 phase。

必须保留最小物理 guard：

```text
candidate energy 不能比本 symbol Top-1 低太多
candidate rank 不能太靠后
line anchors 不足时回退到 multi-offset Top-1
line_rmse 太大时降低 phase 权重
phase-only 选择的 candidate 必须记录 energy_drop_db
```

建议默认：

```text
max_energy_drop_db = 12 ~ 20
min_anchor_count = 4 ~ 8
max_line_rmse_pi = 0.25 ~ 0.40
top_l = 24 or 32
```

这些 guard 是为了防止 phase 在纯噪声里找到一条“很平滑但错误”的轨迹。

## 7. 评估指标

主指标：

```text
center_SER
multi_offset_SER
coherence_selected_SER
phase_line_selected_SER
phase_oracle_SER
CRC_valid_rate          # final metric only
```

phase-line 专属指标：

```text
phase_repair_count
phase_damage_count
phase_repair_rate
phase_damage_rate
oracle_gap
line_rmse_pi
line_r2
selected_mean_energy_drop_db
selected_mean_phase_residual
```

关键对照：

```text
phase_line_selected vs multi_offset_argmax
phase_line_selected vs coherence_selected
phase_line_selected vs phase_oracle
```

如果 phase-line 主线成立，应该看到：

```text
在低 SNR 区间，phase_line_selected 能比 multi_offset_argmax 修复更多 symbol；
与 coherence_selected 相比，收益来源不是 offset coherence；
phase_oracle 和实际 phase_line_selected 之间的 gap 可以逐步缩小。
```

## 8. 推荐实验顺序

```text
1. 导出统一 Top-L candidate 表。
2. 做 phase oracle 上界实验。
3. 做 anchor quality 诊断：哪些 symbol 能作为 line anchor。
4. 实现 alternating fit-select baseline。
5. 实现 beam / Viterbi smooth path baseline。
6. 做 ablation：phase-only、phase+energy guard、phase+tiny coherence tie-break。
7. 和旧 coherence_selected 做同 SNR、同 packet、同候选集对照。
```

## 9. 论文叙事建议

这条线的叙事不要写成：

```text
multi-offset coherence improves weak demodulation
```

而应写成：

```text
Multi-offset FFT is used only to preserve a compact candidate set under low SNR.
The final demodulation decision is made by exploiting a packet-local phase
trajectory of the correct payload FFT bins.
```

中文表达：

```text
第一轮只负责把正确峰尽量留在候选集里；
第二轮利用正确 payload bin 在包内呈现的平滑相位轨迹，从候选集中选出物理上自洽的一条路径。
```

这和 offset coherence baseline 的边界要写清楚，否则容易撞车。

## 10. 当前开发优先级

最小可行版本：

```text
phase_line/evidence.py
phase_line/trajectory.py
phase_line/selector.py
scripts/experiments/phase_line/run_phase_line_oracle.py
scripts/experiments/phase_line/run_phase_line_selector.py
```

第一批结果只需要回答：

```text
1. phase oracle 是否有足够上界空间？
2. 不使用 coherence 主判据时，phase-line selector 能否在 -22 ~ -27 dB 修复 symbol？
3. phase-line selector 相对 coherence_selected 的优势/劣势在哪里？
```

先把问题讲干净，再追求性能数字。

## 11. 当前实现状态（2026-06-19）

已经新增第一版可运行原型：

```text
weak_decoder/phase_line/configs.py
weak_decoder/phase_line/trajectory.py
weak_decoder/phase_line/selector.py
weak_decoder/phase_line/__init__.py
scripts/experiments/phase_line/run_phase_line_threshold_sweep.py
```

当前 selector 采用两阶段结构：

```text
stage 1:
  multi-offset fused energy 生成 Top-L candidates

stage 2:
  从高可靠 Top-1 symbols 建立局部 payload anchor line
  对低置信 symbol 做局部 phase-line guarded override
  coherence 只保留为很小的 tie-break / guard
```

已经验证：

```text
python -m py_compile 通过。
phase=0 时 selector 能退回 multi-offset argmax 下界。
run_phase_line_threshold_sweep.py 可以输出 center / multi / v3 / phase_line / phase_oracle 对照表。
```

当前 probe 结论：

```text
0_0_0_10_14_16, SNR -22/-24 dB:
  v3 coherence-selected 仍明显强于当前 phase_line_selected。
  naive causal phase beam 会自我强化错误轨迹，已从默认路径移除。
  local-anchor delayed selector 更稳，但仍未达到“比 v3 稳定 +1 dB”。
```

重要诊断：

```text
纯 phase oracle 很差，说明 Top-L 里存在大量相位更贴线但不是正确 bin 的噪声候选。
因此 phase-line 不能无能量约束地单独判决。
```

下一步不要继续盲目调权重。优先做真正可靠的 oracle：

```text
使用 clean/header-first CSV 的 clean GT peak phase
或 low_snr_gt_bin CSV 中的 clean_gt_peak_phase / phase_linear_fit
来构建 clean payload phase trajectory；
再在 noisy Top-L candidates 中测试 phase-line guarded rerank 的上界。
```

如果 clean-phase oracle 仍不能超过 v3，说明当前 raw FFT phase-line 对候选选择的可用信息不足；
如果 clean-phase oracle 有明显空间，再继续研究如何用无 GT 的 anchors / EM / RANSAC 去逼近该 oracle。

### 11.1 最新诊断：单点 phase residual 判别力不足

新增候选级诊断脚本：

```text
scripts/experiments/phase_line/diagnose_phase_candidate_ranking.py
```

它统计每个 payload symbol 的 Top-L candidates 中，GT bin 按以下分数的排名：

```text
energy rank
global clean phase residual rank
local clean phase residual rank
phase + energy combined rank
```

在 `0_0_0_10_14_16` 上的 probe：

```text
SNR -22 dB:
  GT recall@24 = 0.969
  v3 error rows = 11
  v3 错误处 GT local phase rank=1 rate = 0.182
  v3 错误处 GT local phase rank<=8 rate = 0.364

SNR -24 dB:
  GT recall@24 = 0.855
  v3 error rows = 58
  v3 错误处 GT local phase rank=1 rate = 0.103
  v3 错误处 GT local phase rank<=8 rate = 0.448
```

结论：

```text
Top-L candidate recall 仍然有空间；
但“单个 candidate 的 phase residual 是否贴 clean/local phase line”不能直接当第二轮主排序。
错误 bin 在 modulo phase 上经常比 GT 更贴线。
```

因此后续 phase-line 主线不应继续做：

```text
candidate_score = mostly phase_residual_to_line
```

更可能的方向是：

```text
1. phase 作为局部一致性 gate，而不是直接 top-1 rank；
2. 用路径级统计，而不是单点 residual；
3. 引入 bin-dependent deterministic phase correction；
4. 使用 pairwise / second-order phase increments；
5. 在 phase-consistent subset 内再由 energy/coherence 选择。
```

换句话说，当前目标仍然是 phase-smooth candidate path selection，
但“phase smoothness”需要定义在路径级或校正后的相位空间里，而不是直接用 raw FFT candidate phase 到一条 line 的距离。

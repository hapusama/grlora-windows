# D2F（Decode-to-Finish）— 弱包检测解码链档案包（2026-10-05）

> **人类可读的步骤说明：[05_docs/D2F_接收流程说明.md](05_docs/D2F_接收流程说明.md)
> （每步做什么/为什么/验证程度，无内部代号）**——新读者先读它。
>
> **再读 [HANDOFF.md](HANDOFF.md)（交接/现状/下一刀）→ [FINDINGS.md](FINDINGS.md)**——高价值发现记录（精炼版：SFO
> 相位账/DeRa 护城河/相位模型驱动定律/SER-PER 解离/搜索税构成/实现
> 坑清单）。

> **系统名 D2F = Decode-to-Finish**（检测/同步/参数估计层自主、解码器
> 可插拔、CRC 仲裁收口——检测还没做完就把解码用起来，用解码结果
> "收尾"同步）。论文、战表、代码仓统一用此名。
> 内部代号对照：M3+ = m3p = D2F 前身编号；m3p2 = D2F + 斜率梯仲裁
> （现行形态）；dep4/dep5 = 检测器第 4/5 代（dep4k16 在役）。脚本注释
> 中的历史编号保留不动，以本 README 对照为准。
> 建包日期 2026-10-05，曾用目录名 `method_m3p_20261005`、
> `方法A_弱包检测解码链_20261005`（均已改名）。动机：项目内多 idea
> 并行开发，本包 = D2F 线（dep4k16 两级检测 × 连续锚 × 双列仲裁 ×
> 斜率梯）的方法、脚本、记录全量快照，与其他 idea 物理隔离。
> 规范：`weak_decoder/EXPERIMENT_PROTOCOL.md`（v1.1，§5A 检测验收=下游
> CRC 判决）为本线唯一实验规范——**不随包复制**（唯一权威，防分叉）。

## 0. 方法一句话与链规格

**Decode-to-finish 系统链**：检测/同步/参数估计层自主（贡献所在），
解码器可插拔（当前 A 列 = 收编 DeRa port，B 列 = 我方 TREL-5 走格），
CRC 逐级仲裁收口。

```
IQ 段（真实 OTA + 纯 AWGN）
 ├─①门扫：能量峰事件分组 top-3（最密窗起点防离群）
 ├─②检测 dep4k16：非相干提名(K16 能量列图) → 全场 12.25 行两群相干
 │   验证（前导+sync 一群 / SFD 镜像一群）；门限 13.74dB@FAR 1e-3
 │   （臂自 H0 池化精确分位，跨 δ/P 单门限）
 ├─③锚：胜点提名抛物线精化 → 连续 ν̂（native ≤0.011 bin）+ δ̂(0.01 格)
 ├─④对齐：ν_rot payload 居中 + 斜率梯 {0,.01,.02,.03} 逐符号 de-walk
 ├─⑤解码：A 列 DeRa port（两段相干合并）/ B 列 TREL-5（κ 格走格）
 ├─⑥仲裁：Δ0∈{0,±1,±2,±3} CRC 逐级早退（56 判次 vs DeRa 25）
 └─⑦fast 救援（2026-10-05 新增，第三级）：①-⑥ 全败时 δ 梯
     {0,±.01,±.02} × 前导盲 τ̂₀（逐候选，含噪段）× ε 精确逆重采样
     + advance(τ̂₀) → **A→P→B 三列** + Δ0 CRC（+70 判次；单调零回归；
     P=逐符号两段斜坡相位补齐 2026-10-06，护栏退回常数）
```

## 1. 目录结构

| 路径 | 内容 | 权威原件（可运行） |
|---|---|---|
| `01_core/` | `m3p_core.py`（dep4k16 盲检测+连续锚）、`m3p2_core.py`（斜率梯仲裁）、`m3p3_core.py`（turbo D 列，**默认停用**，见未决清单 3）、`ref_template.py`（精确模板相关器，用户方案行1/2 参考实现）、`fast_tmpl_core.py`（fast_tmpl 快实现接收机：ε 梯重采样修复+前导盲 τ̂₀+fast 臂，2026-10-05 入列）、`phase_align_core.py`（**P 列两段斜坡相位补齐 + N 非相干消融臂 + fast_arm_sets 多套列一次投影**，2026-10-06 入列） | `data/experiments/e2e_final_20261004/`（m3p*） |
| `02_detector/` | `m2c_core.py`（dep4 两级检测器库）、`m2d_core.py`（dep5 细 δ 轴+连续锚） | `data/experiments/dep_anchor_20261004/` |
| `decode_trellis/` | **B 列 κ 格解调器 D2F 自有副本**（2026-10-05 迁入，逐位一致验证；trellis 去网格化演进在此进行） | `weak_decoder/decoding/kappa_trellis/` |
| `03_battle/` | 战与汇总脚本：`m3p_battle.py`、`m3p2_battle.py`、`m3p_sum.py`、`m3p2_sum.py`、`m2d_smoke.py`（dep5 冒烟）、`m4_run.py`（turbo 相位列）、`sfo_sto_experiment.py`/`sfo_phase_decomp.py`（SFO/STO 战）、`fast_tmpl_check.py`（验收①②③）、`fast_tmpl_mech.py`（机制战）、`fast_tmpl_battle.py`（链战 join m3p/m3p2/dera）、`phase_align_check.py`（P/N 机制实验）、`phase_ab_battle.py`（**模块消融链战：系统阶梯 L0-L4**） | 同上两目录 + `phase_track_20261002/` |
| `04_results/` | `RESULTS_M3P.md`、`RESULTS_M3P2.md`（终战报告）、`RESULTS_FAST_TMPL.md`（fast_tmpl 验收+机制+链战）、**`RESULTS_PHASE.md`（相位补齐+模块消融）**、`m3p_sum_results.txt`、checkpoint 族（m3p 8484 / m3p2 5656 / fast_tmpl_mech 700 / fast_tmpl 4536 / phase_align 728 / phase_ab 4536）、`m2c_roc_results.json`、`m2d_smoke_results.json`、`m4_units.jsonl`；`logs/` 运行日志 | 原目录 |
| `05_docs/` | `dep5_搜索税消解_20261004.md`（设计+杀死裁定+构成定律）、`方法突破_20261004.md`（全夜会战存档 v3-v5 段）、**`D2F_接收流程说明.md`（人类可读步骤文档）**、**`IDEAS.md`（论文级 insight 记录，I1=对齐量不估计要搜索）** | `doc/` |

## 2. 依赖地图（副本不随包复制的原因）

 battle 脚本经绝对 `sys.path` 引用以下库 ⇒ **移动会断链，故原位保留**，
 副本仅作阅读/快照；battle 类副本实际仍可运行（CKPT 解析回原目录，
 join/断点一致）：

- `keystone_battle_20261003/d1_core.py`（场布局/CFAR 闭式/谱原语）、
  `d1_battle.py`（帧构建）、`d2_core.py`（SFO 注入/模板）
- `dep_anchor_20261004/m2_core.py`、`m2b_core.py`（dep2/dep3 前代 +
  O_SYNC/KMIRROR 常量）
- `e2e_final_20261004/m3_core.py`（门扫/盲对齐/demap+Δ0 仲裁/nu_rot）
- `dera_front_battle_20260930/front_runner.py`（DeRa 检测 port + 全链）
- `weak_decoder/decoding/kappa_trellis.py`（B 列）、
  `baselines/dera/paper_dera_demod.py`（A 列）
  ⚠️ **B 列已迁移 D2F 自有副本 `decode_trellis/`（2026-10-05）**：4 文件
  224 行冻结快照，逐位一致验证过（viterbi/bcjr）；唯二改动 = chirp 导入
  改绝对路径 + 溯源头。**包内全部脚本（含 m3p/m3p2 battle 副本）已切换
  import 自家副本，包内零依赖共享 kappa_trellis**（切换后逐位一致复验
  通过；原目录 data/experiments/e2e_final_20261004/ 的可运行原件仍
  import 共享 canonical，保证历史 checkpoint 复跑逐位可比）。并行会话
  曾另放一份快照 01_core/kappa_trellis/，与 decode_trellis 数值体等价，
  已去重删除。
- `phase_track_20261002/e1_common/e2_arm/e3_common`（M4 turbo 相位环）

运行环境：`D:/mysoft2/miniconda3/envs/gr-lora/python.exe`。

## 3. 复现命令

```bash
PY=D:/mysoft2/miniconda3/envs/gr-lora/python.exe
E2E=data/experiments/e2e_final_20261004

# 终战（断点续跑；本线已满额，命令仅供复跑验证）
$PY $E2E/m3p_battle.py 55
$PY $E2E/m3p2_battle.py 55

# 终表复原（本包内离线可跑）：
cd "weak_decoder/D2F_20261005/04_results"
$PY ../03_battle/m3p_sum.py      # 读本目录 m3p_checkpoint.jsonl
$PY ../03_battle/m3p2_sum.py     # 读本目录 m3p2_checkpoint.jsonl
```

## 4. 战果摘要（对 DeRa 全链 port，SF10 OTA 28 帧，纯 AWGN，20 种子）

| 场景 | 判决 | 数字 |
|---|---|---|
| δ=0（无漂移） | **胜（扩大）** | 10%PER −22.1 → −23.0 → **−23.3（+P）**；−24 档 .174 vs .672（3.9×，同分母）；−18 档 .000 vs .002 |
| δ=0.02（真实漂移上界） | **全档首胜（反转）** | −24 **.197 vs .663（3.4×）**；−22 .049 vs .249；−20 .007 vs .191；−18 .002 vs .163；10%PER **≈−23.0**（对 m3p2 −20.76 = +2.2dB）vs DeRa 全档不达 10% |
| δ=0.082（压力档） | 双灭（未入 fast 战） | 预注册失效场景 |
| 模块消融（2026-10-06） | 设计合理定案 | δ=0.02 −24 贡献：**fast .652 ≫ 相干合并 .401 > 梯 .084 > B .019 > P .007**；两段相干合并链级价值 +2.2dB/.40PER 首次量化 |

（fast_tmpl 战表：RESULTS_FAST_TMPL.md §3-4；P 列+消融：RESULTS_PHASE.md
§2-3（同分母声明）；m3p/m3p2 战表见 RESULTS_M3P/M3P2.md。口径注意：
DeRa 检测级 port 瘦身版声明沿袭，v14 重构后双版本改判。）

检测层（诊断级，§5A）：cert +4.0~4.2dB（oracle 标签）；FF_cert8 OTA
等虚警 +0.6~3.7dB（见 `fullfield_cert_detect_20261003`，不在本包）；
dep4 +0.63~2.18dB（半格点 δ 补考在列）。

## 5. 归因（创新含量口径）

- **可写创新**：①证书检测层（固定变换族+闭式 H0 vs DeRa 自适应搜索
  无解析 H0——选择耦合税是后者结构属性）；②marginalize 家族解码
  （B 列 trellis 不估走动走格）；③**模型驱动时间基修复（fast_tmpl：
  ε 精确逆重采样 + 前导盲 τ̂₀ → 常规 dechirp-FFT，2026-10-05 入列；
  附带 τ̂ 仪器定律 = §5A 下沉到估计器 + (δ_c,τ̂) 联合简并）**；
  ④测量式相位（γ 盲差分/turbo 环，第二篇种子）。
- **工程兑现**：连续锚、门限 bug 修复、斜率梯、A 列收编、能量行
  （合法但非创新）。

## 6. 未决清单

1. ~~δ=0.02 深端：port 切分点补偿~~ **已修（fast_tmpl，2026-10-05）**：
   ε 精确逆重采样 + 前导盲 τ̂₀ → −24 .863→.241（vs DeRa .673 反转）。
   剩余：浅端 1~3% 错误地板（DeRa −18/−20 仍略优 .004 vs .002）；
2. τ 细梯（±0.5 采样 ×3 候选）：GT 扫描裁定 −24 天花板 .057 vs 现状
   .086（12 败中 8 信息受限、4 可细梯救）——小收益候选，预算 ×3；
3. **D 列 turbo 相位（已集成、默认停用）**：`01_core/m3p3_core.py`——
   机器端到端验证过，但 e1-harness 增益（10.6%@−24）不向链上转移。
   下一刀 = 同帧同噪 e1 vs 链上中间量差分（q/κ-line/φ̂/rows）；
4. BCJR 深端臂、dep4 半格点 δ 补考（协议 §5A 存量四项）；
5. SF11 扩展（fast_tmpl runner 现成，改 SEED_CONST/数据源即可）、
   **行3-5 跨符号相干（段内相位通道已关账——F11 余量定律；跨符号
   = 榨取更多相干能量的唯一出路，入口 = e3 turbo 转移断点差分）**、
   κ̃-GLRT 跨 bin 相干（Dirichlet 泄漏回收）、second-paper 理论章
   （耦合税构成律 05_docs/dep5 §6）；
6. τ̂ 机理（学术点）：payload |Z|² τ̂ 偏低 2 采样的解析解释（fold 位置
   ×目标函数相互作用）+ (δ_c,τ̂) 联合简并的参数空间几何——F10；
7. 段幅度不平衡定律的解析式（F11：合并相位增益 ∝ 2|F||T|/(|F|²+|T|²)
   对 fold 位置的分布平均）——论文估计器章素材。

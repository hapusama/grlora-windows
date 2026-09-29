# kappa_trellis — κ 细格跨符号解调器（我方方法）

> 2026-09-29 正式成模块。此前实现以内联形式散于实验脚本
> （`data/experiments/{ota_decode_only,bcjr_ablation,dera_battle,sf11_battle}_20260929/`），
> 本模块为唯一权威源，数值与战表脚本逐位一致（`tests/test_kappa_trellis.py`
> 含等价校验）。

## 1. 问题

同步给定（整数+分数 CFO、分数 STO 均已按先验补偿）后，符号音仍坐在
非整数 bin 上：残余 κ ∈ [−0.5, +0.5]（先验误差+SFO 漂移），且逐帧
不同、个别帧越过 ±0.5 触发整 bin 回绕。经典路线是"估 κ 再校正"——
但弱信号区 κ 估计本身不可靠；且 κ=±0.5 处相邻 bin 统计逐 bit 不可分
（半 bin 不可分命题，doc/halfbin_proposition_20260928.md），逐符号
硬判决必掷硬币。**本方法：不估、直接在 κ 假设格上跨符号联合，把 κ
当作 nuisance 边际化/路径化，输出软证据交给标准 LoRa FEC 链。**

## 2. 方法（三层）

**① κ 细格谱（`fine_grid.py`）** —— 每符号：带通预滤 → 4 个过采样
相位的 chip 抽取 → 5 列 κ 旋转（κ=(2−d)/4，步进 0.25 bin）→ 各相位
功率平均，得 (N, 5) 窗口能量 `m[s, d]`。列 d 即"κ=d 假设下"的完整
候选谱。整数 bin 回绕天然被 ±0.5 列覆盖（配合逐帧 δ，见 §4）。

**② 跨符号格架（`trellis.py`）** —— 发射项（突出度）：
λ[k,d] = log(max_s m[s,d] − median_s m[s,d])；转移：|Δd|≤1 均匀
（对应 κ 漂移 ≤0.25 bin/符号，覆盖 SFO 残差）。两种读出：

| 读出 | 算法 | 证据行 | 机制角色 |
|---|---|---|---|
| `grid0` | 无格（κ=0 单列） | m[:,2] | 消融基线（NEW-0） |
| `viterbi` | 联合最优 κ 路径（TREL） | m[s, path[k]] | **硬读出，默认模式** |
| `bcjr` | 前向-后验边际化（BCJR） | Σ_d P(κ_d\|全包)·m[s,d] | 软读出，深端模式 |

**③ 判据衔接（`demodulator.py`）** —— 输出 bin 域软证据行，按值映射
（非 LDRO：value=(bin−1)%N；LDRO：value=((bin−1)%N)//4，值 v 占
bins 4v+1..4v+4）交给 SymFEC 风格软 FEC + CRC16（判据须
header_valid 且 has_crc，见 ota_replay RESULTS 勘误）。

**核心主张（vs baseline 的定位轴）**：Savaux/UniChirp/DeRa 走
估计-补偿或候选-选择（est-compensate/select + 硬判决），本方法在同
一"同步给定"设定下对 κ **联合+边际化**并输出软证据——机制层面无先例
（2026-09-29 文献查新，DeRa=MobiCom'26 为最近邻但为选择式）。

## 3. 为什么两种读出（实验刻画的真实结构）

- **工作区（native ~ −23dB）**：每帧 κ 近似常数 → Viterbi 锁对路径
  即近零 SER，**TREL 全面第一**（含 Savaux）。
- **极深端（≤ −25dB）**：λ 不可靠、列歧义大 → 软混合 + FEC 兑现，
  **BCJR 反超**（与合成 κ=0.5 对抗场景 bcjr_marginal 互证）。
- 一个框架、两种读出、分区适用；格密度（5→9 列）在 OTA 常数 κ 下
  非瓶颈（bcjr_ablation 归因）。

## 4. 约定与判据（复现必读）

- **输入**：已完成整数+分数 CFO 频偏与 STO 亚 chip 分数时延的信号
  （全链统一施加，含全部 baseline——公平铁律，见
  lora-ota-savaux-prior-bug 教训）。
- **δ 冻结**：每帧整 bin 残余映射在干净信号上以
  mode(argmax−期望bin) 冻结（同步先验一部分，噪声档不再调整）；
  `freeze_delta()` 提供该协议。
- **噪声**：有且仅有 AWGN，同一实现喂所有链（用户三令）。
- **SNR 口径**：整包加噪后 SNR（S=段功率−带外噪声底，全采样带）。

## 5. 亮眼结果（全部真实 OTA + 纯 AWGN，同种子配对）

**SF10（28 帧，`ota_decode_only`+`dera_battle`+`bcjr_ablation`）**

| 整包SNR | LoRaTrimmer | Savaux | NEW-0 | **TREL-5** | BCJR-5 |
|---|---|---|---|---|---|
| native(+8) | .032/.143 | .038/.107 | .032/.143 | **.000/.000** | .033/.143 |
| −20 | .067/.226 | .053/.167 | .074/.226 | **.024/.048** | .051/.190 |
| −22 | .101/.310 | .079/.286 | .114/.369 | **.048/.167** | .085/.274 |
| −24 | .220/.929 | **.184**/.762 | .246/.940 | .211/.964 | .192/.786 |

- **对 LoRaTrimmer：native~−23 全档 SER+PER 双胜**，工作区等效门限
  差 ~2.5-3 dB（−20dB 处 SER 2.8×、PER 4.7×）；−24 起打平。
- 对 Savaux：中段领先，−24 起被其相干能量合并反超（边界如实保留）。
- 组件归因（bcjr_ablation）：κ格+路径选择是决定性组件
  （−22dB SER .120→.048）；软硬读出分区依赖；格密度次要。

**SF11（120 帧，LDRO，`sf11_battle`）**

| 整包SNR | LoRaTrimmer | Savaux | **TREL-5** | BCJR-5 |
|---|---|---|---|---|
| native | .016/.000 | .001/.000 | **.000/.000** | .013/.000 |
| −22 | .060/.078 | **.036**/.078 | .035/.081 | .048/.081 |
| −25 | .110/.203 | .093/.339 | **.080/.175** | .089/.150 |
| −26 | .159/.564 | .156/.617 | .133/.542 | **.129**/.389 |

- 对 LoRaTrimmer SER 全档保持领先（幅度收窄——LDRO 4-bin 值格粗化
  bin 级差异）；深端 BCJR 双料最强（PER .389 vs .564）。
- Savaux 在 SF11 中段 SER 最强（2048-chip 长符号相干优势）——方法
  排序随 SF 变化，论文按 SF 分区陈述。

**尚未入列**：DeRa(MobiCom'26) port 挂起（分段公式待精读，WIP 于
`baselines/dera/`）；SF7/SF12 臂待数据。

## 6. 与 baseline 模块的接口对等性

本模块输出与 `baselines/{savaux_oversampled,loratrimmer,unichirp}`
同构的逐符号 bin 域能量行；战 runner（`dera_battle/battle_runner.py`
与 `sf11_battle/sf11_runner.py`）为并行+断点的参考调用方。历史锚点：
合成床消融 `bcjr_marginal_20260928`（12 点中 11 点 BCJR≥旧链）。

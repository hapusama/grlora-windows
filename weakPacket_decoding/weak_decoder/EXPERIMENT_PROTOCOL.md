# EXPERIMENT_PROTOCOL — 弱包解调对比实验规范（v1.1；v1.0 2026-09-30，v1.1 2026-10-04 增补 §5A）

> 本文是唯一权威实验流程。所有新实验（脚本、报告、论文数字）必须遵循；
> 与历史运行的差异须在 RESULTS 中显式声明。维护者：hapusama。
> 参考实现：`data/experiments/{ota_decode_only,bcjr_ablation,dera_battle,
> sf11_battle,full_chain}_2026092*/` 与 `dera_front_battle_20260930/`
> （实验A 系统战型：前端×解调器矩阵，DeRa 检测级 port 入列）；我方方法一律 import
> `weak_decoder.decoding.kappa_trellis`（不得内联拷贝）。

---

## 0. 铁律（违反任何一条 = 结果作废）

1. **噪声有且仅有 AWGN**：复高斯白噪声直接加在真实基带信号上，
   同一实现喂所有链。禁止：窗口错位噪声、色噪声/带宽塑形、κ/偏移注入、
   按链定制噪声、任何利好我方方法的加扰设置（学术红线，2026-09-29
   Savaux 冤案教训：先验只喂一半曾致 baseline 被冤崩）。
2. **公平输入**：所有链拿到同一段信号、同一份同步信息、同一判据。
   baseline 的已知机制前提必须喂足（如 Savaux 相干合并需要分数 CFO +
   STO 亚 chip 时延的全先验对齐）。
3. **GT 只来自原始干净信号**（§2），不得由任何被评估链参与产生。
4. **SNR 口径 = 整包加噪后 SNR**（§3）。对外报告禁止"+xx dB 加噪量"。
5. 判据侧的任何改动必须先过负控（坏 payload / 坏 header 必须被拒），
   防止 `crc_valid` 类旁路回归（2026-09-29 has_crc 空真事故）。

---

## 1. 数据集（进入实验的帧）

**来源**：真实 OTA 捕获（USRP + SX1276/LoraSTMac 发射），现库：
SF10 三文件（preamble 8/16/32，28 帧）+ SF11 lab1_sf11_TP2 十六文件
（LDRO，120 帧）。禁止合成 TX 波形进入对比实验。

**接入管线**（每 capture 一次性，产物入 `data/weak_sync_chain/`）：

```
run_weak_sync_chain.py -i <bin> -o sync_chain/<name>_sync_chain.csv \
    --preamble-len <P> --win-chirps 4 --hop-chirps 1 \
    --min-periodic-peaks 12 --frame-min-preamble-peaks 12
run_header_first_demod.py -i <bin> -s <sync.csv> \
    -o header_first/<name>_..._symbols.csv \
    --frames-output header_first/<name>_..._frames.csv --sf <SF> ...
```

**帧筛选（数据集规则）**：
- `header_valid=1` 且 `payload_len>0`；
- **原生（不加噪）GT 可解**：以 CSV 先验在干净信号上
  `demod_symbol_sequence` 解出全帧符号，`decode_explicit_frame_symbols`
  的 header 校验和与 payload CRC16 **必须通过**——过不了的帧（本身
  SNR 太低）直接剔除，不进任何实验的分母。例：SF11 126 有效帧 →
  120 帧入集。
- LDRO：SF≥11 时 TX 按 LoRaWAN 规范开 LDRO（CSV `ldro` 列为准），
  GT 与值映射走 `payload_ldro=True / ldro_mode=1`。
- 逐帧 preamble 长度（8/16/32）从文件名尾段读取，**同步链配置必须
  逐帧匹配**（曾因统一按 8 配置致 18/28 帧错位的假 SER 0.7）。
- CSV 的 `branch_*` 列为 4 相位多值（`|` 分隔），取首值（=有效支路）。

**GT 冻结**：每帧的 GT header 值、GT payload 符号值、GT 字节在实验
脚本启动时按上法解出并冻结，全程只读。

---

## 2. SNR 与噪声

- **整包 SNR**（dB）= S / (N0 + P_add)，全采样带（500 kS/s）口径：
  - S = 段功率 − 带外噪声底（|f|∈(70,240) kHz 的每 bin 平均，
    Parseval 单位），N0 = 带外噪声底；
  - 目标 SNR 逐包标定：P_add = S/10^(SNR/10) − N0（下限 1e-30）。
- 噪声实现：`(randn + j·randn)·sqrt(P_add/2)`，随机种子由
  (实验日, 档位, 种子号, 帧号) 确定性派生——保证可复现且逐单元独立。
- native（不加噪）档必须包含，作为管线健康检查（§6 冒烟标准）。

---

## 3. 两种实验

### 实验A：全链（整包加噪 → 检测+帧同步+解调，无先验）

对应参考实现 `full_chain_20260929/exp2_runner.py`。

1. 段 = `[hs − (P+6)·NF, hs + (8+psym+2)·NF]`（P=该帧 preamble），
   **噪声加在整个段（含前导）**。
2. 同步全部在带噪信号上重做（与产线同一套模块）：
   `detect_preamble_runs`（按 P 配置）→ 事件按置信度排序逐个尝试 →
   `align_event_start`（radius 8192, step 512, chirps 4）→
   `locate_frame_from_event` → `run_grlora_frame_sync_validation`
   （preamble_len=P）；任一事件通过即用其估计
   （cfo_int/cfo_frac/payload_sto_frac/fine_payload_start）。
3. **同步失败 = 该单元全链 PER 记错**；SER 只在同步成功单元上统计，
   并单独报告**同步成功率**（它是独立的性能轴）。
4. 对齐：整 bin+分数 CFO 频偏 × 全段 + STO 亚 chip FFT 分数时延，
   全链同一份带噪估计。
5. **δ（整 bin 残余映射）= 逐链 CRC 仲裁**：δ∈{0,±1,±2} 候选依次
   过判据，CRC 通过者胜（DeRa 同款 CRC-guided 原语；判据已修复故无
   空真，错误 δ 撞过 CRC 概率 ~2⁻¹⁶/候选）。
   **禁止**：per-capture 常数标定（域不一致）、header 校验和锚定
   （header κ 分裂随机锁错）、sync-word 实测偏移（±1 抖动）——三者
   均已实证翻车。

### 实验B：先验给定（干净同步 → 符号段加噪 → 只比解调）

对应参考实现 `dera_battle_20260929/battle_runner.py`（SF10）与
`sf11_battle_20260929/sf11_runner.py`（SF11/LDRO）。

1. **同步先验在干净信号上取定并冻结**：整 bin+分数 CFO 频偏 +
   STO 亚 chip 分数时延（delay = −sto_frac·OS），对段统一施加。
2. 噪声加在解调段（先验已冻结，不重新同步）；段含前导（UniChirp
   等需要重看前导的链按接收机真实行为用带噪前导训练）。
3. **δ 在干净信号上冻结**：mode over payload 符号 of
   (argmax − 期望 bin)。期望 bin：非 LDRO = v；LDRO = 4v+1。
   噪声档不再调整 δ；逐符号翻转如实计入 SER。
4. 该实验回答"同步给定时的解调器排序"；与实验A的差 = 同步链代价。

---

## 4. 参赛链

| 链 | 角色 | 实现 |
|---|---|---|
| PLAIN | 传统 LoRa 硬解链（"朴实无华"参照） | argmax 硬判决 + `decode_explicit_frame_symbols`（判据含 header_valid） |
| OLD-A | 部署裸抽取（gr-lora frame_sync 风格） | `olda_rows`（offset-0 相位 chip 抽取） |
| TRIMMER | baseline（MobiCom'24） | `baselines/loratrimmer/demod_loratrimmer_symbol` |
| SAVAUX | baseline（TIM'22，相干分支合并） | `baselines/savaux_oversampled/demod_paper_oversampled_symbol` |
| UNICHIRP | baseline（SECON'26，双峰相位模型） | `baselines/unichirp/...`（OTA 段内仅 4 干净前导可训练，天然受限，如实报告） |
| NEW-0 | 我方消融基线（κ=0 单列） | `kappa_trellis` readout="grid0" |
| **TREL-5** | 我方默认（κ 格 Viterbi 硬读出） | `kappa_trellis` readout="viterbi" |
| BCJR-5 | 我方深端模式（前向-后验软读出） | `kappa_trellis` readout="bcjr" |
| DERA | baseline（MobiCom'26）| `baselines/dera/` port v2（2026-09-29 深夜修复：逐候选越界切分+相干合并，native SER .002，对 Trimmer ≈1.8dB@10%SER 与论文自报 +1.9~2.0dB 交叉验证通过）**已入表**（dera_battle_20260929、dera_fusion_battle_20260930）。仍缺 Algorithm 1 Stage 3 与检测级 CFO——E3 补齐前禁止任何"vs DeRa 胜利声明"（神招辩论 §8.1-7 R2 红线，双向不坑 baseline） |

UniChirp 的训练配置：4 个干净 preamble 符号探针定 bin，逐单元（带噪）
重训练——接收机行为；实验B 同。

---

## 5. 指标

- **SER**：各链 bin 选择（argmax，按各自 δ 与值映射）对 GT 符号的
  错误率。demod 级，不进 FEC。值映射：非 LDRO `value=(b−δ) mod N`；
  LDRO `value=((b−δ−1) mod N)//4`。
- **PER**：整包 CRC16 未通过率（判据 = 修复后的
  `decode_symfec_payload_from_evidences` / `decode_explicit_frame_symbols`，
  须 header_valid 且 has_crc=1 路径）。**无 MIC**：payload 为固件固定
  测试图案（非 LoRaWAN 帧），CRC16 是唯一且正确的判据。
- 实验A 的 PER 含同步失败；SER 分母 = 同步成功单元。
- 报告表：每档 n（包数）、SER、PER、（实验A）sync 率。多种子
  （≥3）逐档，种子不同实现独立；建议每档 ≥84 包。

---

## 5A. 检测/估计模块的评估判据（v1.1 增补，2026-10-04）

> 背景：本仓"检测" = 发现包 + **精确参数估计**（κ̂/ν̂/δ̂/σ̂² 随判决
> 一并输出），不是二值告警。检测层等虚警指标（Pd@等虚警交点）测不出
> 对齐精度：E3-lite 已实证"检测级全档 ≥DERA 而 PER 中段未胜"，缺口
> 恰在锚对齐精度（±0.2-0.3 bin）。故检测模块的效果**只能以下游
> decoding 判决**（用户裁定 2026-10-04）。

1. **验收判据 = 下游 decoding**：检测/同步模块的一切"胜/负"声明，
   只能出自实验A式全链、CRC 仲裁的 PER/SER——同一下游、只换检测
   模块、DeRa 同表（E3 形态）。检测层等虚警交点表降级为
   **诊断/归因**用，单独出现时不得写作"胜 DeRa"。
2. **中间层指标（归因用）**：参数误差 |κ̂−κ|、|ν̂−ν|、|δ̂−δ| 对 GT
   的分布（CDF/分位数），用于把 PER 缺口分解到估计器（例：δ̂ 深端
   |误差|>0.04 占比 6~15% = 已知退化点）。
3. **归因防护**：对比时下游解调器必须固定（只换检测/估计模块）；
   否则弱底座冤枉好检测器、好底座掩盖坏检测器，两个方向都会误归因。
4. **注入型检测实验附加规则**（keystone/m2c 系，即"实验C：检测/
   估计型"，升格自 2026-10-03/04 战表）：
   - 测试 δ 必含**半格点值**（如 0.005/0.015/0.025），禁止只测
     搜索格节点；格点红利需用"格加密一倍"消融量化；
   - oracle 臂（GT 模板/无噪拟合参数）必须带 **oracle 标签**，
     与可部署臂分表分报，不得混报；
   - δ 档标注实测界（A3 实测 0.002~0.024 bin/符）：超出上界的
     测试点标"压力档"，不得当默认工况报数；
   - H0 优先真实噪底窗（FF 式）；使用合成 AWGN H0 必须声明。
5. 三层分报：检测实验 RESULTS 必含 **oracle 证书级 / OTA 等虚警级 /
   可部署级** 三层分表 + 未决补考清单（2026-10-04 审计起的存量补考：
   dep4 半格点 δ、格加密消融、真实噪底 H0 重校）。

---

## 6. 运行与健康检查

- **Runner 工程规范**：multiprocessing 并行 + JSONL 断点续跑
  （每完成一个 (档位,种子,帧) 单元即时落盘；重跑自动跳过已完成单元；
  更换链集合需清断点）。段缓存 complex64、worker 数按内存调
  （SF11 120 帧 ≈ 6 workers 上限，曾 OOM）。
- **native 冒烟标准**（不过禁止继续）：
  - 实验B：TREL SER = 0（或 ≈历史值），其余链 SER ≤ 0.06；
  - 实验A：sync 率 = 1.00，TREL SER = 0；
  - 判据负控：单符号损坏 / 坏 header 必须被拒。
- **判据快路径**若向量化，必须与逐 bin 原版逐位等价（启动自校验）；
  移位约定：bin 域 roll(−(δ−1)) 后经 (bin−1) 映射（实验B 路径）。
- 汇总键注意 `"%+d"` 正负号（曾 +10/−10 不匹配致战表漏行）。

---

## 7. 报告模板（每个实验目录的 RESULTS.md 必含）

1. 铁律遵守声明（AWGN/公平输入/SNR 口径）；
2. 数据集（capture/帧数/剔除数/preamble/LDRO）；
3. 战表（§5 格式）+ native 冒烟结果；
4. 与既有实验的关系（同种子配对 / 独立种子重复）；
5. 偏差声明（与本规范的任何差异及理由）；
6. 检测/估计型实验：按 §5A 三层分报 + 补考状态。

## 8. 现行结论速查（详见各 RESULTS）

- 实验B（SF10/28 帧 + SF11/120 帧）：TREL-5 对 LoRaTrimmer 全档
  SER 领先（SF10 工作区 PER 亦全胜，等效门限 ~2.5-3 dB）；深端
  （≤−24 dB）Savaux 相干合并最强；BCJR 软读出 ≤−25 dB 反超 TREL。
- 实验A（SF10/28 帧）：TREL-5 全档第一；**同步门限（sync 46%@−20、
  11%@−22）高于 BCJR 反超点（≤−25）**——全链到不了软读出领地，
  瓶颈在 detection（检测级联的定量论据）。

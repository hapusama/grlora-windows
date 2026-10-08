# weakPacket_decoding 顶层认知（COGNITION）

> **这是本项目最外层、最重要的认知文件。** 结构导览见 `README.md` 与
> `doc/PROJECT_MAP_20260928.md`；本文件回答的是另一个问题：**这个项目在
> 领域地图上站在哪、为什么站在这、牌是什么、哪些路已经裁决关闭。**
> 迷失方向、想换赛道、被新论文打崩的时候，先回来读这一页。
>
> 版本 v1，2026-09-29 定稿（与用户对齐）。依据：弱包两分法灌输 + DeRa
> 全文精读 + 围殴战（`data/experiments/dera_battle_20260929/`）+ 三选项裁决讨论。

---

## 0. 一句话故事（论文的脊柱）

**Perfect sync 是弱包解调的虚假前提。** 现有解码工作（族2）要么估计-补偿-硬判
（LoRaTrimmer/UniChirp/DeRa 解码器），要么干脆假设同步完美（DeRa 的解码器
评估实验）；我们把同步不确定性**边际化**进 κ 格 trellis（marginalize-soft），
让解码器自己吸收同步残差。深端（≤−24dB）相干副本合并（族1 Savaux）接管，
这是必须诚实画出的边界；DeRa 是我们的 est-compensate 对照组 + 检测前端，
不是对手。

## 1. 领域地图：弱包工作两族（用户灌输，2026-09-29）

| | **族1：能量副本叠加** | **族2：物理性质解码** |
|---|---|---|
| 代表 | MALORA / XCOPY / SymFec / Savaux | LoRaTrimmer / UniChirp |
| 底层逻辑 | 单副本能量不够 → 用冗余造/取同包多副本 → 叠能量 | 能量不动 → 挖单副本内可利用的物理性质 |
| 在哪个维度做文章 | **能量**（每比特有效 SNR ≈ ×N） | **信息**（同样能量少丢信息） |
| 代价 | 时延 / 带宽 / 冗余 / 多天线 | 复杂度 |

**我们的位置：族2 内部的机制级分叉。** 族2 现有工作把物理性质变成更强的
估计/校准（estimate-then-compensate → 硬判决）；我们把物理性质变成软信息
（marginalize-soft → κ 格 trellis，对信道参数边缘化）。

> ⚠️ 纪律：MALORA/XCOPY/SymFec/LoRaTrimmer/UniChirp 各自确切机制未现场
> 核对，写 related work 时逐篇读原文，不靠记忆、不靠搜索摘要。

## 2. DeRa 冲击报告（MobiCom'26，已精读定稿）

DeRa: Deep-Range LPWAN IoT via Ultra-sensitive Signal Detection.
Du/Lin/Cao, MSU, DOI 10.1145/3795866.3844157. 主轴 detection
（相干 preamble 累积 + 40kHz CFO 复合失真解析补偿，检测门限 +4.3~4.8dB）；
贡献(3) 是 CRC 引导迭代解码器（两段窗相干合并分裂能量）。

**它杀掉的（认账）：**
1. 检测空地——MobiCom'26 已占。
2. "族2 缺完整链条"的字面版——它自称 closes detection-to-decoding loop。
3. "给定同步下解调无人做"的 firstness——Fig 6c/d perfect-sync 实验
   赢 LoRaTrimmer +1.9~2.0dB（10% SER 门限）/ +1.7~1.9dB（90% PRR），
   与我们实验一同设置同指标。

**它没杀、反而送来的（核实过摘要与 port 后确认）：**
1. **它闭环的是 detection→decoding，不是 sync→decoding。** 三个贡献里没有
   sync；解码器评估必须假设 perfect CFO/TO knowledge——这句话是引言里最硬的
   引文：连最新 MobiCom 工作评估解码器都得绕过同步段，因为弱包 SNR 下
   没人交付这一段。**同步楔子被它引用式地承认了。**
2. **机制正交是硬的**：摘要原话 analytical compensation；全文 0 处
   marginal/trellis/BCJR/LLR/soft。同一物理现象（CFO 越界的符号能量分裂），
   它补偿、我们边缘化。
3. **数字不虚**：它对 LoRaTrimmer +1.9~2.0dB（perfect sync）；我们 TREL-5
   对 LoRaTrimmer **+2.5~3dB**（native~−23dB SER+PER 双料全胜，三次独立
   种子复现，围殴战 + 消融 + exp1）。
4. 它没比全相干 Savaux 过采样接收机——评测矩阵缺口是我们的卖点。
   开源（github.com/daibiaoxuwu/DeRa，当前仅 gen_signal.py）→ port 是
   现成对照组。

**负债（不粉饰）：**
1. 贡献(2)（时漂/频移/泄漏解析补偿维持相干累积）是 sync 邻接——审稿人必问
   "你的同步比它的补偿多什么"。答案：它的补偿服务于 preamble 累积，不交付
   "同步残差→解码器"通路。
2. **port 目前是坏的**（native SER 0.14，分段边界与逐候选 Δ 对不上，
   Eqs.10-23 被 pdftotext 损毁）。不坑 baseline 红线：未修好不入表。
   修好前"TREL-5 > DeRa"只是跨表推断。
3. 深端 ≤−24dB 是 Savaux 相干合并的地盘，别声张全深端。

## 3. 三选项裁决（2026-09-29，与用户对齐）

| 选项 | 判断 | 处置 |
|---|---|---|
| ① 深耕 κ 格 trellis 解码 | **唯一机制独有**的主线；短板是必须升级成"虚假前提"故事 | **做脊柱** |
| ② Savaux 副本数 K 探究 + 同步链完善 | 单独立不住：K-scaling 是族1 地盘上的参数曲线非新机制，且族1 恰是我们软边缘化最弱处 | **吸收**：边界章（深端 Savaux 接管线）+ oracle vs real sync gap 实验 |
| ③ 挤 DeRa 赛道做更牛的同步 | 否：赛道刚被 MobiCom'26 占；我方无检测资产（PC-Ridge 是 sync 非 detection，detection cascade 从未建成） | **关闭**：DeRa 引用为前端 + 对照组，引用它比打败它便宜一个数量级 |

三项没有一项浪费：②的 PC-Ridge（已验证 +15pts @11dB at oracle ceiling）
复活为 sync 链组件；③的 port 变成主表缺的那一列。

## 4. 论文结构与落点

章节骨架（对应上面故事）：引言用 DeRa perfect-sync 引文立"虚假前提" →
机制（κ 格 marginalize-soft，两种读出：TREL 默认 / BCJR 深端）→ 主表
（10 列 + DeRa = 11）→ 同步链实验（oracle vs real gap）→ 深端边界章
（Savaux K 分析）→ 复杂度表。

落点：**TIM 主投**（SCI Q1，Savaux'22 + Ibi'24 阵地）；**SECON'27 是 CCF-B
唯一现实窗口**（里斯本，full paper 约 2026-11-30 截稿，UniChirp 证明该会收
这类工作）；IoT-J（CCF-C）备胎；TVT 已不在 CCF 目录，旧计划作废。

## 5. 证据现状（截至 2026-09-29）

- TREL-5 vs LoRaTrimmer：等效门限差 ~2.5-3dB，native~−23dB 双料全胜，
  三次独立种子（exp1 / 消融 89868f1 / 围殴战 4ccba58）。
- 深端分区：−24 起与 LoRaTrimmer 打平（CR4/5 悬崖），≤−24 Savaux 王，
  ≤−25 BCJR 软读出捞回；OTA 每帧 κ 常数 → 硬读出为默认。
- 机制查新：无人对分数 bin 不确定性做跨符号边际化 + 软证据进 FEC；
  "marginalize vs estimate"与半 bin 命题均无人占。
- 主表已含 PLAIN/UniChirp 背景板（−20dB 出局）。
- runner v2：全矩阵 256s（12.9×），JSONL 断点续跑。

## 6. 工作队列（依赖排序）

1. **修 DeRa port**：需 PDF §3.3-4.4 公式页（Eqs.10-23；pdftotext 已损）。
   修好前不进战表（不坑 baseline 红线）。
2. **mirror 对等实验**：TREL 与 BCJR 都带镜像仲裁再打一次，排除仲裁不对称。
3. **oracle vs real sync gap**：est-compensate 链在真实同步残差下的损失
   vs marginalize 的损失——新故事的第一个数字（PC-Ridge 链在此复活）。
4. 深端 K-scaling 边界章（选项②的探究落点）。
5. 投稿前 gap 清单：SF7/SF12、≥10× OTA 包 + 多种子 + CI、Ibi 定位、复杂度表。

## 7. 纪律条款（本项目学术红线，违反即作废）

1. **不坑 baseline**：任何对比方法未修好/未对齐输入前不入表（v6 教训：
   只喂整数 CFO 导致 Savaux 冤崩；全先验公平版才是准绳）。
2. **"利好我方设置"= 学术红线**：对比设置必须全先验对齐、同一实现喂所有链。
3. **related work 现场核对原文**，不脑补、不轻信搜索摘要的 DOI。
4. 实验要简单真实：纯 AWGN、真实数据 + 已验证组件，确需偏离先问。
5. 未获信源（PDF/DOI）记入 `信源待下载清单.md`，对话结束前向用户报告。

## 变更记录

- v1（2026-09-29）：初版。两分法灌输 + DeRa 精读 + 围殴战 + 三选项裁决，
  与用户对齐定稿。后续论文定位变化必须更新此文件并记版本。

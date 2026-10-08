# 竞争者与近邻占领扫描（2026-10-05，agent 报告归档）

> 来源：web 检索 agent（当日完成）。所有条目带真实 URL；未能核实项见文末。

## A. DeRa 引用/后续

- **DeRa 本体**：MobiCom 2026（ACM DL 2026-09-26 上线），
  [PDF](https://cse.msu.edu/~caozc/papers/mobicom26-du.pdf) ·
  [ACM](https://dl.acm.org/doi/10.1145/3738036.3699596)。
- **零引用、零复现**：官方 GitHub [空壳](https://github.com/daibiaoxuwu/DeRa)
  （2 commits、一个 gen_signal.py）；UniChirp/LZn/RF-SR/CCSC 参考文献均未引它。
  **我方 port 是全球唯一可运行实现**——时间窗优势。
- 同组（MSU Cao）后续转向：NELoRa++ 技术报告、Kairos（ICNP'26，传输控制）、
  Morph/ChirpTransformer（[arXiv 2507.22851](https://arxiv.org/abs/2507.22851)，
  −28.8dB，改发射机+DNN）。

## B. 2025-26 顶会扫描（与本题相关者）

| Paper | Venue | 机制一句话 | 重叠 |
|---|---|---|---|
| [UniChirp](https://cis.temple.edu/~wu/research/publications/Publication_files/1571232589-paper.pdf) | SECON'26 | CFO/TO/SFO 解析相位模型+前导外推，**过采样双峰相位补偿相干相加**，−17.3dB，+1.3dB | **占方向②核心组件**，须正面区分 |
| [B²LoRa](https://arxiv.org/abs/2505.24140) | MobiCom'25 | 重传副本盲对齐+相干合并（卫星 IoT） | 相邻（跨副本非机内） |
| LoRaSeek | MobiCom'25 | 层级特征神经去噪 | 相邻（神经派） |
| [LZn](https://arxiv.org/abs/2604.27672) | arXiv 2026-04 | 前导谱交集+(δ,f) 分数网格 → **多假设同步前端**；明确非相干、单假设交接 | **半占方向①前端** |
| [FCLoRa](https://dl.acm.org/doi/10.1145/3774326) | TOSN'26 | 前导多级相干积累+多网关相干合并（海面） | 半占（空间维度） |
| RF Super Resolution | MobiSys'26 | 深度学习空间增强 | 相邻 |
| CALoRa | IoT-J'26 | chirp 自注意力前导检测 | 相邻 |
| [PreCo](https://ieeexplore.ieee.org/document/10966705) | WoWMoM'25 | 预计算整包相关 | 相邻 |
| [CCSC](https://ieeexplore.ieee.org/abstract/document/11579205) | SECON'26 | 多网关符号级码字修复 | 无关 |
| NSDI'25/26、SIGCOMM'25 | — | 无 LoRa 弱信号接收机论文 | — |

## C. 谱系

- UniChirp 谱系：Ostinato（ICNP'22，过采样双峰穷举对齐）→ UniChirp（解析
  模型替代穷举）。2026 年同谱系新工作：除 UniChirp 外未找到。
- LoRaTrimmer：被引 10-23，全部为 related-work 级引用/对比基线，
  **无人做直接扩展后续**。

## D. 占领判定

- **① 多候选同步 × 相干解码联合：半占**（壳 = DeRa 本身；空位 = 候选
  选择准则的软/GLRT 度量、候选×分数 bin 联合、真实 OTA）。
- **② 过采样相干 × 分数 bin：半占且被啃最狠**（UniChirp 占组件 + DeRa
  含 scalloping 相位；空位 = κ 作为格点假设的 GLRT 读出统一表述）。
- **③ 同步不确定性作软信息贯穿 payload：半占但表述层最干净**（DeRa
  Stage-3 CRC 硬枚举是最简形式；无人做假设加权/不确定性传播到码字级）。

**最强空位组合 = 候选集软保留 × 分数 bin κ 格 GLRT 读出 × UniChirp 式
双峰合并统一 + 真实 OTA——没有任何一家同时具备。**

## E. 未核实清单

DeRa 被引数（S2 429/OpenAlex 限流/GS 反爬；旁证支持零引用）；LoRaSeek/
MoLoRa/RF-SR/CALoRa/INFOCOM'25 直链（ACM/IEEE 反爬，仅标题级）；
PreCo 机制细节；UniChirp 疑有 TMC 期刊版；清华 TNS 组 2025-10 录用
MobiCom'25 标题未公开；"Versatile LoRa Encoding" SIGCOMM 归属疑误；
FCLoRa −35dB 是检测率 40% 非解码下限（勿误引）。

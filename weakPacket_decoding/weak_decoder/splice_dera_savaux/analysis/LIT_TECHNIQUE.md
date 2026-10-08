# 技术先占扫描：分支相干 × 分数 bin × 序列级读出（2026-10-05，agent 报告归档）

## A. Savaux 谱系全表（b<>com 时期 8 篇 LoRa，此后转 AFDM/6G）

- **#8 解调论文**：*A Low-Complexity Demodulation for Oversampled LoRa
  Signal*，**IJMNDI 10(3):148, 2022**（DOI 10.1504/IJMNDI.2022.126452；
  [TechRxiv](https://www.techrxiv.org/doi/10.36227/techrxiv.16657063) ·
  [HAL](https://hal.science/hal-03970220v1)）。
  **⚠️ 无 TIM'22 版本**——baseline 标注 "TIM'22" 是记忆偏差，引用一律用
  IJMNDI'22 + 下条 IoT-J'22 两个 DOI。
- **#7 同步论文**：*On Time-Frequency Synchronization in LoRa System:
  From Analysis to Near-Optimal Algorithm*，IEEE **IoT-J** 9(12):10200,
  2022（DOI 10.1109/JIOT.2021.3120789；[IEEE](https://ieeexplore.ieee.org/document/9577227)）。
  前导上时延+频偏**联合 ML**（代价函数非凹、近优算法恢复凹性）；证据显示
  定义在过采样信号上（亚样本级分辨）——**但与 #8 解调从未联合发表**，
  "同步×过采样解调联合"是 Savaux 自己没填的缝 = 本线拼接战填的缝。

## B. 分数 bin / 过采样相干 / GLRT 先占判定：**半占（组件全有、组合无人做）**

最接近者排序：
1. **Savaux IJMNDI'22**：R 支路 DFT 相干合并（Eq.37 底座）；完美同步假设，
   合并权重只在整数 bin 取值。
2. **Xhonneux et al., IoT-J 2020**（[arXiv:1912.11344](https://arxiv.org/abs/1912.11344)）：
   分数 CFO（相邻前导符号峰相位差）+分数 STO（Jacobsen 型 3-bin 复数插值）
   估计最完整；但"先估→校正→Nyquist argmax"两步式，无支路合并、无解调内
   分数假设检验。
3. **Demeslay et al., Sensors 2022**（[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9611819)）：
   显式建模/估计/校正 CFO_frac+STO_frac（含 0.5 检验）；解调仍非相干 argmax。
4. **LZn arXiv 2026-04**：分数 (δ,f) 假设网格+逐假设 FFT+Zoom-FFT(8×) 细化
   ——精神上最接近 κ 网格，**但止步帧同步**，解调回常规 argmax。
5. ⚠️ **必查**：*Adaptive frequency-resolution receiver design for
   Doppler-impacted LoRa-based satellite IoT*, Computer Networks 2026
   （[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0140366426002501)，
   403）——标题最撞，投稿前人工核对。
- ELRF/FELoRa/Choeropsis：公开索引零命中（疑内部代号，无法评估）。

**判定**：未发现任何工作把支路合并权重取在 k+κ 上、以 κ 网格做**解调内**
联合假设检验（GLRT）——所有人都是"先估分数偏移→校正→argmax"或"完美
同步整数合并"。**（注：本线 validate_v2 已实证整数谱重权式 κ 格修不了
时延型混跳——空位虽空，但正确的实现是时延轴 fd + 序列级读出，不是
频域重权。）**

## C. 序列级 κ/相位跟踪读出：**空占**

- LoRa 内无 Viterbi/BCJR 跟踪分数偏移/公共相干的任何工作（多路检索零命中）。
- 邻居（维度不同）：Sym-FEC（TMC'25，符号级 FEC）、CEC-LoRa/PreCo
  （[Szafranski 组](https://www.areinhardt.de)，无码纠错/前导上下文 ~3.1dB）、
  MoLoRa（跨重复包亚样本相干合并）。
- 教程（[arXiv:2310.10503](https://arxiv.org/abs/2310.10503)）明确
  ±20ppm → ~0.16 样本/符号累积漂移而"同步研究有限"——动机叙事现成。

## D. 非 LoRa 同构参考（related work 用）

1. van de Beek et al., **ML 时频联合估计 OFDM**, IEEE TSP 1997。
2. Raheli/Polydoros/Tzou, **Per-Survivor Processing**, IEEE TCom 1995
   （[IEEE 380054](https://ieeexplore.ieee.org/document/380054)）——
   序列级 κ 读出的方法论模板。
3. Gasior & Gonzalez, FFT 分数 bin 谱插值, IEEE TNS 2004。
4. FMCW 亚 bin 相位测距两篇（Q. Guoqing 2004；Kim et al., Sensors 2022）。

## E. 结论

- Eq.37 支路合并：已占（Savaux）——底座引用。
- × 分数 bin κ 格 GLRT 读出：**空占**（引 Xhonneux/Demeslay/LZn 划界）。
- × 序列级跟踪读出：**空占**（引 PSP 作方法论）。
- 引用勘误三连：无 TIM'22（用 IJMNDI'22）；Savaux 同步 = IoT-J'22；
  同步×过采样解调联合 = 未有人做（Savaux 本人也没做）。

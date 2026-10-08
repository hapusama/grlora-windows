# paper/ — 对比方法源文档库

> 本目录存放**对比方法的原始 PDF**。两族地图与战表定位的权威版本见
> [`../COGNITION.md`](../COGNITION.md)；本目录只管出处、归类与现状。
> 纪律：引用任何一篇的机制细节时，以本目录 PDF 原文为准——不靠记忆、
> 不靠搜索摘要（COGNITION §7 条款 3）。

| 文件 | 论文 | 出处 | 两族归类 | 机制一句话 | 我方战表状态 |
|---|---|---|---|---|---|
| `DeRa_MobiCom26.pdf` | DeRa: Deep-Range LPWAN IoT via Ultra-sensitive Signal Detection（Du/Lin/Cao, MSU） | MobiCom'26, DOI 10.1145/3795866.3844157 | 族2·解析（相干 est-compensate）+ detection 主轴 | CFO 相位闭式模型 → 相干 preamble 累积检测 + 解析补偿（时漂/频移/泄漏）+ CRC 引导迭代解码器（两段窗相干重组合） | **对照组 + 主表缺列**；port WIP 挂起（native SER 0.14），Eqs.10-23 待核对——本 PDF 已可直读，解冻路径打通 |
| `LoRaTrimmer_MobiCom24.pdf` | LoRaTrimmer: Optimal Energy Condensation with Chirp Trimming（Du et al.） | MobiCom'24, DOI 10.1145/3636534.3690681 | 族2·解析 | trim FFT 感知范围切附加噪声 + 概率建模 \|X₁\|²+\|X₂\|² 非相干合并相位跳变能量 | **已进战表**（TRIMMER 列）；TREL-5 对其 +2.5~3dB（三次独立种子） |
| `UniChirp_2026.pdf` | UniChirp: Unwrapping In-Chirp Phase Misalignment | 2026（项目记忆记 SECON'26，正式版待核对） | 族2·解析 | 闭式相位跳变模型（φ_mis[n]=−2π·BW·n·SFO/Fs−2π·BW·TO+φ₀）+ preamble 线性外推 + 双峰相干融合 | **已进主表**（UNICHIRP 列，−20dB 出局背景板）；与 baselines/unichirp/ 存量版 diff 待做 |
| `NELoRa_SenSys21.pdf` | NELoRa: Towards Ultra-low SNR LoRa Communication with Neural-enhanced Demodulation（Li et al., MSU/THU） | SenSys'21, DOI 10.1145/3485730.3485928 | 族2·ML（学习式 est-compensate） | dual-channel 谱图 + mask DNN filter + DNN decoder（利用 CSS 有限码空间过拟合）；1.84~2.35dB over dechirp | 未进战表；ML 支线必引（LoRaTrimmer/GLoRiPHY 的共同 baseline） |
| `GLoRiPHY_SenSys24.pdf` | GLoRiPHY: Channel-Aware Denoising of LoRaPHY Signals（Sabharwal et al., NUS） | SenSys'24, DOI 10.1145/3666025.3699354 | 族2·ML | conformer 生成式信道感知去噪（preamble 估 CIR）→ 标准解调；SER 2.85× lower than NELoRa | 未进战表；related work 用。⚠️ 自述仿真数据不含 CFO/TO（§7 Limitation），与我们全先验口径不同，不可直接同表 |

## 待办钩子

1. **DeRa port 公式核对**（工作队列①，见 `data/experiments/dera_battle_20260929/RESULTS.md`
   挂起说明）：§3.3 Eqs.10-11（两段窗模型）+ §4.3 Eqs.20-23（payload 相位补偿）
   + §4.2（相干验证/20 候选）+ Appendix C Algorithm 1。原 pdftotext 损毁版已被
   本 PDF 绕过，可直接读页核对。
2. **UniChirp 版本 diff**：本目录 PDF 与 `baselines/unichirp/` 存量投稿版是否
   同一版本；正式落地页核对未做（见 proj 根 `信源待下载清单.md`）。
3. **NELoRa / GLoRiPHY**：不进战表；写 related work 的 ML 支线（学习式
   est-compensate）时引用，注意两者输入假设与我们公平口径的差异。

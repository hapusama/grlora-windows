# weakPacket_decoding

真实 LoRa 弱包的离线检测、同步与过采样解调研究目录。**本 README 是结构总入口**；
逐项导览见 `doc/PROJECT_MAP_20260928.md`，最新交接见 `HANDOFF_20260928.md`。

当前两条主线（2026-09-28）：

1. **弱包接收主链**（`weak_decoder/os_lora/system/`）：ambiguity-ridge 候选列表
   → Savaux 似然 → bounded soft-Hamming → CRC 仲裁（decode-to-finish）。
2. **细网格 GLRT 解调线**（2026-09-28 新，见 `doc/paper_method_fine_grid_demod_20260928.md`
   与 `data/experiments/ablation_battle_20260928/`、`symfec_pk_20260928/`）：
   OS 去斜 + 补零细网格 + 跨符号 κ-格 Viterti，已与 baselines 全家四件套对比全胜。

## 目录一图流（✅活跃 / 🗄退役收纳 / 📜历史台账）

```text
weakPacket_decoding/
├── weak_decoder/                 代码包（python 用 conda envs/gr-lora，系统 3.6 不行）
│   ├── chirp.py                ✅ 全仓 chirp/FFT/bin 约定（单一事实源）
│   ├── branch4_profile.py      ✅ STM32 固定帧实验参数（Branch4，见下）
│   ├── run_iq_frontend.py      ✅ raw IQ 前端入口
│   ├── synchronization/        ✅ 检测/帧定位/CFO-STO-SFO 同步
│   ├── decoding/               ✅ LoRa 帧编解码
│   │   ├── payload_codec.py      白化/Hamming/交织/Gray/CRC 编解码
│   │   ├── header_first_demod.py header 先行解调
│   │   └── legacy/             🗄 退役解调探索（adaptive_path/structured_path/
│   │       timing_path/alias_trim/robust_sparse/phase_templates；
│   │       顶层 __init__ 保留再导出，旧 import 兼容）
│   ├── os_lora/                ✅ 主系统（system/ 算法、experiments/ 评估、doc/ 文档）
│   ├── baselines/              ✅ 发表方法忠实复现（对比方法库；四件套已全 PK，
│   │                              集成约定见 PROJECT_MAP §1）
│   ├── rf_super_resolution/    🗄 实验性 RF 前端超分辨（仅单测引用，未接主链）
│   └── tests/                  ✅ 单元测试
├── data/
│   ├── experiments/            📜 实验台账（一实验一目录；近期关键：
│   │                              multicopy_*、ambiguity_ridge_*、
│   │                              oversampling_accounting / enhanced_demod_chain /
│   │                              ablation_battle / symfec_pk _20260928）
│   ├── frontend/               前端 sync CSV 输出
│   └── smoke_tmp/              🗄 冒烟测试输出（原根目录 tmp/）
├── scripts/                    实验脚本（含 experiments/ 历史 runner）
├── doc/                        PROJECT_MAP、论文方法稿、phase_map、history/
├── notes/                      个人笔记与论文 PDF（design/plans/handoffs）
├── noisy_iq/                   加噪 IQ 数据
├── USRP_collector/             USRP 采集 GRC/脚本（含 collect_usrp_iq.py）
└── HANDOFF_*.md                交接时间线（最新 20260928）
```

## 速查：我想 X，去哪 / 放哪

- **跑主链解码**：`weak_decoder/os_lora/system/`（入口 `decode_savaux_sync_candidate`
  / `arbitrate_sync_list_with_crc`；模块职责见 `weak_decoder/os_lora/README.md`）
- **复现 09-28 解调线与 battle**：`data/experiments/*_20260928/`（先读各 RESULTS.md）
- **加新实验**：`data/experiments/<name>_YYYYMMDD/` 一目录（脚本+results+RESULTS.md），
  独立分层、可整目录删除、不污染 system/
- **找/加对比基线**：`weak_decoder/baselines/`（bin→值映射常数 512、UniChirp 需
  none 模式 + guard 符号，坑清单见 PROJECT_MAP §1）
- **历史方法考据**：`weak_decoder/decoding/legacy/`（代码）、`doc/history/`、`notes/`

---

## Branch4 固定帧参数（硬件台账，勿删）

参数来自：

```text
LoraSTMacL1_2019.03.28_修改main函数_实现classA_通用版(Branch4)/apps/main.c
```

```text
RF              487.7 MHz
SF              10
BW              125 kHz
CR              4/7
preamble        32 symbols
sync word       0x34
header          explicit
PHY CRC         enabled
App payload     20 bytes
PHY payload     33 bytes
FCnt            1（固定）
TX power        2 dBm
TX period       3 s
IQ sample rate  500 ksample/s（OSR=4）
```

生成推荐文件名：

```powershell
python -m weak_decoder.branch4_profile --condition high_snr --run 1
```

```text
high_snr/sf10_bw125_fs500_pre32_sw34_r001.bin
```

采集条件建议：`high_snr`（建 FFT-bin ground truth）/ `low_snr` / `noise_only`
（估计有色噪声协方差）/ `interference`。频率、CR、payload、TX power、RX gain 等
记录在 `USRP_collector/data/branch4_fixed/README.md` 与 `.bin.json`，勿只依赖文件名。

## 采集 IQ（RadioConda）

```powershell
Set-Location "D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding"

python USRP_collector\collect_usrp_iq.py `
  --output USRP_collector\data\branch4_fixed\high_snr\sf10_bw125_fs500_pre32_sw34_r001.bin `
  --duration 120 `
  --gain 20 `
  --device-args "serial=2603160"
```

## 前导码检测与 frame sync（gr-lora 环境）

```powershell
python -m weak_decoder.run_iq_frontend `
  --input "USRP_collector\data\branch4_fixed\high_snr\sf10_bw125_fs500_pre32_sw34_r001.bin" `
  --max-packets 20
```

默认输出 `data/frontend/<capture>_sync.csv`，关键字段：
`grlora_fine_payload_start_sample`、`grlora_cfo_int_est / grlora_cfo_frac_est`、
`grlora_payload_sto_frac_est`、`grlora_sfo_hat`、`grlora_branch_sample_phases`、
`grlora_branch_valid`——它们定义 demod 的输入边界与各 branch 同步状态。

## 传统 FFT 参考链

```powershell
python scripts\run_header_first_demod.py `
  --input "USRP_collector\data\<capture>.bin" `
  --sync-csv "data\frontend\<capture>_sync.csv" `
  --output "data\frontend\<capture>_symbols.csv" `
  --frames-output "data\frontend\<capture>_frames.csv" `
  --sf 10 --bw 125000 --samp-rate 500000
```

## Baselines 说明

`weak_decoder/baselines/`（savaux_oversampled / loratrimmer / symfec / unichirp）
保留用于论文对比、负结果与消融，不混入 GLS 权重估计本身。退役自研探索在
`weak_decoder/decoding/legacy/`（2026-09-28 迁入，顶层再导出保持旧 import 兼容）。

## 约定

- 实验目录即台账：可复现、可整目录删除；结果与勘误写回目录内 RESULTS.md。
- `chirp.py` 是 bin/值/符号约定的单一事实源，新解调先对齐它。
- 结构改动以 单测 + `symfec_pk_20260928` selftest 全绿为准（2026-09-28 重构已验证）。

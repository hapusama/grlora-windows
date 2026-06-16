#!/usr/bin/env python3
"""完整的盲解码实验流程。

此脚本协调整个实验：
1. 候选集覆盖率分析（验证 GT bin 在候选集内）
2. 盲解码器评估（block-wise beam search）
3. 结果可视化

使用方法：
    python run_full_blind_decode_experiment.py \\
        --clean-iq data/clean.bin \\
        --gt-csv data/header_first_symbols.csv \\
        --output-dir results/blind_decode_exp \\
        --target-snr-db -10 -15 -20

输出：
    - 候选集覆盖率统计和图表
    - 盲解码器性能评估
    - 对比图表（argmax vs blind decoder）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete blind decode experiment"
    )
    parser.add_argument(
        "--clean-iq",
        type=Path,
        required=True,
        help="Clean complex64 IQ .bin file",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        required=True,
        help="Header-first symbol CSV with GT bins",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for all results",
    )
    parser.add_argument(
        "--target-snr-db",
        type=float,
        nargs="+",
        default=[-10.0, -15.0, -20.0],
        help="Target SNR values in dB (default: -10 -15 -20)",
    )
    parser.add_argument(
        "--packet",
        type=int,
        help="Optional: only process specified packet_index",
    )

    # Candidate set parameters
    parser.add_argument(
        "--amplitude-topk",
        type=int,
        default=8,
        help="Amplitude topK (default: 8)",
    )
    parser.add_argument(
        "--argmax-window",
        type=int,
        default=4,
        help="Argmax ±W window (default: 4)",
    )
    parser.add_argument(
        "--phase-topk",
        type=int,
        default=8,
        help="Phase-residual topK (default: 8)",
    )
    parser.add_argument(
        "--combined-topk",
        type=int,
        default=16,
        help="Combined topK (default: 16)",
    )

    # Decoder parameters
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold (default: 0.6)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=10,
        help="Beam search width (default: 10)",
    )
    parser.add_argument(
        "--max-block-candidates",
        type=int,
        default=100,
        help="Max candidates per block (default: 100)",
    )
    parser.add_argument(
        "--phase-weight",
        type=float,
        default=0.85,
        help="Phase weight (default: 0.85)",
    )

    # Other
    parser.add_argument(
        "--skip-noise-generation",
        action="store_true",
        help="Skip noise generation if noisy IQ already exists",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Plot DPI (default: 150)",
    )

    return parser.parse_args()


def run_command(cmd: list[str], description: str) -> int:
    """Run a subprocess command and report status."""
    print(f"\n{'='*70}")
    print(f"[Step] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(str(c) for c in cmd)}\n")

    result = subprocess.run(cmd, cwd=str(WEAK_ROOT))

    if result.returncode != 0:
        print(f"\n[Error] Command failed with exit code {result.returncode}")
        return result.returncode

    print(f"\n[Success] {description} completed")
    return 0


def main() -> int:
    args = parse_args()

    clean_iq = args.clean_iq.resolve()
    gt_csv = args.gt_csv.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration
    config = {
        "clean_iq": str(clean_iq),
        "gt_csv": str(gt_csv),
        "target_snr_db": args.target_snr_db,
        "packet_filter": args.packet,
        "parameters": {
            "amplitude_topk": args.amplitude_topk,
            "argmax_window": args.argmax_window,
            "phase_topk": args.phase_topk,
            "combined_topk": args.combined_topk,
            "confidence_threshold": args.confidence_threshold,
            "beam_width": args.beam_width,
            "max_block_candidates": args.max_block_candidates,
            "phase_weight": args.phase_weight,
        },
    }

    config_path = output_dir / "experiment_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Experiment configuration saved to: {config_path}")

    # Step 1: Generate noisy IQ (if needed)
    for snr_db in args.target_snr_db:
        snr_label = f"snr_m{abs(int(snr_db)):02d}dB" if snr_db < 0 else f"snr_p{int(snr_db):02d}dB"
        noisy_iq = output_dir / "noisy_iq" / f"{clean_iq.stem}_{snr_label}.bin"

        if args.skip_noise_generation and noisy_iq.exists():
            print(f"\nSkipping noise generation for {snr_label} (file exists)")
            continue

        cmd = [
            sys.executable,
            str(WEAK_ROOT / "scripts" / "experiments" / "run_low_snr_gt_bin_experiment.py"),
            "-i", str(clean_iq),
            "-g", str(gt_csv),
            "-o", str(output_dir / "noisy_iq"),
            "--target-snr-db", str(snr_db),
            "--no-write-noisy-bin",  # We only need the IQ, not features yet
        ]

        if args.packet is not None:
            cmd.extend(["--packet", str(args.packet)])

        ret = run_command(cmd, f"Generate noisy IQ at {snr_db} dB")
        if ret != 0:
            return ret

    # Step 2: Candidate coverage analysis for each SNR
    for snr_db in args.target_snr_db:
        snr_label = f"snr_m{abs(int(snr_db)):02d}dB" if snr_db < 0 else f"snr_p{int(snr_db):02d}dB"
        noisy_iq = output_dir / "noisy_iq" / f"{clean_iq.stem}_{snr_label}.bin"

        if not noisy_iq.exists():
            print(f"\n[Warning] Noisy IQ not found: {noisy_iq}, skipping")
            continue

        coverage_csv = output_dir / "coverage" / snr_label / "candidate_coverage.csv"
        coverage_json = output_dir / "coverage" / snr_label / "summary.json"

        cmd = [
            sys.executable,
            str(WEAK_ROOT / "scripts" / "experiments" / "analyze_gt_bin_candidate_coverage.py"),
            "-i", str(noisy_iq),
            "-g", str(gt_csv),
            "-o", str(coverage_csv),
            "--summary-json", str(coverage_json),
            "--amplitude-topk", str(args.amplitude_topk), "16", "32", "64",
            "--argmax-window", "1", "2", str(args.argmax_window), "8",
            "--phase-topk", "4", str(args.phase_topk), "16", "32",
            "--combined-topk", "4", "8", str(args.combined_topk), "32", "64",
            "--phase-weight", str(args.phase_weight),
        ]

        if args.packet is not None:
            cmd.extend(["--packet", str(args.packet)])

        ret = run_command(cmd, f"Candidate coverage analysis at {snr_db} dB")
        if ret != 0:
            return ret

        # Visualize coverage
        viz_cmd = [
            sys.executable,
            str(WEAK_ROOT / "scripts" / "experiments" / "visualize_candidate_coverage.py"),
            "-i", str(coverage_csv),
            "-o", str(output_dir / "coverage" / snr_label / "plots"),
            "--dpi", str(args.dpi),
        ]

        ret = run_command(viz_cmd, f"Visualize candidate coverage at {snr_db} dB")
        if ret != 0:
            print(f"[Warning] Visualization failed, continuing...")

    # Step 3: Blind decoder evaluation for each SNR
    for snr_db in args.target_snr_db:
        snr_label = f"snr_m{abs(int(snr_db)):02d}dB" if snr_db < 0 else f"snr_p{int(snr_db):02d}dB"
        noisy_iq = output_dir / "noisy_iq" / f"{clean_iq.stem}_{snr_label}.bin"

        if not noisy_iq.exists():
            print(f"\n[Warning] Noisy IQ not found: {noisy_iq}, skipping")
            continue

        decode_csv = output_dir / "decoder" / snr_label / "results.csv"
        decode_json = output_dir / "decoder" / snr_label / "summary.json"

        cmd = [
            sys.executable,
            str(WEAK_ROOT / "scripts" / "experiments" / "run_blind_decoder_evaluation.py"),
            "-i", str(noisy_iq),
            "-g", str(gt_csv),
            "-o", str(decode_csv),
            "--summary-json", str(decode_json),
            "--confidence-threshold", str(args.confidence_threshold),
            "--amplitude-topk", str(args.amplitude_topk),
            "--argmax-window", str(args.argmax_window),
            "--phase-topk", str(args.phase_topk),
            "--combined-topk", str(args.combined_topk),
            "--beam-width", str(args.beam_width),
            "--max-block-candidates", str(args.max_block_candidates),
            "--phase-weight", str(args.phase_weight),
        ]

        if args.packet is not None:
            cmd.extend(["--packet", str(args.packet)])

        ret = run_command(cmd, f"Blind decoder evaluation at {snr_db} dB")
        if ret != 0:
            return ret

    # Step 4: Generate comparison summary
    print(f"\n{'='*70}")
    print("[Summary] Experiment completed")
    print(f"{'='*70}\n")

    print(f"Results saved to: {output_dir}")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    experiment_config.json       # 实验配置")
    print(f"    noisy_iq/                    # 加噪 IQ 文件")
    print(f"    coverage/                    # 候选集覆盖率分析")
    print(f"      <snr_label>/")
    print(f"        candidate_coverage.csv   # 逐符号覆盖率")
    print(f"        summary.json             # 汇总统计")
    print(f"        plots/                   # 可视化图表")
    print(f"    decoder/                     # 盲解码器评估")
    print(f"      <snr_label>/")
    print(f"        results.csv              # 逐包结果")
    print(f"        summary.json             # 汇总统计")

    print(f"\n关键发现：")
    for snr_db in args.target_snr_db:
        snr_label = f"snr_m{abs(int(snr_db)):02d}dB" if snr_db < 0 else f"snr_p{int(snr_db):02d}dB"

        # Load coverage summary
        coverage_json = output_dir / "coverage" / snr_label / "summary.json"
        if coverage_json.exists():
            with coverage_json.open("r", encoding="utf-8") as f:
                coverage = json.load(f)
            print(f"\n  [{snr_db} dB] 候选集覆盖率:")
            print(f"    Argmax 准确率: {coverage.get('argmax_accuracy', 0):.3f}")
            rates = coverage.get('coverage_rates', {})
            print(f"    Amp top{args.amplitude_topk}: {rates.get(f'in_amp_top{args.amplitude_topk}', 0):.3f}")
            print(f"    Argmax ±{args.argmax_window}: {rates.get(f'in_argmax_window_{args.argmax_window}', 0):.3f}")
            print(f"    Phase top{args.phase_topk}: {rates.get(f'in_phase_top{args.phase_topk}', 0):.3f}")
            print(f"    Combined top{args.combined_topk}: {rates.get(f'in_combined_top{args.combined_topk}', 0):.3f}")

        # Load decoder summary
        decode_json = output_dir / "decoder" / snr_label / "summary.json"
        if decode_json.exists():
            with decode_json.open("r", encoding="utf-8") as f:
                decode = json.load(f)
            print(f"\n  [{snr_db} dB] 解码性能:")
            print(f"    Argmax SER: {decode.get('argmax_mean_ser', 1):.3f}")
            print(f"    Blind decoder SER: {decode.get('blind_mean_ser', 1):.3f}")
            improvement = decode.get('argmax_mean_ser', 1) - decode.get('blind_mean_ser', 1)
            print(f"    SER 改进: {improvement:.3f}")
            print(f"    CRC-valid 率: {decode.get('crc_valid_rate', 0):.3f}")
            print(f"    完美解码率: {decode.get('perfect_decode_rate', 0):.3f}")
            print(f"    平均解码时间: {decode.get('avg_decode_time_ms', 0):.1f} ms")

    print(f"\n结论：")
    print(f"  若盲解码器在不使用 payload template、跨包 joint、计数器先验的")
    print(f"  情况下，SER 显著低于 argmax 或 CRC-valid 率提升，则说明")
    print(f"  Phase-MAP 相位/幅度证据 + LoRa code constraint 对单包弱包")
    print(f"  盲解码有真实价值。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

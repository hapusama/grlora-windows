# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""Thin entry point for the modular preamble FFT exporter."""
# 提取packet_features 指令
# conda run --no-capture-output -n gr-lora python gr-lora_sdr/examples/lora_file_preamble_fft.py --all-bin --input-dir gr-lora_sdr/data/USRP_IQ/lab1_sf11_TP2 --sf 11 --preamble-len 16 --samp-rate 500000 --center-freq 487.7e6 --sync-word 0x34 --crc-mode 0 --no-throttle --no-print-header --print-payload none

# 只画前导码 raw dechirp FFT 图，不覆盖 CSV/NPZ：
# conda run --no-capture-output -n gr-lora python gr-lora_sdr/examples/lora_file_preamble_fft.py --all-bin --input-dir gr-lora_sdr/data/USRP_IQ/lab1_sf11_TP2 --sf 11 --preamble-len 16 --samp-rate 500000 --center-freq 487.7e6 --sync-word 0x34 --crc-mode 0 --no-throttle --no-print-header --print-payload none --position-ids 8,9,10,11,12,13 --plot-only --plot-dechirp-fft --plot-output-dir gr-lora_sdr/data/USRP_IQ/lab1_sf11_TP2/preamble_fft_plots --plot-db-floor -60 --plot-zoom-bins 8

from preamble_fft_tools.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

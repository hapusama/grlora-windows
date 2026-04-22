#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Lora File Rx
# Description: 基于 gr-lora_sdr 的离线 IQ 文件接收机，用于解析 USRP 采集的 LoRa 基带数据
# Author: Tapparel Joachim@EPFL,TCL (modified for file source)
#
# 运行前请确保已安装 gr-lora_sdr，并使用 USRP 采集了 LoRa 信号的基带 IQ 数据文件（.fc32 或 .cfile 格式）。
# python .\gr-lora_sdr\examples\lora_file_RX.py -f .\gr-lora_sdr\data\USRP_IQ\0_0_0_10_6_16.bin --sf 10 --bw 125000 --samp-rate 500000 --cr 1 --center-freq 487.7e6 --sync-word 0x34 --preamble-len 16 --ldro-mode 2 --crc-mode 0 --plot-preamble --preamble-plot-max 0


from gnuradio import gr
from gnuradio import blocks
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from pathlib import Path
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import gnuradio.lora_sdr as lora_sdr
import numpy as np
import pmt
import threading


class phy_header_spectrogram_sink(gr.basic_block):
    """
    接收 frame_sync 输出的原始 IQ 索引范围，并保存 PHY header 频谱图。
    """

    def __init__(self, input_file, output_dir, max_plots=3, dpi=150):
        gr.basic_block.__init__(
            self,
            name="phy_header_spectrogram_sink",
            in_sig=None,
            out_sig=None
        )
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_plots = max_plots
        self.dpi = dpi
        self.plot_count = 0
        self.queued_count = 0
        self.pending_plots = []
        self.pending_lock = threading.Lock()
        self.message_port_register_in(pmt.intern("preamble"))
        self.set_msg_handler(pmt.intern("preamble"), self.handle_preamble)

    def _dict_value(self, msg, key, default=None):
        value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
        if pmt.is_null(value):
            return default
        return pmt.to_python(value)

    def _pmt_iq_to_numpy(self, iq_pmt):
        if pmt.is_c32vector(iq_pmt):
            return np.asarray(pmt.c32vector_elements(iq_pmt), dtype=np.complex64)
        if pmt.is_blob(iq_pmt):
            raw = bytes(pmt.blob_data(iq_pmt))
        elif pmt.is_u8vector(iq_pmt):
            raw = bytes(pmt.u8vector_elements(iq_pmt))
        else:
            raise TypeError(f"Unsupported PHY header IQ PMT type: {pmt.write_string(iq_pmt)}")
        return np.frombuffer(raw, dtype=np.complex64)

    def handle_preamble(self, msg):
        if self.max_plots > 0 and self.queued_count >= self.max_plots:
            return
        if not pmt.is_dict(msg):
            print("[phy_header_plot] ignored non-dict message")
            return

        iq_pmt = pmt.dict_ref(msg, pmt.intern("phy_header_iq"), pmt.PMT_NIL)
        if pmt.is_null(iq_pmt):
            iq_pmt = pmt.dict_ref(msg, pmt.intern("preamble_iq"), pmt.PMT_NIL)
        start_sample = self._dict_value(msg, "start_sample", None)
        end_sample = self._dict_value(msg, "end_sample", None)
        if pmt.is_null(iq_pmt) and (start_sample is None or end_sample is None):
            print("[phy_header_plot] phy_header_iq and sample range both missing")
            return

        try:
            iq = None if pmt.is_null(iq_pmt) else self._pmt_iq_to_numpy(iq_pmt).copy()
            sf = int(self._dict_value(msg, "sf", 7))
            default_sps = 1 << sf
            start_sample = None if start_sample is None else int(start_sample)
            end_sample = None if end_sample is None else int(end_sample)
            n_samples = int(self._dict_value(
                msg,
                "n_samples",
                (end_sample - start_sample) if start_sample is not None and end_sample is not None else (iq.size if iq is not None else 0)
            ))
            meta = {
                "frame_count": self._dict_value(msg, "frame_count", self.queued_count + 1),
                "sf": sf,
                "bw": float(self._dict_value(msg, "bw", 125000)),
                "sample_rate": float(self._dict_value(msg, "sample_rate", self._dict_value(msg, "bw", 125000))),
                "samples_per_symbol": int(self._dict_value(msg, "samples_per_symbol", default_sps)),
                "n_symbols": float(self._dict_value(msg, "n_symbols", n_samples / default_sps)),
                "start_sample": start_sample,
                "end_sample": end_sample,
                "n_samples": n_samples,
                "preamble_len": self._dict_value(msg, "preamble_len", None),
                "sync_word_symbols": self._dict_value(msg, "sync_word_symbols", None),
                "sfd_symbols": self._dict_value(msg, "sfd_symbols", None),
                "netid1": int(self._dict_value(msg, "netid1", -1)),
                "netid2": int(self._dict_value(msg, "netid2", -1)),
                "snr_db": self._dict_value(msg, "snr_db", None),
                "cfo": self._dict_value(msg, "cfo", None),
                "sto": self._dict_value(msg, "sto", None),
                "sfo": self._dict_value(msg, "sfo", None),
                "header_valid": bool(self._dict_value(msg, "header_valid", False)),
                "source": self._dict_value(msg, "source", ""),
            }
            with self.pending_lock:
                self.pending_plots.append((iq, meta))
                self.queued_count += 1
            print(
                f"[phy_header_plot] queued frame {int(meta['frame_count'])} "
                f"(samples {meta['start_sample']}:{meta['end_sample']}, {meta['n_symbols']:.2f} symbols)"
            )
        except Exception as exc:
            print(f"[phy_header_plot] failed: {exc}")

    def save_all(self):
        with self.pending_lock:
            pending = list(self.pending_plots)
            self.pending_plots.clear()

        for iq, meta in pending:
            try:
                if iq is None:
                    src = np.memmap(self.input_file, dtype=np.complex64, mode="r")
                    start = int(meta["start_sample"])
                    end = int(meta["end_sample"])
                    start = max(0, min(start, src.size))
                    end = max(start, min(end, src.size))
                    iq = np.asarray(src[start:end], dtype=np.complex64).copy()
                    meta = dict(meta)
                    meta["start_sample"] = start
                    meta["end_sample"] = end
                    meta["n_samples"] = iq.size
                    meta["n_symbols"] = iq.size / max(1, int(meta["samples_per_symbol"]))
                out_path = self.output_dir / f"phy_header_frame_{int(meta['frame_count']):03d}.png"
                print(
                    f"[phy_header_plot] saving frame {int(meta['frame_count'])} "
                    f"({meta.get('n_symbols', 0):.2f} symbols, {iq.size} samples)",
                    flush=True
                )
                self.plot_preamble(iq, meta, out_path)
                self.plot_count += 1
                print(f"[phy_header_plot] saved {out_path}")
            except Exception as exc:
                print(f"[phy_header_plot] failed to save frame {meta.get('frame_count', '?')}: {exc}")

    def _viridis_rgb(self, values):
        values = np.clip(values, 0.0, 1.0)
        stops = np.array([
            [0.267, 0.005, 0.329],
            [0.283, 0.141, 0.458],
            [0.254, 0.265, 0.530],
            [0.207, 0.372, 0.553],
            [0.164, 0.471, 0.558],
            [0.128, 0.567, 0.551],
            [0.135, 0.659, 0.518],
            [0.267, 0.749, 0.441],
            [0.478, 0.821, 0.318],
            [0.741, 0.873, 0.150],
            [0.993, 0.906, 0.144],
        ], dtype=np.float32)
        pos = values * (len(stops) - 1)
        idx = np.floor(pos).astype(np.int32)
        idx = np.clip(idx, 0, len(stops) - 2)
        frac = (pos - idx)[..., None]
        rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def plot_preamble(self, iq, meta, out_path):
        bw = float(meta["bw"])
        fs = float(meta.get("sample_rate", bw))
        samples_per_symbol = max(1, int(meta["samples_per_symbol"]))
        if iq.size < 32:
            raise ValueError("PHY header IQ is too short to plot")
        iq = np.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64, copy=False)
        os_factor = max(1, int(round(fs / bw)))
        if os_factor > 1:
            iq = iq[os_factor // 2::os_factor]
            samples_per_symbol = max(1, samples_per_symbol // os_factor)
            fs = bw
        nperseg = min(512, max(64, samples_per_symbol // 8))    # 每个短时FFT窗口的长度，通常取一个 symbol 的 1/8
        nperseg = min(nperseg, iq.size)
        noverlap = int(nperseg * 0.88)
        if noverlap >= nperseg:
            noverlap = nperseg - 1
        hop = max(1, nperseg - noverlap)
        nfft = max(1024, nperseg * 2)
        starts = np.arange(0, iq.size - nperseg + 1, hop)
        if starts.size == 0:
            starts = np.array([0])
            nperseg = iq.size
            nfft = max(1024, nperseg * 2)

        window = np.hanning(nperseg).astype(np.float32)
        frames = np.empty((starts.size, nperseg), dtype=np.complex64)
        for row, start in enumerate(starts):
            frames[row, :] = iq[start:start + nperseg] * window

        spec = np.abs(np.fft.fft(frames, n=nfft, axis=1)).T
        spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
        freqs = np.fft.fftfreq(nfft, d=1.0 / fs)
        times = (starts + nperseg / 2) / fs

        freqs = np.fft.fftshift(freqs)
        spec = np.fft.fftshift(spec, axes=0)
        spec = np.maximum(spec, 1e-15)
        spec_db = 20 * np.log10(spec / np.max(spec)) - 55.0
        spec_db = np.nan_to_num(spec_db, nan=-170.0, posinf=-55.0, neginf=-170.0)
        spec_db = np.clip(spec_db, -170.0, -55.0)

        time_ms = times * 1e3
        freq_khz = freqs / 1e3
        del time_ms, freq_khz

        from PIL import Image, ImageDraw, ImageFont

        norm = (spec_db - (-170.0)) / 115.0
        rgb = self._viridis_rgb(norm)
        rgb = np.flipud(rgb)

        plot_w = max(900, int(12.0 * self.dpi))
        plot_h = max(300, int(3.2 * self.dpi))
        left = 88
        right = 112
        top = 62
        bottom = 64
        img_w = plot_w + left + right
        img_h = plot_h + top + bottom

        image = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        spec_img = Image.fromarray(rgb, mode="RGB").resize((plot_w, plot_h), Image.Resampling.BILINEAR)
        image.paste(spec_img, (left, top))

        symbol_ms = samples_per_symbol / fs * 1e3
        n_symbols = float(meta["n_symbols"] or (iq.size / samples_per_symbol))
        symbol_edges_ms = np.arange(1, int(np.floor(n_symbols)) + 1, dtype=np.float32) * symbol_ms
        symbol_edges_ms = symbol_edges_ms[symbol_edges_ms < iq.size / fs * 1e3]
        if symbol_edges_ms.size <= 256:
            total_ms = iq.size / fs * 1e3
            for edge_ms in symbol_edges_ms:
                x = left + int(round(edge_ms / total_ms * plot_w))
                draw.line([(x, top), (x, top + plot_h)], fill=(255, 255, 255), width=1)

        axis_color = (20, 20, 20)
        draw.rectangle([left, top, left + plot_w, top + plot_h], outline=axis_color, width=1)

        title = "LoRa Preamble + Sync + SFD Spectrogram"
        draw.text((left, 18), title, fill=axis_color, font=font)

        subtitle = []
        if meta["snr_db"] is not None:
            subtitle.append(f"SNR {float(meta['snr_db']):.1f} dB")
        if meta["cfo"] is not None:
            subtitle.append(f"CFO {float(meta['cfo']):.2f} bins")
        if subtitle:
            text = " | ".join(subtitle)
            bbox = draw.textbbox((0, 0), text, font=font)
            draw.text((left + plot_w - (bbox[2] - bbox[0]), 20), text, fill=(50, 50, 50), font=font)

        total_ms = iq.size / fs * 1e3
        xticks = np.linspace(0, total_ms, 6)
        for tick in xticks:
            x = left + int(round(tick / total_ms * plot_w))
            draw.line([(x, top + plot_h), (x, top + plot_h + 5)], fill=axis_color)
            label = f"{tick:.1f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text((x - (bbox[2] - bbox[0]) / 2, top + plot_h + 8), label, fill=axis_color, font=font)

        yticks = np.linspace(-bw / 2e3, bw / 2e3, 5)
        for tick in yticks:
            y = top + plot_h - int(round((tick + bw / 2e3) / (bw / 1e3) * plot_h))
            draw.line([(left - 5, y), (left, y)], fill=axis_color)
            label = f"{tick:.0f}"
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text((left - 10 - (bbox[2] - bbox[0]), y - 5), label, fill=axis_color, font=font)

        draw.text((left + plot_w / 2 - 35, img_h - 30), "Time (ms)", fill=axis_color, font=font)
        draw.text((8, top + plot_h / 2 - 8), "Frequency (kHz)", fill=axis_color, font=font)

        cbar_x = left + plot_w + 32
        cbar_w = 22
        cbar_values = np.linspace(1.0, 0.0, plot_h, dtype=np.float32).reshape(plot_h, 1)
        cbar_rgb = self._viridis_rgb(cbar_values)
        cbar_img = Image.fromarray(cbar_rgb, mode="RGB").resize((cbar_w, plot_h), Image.Resampling.BILINEAR)
        image.paste(cbar_img, (cbar_x, top))
        draw.rectangle([cbar_x, top, cbar_x + cbar_w, top + plot_h], outline=axis_color, width=1)
        for tick in [-60, -80, -100, -120, -140, -160]:
            y = top + int(round(((-55.0 - tick) / 115.0) * plot_h))
            draw.line([(cbar_x + cbar_w, y), (cbar_x + cbar_w + 5, y)], fill=axis_color)
            draw.text((cbar_x + cbar_w + 9, y - 5), f"{tick} dB", fill=axis_color, font=font)

        image.save(out_path)


class lora_file_RX(gr.top_block):
    """
    LoRa 离线接收流程图
    ===================
    将 USRP 采集的复基带 IQ 数据（float32 复数格式，即 .fc32 或 .cfile）
    通过文件源读入，经完整的 gr-lora_sdr 接收链路进行帧同步、解调、
    解格雷、去交织、汉明译码、包头解析、去白化以及 CRC 校验，
    最终在终端输出接收到的数据包内容。
    """

    def __init__(self, args):
        gr.top_block.__init__(self, "Lora File Rx", catch_exceptions=True)

        ##################################################
        # 参数配置：从命令行读取并保存为实例变量
        ##################################################
        self.input_file = args.input_file      # IQ 文件路径
        self.sf = args.sf                      # 扩频因子 (Spreading Factor)
        self.bw = args.bw                      # 带宽 (Hz)
        self.samp_rate = args.samp_rate        # 采样率 (Hz)
        self.cr = args.cr                      # 编码率 (Coding Rate)
        self.pay_len = args.pay_len            # 预设包长（隐式头模式下有效）
        self.has_crc = args.has_crc            # 是否包含 CRC
        self.impl_head = args.impl_head        # 是否使用隐式头
        self.soft_decoding = args.soft_decoding# 是否启用软判决译码
        # 注意：center_freq 在 gr-lora_sdr 中并不用于混频，而仅用于估计采样频率偏移(SFO)。
        # 即使是基带IQ数据，如果来源于真实USRP射频采集，仍建议填入原始的射频中心频率。
        self.center_freq = args.center_freq    # 中心频率 (Hz)，仅用于帧同步内部的 SFO 估计
        self.sync_word = args.sync_word        # 同步字，默认 0x12
        self.ldro_mode = args.ldro_mode        # 低数据率优化模式
        self.preamble_len = args.preamble_len    # 同步触发阈值；绘图导出 preamble_len + sync word + SFD
        self.plot_preamble = args.plot_preamble

        ##################################################
        # 构建 GNU Radio 信号处理模块
        ##################################################

        # --------------------------------------------------
        # 1) 文件源：读取 USRP 采集的复基带 IQ 数据
        # --------------------------------------------------
        # GNU Radio 的 file_source 默认读取二进制流；
        # USRP 采集的 .fc32 / .cfile 文件即为连续的 gr_complex (float32 I + float32 Q)。
        # repeat=False 表示文件读完即结束，不会循环播放。
        self.blocks_file_source_0 = blocks.file_source(
            gr.sizeof_gr_complex * 1,
            self.input_file,
            False,   # repeat
            0,       # offset
            0        # length
        )
        # 为 frame_sync 预留足够大的输出缓冲区，避免 scheduler 报错
        # frame_sync 的 forecast 会请求 os_factor*(2^sf+2) 个样本
        min_buf = int(np.ceil(self.samp_rate / self.bw * (2**self.sf + 2)))
        self.blocks_file_source_0.set_min_output_buffer(min_buf)

        # --------------------------------------------------
        # 2) 节流阀（Throttle）：按 samp_rate 控制数据流速
        # --------------------------------------------------
        # 对于文件源，GNU Radio 会以 CPU 能处理的最快速度推送数据；
        # 加入 throttle 可以让流图按真实采样率运行，便于观察实时输出，
        # 也避免帧同步块因为数据来得过快而丢失样本（取决于实现）。
        # 如果希望以最快速度处理完文件，可注释掉此模块并在连接时跳过它。
        self.blocks_throttle_0 = blocks.throttle(
            gr.sizeof_gr_complex * 1,
            self.samp_rate,
            True
        )
        self.blocks_throttle_0.set_min_output_buffer(min_buf)

        # --------------------------------------------------
        # 3) LoRa 接收链路（与 lora_RX.py 完全一致）
        # --------------------------------------------------
        # 帧同步：检测前导码、补偿频偏、输出符号块
        #   int(samp_rate/bw) 为每个 chip 的采样点数，
        #   preamble_len 也决定绘图时向前回切的 preamble upchirp 数。
        # 帧同步：center_freq 在此块中仅用于 SFO 估计（sfo_hat = cfo * bw / center_freq），不做混频。
        
        # 这里非常值得注意：传给 frame_sync 的 preamble_len 是连续出现多少个 upchirp 才触发同步的参数；
        # 本脚本的前导码图也按这个值导出 preamble_len + 2 sync + 2.25 SFD。
        self.lora_sdr_frame_sync_0 = lora_sdr.frame_sync(
            int(self.center_freq),
            int(self.bw),
            self.sf,
            self.impl_head,
            [self.sync_word],
            int(self.samp_rate / self.bw),
            int(self.preamble_len)
        )
        

        # FFT 解调：将接收到的 chirp 信号解调为频域峰值（符号）
        self.lora_sdr_fft_demod_0 = lora_sdr.fft_demod(
            self.soft_decoding,
            True     # print_info，调试时可在终端看到解调信息
        )

        # 格雷映射逆映射：将解调出的符号恢复为原始比特序列
        self.lora_sdr_gray_mapping_0 = lora_sdr.gray_mapping(self.soft_decoding)

        # 去交织：按 LoRa 交织矩阵重新排列比特顺序
        self.lora_sdr_deinterleaver_0 = lora_sdr.deinterleaver(self.soft_decoding)

        # 汉明译码：对去交织后的数据进行汉明码纠错/译码
        self.lora_sdr_hamming_dec_0 = lora_sdr.hamming_dec(self.soft_decoding)

        # 包头解析：提取 payload 长度、CR、CRC 标志等，并向 frame_sync 反馈帧信息
        self.lora_sdr_header_decoder_0 = lora_sdr.header_decoder(
            self.impl_head,
            self.cr,
            self.pay_len,
            self.has_crc,
            self.ldro_mode,
            True      # print_header，在终端打印解析出的包头信息
        )

        # 去白化：移除 LoRa 数据白化（whitening）处理
        self.lora_sdr_dewhitening_0 = lora_sdr.dewhitening()

        # CRC 校验：验证数据完整性并在终端打印最终载荷
        # CRC 校验：验证数据完整性并在终端打印最终载荷
        # crc_mode: 0=GRLORA/默认 LoRa PHY 对照, 1=全 payload CRC-16 对照模式
        # CRC 校验：验证数据完整性并在终端打印最终载荷
        crc_mode = lora_sdr.Crc_mode.SX1276 if args.crc_mode == 1 else lora_sdr.Crc_mode.GRLORA
        self.lora_sdr_crc_verif_0 = lora_sdr.crc_verif(
            1,        # print_payload，非 0 表示打印收到的数据包内容
            False,    # output_crc
            crc_mode  # CRC 算法模式
        )

        if self.plot_preamble:
            self.preamble_spectrogram_sink_0 = phy_header_spectrogram_sink(
                args.input_file,
                args.preamble_plot_dir,
                args.preamble_plot_max,
                args.preamble_plot_dpi
            )

        ##################################################
        # 连接信号流
        ##################################################
        # 文件源 -> 节流阀 -> 帧同步 -> FFT 解调 -> 格雷逆映射 -> 去交织
        # -> 汉明译码 -> 包头解析 -> 去白化 -> CRC 校验 -> （终端输出）
        self.connect((self.blocks_file_source_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.lora_sdr_frame_sync_0, 0))
        self.connect((self.lora_sdr_frame_sync_0, 0), (self.lora_sdr_fft_demod_0, 0))
        self.connect((self.lora_sdr_fft_demod_0, 0), (self.lora_sdr_gray_mapping_0, 0))
        self.connect((self.lora_sdr_gray_mapping_0, 0), (self.lora_sdr_deinterleaver_0, 0))
        self.connect((self.lora_sdr_deinterleaver_0, 0), (self.lora_sdr_hamming_dec_0, 0))
        self.connect((self.lora_sdr_hamming_dec_0, 0), (self.lora_sdr_header_decoder_0, 0))
        self.connect((self.lora_sdr_header_decoder_0, 0), (self.lora_sdr_dewhitening_0, 0))
        self.connect((self.lora_sdr_dewhitening_0, 0), (self.lora_sdr_crc_verif_0, 0))

        # 消息连接：包头解析后把帧信息回传给帧同步，以便其知道何时结束当前帧
        self.msg_connect(
            (self.lora_sdr_header_decoder_0, 'frame_info'),
            (self.lora_sdr_frame_sync_0, 'frame_info')
        )
        if self.plot_preamble:
            self.msg_connect(
                (self.lora_sdr_frame_sync_0, 'preamble'),
                (self.preamble_spectrogram_sink_0, 'preamble')
            )

    def save_preamble_plots(self):
        if self.plot_preamble:
            self.preamble_spectrogram_sink_0.save_all()


    def get_sf(self):
        return self.sf

    def set_sf(self, sf):
        self.sf = sf

    def get_bw(self):
        return self.bw

    def set_bw(self, bw):
        self.bw = bw

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

    def get_cr(self):
        return self.cr

    def set_cr(self, cr):
        self.cr = cr

    def get_pay_len(self):
        return self.pay_len

    def set_pay_len(self, pay_len):
        self.pay_len = pay_len

    def get_has_crc(self):
        return self.has_crc

    def set_has_crc(self, has_crc):
        self.has_crc = has_crc

    def get_impl_head(self):
        return self.impl_head

    def set_impl_head(self, impl_head):
        self.impl_head = impl_head

    def get_soft_decoding(self):
        return self.soft_decoding

    def set_soft_decoding(self, soft_decoding):
        self.soft_decoding = soft_decoding

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq

    def get_sync_word(self):
        return self.sync_word

    def set_sync_word(self, sync_word):
        self.sync_word = sync_word

    def get_ldro_mode(self):
        return self.ldro_mode

    def set_ldro_mode(self, ldro_mode):
        self.ldro_mode = ldro_mode

    def get_preamble_len(self):
        return self.preamble_len

    def set_preamble_len(self, preamble_len):
        self.preamble_len = preamble_len


def main():
    """
    命令行入口：解析参数、构建流图、运行并等待处理完成。
    """
    parser = ArgumentParser(
        description="LoRa 离线 IQ 文件接收机 —— 解析 USRP 采集的基带数据"
    )

    # 文件参数
    parser.add_argument(
        "-f", "--input-file",
        type=str,
        required=True,
        help="输入的 IQ 数据文件路径（float32 复数格式，即 gnuradio 的 .fc32 / .cfile）"
    )

    # LoRa 物理层参数
    parser.add_argument(
        "--sf", "--spreading-factor",
        type=int,
        default=7,
        help="扩频因子 SF (默认: 7)"
    )
    parser.add_argument(
        "--bw", "--bandwidth",
        type=eng_float,
        default=125e3,
        help="信号带宽 (Hz)，默认 125000"
    )
    parser.add_argument(
        "--samp-rate",
        type=eng_float,
        default=500e3,
        help="采样率 (Hz)，默认 500000。应与 USRP 采集时的采样率一致"
    )
    parser.add_argument(
        "--cr", "--coding-rate",
        type=int,
        default=1,
        help="编码率 CR (1~4，对应 4/5~4/8)，默认 1"
    )
    parser.add_argument(
        "--pay-len",
        type=int,
        default=255,
        help="预设 payload 长度（仅在 impl_head=True 时生效），默认 255"
    )
    parser.add_argument(
        "--has-crc",
        action="store_true",
        default=True,
        help="发送端是否带 CRC（默认开启）"
    )
    parser.add_argument(
        "--no-crc",
        action="store_false",
        dest="has_crc",
        help="发送端不带 CRC"
    )
    parser.add_argument(
        "--impl-head",
        action="store_true",
        default=False,
        help="使用隐式头模式（默认关闭）"
    )
    parser.add_argument(
        "--soft-decoding",
        action="store_true",
        default=True,
        help="启用软判决译码（默认开启）"
    )
    parser.add_argument(
        "--hard-decoding",
        action="store_false",
        dest="soft_decoding",
        help="使用硬判决译码"
    )
    parser.add_argument(
        "--center-freq",
        type=eng_float,
        default=868.1e6,
        help="中心频率 (Hz)，默认 868.1e6。"
             "在 gr-lora_sdr 中此值仅用于 frame_sync 的采样频率偏移(SFO)估计，不做混频。"
             "若数据来自真实USRP采集(即使已DDC到基带)，建议仍填写USRP当时的射频中心频率。"
    )
    parser.add_argument(
        "--sync-word",
        type=lambda x: int(x, 0),
        default=0x34,
        help="同步字（默认 0x34，支持十进制或 0x 十六进制写法）"
    )
    parser.add_argument(
        "--ldro-mode",
        type=int,
        default=2,
        help="低数据率优化模式 (0=关闭, 1=开启, 2=自动)，默认 2"
    )
    parser.add_argument(
        "--preamble-len",
        type=int,
        default=16,
        help="同步触发阈值参考值（默认 16）。前导码图会导出该值对应的 upchirp 数，再加 2 个 sync word symbol 和 2.25 个 SFD symbol。"
    )
    parser.add_argument(
        "--crc-mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="CRC 算法模式 (0=GRLORA/默认 LoRa PHY 对照, 1=全 payload CRC-16 对照模式)。"
             "LoraSTMacL1/SX1276 LoRa PHY 包优先使用 0。默认 0。"
    )
    parser.add_argument(
        "--plot-preamble",
        "--plot-phy-header",
        dest="plot_preamble",
        action="store_true",
        default=False,
        help="保存 frame_sync 精细对齐后的 LoRa preamble + sync word + SFD 频谱图"
    )
    parser.add_argument(
        "--preamble-plot-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "preamble_plots"),
        help="前导码频谱图输出目录"
    )
    parser.add_argument(
        "--preamble-plot-max",
        type=int,
        default=0,
        help="最多保存多少帧前导码图；0 表示不限制"
    )
    parser.add_argument(
        "--preamble-plot-dpi",
        type=int,
        default=150,
        help="前导码频谱图 DPI，默认 150"
    )

    args = parser.parse_args()

    # 实例化并启动流图
    tb = lora_file_RX(args)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        tb.save_preamble_plots()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.wait()   # 等待文件读取完成（或用户按下 Ctrl+C）
    tb.save_preamble_plots()


if __name__ == '__main__':
    main()

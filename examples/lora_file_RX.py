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

from gnuradio import gr
from gnuradio import blocks
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import gnuradio.lora_sdr as lora_sdr
import numpy as np


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
        self.preamble_len = args.preamble_len    # 前导码长度

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
        #   8 为默认前导码长度（preamble length）。
        # 帧同步：center_freq 在此块中仅用于 SFO 估计（sfo_hat = cfo * bw / center_freq），不做混频。
        self.lora_sdr_frame_sync_0 = lora_sdr.frame_sync(
            int(self.center_freq),
            int(self.bw),
            self.sf,
            self.impl_head,
            [self.sync_word],
            int(self.samp_rate / self.bw),
            self.preamble_len
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
        self.lora_sdr_crc_verif_0 = lora_sdr.crc_verif(
            1,        # print_payload，非 0 表示打印收到的数据包内容
            False     # output_crc
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
        default=0x12,
        help="同步字（默认 0x12，支持十进制或 0x 十六进制写法）"
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
        help="前导码长度（默认 16）"
    )

    args = parser.parse_args()

    # 实例化并启动流图
    tb = lora_file_RX(args)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.wait()   # 等待文件读取完成（或用户按下 Ctrl+C）


if __name__ == '__main__':
    main()

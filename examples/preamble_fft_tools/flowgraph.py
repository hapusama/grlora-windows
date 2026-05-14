# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""GNU Radio receive graph used only to obtain packet synchronization metadata."""

import numpy as np
from gnuradio import blocks
from gnuradio import gr
import gnuradio.lora_sdr as lora_sdr

from .sinks import (
    header_metadata_sink,
    payload_metadata_sink,
    preamble_metadata_sink,
    print_payload_mode,
)


class lora_file_preamble_fft_rx(gr.top_block):
    """Run the same file RX chain far enough to obtain frame_sync preamble ranges."""

    def __init__(self, args):
        gr.top_block.__init__(self, "LoRa File Preamble FFT", catch_exceptions=True)

        # 把已经解析好的采集参数保存到 top_block 中，保证 GNU Radio 流图启动后
        # 不再依赖外部临时状态。
        self.input_file = args.input_file
        self.file_source_path = args.file_source_path
        self.sf = args.sf
        self.bw = args.bw
        self.samp_rate = args.samp_rate
        self.cr = args.cr
        self.pay_len = args.pay_len
        self.has_crc = args.has_crc
        self.impl_head = args.impl_head
        self.soft_decoding = args.soft_decoding
        self.center_freq = args.center_freq
        self.sync_word = args.sync_word
        self.ldro_mode = args.ldro_mode
        self.preamble_len = args.preamble_len

        os_factor = int(round(float(self.samp_rate) / float(self.bw)))
        min_buf = int(np.ceil(os_factor * ((1 << self.sf) + 2)))

        # 从 file_source_path 读取 IQ。Windows 下如果真实路径含中文，这里可能是
        # 一个 ASCII-only 的临时硬链接。
        self.file_source = blocks.file_source(
            gr.sizeof_gr_complex,
            self.file_source_path,
            False,
            0,
            0,
        )
        self.file_source.set_min_output_buffer(min_buf)

        # 标准 gr-lora_sdr 接收链，一直接到 PHY header decoder。默认特征导出
        # 更关心包的时间位置和 header 元数据，不一定需要解出 payload。
        self.frame_sync = lora_sdr.frame_sync(
            int(self.center_freq),
            int(self.bw),
            self.sf,
            self.impl_head,
            [self.sync_word],
            os_factor,
            int(self.preamble_len),
        )
        self.fft_demod = lora_sdr.fft_demod(self.soft_decoding, True)
        self.gray_mapping = lora_sdr.gray_mapping(self.soft_decoding)
        self.deinterleaver = lora_sdr.deinterleaver(self.soft_decoding)
        self.hamming_dec = lora_sdr.hamming_dec(self.soft_decoding)
        self.header_decoder = lora_sdr.header_decoder(
            self.impl_head,
            self.cr,
            self.pay_len,
            self.has_crc,
            self.ldro_mode,
            args.print_header,
        )
        self.require_valid_payload = bool(args.require_valid_payload)
        self.metadata_sink = preamble_metadata_sink()
        self.header_sink = header_metadata_sink()
        if self.require_valid_payload:
            # 可选路径：继续解 payload 并做 CRC 校验，只保留 CRC-valid 的包。
            # 主要用于需要 LoRaWAN FCnt / payload 包号的场景。
            self.dewhitening = lora_sdr.dewhitening()
            crc_mode = lora_sdr.Crc_mode.SX1276 if args.crc_mode == 1 else lora_sdr.Crc_mode.GRLORA
            self.crc_verif = lora_sdr.crc_verif(
                print_payload_mode(args.print_payload),
                True,
                crc_mode,
            )
            self.payload_bytes_null_sink = blocks.null_sink(gr.sizeof_char)
            self.crc_valid_sink = blocks.vector_sink_b()
            self.payload_sink = payload_metadata_sink()
        else:
            self.payload_null_sink = blocks.null_sink(gr.sizeof_char)

        # throttle 适合调试实时播放节奏；批量导出通常使用 --no-throttle 提速。
        if args.throttle:
            self.throttle = blocks.throttle(gr.sizeof_gr_complex, self.samp_rate, True)
            self.throttle.set_min_output_buffer(min_buf)
            self.connect((self.file_source, 0), (self.throttle, 0))
            self.connect((self.throttle, 0), (self.frame_sync, 0))
        else:
            self.connect((self.file_source, 0), (self.frame_sync, 0))

        self.connect((self.frame_sync, 0), (self.fft_demod, 0))
        self.connect((self.fft_demod, 0), (self.gray_mapping, 0))
        self.connect((self.gray_mapping, 0), (self.deinterleaver, 0))
        self.connect((self.deinterleaver, 0), (self.hamming_dec, 0))
        self.connect((self.hamming_dec, 0), (self.header_decoder, 0))
        if self.require_valid_payload:
            # payload-valid 模式保留下游解码链路，让 crc_verif 能输出 payload
            # 消息和 CRC valid 标志。
            self.connect((self.header_decoder, 0), (self.dewhitening, 0))
            self.connect((self.dewhitening, 0), (self.crc_verif, 0))
            self.connect((self.crc_verif, 0), (self.payload_bytes_null_sink, 0))
            self.connect((self.crc_verif, 1), (self.crc_valid_sink, 0))
            self.msg_connect((self.crc_verif, "payload_metadata"), (self.payload_sink, "payload"))
        else:
            self.connect((self.header_decoder, 0), (self.payload_null_sink, 0))

        # header_decoder 会通知 frame_sync 哪些 PHY header 有效；只有 header
        # 通过后，frame_sync 才发布对齐后的 preamble/sync/SFD 样本范围。
        self.msg_connect((self.header_decoder, "frame_info"), (self.frame_sync, "frame_info"))
        self.msg_connect((self.header_decoder, "frame_info"), (self.header_sink, "frame_info"))
        self.msg_connect((self.frame_sync, "preamble"), (self.metadata_sink, "preamble"))

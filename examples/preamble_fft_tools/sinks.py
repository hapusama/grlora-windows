# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0
"""GNU Radio message sinks and payload helpers.

These blocks collect PMT messages emitted by frame_sync, header_decoder, and
crc_verif without changing the C++ block APIs.
"""

import threading

import numpy as np
import pmt
from gnuradio import gr


class preamble_metadata_sink(gr.basic_block):
    """Collect frame_sync preamble messages.

    frame_sync 负责在 IQ 中检测包同步位置。这里收集的是每个包的
    preamble/sync/SFD 对齐样本范围，以及 frame_sync 估计的 SNR/CFO/STO/SFO。
    frame_count 会沿着 header/payload 元数据一起传播，用于避免按 list index 错配。
    """

    def __init__(self):
        gr.basic_block.__init__(
            self,
            name="preamble_metadata_sink",
            in_sig=None,
            out_sig=None,
        )
        self.frames = []
        self._lock = threading.Lock()
        self.message_port_register_in(pmt.intern("preamble"))
        self.set_msg_handler(pmt.intern("preamble"), self.handle_preamble)

    def _dict_value(self, msg, key, default=None):
        value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
        if pmt.is_null(value):
            return default
        return pmt.to_python(value)

    def handle_preamble(self, msg):
        if not pmt.is_dict(msg):
            print("[preamble_fft] ignored non-dict preamble message")
            return

        start_sample = self._dict_value(msg, "start_sample", None)
        end_sample = self._dict_value(msg, "end_sample", None)
        if start_sample is None or end_sample is None:
            print("[preamble_fft] ignored message without sample range")
            return

        frame = {
            "frame_count": int(self._dict_value(msg, "frame_count", 0)),
            "sf": int(self._dict_value(msg, "sf", 7)),
            "bw": float(self._dict_value(msg, "bw", 125000)),
            "sample_rate": float(self._dict_value(msg, "sample_rate", 125000)),
            "samples_per_symbol": int(self._dict_value(msg, "samples_per_symbol", 1 << int(self._dict_value(msg, "sf", 7)))),
            "preamble_len": int(self._dict_value(msg, "preamble_len", 8)),
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "n_samples": int(self._dict_value(msg, "n_samples", int(end_sample) - int(start_sample))),
            "n_symbols": float(self._dict_value(msg, "n_symbols", 0.0)),
            "snr_db": self._dict_value(msg, "snr_db", np.nan),
            "cfo": self._dict_value(msg, "cfo", np.nan),
            "sto": self._dict_value(msg, "sto", np.nan),
            "sfo": self._dict_value(msg, "sfo", np.nan),
            "netid1": int(self._dict_value(msg, "netid1", -1)),
            "netid2": int(self._dict_value(msg, "netid2", -1)),
        }
        for source_key, output_key in (
            ("cr", "cr"),
            ("pay_len", "pay_len"),
            ("crc", "crc"),
            ("ldro_mode", "ldro_mode"),
            ("err", "header_err"),
        ):
            value = self._dict_value(msg, source_key, None)
            if value is not None:
                frame[output_key] = int(value)
        with self._lock:
            self.frames.append(frame)


class header_metadata_sink(gr.basic_block):
    """Collect decoded PHY header messages in detection order.

    header_decoder 解出 payload 长度、编码率、CRC 标志等 PHY header 信息。
    这些信息后面用于估算 packet_end_sample，也就是一个包大概到哪里结束。
    """

    def __init__(self):
        gr.basic_block.__init__(
            self,
            name="header_metadata_sink",
            in_sig=None,
            out_sig=None,
        )
        self.headers = []
        self._lock = threading.Lock()
        self.message_port_register_in(pmt.intern("frame_info"))
        self.set_msg_handler(pmt.intern("frame_info"), self.handle_frame_info)

    def _dict_value(self, msg, key, default=None):
        value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
        if pmt.is_null(value):
            return default
        return pmt.to_python(value)

    def handle_frame_info(self, msg):
        if not pmt.is_dict(msg):
            return

        header = {
            "frame_count": int(self._dict_value(msg, "frame_count", -1)),
            "cr": int(self._dict_value(msg, "cr", -1)),
            "pay_len": int(self._dict_value(msg, "pay_len", -1)),
            "crc": int(self._dict_value(msg, "crc", 0)),
            "ldro_mode": int(self._dict_value(msg, "ldro_mode", 2)),
            "header_err": int(self._dict_value(msg, "err", 1)),
        }
        for key in ("start_sample", "end_sample"):
            value = self._dict_value(msg, key, None)
            if value is not None:
                header[key] = int(value)
        with self._lock:
            self.headers.append(header)


def payload_msg_to_bytes(msg):
    """Convert crc_verif payload PMT into raw bytes."""
    if isinstance(msg, bytes):
        return msg
    if isinstance(msg, bytearray):
        return bytes(msg)
    if isinstance(msg, str):
        return msg.encode("latin-1", errors="ignore")
    if isinstance(msg, np.ndarray):
        return msg.astype(np.uint8, copy=False).tobytes()
    if isinstance(msg, (list, tuple)):
        return bytes(int(item) & 0xFF for item in msg)

    if hasattr(pmt, "is_u8vector") and pmt.is_u8vector(msg):
        return bytes(pmt.u8vector_elements(msg))
    if hasattr(pmt, "is_blob") and pmt.is_blob(msg):
        return bytes(pmt.blob_data(msg))
    if pmt.is_symbol(msg):
        try:
            return pmt.symbol_to_string(msg).encode("latin-1", errors="ignore")
        except UnicodeDecodeError:
            return b""

    try:
        payload = pmt.to_python(msg)
    except UnicodeDecodeError:
        return b""
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode("latin-1", errors="ignore")
    return b""


def extract_payload_packet_number(payload):
    if len(payload) >= 8 and (payload[0] & 0xE0) in (0x40, 0x60, 0x80, 0xA0):
        # Branch4 LoRaWAN-like PHYPayload: MHDR|DevAddr|FCtrl|FCnt|...
        return int(payload[6]) + (int(payload[7]) << 8)
    if len(payload) >= 2:
        return int(payload[0]) + (int(payload[1]) << 8)
    return ""


def print_payload_mode(mode):
    return {"none": 0, "ascii": 1, "hex": 2}.get(str(mode).lower(), 0)


class payload_metadata_sink(gr.basic_block):
    """Collect decoded payload packet numbers and frame metadata."""

    def __init__(self):
        gr.basic_block.__init__(
            self,
            name="payload_metadata_sink",
            in_sig=None,
            out_sig=None,
        )
        self.payloads = []
        self._lock = threading.Lock()
        self.message_port_register_in(pmt.intern("payload"))
        self.set_msg_handler(pmt.intern("payload"), self.handle_payload)

    def _dict_value(self, msg, key, default=None):
        value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
        if pmt.is_null(value):
            return default
        return pmt.to_python(value)

    def _dict_payload_bytes(self, msg):
        value = pmt.dict_ref(msg, pmt.intern("payload_bytes"), pmt.PMT_NIL)
        if pmt.is_null(value):
            value = pmt.dict_ref(msg, pmt.intern("payload"), pmt.PMT_NIL)
        return payload_msg_to_bytes(value)

    def handle_payload(self, msg):
        metadata = {}
        if pmt.is_dict(msg):
            payload = self._dict_payload_bytes(msg)
            for key in ("frame_count", "start_sample", "end_sample"):
                value = self._dict_value(msg, key, None)
                if value is not None:
                    metadata[key] = int(value)
            crc_valid = self._dict_value(msg, "crc_valid", None)
            if crc_valid is not None:
                metadata["crc_valid"] = bool(crc_valid)
            decoded_payload_len = self._dict_value(msg, "decoded_payload_len", len(payload))
        else:
            payload = payload_msg_to_bytes(msg)
            decoded_payload_len = len(payload)
        packet_number = extract_payload_packet_number(payload)

        with self._lock:
            metadata.update(
                {
                    "header_packet_counter": packet_number,
                    "payload_packet_number": packet_number,
                    "decoded_payload_len": int(decoded_payload_len),
                }
            )
            self.payloads.append(metadata)

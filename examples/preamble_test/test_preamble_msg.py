#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_preamble_msg.py
====================
测试脚本：验证 Python 端解析 frame_sync 发送的 PMT preamble 消息。

由于 frame_sync 的 C++ 修改尚未编译，本脚本**模拟** C++ 端发送的消息格式，
验证以下链路：
    1. PMT dict 构造（模拟 C++）
    2. blob → numpy.complex64 转换
    3. 字段完整性检查
    4. 简单的频谱图画图流程（使用模拟 chirp 数据）

运行方式：
    python test_preamble_msg.py

依赖：numpy, pmt (来自 gnuradio), matplotlib, scipy
"""

import numpy as np
import pmt
import sys

# 尝试导入画图库
try:
    import matplotlib.pyplot as plt
    from scipy import signal
    HAS_PLOT = True
except ImportError as e:
    print(f"[WARN] 缺少画图依赖 ({e})，仅执行消息解析测试。")
    HAS_PLOT = False


def build_test_chirp(sf: int = 7, bw: float = 125e3, fs: float = 500e3, nsymb: int = 5):
    """
    生成模拟的 LoRa upchirp 前导码（多个 upchirp 连接）。
    仅用于测试画图链路，不保证与真实 LoRa 调制完全一致。
    """
    N = 1 << sf
    os_factor = int(round(fs / bw))
    sps = N * os_factor  # samples per symbol (oversampled)
    t = np.arange(sps) / fs
    # 理想 upchirp: 频率从 -bw/2 线性扫到 +bw/2
    f0 = -bw / 2
    f1 = +bw / 2
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t[-1]))
    single_chirp = np.exp(1j * phase)
    preamble = np.tile(single_chirp, nsymb)
    return preamble.astype(np.complex64)


def simulate_cpp_preamble_message() -> pmt.pmt_t:
    """
    模拟 frame_sync_impl.cc 中 emit 的 PMT 消息字典。
    对应 C++ 代码：
        pmt::pmt_t iq_blob = pmt::make_blob(&preamble_upchirps[0], nbytes);
        pmt::pmt_t preamb_dict = pmt::make_dict();
        preamb_dict = pmt::dict_add(preamb_dict, pmt::intern("preamble_iq"), iq_blob);
        preamb_dict = pmt::dict_add(preamb_dict, pmt::intern("frame_count"), pmt::from_long(frame_cnt));
        ...
        message_port_pub(pmt::mp("preamble"), preamb_dict);
    """
    sf = 7
    bw = 125e3
    fs = 500e3
    nsymb = 5

    # 生成模拟前导码 IQ（复数数组）
    iq_arr = build_test_chirp(sf=sf, bw=bw, fs=fs, nsymb=nsymb)

    # C++ 端通过 make_blob 发送原始字节；Python 端收到的是 pmt blob
    # pmt 没有直接的 make_blob 对应 Python 函数，但我们可以用 pmt.make_u8vector
    # 然后把字节塞进去，最后再用 pmt.to_python 转换。
    # 更简单：直接用 pmt 的 string/bool 机制，或者直接用 numpy 的 tobytes。
    # 实际上 GNU Radio Python 绑定中 pmt.pmt_to_python 会自动处理 blob。
    # 在 Python 侧构造 blob 的方法：用 pmt.make_u8vector 再转成 blob，或者
    # 直接使用 pmt.pmt_from_uint64 + bytes。这里采用最直接的方式：
    raw_bytes = iq_arr.tobytes()
    # pmt 没有直接的 from_bytes，但 make_u8vector 可以构造等价的 blob
    u8v = pmt.make_u8vector(len(raw_bytes), 0)
    for i, b in enumerate(raw_bytes):
        pmt.u8vector_set(u8v, i, b)
    # u8vector 在通过消息端口传输时会被视为 blob；在 Python 侧解析时，
    # pmt.to_python 会返回 bytes。
    # 不过为了更接近真实场景，我们直接用 pmt.intern 做键，值用 pmt 原生类型。

    preamb_dict = pmt.make_dict()
    preamb_dict = pmt.dict_add(preamb_dict, pmt.intern("preamble_iq"), u8v)
    preamb_dict = pmt.dict_add(preamb_dict, pmt.intern("frame_count"), pmt.from_long(1))
    preamb_dict = pmt.dict_add(preamb_dict, pmt.intern("snr_db"), pmt.from_float(12.5))
    preamb_dict = pmt.dict_add(preamb_dict, pmt.intern("cfo"), pmt.from_float(0.3))
    preamb_dict = pmt.dict_add(preamb_dict, pmt.intern("sto"), pmt.from_float(-0.1))
    preamb_dict = pmt.dict_add(preamb_dict, pmt.intern("sfo"), pmt.from_float(1e-6))
    preamb_dict = pmt.dict_add(preamb_dict, pmt.intern("sf"), pmt.from_long(sf))
    preamb_dict = pmt.dict_add(preamb_dict, pmt.intern("bw"), pmt.from_long(int(bw)))
    preamb_dict = pmt.dict_add(preamb_dict, pmt.intern("nsymb"), pmt.from_long(nsymb))

    return preamb_dict


def parse_preamble_message(msg: pmt.pmt_t) -> dict:
    """
    解析 PMT 消息字典，提取字段并还原 numpy complex64 数组。
    """
    assert pmt.is_dict(msg), "消息不是 PMT dict"

    result = {}
    # 提取字段
    result["frame_count"] = pmt.to_python(pmt.dict_ref(msg, pmt.intern("frame_count"), pmt.PMT_NIL))
    result["snr_db"] = pmt.to_python(pmt.dict_ref(msg, pmt.intern("snr_db"), pmt.PMT_NIL))
    result["cfo"] = pmt.to_python(pmt.dict_ref(msg, pmt.intern("cfo"), pmt.PMT_NIL))
    result["sto"] = pmt.to_python(pmt.dict_ref(msg, pmt.intern("sto"), pmt.PMT_NIL))
    result["sfo"] = pmt.to_python(pmt.dict_ref(msg, pmt.intern("sfo"), pmt.PMT_NIL))
    result["sf"] = pmt.to_python(pmt.dict_ref(msg, pmt.intern("sf"), pmt.PMT_NIL))
    result["bw"] = pmt.to_python(pmt.dict_ref(msg, pmt.intern("bw"), pmt.PMT_NIL))
    result["nsymb"] = pmt.to_python(pmt.dict_ref(msg, pmt.intern("nsymb"), pmt.PMT_NIL))

    # 还原 IQ 数据：PMT u8vector / blob → bytes → numpy.complex64
    iq_pmt = pmt.dict_ref(msg, pmt.intern("preamble_iq"), pmt.PMT_NIL)
    if pmt.is_u8vector(iq_pmt):
        # 从 u8vector 提取字节
        nbytes = pmt.length(iq_pmt)
        raw_bytes = bytes(pmt.u8vector_elements(iq_pmt))
    elif pmt.is_blob(iq_pmt):
        raw_bytes = pmt.blob_data(iq_pmt)
    else:
        raise TypeError(f"preamble_iq 类型不支持: {pmt.write_string(iq_pmt)}")

    iq_arr = np.frombuffer(raw_bytes, dtype=np.complex64)
    result["preamble_iq"] = iq_arr
    return result


def test_message_roundtrip():
    print("=" * 60)
    print("测试 1/3: PMT 消息构造与解析")
    print("=" * 60)

    msg = simulate_cpp_preamble_message()
    parsed = parse_preamble_message(msg)

    assert parsed["frame_count"] == 1
    assert parsed["sf"] == 7
    assert parsed["bw"] == 125000
    assert parsed["nsymb"] == 5
    assert len(parsed["preamble_iq"]) == parsed["nsymb"] * (1 << parsed["sf"]) * 4  # os_factor=4
    assert parsed["preamble_iq"].dtype == np.complex64

    print(f"  frame_count : {parsed['frame_count']}")
    print(f"  sf          : {parsed['sf']}")
    print(f"  bw          : {parsed['bw']} Hz")
    print(f"  nsymb       : {parsed['nsymb']}")
    print(f"  snr_db      : {parsed['snr_db']:.2f} dB")
    print(f"  cfo         : {parsed['cfo']:.3f}")
    print(f"  sto         : {parsed['sto']:.3f}")
    print(f"  sfo         : {parsed['sfo']:.2e}")
    print(f"  IQ 样本数   : {len(parsed['preamble_iq'])}")
    print(f"  IQ dtype    : {parsed['preamble_iq'].dtype}")
    print("  -> 消息解析测试通过 ✔")


def test_spectrogram_plot():
    if not HAS_PLOT:
        print("\n跳过画图测试（缺少 matplotlib/scipy）。")
        return

    print("\n" + "=" * 60)
    print("测试 2/3: 频谱图 (Spectrogram) 绘制")
    print("=" * 60)

    msg = simulate_cpp_preamble_message()
    parsed = parse_preamble_message(msg)

    iq = parsed["preamble_iq"]
    fs = parsed["bw"] * 4  # 模拟 os_factor=4
    sf = parsed["sf"]
    nperseg = (1 << sf) * 4  # 窗口长度 = 一个符号的采样数
    noverlap = int(nperseg * 0.75)

    f, t, Sxx = signal.spectrogram(
        iq,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='spectrum',
        mode='complex'
    )

    # 转换为 dB
    Sxx_dB = 10 * np.log10(np.abs(Sxx) + 1e-12)

    fig, ax = plt.subplots(figsize=(10, 4))
    # 频率范围限制在 [-bw/2, bw/2]
    bw = parsed["bw"]
    freq_khz = f / 1e3 - bw / 2 / 1e3
    time_ms = t * 1e3

    im = ax.pcolormesh(time_ms, freq_khz, Sxx_dB, shading='gouraud', cmap='viridis')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Preamble Symbol Spectrogram (Test)")
    fig.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()

    out_path = "test_preamble_spectrogram.png"
    fig.savefig(out_path, dpi=150)
    print(f"  -> 频谱图已保存到 {out_path}")
    plt.close(fig)


def test_receiver_block_snippet():
    """
    测试 3/3: 打印一个可直接嵌入 lora_file_RX.py 的 Python 消息接收块代码片段。
    """
    print("\n" + "=" * 60)
    print("测试 3/3: 嵌入式 Python 块代码片段 (供 lora_file_RX.py 使用)")
    print("=" * 60)

    snippet = '''
# ---------------------------------------------------------------
# 插入到 lora_file_RX.py 中的 Embedded Python Block 示例
# ---------------------------------------------------------------
import numpy as np
from gnuradio import gr
import pmt

class preamble_collector(gr.basic_block):
    """
    接收 frame_sync 的 preamble 消息，保存到 Python 列表。
    """
    def __init__(self):
        gr.basic_block.__init__(
            self,
            name="preamble_collector",
            in_sig=None,
            out_sig=None
        )
        self.message_port_register_in(pmt.intern("preamble"))
        self.set_msg_handler(pmt.intern("preamble"), self.handle_preamble)
        self.preambles = []

    def handle_preamble(self, msg):
        if not pmt.is_dict(msg):
            return
        # 解析字段
        iq_pmt = pmt.dict_ref(msg, pmt.intern("preamble_iq"), pmt.PMT_NIL)
        if pmt.is_u8vector(iq_pmt):
            raw = bytes(pmt.u8vector_elements(iq_pmt))
        elif pmt.is_blob(iq_pmt):
            raw = pmt.blob_data(iq_pmt)
        else:
            return
        iq = np.frombuffer(raw, dtype=np.complex64)
        self.preambles.append({
            "frame_count": pmt.to_python(pmt.dict_ref(msg, pmt.intern("frame_count"), pmt.PMT_NIL)),
            "snr_db": pmt.to_python(pmt.dict_ref(msg, pmt.intern("snr_db"), pmt.PMT_NIL)),
            "cfo": pmt.to_python(pmt.dict_ref(msg, pmt.intern("cfo"), pmt.PMT_NIL)),
            "sf": pmt.to_python(pmt.dict_ref(msg, pmt.intern("sf"), pmt.PMT_NIL)),
            "bw": pmt.to_python(pmt.dict_ref(msg, pmt.intern("bw"), pmt.PMT_NIL)),
            "nsymb": pmt.to_python(pmt.dict_ref(msg, pmt.intern("nsymb"), pmt.PMT_NIL)),
            "preamble_iq": iq,
        })
        print(f"[preamble_collector] 收到帧 #{len(self.preambles)} 的前导码，"
              f"长度={len(iq)}，SNR={self.preambles[-1]['snr_db']:.1f} dB")
'''
    print(snippet)


if __name__ == "__main__":
    test_message_roundtrip()
    test_spectrogram_plot()
    test_receiver_block_snippet()
    print("\n" + "=" * 60)
    print("全部测试完成。若以上测试通过，说明消息格式设计正确。")
    print("下一步：编译 gr-lora_sdr，然后运行 lora_file_RX.py 集成测试。")
    print("=" * 60)

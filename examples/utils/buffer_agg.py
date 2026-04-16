import numpy as np
from gnuradio import gr


class BufferAggregator(gr.sync_block):
    """
    Accumulate input samples in an internal buffer and emit them in fixed-size
    chunks. This works around gr-lora_sdr blocks whose forecast() exceeds the
    default GNU Radio output-buffer size on Windows (8192 items).
    """

    def __init__(self, chunk_size=16384):
        gr.sync_block.__init__(
            self,
            name="buffer_aggregator",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self.chunk_size = chunk_size
        self._buf = np.array([], dtype=np.complex64)

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        self._buf = np.concatenate((self._buf, in0))

        n_chunks = len(self._buf) // self.chunk_size
        n_out = n_chunks * self.chunk_size
        if n_out == 0:
            return 0

        out[:n_out] = self._buf[:n_out]
        self._buf = self._buf[n_out:]
        return n_out

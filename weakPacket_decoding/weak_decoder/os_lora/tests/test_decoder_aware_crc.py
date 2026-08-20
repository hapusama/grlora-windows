from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from weak_decoder.decoding.payload_codec import encode_explicit_frame_symbols
from weak_decoder.os_lora.system.decoder_aware_crc import (
    DecoderAwareSymbolDecision,
    decode_savaux_sync_candidate,
)


class DecoderAwareCrcTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame_sync = SimpleNamespace(
            valid=False,
            fine_payload_start_sample=0,
            cfo_int_est=0,
            cfo_frac_est=0.0,
            sfo_hat=0.0,
            sfo_cum_initial=0.0,
        )

    def test_strict_mode_rejects_gate_failed_candidate(self) -> None:
        result = decode_savaux_sync_candidate(
            np.zeros(8, dtype=np.complex64),
            self.frame_sync,
            sf=12,
            bw_hz=125_000.0,
            os_factor=8,
        )
        self.assertEqual(result.status, "sync_gate_rejected")
        self.assertFalse(result.candidate_used)

    def test_opt_in_candidate_runs_header_fec_and_crc(self) -> None:
        payload = bytes(range(33))
        header_symbols, payload_symbols = encode_explicit_frame_symbols(
            payload,
            sf=12,
            cr=4,
            has_crc=True,
            ldro=True,
            crc_mode="grlora",
        )
        expected = list(header_symbols) + list(payload_symbols)

        def fake_demod(_samples: np.ndarray, **kwargs: object) -> DecoderAwareSymbolDecision:
            index = int(kwargs["frame_symbol_index"])
            stage = str(kwargs["stage"])
            return DecoderAwareSymbolDecision(
                stage=stage,
                frame_symbol_index=index,
                stage_symbol_index=int(kwargs["stage_symbol_index"]),
                start_sample=int(kwargs["start_sample"]),
                raw_fft_bin=0,
                symbol_value=int(expected[index]),
                peak_margin_db=10.0,
            )

        with patch(
            "weak_decoder.os_lora.system.decoder_aware_crc._demod_symbol",
            side_effect=fake_demod,
        ):
            result = decode_savaux_sync_candidate(
                np.zeros(8, dtype=np.complex64),
                self.frame_sync,
                sf=12,
                bw_hz=125_000.0,
                os_factor=8,
                ldro_mode=1,
                crc_mode="grlora",
                allow_gate_failed_candidate=True,
            )

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.candidate_used)
        self.assertTrue(result.header_valid)
        self.assertTrue(result.crc_valid)
        self.assertEqual(result.payload_bytes, payload)
        self.assertEqual(len(result.header_symbols), 8)
        self.assertEqual(len(result.payload_symbols), len(payload_symbols))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from weak_decoder.decoding.header_first_demod import decode_explicit_header
from weak_decoder.decoding.payload_codec import (
    decode_explicit_frame_symbols,
    encode_explicit_frame_symbols,
)
from weak_decoder.os_lora.system.soft_hamming_crc import (
    decode_soft_hamming_sync_candidate,
    soft_repair_interleaver_block,
)
from weak_decoder.os_lora.system.decoder_aware_crc import (
    DecoderAwareSymbolDecision,
)


class SoftHammingCrcTests(unittest.TestCase):
    @staticmethod
    def _power(symbol: int, *, sf: int, divisor: int) -> np.ndarray:
        n_bins = 1 << int(sf)
        power = np.ones(n_bins, dtype=np.float64)
        raw_bin = (int(symbol) * int(divisor) + 1) % n_bins
        power[raw_bin] = 1_000.0
        return power

    def test_clean_encoded_frame_round_trips_through_soft_blocks(self) -> None:
        sf = 12
        payload = bytes(range(33))
        header_symbols, payload_symbols = encode_explicit_frame_symbols(
            payload,
            sf=sf,
            cr=4,
            has_crc=True,
            ldro=True,
            crc_mode="grlora",
        )
        repaired_header = soft_repair_interleaver_block(
            [self._power(value, sf=sf, divisor=4) for value in header_symbols],
            sf=sf,
            is_header=True,
            cr=4,
            ldro=False,
        )
        self.assertEqual(repaired_header.symbol_values, tuple(header_symbols))
        header = decode_explicit_header(
            repaired_header.symbol_values,
            sf=sf,
            bw=125_000.0,
            ldro_mode=1,
        )
        self.assertTrue(header.header_valid)

        repaired_payload: list[int] = []
        for offset in range(0, len(payload_symbols), 8):
            block = payload_symbols[offset : offset + 8]
            repaired = soft_repair_interleaver_block(
                [self._power(value, sf=sf, divisor=4) for value in block],
                sf=sf,
                is_header=False,
                cr=4,
                ldro=True,
            )
            repaired_payload.extend(repaired.symbol_values)
        self.assertEqual(repaired_payload, payload_symbols)
        frame = decode_explicit_frame_symbols(
            repaired_header.symbol_values,
            repaired_payload,
            sf=sf,
            bw=125_000.0,
            ldro_mode=1,
            crc_mode="grlora",
        )
        self.assertTrue(frame.payload.crc_valid)
        self.assertEqual(frame.payload.payload_bytes, payload)

    def test_invalid_block_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs 8 spectra"):
            soft_repair_interleaver_block(
                [np.ones(64, dtype=np.float64)],
                sf=6,
                is_header=True,
                cr=4,
                ldro=False,
            )

    def test_codeword_constraints_can_override_one_wrong_argmax(self) -> None:
        sf = 12
        header_symbols, _payload_symbols = encode_explicit_frame_symbols(
            bytes(range(33)),
            sf=sf,
            cr=4,
            has_crc=True,
            ldro=True,
            crc_mode="grlora",
        )
        powers = [
            self._power(value, sf=sf, divisor=4) for value in header_symbols
        ]
        true_raw = (int(header_symbols[0]) * 4 + 1) % (1 << sf)
        wrong_symbol = (int(header_symbols[0]) + 1) % (1 << (sf - 2))
        wrong_raw = (wrong_symbol * 4 + 1) % (1 << sf)
        powers[0][true_raw] = 900.0
        powers[0][wrong_raw] = 1_000.0
        self.assertNotEqual(int(np.argmax(powers[0])), true_raw)

        repaired = soft_repair_interleaver_block(
            powers,
            sf=sf,
            is_header=True,
            cr=4,
            ldro=False,
        )
        self.assertEqual(repaired.symbol_values, tuple(header_symbols))

    def test_checksum_valid_header_with_illegal_cr_is_rejected(self) -> None:
        sf = 12
        header_symbols, _payload_symbols = encode_explicit_frame_symbols(
            bytes(range(33)),
            sf=sf,
            cr=4,
            has_crc=True,
            ldro=True,
            crc_mode="grlora",
        )
        valid_header = decode_explicit_header(
            header_symbols,
            sf=sf,
            bw=125_000.0,
            ldro_mode=1,
        )
        invalid_cr_header = replace(valid_header, cr=0, header_valid=True)
        index = 0

        def fake_evidence(*_args: object, **kwargs: object) -> SimpleNamespace:
            nonlocal index
            value = int(header_symbols[index])
            index += 1
            decision = DecoderAwareSymbolDecision(
                stage="header",
                frame_symbol_index=int(kwargs["frame_symbol_index"]),
                stage_symbol_index=int(kwargs["stage_symbol_index"]),
                start_sample=int(kwargs["start_sample"]),
                raw_fft_bin=0,
                symbol_value=value,
                peak_margin_db=10.0,
            )
            return SimpleNamespace(
                decision=decision,
                power=self._power(value, sf=sf, divisor=4),
            )

        frame_sync = SimpleNamespace(
            valid=True,
            fine_payload_start_sample=0,
            cfo_int_est=0,
            cfo_frac_est=0.0,
            sfo_hat=0.0,
            sfo_cum_initial=0.0,
        )
        with (
            patch(
                "weak_decoder.os_lora.system.soft_hamming_crc._demod_symbol_evidence",
                side_effect=fake_evidence,
            ),
            patch(
                "weak_decoder.os_lora.system.soft_hamming_crc.decode_explicit_header",
                return_value=invalid_cr_header,
            ),
        ):
            result = decode_soft_hamming_sync_candidate(
                np.zeros(8, dtype=np.complex64),
                frame_sync,
                sf=sf,
                bw_hz=125_000.0,
                os_factor=8,
            )

        self.assertEqual(result.status, "header_cr_invalid")
        self.assertTrue(result.candidate_used)


if __name__ == "__main__":
    unittest.main()

"""Tests for the ported gr-lora_sdr native soft decoding chain."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from weak_decoder.decoding.header_first_demod import (
    bits_to_int,
    decode_explicit_header,
    int_to_bits_msb,
)
from weak_decoder.decoding.payload_codec import (
    decode_explicit_frame_symbols,
    encode_explicit_frame_symbols,
    encode_hamming_nibble,
)
from weak_decoder.os_lora.experiment_support.noisy_framesync_ota import (
    load_json,
    load_single_packet_sync_module,
    make_sync_config,
    packet_metadata_paths,
)
from weak_decoder.os_lora.system.soft_hamming_crc import _gray_to_binary
from weak_decoder.os_lora.system.upstream_soft_crc import (
    decode_upstream_soft_sync_candidate,
    log_i0,
    upstream_soft_repair_interleaver_block,
    upstream_symbol_bit_llrs,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
DATASET_REPO = WORKSPACE_ROOT / "lora-rfsr-savaux"
OTA_ROOT = DATASET_REPO / "data" / "reference_phy" / "rfsr_db"

SF12 = 12
OS_FACTOR = 8
DEMOD_TAIL_SAMPLES = 256


def _reference_i0_series(x: float) -> float:
    """Independent scalar power-series I0 used only for cross-checking."""
    total = 1.0
    term = 1.0
    k = 1
    while True:
        term *= (x * 0.5) ** 2 / float(k * k)
        total += term
        k += 1
        if term <= total * 1e-20 and k > x:
            break
    return total


def _ideal_power(symbol: int, *, sf: int, divisor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    power = np.ones(n_bins, dtype=np.float64)
    power[(int(symbol) * int(divisor) + 1) % n_bins] = 1_000.0
    return power


class LogI0Tests(unittest.TestCase):
    def test_series_region_matches_reference(self) -> None:
        for x in (0.0, 0.5, 2.0, 5.0, 10.0, 20.0):
            expected = math.log(_reference_i0_series(x)) if x > 0 else 0.0
            self.assertAlmostEqual(
                float(log_i0(np.array([x]))[0]),
                expected,
                delta=max(1e-12, abs(expected) * 1e-12),
                msg=f"x={x}",
            )

    def test_asymptotic_region_properties(self) -> None:
        values = log_i0(np.array([25.0, 50.0, 200.0, 700.0]))
        for x, value in zip((25.0, 50.0, 200.0, 700.0), values):
            leading = x - 0.5 * math.log(2.0 * math.pi * x)
            self.assertGreater(float(value), leading)
            self.assertLess(float(value), leading + 0.01, msg=f"x={x}")

    def test_region_boundary_is_continuous(self) -> None:
        values = log_i0(np.array([19.99, 20.01]))
        slope = float(values[1] - values[0])
        self.assertGreater(slope, 0.018)
        self.assertLess(slope, 0.021)


class UpstreamBitLlrTests(unittest.TestCase):
    def test_full_rate_signs_match_gray_bits(self) -> None:
        sf = 7
        symbol = 0b1011011
        llrs = upstream_symbol_bit_llrs(
            _ideal_power(symbol, sf=sf, divisor=1),
            sf=sf,
            reduced_rate=False,
        )
        gray = symbol ^ (symbol >> 1)
        for bit in range(sf):
            expected = (gray >> bit) & 1
            self.assertEqual(
                int(llrs[sf - 1 - bit] > 0),
                expected,
                msg=f"bit {bit}",
            )

    def test_reduced_rate_low_bits_match(self) -> None:
        sf = 7
        symbol = 0b1101  # 2^(sf-2) symbol space
        llrs = upstream_symbol_bit_llrs(
            _ideal_power(symbol, sf=sf, divisor=4),
            sf=sf,
            reduced_rate=True,
        )
        gray = symbol ^ (symbol >> 1)
        sf_app = sf - 2
        for bit in range(sf_app):
            self.assertEqual(
                int(llrs[sf - 1 - bit] > 0),
                (gray >> bit) & 1,
                msg=f"bit {bit}",
            )


class UpstreamBlockRoundTripTests(unittest.TestCase):
    @staticmethod
    def _block_symbols(
        nibbles: list[int], *, sf: int, is_header: bool, cr: int, ldro: bool
    ) -> tuple[list[int], int, int]:
        cr_app = 4 if is_header else int(cr)
        cw_len = 8 if is_header else 4 + cr_app
        sf_app = sf - 2 if (is_header or ldro) else sf
        codewords = [
            encode_hamming_nibble(nibble, cr_app=cr_app) for nibble in nibbles
        ]
        cw_bits = [int_to_bits_msb(cw, cw_len) for cw in codewords]
        symbols: list[int] = []
        for column in range(cw_len):
            gray_bits = [
                cw_bits[(column - bit_index - 1) % sf_app][column]
                for bit_index in range(sf_app)
            ]
            symbols.append(_gray_to_binary(bits_to_int(gray_bits)))
        return symbols, cw_len, sf_app

    def _round_trip(
        self, *, is_header: bool, cr: int, ldro: bool
    ) -> None:
        sf = SF12
        sf_app = sf - 2 if (is_header or ldro) else sf
        rng = np.random.default_rng(20260914)
        nibbles = [int(v) for v in rng.integers(0, 16, size=sf_app)]
        symbols, cw_len, _ = self._block_symbols(
            nibbles, sf=sf, is_header=is_header, cr=cr, ldro=ldro
        )
        divisor = 4 if (is_header or ldro) else 1
        repaired = upstream_soft_repair_interleaver_block(
            [_ideal_power(symbol, sf=sf, divisor=divisor) for symbol in symbols],
            sf=sf,
            is_header=is_header,
            cr=cr,
            ldro=ldro,
        )
        self.assertEqual(repaired.decoded_nibbles, tuple(nibbles))
        self.assertEqual(repaired.symbol_values, tuple(symbols))
        self.assertEqual(cw_len, 8 if is_header else 4 + cr)

    def test_header_block_round_trips(self) -> None:
        self._round_trip(is_header=True, cr=4, ldro=False)

    def test_payload_blocks_round_trip_all_coding_rates(self) -> None:
        for cr in (1, 2, 3, 4):
            with self.subTest(cr=cr):
                self._round_trip(is_header=False, cr=cr, ldro=False)

    def test_ldro_payload_block_round_trips(self) -> None:
        self._round_trip(is_header=False, cr=4, ldro=True)


class UpstreamFullFrameTests(unittest.TestCase):
    def test_clean_encoded_frame_round_trips(self) -> None:
        sf = SF12
        payload = bytes(range(33))
        header_symbols, payload_symbols = encode_explicit_frame_symbols(
            payload,
            sf=sf,
            cr=4,
            has_crc=True,
            ldro=True,
            crc_mode="grlora",
        )
        repaired_header = upstream_soft_repair_interleaver_block(
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
            repaired = upstream_soft_repair_interleaver_block(
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

    @staticmethod
    def _power(symbol: int, *, sf: int, divisor: int) -> np.ndarray:
        return _ideal_power(symbol, sf=sf, divisor=divisor)


@unittest.skipUnless(
    (OTA_ROOT / "metadata").is_dir(), "OTA reference dataset not present"
)
class UpstreamCleanOtaDecodeTests(unittest.TestCase):
    def test_first_clean_ota_packet_decodes_exactly(self) -> None:
        metadata_path = next(iter(packet_metadata_paths(OTA_ROOT)))
        metadata = load_json(metadata_path)
        reference = load_json(
            OTA_ROOT.parent / "metadata" / f"{int(metadata['reference']['reference_id']):06d}.json"
        )
        samples = np.fromfile(
            OTA_ROOT / str(metadata["ota"]["relative_path"]),
            dtype=np.dtype("<c8"),
        )
        module = load_single_packet_sync_module(DATASET_REPO)
        result = module.run_single_packet_sync(
            samples,
            make_sync_config(
                module,
                float(metadata["capture"]["center_frequency_hz"]),
                sf=SF12,
                bw_hz=125_000.0,
                sample_rate_hz=1_000_000.0,
                preamble_symbols=16,
                sync_word=0x12,
            ),
        )
        self.assertTrue(result.synchronized, msg=str(result.error))
        frame_sync = result.frame_sync
        padded = np.pad(samples, (0, DEMOD_TAIL_SAMPLES)).astype(np.complex64)
        decoded = decode_upstream_soft_sync_candidate(
            padded,
            SimpleNamespace(
                valid=True,
                fine_payload_start_sample=int(
                    frame_sync.fine_payload_start_sample
                ),
                cfo_int_est=int(frame_sync.cfo_int_est),
                cfo_frac_est=float(frame_sync.cfo_frac_est),
                sfo_hat=float(frame_sync.sfo_hat),
                sfo_cum_initial=float(frame_sync.sfo_cum_initial),
            ),
            sf=SF12,
            bw_hz=125_000.0,
            os_factor=OS_FACTOR,
            ldro_mode=1,
            crc_mode="grlora",
        )
        expected = bytes.fromhex(str(reference["packet"]["frame_hex"]))
        self.assertTrue(decoded.header_valid, msg=str(decoded.status))
        self.assertTrue(decoded.crc_valid, msg=str(decoded.status))
        self.assertEqual(decoded.payload_bytes, expected)


if __name__ == "__main__":
    unittest.main()

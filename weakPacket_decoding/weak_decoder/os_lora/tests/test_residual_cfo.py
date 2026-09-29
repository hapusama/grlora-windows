"""Check physical CFO ranking and bounded CRC acceptance semantics."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from weak_decoder.chirp import build_upchirp
from weak_decoder.os_lora.system.residual_cfo import (
    build_residual_cfo_candidates,
    decode_residual_cfo_candidates,
    rank_residual_cfo_from_preamble,
)


class ResidualCfoTests(unittest.TestCase):
    def frame(self, cfo=0, start=10368):
        return SimpleNamespace(valid=True, fine_payload_start_sample=start,
                               cfo_int_est=cfo, cfo_frac_est=0.0,
                               sfo_hat=0.0, sfo_cum_initial=0.0)

    def test_received_preamble_identifies_both_signs_of_integer_error(self):
        sf, os_factor = 7, 4
        symbol_samples = (1 << sf) * os_factor
        samples = np.concatenate([
            np.zeros(os_factor // 2, np.complex64),
            np.tile(build_upchirp(sf, 0, os_factor), 16),
            np.zeros(8 * symbol_samples, np.complex64),
        ])
        for error in (0, -1, 1):
            frame = self.frame(error, int(20.25 * symbol_samples))
            ranked = rank_residual_cfo_from_preamble(
                samples, build_residual_cfo_candidates(frame), sf=sf,
                os_factor=os_factor, preamble_symbols=16,
            )
            self.assertEqual(ranked[0].delta_cfo_bins, -error)
            self.assertGreater(ranked[0].preamble_score, ranked[1].preamble_score)
            for candidate in ranked:
                for field in ("fine_payload_start_sample", "sfo_hat", "sfo_cum_initial", "cfo_frac_est"):
                    self.assertEqual(getattr(candidate, field), getattr(frame, field))

    def test_absent_or_truncated_preamble_keeps_fixed_order(self):
        candidates = build_residual_cfo_candidates(self.frame())
        ranked = rank_residual_cfo_from_preamble(
            np.zeros(20, np.complex64), candidates, sf=7, os_factor=4, preamble_symbols=16,
        )
        self.assertEqual(ranked, candidates)
        self.assertEqual(build_residual_cfo_candidates(None), ())

    def test_zero_iq_keeps_original_candidate_first(self):
        candidates = build_residual_cfo_candidates(self.frame())
        ranked = rank_residual_cfo_from_preamble(
            np.zeros(13000, np.complex64), candidates, sf=7, os_factor=4, preamble_symbols=16,
        )
        self.assertEqual([c.delta_cfo_bins for c in ranked], [0, -1, 1])

    def test_crc_disabled_candidate_cannot_be_accepted(self):
        candidates = build_residual_cfo_candidates(self.frame())
        no_crc = SimpleNamespace(header_valid=True, header=SimpleNamespace(has_crc=False), crc_valid=True)
        accepted = SimpleNamespace(header_valid=True, header=SimpleNamespace(has_crc=True), crc_valid=True)
        with patch("weak_decoder.os_lora.system.residual_cfo.decode_soft_hamming_sync_candidate",
                   side_effect=[no_crc, accepted]) as decoder:
            result = decode_residual_cfo_candidates(
                np.zeros(1, np.complex64), candidates, sf=7, bw_hz=125000, os_factor=4,
            )
        self.assertEqual(decoder.call_count, 2)
        self.assertEqual(result.selected.candidate.delta_cfo_bins, -1)

    def test_budget_one_never_probes_alternatives(self):
        invalid = SimpleNamespace(header_valid=False)
        with patch("weak_decoder.os_lora.system.residual_cfo.decode_soft_hamming_sync_candidate",
                   return_value=invalid) as decoder:
            result = decode_residual_cfo_candidates(
                np.zeros(1, np.complex64), build_residual_cfo_candidates(self.frame()),
                sf=7, bw_hz=125000, os_factor=4, max_attempts=1,
            )
        self.assertEqual(decoder.call_count, 1)
        self.assertIsNone(result.selected)


if __name__ == "__main__":
    unittest.main()

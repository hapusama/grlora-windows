"""Tests for the gated robust sparse-corruption demodulator."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.chirp import build_upchirp
from weak_decoder.decoding.robust_sparse_demod import (
    RobustSparseConfig,
    demod_robust_sparse_symbol,
)


class RobustSparseDemodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sf = 7
        self.os_factor = 4
        self.true_bin = 37

    def test_clean_symbol_keeps_savaux_decision(self) -> None:
        samples = build_upchirp(self.sf, self.true_bin, self.os_factor)
        result = demod_robust_sparse_symbol(
            samples,
            start_sample=0,
            sf=self.sf,
            os_factor=self.os_factor,
        )

        self.assertEqual(self.true_bin, result.savaux_bin)
        self.assertEqual(result.savaux_bin, result.selected_bin)
        self.assertFalse(result.gate_triggered)
        self.assertFalse(result.changed_from_savaux)

    def test_awgn_gate_falls_back_exactly_to_savaux(self) -> None:
        rng = np.random.default_rng(20260819)
        clean = build_upchirp(self.sf, self.true_bin, self.os_factor)
        noise = (
            rng.normal(0.0, np.sqrt(0.5), clean.size)
            + 1j * rng.normal(0.0, np.sqrt(0.5), clean.size)
        )
        samples = (clean + noise).astype(np.complex64)
        result = demod_robust_sparse_symbol(
            samples,
            start_sample=0,
            sf=self.sf,
            os_factor=self.os_factor,
        )

        self.assertEqual(self.true_bin, result.savaux_bin)
        self.assertFalse(result.gate_triggered)
        self.assertEqual(result.savaux_bin, result.selected_bin)

    def test_sparse_wrong_chirp_burst_is_rejected(self) -> None:
        clean = build_upchirp(self.sf, self.true_bin, self.os_factor).astype(
            np.complex128
        )
        wrong_bin = 81
        jammer = build_upchirp(self.sf, wrong_bin, self.os_factor).astype(
            np.complex128
        )
        burst_samples = int(round(0.04 * clean.size))
        burst_start = 37
        corrupted = clean.copy()
        corrupted[burst_start : burst_start + burst_samples] += (
            24.0 * jammer[burst_start : burst_start + burst_samples]
        )
        result = demod_robust_sparse_symbol(
            corrupted,
            start_sample=0,
            sf=self.sf,
            os_factor=self.os_factor,
            config=RobustSparseConfig(),
        )

        self.assertNotEqual(self.true_bin, result.savaux_bin)
        self.assertTrue(result.gate_triggered)
        self.assertTrue(result.changed_from_savaux)
        self.assertEqual(self.true_bin, result.selected_bin)

    def test_invalid_config_is_rejected(self) -> None:
        samples = build_upchirp(self.sf, self.true_bin, self.os_factor)
        with self.assertRaises(ValueError):
            demod_robust_sparse_symbol(
                samples,
                start_sample=0,
                sf=self.sf,
                os_factor=self.os_factor,
                config=RobustSparseConfig(candidate_count=0),
            )


if __name__ == "__main__":
    unittest.main()

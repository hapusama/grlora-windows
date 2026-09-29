"""Regression tests for aliased narrowband trajectory cancellation."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.chirp import build_upchirp
from weak_decoder.decoding.alias_trim import (
    AliasTrimConfig,
    demod_alias_trim_symbol,
)


class AliasTrimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sf = 7
        self.os_factor = 4
        self.true_bin = 37
        self.sample_count = (1 << self.sf) * self.os_factor

    def test_clean_lora_does_not_open_blocker_gate(self) -> None:
        samples = build_upchirp(self.sf, self.true_bin, self.os_factor)
        result = demod_alias_trim_symbol(
            samples, 0, self.sf, self.os_factor
        )

        self.assertEqual(self.true_bin, result.savaux_bin)
        self.assertEqual(result.savaux_bin, result.selected_bin)
        self.assertFalse(result.gate_triggered)
        self.assertEqual((), result.blockers)

    def test_dense_awgn_falls_back_to_savaux(self) -> None:
        clean = build_upchirp(self.sf, self.true_bin, self.os_factor)
        for seed in range(12):
            rng = np.random.default_rng(seed)
            noise = (
                rng.normal(size=self.sample_count)
                + 1j * rng.normal(size=self.sample_count)
            ) / np.sqrt(2.0)
            result = demod_alias_trim_symbol(
                clean + noise,
                0,
                self.sf,
                self.os_factor,
            )
            self.assertFalse(result.gate_triggered)
            self.assertEqual(result.savaux_bin, result.selected_bin)

    def test_strong_aliased_tone_is_removed(self) -> None:
        normalized_frequency = 0.17321
        indexes = np.arange(self.sample_count, dtype=np.float64)
        rng = np.random.default_rng(3)
        samples = build_upchirp(
            self.sf, self.true_bin, self.os_factor
        ).astype(np.complex128)
        samples += 64.0 * np.exp(
            1j * (2.0 * np.pi * normalized_frequency * indexes + 0.7)
        )
        samples += (
            rng.normal(size=self.sample_count)
            + 1j * rng.normal(size=self.sample_count)
        ) / np.sqrt(2.0)
        result = demod_alias_trim_symbol(
            samples, 0, self.sf, self.os_factor
        )

        self.assertNotEqual(self.true_bin, result.savaux_bin)
        self.assertTrue(result.gate_triggered)
        self.assertTrue(result.changed_from_savaux)
        self.assertEqual(self.true_bin, result.selected_bin)
        self.assertGreater(result.clean_gain_db, 20.0)
        self.assertAlmostEqual(
            normalized_frequency,
            result.blockers[0].normalized_frequency,
            places=4,
        )

    def test_invalid_configuration_is_rejected(self) -> None:
        samples = build_upchirp(self.sf, self.true_bin, self.os_factor)
        with self.assertRaises(ValueError):
            demod_alias_trim_symbol(
                samples,
                0,
                self.sf,
                self.os_factor,
                config=AliasTrimConfig(fft_oversampling=0),
            )


if __name__ == "__main__":
    unittest.main()

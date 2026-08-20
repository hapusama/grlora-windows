"""Regression tests for the Savaux synchronization sensitivity experiment."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.baselines.savaux_oversampled import paper_oversampled_demod as savaux
from weak_decoder.os_lora.experiments import analyze_savaux_sync_sensitivity_ota as sensitivity


class SavauxFastPathTests(unittest.TestCase):
    def test_chirp_convolution_matches_dense_branch_dft(self) -> None:
        sf = 5
        os_factor = 4
        n_bins = 1 << sf
        rng = np.random.default_rng(123)
        branch = (
            rng.standard_normal(n_bins) + 1j * rng.standard_normal(n_bins)
        ).astype(np.complex64)
        for branch_index in range(os_factor):
            expected = (
                savaux._branch_dft_matrix(sf, os_factor, branch_index) @ branch
            ) / np.sqrt(float(n_bins))
            actual = savaux._paper_branch_spectrum(
                branch, sf, os_factor, branch_index
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


class FractionalTimingTests(unittest.TestCase):
    def test_integer_timing_offsets_are_exact_slices(self) -> None:
        guard = sensitivity.GUARD_SAMPLES
        count = sensitivity.SYMBOL_SAMPLES
        values = np.arange(count + 2 * guard, dtype=np.float32).astype(np.complex64)
        for offset in (-4, 0, 3):
            actual = sensitivity._fractional_timing_window(values, float(offset))
            expected = values[guard + offset : guard + offset + count]
            np.testing.assert_array_equal(actual, expected)

    def test_fractional_timing_has_expected_tone_phase(self) -> None:
        guard = sensitivity.GUARD_SAMPLES
        count = sensitivity.SYMBOL_SAMPLES
        indexes = np.arange(-guard, count + guard, dtype=np.float64)
        frequency = 0.03125
        values = np.exp(2j * np.pi * frequency * indexes).astype(np.complex64)
        delta = 0.375
        actual = sensitivity._fractional_timing_window(values, delta)
        expected = np.exp(
            2j * np.pi * frequency * (np.arange(count, dtype=np.float64) + delta)
        )
        np.testing.assert_allclose(actual[32:-32], expected[32:-32], atol=2e-5)

    def test_bandlimited_awgn_has_unit_power_and_no_out_of_band_bins(self) -> None:
        count = sensitivity.SYMBOL_SAMPLES + 2 * sensitivity.GUARD_SAMPLES
        noise = sensitivity._unit_lora_band_awgn(np.random.default_rng(9), count)
        spectrum = np.fft.fft(noise)
        passband = int(round(count / sensitivity.OS_FACTOR))
        if passband % 2:
            passband -= 1
        half = passband // 2
        self.assertLess(float(np.max(np.abs(spectrum[half : count - half]))), 2e-3)
        self.assertLess(abs(float(np.mean(np.abs(noise) ** 2)) - 1.0), 0.08)


if __name__ == "__main__":
    unittest.main()

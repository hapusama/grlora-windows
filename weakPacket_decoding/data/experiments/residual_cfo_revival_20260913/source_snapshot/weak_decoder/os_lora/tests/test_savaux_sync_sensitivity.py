"""Regression tests for the Savaux synchronization sensitivity experiment."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.baselines.savaux_oversampled import paper_oversampled_demod as savaux
from weak_decoder.chirp import build_upchirp
from weak_decoder.os_lora.experiments import analyze_savaux_sync_sensitivity_ota as sensitivity
from weak_decoder.synchronization import grlora_frame_sync
from weak_decoder.synchronization.grlora_frame_sync import estimate_virtual_phase_sfd_cfo
from weak_decoder.synchronization.preamble_detector import PreambleDetectorConfig
from weak_decoder.synchronization.preamble_detector import virtual_simo_preamble_spectrum


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

    def test_dechirped_helper_matches_existing_symbol_path(self) -> None:
        sf = 7
        os_factor = 4
        upchirp = build_upchirp(sf, symbol_id=0, os_factor=os_factor)
        downchirp = np.conjugate(upchirp)
        expected, expected_branches, _ = savaux.paper_oversampled_spectrum(
            upchirp, 0, sf, os_factor
        )
        actual, actual_branches = savaux.paper_oversampled_dechirped_spectrum(
            upchirp * downchirp, sf, os_factor
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        for actual_branch, expected_branch in zip(actual_branches, expected_branches):
            np.testing.assert_allclose(
                actual_branch, expected_branch, rtol=2e-6, atol=2e-6
            )


class VirtualPhaseSfdTests(unittest.TestCase):
    def test_polyphase_combine_equals_full_rate_sfd_matched_filter(self) -> None:
        sf = 7
        os_factor = 4
        n_bins = 1 << sf
        values = (
            np.random.default_rng(81).standard_normal(n_bins * os_factor)
            + 1j
            * np.random.default_rng(82).standard_normal(n_bins * os_factor)
        ).astype(np.complex64)
        actual = grlora_frame_sync._virtual_phase_sfd_spectrum(
            values, sf, os_factor
        )
        full = np.fft.fft(values)
        expected = np.asarray(
            [
                full[
                    grlora_frame_sync.signed_fft_bin(raw_bin, n_bins)
                    % (n_bins * os_factor)
                ]
                for raw_bin in range(n_bins)
            ],
            dtype=np.complex64,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_clean_known_sfd_returns_zero_integer_cfo(self) -> None:
        sf = 7
        os_factor = 4
        config = PreambleDetectorConfig(
            sf=sf,
            bw=125_000.0,
            samp_rate=125_000.0 * os_factor,
            win_chirps=8,
            hop_samples=None,
            min_periodic_peaks=6,
            bin_tol=2,
        )
        sfd = np.conjugate(build_upchirp(sf, symbol_id=0, os_factor=os_factor))
        cfo_int, signed_bin, margin_db = estimate_virtual_phase_sfd_cfo(
            sfd, 0, config, cfo_frac=0.0
        )
        self.assertEqual(cfo_int, 0)
        self.assertEqual(signed_bin, 0)
        self.assertGreater(margin_db, 20.0)


class VirtualSimoPreambleTests(unittest.TestCase):
    def test_polyphase_array_equals_full_rate_long_fft(self) -> None:
        chirps = 5
        n_bins = 32
        os_factor = 4
        rng = np.random.default_rng(902)
        values = (
            rng.standard_normal((chirps, n_bins * os_factor))
            + 1j * rng.standard_normal((chirps, n_bins * os_factor))
        ).astype(np.complex64)
        actual = virtual_simo_preamble_spectrum(values, os_factor=os_factor)
        expected = np.fft.fft(values.reshape(-1)).astype(np.complex64)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


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

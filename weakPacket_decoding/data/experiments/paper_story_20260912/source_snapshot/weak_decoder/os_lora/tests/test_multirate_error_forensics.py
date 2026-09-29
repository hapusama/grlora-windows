"""Regression tests for the OTA multi-rate error-bin forensic experiment."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.os_lora.experiments.archive import (
    analyze_multirate_error_bins_ota as forensic,
)


class MultiRateErrorForensicsTests(unittest.TestCase):
    def test_fft_coordinate_mapping_round_trips(self) -> None:
        for symbol in (0, 1, forensic.N_BINS // 2, forensic.N_BINS - 1):
            coordinate = forensic._candidate_coordinate(symbol)
            self.assertEqual(forensic._symbol_from_coordinate(coordinate), symbol)

    def test_candidate_feature_recovers_ideal_amplitude_ratio(self) -> None:
        symbol = forensic.N_BINS // 4
        coordinate = forensic._candidate_coordinate(symbol)
        spectra: dict[int, np.ndarray] = {}
        for q in forensic.RATES:
            spectrum = np.zeros(q * forensic.N_BINS, dtype=np.complex64)
            spectrum[coordinate] = forensic.N_BINS - symbol
            spectrum[(coordinate - forensic.N_BINS) % spectrum.size] = symbol
            spectra[q] = spectrum
        features = forensic._candidate_features(spectra, symbol, 1.0)
        for q in forensic.RATES:
            self.assertAlmostEqual(features[f"r_q{q}"], 0.25, places=7)
            self.assertAlmostEqual(features[f"dev_q{q}"], 0.0, places=7)
        self.assertAlmostEqual(features["ratio_variance"], 0.0, places=12)
        self.assertAlmostEqual(features["c_1m"], 0.0, places=12)
        self.assertAlmostEqual(features["c_multirate"], 0.0, places=12)

    def test_awgn_is_flat_in_band_and_zero_outside_lora_band(self) -> None:
        noise = forensic._unit_lora_band_awgn(np.random.default_rng(1234))
        spectrum = np.fft.fft(noise)
        half = forensic.N_BINS // 2
        outside = spectrum[half : spectrum.size - half]
        self.assertLess(float(np.max(np.abs(outside))), 1e-3)
        self.assertLess(abs(float(np.mean(np.abs(noise) ** 2)) - 1.0), 0.08)


if __name__ == "__main__":
    unittest.main()

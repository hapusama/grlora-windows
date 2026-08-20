"""Regression tests for the single-symbol multi-rate LoRa detector."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.chirp import build_upchirp
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)
from weak_decoder.os_lora.system.multirate_structure import (
    awgn_multirate_glrt_scores,
    build_multirate_spectra,
    coherent_ml_scores,
    detect_multirate_structure,
    fold_pair_components,
    fold_pair_energy_scores,
    fold_profile_scores,
    mapped_fft_argmax_scores,
)


class MultiRateStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sf = 6
        self.n_bins = 1 << self.sf
        self.source_os_factor = 8
        self.rates = (8, 4, 2, 1)

    def test_nested_views_match_native_rate_chirps(self) -> None:
        symbol = 19
        high = build_upchirp(self.sf, symbol, self.source_os_factor)
        observation = build_multirate_spectra(
            high,
            self.sf,
            self.source_os_factor,
            self.rates,
        )
        for rate in self.rates:
            stride = self.source_os_factor // rate
            expected = build_upchirp(self.sf, symbol, rate)
            np.testing.assert_allclose(high[::stride], expected, atol=1e-6)
            self.assertEqual(observation.spectrum(rate).size, self.n_bins * rate)

    def test_clean_pair_spacing_and_wrap_ratio(self) -> None:
        for symbol in (0, 1, 9, self.n_bins // 2, self.n_bins - 1):
            high = build_upchirp(self.sf, symbol, self.source_os_factor)
            observation = build_multirate_spectra(
                high,
                self.sf,
                self.source_os_factor,
                self.rates,
            )
            for rate in self.rates[:-1]:
                spectrum = observation.spectrum(rate)
                primary, secondary = fold_pair_components(
                    spectrum, self.sf, rate
                )
                self.assertIsNotNone(secondary)
                length = self.n_bins * rate
                primary_expected = rate * (self.n_bins - symbol) / np.sqrt(length)
                secondary_expected = rate * symbol / np.sqrt(length)
                self.assertAlmostEqual(
                    float(abs(primary[symbol])), float(primary_expected), places=4
                )
                self.assertAlmostEqual(
                    float(abs(secondary[symbol])), float(secondary_expected), places=4
                )
                primary_index = symbol
                secondary_index = symbol + length - self.n_bins
                self.assertEqual(
                    (primary_index - secondary_index) % length,
                    self.n_bins,
                )

    def test_clean_detectors_select_the_symbol(self) -> None:
        for symbol in (0, 1, 13, self.n_bins // 2, self.n_bins - 1):
            high = build_upchirp(self.sf, symbol, self.source_os_factor)
            result = detect_multirate_structure(
                high,
                self.sf,
                self.source_os_factor,
                self.rates,
            )
            self.assertEqual(result.selected_bin, symbol)
            for rate in self.rates:
                spectrum = result.observation.spectrum(rate)
                self.assertEqual(
                    int(np.argmax(mapped_fft_argmax_scores(spectrum, self.sf, rate))),
                    symbol,
                )
                self.assertEqual(
                    int(np.argmax(fold_pair_energy_scores(spectrum, self.sf, rate))),
                    symbol,
                )
                self.assertEqual(
                    int(np.argmax(fold_profile_scores(spectrum, self.sf, rate))),
                    symbol,
                )

    def test_fast_coherent_ml_matches_direct_template_bank(self) -> None:
        rng = np.random.default_rng(37)
        length = self.n_bins * self.source_os_factor
        samples = build_upchirp(self.sf, 23, self.source_os_factor).astype(
            np.complex128
        )
        samples += 0.35 * (
            rng.normal(size=length) + 1j * rng.normal(size=length)
        ) / np.sqrt(2.0)
        fast = coherent_ml_scores(samples, self.sf, self.source_os_factor)
        templates = np.stack(
            [
                build_upchirp(self.sf, candidate, self.source_os_factor)
                for candidate in range(self.n_bins)
            ]
        )
        direct = np.abs(templates.conj() @ samples) ** 2 / float(length)
        np.testing.assert_allclose(fast, direct, rtol=2e-6, atol=2e-6)

    def test_covariance_aware_multirate_glrt_collapses_to_source_pair(self) -> None:
        rng = np.random.default_rng(41)
        length = self.n_bins * self.source_os_factor
        samples = build_upchirp(self.sf, 17, self.source_os_factor)
        samples = samples + 0.5 * (
            rng.normal(size=length) + 1j * rng.normal(size=length)
        ) / np.sqrt(2.0)
        observation = build_multirate_spectra(
            samples,
            self.sf,
            self.source_os_factor,
            self.rates,
        )
        expected = fold_profile_scores(
            observation.spectrum(self.source_os_factor),
            self.sf,
            self.source_os_factor,
        )
        actual = awgn_multirate_glrt_scores(observation)
        np.testing.assert_array_equal(actual, expected)
        detected = detect_multirate_structure(
            samples,
            self.sf,
            self.source_os_factor,
            self.rates,
        )
        self.assertEqual(detected.fusion_mode, "awgn_glrt")
        np.testing.assert_array_equal(detected.scores, expected)

    def test_explicit_cross_rate_covariance_has_no_extra_awgn_signal(self) -> None:
        sf = 4
        n_bins = 1 << sf
        source = 8
        rates = (8, 4, 2, 1)
        length = n_bins * source
        rng = np.random.default_rng(43)
        samples = build_upchirp(sf, 7, source) + 0.6 * (
            rng.normal(size=length) + 1j * rng.normal(size=length)
        ) / np.sqrt(2.0)
        observation = build_multirate_spectra(samples, sf, source, rates)
        feature_columns: list[np.ndarray] = []
        for rate in rates:
            primary, secondary = fold_pair_components(
                observation.spectrum(rate), sf, rate
            )
            feature_columns.append(primary)
            if secondary is not None:
                feature_columns.append(secondary)
        candidate_observations = np.stack(feature_columns, axis=1)

        explicit_scores = np.empty(n_bins, dtype=np.float64)
        for candidate in range(n_bins):
            kernels: list[np.ndarray] = []
            for rate in rates:
                rate_length = n_bins * rate
                stride = source // rate
                indexes = np.arange(rate_length, dtype=np.int64)
                downchirp = np.conjugate(build_upchirp(sf, 0, rate))
                bins = (
                    (candidate,)
                    if rate == 1
                    else (candidate, candidate + rate_length - n_bins)
                )
                for fft_bin in bins:
                    kernel = np.zeros(length, dtype=np.complex128)
                    kernel[stride * indexes] = downchirp * np.exp(
                        -2j * np.pi * fft_bin * indexes / float(rate_length)
                    ) / np.sqrt(float(rate_length))
                    kernels.append(kernel)
            matrix = np.stack(kernels)
            covariance = matrix @ matrix.conj().T
            clean = build_upchirp(sf, candidate, source)
            steering = matrix @ clean
            inverse = np.linalg.pinv(covariance, hermitian=True, rcond=1e-10)
            weight = inverse @ steering
            information = float(np.real(np.vdot(steering, weight)))
            projection = np.vdot(weight, candidate_observations[candidate])
            explicit_scores[candidate] = abs(projection) ** 2 / information

        source_profile = fold_profile_scores(
            observation.spectrum(source), sf, source
        )
        np.testing.assert_allclose(
            explicit_scores, source_profile, rtol=2e-6, atol=2e-6
        )

    def test_coherent_ml_matches_existing_savaux_awgn_metric(self) -> None:
        rng = np.random.default_rng(53)
        length = self.n_bins * self.source_os_factor
        samples = build_upchirp(self.sf, 29, self.source_os_factor)
        samples = samples + 0.4 * (
            rng.normal(size=length) + 1j * rng.normal(size=length)
        ) / np.sqrt(2.0)
        ml = coherent_ml_scores(samples, self.sf, self.source_os_factor)
        savaux, _branches, _phase = paper_oversampled_spectrum(
            samples,
            0,
            self.sf,
            self.source_os_factor,
        )
        savaux_power = np.abs(savaux).astype(np.float64) ** 2
        np.testing.assert_allclose(
            ml / np.max(ml),
            savaux_power / np.max(savaux_power),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_invalid_non_nested_rate_is_rejected(self) -> None:
        samples = build_upchirp(self.sf, 0, self.source_os_factor)
        with self.assertRaises(ValueError):
            build_multirate_spectra(
                samples,
                self.sf,
                self.source_os_factor,
                rates=(8, 3, 1),
            )


if __name__ == "__main__":
    unittest.main()

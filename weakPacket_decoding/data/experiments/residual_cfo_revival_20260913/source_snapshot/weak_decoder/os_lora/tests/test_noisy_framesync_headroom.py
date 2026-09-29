"""Tests for the whole-packet noisy FrameSync headroom experiment."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from weak_decoder.os_lora.experiments import analyze_noisy_framesync_headroom_ota as headroom


class NoisyFrameSyncHeadroomTests(unittest.TestCase):
    def test_gate_failure_audit_reports_each_hard_condition(self) -> None:
        config = SimpleNamespace(
            preamble_symbols=16,
            bin_tolerance=2,
            sync_bin_tolerance=4,
            sfd_bin_tolerance=4,
        )
        location = SimpleNamespace(
            preamble_stable_count=13,
            sync1_distance=5,
            sync2_distance=0,
            sfd_bin_distance=6,
        )
        frame_sync = SimpleNamespace(
            preamble_bin0_count=15,
            preamble_peak_count=16,
            netid_valid=False,
        )
        failures = headroom._sync_gate_failures(
            SimpleNamespace(frame_location=location, frame_sync=frame_sync), config
        )
        self.assertEqual(
            failures,
            (
                "locator_preamble_stability",
                "locator_sync1",
                "locator_sfd",
                "framesync_preamble_all_bin0",
                "framesync_netid",
            ),
        )

    def test_whole_packet_awgn_is_unit_power_and_bandlimited(self) -> None:
        count = headroom.SYMBOL_SAMPLES
        noise = headroom._unit_lora_band_awgn(np.random.default_rng(19), count)
        spectrum = np.fft.fft(noise)
        passband = int(round(count / headroom.OS_FACTOR))
        half = passband // 2
        self.assertLess(float(np.max(np.abs(spectrum[half : count - half]))), 2e-3)
        self.assertLess(abs(float(np.mean(np.abs(noise) ** 2)) - 1.0), 0.08)

    def test_summary_separates_local_and_catastrophic_estimates(self) -> None:
        packets = [
            {
                "esn0_db": 14.0,
                "strict_sync_success": 1,
                "estimate_available": 1,
                "local_sync_estimate": 1,
                "cfo_error_bins": 0.4,
                "abs_cfo_error_bins": 0.4,
                "payload_sto_error_samples_1m": 1,
            },
            {
                "esn0_db": 14.0,
                "strict_sync_success": 0,
                "estimate_available": 1,
                "local_sync_estimate": 0,
                "cfo_error_bins": 10.0,
                "abs_cfo_error_bins": 10.0,
                "payload_sto_error_samples_1m": 80,
            },
        ]
        symbols = [
            {
                "esn0_db": 14.0,
                "local_sync_estimate": 1,
                "oracle_correct": 1,
                "noisy_sync_correct": 0,
                "strict_end_to_end_correct": 0,
            },
            {
                "esn0_db": 14.0,
                "local_sync_estimate": 0,
                "oracle_correct": 1,
                "noisy_sync_correct": 0,
                "strict_end_to_end_correct": 0,
            },
        ]
        row = headroom._summary_by_snr(packets, symbols, (14.0,))[0]
        self.assertEqual(row["estimate_available_rate"], 1.0)
        self.assertEqual(row["local_estimate_rate"], 0.5)
        self.assertEqual(row["catastrophic_fraction_given_estimate"], 0.5)
        self.assertEqual(row["local_fraction_abs_cfo_0p25_to_0p75"], 1.0)
        self.assertEqual(row["strict_success_rate_given_local_estimate"], 1.0)
        self.assertEqual(
            row["fraction_abs_cfo_sto_coupled_0p25_to_0p75"], 0.5
        )
        self.assertEqual(row["local_conditional_savaux_ser_gap"], 1.0)
        self.assertEqual(row["strict_end_to_end_ser"], 1.0)


if __name__ == "__main__":
    unittest.main()

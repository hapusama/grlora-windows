"""Tests for decoder-aware FrameSync gate analysis."""

from __future__ import annotations

import unittest

from weak_decoder.os_lora.experiments import analyze_framesync_gate_audit as audit


class FrameSyncGateAuditTests(unittest.TestCase):
    def test_decoder_aware_gate_drops_only_bin0_purity(self) -> None:
        row = {
            "estimate_available": "1",
            "locator_valid": "1",
            "netid_valid": "1",
            "coarse_preamble_bin0_count": "14",
            "coarse_preamble_peak_count": "16",
        }
        self.assertTrue(audit._decoder_aware_accept(row))
        row["netid_valid"] = "0"
        self.assertFalse(audit._decoder_aware_accept(row))

    def test_policy_summary_counts_false_reject_recovery(self) -> None:
        packets = [
            {
                "trial_id": "a",
                "esn0_db": "15",
                "strict_sync_success": "0",
                "estimate_available": "1",
                "locator_valid": "1",
                "netid_valid": "1",
            },
            {
                "trial_id": "b",
                "esn0_db": "15",
                "strict_sync_success": "1",
                "estimate_available": "1",
                "locator_valid": "1",
                "netid_valid": "1",
            },
        ]
        symbols = [
            {
                "trial_id": trial_id,
                "noisy_sync_correct": "1",
            }
            for trial_id in ("a", "b")
            for _ in range(4)
        ]
        augmented = audit._augment_packets(packets, symbols)
        rows = audit._policy_summary(augmented, (15.0,))
        by_policy = {row["policy"]: row for row in rows}
        self.assertEqual(by_policy["strict"]["packet_delivery_rate"], 0.5)
        self.assertEqual(
            by_policy["strict"]["decodable_false_rejects"], 1
        )
        self.assertEqual(
            by_policy["decoder_aware"]["packet_delivery_rate"], 1.0
        )
        self.assertEqual(by_policy["decoder_aware"]["end_to_end_ser"], 0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from weak_decoder.chirp import build_upchirp
from weak_decoder.os_lora.system.ambiguity_ridge_list import (
    AmbiguityRidgeCandidate,
    arbitrate_sync_list_with_crc,
    build_ambiguity_ridge_sync_list,
)


class AmbiguityRidgeListTests(unittest.TestCase):
    @staticmethod
    def _candidate(index: int) -> AmbiguityRidgeCandidate:
        return AmbiguityRidgeCandidate(
            candidate_index=index,
            source="test",
            sfd_peak_rank=index + 1,
            sfd_peak_signed_bin=0,
            sfd_relative_power_db=0.0,
            cfo_int_est=0,
            cfo_frac_est=0.0,
            cfo_total_est=0.0,
            cfo_hz_est=0.0,
            sfo_hat=0.0,
            sfo_cum_initial=0.0,
            fine_preamble_start_sample=0,
            fine_payload_start_sample=0,
            netid1_est=0,
            netid2_est=0,
            netid_offset=0,
            netid_valid=True,
            netid_margin_db=0.0,
            delta_cfo_bins=0.0,
            delta_payload_samples=0,
            ridge_residual_bins=0.0,
        )

    def test_sfd_alternative_couples_integer_cfo_and_payload_timing(self) -> None:
        sf = 6
        n_bins = 1 << sf
        os_factor = 4
        symbol_samples = n_bins * os_factor
        synchronized_preamble = 512
        synchronized_sfd = synchronized_preamble + 18 * symbol_samples
        synchronized_payload = synchronized_sfd + int(2.25 * symbol_samples)
        samples = np.zeros(synchronized_payload + 8 * symbol_samples, dtype=np.complex64)

        # A downchirp carrying signed SFD bin -6 corresponds to CFO_int=-3.
        desired_cfo = -3
        signed_sfd_bin = 2 * desired_cfo
        chips = np.conjugate(build_upchirp(sf)) * np.exp(
            2j * np.pi * signed_sfd_bin * np.arange(n_bins) / n_bins
        )
        sfd2_start = synchronized_sfd + symbol_samples
        indexes = sfd2_start + os_factor // 2 + os_factor * np.arange(n_bins)
        samples[indexes] = chips.astype(np.complex64)

        selected_cfo = 5
        frame_sync = SimpleNamespace(
            cfo_int_est=selected_cfo,
            cfo_frac_est=0.0,
            cfo_total_est=float(selected_cfo),
            down_val_signed_bin=2 * selected_cfo,
            sto_frac_initial=0.0,
            sto_frac_used=0.0,
            sto_sample_correction=0,
            up_symbols_used=12,
            sfo_hat=0.0,
            sfo_cum_initial=0.0,
            synced_preamble_start_sample=synchronized_preamble,
            synced_sfd_start_sample=synchronized_sfd,
            synced_payload_start_sample=synchronized_payload,
            fine_preamble_start_sample=synchronized_preamble,
            fine_payload_start_sample=synchronized_payload
            + os_factor * selected_cfo,
            netid1_est=-1,
            netid2_est=-1,
            netid_offset=0,
            netid_valid=False,
        )
        result = build_ambiguity_ridge_sync_list(
            samples,
            frame_sync,
            sf=sf,
            bw_hz=125_000.0,
            os_factor=os_factor,
            center_frequency_hz=487_700_000.0,
            preamble_symbols=16,
            sync_word=0x12,
            top_k=2,
            sfd_peak_pool=8,
        )

        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.candidates[0].source, "selected")
        alternative = result.candidates[1]
        self.assertEqual(alternative.cfo_int_est, desired_cfo)
        self.assertEqual(alternative.sfd_peak_rank, 1)
        self.assertEqual(
            alternative.fine_payload_start_sample,
            synchronized_payload + os_factor * desired_cfo,
        )
        self.assertAlmostEqual(alternative.ridge_residual_bins, 0.0)

    def test_invalid_list_sizes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            build_ambiguity_ridge_sync_list(
                np.zeros(32, dtype=np.complex64),
                SimpleNamespace(),
                sf=6,
                bw_hz=125_000.0,
                os_factor=4,
                center_frequency_hz=487_700_000.0,
                top_k=0,
            )

    def test_crc_arbitration_stops_early_by_default(self) -> None:
        decoded = SimpleNamespace(
            header=SimpleNamespace(has_crc=True),
            header_valid=True,
            crc_valid=True,
        )
        candidates = (self._candidate(0), self._candidate(1))
        with patch(
            "weak_decoder.os_lora.system.ambiguity_ridge_list."
            "decode_savaux_sync_candidate",
            return_value=decoded,
        ) as decoder:
            result = arbitrate_sync_list_with_crc(
                np.zeros(1, dtype=np.complex64),
                candidates,
                sf=6,
                bw_hz=125_000.0,
                os_factor=4,
            )
        self.assertEqual(result.selected_index, 0)
        self.assertEqual(len(result.attempts), 1)
        decoder.assert_called_once()

    def test_crc_arbitration_can_audit_the_full_list(self) -> None:
        decoded = SimpleNamespace(
            header=SimpleNamespace(has_crc=True),
            header_valid=True,
            crc_valid=True,
        )
        candidates = (self._candidate(0), self._candidate(1))
        with patch(
            "weak_decoder.os_lora.system.ambiguity_ridge_list."
            "decode_savaux_sync_candidate",
            return_value=decoded,
        ) as decoder:
            result = arbitrate_sync_list_with_crc(
                np.zeros(1, dtype=np.complex64),
                candidates,
                sf=6,
                bw_hz=125_000.0,
                os_factor=4,
                stop_on_crc=False,
            )
        self.assertEqual(result.selected_index, 0)
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(decoder.call_count, 2)

    def test_unknown_decoder_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown list decoder"):
            arbitrate_sync_list_with_crc(
                np.zeros(1, dtype=np.complex64),
                (),
                sf=6,
                bw_hz=125_000.0,
                os_factor=4,
                decoder_mode="not-a-decoder",
            )


if __name__ == "__main__":
    unittest.main()

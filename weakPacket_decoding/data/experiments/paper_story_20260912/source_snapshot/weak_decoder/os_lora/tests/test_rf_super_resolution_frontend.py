"""Optional integration checks for the external RF-SR checkout."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

from weak_decoder.rf_super_resolution import (
    DEFAULT_SYNTHETIC_CHECKPOINT,
    RFSRFrontendConfig,
    RFSuperResolutionFrontend,
)


RFSR_REPO = os.environ.get("RFSR_REPO", "").strip()


@unittest.skipUnless(RFSR_REPO, "set RFSR_REPO to run the optional author-model test")
class RFSuperResolutionFrontendTests(unittest.TestCase):
    def test_shapes_and_chunk_boundaries(self) -> None:
        repo = Path(RFSR_REPO).resolve()
        rng = np.random.default_rng(1701)
        values = (
            rng.normal(size=300) + 1j * rng.normal(size=300)
        ).astype(np.complex64)
        common = {
            "repo_root": repo,
            "checkpoint_name": DEFAULT_SYNTHETIC_CHECKPOINT,
            "device": "cpu",
        }
        whole = RFSuperResolutionFrontend(
            RFSRFrontendConfig(**common, chunk_input_samples=1_000)
        )
        chunked = RFSuperResolutionFrontend(
            RFSRFrontendConfig(
                **common,
                chunk_input_samples=97,
                overlap_input_samples=68,
            )
        )
        for mode in ("interpolation", "rfsr"):
            expected = whole.transform(values, mode, snr_db=-20.0)
            actual = chunked.transform(values, mode, snr_db=-20.0)
            self.assertEqual((1_200,), actual.shape)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-7)


if __name__ == "__main__":
    unittest.main()

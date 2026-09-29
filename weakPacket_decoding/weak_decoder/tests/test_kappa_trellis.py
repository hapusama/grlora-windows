# -*- coding: utf-8 -*-
"""kappa_trellis 模块单测：与战 runner 内联实现的数值等价 + 冒烟。

运行：D:/mysoft2/miniconda3/envs/gr-lora/python.exe -m unittest
weak_decoder.tests.test_kappa_trellis -v
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding")

from weak_decoder.decoding.kappa_trellis import (
    FineGrid, KappaTrellisDemodulator, forward_backward, viterbi,
    emission_prominence)


class TestEquivalenceWithRunner(unittest.TestCase):
    """模块 FineGrid.symbol_spectrum ≡ battle_runner.wm（SF10 内联版）。"""

    def test_wm_equivalence_sf10(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "br", r"D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding\data"
                  r"\experiments\dera_battle_20260929\battle_runner.py")
        br = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(br)
        rng = np.random.default_rng(7)
        x = (rng.standard_normal(6 * 4096) + 1j * rng.standard_normal(6 * 4096))
        fg = FineGrid(10, 4, 5)
        for k in range(6):
            a = fg.symbol_spectrum(x, k)
            b = br.wm(x, k)
            self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-9),
                            f"wm 不一致 @k={k}")


class TestDemodulator(unittest.TestCase):
    def _synthetic(self, sf, ldro):
        """干净合成段：LDRO 时值 v 的 chirp 移位 4v+1（音落 4v+1）。"""
        n = 1 << sf
        nf = n * 4
        t = np.arange(nf) / 4.0
        rng = np.random.default_rng(sf * 3 + int(ldro))
        vals = rng.integers(0, n // 4 if ldro else n, 12)
        segs = []
        for v in vals:
            vv = (4 * int(v) + 1) if ldro else int(v)
            lin = np.where(t < (n - vv), (vv / n - 0.5) * t, (vv / n - 1.5) * t)
            segs.append(np.exp(2j * np.pi * (t ** 2 / (2 * n) + lin)))
        x = np.concatenate(segs).astype(np.complex128)
        return x, [int(v) for v in vals], n

    def test_sf10_clean_decode_all_readouts(self):
        x, vals, n = self._synthetic(10, False)
        demod = KappaTrellisDemodulator(10, 4)
        for readout in ("grid0", "viterbi", "bcjr"):
            rows = demod.demod_payload(x, 0, 12, readout=readout)
            got = [demod.value_from_bin(int(np.argmax(rows[k])), 0, False)
                   for k in range(12)]
            self.assertEqual(got, vals, f"{readout} 干净合成应全对")

    def test_sf11_ldro_clean_decode(self):
        x, vals, n = self._synthetic(11, True)
        demod = KappaTrellisDemodulator(11, 4)
        for readout in ("viterbi", "bcjr"):
            rows = demod.demod_payload(x, 0, 12, readout=readout)
            got = [demod.value_from_bin(int(np.argmax(rows[k])), 0, True)
                   for k in range(12)]
            self.assertEqual(got, vals, f"SF11 LDRO {readout} 应全对")

    def test_freeze_delta_roundtrip(self):
        x, vals, n = self._synthetic(10, False)
        demod = KappaTrellisDemodulator(10, 4)
        d = demod.freeze_delta(x, 0, 12, vals, ldro=False)
        self.assertEqual(d, 0)

    def test_posterior_rows_sum_to_one(self):
        rng = np.random.default_rng(1)
        lam = np.log(rng.random((9, 5)))
        post = forward_backward(lam)
        self.assertTrue(np.allclose(post.sum(axis=1), 1.0))
        path = viterbi(lam)
        self.assertEqual(path.shape, (9,))
        self.assertTrue(((path[1:] - path[:-1]) <= 1).all()
                        and ((path[1:] - path[:-1]) >= -1).all())


if __name__ == "__main__":
    unittest.main()

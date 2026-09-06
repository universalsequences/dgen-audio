"""Regressions for the phase-independent hat dynamics objective."""
import unittest
import numpy as np
from fit_hat import BandBalanceStats, DynamicsStats


class DynamicsStatsTests(unittest.TestCase):
    def setUp(self):
        self.sr = 48000
        self.target = np.random.default_rng(73).normal(0, 0.2, self.sr // 4)
        self.stats = DynamicsStats(self.target, self.sr)

    def test_identity_and_polarity_have_zero_loss(self):
        self.assertEqual(self.stats.loss(self.target), 0)
        self.assertAlmostEqual(self.stats.loss(-self.target), 0)

    def test_clipping_is_penalized_even_with_rms_restored(self):
        clipped = np.clip(self.target, -0.12, 0.12)
        for window in self.stats.slices:
            clipped[window] *= np.sqrt(np.mean(self.target[window] ** 2)
                                       / np.mean(clipped[window] ** 2))
        np.testing.assert_allclose(self.stats.stats(clipped)[:, 0], self.stats.ref[:, 0])
        self.assertGreater(self.stats.loss(clipped), 0.8)

    def test_gain_changes_level_not_kurtosis(self):
        quieter = self.stats.stats(self.target * 0.5)
        np.testing.assert_allclose(quieter[:, 1], self.stats.ref[:, 1])
        self.assertAlmostEqual(self.stats.loss(self.target * 0.5), np.log(2))


class BandBalanceStatsTests(unittest.TestCase):
    def setUp(self):
        self.sr = 48000
        self.target = np.random.default_rng(37).normal(0, 0.2, self.sr // 4)
        self.stats = BandBalanceStats(self.target, self.sr)

    def test_gain_and_polarity_do_not_change_balance(self):
        self.assertEqual(self.stats.loss(self.target), 0)
        self.assertAlmostEqual(self.stats.loss(-self.target * 0.1), 0)

    def test_missing_body_is_penalized_despite_loudness_matching(self):
        freq = np.fft.rfftfreq(len(self.target), 1 / self.sr)
        thin = np.fft.irfft(np.fft.rfft(self.target) * np.where(freq < 1500, 0.1, 1))
        thin *= np.sqrt(np.mean(self.target ** 2) / np.mean(thin ** 2))
        self.assertGreater(self.stats.loss(thin), 0.8)
        # Body bands must register as deficient, not be hidden by a per-bin floor.
        self.assertTrue(np.all(self.stats.ref[:, :3] > self.stats.stats(thin)[:, :3]))


if __name__ == '__main__':
    unittest.main()

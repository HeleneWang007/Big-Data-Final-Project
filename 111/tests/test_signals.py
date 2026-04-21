import unittest

import numpy as np
import pandas as pd

from index_enhancement.signals import (
    build_month_end_schedule,
    compute_momentum_signal,
    zscore_row,
)


class TestSignals(unittest.TestCase):
    def test_zscore_row_has_zero_mean_and_unit_std(self):
        row = pd.Series([1.0, 2.0, 3.0])
        out = zscore_row(row)
        self.assertAlmostEqual(float(out.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(out.std(ddof=0)), 1.0, places=7)

    def test_zscore_constant_row_returns_zero(self):
        row = pd.Series([5.0, 5.0, 5.0])
        out = zscore_row(row)
        self.assertTrue((out == 0.0).all())

    def test_zscore_handles_nan(self):
        row = pd.Series([1.0, np.nan, 3.0])
        out = zscore_row(row)
        self.assertTrue(np.isfinite(out.dropna()).all())

    def test_compute_momentum_signal_scales_factor_panel(self):
        idx = pd.bdate_range("2023-01-02", periods=320)
        close = pd.DataFrame(
            {
                "A": np.linspace(100, 180, len(idx)),
                "B": np.linspace(100, 140, len(idx)),
                "C": np.linspace(100, 110, len(idx)),
            },
            index=idx,
        )
        mom, factor_panel, mu_panel = compute_momentum_signal(close, alpha=0.001)

        self.assertEqual(mom.shape, close.shape)
        self.assertEqual(factor_panel.shape, close.shape)
        self.assertEqual(mu_panel.shape, close.shape)

        check_day = idx[-1]
        expected = (0.001 * factor_panel.loc[check_day]).fillna(0.0)
        pd.testing.assert_series_equal(mu_panel.loc[check_day], expected, check_names=False)

    def test_build_month_end_schedule_returns_valid_dates_and_start_idx(self):
        idx = pd.bdate_range("2023-01-02", periods=320)
        close = pd.DataFrame(
            {
                "A": np.linspace(100, 180, len(idx)),
                "B": np.linspace(90, 150, len(idx)),
            },
            index=idx,
        )

        rebalance_days, holdto_days, coverage, start_idx = build_month_end_schedule(close, min_coverage=0.8)

        self.assertEqual(len(rebalance_days), len(holdto_days))
        self.assertGreater(len(rebalance_days), 0)
        self.assertGreaterEqual(start_idx, 0)
        self.assertGreaterEqual(float(coverage.iloc[start_idx]), 0.8)

    def test_build_month_end_schedule_raises_when_no_valid_coverage(self):
        idx = pd.bdate_range("2024-01-02", periods=60)
        close = pd.DataFrame({"A": np.linspace(100, 120, len(idx))}, index=idx)

        with self.assertRaises(ValueError):
            build_month_end_schedule(close, min_coverage=0.8)


if __name__ == "__main__":
    unittest.main()

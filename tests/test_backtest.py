import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from index_enhancement.backtest import (
    BacktestResult,
    _compute_period_return,
    run_backtest,
    run_parameter_grid,
    summarize_backtest,
)


class TestBacktestUtilities(unittest.TestCase):
    def test_compute_period_return_reweights_after_missing_returns(self):
        dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
        close = pd.DataFrame(
            {
                "A": [100.0, 110.0],
                "B": [100.0, np.nan],
            },
            index=dates,
        )
        out = _compute_period_return(close, dates[0], dates[1], ["A", "B"], np.array([0.5, 0.5]))
        self.assertAlmostEqual(out, 0.10, places=7)

    def test_summarize_backtest_basic_stats(self):
        perf = pd.DataFrame({"port_ret": [0.01, 0.02, -0.01]})
        turnover = pd.Series([0.1, 0.2, 0.3])

        out = summarize_backtest(perf, turnover)
        self.assertIn("ann_return", out)
        self.assertIn("ann_vol", out)
        self.assertAlmostEqual(out["avg_turnover"], 0.2, places=7)
        self.assertAlmostEqual(out["median_turnover"], 0.2, places=7)

    @patch("index_enhancement.backtest.solve_lp")
    def test_run_backtest_returns_expected_shapes(self, mock_solve_lp):
        dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
        close = pd.DataFrame(
            {
                "A": [100.0, 110.0, 121.0],
                "B": [100.0, 100.0, 100.0],
            },
            index=dates,
        )
        mu_panel = pd.DataFrame(
            {
                "A": [0.01, 0.02, 0.00],
                "B": [0.00, -0.01, 0.00],
            },
            index=dates,
        )
        universe = ["A", "B"]
        S = np.array([[1.0], [1.0]])
        b = np.array([0.5, 0.5])
        rebalance_days = dates[:2]
        holdto_days = dates[1:]

        mock_solve_lp.side_effect = [
            (np.array([0.6, 0.4]), np.array([0.1, 0.1]), {"status": "ok"}),
            (np.array([0.7, 0.3]), np.array([0.1, 0.1]), {"status": "ok"}),
        ]

        result = run_backtest(
            close=close,
            mu_panel=mu_panel,
            universe=universe,
            S=S,
            b=b,
            rebalance_days=rebalance_days,
            holdto_days=holdto_days,
            start_idx=0,
            delta=0.05,
            tau=0.20,
            gamma_sector=0.10,
            c=0.0005,
            solver_name="highs",
        )

        self.assertEqual(result.weights.shape, (2, 2))
        self.assertEqual(result.perf.shape[0], 2)
        self.assertEqual(len(result.turnover), 2)
        self.assertEqual(len(result.benchmark_returns), 2)
        self.assertEqual(len(result.benchmark_cumulative), 2)
        self.assertAlmostEqual(float(result.perf.iloc[0]["port_ret"]), 0.06, places=7)

    def test_run_parameter_grid_writes_summary_and_output_files(self):
        perf = pd.DataFrame(
            {
                "port_ret": [0.01, 0.02],
                "cum_ret": [1.01, 1.0302],
            },
            index=pd.to_datetime(["2024-02-29", "2024-03-31"]),
        )
        perf.index.name = "date"
        dummy_result = BacktestResult(
            weights=pd.DataFrame([[0.5, 0.5]], columns=["A", "B"], index=[pd.Timestamp("2024-01-31")]),
            perf=perf,
            turnover=pd.Series([0.1], index=[pd.Timestamp("2024-01-31")], name="turnover"),
            benchmark_returns=pd.Series([0.01], index=[pd.Timestamp("2024-02-29")], name="benchmark_ret"),
            benchmark_cumulative=pd.Series([1.01], index=[pd.Timestamp("2024-02-29")]),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("index_enhancement.backtest.run_backtest", return_value=dummy_result):
                summary = run_parameter_grid(
                    close=pd.DataFrame(),
                    mu_panel=pd.DataFrame(),
                    universe=["A", "B"],
                    S=np.array([[1.0], [1.0]]),
                    b=np.array([0.5, 0.5]),
                    rebalance_days=pd.to_datetime(["2024-01-31"]),
                    holdto_days=pd.to_datetime(["2024-02-29"]),
                    start_idx=0,
                    grid_delta=[0.01],
                    grid_tau=[0.2],
                    grid_gamma=[0.02],
                    grid_cost=[0.0005],
                    results_dir=tmp,
                    solver_name="highs",
                )

            self.assertEqual(summary.shape[0], 1)
            self.assertIn("ann_IR", summary.columns)
            folder = Path(tmp) / "delta=0.01_tau=0.2_gamma=0.02_c=0.0005"
            self.assertTrue((folder / "weights.csv").exists())
            self.assertTrue((folder / "perf_full_period.csv").exists())
            self.assertTrue((folder / "turnover.csv").exists())


if __name__ == "__main__":
    unittest.main()

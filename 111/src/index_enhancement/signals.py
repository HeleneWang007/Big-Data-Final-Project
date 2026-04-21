from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_row(row: pd.Series) -> pd.Series:
    """Cross-sectional z-score with zero output for degenerate rows."""
    mean_val = row.mean(skipna=True)
    std_val = row.std(skipna=True, ddof=0)
    if std_val == 0 or np.isnan(std_val):
        return row * 0.0
    return (row - mean_val) / std_val


def compute_momentum_signal(close: pd.DataFrame, alpha: float = 0.0004):
    """Compute 12-1 momentum and the scaled expected-return panel."""
    mom = close.shift(21) / close.shift(252) - 1
    factor_panel = mom.apply(zscore_row, axis=1)
    mu_panel = (alpha * factor_panel).fillna(0.0)
    return mom, factor_panel, mu_panel


def build_month_end_schedule(close: pd.DataFrame, min_coverage: float = 0.80):
    """Build monthly rebalance dates and detect the first valid start date."""
    last_days = close.index.to_series().groupby(pd.Grouper(freq="M")).max()
    rebalance_days = last_days.values[:-1]
    holdto_days = last_days.values[1:]

    mom = close.shift(21) / close.shift(252) - 1
    coverage = mom.reindex(rebalance_days).notna().mean(axis=1)
    valid = np.flatnonzero(coverage.values >= min_coverage)
    if len(valid) == 0:
        raise ValueError("No rebalance date meets the requested coverage threshold.")
    start_idx = int(valid[0])
    return rebalance_days, holdto_days, coverage, start_idx

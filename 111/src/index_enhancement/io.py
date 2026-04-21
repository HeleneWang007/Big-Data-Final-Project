from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_result_folder(results_dir: str | Path, delta, tau, gamma_sector, c) -> Path:
    folder = Path(results_dir) / f"delta={delta}_tau={tau}_gamma={gamma_sector}_c={c}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def load_cumulative_from_saved(delta, tau, gamma_sector, c, results_dir: str | Path = "results") -> pd.Series:
    """Load cumulative return series for a stored parameter configuration."""
    folder = make_result_folder(results_dir, delta, tau, gamma_sector, c)
    path = folder / "perf_full_period.csv"
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df["cum_ret"]

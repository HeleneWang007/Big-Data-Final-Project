"""Optimization-based index enhancement package."""

from .data import (
    CSV_PATH,
    extract_sp500_table,
    fetch_sp500_from_wikipedia,
    build_sector_matrix,
    download_close_prices,
)
from .signals import zscore_row, compute_momentum_signal, build_month_end_schedule
from .optimize import solve_lp
from .backtest import (
    BacktestResult,
    run_backtest,
    summarize_backtest,
    run_parameter_grid,
)
from .io import load_cumulative_from_saved, make_result_folder

__all__ = [
    "CSV_PATH",
    "extract_sp500_table",
    "fetch_sp500_from_wikipedia",
    "build_sector_matrix",
    "download_close_prices",
    "zscore_row",
    "compute_momentum_signal",
    "build_month_end_schedule",
    "solve_lp",
    "BacktestResult",
    "run_backtest",
    "summarize_backtest",
    "run_parameter_grid",
    "load_cumulative_from_saved",
    "make_result_folder",
]

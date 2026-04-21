from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from index_enhancement import (
    CSV_PATH,
    build_month_end_schedule,
    build_sector_matrix,
    compute_momentum_signal,
    download_close_prices,
    fetch_sp500_from_wikipedia,
    run_parameter_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the index enhancement pipeline.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sp500 = fetch_sp500_from_wikipedia(CSV_PATH)
    sp500, sector_onehot, universe, S = build_sector_matrix(sp500)
    close = download_close_prices(universe, start=args.start, end=args.end)
    _, _, mu_panel = compute_momentum_signal(close)
    rebalance_days, holdto_days, coverage, start_idx = build_month_end_schedule(close)
    b = np.ones(len(universe)) / len(universe)

    grid_df = run_parameter_grid(
        close=close,
        mu_panel=mu_panel,
        universe=universe,
        S=S,
        b=b,
        rebalance_days=rebalance_days,
        holdto_days=holdto_days,
        start_idx=start_idx,
        grid_delta=[0.005, 0.010, 0.020],
        grid_tau=[0.10, 0.20, 0.30],
        grid_gamma=[0.01, 0.02],
        grid_cost=[0.0003, 0.0005],
        results_dir=args.results_dir,
    )
    output_path = Path(args.results_dir) / "grid_backtest_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(output_path, index=False)
    print(f"Saved summary results to {output_path}")
    print(f"First valid rebalance day: {rebalance_days[start_idx]} | coverage={coverage.iloc[start_idx]:.1%}")


if __name__ == "__main__":
    main()

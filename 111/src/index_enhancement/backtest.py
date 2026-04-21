from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import make_result_folder
from .optimize import solve_lp


@dataclass
class BacktestResult:
    weights: pd.DataFrame
    perf: pd.DataFrame
    turnover: pd.Series
    benchmark_returns: pd.Series
    benchmark_cumulative: pd.Series


def _compute_period_return(close: pd.DataFrame, d, d_next, universe: list[str], weights: np.ndarray) -> float:
    r_next_vec = close.loc[d_next, universe] / close.loc[d, universe] - 1.0
    r_next = r_next_vec.to_numpy(dtype=float)
    mask = np.isfinite(r_next)
    w_eff = weights.copy()
    w_eff[~mask] = 0.0
    total = w_eff.sum()
    if total > 0:
        w_eff /= total
    return float((w_eff * np.nan_to_num(r_next, nan=0.0)).sum())


def run_backtest(
    close: pd.DataFrame,
    mu_panel: pd.DataFrame,
    universe: list[str],
    S: np.ndarray,
    b: np.ndarray,
    rebalance_days,
    holdto_days,
    start_idx: int,
    delta: float,
    tau: float,
    gamma_sector: float,
    c: float,
    solver_name: str = "appsi_highs",
) -> BacktestResult:
    """Run the monthly backtest for one parameter set."""
    weights_rows, port_rets, turnovers = [], [], []
    w_prev = b.copy()

    for d, d_next in zip(rebalance_days[start_idx:], holdto_days[start_idx:]):
        mu = mu_panel.loc[d, universe].fillna(0.0).to_numpy(dtype=float)
        w, t, _ = solve_lp(
            mu_vec=mu,
            b_vec=b,
            S_mat=S,
            w_prev=w_prev,
            delta=delta,
            tau=tau,
            gamma_sector=gamma_sector,
            c=c,
            solver_name=solver_name,
        )
        port_rets.append((d_next, _compute_period_return(close, d, d_next, universe, w)))
        turnovers.append((d, float(t.sum())))
        weights_rows.append(pd.Series(w, index=universe, name=d))
        w_prev = w

    weights = pd.DataFrame(weights_rows)
    perf = pd.DataFrame(port_rets, columns=["date", "port_ret"]).set_index("date")
    perf["cum_ret"] = (1.0 + perf["port_ret"]).cumprod()
    turnover = pd.Series(dict(turnovers), name="turnover")

    bench_seq = []
    for d, d_next in zip(rebalance_days[start_idx:], holdto_days[start_idx:]):
        r_next_vec = close.loc[d_next, universe] / close.loc[d, universe] - 1.0
        r_next = r_next_vec.to_numpy(dtype=float)
        mask = np.isfinite(r_next)
        bench_seq.append(np.nanmean(r_next[mask]) if mask.any() else 0.0)
    benchmark_returns = pd.Series(bench_seq, index=holdto_days[start_idx:], name="benchmark_ret")
    benchmark_cumulative = (1.0 + benchmark_returns).cumprod()

    return BacktestResult(
        weights=weights,
        perf=perf,
        turnover=turnover,
        benchmark_returns=benchmark_returns,
        benchmark_cumulative=benchmark_cumulative,
    )


def summarize_backtest(perf: pd.DataFrame, turnover: pd.Series) -> dict[str, float]:
    """Compute basic annualized summary statistics."""
    mret = perf["port_ret"].astype(float)
    ann_return = (1.0 + mret).prod() ** (12 / len(mret)) - 1.0
    ann_vol = mret.std(ddof=0) * np.sqrt(12)
    ann_ir = ann_return / ann_vol if ann_vol > 0 else np.nan
    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "ann_IR": float(ann_ir),
        "avg_turnover": float(turnover.mean()),
        "median_turnover": float(turnover.median()),
    }


def run_parameter_grid(
    close: pd.DataFrame,
    mu_panel: pd.DataFrame,
    universe: list[str],
    S: np.ndarray,
    b: np.ndarray,
    rebalance_days,
    holdto_days,
    start_idx: int,
    grid_delta,
    grid_tau,
    grid_gamma,
    grid_cost,
    results_dir: str | Path = "results",
    solver_name: str = "appsi_highs",
) -> pd.DataFrame:
    """Run a full-grid backtest and save per-configuration results."""
    rows = []
    results_dir = Path(results_dir)

    for delta in grid_delta:
        for tau in grid_tau:
            for gamma_sector in grid_gamma:
                for c in grid_cost:
                    result = run_backtest(
                        close=close,
                        mu_panel=mu_panel,
                        universe=universe,
                        S=S,
                        b=b,
                        rebalance_days=rebalance_days,
                        holdto_days=holdto_days,
                        start_idx=start_idx,
                        delta=delta,
                        tau=tau,
                        gamma_sector=gamma_sector,
                        c=c,
                        solver_name=solver_name,
                    )
                    summary = summarize_backtest(result.perf, result.turnover)

                    folder = make_result_folder(results_dir, delta, tau, gamma_sector, c)
                    result.weights.to_csv(folder / "weights.csv")
                    result.perf.reset_index().rename(columns={"index": "date"}).to_csv(folder / "perf_full_period.csv", index=False)
                    result.turnover.to_csv(folder / "turnover.csv")

                    rows.append(
                        {
                            "delta": delta,
                            "tau": tau,
                            "gamma_sector": gamma_sector,
                            "cost_c": c,
                            **summary,
                        }
                    )

    return pd.DataFrame(rows).sort_values("ann_IR", ascending=False).reset_index(drop=True)

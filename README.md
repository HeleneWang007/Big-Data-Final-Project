# Optimization-Based Index Enhancement for the S&P 500

## Project Overview

This project develops an optimization-based index-enhancement strategy for the S&P 500 universe. The goal is to construct a portfolio that remains close to the S&P 500 benchmark while seeking modest excess return through factor-based stock selection.

The model adapts an index-enhancement framework originally designed for the Chinese A-share market to the U.S. equity market. Since S&P 500 constituents are generally tradable and are not subject to daily price limits, the U.S. version focuses on the core portfolio-construction constraints used in practical benchmark-aware investing: active-weight limits, sector neutrality, turnover control, and transaction-cost awareness.

At each monthly rebalancing date, the optimizer takes the current benchmark weights, a cross-sectional factor score, previous portfolio weights, and sector exposure data as inputs. It then solves a linear programming problem to produce a new portfolio weight vector.

## Research Motivation

Index enhancement aims to outperform a broad benchmark while preserving similar risk and exposure characteristics. For institutional investors, even small improvements over a benchmark can be economically meaningful when the capital base is large. However, simple factor tilting may introduce unintended sector bets, excessive turnover, or concentrated single-stock exposures.

This project uses optimization as the layer that converts factor signals into an implementable portfolio. The optimizer allows the strategy to express alpha views while explicitly controlling benchmark-relative risk, trading intensity, and sector drift.

## Optimization Model

The decision variable is the post-trade portfolio weight vector:

$$
w \in \mathbb{R}^N
$$

where $N$ is the number of stocks in the investable S&P 500 universe.

The expected return vector is constructed from a standardized cross-sectional factor score:

$$
\mu = \alpha f
$$

The objective maximizes expected return net of linear transaction costs:

$$
\max_w \ \mu^\top w - c \sum_{i=1}^{N} |w_i - w_i^{prev}|
$$

where:

- $w_i$ is the new portfolio weight of stock $i$
- $w_i^{prev}$ is the previous-period portfolio weight
- $c$ is the transaction-cost coefficient
- $f$ is the standardized factor score
- $\alpha$ scales the factor score into expected return

Because the absolute-value turnover term is not directly linear, the model introduces auxiliary variables $t_i$ such that:

$$
t_i \geq w_i - w_i^{prev}, \quad
t_i \geq -(w_i - w_i^{prev})
$$

The final model is a linear program.

## Main Constraints

The portfolio is subject to the following constraints:

1. **Full investment**

$$
\sum_i w_i = 1
$$

2. **Long-only and single-name cap**

$$
0 \leq w_i \leq 0.10
$$

3. **Stock-level active-weight bounds**

$$
-\delta \leq w_i - b_i \leq \delta
$$

where $b_i$ is the benchmark weight.

4. **Sector neutrality**

$$
-\gamma^{sector}\mathbf{1} \leq S^\top w - S^\top b \leq \gamma^{sector}\mathbf{1}
$$

where $S$ is the sector exposure matrix.

5. **Turnover cap**

$$
\sum_i t_i \leq \tau
$$

These constraints keep the optimized portfolio close to the benchmark while allowing controlled active tilts.

## Solution Workflow

The strategy is implemented as a monthly backtest.

For each rebalance date:

1. Load the current S&P 500 universe and benchmark weights.
2. Construct or load the sector exposure matrix.
3. Compute cross-sectional factor scores.
4. Fill missing scores with zero to preserve feasibility.
5. Solve the linear programming problem.
6. Clip small numerical errors and renormalize weights.
7. Compute realized portfolio return over the next holding period.
8. Record portfolio weights, returns, turnover, and diagnostics.
9. Repeat the process across all monthly rebalance dates.

The model is solved using Pyomo with an LP solver such as HiGHS, Gurobi, CBC, CPLEX, or XPRESS.

## Hyperparameter Grid Search

The project evaluates model performance over a grid of economically meaningful hyperparameters:

- Active-weight tolerance: $\delta \in \{0.005, 0.010, 0.020\}$
- Turnover cap: $\tau \in \{0.10, 0.20, 0.30\}$
- Sector-exposure tolerance: $\gamma^{sector} \in \{0.01, 0.02\}$
- Transaction-cost coefficient: $c \in \{0.0003, 0.0005\}$

This produces 36 parameter configurations.

The main evaluation metrics are:

- Annualized return
- Annualized volatility
- Annualized information ratio
- Tracking error
- Average monthly turnover
- Maximum sector deviation
- Number of binding active-weight constraints

## Main Empirical Findings

The highest-information-ratio configuration is:

$$
(\delta, \tau, \gamma^{sector}, c) = (0.020, 0.300, 0.010, 0.0003)
$$

This configuration achieves the highest annualized information ratio, but it also has relatively high turnover.

A more balanced configuration is:

$$
(\delta, \tau, \gamma^{sector}, c) = (0.020, 0.100, 0.010, 0.0003)
$$

This setting provides strong risk-adjusted performance while maintaining lower turnover and tighter benchmark discipline.

The results show that moderate flexibility in active weights and turnover allows the optimizer to express factor information effectively. Extremely tight constraints restrict the strategy’s ability to generate excess return, while excessive turnover may increase trading intensity without proportional performance improvement.

## Repository Structure

```text
Big-Data-Final-Project/
├── README.md
├── pyproject.toml
├── notebooks/
│   └── Optimization_final_project.ipynb
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── index_enhancement/
│       ├── __init__.py
│       ├── backtest.py
│       ├── data.py
│       ├── io.py
│       ├── optimize.py
│       └── signals.py
└── tests/
    ├── test_data.py
    ├── test_io.py
    ├── test_optimize.py
    └── test_signals.py

## Main Project Files

- `data.py`: constituents download/parsing, sector matrix construction, price download
- `signals.py`: momentum signal and cross-sectional z-scoring
- `optimize.py`: the LP portfolio optimizer
- `backtest.py`: monthly backtest runner and grid search
- `io.py`: saved-result loading utilities

## Installation

From the project root:

```bash
pip install -e .
```

## Running the pipeline

```bash
python scripts/run_pipeline.py --start 2018-01-01 --end 2025-01-01 --results-dir results
```

## Running the tests

With `unittest` discovery:

```bash
python -m unittest discover -s tests
```

Or with pytest:

```bash
pytest
```


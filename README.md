# Optimization-Based Index Enhancement Project

## Project purpose

The project studies an optimization-based index-enhancement strategy on the S&P 500. It builds a sector-constrained portfolio using momentum-based expected returns and linear programming, then evaluates full-period backtest performance over a grid of parameter settings.

## Package layout

```text
index_enhancement_project/
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
```

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


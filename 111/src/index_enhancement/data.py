from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf

CSV_PATH = "sp500_constituents.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def extract_sp500_table(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Find and normalize the S&P 500 constituents table."""
    tables = list(tables)
    if not tables:
        raise ValueError("No HTML tables were provided.")

    candidates = []
    for df in tables:
        cols_norm = [str(c).strip().lower() for c in df.columns]
        has_symbol = any(("symbol" in c) or ("ticker" in c) for c in cols_norm)
        has_sector = any(("gics" in c) and ("sector" in c) for c in cols_norm)
        if has_symbol and has_sector:
            candidates.append(df.copy())

    df = candidates[0] if candidates else max(tables, key=lambda x: (x.shape[0], x.shape[1])).copy()

    rename = {}
    for col in df.columns:
        lowered = str(col).strip().lower()
        if ("symbol" in lowered) or ("ticker" in lowered):
            rename[col] = "Symbol"
        elif lowered == "security" or "company" in lowered:
            rename[col] = "Security"
        elif ("gics" in lowered) and ("sector" in lowered):
            rename[col] = "GICS Sector"
        elif ("gics" in lowered) and ("sub" in lowered):
            rename[col] = "GICS Sub-Industry"
    df = df.rename(columns=rename)

    if "Symbol" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Symbol"})

    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df = df[df["Symbol"].ne("")]
    keep = [c for c in ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"] if c in df.columns]
    return df[keep] if keep else df


def fetch_sp500_from_wikipedia(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Fetch and optionally cache the S&P 500 constituents table."""
    if csv_path is not None and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(WIKI_URL, headers=headers, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(response.text)
    df = extract_sp500_table(tables)

    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def build_sector_matrix(sp500: pd.DataFrame):
    """Return cleaned constituents, one-hot sector matrix, universe, and numeric sector matrix."""
    df = sp500.copy()
    df["YahooTicker"] = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    sector_onehot = pd.get_dummies(
        df.set_index("YahooTicker")["GICS Sector"].astype(str),
        dtype=float,
    )
    universe = list(sector_onehot.index)
    S = sector_onehot.to_numpy(dtype=float)
    return df, sector_onehot, universe, S


def download_close_prices(universe: list[str], start: str, end: str, auto_adjust: bool = True) -> pd.DataFrame:
    """Download close prices for a ticker universe."""
    raw = yf.download(
        universe,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if "Close" in raw.columns:
        close = raw["Close"].copy()
    elif isinstance(raw, pd.DataFrame):
        close_like = [c for c in ["Adj Close", "Close", "close", "adjclose"] if c in raw.columns]
        if not close_like:
            raise RuntimeError(f"No close-like column found. Columns={list(raw.columns)}")
        close = raw[close_like[0]].copy()
    else:
        raise RuntimeError("Unexpected structure returned by yfinance download.")

    if isinstance(close, pd.Series):
        close = close.to_frame(name=universe[0])
    return close.sort_index()

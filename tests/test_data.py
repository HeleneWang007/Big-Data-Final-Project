import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from index_enhancement.data import (
    build_sector_matrix,
    download_close_prices,
    extract_sp500_table,
    fetch_sp500_from_wikipedia,
)


class TestExtractSP500Table(unittest.TestCase):
    def test_extracts_constituents_and_standardizes_columns(self):
        wrong = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        target = pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT"],
                "Company": ["Apple", "Microsoft"],
                "GICS Sector": ["Information Technology", "Information Technology"],
            }
        )
        out = extract_sp500_table([wrong, target])
        self.assertEqual(list(out.columns), ["Symbol", "Security", "GICS Sector"])
        self.assertEqual(out.loc[0, "Symbol"], "AAPL")
        self.assertEqual(out.loc[1, "Security"], "Microsoft")

    def test_falls_back_to_first_column_as_symbol(self):
        table = pd.DataFrame({"Code": ["AAA", "BBB"], "Sector": ["X", "Y"]})
        out = extract_sp500_table([table])
        self.assertIn("Symbol", out.columns)
        self.assertEqual(out["Symbol"].tolist(), ["AAA", "BBB"])

    def test_drops_blank_symbols(self):
        table = pd.DataFrame(
            {
                "Symbol": ["AAPL", " ", "MSFT"],
                "GICS Sector": ["Tech", "Tech", "Tech"],
            }
        )
        out = extract_sp500_table([table])
        self.assertEqual(out["Symbol"].tolist(), ["AAPL", "MSFT"])

    def test_raises_on_empty_tables(self):
        with self.assertRaises(ValueError):
            extract_sp500_table([])

    def test_renames_subindustry_when_present(self):
        table = pd.DataFrame(
            {
                "Ticker": ["AAPL"],
                "Company": ["Apple"],
                "GICS Sector": ["Tech"],
                "GICS Sub Industry": ["Consumer Electronics"],
            }
        )
        out = extract_sp500_table([table])
        self.assertEqual(
            list(out.columns),
            ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"],
        )


class TestDataUtilities(unittest.TestCase):
    def test_build_sector_matrix_outputs_expected_shapes(self):
        sp500 = pd.DataFrame(
            {
                "Symbol": ["BRK.B", "AAPL", "MSFT"],
                "GICS Sector": ["Financials", "Tech", "Tech"],
            }
        )
        df, sector_onehot, universe, S = build_sector_matrix(sp500)
        self.assertEqual(df["YahooTicker"].tolist(), ["BRK-B", "AAPL", "MSFT"])
        self.assertEqual(universe, ["BRK-B", "AAPL", "MSFT"])
        self.assertEqual(S.shape, (3, 2))
        self.assertAlmostEqual(float(sector_onehot.loc["AAPL", "Tech"]), 1.0)

    def test_fetch_sp500_uses_cached_csv_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sp500.csv"
            cached = pd.DataFrame({"Symbol": ["AAPL"], "GICS Sector": ["Tech"]})
            cached.to_csv(path, index=False)

            out = fetch_sp500_from_wikipedia(path)
            self.assertEqual(out["Symbol"].tolist(), ["AAPL"])

    @patch("index_enhancement.data.extract_sp500_table")
    @patch("index_enhancement.data.pd.read_html")
    @patch("index_enhancement.data.requests.get")
    def test_fetch_sp500_downloads_parses_and_saves(self, mock_get, mock_read_html, mock_extract):
        response = MagicMock()
        response.text = "<html></html>"
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        mock_read_html.return_value = [pd.DataFrame({"a": [1]})]
        expected = pd.DataFrame({"Symbol": ["MSFT"], "GICS Sector": ["Tech"]})
        mock_extract.return_value = expected

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "sp500.csv"
            out = fetch_sp500_from_wikipedia(path)

            self.assertTrue(path.exists())
            self.assertEqual(out.to_dict(orient="list"), expected.to_dict(orient="list"))
            mock_get.assert_called_once()
            mock_read_html.assert_called_once_with("<html></html>")
            mock_extract.assert_called_once()

    @patch("index_enhancement.data.yf.download")
    def test_download_close_prices_handles_multicolumn_close(self, mock_download):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        close_df = pd.DataFrame({"AAPL": [10.0, 11.0], "MSFT": [20.0, 21.0]}, index=idx)
        mock_download.return_value = pd.concat({"Close": close_df}, axis=1)

        out = download_close_prices(["AAPL", "MSFT"], "2024-01-01", "2024-01-10")
        self.assertEqual(list(out.columns), ["AAPL", "MSFT"])
        self.assertEqual(out.index.tolist(), idx.tolist())

    @patch("index_enhancement.data.yf.download")
    def test_download_close_prices_handles_single_series_output(self, mock_download):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        mock_download.return_value = pd.DataFrame({"Close": [10.0, 11.0]}, index=idx)

        out = download_close_prices(["AAPL"], "2024-01-01", "2024-01-10")
        self.assertEqual(list(out.columns), ["AAPL"])
        self.assertEqual(out.iloc[:, 0].tolist(), [10.0, 11.0])

    @patch("index_enhancement.data.yf.download")
    def test_download_close_prices_raises_when_no_close_like_column(self, mock_download):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        mock_download.return_value = pd.DataFrame({"Open": [10.0, 11.0]}, index=idx)

        with self.assertRaises(RuntimeError):
            download_close_prices(["AAPL"], "2024-01-01", "2024-01-10")


if __name__ == "__main__":
    unittest.main()

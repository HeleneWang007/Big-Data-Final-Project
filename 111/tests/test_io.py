import tempfile
import unittest

import pandas as pd

from index_enhancement.io import load_cumulative_from_saved, make_result_folder


class TestIO(unittest.TestCase):
    def test_make_result_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = make_result_folder(tmp, 0.01, 0.2, 0.02, 0.0005)
            self.assertTrue(folder.exists())
            self.assertIn("delta=0.01", str(folder))

    def test_load_cumulative_from_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = make_result_folder(tmp, 0.01, 0.2, 0.02, 0.0005)
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
                    "cum_ret": [1.01, 1.03],
                }
            )
            df.to_csv(folder / "perf_full_period.csv", index=False)
            out = load_cumulative_from_saved(0.01, 0.2, 0.02, 0.0005, results_dir=tmp)
            self.assertEqual(out.name, "cum_ret")
            self.assertEqual(len(out), 2)
            self.assertAlmostEqual(float(out.iloc[-1]), 1.03)

    def test_load_cumulative_from_saved_raises_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_cumulative_from_saved(0.01, 0.2, 0.02, 0.0005, results_dir=tmp)

    def test_load_cumulative_from_saved_raises_when_column_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = make_result_folder(tmp, 0.01, 0.2, 0.02, 0.0005)
            pd.DataFrame({"date": ["2024-01-31"], "port_ret": [0.01]}).to_csv(
                folder / "perf_full_period.csv", index=False
            )

            with self.assertRaises(KeyError):
                load_cumulative_from_saved(0.01, 0.2, 0.02, 0.0005, results_dir=tmp)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np

from index_enhancement.optimize import solve_lp


class DummySolver:
    def __init__(self, available, w_values=None, t_values=None):
        self._available = available
        self.w_values = w_values or []
        self.t_values = t_values or []

    def available(self):
        return self._available

    def solve(self, model, tee=False):
        for i, val in enumerate(self.w_values):
            model.w[i].set_value(val)
        for i, val in enumerate(self.t_values):
            model.t[i].set_value(val)
        return {"status": "ok"}


class TestOptimize(unittest.TestCase):
    def test_solve_lp_returns_mocked_solution(self):
        mu = np.array([0.03, 0.01, 0.00])
        b = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        S = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
        w_prev = b.copy()
        expected_w = np.array([0.38, 0.28, 0.34])
        expected_t = np.abs(expected_w - w_prev)

        with patch(
            "index_enhancement.optimize.pyo.SolverFactory",
            return_value=DummySolver(True, expected_w.tolist(), expected_t.tolist()),
        ):
            w, t, res = solve_lp(
                mu_vec=mu,
                b_vec=b,
                S_mat=S,
                w_prev=w_prev,
                delta=0.05,
                tau=0.20,
                gamma_sector=0.10,
                c=0.0005,
                solver_name="highs",
            )

        np.testing.assert_allclose(w, expected_w)
        np.testing.assert_allclose(t, expected_t)
        self.assertEqual(res["status"], "ok")

    def test_solve_lp_falls_back_to_glpk_when_primary_solver_unavailable(self):
        mu = np.array([0.03, 0.01])
        b = np.array([0.5, 0.5], dtype=float)
        S = np.array([[1], [1]], dtype=float)
        w_prev = b.copy()
        calls = []

        def factory(name):
            calls.append(name)
            if name == "highs":
                return DummySolver(False)
            if name == "glpk":
                return DummySolver(True, [0.5, 0.5], [0.0, 0.0])
            raise AssertionError(f"Unexpected solver name: {name}")

        with patch("index_enhancement.optimize.pyo.SolverFactory", side_effect=factory):
            w, t, _ = solve_lp(
                mu_vec=mu,
                b_vec=b,
                S_mat=S,
                w_prev=w_prev,
                solver_name="highs",
            )

        self.assertEqual(calls, ["highs", "glpk"])
        np.testing.assert_allclose(w, [0.5, 0.5])
        np.testing.assert_allclose(t, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

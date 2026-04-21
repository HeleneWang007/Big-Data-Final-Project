from __future__ import annotations

import numpy as np
import pyomo.environ as pyo


def solve_lp(
    mu_vec,
    b_vec,
    S_mat,
    w_prev,
    delta=0.01,
    tau=0.20,
    gamma_sector=0.02,
    c=0.0005,
    solver_name="highs",
):
    """Solve the one-period index-enhancement linear program."""
    n = len(mu_vec)
    k = S_mat.shape[1]
    sbmk = S_mat.T @ b_vec

    model = pyo.ConcreteModel()
    model.I = range(n)
    model.J = range(k)

    model.w = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.t = pyo.Var(model.I, domain=pyo.NonNegativeReals)

    model.obj = pyo.Objective(
        expr=sum(mu_vec[i] * model.w[i] for i in model.I) - c * sum(model.t[i] for i in model.I),
        sense=pyo.maximize,
    )
    model.full = pyo.Constraint(expr=sum(model.w[i] for i in model.I) == 1)

    model.stock_constraints = pyo.ConstraintList()
    for i in model.I:
        lb = max(0.0, b_vec[i] - delta)
        ub = min(0.10, b_vec[i] + delta)
        model.stock_constraints.add(model.w[i] >= lb)
        model.stock_constraints.add(model.w[i] <= ub)
        model.stock_constraints.add(model.t[i] >= model.w[i] - w_prev[i])
        model.stock_constraints.add(model.t[i] >= -(model.w[i] - w_prev[i]))

    model.sector_upper = pyo.ConstraintList()
    model.sector_lower = pyo.ConstraintList()
    for j in model.J:
        exposure_expr = sum(S_mat[i, j] * model.w[i] for i in model.I) - sbmk[j]
        model.sector_upper.add(exposure_expr <= gamma_sector)
        model.sector_lower.add(-exposure_expr <= gamma_sector)

    model.turncap = pyo.Constraint(expr=sum(model.t[i] for i in model.I) <= tau)

    opt = pyo.SolverFactory(solver_name)
    if (not opt.available()) and solver_name != "glpk":
        opt = pyo.SolverFactory("glpk")
    res = opt.solve(model, tee=False)

    w = np.array([pyo.value(model.w[i]) for i in model.I], dtype=float)
    t = np.array([pyo.value(model.t[i]) for i in model.I], dtype=float)
    return w, t, res

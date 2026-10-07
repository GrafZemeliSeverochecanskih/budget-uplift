"""Budget-constrained assignment of at most one e-mail per customer.

Problem (multiple-choice knapsack): x[i, a] in {0, 1}, sum_a x[i, a] <= 1,
sum_{i,a} cost[a] x[i, a] <= B, maximise sum_{i,a} tau[i, a] x[i, a].

Assumptions: tau-hat is taken at face value (no uncertainty; see ``robust_allocate`` for that);
effects are additive across customers (SUTVA, no cannibalisation); costs are known constants.
Assignments are coded 0 = no e-mail, a = treated arm a (column a-1 of tau).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FIGURES_DIR, CACHE_DIR, TABLES_DIR, arm_costs, budget_for, ensure_output_dirs, load_config
from .plotting import SLOTS, TEXT_2, apply_style, plt, save


@dataclass
class Allocation:
    assign: np.ndarray        # (n,) in {0, ..., K}
    objective: float          # sum of tau-hat over assigned (customer, arm)
    spend: float              # total cost
    seconds: float
    status: str = ""
    bound: float = float("nan")   # upper bound on the optimum (IP: best bound; Lagrangian: dual value)
    lam: float = float("nan")


def evaluate_assignment(assign: np.ndarray, tau: np.ndarray, cost: np.ndarray) -> tuple[float, float]:
    """Objective and spend of an assignment; ``cost`` has length K (treated arms only)."""
    treated = assign > 0
    rows = np.flatnonzero(treated)
    return float(tau[rows, assign[treated] - 1].sum()), float(cost[assign[treated] - 1].sum())


# exact -----------------------------------------------------------------------
def _useful_pairs(tau: np.ndarray, cost: np.ndarray, budget: float) -> list[tuple[int, int]]:
    """(customer, arm) pairs worth a variable: tau > 0 and affordable on their own.

    Dropping a tau <= 0 pair from any feasible solution keeps it feasible and does not lower the
    objective, so the optimum is unchanged by this pruning.
    """
    return [(int(i), int(a)) for i, a in zip(*np.nonzero(tau > 0)) if cost[a] <= budget]


def ip_allocate(tau: np.ndarray, cost: np.ndarray, budget: float, time_limit: float = 60.0,
                workers: int = 8, scale: int = 10_000, solver: str = "scip") -> Allocation:
    """Solve the integer program exactly: SCIP (LP-based branch and bound) or CP-SAT."""
    if solver == "cpsat":
        return cpsat_allocate(tau, cost, budget, time_limit, workers, scale)
    from ortools.linear_solver import pywraplp

    start = time.perf_counter()
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(int(time_limit * 1000))
    pairs = _useful_pairs(tau, cost, budget)
    x = [s.BoolVar(f"x{i}_{a}") for i, a in pairs]
    knap = s.Constraint(-s.infinity(), float(budget))
    obj = s.Objective()
    one_email: dict[int, pywraplp.Constraint] = {}
    for (i, a), v in zip(pairs, x):
        knap.SetCoefficient(v, float(cost[a]))
        obj.SetCoefficient(v, float(tau[i, a]))
        if i not in one_email:
            one_email[i] = s.Constraint(0, 1)
        one_email[i].SetCoefficient(v, 1)
    obj.SetMaximization()
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(params.RELATIVE_MIP_GAP, 1e-6)
    status = s.Solve(params)
    assign = np.zeros(len(tau), dtype=int)
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for (i, a), v in zip(pairs, x):
            if v.solution_value() > 0.5:
                assign[i] = a + 1
    names = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE"}
    obj_val, spend = evaluate_assignment(assign, tau, cost)
    bound = obj.BestBound() if status in names else float("nan")
    return Allocation(assign, obj_val, spend, time.perf_counter() - start,
                      names.get(status, f"STATUS_{status}"), bound=bound)


def cpsat_allocate(tau: np.ndarray, cost: np.ndarray, budget: float, time_limit: float = 60.0,
                   workers: int = 8, scale: int = 10_000) -> Allocation:
    """Same integer program with OR-Tools CP-SAT.

    CP-SAT needs integer data, so tau and costs are multiplied by ``scale`` and rounded (budget is
    floored so the scaled solution is always feasible in the original units).
    """
    from ortools.sat.python import cp_model

    start = time.perf_counter()
    n, k = tau.shape
    model = cp_model.CpModel()
    tau_int = np.round(tau * scale).astype(np.int64)
    cost_int = np.round(cost * scale).astype(np.int64)
    pairs = [(i, a) for i, a in _useful_pairs(tau, cost, budget) if tau_int[i, a] > 0]
    x = {p: model.NewBoolVar("") for p in pairs}
    by_customer: dict[int, list] = {}
    for (i, a), var in x.items():
        by_customer.setdefault(i, []).append(var)
    for vars_ in by_customer.values():
        if len(vars_) > 1:
            model.AddAtMostOne(vars_)
    model.Add(sum(int(cost_int[a]) * v for (i, a), v in x.items()) <= int(np.floor(budget * scale)))
    model.Maximize(sum(int(tau_int[i, a]) * v for (i, a), v in x.items()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.Solve(model)
    assign = np.zeros(n, dtype=int)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (i, a), v in x.items():
            if solver.Value(v):
                assign[i] = a + 1
    obj, spend = evaluate_assignment(assign, tau, cost)
    return Allocation(assign, obj, spend, time.perf_counter() - start, solver.StatusName(status),
                      bound=solver.BestObjectiveBound() / scale)


def brute_force(tau: np.ndarray, cost: np.ndarray, budget: float) -> tuple[float, np.ndarray]:
    """Enumerate all (K+1)^n assignments. Only for tiny test instances."""
    n, k = tau.shape
    best, best_assign = 0.0, np.zeros(n, dtype=int)
    for combo in itertools.product(range(k + 1), repeat=n):
        assign = np.array(combo)
        obj, spend = evaluate_assignment(assign, tau, cost)
        if spend <= budget + 1e-12 and obj > best:
            best, best_assign = obj, assign
    return best, best_assign


# heuristic -------------------------------------------------------------------
def lagrangian_choice(tau: np.ndarray, cost: np.ndarray, lam: float) -> np.ndarray:
    """a*(i) = argmax over {no e-mail: 0} U {tau[i, a] - lam * cost[a]}."""
    scores = np.column_stack([np.zeros(len(tau)), tau - lam * cost])
    return scores.argmax(axis=1)


def dual_value(tau: np.ndarray, cost: np.ndarray, budget: float, lam: float) -> float:
    """Lagrangian dual L(lam) = sum_i max(0, max_a tau - lam c) + lam B; an upper bound for lam >= 0."""
    return float(np.maximum(0, (tau - lam * cost).max(axis=1)).sum() + lam * budget)


def lagrangian_allocate(tau: np.ndarray, cost: np.ndarray, budget: float, max_iter: int = 100) -> Allocation:
    """Binary-search the smallest lam >= 0 whose greedy choice spends <= B.

    Spend is non-increasing in lam, so bisection works; at lam_hi = max(tau/c) + 1 nobody is
    e-mailed. The returned solution is always feasible; leftover budget is the price of not
    splitting ties/fractional customers (the gap to the IP).
    """
    start = time.perf_counter()

    def spend_at(lam: float) -> float:
        return evaluate_assignment(lagrangian_choice(tau, cost, lam), tau, cost)[1]

    lo, hi = 0.0, float(max(tau.max() / cost.min(), 0.0) + 1.0)
    if spend_at(lo) <= budget:
        hi = lo
    else:
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            if spend_at(mid) <= budget:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-12:
                break
    assign = lagrangian_choice(tau, cost, hi)
    obj, spend = evaluate_assignment(assign, tau, cost)
    return Allocation(assign, obj, spend, time.perf_counter() - start, "feasible",
                      bound=dual_value(tau, cost, budget, hi), lam=hi)


# experiment ------------------------------------------------------------------
def selected_tau_all() -> np.ndarray:
    """tau-hat for spend from the train-selected learner, fit on train, predicted for all rows.

    Used only for the solver comparison (runtime/objective), where in-sample predictions are fine.
    """
    from .cate import load_choice, make_learner
    from .data import load_hillstrom, train_test

    path = CACHE_DIR / "tau_all_spend.npy"
    if path.exists():
        return np.load(path)
    cfg = load_config()
    choice = load_choice()
    (Xtr, ttr, Ytr), _ = train_test()
    model = make_learner(choice["selected"]["spend"], choice["params"]["spend"], cfg["seed"])
    model.fit(Xtr.to_numpy(), ttr, Ytr["spend"].to_numpy())
    X, _, _ = load_hillstrom()
    tau = model.effect(X.to_numpy())
    np.save(path, tau)
    return tau


def compare(tau_all: np.ndarray, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    cost = arm_costs(cfg)[1:]
    a_cfg = cfg["allocate"]
    rows = []
    for n in a_cfg["runtime_sizes"]:
        tau = tau_all[rng.choice(len(tau_all), size=min(n, len(tau_all)), replace=False)]
        for frac in cfg["budget_fractions"]:
            budget = budget_for(frac, len(tau), cfg)
            ip = ip_allocate(tau, cost, budget, a_cfg["ip_time_limit_s"], a_cfg["ip_workers"])
            sols = {"ip_scip": ip,
                    "lagrangian": lagrangian_allocate(tau, cost, budget, a_cfg["lagrange_max_iter"])}
            if len(tau) <= a_cfg["cpsat_max_n"]:
                sols["ip_cpsat"] = cpsat_allocate(tau, cost, budget, a_cfg["cpsat_time_limit_s"],
                                                  a_cfg["ip_workers"], a_cfg["int_scale"])
            for method, sol in sols.items():
                rows.append({
                    "n": len(tau), "budget_fraction": frac, "budget": budget, "method": method,
                    "objective": sol.objective, "spend": sol.spend, "upper_bound": sol.bound,
                    "seconds": sol.seconds, "status": sol.status, "lambda": sol.lam,
                    "n_emailed": int((sol.assign > 0).sum()),
                    "n_mens": int((sol.assign == 1).sum()), "n_womens": int((sol.assign == 2).sum()),
                    "gap_to_ip_pct": 100 * (1 - sol.objective / ip.objective) if ip.objective > 0 else 0.0,
                })
    return pd.DataFrame(rows)


def plot_runtime(df: pd.DataFrame) -> plt.Figure:
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    styles = {"ip_scip": (SLOTS[0], "o", "Integer program (SCIP)"), "lagrangian": (SLOTS[1], "s", "Lagrangian heuristic"),
              "ip_cpsat": (SLOTS[2], "^", "Integer program (CP-SAT)")}
    nonzero = df[df.budget_fraction > 0]
    for method, (color, marker, label) in styles.items():
        g = nonzero[nonzero.method == method].groupby("n")["seconds"]
        if g.ngroups == 0:
            continue
        med, lo, hi = g.median(), g.min(), g.max()
        ax1.plot(med.index, med.values, color=color, marker=marker, markersize=6, label=label)
        ax1.fill_between(med.index, lo.values, hi.values, color=color, alpha=0.15, linewidth=0)
    failed = nonzero[~nonzero.status.isin(["OPTIMAL", "FEASIBLE", "feasible"])]
    for (method, n), g in failed.groupby(["method", "n"]):
        total = ((nonzero.method == method) & (nonzero.n == n)).sum()
        ax1.annotate(f"{len(g)}/{total} budgets:\nno solution "
                     f"in {g.seconds.max():.0f}s", (n, g.seconds.median()), xytext=(8, -4),
                     textcoords="offset points", fontsize=7, color=TEXT_2, va="top")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Customers (n)")
    ax1.set_ylabel("Seconds (median; band = min–max over budgets > 0)")
    ax1.set_title("Solve time", loc="left")
    ax1.legend(loc="lower right")
    lg = df[(df.method == "lagrangian") & (df.budget_fraction > 0)]
    for color, (n, g) in zip((SLOTS[3], SLOTS[6], SLOTS[4]), lg.groupby("n")):
        ax2.plot(g.budget_fraction, g.gap_to_ip_pct, color=color, marker="o", markersize=5, label=f"n = {n:,}")
    ax2.set_xlabel("Budget (fraction of customers at mean e-mail cost)")
    ax2.set_ylabel("Objective gap to IP (%)")
    ax2.set_title("Lagrangian optimality gap", loc="left")
    ax2.legend(loc="upper right")
    fig.text(0.01, -0.03, "Objective = predicted incremental spend (train-selected learner). "
             "Gap = 1 − Lagrangian / IP.", color=TEXT_2, fontsize=8)
    fig.tight_layout()
    return fig


def run() -> pd.DataFrame:
    cfg = load_config()
    ensure_output_dirs()
    df = compare(selected_tau_all(), cfg)
    df.to_csv(TABLES_DIR / "allocation_comparison.csv", index=False)
    save(plot_runtime(df), FIGURES_DIR / "runtime.png")
    return df

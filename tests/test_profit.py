import copy

import numpy as np
import pytest

from uplift_alloc import profit
from uplift_alloc.config import load_config


def test_combine_repeats_matches_dml_rule():
    est = np.array([1.0, 2.0, 3.0])
    se = np.array([0.1, 0.1, 0.1])
    theta, s, lo, hi = profit.combine_repeats(est, se, 0.05)
    assert theta == 2.0
    assert s == pytest.approx(np.sqrt(np.median(0.01 + (est - 2.0) ** 2)))  # between-repeat spread included
    assert lo < theta < hi


def test_optimal_budget_finds_argmax_with_interval():
    rng = np.random.default_rng(0)
    fractions = [0.0, 0.5, 1.0]
    scores = rng.normal(size=(5000, 3)) * 0.1 + np.array([0.0, 1.0, 0.5])
    res = profit.optimal_budget(scores, fractions, reps=100, seed=0, alpha=0.05)
    assert res["optimal_fraction"] == 0.5
    assert res["fraction_ci_low"] == res["fraction_ci_high"] == 0.5  # unambiguous optimum


def simulate(n: int = 6000, seed: int = 0):
    """Arm 2 is profitable (m*tau > cost) for half the customers and loss-making for the rest."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = rng.integers(0, 3, n)
    tau = np.column_stack([np.full(n, 0.05), np.where(X[:, 0] > 0, 2.0, 0.15)])  # margin 0.3: 0.6 vs 0.045
    y = X[:, 1] + np.select([t == 1, t == 2], [tau[:, 0], tau[:, 1]], 0.0) + rng.normal(scale=0.5, size=n)
    return X, t, y, tau


@pytest.fixture(scope="module")
def tiny():
    cfg = copy.deepcopy(load_config())
    cfg["profit"].update(budget_fractions=[0.3, 1.2], margin_sensitivity=[], crossfit={"folds": 2, "repeats": 2})
    X, t, y, tau = simulate()
    params = {"n_estimators": 80, "num_leaves": 7, "min_child_samples": 50, "learning_rate": 0.1, "verbose": -1}
    reps = [profit.crossfit_repeat(X, t, y, "T", params, cfg, r, tau) for r in range(2)]
    return cfg, reps


def test_crossfit_covers_every_customer_and_profit_beats_revenue(tiny):
    cfg, reps = tiny
    name = profit.profit_label(cfg["profit"]["margin"])
    assert set(reps[0]) >= {name, "Revenue-targeted (v1.0)", "Random targeting", "Oracle (true τ, profit)"}
    emailed = {k: (reps[0][k]["cost"][:, 1] > 0).mean() for k in (name, "Revenue-targeted (v1.0)")}
    assert emailed["Revenue-targeted (v1.0)"] > emailed[name] + 0.2  # revenue also e-mails loss-making customers
    curves = profit.profit_curves(reps, cfg["profit"]["budget_fractions"], [cfg["profit"]["margin"]], 0.05, "sim")
    at = curves[curves.budget_fraction == 1.2].set_index("policy")
    # unconstrained: revenue targeting also e-mails loss-making customers; profit targeting does not
    assert at.loc[name, "true_profit"] > at.loc["Revenue-targeted (v1.0)", "true_profit"]
    assert at.loc[name, "profit_ci_low"] < at.loc[name, "true_profit"] < at.loc[name, "profit_ci_high"]


def test_paired_differences_and_optima_shapes(tiny):
    cfg, reps = tiny
    fr, m = cfg["profit"]["budget_fractions"], cfg["profit"]["margin"]
    d = profit.paired_differences(reps, fr, m, 0.05, "sim")
    assert len(d) == len(profit.BASELINES) * len(fr)
    o = profit.optimal_budgets(reps, fr, [m], {**cfg, "profit": {**cfg["profit"], "optimum_bootstrap_reps": 20}}, "sim")
    assert {"optimal_fraction", "fraction_ci_low", "true_optimal_fraction"} <= set(o.columns)

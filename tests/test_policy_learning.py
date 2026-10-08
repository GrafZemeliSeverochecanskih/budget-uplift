import copy

import numpy as np
import pytest

from uplift_alloc import policy_learning as pl
from uplift_alloc.config import load_config

PARAMS = {"n_estimators": 60, "num_leaves": 7, "min_child_samples": 50, "learning_rate": 0.1, "verbose": -1}


def simulate(n: int = 6000, seed: int = 0):
    """Arm 2 pays (2.0) only when x0 > 0; arm 1 pays a little everywhere; propensity 1/3."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = rng.integers(0, 3, n)
    tau = np.column_stack([np.full(n, 0.3), np.where(X[:, 0] > 0, 2.0, 0.0)])
    y = X[:, 1] + np.select([t == 1, t == 2], [tau[:, 0], tau[:, 1]], 0.0) + rng.normal(scale=0.5, size=n)
    return X, t, y, tau


def test_dr_pseudo_outcomes_recover_ate():
    X, t, y, tau = simulate(20_000)
    gamma = pl.dr_pseudo_outcomes(X, t, y, PARAMS, folds=2, seed=0)
    assert (gamma[:, 1:] - gamma[:, [0]]).mean(axis=0) == pytest.approx(tau.mean(axis=0), abs=0.05)


def test_tree_learns_rule_and_renders_it():
    X, t, y, tau = simulate()
    trees = pl.fit_budgeted_trees(X, np.column_stack([np.zeros(len(y)), tau]), np.array([0.0, 0.1, 0.1]),
                                  np.array([0.0]), depth=1, min_leaf=50, seed=0)
    tree = trees[0][1]
    assign = tree.predict(X)
    assert ((assign == 2) == (X[:, 0] > 0)).mean() > 0.98
    text = pl.render_tree(tree, ["x0", "x1", "x2"])
    assert "if x0 <=" in text and "Womens E-Mail" in text and "Mens E-Mail" in text


def test_pick_tree_takes_smallest_feasible_lambda():
    trees = [(0.0, "a", 0.09), (0.5, "b", 0.05), (1.0, "c", 0.01)]
    assert pl.pick_tree(trees, 0.06) == (0.5, "b")
    assert pl.pick_tree(trees, 0.001) == (1.0, "c")  # none fits: most restrictive


def test_crossfit_repeat_tree_beats_random(monkeypatch):
    cfg = copy.deepcopy(load_config())
    cfg["policy_learning"].update(budget_fractions=[0.3, 1.2], crossfit={"folds": 2, "repeats": 1},
                                  tree_depths=[2], lambda_grid={"zero": True, "min": 0.01, "max": 20, "size": 6},
                                  min_samples_leaf=50)
    cfg["profit"]["lgbm_params"] = PARAMS
    X, t, y, tau = simulate()
    out = pl.crossfit_repeat(X, t, y, "T", cfg, 0, tau)
    m = cfg["policy_learning"]["margin"]
    true = {k: (m * v["true_rev"] - v["cost"]).mean(axis=0) for k, v in out.items()}
    assert true[pl.tree_label(2)][1] > true["Random targeting"][1] + 0.1
    assert true[pl.eto_label("T")][1] > true["Random targeting"][1] + 0.1
    spend = out[pl.tree_label(2)]["cost"][:, 0].mean()
    assert spend < 1.25 * 0.3 * 0.09  # budgeted tree roughly respects the budget on new customers

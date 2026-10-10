import copy

import numpy as np
import pytest

from uplift_alloc import chance, chance15
from uplift_alloc.config import load_config

COST = np.array([0.0, 0.10, 0.08])
V = 5.0


def world(n: int, seed: int, sigma: float = 0.3):
    """RCT with known uptake p; p-hat = p * mean-one lognormal noise (calibrated on average, noisy per customer)."""
    rng = np.random.default_rng(seed)
    p = np.column_stack([np.full(n, 0.01), rng.uniform(0.01, 0.08, (n, 2))])
    p_hat = np.clip(p * rng.lognormal(-sigma**2 / 2, sigma, (n, 3)), 1e-4, 0.99)
    tau = 1.0 + rng.normal(0, 0.1, (n, 2))
    t = rng.integers(0, 3, n)
    conv = (rng.random(n) < p[np.arange(n), t]).astype(float)
    return p, p_hat, tau, t, conv


def test_kappa_recovers_selection_bias():
    p, p_hat, tau, t, conv = world(600_000, 0)  # large: the IPW numerator has ~sqrt(3 / sum p) relative error
    S, C, _ = chance15.score_cost_var(tau, p_hat, COST, V, 0.3)
    assign = chance.allocate_rowcost(S, C, 0.04 * len(t))
    num, den = chance15.kappa_from_selection(assign, t, conv, p_hat)
    sel = np.flatnonzero(assign > 0)
    true_ratio = p[sel, assign[sel]].sum() / p_hat[sel, assign[sel]].sum()
    assert true_ratio > 1.1  # selection on cheap-looking uptake under-estimates it
    assert num / den == pytest.approx(true_ratio, rel=0.07)


def test_recalibration_restores_the_overrun_target():
    eps, b = 0.05, 0.04
    # kappa from an independent calibration world, applied to a fresh campaign
    p, p_hat, tau, t, conv = world(600_000, 1)
    S, C, _ = chance15.score_cost_var(tau, p_hat, COST, V, 0.3)
    num, den = chance15.kappa_from_selection(chance.allocate_rowcost(S, C, b * len(t)), t, conv, p_hat)
    kappa = max(1.0, num / den)

    p, p_hat, tau, _, _ = world(20_000, 2)
    budget = b * len(p)
    S, C, Vv = chance15.score_cost_var(tau, p_hat, COST, V, 0.3)
    plug, _ = chance.chance_allocate(S, C, Vv, budget, eps)
    p_k = np.minimum(kappa * p_hat, 0.99)
    S_k, C_k, V_k = chance15.score_cost_var(tau, p_k, COST, V, 0.3)
    recal, _ = chance.chance_allocate(S_k, C_k, V_k, budget, eps)
    rng = np.random.default_rng(3)
    viol_plug = chance.violation_mc(plug, COST, p, V, budget, 2000, rng)
    viol_recal = chance.violation_mc(recal, COST, p, V, budget, 2000, rng)
    assert viol_plug > 0.3          # the v1.3 failure, reproduced
    assert viol_recal < 0.15        # back near epsilon (MC + kappa estimation error allowed)
    assert viol_recal < viol_plug


def test_kappa_floor_and_empty_selection():
    t = np.array([1, 2, 0])
    assert chance15.kappa_from_selection(np.zeros(3, dtype=int), t, np.ones(3), np.full((3, 3), 0.1)) == (0.0, 0.0)


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 4))
    t = rng.integers(0, 3, n)
    p = np.clip(0.02 + 0.02 * (X[:, :1] > 0) + np.zeros((n, 3)), 0, 1)
    conv = (rng.random(n) < p[np.arange(n), t]).astype(float)
    spend = conv * 100.0
    cfg = copy.deepcopy(load_config())
    cfg["chance"]["budget_per_customer"] = [0.04]
    cfg["chance_v15"].update(bootstrap_fits=3)
    params = {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 50, "learning_rate": 0.1, "verbose": -1}
    return X, t, spend, conv, cfg, params


def test_estimate_kappa_runs_and_is_floored(small):
    X, t, spend, conv, cfg, params = small
    k = chance15.estimate_kappa(X, t, spend, conv, "T", params, cfg, seed=0)
    assert set(k) == {0.04}
    assert k[0.04]["kappa"] >= 1.0 and k[0.04]["n_selected"] > 0


def test_bootstrap_upper_p_shape_and_range(small):
    X, t, spend, conv, cfg, params = small
    up = chance15.bootstrap_upper_p(X, t, conv, X[:100], params, fits=3, quantile=0.9, seed=0)
    assert up.shape == (100, 3) and (up > 0).all() and (up <= 0.99).all()

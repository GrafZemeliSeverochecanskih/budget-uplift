import numpy as np

from uplift_alloc import robust_allocate

PARAMS = {"n_estimators": 60, "num_leaves": 7, "min_child_samples": 50, "learning_rate": 0.1, "verbose": -1}


def test_bootstrap_tau_shape_and_lcb_below_mean():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 3))
    t = rng.integers(0, 3, n)
    y = X[:, 0] + (t == 1) * (X[:, 1] > 0) + rng.normal(size=n)
    draws = robust_allocate.bootstrap_tau("T", PARAMS, X, t, y, X[:200], fits=5, seed=0)
    assert draws.shape == (5, 200, 2)
    lcb = robust_allocate.lower_bound(draws, 0.1)
    assert (lcb <= draws.mean(axis=0) + 1e-12).all()


def test_summarise_quantile_and_ci_ordering():
    draws = np.random.default_rng(1).normal(1.0, 0.5, 10_000)
    s = robust_allocate.summarise(draws, 0.05, 0.05)
    assert s["ci_low"] < s["q05"] < s["mean"] < s["ci_high"]
    assert abs(s["q05"] - (1.0 - 1.645 * 0.5)) < 0.05

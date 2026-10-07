import numpy as np
import pytest

from uplift_alloc import cate, metrics

PARAMS = {"n_estimators": 150, "num_leaves": 7, "min_child_samples": 50, "learning_rate": 0.05, "verbose": -1}


def simulate(n: int = 9000, seed: int = 0):
    """Three-arm RCT (propensity 1/3) with heterogeneous effects of opposite sign across arms."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    t = rng.integers(0, 3, n)
    tau = np.column_stack([1.0 * (X[:, 0] > 0), np.clip(X[:, 1], -1, 1)])  # true tau_1, tau_2
    base = X[:, 2] + 0.5 * X[:, 3]
    y = base + np.select([t == 1, t == 2], [tau[:, 0], tau[:, 1]], 0.0) + rng.normal(scale=0.5, size=n)
    return X, t, y, tau


@pytest.fixture(scope="module")
def sim():
    X, t, y, tau = simulate()
    Xte, tte, yte, taute = simulate(seed=1)
    return X, t, y, Xte, tte, yte, taute


LEARNERS = {
    "S": lambda: cate.SLearner(PARAMS),
    "T": lambda: cate.TLearner(PARAMS),
    "X": lambda: cate.XLearner(PARAMS),
    "DR": lambda: cate.DRLearnerMulti(PARAMS),
    "CausalForest": lambda: cate.CausalForestMulti(PARAMS, n_estimators=100, min_samples_leaf=20),
}


@pytest.mark.parametrize("name", list(LEARNERS))
def test_learner_beats_random_on_qini(sim, name):
    X, t, y, Xte, tte, yte, taute = sim
    tau_hat = LEARNERS[name]().fit(X, t, y).effect(Xte)
    assert tau_hat.shape == (len(Xte), 2)
    rng = np.random.default_rng(0)
    for a in (1, 2):
        m = (tte == a) | (tte == 0)
        d = (tte[m] == a).astype(float)
        learned = metrics.qini_coefficient(tau_hat[m, a - 1], yte[m], d)
        random = np.mean([metrics.qini_coefficient(rng.random(m.sum()), yte[m], d) for _ in range(20)])
        oracle = metrics.qini_coefficient(taute[m, a - 1], yte[m], d)
        assert learned > random + 0.25 * (oracle - random), (name, a, learned, random, oracle)
    # and the estimates track the truth
    assert np.corrcoef(tau_hat[:, 0], taute[:, 0])[0, 1] > 0.6


def test_known_propensity_is_uniform():
    clf = cate.KnownPropensity().fit(np.zeros((6, 1)), np.array([0, 1, 2, 0, 1, 2]))
    assert np.allclose(clf.predict_proba(np.zeros((4, 1))), 1 / 3)


def test_qini_endpoint_equals_ate_times_treated_share():
    rng = np.random.default_rng(3)
    n = 20_000
    d = rng.integers(0, 2, n).astype(float)
    y = 1.0 + 0.5 * d + rng.normal(size=n)
    x, q = metrics.qini_curve(rng.random(n), y, d)
    ate = y[d == 1].mean() - y[d == 0].mean()
    assert np.isclose(q[-1], ate * d.mean())
    assert x[0] == 0 and x[-1] == 1


def test_qini_oracle_positive_random_near_zero():
    rng = np.random.default_rng(4)
    n = 20_000
    s = rng.normal(size=n)
    d = rng.integers(0, 2, n).astype(float)
    y = d * (s > 0) + rng.normal(scale=0.3, size=n)
    assert metrics.qini_coefficient(s, y, d) > 0.05
    assert abs(metrics.qini_coefficient(rng.random(n), y, d)) < 0.02
    assert metrics.auuc_lift(s, y, d) > 0.05


def test_tune_lgbm_returns_grid_member():
    X, t, y, _ = simulate(n=1500)
    best, table = cate.tune_lgbm(X, t, y, {"num_leaves": [3, 7]}, {"n_estimators": 30, "verbose": -1}, 2, 0)
    assert best["num_leaves"] in (3, 7) and len(table) == 2

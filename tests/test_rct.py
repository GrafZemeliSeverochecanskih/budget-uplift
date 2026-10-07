import numpy as np
import pandas as pd

from uplift_alloc import rct


def simulate_rct(n: int = 60_000, ate: float = 0.5, seed: int = 0):
    """Three-arm RCT, propensity 1/3; arm 1 effect = ate, arm 2 effect = ate/2, strong prognostic x."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    t = rng.integers(0, 3, size=n)
    tau = np.select([t == 1, t == 2], [ate + 0.3 * x[:, 0], ate / 2], 0.0)  # E[tau | arm1] = ate
    y = 2.0 * x[:, 0] + x[:, 1] + tau + rng.normal(size=n)
    return x, t, y


def test_diff_in_means_recovers_ate():
    x, t, y = simulate_rct()
    for arm, true in ((1, 0.5), (2, 0.25)):
        mask = (t == arm) | (t == 0)
        est = rct.diff_in_means(y[mask], (t[mask] == arm).astype(float))
        assert est.ci_low < true < est.ci_high
        assert abs(est.estimate - true) < 3 * est.se


def test_regression_adjustment_recovers_ate_and_is_tighter():
    x, t, y = simulate_rct()
    mask = (t == 1) | (t == 0)
    d = (t[mask] == 1).astype(float)
    adj = rct.regression_adjusted(y[mask], d, x[mask])
    dim = rct.diff_in_means(y[mask], d)
    assert adj.ci_low < 0.5 < adj.ci_high
    assert adj.width < 0.6 * dim.width  # x explains most of the outcome variance


def test_smd_zero_for_identical_and_known_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(size=100_000)
    assert abs(rct.smd(a, a)) < 1e-12
    assert abs(rct.smd(a + 0.2, a) - 0.2) < 1e-3


def test_balance_table_flags_imbalance():
    t = np.repeat([0, 1, 2], 1000)
    X = pd.DataFrame({"ok": np.tile(np.arange(10.0), 300), "bad": (t == 1).astype(float) + np.tile(np.arange(10.0), 300) / 10})
    bal = rct.balance_table(X, t)
    assert not bal.query("covariate == 'ok'")["flag"].any()
    assert bal.query("covariate == 'bad' and arm == 'Mens E-Mail'")["flag"].all()


def test_power_formulas_are_inverse():
    n = rct.n_per_arm(effect=0.2, sd=1.0)
    assert abs(n - 392.4) < 1  # textbook value for d = 0.2
    assert abs(rct.mde(n, sd=1.0) - 0.2) < 1e-9

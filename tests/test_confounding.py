import numpy as np
import pandas as pd

from uplift_alloc import confounding


def simulate(n: int = 60_000, seed: int = 0):
    """RCT with outcome rising in history; true ATE of arm 1 = 0.5, arm 2 = 0.2."""
    rng = np.random.default_rng(seed)
    history = rng.lognormal(5, 1, n)
    other = rng.normal(size=n)
    t = rng.integers(0, 3, n)
    y = 0.8 * np.log1p(history) + 0.3 * other + np.select([t == 1, t == 2], [0.5, 0.2], 0) + rng.normal(size=n)
    X = pd.DataFrame({"history": history, "other": other, "zip_code_Rural": 0.0, "channel_Multichannel": 0.0})
    return X, t, y


def test_selection_biases_naive_and_adjustment_recovers():
    X, t, y = simulate()
    keep = confounding.make_observational(t, X["history"].to_numpy(), 0.0, 1.5, np.random.default_rng(1))
    Z = confounding.design(X[keep])
    for arm, truth in ((1, 0.5), (2, 0.2)):
        est = confounding.estimates(Z, t[keep], y[keep], arm)
        assert est["naive"] - truth > 0.3              # selection on history -> large upward bias
        assert abs(est["ipw"] - truth) < 0.1
        assert abs(est["dr"] - truth) < 0.05


def test_controls_always_kept():
    t = np.array([0, 0, 1, 2] * 100)
    keep = confounding.make_observational(t, np.arange(400.0), -5, 0.0, np.random.default_rng(0))
    assert keep[t == 0].all() and keep[t > 0].mean() < 0.05

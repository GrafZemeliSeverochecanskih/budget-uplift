import numpy as np
import pytest

from uplift_alloc import ope


def simulate(n: int = 200_000, seed: int = 0):
    """Three-arm RCT with known potential-outcome means mu_true[i, a]."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mu_true = np.column_stack([1 + x, 1.5 + x + 0.5 * (x > 0), 1.2 + x - 0.3 * x])
    t = rng.integers(0, 3, n)
    y = mu_true[np.arange(n), t] + rng.normal(scale=2.0, size=n)
    return x, t, y, mu_true


@pytest.fixture(scope="module")
def sim():
    return simulate()


def fixed_policy(x):
    return np.where(x > 0, 1, np.where(x < -1, 2, 0))


def test_ipw_recovers_fixed_policy_value(sim):
    x, t, y, mu_true = sim
    pi = ope.one_hot(fixed_policy(x))
    truth = (pi * mu_true).sum(axis=1).mean()
    assert ope.ipw_value(pi, t, y) == pytest.approx(truth, abs=0.03)


@pytest.mark.parametrize("mu_kind", ["correct", "wrong"])
def test_dr_recovers_fixed_policy_value(sim, mu_kind):
    x, t, y, mu_true = sim
    pi = ope.one_hot(fixed_policy(x))
    truth = (pi * mu_true).sum(axis=1).mean()
    mu = mu_true if mu_kind == "correct" else np.zeros_like(mu_true) + 5.0  # DR is unbiased either way
    assert ope.dr_value(pi, t, y, mu) == pytest.approx(truth, abs=0.03)


def test_dr_with_correct_model_has_lower_variance():
    ipw, dr = [], []
    for s in range(30):
        x, t, y, mu_true = simulate(n=5000, seed=s)
        pi = ope.one_hot(fixed_policy(x))
        ipw.append(ope.ipw_value(pi, t, y))
        dr.append(ope.dr_value(pi, t, y, mu_true))
    assert np.var(dr) < np.var(ipw)


def test_stochastic_policies_spend_budget_in_expectation():
    cost = np.array([0.0, 0.10, 0.08])
    n, budget = 1000, 45.0
    for pi in (ope.random_policy(n, budget, cost), ope.single_arm_policy(n, budget, cost, 1)):
        assert np.allclose(pi.sum(axis=1), 1)
        assert (pi * cost).sum() == pytest.approx(budget)
    assert (ope.random_policy(n, 0.0, cost)[:, 0] == 1).all()


def test_bootstrap_ci_covers_truth_and_increment_is_paired(sim):
    x, t, y, mu_true = sim
    x, t, y, mu_true = x[:20_000], t[:20_000], y[:20_000], mu_true[:20_000]
    n = len(y)
    policies = {"fixed": ope.one_hot(fixed_policy(x)), "Send nobody": ope.nobody_policy(n)}
    tab = ope.evaluate_policies(policies, t, y, mu_true, reps=200, seed=0)
    truth = (policies["fixed"] * mu_true).sum(axis=1).mean() - mu_true[:, 0].mean()
    for _, r in tab[tab.policy == "fixed"].iterrows():
        assert r.incremental_ci_low < truth < r.incremental_ci_high
    nobody = tab[tab.policy == "Send nobody"]
    assert np.allclose(nobody.incremental, 0) and np.allclose(nobody.incremental_ci_high, 0)

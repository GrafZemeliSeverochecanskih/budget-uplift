import itertools

import numpy as np
import pytest

from uplift_alloc import chance


def brute(S, C, budget):
    best = 0.0
    for combo in itertools.product(range(3), repeat=len(S)):
        a = np.array(combo)
        rows = np.flatnonzero(a > 0)
        if C[rows, a[rows] - 1].sum() <= budget + 1e-12:
            best = max(best, S[rows, a[rows] - 1].sum())
    return best


@pytest.mark.parametrize("seed", range(4))
def test_rowcost_allocation_is_feasible_and_near_optimal(seed):
    rng = np.random.default_rng(seed)
    S, C = rng.normal(0.3, 0.5, (7, 2)), rng.uniform(0.5, 1.5, (7, 2))
    for budget in (0.0, 1.0, 2.5, 100.0):
        a = chance.allocate_rowcost(S, C, budget)
        rows = np.flatnonzero(a > 0)
        assert C[rows, a[rows] - 1].sum() <= budget + 1e-9
        opt = brute(S, C, budget)
        assert S[rows, a[rows] - 1].sum() >= 0.8 * opt - 1e-9  # heuristic, but close on random instances


def test_chance_policy_controls_true_violation_and_mean_policy_does_not():
    rng = np.random.default_rng(0)
    n, v = 20_000, 5.0
    p = np.column_stack([np.full(n, 0.01), rng.uniform(0.01, 0.10, (n, 2))])
    cost = np.array([0.0, 0.10, 0.08])
    S = 1.0 - cost[1:] - v * p[:, 1:] + rng.normal(0, 0.2, (n, 2))
    C, V = cost[1:] + v * p[:, 1:], v**2 * p[:, 1:] * (1 - p[:, 1:])
    budget = 0.05 * n
    mean_a = chance.allocate_rowcost(S, C, budget)
    chance_a, eff = chance.chance_allocate(S, C, V, budget, epsilon=0.05)
    draws = np.random.default_rng(1)
    viol_mean = chance.violation_mc(mean_a, cost, p, v, budget, 2000, draws)
    viol_chance = chance.violation_mc(chance_a, cost, p, v, budget, 2000, draws)
    assert viol_mean > 0.3          # spending the budget in expectation overruns often
    assert viol_chance < 0.08       # the normal approximation holds the 5% target (MC error allowed)
    assert eff < budget

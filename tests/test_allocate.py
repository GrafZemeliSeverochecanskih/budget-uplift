import numpy as np
import pytest

from uplift_alloc import allocate

COST = np.array([1.0, 0.8])


@pytest.mark.parametrize("solver", ["scip", "cpsat"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("budget", [0.0, 1.0, 2.5, 4.0, 10.0])
def test_ip_matches_brute_force(seed, budget, solver):
    rng = np.random.default_rng(seed)
    tau = np.round(rng.normal(0.2, 0.5, size=(7, 2)), 3)  # rounded so integer scaling is exact
    best, _ = allocate.brute_force(tau, COST, budget)
    ip = allocate.ip_allocate(tau, COST, budget, time_limit=10, workers=1, solver=solver)
    assert ip.status == "OPTIMAL"
    assert ip.spend <= budget + 1e-9
    assert ip.objective == pytest.approx(best, abs=1e-9)


def test_lagrangian_feasible_and_close_to_ip():
    rng = np.random.default_rng(0)
    tau = rng.gamma(2.0, 0.5, size=(2000, 2)) - 0.3
    gaps = []
    for budget in (50.0, 200.0, 600.0, 1500.0):
        ip = allocate.ip_allocate(tau, COST, budget, time_limit=20, workers=4)
        lg = allocate.lagrangian_allocate(tau, COST, budget)
        assert lg.spend <= budget + 1e-9
        assert lg.objective <= ip.objective + 1e-6
        assert ip.objective <= lg.bound + 1e-3  # the dual value bounds the optimum
        gaps.append(1 - lg.objective / ip.objective)
    print("Lagrangian gaps to IP:", np.round(gaps, 5))
    assert max(gaps) < 0.02


def test_lagrangian_unconstrained_takes_best_positive_arm():
    tau = np.array([[0.5, 0.2], [-0.1, 0.3], [-0.2, -0.1]])
    lg = allocate.lagrangian_allocate(tau, COST, budget=100)
    assert lg.assign.tolist() == [1, 2, 0]
    assert lg.lam == 0


def test_zero_budget_emails_nobody():
    tau = np.ones((5, 2))
    assert (allocate.ip_allocate(tau, COST, 0.0).assign == 0).all()
    assert (allocate.lagrangian_allocate(tau, COST, 0.0).assign == 0).all()


def test_repair_fixes_identical_tau_groups_and_matches_ip():
    """With identical tau no lam selects part of the group: plain Lagrangian e-mails nobody."""
    tau = np.tile([0.2, 0.9], (100, 1))
    cost = np.array([0.10, 0.08])
    budget = 3.0  # affords 37 Womens e-mails, not 100
    plain = allocate.lagrangian_allocate(tau, cost, budget)
    fixed = allocate.lagrangian_allocate(tau, cost, budget, repair=True)
    ip = allocate.ip_allocate(tau, cost, budget, time_limit=10, workers=1)
    assert plain.objective == 0
    assert fixed.spend <= budget + 1e-9
    assert fixed.objective == pytest.approx(ip.objective)


def test_repair_never_lowers_objective():
    rng = np.random.default_rng(1)
    tau = rng.normal(0.1, 0.5, size=(3000, 2))
    for budget in (10.0, 100.0, 250.0):
        plain = allocate.lagrangian_allocate(tau, COST, budget)
        fixed = allocate.lagrangian_allocate(tau, COST, budget, repair=True)
        assert plain.objective <= fixed.objective + 1e-12 and fixed.spend <= budget + 1e-9

import numpy as np
import pandas as pd
import pytest

from uplift_alloc import positive_control as pcm
from uplift_alloc.config import load_config

PC = load_config()["positive_control"]


def fake_frame(n: int = 300_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"history": rng.lognormal(5, 1, n), "womens": rng.integers(0, 2, n),
                         "mens": rng.integers(0, 2, n), "recency": rng.integers(1, 13, n)})


@pytest.fixture(scope="module")
def sim():
    df = fake_frame()
    t = np.random.default_rng(1).integers(0, 3, len(df))
    Y = pcm.simulate(df, t, PC, np.random.default_rng(2))
    return df, t, Y, pcm.true_tau(df, PC), pcm.segment_mask(df, PC["segment"])


def test_true_spend_tau_matches_config(sim):
    df, t, Y, tau, seg = sim
    mean = PC["amount"]["mean"]
    assert np.allclose(tau["spend"][:, 0], PC["effects"]["mens"]["purchase"] * mean)        # Mens: everywhere
    assert np.allclose(tau["spend"][seg, 1], PC["effects"]["womens"]["purchase"] * mean)   # Womens: in segment
    assert np.allclose(tau["spend"][~seg, 1], 0) and np.allclose(tau["visit"][~seg, 1], 0)  # and nowhere else


def test_segment_rule(sim):
    df, *_, seg = sim
    assert np.array_equal(seg, (df.womens == 1).to_numpy() & (df.recency <= 6).to_numpy())


@pytest.mark.parametrize("outcome", ["spend", "visit"])
def test_simulated_effects_match_true_tau(sim, outcome):
    """Difference in means within the segment recovers the analytic tau (Monte Carlo check)."""
    df, t, Y, tau, seg = sim
    y = Y[outcome].to_numpy()
    for a in (1, 2):
        for mask in (seg, ~seg):
            y1, y0 = y[mask & (t == a)], y[mask & (t == 0)]
            se = np.sqrt(y1.var() / len(y1) + y0.var() / len(y0))
            assert abs((y1.mean() - y0.mean()) - tau[outcome][mask, a - 1].mean()) < 4 * se


def test_visit_implies_purchase_consistency(sim):
    _, _, Y, _, _ = sim
    assert ((Y.conversion == 1) <= (Y.visit == 1)).all()
    assert ((Y.spend > 0) == (Y.conversion == 1)).all()


def test_tau_recovery_perfect_for_truth(sim):
    _, _, _, tau, _ = sim
    tab = pcm.tau_recovery({"truth": tau["spend"][:2000]}, tau["spend"][:2000], reps=20, seed=0, alpha=0.05)
    womens = tab[tab.arm == "Womens E-Mail"].iloc[0]
    assert womens.corr_true_tau == pytest.approx(1.0) and womens.rmse == 0
    assert np.isnan(tab[tab.arm == "Mens E-Mail"].iloc[0].corr_true_tau)  # constant true tau: undefined

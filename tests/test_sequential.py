import copy

import numpy as np
import pandas as pd
import pytest

from uplift_alloc import sequential as sq
from uplift_alloc.config import arm_costs, load_config


def fake_frame(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "recency": rng.integers(1, 13, n), "history": rng.lognormal(5, 1, n),
        "mens": rng.integers(0, 2, n), "womens": rng.integers(0, 2, n), "newbie": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Surburban", "Rural"], n),
        "channel": rng.choice(["Phone", "Web", "Multichannel"], n),
    })


@pytest.fixture(scope="module")
def tiny_cfg():
    cfg = copy.deepcopy(load_config())
    cfg["sequential"].update(customers=6000, batch_size=1500, replications=1, ett_explore_batches=[1])
    cfg["sweep"]["womens_mean_effect"] = {"purchase": 0.05, "visit": 0.05}  # strong, so a tiny run is informative
    cfg["profit"]["lgbm_params"] = {**cfg["profit"]["lgbm_params"], "min_child_samples": 50}
    return cfg


def test_potential_outcomes_are_coupled_and_monotone(tiny_cfg):
    env = sq.environment(fake_frame(), 5000, 1.0, tiny_cfg, np.random.default_rng(0))
    bought = env["spend"] > 0
    assert env["spend"].shape == (5000, 3)
    # effects are non-negative, so with a shared uniform nobody buys under control but not under an e-mail
    assert (bought[:, 1] >= bought[:, 0]).all() and (bought[:, 2] >= bought[:, 0]).all()


def test_random_assign_respects_budget():
    cost = np.array([0.0, 0.10, 0.08])
    a = sq.random_assign(1000, 20.0, cost, np.random.default_rng(0))
    assert sq.spent(a, cost) <= 20.0 + 1e-9 and (a > 0).sum() > 150


def test_tiny_campaign_oracle_dominates_and_budget_holds(tiny_cfg):
    df = sq.run_replication(fake_frame(), "het", 1.0, 0, tiny_cfg)
    totals = df.groupby("policy").profit.sum()
    assert set(totals.index) == {sq.ORACLE, sq.BANDIT, sq.ett_label(1), sq.RANDOM}
    assert (totals[sq.ORACLE] >= totals.drop(sq.ORACLE) - 1e-9).all()
    assert totals[sq.BANDIT] > totals[sq.RANDOM]
    budget = tiny_cfg["sequential"]["budget_fraction"] * 6000 * arm_costs(tiny_cfg)[1:].mean()
    spent = df.groupby("policy").spent.sum()
    assert (spent.drop(sq.ett_label(1)) <= budget * 1.001).all()
    assert df.batch.max() == 4


def test_summarise_pairs_bandit_against_each_policy(tiny_cfg):
    df = pd.concat([sq.run_replication(fake_frame(seed=s), "het", 1.0, s, tiny_cfg) for s in range(2)])
    totals, diffs = sq.summarise(df, 6000, 0.05)
    assert len(diffs) == 3 and totals.share_of_oracle.max() == pytest.approx(1.0)

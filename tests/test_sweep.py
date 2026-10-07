import copy

import numpy as np
import pandas as pd
import pytest

from uplift_alloc import ope, sweep
from uplift_alloc.config import load_config
from uplift_alloc.positive_control import segment_mask, true_tau


def fake_frame(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Covariates with the columns the generator and the feature encoder need."""
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
    cfg["sweep"].update(heterogeneity=[0.0, 1.0], sample_sizes=[6000], replications=2)
    # stronger, cheaper setting so a tiny grid is informative
    cfg["sweep"]["womens_mean_effect"] = {"purchase": 0.05, "visit": 0.05}
    cfg["sweep"]["lgbm_params"] = {**cfg["sweep"]["lgbm_params"], "min_child_samples": 50}
    return cfg


@pytest.mark.parametrize("h", [0.0, 0.3, 1.0])
def test_generator_keeps_mean_effect_and_concentrates_it(h):
    df = fake_frame()
    pc0 = load_config()["positive_control"]
    seg = segment_mask(df, pc0["segment"])
    share = seg.mean()
    mean = {"purchase": 0.01, "visit": 0.03}
    tau = true_tau(df, sweep.generator(pc0, h, mean, share))["spend"][:, 1]
    amount = pc0["amount"]["mean"]
    assert tau.mean() == pytest.approx(mean["purchase"] * amount)
    assert np.allclose(tau[~seg], mean["purchase"] * amount * (1 - h))
    if h == 1.0:  # reproduces the v1.0 positive control's structure: effect only in the segment
        assert np.allclose(tau[~seg], 0)


def test_generator_rejects_bad_h():
    with pytest.raises(ValueError):
        sweep.generator(load_config()["positive_control"], 1.5, {"purchase": 0.01, "visit": 0.01}, 0.3)


def test_dr_scores_mean_is_dr_value():
    rng = np.random.default_rng(0)
    n = 500
    t, y, mu = rng.integers(0, 3, n), rng.random(n), rng.random((n, 3))
    pi = ope.one_hot(rng.integers(0, 3, n))
    assert ope.dr_scores(pi, t, y, mu).mean() == pytest.approx(ope.dr_value(pi, t, y, mu))


def test_tiny_grid_runs_and_heterogeneity_helps(tiny_cfg):
    df = fake_frame()
    share = sweep.segment_share(df, tiny_cfg["positive_control"]["segment"])
    reps = sweep.run_grid(tiny_cfg, df, share, n_jobs=1)
    assert len(reps) == 4 and set(reps.heterogeneity) == {0.0, 1.0}
    summary = sweep.summarise(reps, 0.05)
    assert {"true_gain_vs_random", "excludes_0_vs_best_arm", "detection_rate_vs_random"} <= set(summary.columns)
    by_h = summary.set_index("heterogeneity")
    # the oracle can only gain over the best single arm when there is heterogeneity to exploit
    assert by_h.loc[1.0, "oracle_gain_vs_best_arm"] > by_h.loc[0.0, "oracle_gain_vs_best_arm"] + 0.1
    assert by_h.loc[1.0, "true_gain_vs_random"] > 0
    assert (reps.est_ci_low_vs_random <= reps.est_gain_vs_random).all()
    fig = sweep.plot_heatmap(summary, hillstrom_n=6000)
    assert len(fig.axes) == 2


def test_replications_are_reproducible(tiny_cfg):
    df = fake_frame()
    a = sweep.run_replication(df, 3000, 1.0, 0, tiny_cfg, 0.3)
    b = sweep.run_replication(df, 3000, 1.0, 0, tiny_cfg, 0.3)
    assert a == pytest.approx(b, nan_ok=True)

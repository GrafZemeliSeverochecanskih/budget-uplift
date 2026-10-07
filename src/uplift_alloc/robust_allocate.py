"""Allocation on a bootstrap lower confidence bound of tau-hat instead of the point estimate.

The CATE model is refit on bootstrap resamples of TRAIN; for every test customer and arm the
``lcb_quantile`` of the refit predictions is the pessimistic effect. Customers whose effect is
uncertain are then less attractive, so the policy prefers confidently-positive uplift.

Assumptions: as in ``allocate`` and ``ope``; the bootstrap spread of tau-hat is taken as a
proxy for its estimation uncertainty (it ignores bias).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FIGURES_DIR, CACHE_DIR, TABLES_DIR, arm_costs, budget_for, ensure_output_dirs, load_config
from .plotting import SLOTS, TEXT_2, apply_style, plt, save


def bootstrap_tau(learner: str, params: dict, X: np.ndarray, t: np.ndarray, y: np.ndarray, X_pred: np.ndarray,
                  fits: int, seed: int) -> np.ndarray:
    """(fits, n_pred, n_arms - 1) tau-hat from refits on arm-stratified bootstrap resamples."""
    from .cate import make_learner

    rng = np.random.default_rng(seed)
    out = []
    for b in range(fits):
        idx = np.concatenate([rng.choice(np.flatnonzero(t == a), (t == a).sum()) for a in np.unique(t)])
        out.append(make_learner(learner, params, seed + b).fit(X[idx], t[idx], y[idx]).effect(X_pred))
    return np.stack(out)


def lower_bound(draws: np.ndarray, quantile: float) -> np.ndarray:
    return np.quantile(draws, quantile, axis=0)


def summarise(draws: np.ndarray, eval_quantile: float, alpha: float) -> dict[str, float]:
    """Mean, lower-tail quantile and CI of a bootstrap distribution of incremental policy value."""
    return {"mean": float(draws.mean()), f"q{int(round(eval_quantile * 100)):02d}": float(np.quantile(draws, eval_quantile)),
            "ci_low": float(np.quantile(draws, alpha / 2)), "ci_high": float(np.quantile(draws, 1 - alpha / 2))}


def run() -> pd.DataFrame:
    from .allocate import ip_allocate
    from .cate import load_choice
    from .data import train_test
    from .ope import evaluate_policies, nobody_policy, one_hot

    cfg = load_config()
    ensure_output_dirs()
    r_cfg, seed, alpha = cfg["robust"], cfg["seed"], cfg["alpha"]
    choice = load_choice()
    learner, params = choice["selected"]["spend"], choice["params"]["spend"]
    (Xtr, ttr, Ytr), (Xte, tte, Yte) = train_test()
    y = Yte["spend"].to_numpy()
    tau_point = dict(np.load(CACHE_DIR / "cate_test_spend.npz"))[learner]
    draws_path = CACHE_DIR / "tau_bootstrap_test.npy"
    if draws_path.exists():
        draws = np.load(draws_path)
    else:
        draws = bootstrap_tau(learner, params, Xtr.to_numpy(), ttr, Ytr["spend"].to_numpy(), Xte.to_numpy(),
                              r_cfg["bootstrap_fits"], seed)
        np.save(draws_path, draws)
    tau_lcb = lower_bound(draws, r_cfg["lcb_quantile"])
    mu = np.load(CACHE_DIR / "mu_test_spend.npy")
    cost = arm_costs(cfg)
    n = len(y)
    rows = []
    for frac in cfg["budget_fractions"][1:]:
        budget = budget_for(frac, n, cfg)
        policies = {}
        for name, tau in (("point estimate", tau_point), (f"LCB (q={r_cfg['lcb_quantile']})", tau_lcb)):
            alloc = ip_allocate(tau, cost[1:], budget, cfg["allocate"]["ip_time_limit_s"], cfg["allocate"]["ip_workers"])
            policies[name] = one_hot(alloc.assign)
        policies["Send nobody"] = nobody_policy(n)
        tab = evaluate_policies(policies, tte, y, mu, cfg["bootstrap_reps"], seed, alpha)
        tab = tab[tab.estimator == "dr"]
        base_draws = tab.loc[tab.policy == "Send nobody", "_draws"].iloc[0]
        for _, r in tab[tab.policy != "Send nobody"].iterrows():
            inc = r["_draws"] - base_draws
            rows.append({"budget_fraction": frac, "policy": r.policy, "incremental": r.incremental,
                         **summarise(inc, r_cfg["eval_quantile"], alpha),
                         "n_emailed": int((policies[r.policy][:, 0] == 0).sum()),
                         "spend_used": float((policies[r.policy] * cost).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "robust_allocation.csv", index=False)
    save(plot(table, r_cfg["eval_quantile"]), FIGURES_DIR / "robust_allocation.png")
    return table


def plot(table: pd.DataFrame, eval_quantile: float) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    qcol = f"q{int(round(eval_quantile * 100)):02d}"
    for color, (name, g) in zip((SLOTS[0], SLOTS[1]), table.groupby("policy", sort=False)):
        ax.fill_between(g.budget_fraction, g.ci_low, g.ci_high, color=color, alpha=0.13, linewidth=0)
        ax.plot(g.budget_fraction, g["mean"], color=color, marker="o", markersize=4, label=f"{name}: mean")
        ax.plot(g.budget_fraction, g[qcol], color=color, linestyle=":", label=f"{name}: {int(eval_quantile * 100)}th pct")
    ax.axhline(0, color=TEXT_2, lw=0.8)
    ax.set_xlabel("Budget (fraction of customers e-mailable at mean e-mail cost)")
    ax.set_ylabel("Incremental spend per customer ($, DR)")
    ax.set_title("Point-estimate vs lower-confidence-bound allocation", loc="left")
    ax.legend(loc="upper left")
    fig.text(0.01, -0.03, "Bands: 95% bootstrap CI over test resamples. Dotted: 5th percentile of the same distribution.",
             color=TEXT_2, fontsize=8)
    fig.tight_layout()
    return fig

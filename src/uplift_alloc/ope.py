"""Off-policy evaluation of targeting policies on the randomised test split.

A policy is an (n, K+1) matrix of assignment probabilities pi[i, a] (rows sum to 1; column 0 =
no e-mail); deterministic policies are one-hot. Value = expected outcome per customer.

Assumptions: SUTVA; the logging policy is the RCT design with known propensity 1/3 for every
arm, so both estimators below are unbiased (IPW exactly; DR for any fixed outcome model mu-hat).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ARM_NAMES, FIGURES_DIR, CACHE_DIR, TABLES_DIR, arm_costs, budget_for, ensure_output_dirs, load_config
from .plotting import NEUTRAL, SLOTS, TEXT, TEXT_2, apply_style, plt, save

N_ARMS = 3


def one_hot(assign: np.ndarray, n_arms: int = N_ARMS) -> np.ndarray:
    return np.eye(n_arms)[assign]


def ipw_value(pi: np.ndarray, t: np.ndarray, y: np.ndarray, propensity: float = 1 / 3) -> float:
    """Horvitz-Thompson: mean_i pi[i, T_i] * Y_i / e(T_i)."""
    return float(np.mean(pi[np.arange(len(t)), t] * y / propensity))


def dr_scores(pi: np.ndarray, t: np.ndarray, y: np.ndarray, mu: np.ndarray, propensity: float = 1 / 3) -> np.ndarray:
    """Per-customer doubly robust scores sum_a pi[i,a] mu[i,a] + pi[i,T_i] (Y_i - mu[i,T_i]) / e(T_i).

    Their mean is the DR policy value and their standard deviation / sqrt(n) its influence-function SE.
    """
    rows = np.arange(len(t))
    return (pi * mu).sum(axis=1) + pi[rows, t] * (y - mu[rows, t]) / propensity


def dr_value(pi: np.ndarray, t: np.ndarray, y: np.ndarray, mu: np.ndarray, propensity: float = 1 / 3) -> float:
    """Doubly robust policy value: mean of ``dr_scores``."""
    return float(np.mean(dr_scores(pi, t, y, mu, propensity)))


# policies ----------------------------------------------------------------------
def random_policy(n: int, budget: float, cost: np.ndarray) -> np.ndarray:
    """E-mail each customer with prob q (arm uniform among treated arms) so expected cost = B."""
    q = min(1.0, budget / (n * cost[1:].mean())) if budget > 0 else 0.0
    pi = np.zeros((n, len(cost)))
    pi[:, 0] = 1 - q
    pi[:, 1:] = q / (len(cost) - 1)
    return pi


def single_arm_policy(n: int, budget: float, cost: np.ndarray, arm: int) -> np.ndarray:
    """Send ``arm`` to a uniformly random subset of customers whose expected cost is B."""
    q = min(1.0, budget / (n * cost[arm])) if budget > 0 else 0.0
    pi = np.zeros((n, len(cost)))
    pi[:, 0], pi[:, arm] = 1 - q, q
    return pi


def nobody_policy(n: int, n_arms: int = N_ARMS) -> np.ndarray:
    return one_hot(np.zeros(n, dtype=int), n_arms)


# evaluation --------------------------------------------------------------------
def evaluate_policies(policies: dict[str, np.ndarray], t: np.ndarray, y: np.ndarray, mu: np.ndarray,
                      reps: int, seed: int, alpha: float = 0.05, baseline: str = "Send nobody") -> pd.DataFrame:
    """IPW and DR value per policy, plus incremental value over ``baseline``, with bootstrap CIs.

    Rows of the test set are resampled; all policies share each resample (paired comparisons).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    est = {"ipw": lambda p, i: ipw_value(p[i], t[i], y[i]), "dr": lambda p, i: dr_value(p[i], t[i], y[i], mu[i])}
    full = np.arange(n)
    point = {(m, k): f(p, full) for m, f in est.items() for k, p in policies.items()}
    draws = {key: np.empty(reps) for key in point}
    for r in range(reps):
        idx = rng.integers(0, n, n)
        for m, f in est.items():
            for k, p in policies.items():
                draws[(m, k)][r] = f(p, idx)
    q = [alpha / 2, 1 - alpha / 2]
    rows = []
    for (m, k), v in point.items():
        inc = draws[(m, k)] - draws[(m, baseline)]
        rows.append({
            "estimator": m, "policy": k, "value": v, "value_ci_low": np.quantile(draws[(m, k)], q[0]),
            "value_ci_high": np.quantile(draws[(m, k)], q[1]), "incremental": v - point[(m, baseline)],
            "incremental_ci_low": np.quantile(inc, q[0]), "incremental_ci_high": np.quantile(inc, q[1]),
            "_draws": draws[(m, k)],
        })
    return pd.DataFrame(rows)


def add_paired_differences(table: pd.DataFrame, ours: str, others: list[str], alpha: float) -> pd.DataFrame:
    """Value of ``ours`` minus each other policy, with paired bootstrap CIs."""
    rows = []
    for m, g in table.groupby("estimator"):
        g = g.set_index("policy")
        for other in others:
            diff = g.loc[ours, "_draws"] - g.loc[other, "_draws"]
            rows.append({"estimator": m, "policy": f"{ours} − {other}",
                         "incremental": g.loc[ours, "value"] - g.loc[other, "value"],
                         "incremental_ci_low": np.quantile(diff, alpha / 2),
                         "incremental_ci_high": np.quantile(diff, 1 - alpha / 2)})
    return pd.DataFrame(rows)


# pipeline ----------------------------------------------------------------------
def outcome_model_test(train, test, outcome: str, params: dict, seed: int) -> np.ndarray:
    """mu-hat(x, a) on test from per-arm LightGBM models fit on train (no test data used in fitting)."""
    from .cate import TLearner

    (Xtr, ttr, Ytr), (Xte, _, _) = train, test
    model = TLearner(params, seed).fit(np.asarray(Xtr, dtype=float), ttr, Ytr[outcome].to_numpy())
    return model.mu(np.asarray(Xte, dtype=float))


def best_single_arm(train, outcome: str) -> int:
    """Arm with the largest difference-in-means effect on TRAIN."""
    _, ttr, Ytr = train
    y = Ytr[outcome].to_numpy()
    return int(np.argmax([y[ttr == a].mean() - y[ttr == 0].mean() for a in (1, 2)]) + 1)


def true_incremental(pi: np.ndarray, tau_true: np.ndarray) -> float:
    """Known incremental value vs no e-mail: mean_i sum_a pi[i, a] tau[i, a] (positive control only)."""
    return float((pi[:, 1:] * tau_true).sum(axis=1).mean())


def evaluate_budgets(train, test, outcome: str, tag: str = "", tau_true: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate with the IP on the train-selected learner's tau-hat and evaluate every policy at every budget.

    With ``tau_true`` (simulated data) an oracle policy (IP on the true tau) is added and every
    policy also gets its known true incremental value.
    """
    from .allocate import ip_allocate
    from .cate import load_choice, load_test_predictions

    cfg = load_config()
    seed, reps, alpha = cfg["seed"], cfg["bootstrap_reps"], cfg["alpha"]
    choice = load_choice(tag)
    learner = choice["selected"][outcome]
    tau = load_test_predictions(outcome, tag)[learner]
    _, tte, Yte = test
    y = Yte[outcome].to_numpy()
    mu = outcome_model_test(train, test, outcome, choice["params"][outcome], seed)
    np.save(CACHE_DIR / f"{tag}mu_test_{outcome}.npy", mu)
    cost = arm_costs(cfg)
    best_arm = best_single_arm(train, outcome)
    ours = f"Uplift ({learner}-learner) + IP"
    best = f"Best single arm ({ARM_NAMES[best_arm]})"
    n = len(y)

    def ip(scores: np.ndarray, budget: float) -> np.ndarray:
        return ip_allocate(scores, cost[1:], budget, cfg["allocate"]["ip_time_limit_s"], cfg["allocate"]["ip_workers"]).assign

    tables, diffs = [], []
    for frac in cfg["budget_fractions"]:
        budget = budget_for(frac, n, cfg)
        policies = {ours: one_hot(ip(tau, budget))}
        if tau_true is not None:
            policies["Oracle (true τ) + IP"] = one_hot(ip(tau_true, budget))
        policies.update({"Random targeting": random_policy(n, budget, cost),
                         best: single_arm_policy(n, budget, cost, best_arm), "Send nobody": nobody_policy(n)})
        tab = evaluate_policies(policies, tte, y, mu, reps, seed, alpha)
        diffs.append(add_paired_differences(tab, ours, ["Random targeting", best], alpha).assign(budget_fraction=frac))
        tab["spend_used"] = tab["policy"].map({k: float((p * cost).sum()) for k, p in policies.items()})
        if tau_true is not None:
            tab["true_incremental"] = tab["policy"].map({k: true_incremental(p, tau_true) for k, p in policies.items()})
        tables.append(tab.drop(columns="_draws").assign(budget_fraction=frac, budget=budget))
    table = pd.concat(tables, ignore_index=True).assign(outcome=outcome)
    return table, pd.concat(diffs, ignore_index=True).assign(outcome=outcome)


def run_outcomes(train, test, outcomes: list[str], tag: str = "",
                 tau_true: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    ensure_output_dirs()
    tables, diffs = [], []
    for outcome in outcomes:
        table, diff = evaluate_budgets(train, test, outcome, tag, None if tau_true is None else tau_true[outcome])
        tables.append(table)
        diffs.append(diff)
        name = "policy_value_vs_budget.png" if outcome == "spend" else f"policy_value_vs_budget_{outcome}.png"
        save(plot_policy_values(table, outcome, tag), FIGURES_DIR / f"{tag}{name}")
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(TABLES_DIR / f"{tag}policy_values.csv", index=False)
    pd.concat(diffs, ignore_index=True).to_csv(TABLES_DIR / f"{tag}policy_differences.csv", index=False)
    return table


def run() -> pd.DataFrame:
    from .data import train_test

    train, test = train_test()
    return run_outcomes(train, test, load_config()["ope"]["outcomes"])


UNITS = {"spend": "Incremental spend per customer vs no e-mail ($)",
         "visit": "Incremental visit probability vs no e-mail"}


def plot_policy_values(table: pd.DataFrame, outcome: str, tag: str = "", estimator: str = "dr") -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.3))
    sub = table[table.estimator == estimator]
    policies = list(dict.fromkeys(sub.policy))
    palette = iter([SLOTS[0], SLOTS[1], SLOTS[2]])
    colors = {k: TEXT if k.startswith("Oracle") else NEUTRAL if k == "Send nobody" else next(palette) for k in policies}
    for k in policies:
        g = sub[sub.policy == k].sort_values("budget_fraction")
        ax.fill_between(g.budget_fraction, g.incremental_ci_low, g.incremental_ci_high, color=colors[k],
                        alpha=0.12, linewidth=0)
        ax.plot(g.budget_fraction, g.incremental, color=colors[k], marker="o", markersize=4, label=k,
                linestyle="--" if k == "Send nobody" else ":" if k.startswith("Oracle") else "-")
        if "true_incremental" in g:
            ax.plot(g.budget_fraction, g.true_incremental, color=colors[k], marker="x", markersize=6,
                    linestyle="none", label="_nolegend_")
    if "true_incremental" in sub:
        ax.plot([], [], color=TEXT_2, marker="x", linestyle="none", label="true value (known)")
    ax.set_xlabel("Budget (fraction of customers e-mailable at mean e-mail cost)")
    ax.set_ylabel(UNITS.get(outcome, f"Incremental {outcome} per customer"))
    data = "positive control (simulated outcomes)" if tag else "Hillstrom test split"
    ax.set_title(f"Policy value, {outcome}: {data}", loc="left")
    ax.legend(loc="upper left")
    fig.text(0.01, -0.03, "Doubly robust estimate on the test split (n = 19,200), propensity 1/3 known by design. "
             "Bands: 95% bootstrap CI of the difference to 'Send nobody'.", color=TEXT_2, fontsize=8)
    fig.tight_layout()
    return fig

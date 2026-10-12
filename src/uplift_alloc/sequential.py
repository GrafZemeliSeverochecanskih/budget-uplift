"""v1.4: sequential allocation — budgeted Thompson-sampling bandit vs one-shot "experiment, then target".

A campaign of N customers arrives in batches; each batch's outcomes are observed before the next
batch is allocated. One total budget is paced evenly (remaining budget / remaining batches). Outcomes
come from the v1.1 generator (``sweep.generator``) at heterogeneity h, and every policy is scored by
the KNOWN expected profit of its assignments, sum_i (margin * tau[i, a_i] - cost[a_i]).

Pairing: within a replication every policy faces the same customers and the same outcome draws.
Potential outcomes are coupled across arms (one uniform per customer decides purchase under every
arm, one amount per customer), so differences between policies come only from their choices.

Assumptions: SUTVA across customers and batches (no carry-over or fatigue), stationary customer
population, outcomes observed before the next batch, fixed costs and margin.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import FIGURES_DIR, TABLES_DIR, arm_costs, ensure_output_dirs, load_config
from .plotting import NEUTRAL, SLOTS, TEXT, TEXT_2, apply_style, plt, save


def ett_label(k: int) -> str:
    return f"Experiment-then-target (k={k})"


BANDIT, ORACLE, RANDOM = "Thompson-sampling bandit", "Oracle (true τ)", "Random targeting"


# environment ----------------------------------------------------------------------
def environment(df_all: pd.DataFrame, n: int, h: float, cfg: dict, rng: np.random.Generator) -> dict[str, Any]:
    """Customers, features, true tau (spend) and coupled potential spend outcomes under every arm."""
    from .data import encode_features
    from .positive_control import baseline_rates, effect_increments, true_tau
    from .sweep import generator, segment_share

    share = segment_share(df_all, cfg["positive_control"]["segment"])
    pc = generator(cfg["positive_control"], h, cfg["sweep"]["womens_mean_effect"], share)
    df = df_all.iloc[rng.integers(0, len(df_all), n)].reset_index(drop=True)
    _, c0 = baseline_rates(df["history"].to_numpy(), pc["baseline"])
    _, dc = effect_increments(df, pc)
    u = rng.random(n)
    sigma = pc["amount"]["sigma"]
    amount = rng.lognormal(np.log(pc["amount"]["mean"]) - sigma**2 / 2, sigma, n)
    spend = (u[:, None] < (c0[:, None] + dc)) * amount[:, None]          # (n, 3): outcome under each arm
    return {"X": encode_features(df).to_numpy(), "tau": true_tau(df, pc)["spend"], "spend": spend}


# helpers --------------------------------------------------------------------------
def random_assign(n: int, budget: float, cost: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random customers, random e-mail, until the next e-mail would exceed the budget."""
    assign = np.zeros(n, dtype=int)
    arms = rng.integers(1, 3, n)
    order = rng.permutation(n)
    affordable = np.cumsum(cost[arms[order]]) <= budget + 1e-9
    assign[order[affordable]] = arms[order[affordable]]
    return assign


def uniform_assign(n: int, rng: np.random.Generator) -> np.ndarray:
    """RCT assignment: each arm (incl. no e-mail) with probability 1/3."""
    return rng.integers(0, 3, n)


def profit_allocate(tau: np.ndarray, budget: float, margin: float, cost: np.ndarray) -> np.ndarray:
    from .allocate import lagrangian_allocate

    if budget <= 0:
        return np.zeros(len(tau), dtype=int)
    return lagrangian_allocate(margin * tau - cost[1:], cost[1:], budget, repair=True).assign


def spent(assign: np.ndarray, cost: np.ndarray) -> float:
    return float(cost[assign].sum())


def expected_profit(assign: np.ndarray, tau: np.ndarray, margin: float, cost: np.ndarray) -> float:
    rows = np.flatnonzero(assign > 0)
    return float((margin * tau[rows, assign[rows] - 1] - cost[assign[rows]]).sum())


# policies -------------------------------------------------------------------------
def run_policy(name: str, env: dict[str, Any], cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Play one policy through the whole campaign; returns one row per batch."""
    from .cate import make_learner

    sq = cfg["sequential"]
    cost, m = arm_costs(cfg), sq["margin"]
    X, tau, spend_po = env["X"], env["tau"], env["spend"]
    n, bs = len(X), sq["batch_size"]
    batches = [np.arange(s, min(s + bs, n)) for s in range(0, n, bs)]
    remaining = sq["budget_fraction"] * n * cost[1:].mean()
    logX, logt, logy = [], [], []
    model, rows = None, []
    k = int(name.split("k=")[1].rstrip(")")) if name.startswith("Experiment") else 0
    for b, idx in enumerate(batches):
        budget = remaining / (len(batches) - b)
        Xb = X[idx]
        if name == ORACLE:
            assign = profit_allocate(tau[idx], budget, m, cost)
        elif name == RANDOM:
            assign = random_assign(len(idx), budget, cost, rng)
        elif k:  # experiment-then-target
            if b < k:
                assign = uniform_assign(len(idx), rng)
            else:
                if model is None:
                    model = make_learner("S", cfg["profit"]["lgbm_params"], cfg["seed"]).fit(
                        np.vstack(logX), np.concatenate(logt), np.concatenate(logy))
                assign = profit_allocate(model.effect(Xb), budget, m, cost)
        elif name == BANDIT:
            if b == 0:
                assign = random_assign(len(idx), budget, cost, rng)
            else:
                lx, lt, ly = np.vstack(logX), np.concatenate(logt), np.concatenate(logy)
                boot = rng.integers(0, len(ly), len(ly))
                ts = make_learner(sq["bandit"]["learner"], cfg["profit"]["lgbm_params"], int(rng.integers(2**31)))
                tau_draw = ts.fit(lx[boot], lt[boot], ly[boot]).effect(Xb)
                explore = rng.random(len(idx)) < sq["bandit"]["epsilon"]
                assign = np.zeros(len(idx), dtype=int)
                assign[explore] = uniform_assign(int(explore.sum()), rng)
                left = budget - spent(assign, cost)
                exploit = np.flatnonzero(~explore)
                assign[exploit] = profit_allocate(tau_draw[exploit], left, m, cost)
        else:
            raise ValueError(name)
        cost_b = spent(assign, cost)
        remaining -= cost_b
        y = spend_po[idx, assign]
        logX.append(Xb)
        logt.append(assign)
        logy.append(y)
        rows.append({"policy": name, "batch": b + 1, "profit": expected_profit(assign, tau[idx], m, cost),
                     "spent": cost_b, "batch_budget": budget, "n_emailed": int((assign > 0).sum()),
                     "share_womens": float((assign == 2).mean())})
    return pd.DataFrame(rows)


def run_replication(df_all: pd.DataFrame, env_name: str, h: float, rep: int, cfg: dict) -> pd.DataFrame:
    sq = cfg["sequential"]
    env = environment(df_all, sq["customers"], h, cfg, np.random.default_rng([cfg["seed"], int(h * 1000), rep]))
    policies = [ORACLE, BANDIT, *[ett_label(k) for k in sq["ett_explore_batches"]], RANDOM]
    out = []
    for j, name in enumerate(policies):
        res = run_policy(name, env, cfg, np.random.default_rng([cfg["seed"], int(h * 1000), rep, j]))
        out.append(res.assign(environment=env_name, rep=rep))
    df = pd.concat(out, ignore_index=True)
    df["cum_profit_per_customer"] = df.groupby("policy").profit.cumsum() / sq["customers"]
    return df


# summaries ------------------------------------------------------------------------
def _tci(v: np.ndarray, alpha: float) -> tuple[float, float, float]:
    mean = float(np.mean(v))
    if len(v) < 2:
        return mean, np.nan, np.nan
    half = stats.t.ppf(1 - alpha / 2, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))
    return mean, mean - half, mean + half


def summarise(batches: pd.DataFrame, customers: int, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Total profit per customer per policy, and paired differences of the bandit vs every other policy."""
    totals = batches.groupby(["environment", "rep", "policy"]).profit.sum().div(customers).unstack("policy")
    rows, diffs = [], []
    for env, g in totals.groupby(level="environment"):
        oracle = g[ORACLE].to_numpy()
        for pol in g.columns:
            mean, lo, hi = _tci(g[pol].to_numpy(), alpha)
            share, s_lo, s_hi = _tci(g[pol].to_numpy() / oracle, alpha)
            rows.append({"environment": env, "policy": pol, "profit_per_customer": mean, "ci_low": lo, "ci_high": hi,
                         "share_of_oracle": share, "share_ci_low": s_lo, "share_ci_high": s_hi})
            if pol != BANDIT:
                d, d_lo, d_hi = _tci((g[BANDIT] - g[pol]).to_numpy(), alpha)
                diffs.append({"environment": env, "comparison": f"{BANDIT} − {pol}", "difference": d,
                              "ci_low": d_lo, "ci_high": d_hi, "excludes_0": bool(d_lo > 0 or d_hi < 0)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def plot(batches: pd.DataFrame, alpha: float) -> plt.Figure:
    apply_style()
    envs = list(dict.fromkeys(batches.environment))
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 4.3), sharey=False)
    pols = list(dict.fromkeys(batches.policy))
    ett = [p for p in pols if p.startswith("Experiment")]
    colors = {ORACLE: TEXT, BANDIT: SLOTS[0], RANDOM: SLOTS[1]}
    colors.update(dict(zip(ett, (SLOTS[2], SLOTS[6], SLOTS[4]))))
    titles = {"heterogeneous": "Heterogeneous effect (h = 1, positive control)",
              "homogeneous": "Homogeneous effect (h = 0)"}
    for ax, env in zip(np.atleast_1d(axes), envs):
        sub = batches[batches.environment == env]
        for pol in pols:
            g = sub[sub.policy == pol].groupby("batch").cum_profit_per_customer
            mean = g.mean()
            half = stats.t.ppf(1 - alpha / 2, max(g.count().iloc[0] - 1, 1)) * g.std(ddof=1) / np.sqrt(g.count())
            ax.fill_between(mean.index, mean - half, mean + half, color=colors[pol], alpha=0.12, linewidth=0)
            ax.plot(mean.index, mean, color=colors[pol], linewidth=2, label=pol,
                    linestyle=":" if pol == ORACLE else "-")
        ax.axhline(0, color=NEUTRAL, linestyle="--", linewidth=1)
        ax.set_xlabel("Batch (4,000 customers each)")
        ax.set_ylabel("Cumulative true profit per campaign customer ($)")
        ax.set_title(titles.get(env, env), loc="left")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.suptitle("Budgeted bandit vs experiment-then-target (same budget, paced evenly; 95% CI over replications)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, -0.03, "Profit = margin × true τ − e-mail cost of each assignment (known from the generator); "
             "every policy sees the same customers and outcome draws.", color=TEXT_2, fontsize=8)
    return fig


def run() -> dict[str, pd.DataFrame]:
    from joblib import Parallel, delayed

    from .data import load_frame

    cfg = load_config()
    ensure_output_dirs()
    sq = cfg["sequential"]
    df_all = load_frame()
    jobs = [(e, spec["heterogeneity"], r) for e, spec in sq["environments"].items() for r in range(sq["replications"])]
    frames = Parallel(n_jobs=4)(delayed(run_replication)(df_all, e, h, r, cfg) for e, h, r in jobs)
    batches = pd.concat(frames, ignore_index=True)
    totals, diffs = summarise(batches, sq["customers"], cfg["alpha"])
    batches.to_csv(TABLES_DIR / "sequential_batches.csv", index=False)
    totals.to_csv(TABLES_DIR / "sequential_totals.csv", index=False)
    diffs.to_csv(TABLES_DIR / "sequential_differences.csv", index=False)
    save(plot(batches, cfg["alpha"]), FIGURES_DIR / "sequential.png")
    return {"totals": totals, "differences": diffs}

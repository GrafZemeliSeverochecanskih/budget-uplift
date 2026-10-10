"""v1.3b: chance-constrained budget when customer uptake (voucher redemption) is random.

Every e-mail carries a voucher worth v, paid on every purchase by a targeted customer. For customer i
and arm a with purchase probability p[i, a]:

    expected profit  S[i, a] = margin * tau[i, a] - cost[a] - v * p[i, a]
    expected cost    C[i, a] = cost[a] + v * p[i, a]
    cost variance    V[i, a] = v^2 * p[i, a] (1 - p[i, a])

The campaign's total cost is random. The mean-budget policy only requires E[cost] <= B and therefore
overspends about half the time. The chance-constrained policy requires P(cost > B) <= epsilon, using
the normal approximation E[cost] + z_(1-eps) sd[cost] <= B. It is found by bisection on the effective
budget given to the mean-budget allocator.

Assumptions: redemptions are independent across customers given x (SUTVA); p and tau are taken from
models fit on training folds; the voucher does not change behaviour beyond what the outcome model
captures (it is part of the e-mail treatment).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from .config import FIGURES_DIR, TABLES_DIR, arm_costs, ensure_output_dirs, load_config
from .plotting import SLOTS, TEXT, TEXT_2, apply_style, plt, save


# allocation with per-customer costs ----------------------------------------------
def _choice(S: np.ndarray, C: np.ndarray, lam: float) -> np.ndarray:
    scores = np.column_stack([np.zeros(len(S)), S - lam * C])
    return scores.argmax(axis=1)


def _spend(assign: np.ndarray, C: np.ndarray) -> float:
    rows = np.flatnonzero(assign > 0)
    return float(C[rows, assign[rows] - 1].sum())


def allocate_rowcost(S: np.ndarray, C: np.ndarray, budget: float, max_iter: int = 100) -> np.ndarray:
    """Max sum S subject to sum C <= budget, at most one arm each: Lagrangian bisection + greedy repair."""
    if _spend(_choice(S, C, 0.0), C) <= budget:
        return _choice(S, C, 0.0)
    pos = S > 0
    lo, hi = 0.0, float((S[pos] / C[pos]).max()) + 1.0 if pos.any() else 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if _spend(_choice(S, C, mid), C) <= budget else (mid, hi)
        if hi - lo < 1e-12:
            break
    assign = _choice(S, C, hi)
    left = budget - _spend(assign, C)
    free = np.flatnonzero(assign == 0)
    ci, ca = np.nonzero(S[free] > 0)
    for k in np.argsort(-(S[free][ci, ca] / C[free][ci, ca]), kind="stable"):
        i, a = free[ci[k]], ca[k]
        if assign[i] == 0 and C[i, a] <= left:
            assign[i] = a + 1
            left -= C[i, a]
    return assign


def cost_moments(assign: np.ndarray, C: np.ndarray, V: np.ndarray) -> tuple[float, float]:
    rows = np.flatnonzero(assign > 0)
    return float(C[rows, assign[rows] - 1].sum()), float(np.sqrt(V[rows, assign[rows] - 1].sum()))


def chance_allocate(S: np.ndarray, C: np.ndarray, V: np.ndarray, budget: float, epsilon: float,
                    iters: int = 40) -> tuple[np.ndarray, float]:
    """Largest effective budget B' <= B whose allocation satisfies E[cost] + z sd[cost] <= B."""
    z = stats.norm.ppf(1 - epsilon)

    def ok(b: float) -> bool:
        mean, sd = cost_moments(allocate_rowcost(S, C, b), C, V)
        return mean + z * sd <= budget

    lo, hi = 0.0, budget
    if ok(hi):
        return allocate_rowcost(S, C, hi), hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return allocate_rowcost(S, C, lo), lo


def violation_mc(assign: np.ndarray, cost: np.ndarray, p: np.ndarray, voucher: float, budget: float,
                 draws: int, rng: np.random.Generator) -> float:
    """P(total cost > budget) by Monte Carlo with purchase probabilities p (n, 3)."""
    rows = np.flatnonzero(assign > 0)
    fixed = cost[assign[rows]].sum()
    probs = p[rows, assign[rows]]
    over, chunk = 0, max(1, 2_000_000 // max(len(rows), 1))
    for start in range(0, draws, chunk):
        k = min(chunk, draws - start)
        buys = (rng.random((k, len(rows))) < probs).sum(axis=1)
        over += int((fixed + voucher * buys > budget + 1e-9).sum())
    return over / draws


# data ---------------------------------------------------------------------------
def chance_arrays(name: str, cfg: dict) -> dict[str, Any]:
    """X, arm, spend, conversion and (positive control) true tau for spend and true purchase probabilities."""
    from .data import encode_arm, encode_features, load_frame
    from .positive_control import baseline_rates, effect_increments, simulate, true_tau

    df = load_frame()
    out = {"X": encode_features(df).to_numpy(), "t": encode_arm(df["segment"]), "tau": None, "p": None}
    if name == "hillstrom":
        out.update(spend=df["spend"].to_numpy(float), conv=df["conversion"].to_numpy(float))
        return out
    pc = cfg["positive_control"]
    Y = simulate(df, out["t"], pc, np.random.default_rng(cfg["seed"] + pc["seed_offset"]))
    _, c0 = baseline_rates(df["history"].to_numpy(), pc["baseline"])
    _, dc = effect_increments(df, pc)
    out.update(spend=Y["spend"].to_numpy(), conv=Y["conversion"].to_numpy(),
               tau=true_tau(df, pc)["spend"], p=c0[:, None] + dc)
    return out


# experiment ---------------------------------------------------------------------
def crossfit_repeat(d: dict[str, Any], learner: str, cfg: dict, repeat: int) -> list[dict[str, Any]]:
    """Mean-budget vs chance-constrained policies on every held-out fold (one campaign per fold)."""
    from .cate import TLearner, make_learner

    ch, params = cfg["chance"], cfg["profit"]["lgbm_params"]
    cost, v, m, eps = arm_costs(cfg), ch["voucher_value"], ch["margin"], ch["epsilon"]
    X, t, spend, conv = d["X"], d["t"], d["spend"], d["conv"]
    rng = np.random.default_rng([cfg["seed"], repeat])
    rows = []
    folds = StratifiedKFold(ch["crossfit"]["folds"], shuffle=True, random_state=cfg["seed"] + repeat)
    for f, (tr, te) in enumerate(folds.split(X, t)):
        seed = cfg["seed"] + repeat
        tau_hat = make_learner(learner, params, seed).fit(X[tr], t[tr], spend[tr]).effect(X[te])
        p_hat = np.clip(TLearner(params, seed).fit(X[tr], t[tr], conv[tr]).mu(X[te]), 1e-4, 0.99)
        mu_spend = TLearner(params, seed).fit(X[tr], t[tr], spend[tr]).mu(X[te])
        S = m * tau_hat - cost[1:] - v * p_hat[:, 1:]
        C = cost[1:] + v * p_hat[:, 1:]
        V = v**2 * p_hat[:, 1:] * (1 - p_hat[:, 1:])
        ctx = campaign_context(d, te, t, spend, conv, mu_spend, p_hat, cfg, rng)
        for b_pc in ch["budget_per_customer"]:
            budget = b_pc * len(te)
            mean_assign = allocate_rowcost(S, C, budget)
            chance_assign, eff = chance_allocate(S, C, V, budget, eps)
            for name, assign, share in (("Mean budget", mean_assign, 1.0),
                                        (f"Chance-constrained (ε={eps:g})", chance_assign, eff / budget)):
                rows.append({"repeat": repeat, "fold": f, "policy": name, "budget_per_customer": b_pc,
                             "effective_budget_share": share, **campaign_row(assign, C, V, budget, ctx)})
    return rows


def campaign_context(d: dict[str, Any], te: np.ndarray, t: np.ndarray, spend: np.ndarray, conv: np.ndarray,
                     mu_spend: np.ndarray, p_hat: np.ndarray, cfg: dict, rng: np.random.Generator) -> dict[str, Any]:
    """Everything needed to score any policy on one held-out campaign (shared by v1.3b and v1.5)."""
    from .ope import dr_scores, nobody_policy

    ch = cfg["chance"]
    return {"t": t[te], "spend": spend[te], "conv": conv[te], "mu_spend": mu_spend, "p_hat": p_hat,
            "base": dr_scores(nobody_policy(len(te)), t[te], spend[te], mu_spend),
            "cost": arm_costs(cfg), "v": ch["voucher_value"], "m": ch["margin"], "draws": ch["uptake_draws"],
            "p_true": None if d["p"] is None else d["p"][te], "tau_true": None if d["tau"] is None else d["tau"][te],
            "rng": rng, "n": len(te)}


def campaign_row(assign: np.ndarray, C: np.ndarray, V: np.ndarray, budget: float, ctx: dict[str, Any]) -> dict[str, Any]:
    """DR-estimated profit, plug-in overrun (from the C, V the policy planned with) and, if known, the truth."""
    from .ope import dr_scores, one_hot

    cost, v, m = ctx["cost"], ctx["v"], ctx["m"]
    pi = one_hot(assign)
    treated = pi.copy()
    treated[:, 0] = 0.0
    rev = dr_scores(pi, ctx["t"], ctx["spend"], ctx["mu_spend"]) - ctx["base"]
    voucher = v * dr_scores(treated, ctx["t"], ctx["conv"], ctx["p_hat"])
    mean_c, sd_c = cost_moments(assign, C, V)
    row = {"n_targeted": int((assign > 0).sum()),
           "est_profit": float((m * rev - pi @ cost - voucher).mean()),
           "plugin_violation": float(stats.norm.sf((budget - mean_c) / sd_c)) if sd_c > 0 else float(mean_c > budget)}
    if ctx["p_true"] is not None:
        p_te, tau_te, p_hat = ctx["p_true"], ctx["tau_true"], ctx["p_hat"]
        true_s = m * tau_te - cost[1:] - v * p_te[:, 1:]
        sel = np.flatnonzero(assign > 0)
        row["true_profit"] = float(true_s[sel, assign[sel] - 1].sum() / ctx["n"])
        row["true_violation"] = violation_mc(assign, cost, p_te, v, budget, ctx["draws"], ctx["rng"])
        row["true_expected_cost_share"] = float((cost[assign[sel]].sum() + v * p_te[sel, assign[sel]].sum()) / budget)
        # diagnostic: is uptake under-estimated for the customers the allocator picked?
        row["p_hat_selected"] = float(p_hat[sel, assign[sel]].mean()) if len(sel) else np.nan
        row["p_true_selected"] = float(p_te[sel, assign[sel]].mean()) if len(sel) else np.nan
        row["p_hat_all"] = float(p_hat[:, 1:].mean())
        row["p_true_all"] = float(p_te[:, 1:].mean())
    return row


def summarise(rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Mean over campaigns (repeat × fold) with t-intervals."""
    metrics = [c for c in ("est_profit", "true_profit", "plugin_violation", "true_violation",
                           "true_expected_cost_share", "effective_budget_share", "n_targeted", "p_hat_selected",
                           "p_true_selected", "p_hat_all", "p_true_all") if c in rows]
    out = []
    for (ds, pol, b), g in rows.groupby(["dataset", "policy", "budget_per_customer"], sort=False):
        rec: dict[str, Any] = {"dataset": ds, "policy": pol, "budget_per_customer": b, "campaigns": len(g)}
        tq = stats.t.ppf(1 - alpha / 2, max(len(g) - 1, 1))
        for c in metrics:
            vals = g[c].dropna()
            if vals.empty:
                continue
            se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            rec.update({c: vals.mean(), f"{c}_ci_low": vals.mean() - tq * se, f"{c}_ci_high": vals.mean() + tq * se})
        out.append(rec)
    return pd.DataFrame(out)


def plot(summary: pd.DataFrame, epsilon: float) -> plt.Figure:
    apply_style()
    pc = summary[summary.dataset == "positive_control"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = dict(zip(dict.fromkeys(pc.policy), (SLOTS[1], SLOTS[0])))
    for name, g in pc.groupby("policy", sort=False):
        x = g.budget_per_customer
        ax1.errorbar(x, g.true_violation, yerr=[g.true_violation - g.true_violation_ci_low,
                                                g.true_violation_ci_high - g.true_violation],
                     fmt="o-", color=colors[name], capsize=3, markersize=5, label=name)
        ax2.errorbar(x, g.true_profit, yerr=[g.true_profit - g.true_profit_ci_low, g.true_profit_ci_high - g.true_profit],
                     fmt="o-", color=colors[name], capsize=3, markersize=5, label=name)
    ax1.axhline(epsilon, color=TEXT_2, linestyle="--", linewidth=1)
    ax1.annotate(f"target ε = {epsilon:g}", (pc.budget_per_customer.min(), epsilon), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color=TEXT_2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("P(campaign cost > budget), true uptake")
    ax1.set_title("Budget overrun probability", loc="left")
    ax2.set_ylabel("True expected profit per customer ($)")
    ax2.set_title("Price of safety", loc="left")
    for ax in (ax1, ax2):
        ax.set_xlabel("Budget ($ per customer in the campaign)")
    ax1.legend(loc="center right")
    fig.suptitle("v1.3: chance-constrained vs mean budget under random voucher uptake (positive control)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "Each point: mean over 15 cross-fitted campaigns of ~12,800 customers, 95% t-interval. "
             "Violation evaluated with the TRUE purchase probabilities.", color=TEXT_2, fontsize=8)
    fig.tight_layout()
    return fig


def run() -> pd.DataFrame:
    from joblib import Parallel, delayed

    cfg = load_config()
    ensure_output_dirs()
    ch = cfg["chance"]
    frames = []
    for ds, spec in ch["datasets"].items():
        d = chance_arrays(ds, cfg)
        reps = Parallel(n_jobs=3)(delayed(crossfit_repeat)(d, spec["learner"], cfg, r)
                                  for r in range(ch["crossfit"]["repeats"]))
        frames.append(pd.DataFrame([row for rep in reps for row in rep]).assign(dataset=ds))
    rows = pd.concat(frames, ignore_index=True)
    summary = summarise(rows, cfg["alpha"])
    rows.to_csv(TABLES_DIR / "chance_campaigns.csv", index=False)
    summary.to_csv(TABLES_DIR / "chance_summary.csv", index=False)
    save(plot(summary, ch["epsilon"]), FIGURES_DIR / "chance_constraint.png")
    return summary

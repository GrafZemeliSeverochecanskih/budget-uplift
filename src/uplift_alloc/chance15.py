"""v1.5: selection-aware chance constraint (fixing the winner's curse found in v1.3b).

v1.3b showed that estimated uptake p-hat is calibrated over all customers but too low among the
customers the allocator selects: the voucher cost enters the score as v * p-hat, so customers whose
uptake is under-estimated look cheap and are preferentially selected. A chance constraint planned
with the same p-hat then overruns the budget far more often than its nominal epsilon.

Two corrections, both estimated on the training fold only:

A) recalibrated (``estimate_kappa``): split the training fold into inner parts; fit tau-hat and p-hat
   on one part, let the allocator select customers on the other part at the same budget per
   customer, and compare the IPW estimate of their purchases under the assigned arm (unbiased because
   arms are randomised with known propensity 1/3) with the sum of p-hat over the same pairs:

       kappa_b = sum_sel 1[T_i = a_i] conv_i / e  /  sum_sel p-hat[i, a_i]   (floored at 1)

   In the campaign p-hat -> min(kappa_b * p-hat, 0.99) in the profit score, the expected cost and the
   variance, and the v1.3 chance-constrained allocator runs unchanged.
B) conservative (``bootstrap_upper_p``): p-hat refit on bootstrap resamples of the training fold;
   its upper quantile replaces p-hat in the constraint's mean and variance (score keeps p-hat).

Assumptions: those of ``chance`` (independent redemptions, SUTVA, known propensity in the RCT);
the selection bias measured on the calibration part transfers to the campaign (same allocator,
same budget per customer, models fit on similarly sized data).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .chance import (allocate_rowcost, campaign_context, campaign_row, chance_allocate, chance_arrays,
                     summarise)
from .config import FIGURES_DIR, TABLES_DIR, arm_costs, ensure_output_dirs, load_config
from .plotting import SLOTS, TEXT_2, apply_style, plt, save

PROPENSITY = 1 / 3
P_MAX = 0.99


def mean_budget_label() -> str:
    return "Mean budget"


def plugin_label(eps: float) -> str:
    return f"Chance-constrained (eps={eps:g})"


def recal_label(eps: float) -> str:
    return f"Chance, recalibrated uptake (eps={eps:g})"


def upper_label(eps: float) -> str:
    return f"Chance, bootstrap-upper uptake (eps={eps:g})"


def score_cost_var(tau_hat: np.ndarray, p: np.ndarray, cost: np.ndarray, voucher: float,
                   margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Profit score S, expected cost C and cost variance V for arms 1..K from uptake p (n, K+1)."""
    pa = p[:, 1:]
    return (margin * tau_hat - cost[1:] - voucher * pa, cost[1:] + voucher * pa,
            voucher**2 * pa * (1 - pa))


def kappa_from_selection(assign: np.ndarray, t: np.ndarray, conv: np.ndarray, p_hat: np.ndarray,
                         propensity: float = PROPENSITY) -> tuple[float, float]:
    """(IPW purchases, sum of p-hat) over the selected (customer, assigned arm) pairs."""
    sel = np.flatnonzero(assign > 0)
    if not len(sel):
        return 0.0, 0.0
    match = t[sel] == assign[sel]
    num = float((conv[sel] * match).sum() / propensity)
    den = float(p_hat[sel, assign[sel]].sum())
    return num, den


def estimate_kappa(X: np.ndarray, t: np.ndarray, spend: np.ndarray, conv: np.ndarray, learner: str,
                   params: dict, cfg: dict, seed: int) -> dict[float, dict[str, float]]:
    """Per-budget uptake bias factor among selected customers, from inner splits of a training fold."""
    from .cate import TLearner, make_learner

    ch, k15 = cfg["chance"], cfg["chance_v15"]
    cost, v, m = arm_costs(cfg), ch["voucher_value"], ch["margin"]
    acc = {b: [0.0, 0.0, 0] for b in ch["budget_per_customer"]}
    inner = StratifiedKFold(k15["calibration_inner_folds"], shuffle=True, random_state=seed)
    for fit, cal in inner.split(X, t):
        tau_hat = make_learner(learner, params, seed).fit(X[fit], t[fit], spend[fit]).effect(X[cal])
        p_hat = np.clip(TLearner(params, seed).fit(X[fit], t[fit], conv[fit]).mu(X[cal]), 1e-4, P_MAX)
        S, C, _ = score_cost_var(tau_hat, p_hat, cost, v, m)
        for b in acc:
            assign = allocate_rowcost(S, C, b * len(cal))
            num, den = kappa_from_selection(assign, t[cal], conv[cal], p_hat)
            acc[b][0] += num
            acc[b][1] += den
            acc[b][2] += int((assign > 0).sum())
    floor = k15["kappa_floor"]
    return {b: {"kappa": max(floor, num / den) if den > 0 else floor, "kappa_raw": num / den if den > 0 else np.nan,
                "ipw_purchases": num, "sum_p_hat": den, "n_selected": n} for b, (num, den, n) in acc.items()}


def bootstrap_upper_p(X_tr: np.ndarray, t_tr: np.ndarray, conv_tr: np.ndarray, X_te: np.ndarray, params: dict,
                      fits: int, quantile: float, seed: int) -> np.ndarray:
    """Upper quantile over bootstrap refits of the per-arm uptake model, (n_te, 3)."""
    from .cate import TLearner

    rng = np.random.default_rng(seed)
    draws = []
    for b in range(fits):
        idx = np.concatenate([rng.choice(np.flatnonzero(t_tr == a), (t_tr == a).sum()) for a in np.unique(t_tr)])
        draws.append(TLearner(params, seed + b).fit(X_tr[idx], t_tr[idx], conv_tr[idx]).mu(X_te))
    return np.clip(np.quantile(np.stack(draws), quantile, axis=0), 1e-4, P_MAX)


def crossfit_repeat(d: dict[str, Any], learner: str, cfg: dict, repeat: int) -> tuple[list[dict], list[dict]]:
    """All four policies on every held-out fold of one repeat (same folds as v1.3b)."""
    from .cate import TLearner, make_learner

    ch, k15, params = cfg["chance"], cfg["chance_v15"], cfg["profit"]["lgbm_params"]
    cost, v, m, eps = arm_costs(cfg), ch["voucher_value"], ch["margin"], ch["epsilon"]
    X, t, spend, conv = d["X"], d["t"], d["spend"], d["conv"]
    rng = np.random.default_rng([cfg["seed"], repeat, 15])
    rows, kappas = [], []
    folds = StratifiedKFold(ch["crossfit"]["folds"], shuffle=True, random_state=cfg["seed"] + repeat)
    for f, (tr, te) in enumerate(folds.split(X, t)):
        seed = cfg["seed"] + repeat
        tau_hat = make_learner(learner, params, seed).fit(X[tr], t[tr], spend[tr]).effect(X[te])
        p_hat = np.clip(TLearner(params, seed).fit(X[tr], t[tr], conv[tr]).mu(X[te]), 1e-4, P_MAX)
        mu_spend = TLearner(params, seed).fit(X[tr], t[tr], spend[tr]).mu(X[te])
        kap = estimate_kappa(X[tr], t[tr], spend[tr], conv[tr], learner, params, cfg, seed)
        p_up = bootstrap_upper_p(X[tr], t[tr], conv[tr], X[te], params, k15["bootstrap_fits"],
                                 k15["upper_quantile"], seed)
        S, C, V = score_cost_var(tau_hat, p_hat, cost, v, m)
        _, C_up, V_up = score_cost_var(tau_hat, p_up, cost, v, m)
        ctx = campaign_context(d, te, t, spend, conv, mu_spend, p_hat, cfg, rng)
        for b_pc in ch["budget_per_customer"]:
            budget = b_pc * len(te)
            kappa = kap[b_pc]["kappa"]
            p_k = np.minimum(kappa * p_hat, P_MAX)
            p_k[:, 0] = p_hat[:, 0]
            S_k, C_k, V_k = score_cost_var(tau_hat, p_k, cost, v, m)
            mean_a = allocate_rowcost(S, C, budget)
            plug_a, plug_eff = chance_allocate(S, C, V, budget, eps)
            recal_a, recal_eff = chance_allocate(S_k, C_k, V_k, budget, eps)
            up_a, up_eff = chance_allocate(S, C_up, V_up, budget, eps)  # score from p-hat, moments from upper p
            planned = {mean_budget_label(): (mean_a, C, V, 1.0), plugin_label(eps): (plug_a, C, V, plug_eff / budget),
                       recal_label(eps): (recal_a, C_k, V_k, recal_eff / budget),
                       upper_label(eps): (up_a, C_up, V_up, up_eff / budget)}
            for name, (assign, Cp, Vp, share) in planned.items():
                rows.append({"repeat": repeat, "fold": f, "policy": name, "budget_per_customer": b_pc,
                             "effective_budget_share": share, "kappa": kappa if "recalibrated" in name else np.nan,
                             **campaign_row(assign, Cp, Vp, budget, ctx)})
            kappas.append({"repeat": repeat, "fold": f, "budget_per_customer": b_pc, **kap[b_pc]})
    return rows, kappas


def plot(summary: pd.DataFrame, epsilon: float) -> plt.Figure:
    apply_style()
    pc = summary[summary.dataset == "positive_control"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(dict.fromkeys(pc.policy))
    colors = dict(zip(names, (SLOTS[1], SLOTS[0], SLOTS[2], SLOTS[6])))
    offsets = dict(zip(names, np.linspace(-0.0018, 0.0018, len(names))))
    for name, g in pc.groupby("policy", sort=False):
        x = g.budget_per_customer + offsets[name]
        ax1.errorbar(x, g.true_violation, yerr=[g.true_violation - g.true_violation_ci_low,
                                                g.true_violation_ci_high - g.true_violation],
                     fmt="o-", color=colors[name], capsize=3, markersize=5, label=name)
        ax2.errorbar(x, g.true_profit, yerr=[g.true_profit - g.true_profit_ci_low, g.true_profit_ci_high - g.true_profit],
                     fmt="o-", color=colors[name], capsize=3, markersize=5, label=name)
    ax1.axhline(epsilon, color=TEXT_2, linestyle="--", linewidth=1)
    ax1.annotate(f"target eps = {epsilon:g}", (pc.budget_per_customer.min(), epsilon), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color=TEXT_2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("P(campaign cost > budget), true uptake")
    ax1.set_title("Budget overrun probability", loc="left")
    ax2.set_ylabel("True expected profit per customer ($)")
    ax2.set_title("Price of safety", loc="left")
    for ax in (ax1, ax2):
        ax.set_xlabel("Budget ($ per customer in the campaign)")
    handles, labels = ax1.get_legend_handles_labels()
    fig.suptitle("Selection-aware chance constraints under random voucher uptake (positive control)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, -0.03, "Mean over cross-fitted campaigns of ~12,800 customers, 95% t-interval. Overrun evaluated "
             "with the TRUE purchase probabilities.", color=TEXT_2, fontsize=8)
    return fig


def run() -> dict[str, pd.DataFrame]:
    from joblib import Parallel, delayed

    cfg = load_config()
    ensure_output_dirs()
    ch = cfg["chance"]
    frames, kframes = [], []
    for ds, spec in ch["datasets"].items():
        d = chance_arrays(ds, cfg)
        reps = Parallel(n_jobs=3)(delayed(crossfit_repeat)(d, spec["learner"], cfg, r)
                                  for r in range(ch["crossfit"]["repeats"]))
        frames.append(pd.DataFrame([row for rows, _ in reps for row in rows]).assign(dataset=ds))
        kframes.append(pd.DataFrame([k for _, ks in reps for k in ks]).assign(dataset=ds))
    rows = pd.concat(frames, ignore_index=True)
    kappas = pd.concat(kframes, ignore_index=True)
    summary = summarise(rows, cfg["alpha"])
    kappa_summary = (kappas.groupby(["dataset", "budget_per_customer"])
                     .agg(kappa=("kappa", "mean"), kappa_sd=("kappa", "std"), kappa_raw=("kappa_raw", "mean"),
                          campaigns=("kappa", "size")).reset_index())
    rows.to_csv(TABLES_DIR / "chance15_campaigns.csv", index=False)
    summary.to_csv(TABLES_DIR / "chance15_summary.csv", index=False)
    kappas.to_csv(TABLES_DIR / "chance15_kappa.csv", index=False)
    kappa_summary.to_csv(TABLES_DIR / "chance15_kappa_summary.csv", index=False)
    save(plot(summary, ch["epsilon"]), FIGURES_DIR / "chance15.png")
    return {"summary": summary, "kappa": kappa_summary}

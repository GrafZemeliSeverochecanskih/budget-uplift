"""v1.3a: direct (doubly robust) policy learning vs estimate-then-optimise, under a budget.

Policy learning (Athey & Wager) skips the CATE model: it builds per-customer, per-arm reward
estimates and picks the policy in a restricted class that maximises their sum. Here

    R[i, a] = margin * Gamma[i, a] - cost[a],
    Gamma[i, a] = mu_a(x_i) + 1[T_i = a] (Y_i - mu_a(x_i)) / e(a)      (DR pseudo-outcome, e = 1/3)

and the class is depth-limited policy trees (EconML ``PolicyTree``). Within each cross-fitting
training fold, mu is itself cross-fitted (``inner_folds``) so the tree never sees rewards built from
a model fit on the same customer. The budget enters as a penalty lam * cost[a], with lam the
smallest grid value whose training-fold spend fits the budget.

Evaluation reuses v1.2: repeated cross-fitting on all customers, DR profit scores on held-out
folds, DML-combined intervals. Assumptions: SUTVA, known propensity 1/3, fixed margin and costs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .config import FIGURES_DIR, TABLES_DIR, arm_costs, budget_for, ensure_output_dirs, load_config
from .plotting import NEUTRAL, SLOTS, TEXT, TEXT_2, apply_style, plt, save

PROPENSITY = 1 / 3


def tree_label(depth: int) -> str:
    return f"Policy tree (depth {depth})"


def eto_label(learner: str) -> str:
    return f"Estimate-then-optimise ({learner})"


def lambda_grid(spec: dict[str, Any]) -> np.ndarray:
    grid = np.geomspace(spec["min"], spec["max"], spec["size"])
    return np.r_[0.0, grid] if spec.get("zero", True) else grid


def dr_pseudo_outcomes(X: np.ndarray, t: np.ndarray, y: np.ndarray, params: dict, folds: int,
                       seed: int) -> np.ndarray:
    """(n, 3) Gamma[i, a] with mu cross-fitted over ``folds`` splits of the given customers."""
    from .cate import TLearner

    gamma = np.zeros((len(y), 3))
    for fit_idx, pred_idx in KFold(folds, shuffle=True, random_state=seed).split(X):
        mu = TLearner(params, seed).fit(X[fit_idx], t[fit_idx], y[fit_idx]).mu(X[pred_idx])
        k, arm = np.arange(len(pred_idx)), t[pred_idx]
        mu[k, arm] += (y[pred_idx] - mu[k, arm]) / PROPENSITY
        gamma[pred_idx] = mu
    return gamma


def fit_budgeted_trees(X: np.ndarray, rewards: np.ndarray, cost: np.ndarray, lambdas: np.ndarray, depth: int,
                       min_leaf: int, seed: int) -> list[tuple[float, Any, float]]:
    """Fit one tree per lam on penalised rewards; returns (lam, tree, mean training cost per customer)."""
    from econml.policy import PolicyTree

    out = []
    for lam in lambdas:
        tree = PolicyTree(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed)
        tree.fit(X, rewards - lam * cost)
        out.append((float(lam), tree, float(cost[tree.predict(X)].mean())))
    return out


def pick_tree(trees: list[tuple[float, Any, float]], budget_per_customer: float) -> tuple[float, Any]:
    """Smallest lam whose training spend per customer fits the budget (largest lam if none does)."""
    for lam, tree, spend in trees:
        if spend <= budget_per_customer + 1e-12:
            return lam, tree
    return trees[-1][0], trees[-1][1]


def crossfit_repeat(X: np.ndarray, t: np.ndarray, y: np.ndarray, learner: str, cfg: dict, repeat: int,
                    tau_true: np.ndarray | None = None) -> dict[str, dict[str, np.ndarray]]:
    """One repeat: trees and estimate-then-optimise on identical folds; v1.2-style score matrices."""
    from .allocate import lagrangian_allocate
    from .cate import TLearner, make_learner
    from .ope import dr_scores, nobody_policy, one_hot, random_policy, single_arm_policy

    pl, params = cfg["policy_learning"], cfg["profit"]["lgbm_params"]
    fractions, m, seed = pl["budget_fractions"], pl["margin"], cfg["seed"] + repeat
    cost = arm_costs(cfg)
    lambdas = lambda_grid(pl["lambda_grid"])
    n, nb = len(y), len(fractions)
    out: dict[str, dict[str, np.ndarray]] = {}
    folds = StratifiedKFold(pl["crossfit"]["folds"], shuffle=True, random_state=cfg["seed"] + repeat)
    for tr, te in folds.split(X, t):
        gamma = dr_pseudo_outcomes(X[tr], t[tr], y[tr], params, pl["inner_folds"], seed)
        rewards = m * gamma - cost
        trees = {d: fit_budgeted_trees(X[tr], rewards, cost, lambdas, d, pl["min_samples_leaf"], seed)
                 for d in pl["tree_depths"]}
        eto = make_learner(learner, params, seed).fit(X[tr], t[tr], y[tr])
        tau_hat = eto.effect(X[te])
        mu = TLearner(params, seed).fit(X[tr], t[tr], y[tr]).mu(X[te])  # evaluation nuisance, train folds only
        best_arm = int(np.argmax([y[tr][t[tr] == a].mean() - y[tr][t[tr] == 0].mean() for a in (1, 2)]) + 1)
        base = dr_scores(nobody_policy(len(te)), t[te], y[te], mu)
        for b, frac in enumerate(fractions):
            per_customer = budget_for(frac, 1, cfg)
            budget = per_customer * len(te)
            pols = {tree_label(d): one_hot(pick_tree(trees[d], per_customer)[1].predict(X[te])) for d in trees}
            pols[eto_label(learner)] = one_hot(
                lagrangian_allocate(m * tau_hat - cost[1:], cost[1:], budget, repair=True).assign)
            pols["Best single arm"] = single_arm_policy(len(te), budget, cost, best_arm)
            pols["Random targeting"] = random_policy(len(te), budget, cost)
            for name, pi in pols.items():
                slot = out.setdefault(name, {k: np.zeros((n, nb)) for k in ("rev", "cost", "true_rev")})
                slot["rev"][te, b] = dr_scores(pi, t[te], y[te], mu) - base
                slot["cost"][te, b] = pi @ cost
                if tau_true is not None:
                    slot["true_rev"][te, b] = (pi[:, 1:] * tau_true[te]).sum(axis=1)
    return out


def final_tree_rules(X: np.ndarray, t: np.ndarray, y: np.ndarray, feature_names: list[str], cfg: dict,
                     depth: int = 2) -> str:
    """Depth-2 tree on ALL customers (unconstrained) rendered as text, for interpretation only."""
    from econml.policy import PolicyTree

    pl = cfg["policy_learning"]
    gamma = dr_pseudo_outcomes(X, t, y, cfg["profit"]["lgbm_params"], pl["inner_folds"], cfg["seed"])
    tree = PolicyTree(max_depth=depth, min_samples_leaf=pl["min_samples_leaf"], random_state=cfg["seed"])
    tree.fit(X, pl["margin"] * gamma - arm_costs(cfg))
    return render_tree(tree, feature_names)


def render_tree(tree: Any, feature_names: list[str]) -> str:
    """Plain-text rules of a fitted EconML PolicyTree (arm names from ``ARM_NAMES``)."""
    from .config import ARM_NAMES

    t_ = tree.tree_
    lines: list[str] = []

    def walk(node: int, indent: str) -> None:
        if t_.children_left[node] == -1:
            lines.append(f"{indent}-> {ARM_NAMES[int(np.argmax(t_.value[node]))]} ({int(t_.n_node_samples[node])} customers)")
            return
        name, thr = feature_names[t_.feature[node]], t_.threshold[node]
        lines.append(f"{indent}if {name} <= {thr:.2f}:")
        walk(t_.children_left[node], indent + "    ")
        lines.append(f"{indent}else:  # {name} > {thr:.2f}")
        walk(t_.children_right[node], indent + "    ")

    walk(0, "")
    return "\n".join(lines)


def plot(curves: pd.DataFrame, margin: float) -> plt.Figure:
    apply_style()
    datasets = list(dict.fromkeys(curves.dataset))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.3))
    roles = {"Estimate-then-optimise": SLOTS[0], "Policy tree (depth 2)": SLOTS[6], "Policy tree (depth 3)": SLOTS[4],
             "Best single arm": SLOTS[2], "Random targeting": SLOTS[1]}

    def color(name: str) -> str:
        return next(c for role, c in roles.items() if name.startswith(role))

    fr = sorted(curves.budget_fraction.unique())
    offsets = np.linspace(-0.035, 0.035, len(roles))
    titles = {"hillstrom": "Real Hillstrom outcomes", "positive_control": "Positive control (known effect)"}
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        sub = curves[curves.dataset == ds]
        names = sorted(dict.fromkeys(sub.policy), key=lambda k: list(roles).index(next(r for r in roles if k.startswith(r))))
        for off, name in zip(offsets, names):
            g = sub[sub.policy == name].set_index("budget_fraction").loc[fr]
            x = np.arange(len(fr)) + off * 4
            ax.errorbar(x, g.profit, yerr=[g.profit - g.profit_ci_low, g.profit_ci_high - g.profit], fmt="o",
                        color=color(name), capsize=2, markersize=5, label=name)
            if ds != "hillstrom":
                ax.plot(x, g.true_profit, "x", color=TEXT, markersize=5, label="_nolegend_")
        ax.axhline(0, color=NEUTRAL, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(fr)), [("unconstrained" if f > 1 else f"{f:g}") for f in fr])
        ax.set_xlabel("Budget (fraction of customers at mean e-mail cost)")
        ax.set_ylabel(f"Incremental profit per customer ($, margin {margin:.0%})")
        ax.set_title(titles.get(ds, ds), loc="left")
    handles, labels = [], []
    for ax in np.atleast_1d(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.suptitle("Direct policy learning vs estimate-then-optimise (cross-fitted, 95% CI)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=(len(labels) + 1) // 2, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, -0.07, "× = known true profit (positive control). Same folds and customers for every policy.",
             color=TEXT_2, fontsize=8)
    return fig


def run() -> dict[str, pd.DataFrame]:
    from joblib import Parallel, delayed

    from .data import encode_features, load_frame
    from .profit import dataset_arrays, paired_differences, profit_curves

    cfg = load_config()
    ensure_output_dirs()
    pl = cfg["policy_learning"]
    curves, diffs = [], []
    for ds, spec in pl["datasets"].items():
        X, t, y, tau = dataset_arrays(ds, cfg)
        repeats = Parallel(n_jobs=4)(delayed(crossfit_repeat)(X, t, y, spec["learner"], cfg, r, tau)
                                     for r in range(pl["crossfit"]["repeats"]))
        curves.append(profit_curves(repeats, pl["budget_fractions"], [pl["margin"]], cfg["alpha"], ds))
        diffs.append(paired_differences(repeats, pl["budget_fractions"], pl["margin"], cfg["alpha"], ds,
                                        ours=eto_label(spec["learner"]),
                                        others=[tree_label(d) for d in pl["tree_depths"]]))
    out = {"curves": pd.concat(curves, ignore_index=True), "differences": pd.concat(diffs, ignore_index=True)}
    for name, table in out.items():
        table.to_csv(TABLES_DIR / f"policylearn_{name}.csv", index=False)
    save(plot(out["curves"], pl["margin"]), FIGURES_DIR / "policy_learning.png")
    X, t, y, _ = dataset_arrays("hillstrom", cfg)
    rules = final_tree_rules(X, t, y, list(encode_features(load_frame()).columns), cfg)
    (TABLES_DIR / "policylearn_tree_hillstrom.txt").write_text(rules + "\n", encoding="utf-8")
    return out

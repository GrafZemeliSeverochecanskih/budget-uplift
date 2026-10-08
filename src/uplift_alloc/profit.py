"""v1.2: profit objective with cross-fitted, repeated policy evaluation.

Incremental profit of a policy pi vs sending nothing, per customer:
    margin * (DR_i(pi) - DR_i(nobody)) - sum_a pi[i, a] * cost[a]
where DR_i are the doubly robust scores of ``ope.dr_scores`` (propensity 1/3 by design). The cost
term is known exactly; only the revenue term is estimated.

Cross-fitting: customers are split into K arm-stratified folds; for each fold the learner and its
outcome model are fit on the other K-1 folds, the policy is allocated on the held-out fold and
scored there. Every customer is evaluated by a model that never saw them, all customers are used
for evaluation, and repeating with R different fold splits lets the interval include model-refit
variance (DML-style combination, ``combine_repeats``).

Assumptions: SUTVA; known propensity 1/3; costs and margin are fixed, assumed constants; the
margin applies to incremental spend (no returns, no cross-period effects).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from .config import FIGURES_DIR, TABLES_DIR, arm_costs, budget_for, ensure_output_dirs, load_config
from .plotting import NEUTRAL, SLOTS, TEXT, TEXT_2, apply_style, plt, save

BASELINES = ["Revenue-targeted (v1.0)", "Best single arm", "Random targeting"]


def profit_label(margin: float) -> str:
    return f"Profit-targeted (m={margin:g})"


def outcome_means(model: Any, X: np.ndarray) -> np.ndarray:
    """mu-hat(x, a) from the learner's own outcome model (S: joint model; X: its T-learner stage)."""
    if hasattr(model, "mu"):
        return model.mu(X)
    if hasattr(model, "t_learner"):
        return model.t_learner.mu(X)
    raise TypeError(f"{type(model).__name__} exposes no outcome model")


def fold_policies(tau_hat: np.ndarray, n: int, budget: float, cost: np.ndarray, margins: list[float],
                  best_arm: int, tau_true: np.ndarray | None, margin_main: float) -> dict[str, np.ndarray]:
    """All policies for one held-out fold at one budget, as (n, 3) assignment-probability matrices."""
    from .allocate import lagrangian_allocate
    from .ope import one_hot, random_policy, single_arm_policy

    def alloc(scores: np.ndarray) -> np.ndarray:
        return one_hot(lagrangian_allocate(scores, cost[1:], budget, repair=True).assign)

    pols = {"Revenue-targeted (v1.0)": alloc(tau_hat),
            "Best single arm": single_arm_policy(n, budget, cost, best_arm),
            "Random targeting": random_policy(n, budget, cost)}
    for m in margins:
        pols[profit_label(m)] = alloc(m * tau_hat - cost[1:])
    if tau_true is not None:
        pols["Oracle (true τ, profit)"] = alloc(margin_main * tau_true - cost[1:])
    return pols


def crossfit_repeat(X: np.ndarray, t: np.ndarray, y: np.ndarray, learner: str, params: dict, cfg: dict,
                    repeat: int, tau_true: np.ndarray | None = None) -> dict[str, dict[str, np.ndarray]]:
    """One repeat of K-fold cross-fitting.

    Returns, per policy, (n, n_budgets) matrices: ``rev`` = DR incremental revenue score,
    ``cost`` = expected e-mail cost per customer and, if ``tau_true`` is given, ``true_rev``.
    """
    from .cate import make_learner
    from .ope import dr_scores, nobody_policy

    pc = cfg["profit"]
    fractions, margins = pc["budget_fractions"], [pc["margin"], *pc["margin_sensitivity"]]
    cost = arm_costs(cfg)
    n, nb = len(y), len(fractions)
    out: dict[str, dict[str, np.ndarray]] = {}
    folds = StratifiedKFold(pc["crossfit"]["folds"], shuffle=True, random_state=cfg["seed"] + repeat)
    for tr, te in folds.split(X, t):
        model = make_learner(learner, params, cfg["seed"] + repeat).fit(X[tr], t[tr], y[tr])
        tau_hat, mu = model.effect(X[te]), outcome_means(model, X[te])
        best_arm = int(np.argmax([y[tr][t[tr] == a].mean() - y[tr][t[tr] == 0].mean() for a in (1, 2)]) + 1)
        base = dr_scores(nobody_policy(len(te)), t[te], y[te], mu)
        for b, frac in enumerate(fractions):
            budget = budget_for(frac, len(te), cfg)
            pols = fold_policies(tau_hat, len(te), budget, cost, margins, best_arm,
                                 None if tau_true is None else tau_true[te], pc["margin"])
            for name, pi in pols.items():
                slot = out.setdefault(name, {k: np.zeros((n, nb)) for k in ("rev", "cost", "true_rev")})
                slot["rev"][te, b] = dr_scores(pi, t[te], y[te], mu) - base
                slot["cost"][te, b] = pi @ cost
                if tau_true is not None:
                    slot["true_rev"][te, b] = (pi[:, 1:] * tau_true[te]).sum(axis=1)
    return out


def combine_repeats(estimates: np.ndarray, ses: np.ndarray, alpha: float) -> tuple[float, float, float, float]:
    """Median-aggregated estimate over repeats with sigma^2 = median(se_r^2 + (theta_r - theta)^2)."""
    theta = float(np.median(estimates))
    se = float(np.sqrt(np.median(ses**2 + (estimates - theta) ** 2)))
    z = stats.norm.ppf(1 - alpha / 2)
    return theta, se, theta - z * se, theta + z * se


def _per_repeat(mats: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(R, n_budgets) means and standard errors of per-customer score matrices."""
    est = np.array([m.mean(axis=0) for m in mats])
    se = np.array([m.std(axis=0, ddof=1) / np.sqrt(m.shape[0]) for m in mats])
    return est, se


def profit_curves(repeats: list[dict], fractions: list[float], margins: list[float], alpha: float,
                  dataset: str) -> pd.DataFrame:
    """Profit (and revenue) per policy, margin and budget with DML-combined cross-fitted CIs."""
    rows = []
    policies = list(repeats[0])
    for m in margins:
        for name in policies:
            if name.startswith("Profit-targeted") and name != profit_label(m):
                continue  # each profit-targeted policy is evaluated at its own margin
            if name.startswith("Oracle") and m != margins[0]:
                continue
            prof = [m * r[name]["rev"] - r[name]["cost"] for r in repeats]
            p_est, p_se = _per_repeat(prof)
            r_est, r_se = _per_repeat([r[name]["rev"] for r in repeats])
            truth = [m * r[name]["true_rev"] - r[name]["cost"] for r in repeats]
            for b, frac in enumerate(fractions):
                p = combine_repeats(p_est[:, b], p_se[:, b], alpha)
                rv = combine_repeats(r_est[:, b], r_se[:, b], alpha)
                rows.append({
                    "dataset": dataset, "margin": m, "policy": name, "budget_fraction": frac,
                    "profit": p[0], "profit_se": p[1], "profit_ci_low": p[2], "profit_ci_high": p[3],
                    "revenue": rv[0], "revenue_ci_low": rv[2], "revenue_ci_high": rv[3],
                    "revenue_ci_width": rv[3] - rv[2],
                    "cost_spent": float(np.median([r[name]["cost"][:, b].mean() for r in repeats])),
                    "true_profit": float(np.median([tr[:, b].mean() for tr in truth])) if dataset != "hillstrom" else np.nan,
                })
    return pd.DataFrame(rows)


def paired_differences(repeats: list[dict], fractions: list[float], margin: float, alpha: float,
                       dataset: str) -> pd.DataFrame:
    """Profit-targeted minus each baseline, per budget, at the main margin (paired, DML-combined)."""
    ours = profit_label(margin)
    rows = []
    for other in BASELINES:
        diffs = [(margin * r[ours]["rev"] - r[ours]["cost"]) - (margin * r[other]["rev"] - r[other]["cost"])
                 for r in repeats]
        est, se = _per_repeat(diffs)
        for b, frac in enumerate(fractions):
            d = combine_repeats(est[:, b], se[:, b], alpha)
            rows.append({"dataset": dataset, "comparison": f"{ours} − {other}", "budget_fraction": frac,
                         "difference": d[0], "ci_low": d[2], "ci_high": d[3]})
    return pd.DataFrame(rows)


def optimal_budget(scores: np.ndarray, fractions: list[float], reps: int, seed: int, alpha: float) -> dict[str, float]:
    """Argmax of mean profit over the budget grid, with a customer-bootstrap percentile interval."""
    rng = np.random.default_rng(seed)
    n = scores.shape[0]
    fr = np.asarray(fractions)
    point = int(np.argmax(scores.mean(axis=0)))
    boot = np.array([fr[np.argmax(scores[rng.integers(0, n, n)].mean(axis=0))] for _ in range(reps)])
    return {"optimal_fraction": float(fr[point]), "optimal_profit": float(scores.mean(axis=0)[point]),
            "fraction_ci_low": float(np.quantile(boot, alpha / 2)),
            "fraction_ci_high": float(np.quantile(boot, 1 - alpha / 2)),
            "share_bootstrap_at_point": float(np.mean(boot == fr[point]))}


def optimal_budgets(repeats: list[dict], fractions: list[float], margins: list[float], cfg: dict,
                    dataset: str) -> pd.DataFrame:
    rows = []
    for m in margins:
        for name in [profit_label(m), *BASELINES]:
            avg = np.mean([m * r[name]["rev"] - r[name]["cost"] for r in repeats], axis=0)
            res = optimal_budget(avg, fractions, cfg["profit"]["optimum_bootstrap_reps"], cfg["seed"], cfg["alpha"])
            if dataset != "hillstrom":
                true_curve = np.mean([(m * r[name]["true_rev"] - r[name]["cost"]).mean(axis=0) for r in repeats], axis=0)
                res["true_optimal_fraction"] = float(np.asarray(fractions)[np.argmax(true_curve)])
            rows.append({"dataset": dataset, "margin": m, "policy": name, **res})
    return pd.DataFrame(rows)


def dataset_arrays(name: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """X, arm, spend and (positive control only) true tau for all 64,000 customers."""
    from .data import encode_arm, encode_features, load_frame
    from .positive_control import simulate, true_tau

    df = load_frame()
    X, t = encode_features(df).to_numpy(), encode_arm(df["segment"])
    if name == "hillstrom":
        return X, t, df["spend"].to_numpy(dtype=float), None
    pc = cfg["positive_control"]
    y = simulate(df, t, pc, np.random.default_rng(cfg["seed"] + pc["seed_offset"]))["spend"].to_numpy()
    return X, t, y, true_tau(df, pc)["spend"]


def run_dataset(name: str, cfg: dict, n_jobs: int = 4) -> tuple[list[dict], np.ndarray | None]:
    from joblib import Parallel, delayed

    X, t, y, tau = dataset_arrays(name, cfg)
    learner = cfg["profit"]["datasets"][name]["learner"]
    repeats = Parallel(n_jobs=n_jobs)(
        delayed(crossfit_repeat)(X, t, y, learner, cfg["profit"]["lgbm_params"], cfg, r, tau)
        for r in range(cfg["profit"]["crossfit"]["repeats"]))
    return repeats, tau


def split_vs_crossfit(curves: pd.DataFrame) -> pd.DataFrame:
    """v1.0 test-split DR CI width vs v1.2 cross-fitted CI width, revenue-targeted policy, real spend."""
    path = TABLES_DIR / "policy_values.csv"
    if not path.exists():
        return pd.DataFrame()
    v10 = pd.read_csv(path)
    v10 = v10[(v10.estimator == "dr") & (v10.outcome == "spend") & v10.policy.str.startswith("Uplift")]
    cf = curves[(curves.dataset == "hillstrom") & (curves.policy == "Revenue-targeted (v1.0)")
                & (curves.margin == curves.margin.iloc[0])]
    m = v10.merge(cf, on="budget_fraction")
    return pd.DataFrame({
        "budget_fraction": m.budget_fraction,
        "v10_split_estimate": m.incremental, "v10_split_ci_width": m.incremental_ci_high - m.incremental_ci_low,
        "v12_crossfit_estimate": m.revenue, "v12_crossfit_ci_width": m.revenue_ci_width,
    }).assign(width_ratio=lambda d: d.v12_crossfit_ci_width / d.v10_split_ci_width)


def plot_profit(curves: pd.DataFrame, optima: pd.DataFrame, margin: float) -> plt.Figure:
    apply_style()
    datasets = list(dict.fromkeys(curves.dataset))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.3))
    colors = {profit_label(margin): SLOTS[0], "Revenue-targeted (v1.0)": SLOTS[6], "Best single arm": SLOTS[2],
              "Random targeting": SLOTS[1], "Oracle (true τ, profit)": TEXT}
    titles = {"hillstrom": "Real Hillstrom outcomes", "positive_control": "Positive control (known effect)"}
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        sub = curves[(curves.dataset == ds) & (curves.margin == margin)]
        for name, color in colors.items():
            g = sub[sub.policy == name].sort_values("budget_fraction")
            if g.empty:
                continue
            oracle = name.startswith("Oracle")
            ax.fill_between(g.budget_fraction, g.profit_ci_low, g.profit_ci_high, color=color, alpha=0.10, linewidth=0)
            ax.plot(g.budget_fraction, g.profit, color=color, marker="o", markersize=3.5, label=name,
                    linestyle=":" if oracle else "-")
            if ds != "hillstrom":
                ax.plot(g.budget_fraction, g.true_profit, color=color, marker="x", markersize=5, linestyle="none")
        opt = optima[(optima.dataset == ds) & (optima.margin == margin) & (optima.policy == profit_label(margin))].iloc[0]
        ytop = ax.get_ylim()[1]
        ax.errorbar(opt.optimal_fraction, ytop * 0.93,
                    xerr=[[opt.optimal_fraction - opt.fraction_ci_low], [opt.fraction_ci_high - opt.optimal_fraction]],
                    fmt="v", color=SLOTS[0], capsize=3, markersize=7)
        ax.annotate("profit-optimal budget (95% CI)", (opt.optimal_fraction, ytop * 0.93), xytext=(6, 6),
                    textcoords="offset points", fontsize=7, color=TEXT_2)
        ax.axhline(0, color=NEUTRAL, linestyle="--", linewidth=1)
        ax.set_title(titles.get(ds, ds), loc="left")
        ax.set_xlabel("Budget (fraction of customers at mean e-mail cost; 1.2 = unconstrained)")
        ax.set_ylabel(f"Incremental profit per customer ($, margin {margin:.0%})")
    handles, labels = np.atleast_1d(axes)[-1].get_legend_handles_labels()
    fig.suptitle("v1.2: profit vs budget, cross-fitted on all 64,000 customers (95% CI incl. refit variance)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
    fig.text(0.01, -0.04, "× = known true profit (positive control). Profit = margin × DR incremental spend − e-mail "
             "cost spent. Profit- and revenue-targeted coincide while every top-ranked customer is profitable.",
             color=TEXT_2, fontsize=8)
    return fig


def run() -> dict[str, pd.DataFrame]:
    cfg = load_config()
    ensure_output_dirs()
    pc = cfg["profit"]
    fractions, margins = pc["budget_fractions"], [pc["margin"], *pc["margin_sensitivity"]]
    curves, diffs, optima = [], [], []
    for ds in pc["datasets"]:
        repeats, _ = run_dataset(ds, cfg)
        curves.append(profit_curves(repeats, fractions, margins, cfg["alpha"], ds))
        diffs.append(paired_differences(repeats, fractions, pc["margin"], cfg["alpha"], ds))
        optima.append(optimal_budgets(repeats, fractions, margins, cfg, ds))
    out = {"curves": pd.concat(curves, ignore_index=True), "differences": pd.concat(diffs, ignore_index=True),
           "optima": pd.concat(optima, ignore_index=True)}
    out["split_vs_crossfit"] = split_vs_crossfit(out["curves"])
    for name, table in out.items():
        table.to_csv(TABLES_DIR / f"profit_{name}.csv", index=False)
    save(plot_profit(out["curves"], out["optima"], pc["margin"]), FIGURES_DIR / "profit_vs_budget.png")
    return out

"""v1.1: when does targeting pay off? Sweep effect heterogeneity x sample size on the semi-synthetic generator.

Each replication draws n customers (covariates resampled from Hillstrom, arms randomised with
propensity 1/3), simulates outcomes from the v1.0 positive-control generator with heterogeneity h
(see ``generator``), splits 70/30, fits the learner on train, allocates the test customers under the
budget, and records

* the TRUE incremental value of the targeted policy minus random targeting / the best single arm
  (known, because the generator's tau is known), and
* the doubly robust ESTIMATE of the same paired difference with an influence-function 95% CI, i.e.
  what an analyst holding one experiment of size n would conclude.

Assumptions: those of the generator (SUTVA, randomisation, additive effects) and of ``ope``
(known propensity 1/3). The average Womens effect is held fixed across h (DECISIONS V1).
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import FIGURES_DIR, TABLES_DIR, arm_costs, budget_for, ensure_output_dirs, load_config
from .plotting import SLOTS, SURFACE, TEXT, TEXT_2, apply_style, plt, save


def generator(pc: dict[str, Any], h: float, mean_effect: dict[str, float], share: float) -> dict[str, Any]:
    """Positive-control generator whose Womens effect has mean ``mean_effect`` and concentration h.

    inside the segment: mean * (1 + h (1 - s) / s); outside: mean * (1 - h). h = 0 is uniform,
    h = 1 puts the whole effect in the segment. The customer-average effect is ``mean`` for any h.
    """
    if not 0 <= h <= 1:
        raise ValueError("heterogeneity must be in [0, 1]")
    out = copy.deepcopy(pc)
    womens = out["effects"]["womens"]
    for key in ("purchase", "visit"):
        womens[key] = mean_effect[key] * (1 + h * (1 - share) / share)
        womens[f"{key}_outside"] = mean_effect[key] * (1 - h)
    return out


def run_replication(df_all: pd.DataFrame, n: int, h: float, rep: int, cfg: dict[str, Any], share: float) -> dict[str, Any]:
    """One simulated experiment of size n at heterogeneity h; returns true and estimated gains."""
    from .allocate import lagrangian_allocate
    from .cate import SLearner, make_learner
    from .data import encode_features, make_split
    from .ope import dr_scores, one_hot, random_policy, single_arm_policy, true_incremental
    from .positive_control import simulate, true_tau

    sw, seed = cfg["sweep"], cfg["seed"]
    rng = np.random.default_rng([seed, int(round(h * 1000)), n, rep])
    df = df_all.iloc[rng.integers(0, len(df_all), n)].reset_index(drop=True)
    t = rng.integers(0, 3, n)
    pc = generator(cfg["positive_control"], h, sw["womens_mean_effect"], share)
    y = simulate(df, t, pc, rng)[sw["outcome"]].to_numpy()
    tau = true_tau(df, pc)[sw["outcome"]]
    X = encode_features(df).to_numpy()
    split = make_split(t, cfg["split"]["test_size"], int(rng.integers(2**31)))
    tr, te = np.array(split["train"]), np.array(split["test"])

    model = make_learner(sw["learner"], sw["lgbm_params"], seed)
    model.fit(X[tr], t[tr], y[tr])
    tau_hat = model.effect(X[te])
    # DR nuisance: the S-learner's own outcome model if available, else a separate T-learner fit on train.
    mu = model.mu(X[te]) if isinstance(model, SLearner) else make_learner("T", sw["lgbm_params"], seed).fit(
        X[tr], t[tr], y[tr]).mu(X[te])

    cost = arm_costs(cfg)
    n_te = len(te)
    budget = budget_for(sw["budget_fraction"], n_te, cfg)
    repair = sw["allocation"] == "lagrangian_repair"

    def allocate(scores: np.ndarray) -> np.ndarray:
        return lagrangian_allocate(scores, cost[1:], budget, repair=repair).assign

    best_arm = int(np.argmax([y[tr][t[tr] == a].mean() - y[tr][t[tr] == 0].mean() for a in (1, 2)]) + 1)
    policies = {
        "targeted": one_hot(allocate(tau_hat)),
        "oracle": one_hot(allocate(tau[te])),
        "random": random_policy(n_te, budget, cost),
        "best_arm": single_arm_policy(n_te, budget, cost, best_arm),
    }
    true = {k: true_incremental(p, tau[te]) for k, p in policies.items()}
    scores = {k: dr_scores(p, t[te], y[te], mu) for k, p in policies.items()}
    z = stats.norm.ppf(1 - cfg["alpha"] / 2)
    row: dict[str, Any] = {"heterogeneity": h, "n": n, "rep": rep, "n_test": n_te, "best_arm": best_arm}
    for base in ("random", "best_arm"):
        diff = scores["targeted"] - scores[base]
        est, se = diff.mean(), diff.std(ddof=1) / np.sqrt(n_te)
        row.update({
            f"true_gain_vs_{base}": true["targeted"] - true[base],
            f"oracle_gain_vs_{base}": true["oracle"] - true[base],
            f"est_gain_vs_{base}": est, f"est_ci_low_vs_{base}": est - z * se, f"est_ci_high_vs_{base}": est + z * se,
            f"detected_vs_{base}": bool(est - z * se > 0),
        })
    w = tau[te][:, 1]
    row["corr_tau_womens"] = float(np.corrcoef(tau_hat[:, 1], w)[0, 1]) if np.ptp(w) > 0 else np.nan
    return row


def run_grid(cfg: dict[str, Any], df_all: pd.DataFrame, share: float, n_jobs: int = 4) -> pd.DataFrame:
    """All replications of all (h, n) cells. Each replication is seeded by (seed, h, n, rep)."""
    from joblib import Parallel, delayed

    sw = cfg["sweep"]
    jobs = [(n, h, r) for n in sw["sample_sizes"] for h in sw["heterogeneity"] for r in range(sw["replications"])]
    rows = Parallel(n_jobs=n_jobs)(delayed(run_replication)(df_all, n, h, r, cfg, share) for n, h, r in jobs)
    return pd.DataFrame(rows)


def summarise(reps: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Per cell: mean true gain with a t-interval across replications, and detection rates."""
    rows = []
    for (h, n), g in reps.groupby(["heterogeneity", "n"]):
        row: dict[str, Any] = {"heterogeneity": h, "n": n, "replications": len(g),
                               "corr_tau_womens": g["corr_tau_womens"].mean()}
        tq = stats.t.ppf(1 - alpha / 2, len(g) - 1) if len(g) > 1 else np.nan
        for base in ("random", "best_arm"):
            v = g[f"true_gain_vs_{base}"]
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            lo, hi = v.mean() - tq * se, v.mean() + tq * se
            row.update({
                f"true_gain_vs_{base}": v.mean(), f"ci_low_vs_{base}": lo, f"ci_high_vs_{base}": hi,
                f"excludes_0_vs_{base}": bool(lo > 0 or hi < 0),
                f"oracle_gain_vs_{base}": g[f"oracle_gain_vs_{base}"].mean(),
                f"mean_est_gain_vs_{base}": g[f"est_gain_vs_{base}"].mean(),
                f"detection_rate_vs_{base}": g[f"detected_vs_{base}"].mean(),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("gain", [SLOTS[1], "#efeeea", SLOTS[0]])


def plot_heatmap(summary: pd.DataFrame, hillstrom_n: int = 64_000) -> plt.Figure:
    """Mean true gain per customer over h x n, vs random targeting and vs the best single arm."""
    apply_style()
    hs = sorted(summary.heterogeneity.unique(), reverse=True)
    ns = sorted(summary.n.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    titles = {"random": "Targeted - random targeting", "best_arm": "Targeted - best single arm"}
    for ax, base in zip(axes, ("random", "best_arm")):
        piv = summary.pivot(index="heterogeneity", columns="n", values=f"true_gain_vs_{base}").loc[hs, ns]
        lim = float(np.nanmax(np.abs(piv.to_numpy()))) or 1.0
        ax.imshow(piv.to_numpy(), cmap=_diverging_cmap(), vmin=-lim, vmax=lim, aspect="auto")
        for i, h in enumerate(hs):
            for j, n in enumerate(ns):
                r = summary[(summary.heterogeneity == h) & (summary.n == n)].iloc[0]
                sig = r[f"excludes_0_vs_{base}"]
                ax.text(j, i - 0.12, f"{r[f'true_gain_vs_{base}']:+.2f}", ha="center", va="center", fontsize=9,
                        color=TEXT, fontweight="bold" if sig else "normal")
                ax.text(j, i + 0.22, f"detected {r[f'detection_rate_vs_{base}']:.0%}", ha="center", va="center",
                        fontsize=7, color=TEXT)
                if sig:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=TEXT, lw=1.6))
        ax.set_xticks(range(len(ns)), [f"{n // 1000}k" + (" (Hillstrom)" if n == hillstrom_n else "") for n in ns])
        ax.set_yticks(range(len(hs)), [f"{h:g}" for h in hs])
        ax.set_xlabel("Customers in the experiment (n)")
        ax.set_ylabel("Effect heterogeneity h (0 = uniform, 1 = all in segment)")
        ax.set_title(titles[base], loc="left")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("True gain from model-based targeting ($ of spend per customer, budget 30%)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.04, "Cell value: mean over replications of the known true gain. Bold + outlined: 95% interval "
             "across replications excludes 0.\n'detected': share of single experiments whose doubly robust 95% CI "
             "for the gain is above 0. Blue = targeting better, orange = worse.", color=TEXT_2, fontsize=8)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def segment_share(df_all: pd.DataFrame, rule: str) -> float:
    from .positive_control import segment_mask

    return float(segment_mask(df_all, rule).mean())


def run() -> pd.DataFrame:
    from .data import load_frame

    cfg = load_config()
    ensure_output_dirs()
    df_all = load_frame()
    share = segment_share(df_all, cfg["positive_control"]["segment"])
    reps = run_grid(cfg, df_all, share)
    summary = summarise(reps, cfg["alpha"]).assign(segment_share=share)
    reps.to_csv(TABLES_DIR / "sweep_replications.csv", index=False)
    summary.to_csv(TABLES_DIR / "sweep_summary.csv", index=False)
    save(plot_heatmap(summary), FIGURES_DIR / "sweep_heatmap.png")
    return summary

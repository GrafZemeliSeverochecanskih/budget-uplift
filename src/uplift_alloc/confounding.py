"""Deliberately confounded version of the RCT, and recovery of the ATE by adjustment.

Construction: keep every control customer; keep a treated customer with probability
sigmoid(a + b * z), z = standardised log(1 + history). In the kept sample, treatment is more
likely for high-history customers, who also visit/spend more, so the naive difference in means
is biased upwards. The RCT ATE on the full data is the benchmark.

Assumptions for recovery: SUTVA; no unmeasured confounding (selection depends only on observed
``history``, true by construction); overlap (keep probability is bounded away from 0 for all z
in the data). Propensities are estimated here, unlike everywhere else in the repo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from .config import ARM_NAMES, FIGURES_DIR, TABLES_DIR, ensure_output_dirs, load_config
from .plotting import SLOTS, SURFACE, TEXT, TEXT_2, apply_style, plt, save

OUTCOMES = ["visit", "spend"]


def selection_z(history: np.ndarray) -> np.ndarray:
    lh = np.log1p(history)
    return (lh - lh.mean()) / lh.std()


def make_observational(t: np.ndarray, history: np.ndarray, intercept: float, slope: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Boolean keep-mask: all controls, treated with probability rising in history."""
    p_keep = 1 / (1 + np.exp(-(intercept + slope * selection_z(history))))
    return (t == 0) | (rng.random(len(t)) < p_keep)


def design(X: pd.DataFrame) -> np.ndarray:
    Z = X.drop(columns=["zip_code_Rural", "channel_Multichannel"]).copy()
    Z["log_history"] = np.log1p(Z["history"])
    return Z.to_numpy(dtype=float)


def propensity(Z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Estimated P(T = a | X): multinomial logistic regression on standardised covariates."""
    Zs = (Z - Z.mean(0)) / Z.std(0)
    model = LogisticRegression(max_iter=2000).fit(Zs, t)
    return np.clip(model.predict_proba(Zs), 1e-3, 1)


def outcome_means(Z: np.ndarray, t: np.ndarray, y: np.ndarray, n_arms: int = 3) -> np.ndarray:
    """mu-hat(x, a) by per-arm OLS (a deliberately simple, possibly misspecified model)."""
    return np.column_stack([LinearRegression().fit(Z[t == a], y[t == a]).predict(Z) for a in range(n_arms)])


def estimates(Z: np.ndarray, t: np.ndarray, y: np.ndarray, arm: int) -> dict[str, float]:
    """Naive, IPW (Hajek, estimated propensity) and AIPW/DR ATE of ``arm`` vs control."""
    e = propensity(Z, t)
    mu = outcome_means(Z, t, y)
    d1, d0 = (t == arm), (t == 0)
    w1, w0 = d1 / e[:, arm], d0 / e[:, 0]
    ipw = np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0)
    aipw = np.mean(mu[:, arm] - mu[:, 0] + w1 * (y - mu[:, arm]) - w0 * (y - mu[:, 0]))
    return {"naive": y[d1].mean() - y[d0].mean(), "ipw": float(ipw), "dr": float(aipw)}


def bootstrap(Z: np.ndarray, t: np.ndarray, y: np.ndarray, arm: int, reps: int, seed: int,
              alpha: float) -> pd.DataFrame:
    """Point estimates and percentile CIs; propensity and outcome models refit in every resample."""
    rng = np.random.default_rng(seed)
    point = estimates(Z, t, y, arm)
    draws = {k: np.empty(reps) for k in point}
    n = len(y)
    for r in range(reps):
        idx = rng.integers(0, n, n)
        for k, v in estimates(Z[idx], t[idx], y[idx], arm).items():
            draws[k][r] = v
    return pd.DataFrame([{"method": k, "estimate": v, "ci_low": np.quantile(draws[k], alpha / 2),
                          "ci_high": np.quantile(draws[k], 1 - alpha / 2)} for k, v in point.items()])


def rct_benchmark(t: np.ndarray, y: np.ndarray, arm: int, alpha: float) -> dict[str, float]:
    from .rct import diff_in_means

    m = (t == arm) | (t == 0)
    est = diff_in_means(y[m], (t[m] == arm).astype(float), alpha)
    return {"method": "rct_benchmark", "estimate": est.estimate, "ci_low": est.ci_low, "ci_high": est.ci_high}


# DAG ---------------------------------------------------------------------------
def plot_dag() -> plt.Figure:
    """Assumed DAG of the confounded sample: history -> T, history -> Y, other X -> Y, T -> Y."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.set_axis_off()
    nodes = {
        "H": (0.15, 0.78, "history\n(past-year $)"),
        "X": (0.15, 0.22, "other covariates\n(recency, channel, ...)"),
        "T": (0.55, 0.78, "e-mail arm T\n(kept treated ~ history)"),
        "Y": (0.85, 0.40, "outcome Y\n(visit, spend)"),
    }
    # (start, end) points on the box borders, axes fraction
    edges = [((0.27, 0.78), (0.40, 0.78), SLOTS[1]), ((0.27, 0.72), (0.74, 0.44), SLOTS[1]),
             ((0.30, 0.25), (0.74, 0.36), TEXT_2), ((0.62, 0.68), (0.76, 0.50), SLOTS[0])]
    for start, end, color in edges:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2, shrinkA=0, shrinkB=0, mutation_scale=14))
    for key, (x, y, label) in nodes.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=9, color=TEXT, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=SLOTS[0] if key in "TY" else TEXT_2, lw=1.4))
    ax.set_title("Assumed DAG for the confounded sample", loc="left")
    fig.text(0.01, 0.0, "Orange: the back-door path T <- history -> Y that biases the naive estimate. "
             "Adjusting for history blocks it.", color=TEXT_2, fontsize=8)
    return fig


def plot_estimates(table: pd.DataFrame) -> plt.Figure:
    apply_style()
    labels = {"rct_benchmark": "RCT benchmark (full data)", "naive": "Naive diff. in means",
              "ipw": "IPW (estimated propensity)", "dr": "Doubly robust (AIPW)"}
    colors = {"rct_benchmark": TEXT_2, "naive": SLOTS[7], "ipw": SLOTS[2], "dr": SLOTS[0]}
    groups = list(dict.fromkeys(zip(table.outcome, table.arm)))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.2))
    for ax, (outcome, arm) in zip(np.atleast_1d(axes), groups):
        g = table[(table.outcome == outcome) & (table.arm == arm)].set_index("method").loc[list(labels)]
        ys = np.arange(len(g))[::-1]
        for yv, (m, r) in zip(ys, g.iterrows()):
            ax.plot([r.ci_low, r.ci_high], [yv, yv], color=colors[m], lw=2)
            ax.plot(r.estimate, yv, "o", color=colors[m], markersize=7, markeredgecolor="white")
        ax.axvline(g.loc["rct_benchmark", "estimate"], color=TEXT_2, lw=0.8, ls="--")
        ax.set_yticks(ys, [labels[m] for m in g.index] if ax is np.atleast_1d(axes)[0] else [])
        ax.set_title(f"{outcome} - {arm.replace(' E-Mail', '')}", loc="left", fontsize=9)
    fig.suptitle("Confounded sample: naive vs adjusted ATE (95% CI)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def run() -> pd.DataFrame:
    from .data import load_hillstrom

    cfg = load_config()
    ensure_output_dirs()
    c = cfg["confounding"]
    X, t, Y = load_hillstrom()
    keep = make_observational(t, X["history"].to_numpy(), c["keep_intercept"], c["keep_slope"],
                              np.random.default_rng(cfg["seed"]))
    Z = design(X)
    rows = []
    for outcome in OUTCOMES:
        y = Y[outcome].to_numpy()
        for arm in (1, 2):
            bench = rct_benchmark(t, y, arm, cfg["alpha"])
            tab = bootstrap(Z[keep], t[keep], y[keep], arm, cfg["bootstrap_reps"], cfg["seed"] + arm, cfg["alpha"])
            tab = pd.concat([pd.DataFrame([bench]), tab], ignore_index=True)
            tab["bias_vs_rct"] = tab["estimate"] - bench["estimate"]
            tab.insert(0, "arm", ARM_NAMES[arm])
            tab.insert(0, "outcome", outcome)
            rows.append(tab)
    table = pd.concat(rows, ignore_index=True)
    table["n_kept"] = int(keep.sum())
    table["kept_share_treated"] = float(keep[t > 0].mean())
    table.to_csv(TABLES_DIR / "confounding.csv", index=False)
    save(plot_dag(), FIGURES_DIR / "dag.png")
    save(plot_estimates(table), FIGURES_DIR / "confounding_estimates.png")
    return table

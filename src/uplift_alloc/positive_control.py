"""Semi-synthetic positive control: real covariates and arm assignment, simulated outcomes with a known effect.

The real Hillstrom data shows no learnable targeting signal. This module asks whether the
unchanged pipeline would have found one if it existed. Covariates, the randomised arm of every
customer and the train/test split are the real ones; only ``visit`` and ``spend`` are replaced by
draws from the generator specified in ``config.yaml`` (``positive_control``, frozen before any
run), whose conditional effects tau(x, a) are known exactly.

Assumptions of the generator (by construction, so they hold): SUTVA; randomisation (the real
assignment is independent of the simulated potential outcomes given X); additive effects on the
purchase and visit-draw probabilities; purchase amount independent of X and arm.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import ARM_NAMES, FIGURES_DIR, TABLES_DIR, ensure_output_dirs, load_config
from .plotting import SLOTS, TEXT_2, apply_style, plt, save

ARM_KEYS = {1: "mens", 2: "womens"}
OUTCOMES = ["spend", "visit"]


def segment_mask(df: pd.DataFrame, rule: str) -> np.ndarray:
    return df.eval(rule).to_numpy(dtype=bool)


def baseline_rates(history: np.ndarray, base: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """No-e-mail visit-draw and purchase probabilities, rising with log history (prognostic only)."""
    lh = np.log1p(history)
    scale = np.exp(base["history_slope"] * (lh - lh.mean()) / lh.std())
    return base["visit_rate"] * scale, base["purchase_rate"] * scale


def effect_increments(df: pd.DataFrame, pc: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(n, 3) additive increments to visit-draw and purchase probabilities per arm (column 0 = control).

    An arm with ``where: segment`` gets ``visit``/``purchase`` inside the segment and the optional
    ``visit_outside``/``purchase_outside`` (default 0) elsewhere.
    """
    n = len(df)
    seg = segment_mask(df, pc["segment"])
    dv, dc = np.zeros((n, 3)), np.zeros((n, 3))
    for a, key in ARM_KEYS.items():
        eff = pc["effects"][key]
        inside = np.ones(n, bool) if eff["where"] == "all" else seg
        dv[:, a] = np.where(inside, eff["visit"], eff.get("visit_outside", 0.0))
        dc[:, a] = np.where(inside, eff["purchase"], eff.get("purchase_outside", 0.0))
    return dv, dc


def true_tau(df: pd.DataFrame, pc: dict[str, Any]) -> dict[str, np.ndarray]:
    """Exact tau(x, a) for a = 1, 2 (columns), for spend and visit.

    spend: E[purchase * amount] = p_c * mean amount, so tau = d_purchase * mean.
    visit = visit draw OR purchase, so P(visit) = 1 - (1 - p_v)(1 - p_c).
    """
    v0, c0 = baseline_rates(df["history"].to_numpy(), pc["baseline"])
    dv, dc = effect_increments(df, pc)
    p_visit = 1 - (1 - (v0[:, None] + dv)) * (1 - (c0[:, None] + dc))
    return {"spend": dc[:, 1:] * pc["amount"]["mean"], "visit": p_visit[:, 1:] - p_visit[:, [0]]}


def simulate(df: pd.DataFrame, t: np.ndarray, pc: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Draw visit and spend for every customer under the arm they actually received."""
    v0, c0 = baseline_rates(df["history"].to_numpy(), pc["baseline"])
    dv, dc = effect_increments(df, pc)
    rows = np.arange(len(df))
    p_v, p_c = v0 + dv[rows, t], c0 + dc[rows, t]
    assert (p_v <= 1).all() and (p_c <= 1).all(), "simulated probabilities exceed 1"
    purchase = rng.random(len(df)) < p_c
    visit = (rng.random(len(df)) < p_v) | purchase
    sigma = pc["amount"]["sigma"]
    amount = rng.lognormal(np.log(pc["amount"]["mean"]) - sigma**2 / 2, sigma, len(df))
    return pd.DataFrame({"visit": visit.astype(float), "spend": purchase * amount,
                         "conversion": purchase.astype(float)})


def tau_recovery(preds: dict[str, np.ndarray], tau: np.ndarray, reps: int, seed: int, alpha: float) -> pd.DataFrame:
    """Pearson correlation of tau-hat with the true tau per arm, with a bootstrap CI over test rows."""
    rng = np.random.default_rng(seed)
    n = len(tau)
    idx = [rng.integers(0, n, n) for _ in range(reps)]
    rows = []
    for name, p in preds.items():
        for a in (1, 2):
            x, y = p[:, a - 1], tau[:, a - 1]
            varies = np.ptp(y) > 0  # correlation is undefined for a constant true tau (Mens arm)
            r = np.corrcoef(x, y)[0, 1] if varies else np.nan
            draws = [np.corrcoef(x[i], y[i])[0, 1] for i in idx] if varies else [np.nan]
            rows.append({"learner": name, "arm": ARM_NAMES[a], "corr_true_tau": r,
                         "corr_ci_low": np.nanquantile(draws, alpha / 2), "corr_ci_high": np.nanquantile(draws, 1 - alpha / 2),
                         "mean_tau_hat": x.mean(), "mean_true_tau": y.mean(),
                         "rmse": float(np.sqrt(np.mean((x - y) ** 2)))})
    return pd.DataFrame(rows)


def plot_tau_recovery(preds: dict[str, np.ndarray], tau: np.ndarray, seg: np.ndarray, learner: str) -> plt.Figure:
    """tau-hat of the selected learner by true segment membership, per arm (spend)."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for a, ax in zip((1, 2), axes):
        p = preds[learner][:, a - 1]
        groups = [p[~seg], p[seg]]
        parts = ax.violinplot(groups, showextrema=False, widths=0.8)
        for body, color in zip(parts["bodies"], (SLOTS[1], SLOTS[0])):
            body.set_facecolor(color)
            body.set_alpha(0.35)
        for k, (g, s_) in enumerate(zip(groups, (~seg, seg)), 1):
            truth = tau[s_, a - 1].mean()
            ax.plot([k - 0.4, k + 0.4], [truth, truth], color=TEXT_2, lw=2, ls="--",
                    label="true τ" if k == 1 else "_nolegend_")
            ax.plot(k, np.median(g), "o", color="white", markeredgecolor=TEXT_2, markersize=6,
                    label="median τ̂" if k == 1 else "_nolegend_")
        ax.set_xticks([1, 2], ["outside segment", "in segment"])
        ax.set_title(f"{ARM_NAMES[a]}: τ̂ for spend ({learner}-learner)", loc="left", fontsize=10)
    axes[0].set_ylabel("Predicted incremental spend ($)")
    axes[0].legend(loc="upper left")
    fig.text(0.01, -0.03, "Segment: womens == 1 and recency <= 6. Womens e-mail truly helps only inside it.",
             color=TEXT_2, fontsize=8)
    fig.tight_layout()
    return fig


def run() -> dict[str, pd.DataFrame]:
    from . import cate, ope
    from .data import encode_arm, encode_features, load_frame, load_split, subset

    cfg = load_config()
    ensure_output_dirs()
    pc, seed, alpha, reps = cfg["positive_control"], cfg["seed"], cfg["alpha"], cfg["bootstrap_reps"]
    df = load_frame()
    X, t = encode_features(df), encode_arm(df["segment"])
    Y = simulate(df, t, pc, np.random.default_rng(seed + pc["seed_offset"]))
    tau = true_tau(df, pc)
    seg = segment_mask(df, pc["segment"])
    split = load_split()
    tr, te = np.array(split["train"]), np.array(split["test"])
    train, test = subset((X, t, Y), tr), subset((X, t, Y), te)
    tau_te = {k: v[te] for k, v in tau.items()}

    choice = cate.run_outcomes(train, test, OUTCOMES, tag="pc_", oracle=tau_te)
    recovery = []
    for outcome in OUTCOMES:
        preds = cate.load_test_predictions(outcome, "pc_")
        recovery.append(tau_recovery(preds, tau_te[outcome], reps, seed, alpha)
                        .assign(outcome=outcome, selected=lambda d: d.learner == choice["selected"][outcome]))
    recovery_table = pd.concat(recovery, ignore_index=True)
    recovery_table.to_csv(TABLES_DIR / "pc_tau_recovery.csv", index=False)
    save(plot_tau_recovery(cate.load_test_predictions("spend", "pc_"), tau_te["spend"], seg[te],
                           choice["selected"]["spend"]), FIGURES_DIR / "pc_tau_recovery.png")
    policy = ope.run_outcomes(train, test, OUTCOMES, tag="pc_", tau_true=tau_te)

    sim = pd.DataFrame({
        "quantity": ["segment_share", "test_visit_rate", "test_conversion_rate", "test_mean_spend",
                     "true_ate_spend_mens", "true_ate_spend_womens", "true_ate_visit_mens", "true_ate_visit_womens"],
        "value": [seg.mean(), Y["visit"].iloc[te].mean(), Y["conversion"].iloc[te].mean(), Y["spend"].iloc[te].mean(),
                  *tau["spend"].mean(0), *tau["visit"].mean(0)],
    })
    sim.to_csv(TABLES_DIR / "pc_simulation_summary.csv", index=False)
    return {"recovery": recovery_table, "policy": policy, "simulation": sim}

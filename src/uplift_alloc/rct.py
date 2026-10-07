"""Analysis of the Hillstrom RCT: balance, ATEs, regression adjustment, power.

Assumptions: SUTVA (one customer's e-mail does not affect another's outcome); arms randomised
independently of covariates, so the propensity is 1/3 and difference in means is unbiased.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import ARM_NAMES, FIGURES_DIR, TABLES_DIR, ensure_output_dirs, load_config
from .plotting import ARM_COLORS, TEXT_2, apply_style, plt, save


@dataclass(frozen=True)
class Estimate:
    estimate: float
    se: float
    ci_low: float
    ci_high: float

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low


def _normal_ci(est: float, se: float, alpha: float) -> Estimate:
    z = stats.norm.ppf(1 - alpha / 2)
    return Estimate(est, se, est - z * se, est + z * se)


# balance ---------------------------------------------------------------------
def smd(x_treated: np.ndarray, x_control: np.ndarray) -> float:
    """Standardised mean difference with the pooled (average-variance) denominator."""
    pooled = np.sqrt((np.var(x_treated, ddof=1) + np.var(x_control, ddof=1)) / 2)
    return 0.0 if pooled == 0 else float((x_treated.mean() - x_control.mean()) / pooled)


def balance_table(X: pd.DataFrame, t: np.ndarray, flag: float = 0.1) -> pd.DataFrame:
    """SMD of every (encoded) covariate, each treated arm vs control; flag |SMD| > ``flag``."""
    rows = [
        {"covariate": col, "arm": ARM_NAMES[a],
         "smd": smd(X.loc[t == a, col].to_numpy(), X.loc[t == 0, col].to_numpy())}
        for a in (1, 2) for col in X.columns
    ]
    df = pd.DataFrame(rows)
    df["flag"] = df["smd"].abs() > flag
    return df


# ATEs ------------------------------------------------------------------------
def diff_in_means(y: np.ndarray, d: np.ndarray, alpha: float = 0.05) -> Estimate:
    """Unadjusted ATE of d=1 vs d=0 with a Welch (unequal-variance) normal CI."""
    y1, y0 = y[d == 1], y[d == 0]
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return _normal_ci(float(y1.mean() - y0.mean()), float(se), alpha)


def regression_adjusted(y: np.ndarray, d: np.ndarray, Z: np.ndarray, alpha: float = 0.05) -> Estimate:
    """Lin (2013) fully-interacted regression adjustment, HC2 standard errors.

    y ~ d + Zc + d:Zc with Zc = Z centred at the pooled mean, so the coefficient on d is the ATE.
    With Z = pre-period ``history`` this is CUPED; extra covariates only add precision.
    Consistent under randomisation regardless of whether the linear model is correct.
    """
    Zc = Z - Z.mean(axis=0)
    design = np.column_stack([np.ones_like(y), d, Zc, d[:, None] * Zc])
    fit = sm.OLS(y, design).fit(cov_type="HC2")
    return _normal_ci(float(fit.params[1]), float(fit.bse[1]), alpha)


def adjustment_covariates(X: pd.DataFrame) -> np.ndarray:
    """Covariates for adjustment: drop one level of each one-hot block to avoid collinearity."""
    keep = [c for c in X.columns if c not in ("zip_code_Rural", "channel_Multichannel")]
    Z = X[keep].copy()
    Z["log_history"] = np.log1p(Z["history"])
    return Z.to_numpy(dtype=float)


def ate_table(X: pd.DataFrame, t: np.ndarray, Y: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """ATE per arm vs control and outcome: difference in means and regression-adjusted."""
    Z_all = adjustment_covariates(X)
    rows = []
    for a in (1, 2):
        mask = (t == a) | (t == 0)
        d = (t[mask] == a).astype(float)
        for outcome in Y.columns:
            y = Y[outcome].to_numpy(dtype=float)[mask]
            dim = diff_in_means(y, d, alpha)
            adj = regression_adjusted(y, d, Z_all[mask], alpha)
            for method, est in (("diff_in_means", dim), ("regression_adjusted", adj)):
                rows.append({"arm": ARM_NAMES[a], "outcome": outcome, "method": method,
                             **asdict(est), "ci_width": est.width,
                             "ci_width_reduction_pct": 100 * (1 - est.width / dim.width)})
    return pd.DataFrame(rows)


# power -----------------------------------------------------------------------
def n_per_arm(effect: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Per-arm n for a two-sided two-sample z-test with equal arms and common SD."""
    if effect == 0:
        return float("inf")
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(2 * (z * sd / effect) ** 2)


def mde(n: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect with n customers per arm."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd * np.sqrt(2 / n))


def power_table(t: np.ndarray, y: np.ndarray, ates: pd.DataFrame, alpha: float, power: float) -> pd.DataFrame:
    """Sample size to detect the observed spend effect; the interval maps the ATE's CI ends.

    A CI end at or below zero means no finite sample size is guaranteed (reported as inf).
    """
    rows = []
    for a in (1, 2):
        sd = float(np.sqrt((y[t == a].var(ddof=1) + y[t == 0].var(ddof=1)) / 2))
        r = ates.query("arm == @ARM_NAMES[@a] and outcome == 'spend' and method == 'diff_in_means'").iloc[0]
        n_obs = int(min((t == a).sum(), (t == 0).sum()))
        n_bounds = sorted(n_per_arm(e, sd, alpha, power) if e > 0 else float("inf")
                          for e in (r.ci_low, r.ci_high))
        rows.append({
            "arm": ARM_NAMES[a], "observed_effect": r.estimate, "effect_ci_low": r.ci_low,
            "effect_ci_high": r.ci_high, "sd_spend": sd,
            "n_per_arm_required": n_per_arm(r.estimate, sd, alpha, power),
            "n_required_ci_low": n_bounds[0], "n_required_ci_high": n_bounds[1],
            "n_per_arm_observed": n_obs, "mde_at_observed_n": mde(n_obs, sd, alpha, power),
        })
    return pd.DataFrame(rows)


# figure ----------------------------------------------------------------------
def plot_balance(bal: pd.DataFrame, flag: float) -> plt.Figure:
    """Love plot: SMD per covariate for each arm, with +/-flag guides."""
    apply_style()
    covs = list(dict.fromkeys(bal["covariate"]))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for k, a in enumerate((1, 2)):
        sub = bal[bal["arm"] == ARM_NAMES[a]].set_index("covariate").loc[covs]
        ax.scatter(sub["smd"], np.arange(len(covs)) + (k - 0.5) * 0.25, s=36,
                   color=ARM_COLORS[a], edgecolor="white", linewidth=1, label=f"{ARM_NAMES[a]} vs control", zorder=3)
    for x in (-flag, flag):
        ax.axvline(x, color=TEXT_2, linestyle="--", linewidth=1)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(np.arange(len(covs)), covs)
    ax.invert_yaxis()
    lim = max(flag * 1.5, bal["smd"].abs().max() * 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Standardised mean difference")
    ax.set_title(f"Covariate balance: max |SMD| = {bal['smd'].abs().max():.3f} (flag at {flag})", loc="left")
    ax.legend(loc="upper left")
    return fig


def run() -> dict[str, pd.DataFrame]:
    """Full RCT analysis on all 64,000 rows (nothing here is tuned, so the test split is not at risk)."""
    from .data import load_hillstrom

    cfg = load_config()
    ensure_output_dirs()
    X, t, Y = load_hillstrom()
    alpha, flag = cfg["alpha"], cfg["rct"]["smd_flag"]
    bal = balance_table(X, t, flag)
    ates = ate_table(X, t, Y, alpha)
    power = power_table(t, Y["spend"].to_numpy(), ates, alpha, cfg["rct"]["power"])
    ates.to_csv(TABLES_DIR / "rct_summary.csv", index=False)
    bal.to_csv(TABLES_DIR / "balance.csv", index=False)
    power.to_csv(TABLES_DIR / "rct_power.csv", index=False)
    save(plot_balance(bal, flag), FIGURES_DIR / "balance.png")
    return {"balance": bal, "ates": ates, "power": power}

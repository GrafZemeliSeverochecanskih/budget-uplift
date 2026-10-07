"""Uplift evaluation: Qini and uplift (AUUC) curves for one treated arm vs control.

Assumption: the evaluation sample is randomised (here, the RCT test split), so within any score
quantile the treated/control comparison is unbiased.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cumulative(score: np.ndarray, y: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(-score, kind="stable")
    y, d = y[order], d[order]
    n_t = np.cumsum(d)
    n_c = np.cumsum(1 - d)
    y_t = np.cumsum(y * d)
    y_c = np.cumsum(y * (1 - d))
    return n_t, n_c, y_t, y_c


def qini_curve(score: np.ndarray, y: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fraction targeted and Qini value Q(k) = Y_t(k) - Y_c(k) * n_t(k)/n_c(k), divided by N.

    Q(1) / fraction-treated equals the ATE; y units are outcome per customer in the sample.
    """
    n_t, n_c, y_t, y_c = _cumulative(score, y, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, y_t - y_c * n_t / np.maximum(n_c, 1), y_t)
    n = len(y)
    return np.r_[0.0, np.arange(1, n + 1) / n], np.r_[0.0, q / n]


def uplift_curve(score: np.ndarray, y: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fraction targeted and U(k) = (mean_t(k) - mean_c(k)) * k / N (incremental outcome per customer)."""
    n_t, n_c, y_t, y_c = _cumulative(score, y, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.where((n_t > 0) & (n_c > 0), y_t / np.maximum(n_t, 1) - y_c / np.maximum(n_c, 1), 0.0)
    n = len(y)
    k = np.arange(1, n + 1)
    return np.r_[0.0, k / n], np.r_[0.0, diff * k / n]


def _area_above_random(x: np.ndarray, curve: np.ndarray) -> float:
    random_line = x * curve[-1]
    return float(np.trapezoid(curve - random_line, x))


def qini_coefficient(score: np.ndarray, y: np.ndarray, d: np.ndarray) -> float:
    """Area between the Qini curve and the random-targeting diagonal (0 = random, >0 = better)."""
    return _area_above_random(*qini_curve(score, y, d))


def auuc(score: np.ndarray, y: np.ndarray, d: np.ndarray) -> float:
    """Area under the uplift curve (random targeting gives ATE / 2)."""
    x, u = uplift_curve(score, y, d)
    return float(np.trapezoid(u, x))


def auuc_lift(score: np.ndarray, y: np.ndarray, d: np.ndarray) -> float:
    """AUUC minus the random-targeting area."""
    return _area_above_random(*uplift_curve(score, y, d))


def random_null(y: np.ndarray, d: np.ndarray, reps: int, seed: int) -> dict[str, np.ndarray]:
    """Metric values of ``reps`` random orderings of the same sample (the random-targeting null)."""
    rng = np.random.default_rng(seed)
    fns = {"qini": qini_coefficient, "auuc": auuc, "auuc_lift": auuc_lift}
    draws = {m: np.empty(reps) for m in fns}
    for r in range(reps):
        s = rng.random(len(y))
        for m, fn in fns.items():
            draws[m][r] = fn(s, y, d)
    return draws


def bootstrap_metrics(
    scores: dict[str, np.ndarray], y: np.ndarray, d: np.ndarray, reps: int, seed: int, alpha: float = 0.05
) -> pd.DataFrame:
    """Point estimate + percentile bootstrap CI for Qini coefficient, AUUC and AUUC lift.

    Every scorer is evaluated on the same resamples so CIs are comparable; predictions stay fixed
    (the interval covers evaluation-sample noise, not model-refit noise).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    fns = {"qini": qini_coefficient, "auuc": auuc, "auuc_lift": auuc_lift}
    draws = {name: {m: np.empty(reps) for m in fns} for name in scores}
    for r in range(reps):
        idx = rng.integers(0, n, n)
        for name, s in scores.items():
            for m, fn in fns.items():
                draws[name][m][r] = fn(s[idx], y[idx], d[idx])
    rows = []
    for name, s in scores.items():
        row: dict[str, float | str] = {"learner": name}
        for m, fn in fns.items():
            lo, hi = np.quantile(draws[name][m], [alpha / 2, 1 - alpha / 2])
            row.update({m: fn(s, y, d), f"{m}_ci_low": lo, f"{m}_ci_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)

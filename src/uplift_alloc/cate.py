"""Multi-arm CATE (uplift) learners with LightGBM base models.

Every learner exposes ``fit(X, t, y)`` and ``effect(X) -> ndarray (n, n_arms - 1)`` where column
``a - 1`` is the estimated tau_a(x) = E[Y(a) - Y(0) | X = x].

Assumptions: SUTVA; unconfoundedness and overlap hold by design (RCT); the propensity of every
arm is known and equal to 1/n_arms, and is plugged in rather than estimated.
"""
from __future__ import annotations

import itertools
import json
import time
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

from .config import ARM_NAMES, FIGURES_DIR, CACHE_DIR, TABLES_DIR, ensure_output_dirs, load_config
from .metrics import bootstrap_metrics, qini_coefficient, qini_curve, random_null
from .plotting import ARM_COLORS, NEUTRAL, SLOTS, TEXT, TEXT_2, apply_style, plt, save

N_ARMS = 3


class CateLearner(Protocol):
    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "CateLearner": ...
    def effect(self, X: np.ndarray) -> np.ndarray: ...


def lgbm(params: dict[str, Any], seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**params, random_state=seed, n_jobs=4)


def _onehot(t: np.ndarray, n_arms: int) -> np.ndarray:
    return np.eye(n_arms)[t]


# meta-learners ---------------------------------------------------------------
class SLearner:
    """One model mu(x, a) on [X, onehot(a)]; tau_a(x) = mu(x, a) - mu(x, 0)."""

    def __init__(self, params: dict[str, Any], seed: int = 0, n_arms: int = N_ARMS):
        self.params, self.seed, self.n_arms = params, seed, n_arms

    def fit(self, X, t, y):
        self.model = lgbm(self.params, self.seed).fit(np.column_stack([X, _onehot(t, self.n_arms)]), y)
        return self

    def mu(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([
            self.model.predict(np.column_stack([X, np.tile(np.eye(self.n_arms)[a], (n, 1))]))
            for a in range(self.n_arms)
        ])

    def effect(self, X):
        m = self.mu(X)
        return m[:, 1:] - m[:, [0]]


class TLearner:
    """One outcome model per arm; tau_a(x) = mu_a(x) - mu_0(x)."""

    def __init__(self, params: dict[str, Any], seed: int = 0, n_arms: int = N_ARMS):
        self.params, self.seed, self.n_arms = params, seed, n_arms

    def fit(self, X, t, y):
        self.models = [lgbm(self.params, self.seed).fit(X[t == a], y[t == a]) for a in range(self.n_arms)]
        return self

    def mu(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models])

    def effect(self, X):
        m = self.mu(X)
        return m[:, 1:] - m[:, [0]]


class XLearner:
    """Künzel et al. X-learner, one pairwise (arm a vs control) problem per arm.

    Imputed effects D1 = Y - mu_0(X) on arm-a units and D0 = mu_a(X) - Y on controls are each
    regressed on X and blended with the known within-pair propensity g = P(a | a or 0) = 1/2.
    """

    def __init__(self, params: dict[str, Any], seed: int = 0, n_arms: int = N_ARMS, propensity: float = 0.5):
        self.params, self.seed, self.n_arms, self.g = params, seed, n_arms, propensity

    def fit(self, X, t, y):
        self.t_learner = TLearner(self.params, self.seed, self.n_arms).fit(X, t, y)
        mu = self.t_learner.mu(X)
        c = t == 0
        self.stage2 = []
        for a in range(1, self.n_arms):
            tr = t == a
            tau1 = lgbm(self.params, self.seed).fit(X[tr], y[tr] - mu[tr, 0])
            tau0 = lgbm(self.params, self.seed).fit(X[c], mu[c, a] - y[c])
            self.stage2.append((tau0, tau1))
        return self

    def effect(self, X):
        return np.column_stack([self.g * t0.predict(X) + (1 - self.g) * t1.predict(X) for t0, t1 in self.stage2])


# EconML learners -------------------------------------------------------------
class KnownPropensity(ClassifierMixin, BaseEstimator):
    """Classifier that ignores X and returns the design propensity 1/K for each of K arms."""

    def fit(self, X, y, **kwargs):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        k = len(self.classes_)
        return np.full((len(X), k), 1.0 / k)

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


class DRLearnerMulti:
    """EconML DR-learner: AIPW pseudo-outcomes with known propensity, LightGBM final stage."""

    def __init__(self, params: dict[str, Any], seed: int = 0, n_arms: int = N_ARMS):
        from econml.dr import DRLearner

        self.n_arms = n_arms
        self.est = DRLearner(
            model_propensity=KnownPropensity(), model_regression=lgbm(params, seed),
            model_final=lgbm(params, seed), cv=3, random_state=seed,
        )

    def fit(self, X, t, y):
        self.est.fit(y, t, X=X)
        return self

    def effect(self, X):
        return np.column_stack([self.est.effect(X, T0=0, T1=a) for a in range(1, self.n_arms)])


class CausalForestMulti:
    """EconML CausalForestDML with discrete treatment, known propensity, LightGBM outcome nuisance."""

    def __init__(self, params: dict[str, Any], seed: int = 0, n_arms: int = N_ARMS,
                 n_estimators: int = 400, min_samples_leaf: int = 50):
        from econml.dml import CausalForestDML

        self.n_arms = n_arms
        self.est = CausalForestDML(
            model_y=lgbm(params, seed), model_t=KnownPropensity(), discrete_treatment=True,
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, cv=3,
            random_state=seed, inference=False, n_jobs=-1,
        )

    def fit(self, X, t, y):
        self.est.fit(y, t, X=X)
        return self

    def effect(self, X):
        return np.column_stack([self.est.effect(X, T0=0, T1=a) for a in range(1, self.n_arms)])


LEARNERS = ["S", "T", "X", "DR", "CausalForest"]


def make_learner(name: str, params: dict[str, Any], seed: int, cfg: dict[str, Any] | None = None) -> CateLearner:
    cfg = cfg or load_config()
    if name == "S":
        return SLearner(params, seed)
    if name == "T":
        return TLearner(params, seed)
    if name == "X":
        return XLearner(params, seed)
    if name == "DR":
        return DRLearnerMulti(params, seed)
    if name == "CausalForest":
        return CausalForestMulti(params, seed, **cfg["cate"]["causal_forest"])
    raise ValueError(name)


# hyperparameters ---------------------------------------------------------------
def tune_lgbm(X: np.ndarray, t: np.ndarray, y: np.ndarray, grid: dict[str, list], fixed: dict[str, Any],
              folds: int, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Grid-search LightGBM for the outcome model mu(x, a) by K-fold CV MSE (train data only).

    The winning parameters are shared by every base model of every learner. Choosing nuisance
    models by outcome MSE is standard; it does not directly optimise CATE accuracy.
    """
    Xa = np.column_stack([X, _onehot(t, N_ARMS)])
    kf = KFold(folds, shuffle=True, random_state=seed)
    candidates = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    rows = []
    for cand in candidates:
        params = {**fixed, **cand}
        mse = [
            np.mean((lgbm(params, seed).fit(Xa[tr], y[tr]).predict(Xa[va]) - y[va]) ** 2)
            for tr, va in kf.split(Xa)
        ]
        rows.append({**cand, "cv_mse": float(np.mean(mse))})
    table = pd.DataFrame(rows)
    best = {**fixed, **candidates[int(table["cv_mse"].idxmin())]}
    return best, table.sort_values("cv_mse")


# evaluation --------------------------------------------------------------------
def pairwise_qini(tau: np.ndarray, t: np.ndarray, y: np.ndarray) -> list[float]:
    """Qini coefficient of tau[:, a-1] on the arm-a-vs-control subsample, per arm."""
    out = []
    for a in range(1, tau.shape[1] + 1):
        m = (t == a) | (t == 0)
        out.append(qini_coefficient(tau[m, a - 1], y[m], (t[m] == a).astype(float)))
    return out


def evaluate(preds: dict[str, np.ndarray], t: np.ndarray, y: np.ndarray, reps: int, seed: int,
             alpha: float) -> pd.DataFrame:
    """Qini/AUUC with bootstrap CIs per learner and arm, plus the random-targeting baseline.

    The "Random" row is the permutation null: mean and central 1-alpha band of the metric over
    ``reps`` random orderings of the test sample. ``p_vs_random`` is the one-sided share of random
    orderings whose Qini is at least the learner's.
    """
    tables = []
    for a in range(1, N_ARMS):
        m = (t == a) | (t == 0)
        d = (t[m] == a).astype(float)
        scores = {name: p[m, a - 1] for name, p in preds.items()}
        tab = bootstrap_metrics(scores, y[m], d, reps, seed + a, alpha)
        null = random_null(y[m], d, reps, seed + 10 + a)
        tab["p_vs_random"] = [(1 + np.sum(null["qini"] >= q)) / (1 + reps) for q in tab["qini"]]
        row: dict[str, Any] = {"learner": "Random"}
        for metric, draws in null.items():
            lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
            row.update({metric: draws.mean(), f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
        tab = pd.concat([tab, pd.DataFrame([row])], ignore_index=True)
        tab.insert(1, "arm", ARM_NAMES[a])
        tables.append(tab)
    return pd.concat(tables, ignore_index=True)


def plot_qini(preds: dict[str, np.ndarray], t: np.ndarray, y: np.ndarray, outcome: str) -> plt.Figure:
    apply_style()
    fig, axes = plt.subplots(1, N_ARMS - 1, figsize=(10, 4), sharey=False)
    colors = dict(zip(preds, SLOTS[2:] + SLOTS[:2]))  # arm colours (slots 1-2) are kept for arms
    for a, ax in zip(range(1, N_ARMS), axes):
        m = (t == a) | (t == 0)
        d = (t[m] == a).astype(float)
        rng = np.random.default_rng(a)
        null = np.array([qini_curve(rng.random(m.sum()), y[m], d)[1] for _ in range(200)])
        x = np.linspace(0, 1, null.shape[1])
        lo, hi = np.quantile(null, [0.025, 0.975], axis=0)
        ax.fill_between(x, lo, hi, color=NEUTRAL, alpha=0.18, linewidth=0, label="Random, 95% band")
        for name, p in preds.items():
            x, q = qini_curve(p[m, a - 1], y[m], d)
            oracle = name.startswith("Oracle")
            ax.plot(x, q, color=TEXT if oracle else colors[name], linewidth=1.6, label=name,
                    linestyle=":" if oracle else "-")
        ax.plot([0, 1], [0, q[-1]], color=NEUTRAL, linestyle="--", linewidth=1.2, label="Random (expected)")
        ax.set_title(f"{ARM_NAMES[a]} vs control", loc="left", color=ARM_COLORS[a])
        ax.set_xlabel("Fraction of customers targeted (by predicted uplift)")
        ax.set_ylabel(f"Incremental {outcome} per customer (Qini)")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Qini curves on the test split — outcome: {outcome}", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "Curve above the dashed diagonal = better than random targeting; grey band = 95% of random orderings.", color=TEXT_2, fontsize=8)
    fig.tight_layout()
    return fig


# pipeline ----------------------------------------------------------------------
def fit_predict(names: list[str], params: dict[str, Any], seed: int, X_fit, t_fit, y_fit, X_pred) -> tuple[dict, dict]:
    preds, times = {}, {}
    for name in names:
        start = time.perf_counter()
        model = make_learner(name, params, seed).fit(X_fit, t_fit, y_fit)
        preds[name] = model.effect(X_pred)
        times[name] = time.perf_counter() - start
    return preds, times


def select_learner(X, t, y, params, cfg) -> tuple[str, pd.DataFrame]:
    """Pick the downstream learner on a held-out slice of TRAIN (sum of per-arm Qini on this outcome)."""
    rng = np.random.default_rng(cfg["seed"])
    n = len(y)
    sel = np.zeros(n, bool)
    for a in range(N_ARMS):  # stratified by arm
        idx = np.flatnonzero(t == a)
        sel[rng.choice(idx, int(round(cfg["cate"]["selection_fraction"] * len(idx))), replace=False)] = True
    preds, _ = fit_predict(LEARNERS, params, cfg["seed"], X[~sel], t[~sel], y[~sel], X[sel])
    rows = [{"learner": k, **{f"qini_{ARM_NAMES[a]}": q for a, q in enumerate(pairwise_qini(p, t[sel], y[sel]), 1)}}
            for k, p in preds.items()]
    table = pd.DataFrame(rows)
    table["qini_sum"] = table.filter(like="qini_").sum(axis=1)
    return str(table.sort_values("qini_sum", ascending=False).iloc[0]["learner"]), table


def run_outcomes(train, test, outcomes: list[str], tag: str = "",
                 oracle: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Tune on train, choose the learner on a train slice, fit all learners on train, evaluate on test.

    ``tag`` prefixes every output file (e.g. ``pc_`` for the positive control). ``oracle`` maps an
    outcome to known true tau on test; it is scored like a learner but never selected.
    Test predictions are cached as ``{tag}cate_test_{outcome}.npz``.
    """
    cfg = load_config()
    ensure_output_dirs()
    seed, reps, alpha = cfg["seed"], cfg["bootstrap_reps"], cfg["alpha"]
    (Xtr, ttr, Ytr), (Xte, tte, Yte) = train, test
    Xtr_np, Xte_np = np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float)
    results: dict[str, Any] = {"params": {}, "selected": {}}
    comparison, selection_tables, tuning_tables = [], [], []
    for outcome in outcomes:
        ytr, yte = Ytr[outcome].to_numpy(), Yte[outcome].to_numpy()
        params, tuning = tune_lgbm(Xtr_np, ttr, ytr, cfg["cate"]["lgbm_grid"], cfg["cate"]["lgbm_fixed"],
                                   cfg["cate"]["cv_folds"], seed)
        tuning_tables.append(tuning.assign(outcome=outcome))
        selected, sel_table = select_learner(Xtr_np, ttr, ytr, params, cfg)
        selection_tables.append(sel_table.assign(outcome=outcome, selected=lambda d: d.learner == selected))
        preds, times = fit_predict(LEARNERS, params, seed, Xtr_np, ttr, ytr, Xte_np)
        np.savez(CACHE_DIR / f"{tag}cate_test_{outcome}.npz", **preds)
        scored = {**preds, **({"Oracle (true τ)": oracle[outcome]} if oracle else {})}
        table = evaluate(scored, tte, yte, reps, seed, alpha)
        table["fit_seconds"] = table["learner"].map(times)
        table.insert(0, "outcome", outcome)
        comparison.append(table)
        name = "qini_curves.png" if outcome == "spend" else f"qini_curves_{outcome}.png"
        save(plot_qini(scored, tte, yte, outcome), FIGURES_DIR / f"{tag}{name}")
        results["params"][outcome], results["selected"][outcome] = params, selected
    pd.concat(comparison).to_csv(TABLES_DIR / f"{tag}learner_comparison.csv", index=False)
    pd.concat(selection_tables).to_csv(TABLES_DIR / f"{tag}learner_selection_train.csv", index=False)
    pd.concat(tuning_tables).to_csv(TABLES_DIR / f"{tag}lgbm_tuning_cv.csv", index=False)
    (CACHE_DIR / f"{tag}cate_choice.json").write_text(json.dumps(results, indent=2, default=float))
    return results


def run() -> dict[str, Any]:
    """CATE learners on the real Hillstrom train/test split for every configured outcome."""
    from .data import train_test

    train, test = train_test()
    return run_outcomes(train, test, load_config()["cate"]["outcomes"])


def load_choice(tag: str = "") -> dict[str, Any]:
    """Tuned LightGBM params and the train-selected learner per outcome (written by ``run_outcomes``)."""
    path = CACHE_DIR / f"{tag}cate_choice.json"
    if not path.exists() and not tag:
        run()
    return json.loads(path.read_text())


def load_test_predictions(outcome: str, tag: str = "") -> dict[str, np.ndarray]:
    return dict(np.load(CACHE_DIR / f"{tag}cate_test_{outcome}.npz"))

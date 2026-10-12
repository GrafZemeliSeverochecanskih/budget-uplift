"""Text-embedding features: describe each customer in one sentence, embed, reduce, add to the learner.

The sentence is a deterministic rendering of the same tabular covariates, so the embedding can
only help if the language model's geometry encodes useful interactions/non-linearities; it adds
no new information. That makes this a test of representation, not of data.

PCA is fit on train rows only. LightGBM parameters are those tuned without embeddings (Sec. 2).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import ARM_NAMES, PROCESSED_DIR, TABLES_DIR, ensure_output_dirs, load_config
from .metrics import qini_coefficient


def describe(row: pd.Series) -> str:
    """One plain-English sentence per customer from the raw covariates."""
    bought = [c for c, flag in (("men's", row.mens), ("women's", row.womens)) if flag]
    bought_txt = " and ".join(bought) + " merchandise" if bought else "no men's or women's merchandise"
    months = int(row.recency)
    return (
        f"A {'new' if row.newbie else 'returning'} customer from a "
        f"{str(row.zip_code).replace('Surburban', 'suburban').lower()} area who last purchased "
        f"{months} month{'s' if months != 1 else ''} ago, spent ${row.history:,.2f} in the past year, "
        f"bought {bought_txt}, and shops by {str(row.channel).lower()}."
    )


def embed(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    """Sentence embeddings, cached on disk by a hash of model name + texts."""
    key = hashlib.sha256((model_name + "\n".join(texts)).encode()).hexdigest()[:16]
    path = PROCESSED_DIR / f"embeddings_{key}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    np.save(path, emb)
    return emb


def paired_qini_diff(s_a: np.ndarray, s_b: np.ndarray, y: np.ndarray, d: np.ndarray, reps: int, seed: int,
                     alpha: float) -> tuple[float, float, float]:
    """Qini(a) - Qini(b) with a paired percentile bootstrap CI over evaluation rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, n, n)
        diffs[r] = qini_coefficient(s_a[i], y[i], d[i]) - qini_coefficient(s_b[i], y[i], d[i])
    point = qini_coefficient(s_a, y, d) - qini_coefficient(s_b, y, d)
    return point, float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def run() -> pd.DataFrame:
    from .cate import evaluate, load_choice, make_learner
    from .data import load_frame, load_split, train_test

    cfg = load_config()
    ensure_output_dirs()
    l_cfg, seed, alpha, reps = cfg["llm_features"], cfg["seed"], cfg["alpha"], cfg["bootstrap_reps"]
    df = load_frame()
    emb = embed([describe(r) for r in df.itertuples()], l_cfg["model"], l_cfg["batch_size"])
    split = load_split()
    tr, te = np.array(split["train"]), np.array(split["test"])
    pca = PCA(l_cfg["pca_components"], random_state=seed).fit(emb[tr])
    (Xtr, ttr, Ytr), (Xte, tte, Yte) = train_test()
    choice = load_choice()
    learner, params = choice["selected"]["spend"], choice["params"]["spend"]
    ytr, yte = Ytr["spend"].to_numpy(), Yte["spend"].to_numpy()
    feats = {
        "tabular": (Xtr.to_numpy(), Xte.to_numpy()),
        "tabular + embedding PCs": (np.column_stack([Xtr, pca.transform(emb[tr])]),
                                    np.column_stack([Xte, pca.transform(emb[te])])),
    }
    preds = {name: make_learner(learner, params, seed).fit(a, ttr, ytr).effect(b) for name, (a, b) in feats.items()}
    table = evaluate(preds, tte, yte, reps, seed, alpha)
    table = table[table.learner != "Random"].rename(columns={"learner": "features"})
    diffs = []
    for a in (1, 2):
        m = (tte == a) | (tte == 0)
        d = (tte[m] == a).astype(float)
        point, lo, hi = paired_qini_diff(preds["tabular + embedding PCs"][m, a - 1], preds["tabular"][m, a - 1],
                                         yte[m], d, reps, seed + a, alpha)
        diffs.append({"arm": ARM_NAMES[a], "qini_gain": point, "qini_gain_ci_low": lo, "qini_gain_ci_high": hi})
    table = table.merge(pd.DataFrame(diffs), on="arm")
    table.loc[table.features == "tabular", ["qini_gain", "qini_gain_ci_low", "qini_gain_ci_high"]] = np.nan
    table.insert(0, "learner", learner)
    table["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
    table.to_csv(TABLES_DIR / "llm_features.csv", index=False)
    return table

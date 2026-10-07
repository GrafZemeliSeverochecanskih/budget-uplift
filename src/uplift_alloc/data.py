"""Load, validate, version and encode the Hillstrom email RCT.

Implements the six steps in CLAUDE.md ("Primary: how to load"):
fetch -> parquet -> assertions -> SHA-256 in DATA_VERSION.txt -> encoding -> ``load_hillstrom()``.

Assumption: randomisation is uniform over the three arms, so the true propensity is 1/3.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ARM_NAMES, DATA_DIR, PROCESSED_DIR, RAW_DIR, load_config

log = logging.getLogger(__name__)

N_ROWS = 64_000
OUTCOMES = ["visit", "conversion", "spend"]
ARM_CODES = {name: code for code, name in ARM_NAMES.items()}
PARQUET_PATH = PROCESSED_DIR / "hillstrom.parquet"
SPLIT_PATH = PROCESSED_DIR / "split.json"
VERSION_PATH = DATA_DIR / "DATA_VERSION.txt"
PROPENSITY = 1 / 3

Arrays = tuple[pd.DataFrame, np.ndarray, pd.DataFrame]


class DataVersionError(RuntimeError):
    """The raw file no longer matches the hash recorded in DATA_VERSION.txt."""


# 1. fetch --------------------------------------------------------------------
def fetch_raw(data_home: Path = RAW_DIR) -> pd.DataFrame:
    """Download (once) via scikit-uplift; return covariates, outcomes and ``segment``."""
    from sklift.datasets import fetch_hillstrom

    data_home.mkdir(parents=True, exist_ok=True)
    bunch = fetch_hillstrom(target_col="all", data_home=str(data_home))
    df = pd.concat([bunch.data, bunch.target, bunch.treatment.rename("segment")], axis=1)
    return df.reset_index(drop=True)


def raw_file(data_home: Path = RAW_DIR) -> Path:
    """The single raw CSV(.gz) file sklift caches."""
    files = sorted(data_home.glob("hill*.csv*"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one Hillstrom raw file in {data_home}, found {files}")
    return files[0]


# 3. assertions ---------------------------------------------------------------
def validate(df: pd.DataFrame) -> dict[str, int]:
    """Assert the documented properties; return soft-check counts (reported, never dropped)."""
    assert len(df) == N_ROWS, f"expected {N_ROWS} rows, got {len(df)}"
    assert not df.isna().any().any(), "missing values present"
    arms = set(df["segment"].unique())
    assert arms == set(ARM_CODES), f"unexpected arm labels {arms}"
    shares = df["segment"].value_counts(normalize=True)
    assert ((shares - 1 / 3).abs() < 0.01).all(), f"arm shares not ~1/3: {shares.to_dict()}"
    for col in ("visit", "conversion"):
        assert set(df[col].unique()) <= {0, 1}, f"{col} not binary"
    assert (df["spend"] >= 0).all(), "negative spend"
    violations = {
        "conversion_without_visit": int(((df["conversion"] == 1) & (df["visit"] == 0)).sum()),
        "spend_without_conversion": int(((df["spend"] > 0) & (df["conversion"] == 0)).sum()),
    }
    for name, count in violations.items():
        if count:
            log.warning("data check %s: %d rows (kept)", name, count)
    return violations


# 4. versioning ---------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_version(version_path: Path = VERSION_PATH) -> dict[str, str]:
    lines = version_path.read_text().splitlines()
    return dict(line.split(": ", 1) for line in lines if ": " in line)


def check_version(raw: Path, version_path: Path = VERSION_PATH) -> str:
    """Record the raw file hash on first run; raise ``DataVersionError`` if it later changes."""
    digest = sha256(raw)
    if version_path.exists():
        recorded = read_version(version_path).get("sha256")
        if recorded != digest:
            raise DataVersionError(
                f"{raw.name} has sha256 {digest} but {version_path} records {recorded}. "
                "The upstream data changed; investigate before re-running any analysis."
            )
        return digest
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(
        f"file: {raw.name}\nsha256: {digest}\ndownloaded: {dt.date.today().isoformat()}\n"
    )
    return digest


# 5. encoding -----------------------------------------------------------------
def encode_arm(segment: pd.Series) -> np.ndarray:
    """'No E-Mail' -> 0, 'Mens E-Mail' -> 1, 'Womens E-Mail' -> 2."""
    return segment.map(ARM_CODES).to_numpy(dtype=int)


def decode_arm(t: np.ndarray) -> pd.Series:
    return pd.Series(t).map(ARM_NAMES)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model features: numeric covariates + one-hot zip_code/channel; history_segment dropped."""
    X = df[["recency", "history", "mens", "womens", "newbie"]].astype(float)
    dummies = pd.get_dummies(df[["zip_code", "channel"]].astype(str), dtype=float)
    dummies.columns = [c.replace(" ", "_").replace("-", "_") for c in dummies.columns]
    return pd.concat([X, dummies], axis=1)


# 2 + orchestration -----------------------------------------------------------
def prepare() -> pd.DataFrame:
    """Fetch, version-check, validate and cache the combined frame as parquet."""
    df = fetch_raw()
    check_version(raw_file())
    validate(df)
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    return df


def load_frame() -> pd.DataFrame:
    """The combined raw frame (incl. ``history_segment`` for reporting)."""
    if not PARQUET_PATH.exists():
        prepare()
    return pd.read_parquet(PARQUET_PATH)


# 6. public loader ------------------------------------------------------------
def load_hillstrom() -> Arrays:
    """Return encoded features X, integer arm T (0 = control) and outcomes Y."""
    df = load_frame()
    return encode_features(df), encode_arm(df["segment"]), df[OUTCOMES].astype(float)


# split -----------------------------------------------------------------------
def make_split(t: np.ndarray, test_size: float, seed: int) -> dict[str, list[int]]:
    idx = np.arange(len(t))
    train, test = train_test_split(idx, test_size=test_size, stratify=t, random_state=seed)
    return {"train": sorted(map(int, train)), "test": sorted(map(int, test))}


def save_split(force: bool = False) -> dict:
    """Create the stratified split once; afterwards always reuse the saved indices."""
    if SPLIT_PATH.exists() and not force:
        return load_split()
    cfg = load_config()
    _, t, _ = load_hillstrom()
    split: dict = make_split(t, cfg["split"]["test_size"], cfg["seed"])
    split["meta"] = {"seed": cfg["seed"], "test_size": cfg["split"]["test_size"], "stratify": "arm"}
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(split))
    return split


def load_split() -> dict:
    return json.loads(SPLIT_PATH.read_text())


def subset(arrays: Arrays, idx: np.ndarray) -> Arrays:
    X, t, Y = arrays
    return X.iloc[idx].reset_index(drop=True), t[idx], Y.iloc[idx].reset_index(drop=True)


def train_test() -> tuple[Arrays, Arrays]:
    """(X, T, Y) for train and test, using the saved split."""
    arrays = load_hillstrom()
    split = save_split()
    return subset(arrays, np.array(split["train"])), subset(arrays, np.array(split["test"]))


# data card -------------------------------------------------------------------
COVARIATE_DESCRIPTIONS = {
    "recency": "Months since last purchase (1-12).",
    "history_segment": "Binned past-year spend; reporting only (a coarsened copy of `history`).",
    "history": "Past-year spend in $.",
    "mens": "1 if the customer bought Mens merchandise in the past year.",
    "womens": "1 if the customer bought Womens merchandise in the past year.",
    "zip_code": "Urban / Surburban / Rural (sic, the raw label is 'Surburban'); one-hot encoded.",
    "newbie": "1 if the customer is new in the past 12 months.",
    "channel": "Purchase channel in the past year: Phone / Web / Multichannel; one-hot encoded.",
}


def data_card_markdown(df: pd.DataFrame, violations: dict[str, int]) -> str:
    """Render DATA_CARD.md from the validated frame (all numbers computed, none typed by hand)."""
    version = read_version()
    by_arm = df.groupby("segment")[OUTCOMES].agg(["mean"]).droplevel(1, axis=1)
    counts = df["segment"].value_counts()
    rows = [
        f"| {arm} | {counts[arm]:,} | {counts[arm] / len(df):.3f} | {r.visit:.4f} | "
        f"{r.conversion:.4f} | {r.spend:.3f} |"
        for arm, r in by_arm.loc[list(ARM_CODES)].iterrows()
    ]
    overall = df[OUTCOMES].mean()
    cov = [f"| `{k}` | {v} |" for k, v in COVARIATE_DESCRIPTIONS.items()]
    return "\n".join([
        "# Data card: Hillstrom email RCT",
        "",
        "Kevin Hillstrom's MineThatData E-Mail Analytics challenge (2008). 64,000 customers who",
        "purchased in the past 12 months were randomised in equal thirds to a Mens e-mail, a Womens",
        "e-mail, or no e-mail; outcomes were tracked for the two weeks after the campaign.",
        "",
        f"- Source: `sklift.datasets.fetch_hillstrom` (scikit-uplift 0.5.1), file `{version.get('file')}`",
        f"- SHA-256: `{version.get('sha256')}` (first downloaded {version.get('downloaded')})",
        "- Propensity: 1/3 per arm by design (uniform randomisation).",
        "- Generated by `scripts/download_data.py`; do not edit by hand.",
        "",
        "## Rows and outcome base rates per arm",
        "",
        "| Arm | Rows | Share | Visit rate | Conversion rate | Mean spend ($) |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        f"| **All** | {len(df):,} | 1.000 | {overall.visit:.4f} | {overall.conversion:.4f} | {overall.spend:.3f} |",
        "",
        f"Share of customers with spend > 0: {(df['spend'] > 0).mean():.4f}. "
        f"Mean spend among buyers: ${df.loc[df['spend'] > 0, 'spend'].mean():.2f}.",
        "",
        "## Covariates",
        "",
        "| Column | Description |",
        "|---|---|",
        *cov,
        "",
        "## Outcomes",
        "",
        "| Column | Description |",
        "|---|---|",
        "| `visit` | 1 if the customer visited the website in the 2 weeks after the e-mail. |",
        "| `conversion` | 1 if the customer purchased in the 2 weeks after the e-mail. |",
        "| `spend` | $ spent in the 2 weeks after the e-mail (0 for non-buyers; heavily skewed). |",
        "",
        "## Soft checks (reported, rows kept)",
        "",
        *[f"- `{k}`: {v} rows" for k, v in violations.items()],
        "",
    ])

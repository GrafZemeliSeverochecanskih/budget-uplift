"""Project paths and the analysis config (``config/config.yaml``).

Quick mode (``UPLIFT_QUICK=1``, set by ``run_all --quick`` / ``make quick``) merges the config's
``quick`` block over the full config and redirects outputs to ``results/quick`` and caches to
``data/processed/quick`` so the full results are never overwritten. It must be set before this
module is imported.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

QUICK = os.environ.get("UPLIFT_QUICK") == "1"
RESULTS_DIR = ROOT / "results" / ("quick" if QUICK else "")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CACHE_DIR = PROCESSED_DIR / "quick" if QUICK else PROCESSED_DIR

ARM_NAMES = {0: "No E-Mail", 1: "Mens E-Mail", 2: "Womens E-Mail"}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config once per process (with the ``quick`` overrides in quick mode)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _merge(cfg, cfg.get("quick", {})) if QUICK else cfg


def arm_costs(cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Cost per email indexed by arm (index 0 = no email, cost 0)."""
    cfg = cfg or load_config()
    return np.array([0.0, cfg["costs"]["mens"], cfg["costs"]["womens"]])


def budget_for(fraction: float, n: int, cfg: dict[str, Any] | None = None) -> float:
    """Budget B = fraction * n * mean email cost (see config comment)."""
    return float(fraction * n * arm_costs(cfg)[1:].mean())


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

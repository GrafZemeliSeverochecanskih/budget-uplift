"""Reproduce every table and figure: ``python -m uplift_alloc.run_all [--quick]``.

Cached intermediates (predictions, embeddings) are deleted first so the run starts from the
validated raw data; pass ``--keep-cache`` to reuse them. ``--quick`` runs the same pipeline with
the config's ``quick`` overrides (50 bootstrap reps, lighter solver benchmark) and writes to
``results/quick/`` without touching the full results.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import time

STEPS = ["rct", "cate", "allocate", "ope", "positive_control", "confounding", "robust_allocate",
         "llm_features", "sweep", "profit", "policy_learning", "chance", "sequential", "chance15", "scale_criteo"]
KEEP = {"hillstrom.parquet", "split.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", nargs="*", choices=STEPS)
    args = parser.parse_args()
    if args.quick:
        os.environ["UPLIFT_QUICK"] = "1"  # must precede the first import of uplift_alloc.config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("run_all")

    from .config import CACHE_DIR, QUICK, ROOT, ensure_output_dirs, load_config
    from .data import data_card_markdown, prepare, save_split, validate

    df = prepare()
    save_split()
    if not QUICK:
        (ROOT / "DATA_CARD.md").write_text(data_card_markdown(df, validate(df)), encoding="utf-8")
    ensure_output_dirs()
    if not args.keep_cache and not args.only:
        for p in CACHE_DIR.glob("*"):
            if p.is_file() and p.name not in KEEP:
                p.unlink()
    cfg = load_config()
    log.info("mode: %s, bootstrap reps: %d", "quick" if QUICK else "full", cfg["bootstrap_reps"])
    for step in args.only or STEPS:
        if step == "scale_criteo" and not cfg["criteo"]["enabled"]:
            log.info("skip scale_criteo (criteo.enabled is false in config)")
            continue
        if step == "llm_features" and importlib.util.find_spec("sentence_transformers") is None:
            log.warning("skip llm_features: install the [llm] extra (pip install -e '.[llm]')")
            continue
        start = time.perf_counter()
        log.info("start %s", step)
        importlib.import_module(f"uplift_alloc.{step}").run()
        log.info("done %s in %.0fs", step, time.perf_counter() - start)


if __name__ == "__main__":
    main()

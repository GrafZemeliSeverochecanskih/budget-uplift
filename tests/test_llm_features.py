import numpy as np
import pandas as pd

from uplift_alloc import llm_features


def test_describe_renders_all_fields():
    row = pd.Series({"recency": 1, "history": 1234.5, "mens": 1, "womens": 0, "newbie": 1,
                     "zip_code": "Surburban", "channel": "Web"})
    s = llm_features.describe(row)
    assert s == ("A new customer from a suburban area who last purchased 1 month ago, spent $1,234.50 "
                 "in the past year, bought men's merchandise, and shops by web.")


def test_paired_qini_diff_zero_for_identical_scores():
    rng = np.random.default_rng(0)
    s = rng.random(2000)
    y = rng.random(2000)
    d = rng.integers(0, 2, 2000).astype(float)
    point, lo, hi = llm_features.paired_qini_diff(s, s, y, d, reps=50, seed=0, alpha=0.05)
    assert point == lo == hi == 0

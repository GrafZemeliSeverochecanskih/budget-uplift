import numpy as np
import pandas as pd
import pytest

from uplift_alloc import data
from uplift_alloc.config import ARM_NAMES

needs_data = pytest.mark.skipif(
    not data.PARQUET_PATH.exists(), reason="run `make data` first"
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return data.load_frame()


@needs_data
def test_row_count(frame):
    assert len(frame) == 64_000
    X, t, Y = data.load_hillstrom()
    assert len(X) == len(t) == len(Y) == 64_000


@needs_data
def test_arm_labels_and_shares(frame):
    assert set(frame["segment"]) == set(ARM_NAMES.values())
    shares = frame["segment"].value_counts(normalize=True)
    assert np.allclose(shares.to_numpy(), 1 / 3, atol=0.01)


@needs_data
def test_value_ranges(frame):
    assert set(frame["visit"].unique()) <= {0, 1}
    assert set(frame["conversion"].unique()) <= {0, 1}
    assert (frame["spend"] >= 0).all()
    assert frame["recency"].between(1, 12).all()
    assert (frame["history"] >= 0).all()
    assert not frame.isna().any().any()


@needs_data
def test_features_exclude_history_segment():
    X, _, _ = data.load_hillstrom()
    assert "history_segment" not in X.columns
    assert X.notna().all().all()
    # one-hot blocks sum to one per row
    for prefix in ("zip_code_", "channel_"):
        assert (X.filter(like=prefix).sum(axis=1) == 1).all()


def test_arm_encoding_round_trip():
    labels = pd.Series(["No E-Mail", "Mens E-Mail", "Womens E-Mail", "Mens E-Mail"])
    codes = data.encode_arm(labels)
    assert codes.tolist() == [0, 1, 2, 1]
    assert data.decode_arm(codes).tolist() == labels.tolist()


def test_split_is_disjoint_and_stratified():
    t = np.repeat([0, 1, 2], 1000)
    split = data.make_split(t, test_size=0.3, seed=0)
    train, test = set(split["train"]), set(split["test"])
    assert not train & test
    assert train | test == set(range(len(t)))
    assert np.bincount(t[list(test)]).tolist() == [300, 300, 300]


@needs_data
def test_saved_split_no_overlap():
    split = data.load_split()
    train, test = set(split["train"]), set(split["test"])
    assert not train & test
    assert len(train) + len(test) == 64_000


def test_version_check_detects_change(tmp_path):
    raw = tmp_path / "raw.csv"
    version = tmp_path / "DATA_VERSION.txt"
    raw.write_text("a,b\n1,2\n")
    digest = data.check_version(raw, version)
    assert data.check_version(raw, version) == digest
    raw.write_text("a,b\n1,3\n")
    with pytest.raises(data.DataVersionError):
        data.check_version(raw, version)


def test_validate_rejects_bad_frame():
    df = pd.DataFrame({"segment": ["No E-Mail"] * 10, "visit": 0, "conversion": 0, "spend": 0.0})
    with pytest.raises(AssertionError):
        data.validate(df)

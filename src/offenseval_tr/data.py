"""OffensEval-TR 2020 loading and the fixed split used by every system.

The official zip is loaded directly (the Hugging Face copy is only a loading
script). Test labels are joined by tweet id, never by row order.
"""
from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DEV_SIZE = 1756
URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
TRAIN_TSV = "offenseval2020-turkish/offenseval-tr-training-v1/offenseval-tr-training-v1.tsv"
TEST_TSV = "offenseval2020-turkish/offenseval-tr-testset-v1/offenseval-tr-testset-v1.tsv"
TEST_LAB = "offenseval2020-turkish/offenseval-tr-testset-v1/offenseval-tr-labela-v1.tsv"
ZIP_SHA256 = "7977e96255dbc9b8d14893f1b14cbe3dec53c70358503c062c5a59720ec9c2f2"


def load(zip_path: str = "offenseval2020-turkish.zip") -> dict[str, pd.DataFrame]:
    """Return {"train": ..., "test": ...} with columns id, text, label."""
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(URL, zip_path)
    with open(zip_path, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == ZIP_SHA256, (
            f"{zip_path} differs from the release used for the reported results"
        )
    with zipfile.ZipFile(zip_path) as z:
        def read(path: str) -> pd.DataFrame:
            return pd.read_csv(io.BytesIO(z.read(path)), sep="\t", quoting=3,
                               dtype={"id": str}, keep_default_na=False)
        train = read(TRAIN_TSV)
        test = read(TEST_TSV)
        labels = pd.read_csv(io.BytesIO(z.read(TEST_LAB)), header=None,
                             names=["id", "subtask_a"], dtype=str)
    test = test.merge(labels, on="id", how="left", validate="one_to_one")
    assert test.subtask_a.notna().all(), "test tweet without a label"
    out = {}
    for key, frame in (("train", train), ("test", test)):
        out[key] = pd.DataFrame({
            "id": frame["id"],
            "text": frame["tweet"].astype(str),
            "label": frame["subtask_a"].str.strip().str.upper(),
        })
        assert set(out[key].label) == {"NOT", "OFF"}
        assert out[key].id.is_unique
    return out


def tr_lower(text: str) -> str:
    """Turkish-aware lowercasing: I -> ı, İ -> i, then lower()."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def make_splits(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 30,000 / 1,756 train/dev split, seed 42."""
    train, dev = train_test_split(full, test_size=DEV_SIZE, stratify=full.label,
                                  random_state=SEED)
    return train.reset_index(drop=True), dev.reset_index(drop=True)


def subset(train: pd.DataFrame, n: int, k: int) -> pd.DataFrame:
    """Random sample k of size n from train, stratified, seed 42 + k."""
    if n >= len(train):
        return train
    return train_test_split(train, train_size=n, stratify=train.label,
                            random_state=SEED + k)[0]

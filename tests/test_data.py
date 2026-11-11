import os

import pytest
from sklearn.model_selection import train_test_split

from offenseval_tr.data import DEV_SIZE, SEED, make_splits, subset


@pytest.mark.skipif(not os.path.exists("offenseval2020-turkish.zip"), reason="data zip not present")
def test_split_counts_and_ratio():
    import offenseval_tr.data as data
    full = data.load()["train"]
    train, dev = make_splits(full)
    assert len(train) == 30000 and len(dev) == DEV_SIZE
    assert abs((train.label == "OFF").mean() - (dev.label == "OFF").mean()) < 0.01
    assert set(train.id).isdisjoint(set(dev.id))


def test_subset_is_deterministic():
    import pandas as pd
    train = pd.DataFrame({"id": [str(i) for i in range(5000)],
                          "text": ["x"] * 5000,
                          "label": ["OFF"] * 1000 + ["NOT"] * 4000})
    a = subset(train, 250, 1)
    b = subset(train, 250, 1)
    assert a.id.tolist() == b.id.tolist()
    assert len(a) == 250
    assert abs((a.label == "OFF").mean() - 0.2) < 0.02

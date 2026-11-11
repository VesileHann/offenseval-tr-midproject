#!/usr/bin/env python
"""TF-IDF + LR baseline on dev, with the fixed split and saved subsets."""
import argparse
import json
import os
import platform

import numpy as np
import pandas as pd

from offenseval_tr import baselines, data, metrics

SIZES = {250: 10, 750: 10, 2000: 10, 6000: 3, 30000: 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--zip", default="offenseval2020-turkish.zip")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    loaded = data.load(args.zip)
    full, test = loaded["train"], loaded["test"]
    train, dev = data.make_splits(full)
    for name, frame in (("train", train), ("dev", dev), ("test", test)):
        frame[["id", "label"]].to_csv(f"{args.out}/{name}_ids.csv", index=False)
    pd.concat([
        data.subset(train, n, k)[["id", "label"]].assign(size=n, subset=k)
        for n, runs in SIZES.items() if n < len(train) for k in range(1, runs + 1)
    ])[["size", "subset", "id", "label"]].to_csv(f"{args.out}/subsets.csv", index=False)

    pred = baselines.tfidf_lr(train.text, train.label)(dev.text)
    res = {
        "sizes": {
            "official_train": {"n": len(full), "OFF": int((full.label == "OFF").sum())},
            "train": {"n": len(train), "OFF": int((train.label == "OFF").sum())},
            "dev": {"n": len(dev), "OFF": int((dev.label == "OFF").sum())},
            "test": {"n": len(test), "OFF": int((test.label == "OFF").sum())},
        },
        "majority_dev_macro_f1": metrics.macro_f1(dev.label, ["NOT"] * len(dev)),
        "tfidf_lr_30000_dev": {
            "macro_f1": metrics.macro_f1(dev.label, pred),
            "ci95": metrics.bootstrap_ci(dev.label, pred).tolist(),
        },
        "curve": {},
    }
    if args.curve:
        for n, runs in SIZES.items():
            scores = []
            for k in range(1, runs + 1):
                tr = data.subset(train, n, k)
                p = baselines.tfidf_lr(tr.text, tr.label)(dev.text)
                scores.append(metrics.macro_f1(dev.label, p))
            res["curve"][str(n)] = scores
    with open(f"{args.out}/dev_curve.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"dev macro-F1: {res['tfidf_lr_30000_dev']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

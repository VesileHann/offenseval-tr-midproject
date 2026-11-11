#!/usr/bin/env python
"""Score the deterministic TF-IDF baseline on the frozen official test set."""
import argparse
import json
import os

import pandas as pd
from sklearn.metrics import classification_report

from offenseval_tr import baselines, data, metrics

SIZES = {250: 10, 750: 10, 2000: 10, 6000: 3, 30000: 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--zip", default="offenseval2020-turkish.zip")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    loaded = data.load(args.zip)
    train, dev = data.make_splits(loaded["train"])
    test = loaded["test"]
    y = (test.label == "OFF").to_numpy()

    pred = baselines.tfidf_lr(train.text, train.label)(test.text)
    result = {
        "test_n": len(test), "test_off": int(y.sum()),
        "tfidf_30000": {
            "macro_f1": metrics.macro_f1(y, pred == "OFF"),
            "ci95": metrics.bootstrap_ci(test.label, pred).tolist(),
            "report": classification_report(y, pred == "OFF", output_dict=True, zero_division=0),
        },
        "curve": {},
    }
    pd.DataFrame({"id": test.id, "pred": pred}).to_csv(
        f"{args.out}/test_predictions.csv", index=False)
    for n, runs in SIZES.items():
        scores = []
        for k in range(1, runs + 1):
            tr = data.subset(train, n, k)
            p = baselines.tfidf_lr(tr.text, tr.label)(test.text)
            scores.append(metrics.macro_f1(test.label, p))
        result["curve"][str(n)] = scores
    with open(f"{args.out}/test_curve.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"test macro-F1: {result['tfidf_30000']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

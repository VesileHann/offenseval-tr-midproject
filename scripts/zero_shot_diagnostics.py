#!/usr/bin/env python
"""Quantify why threshold calibration helps some models more than others.

For each model's zero-shot dev scores:
  F1 at threshold 0, best-threshold F1, ROC-AUC, mean score by gold class,
  share predicted offensive at 0.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from offenseval_tr.metrics import best_threshold, macro_f1


def main():
    out = {}
    for path in sorted(glob.glob("results/llm/*/zeroshot_run1_dev_scores.csv")):
        model = path.split("/")[2]
        d = pd.read_csv(path)
        y = d.label == "OFF"
        scores = d.score.to_numpy()
        threshold, best_f1 = best_threshold(y, scores)
        out[model] = {
            "f1_at_zero": macro_f1(y, scores > 0),
            "best_threshold": threshold,
            "best_f1": best_f1,
            "auc": roc_auc_score(y, scores),
            "mean_off": float(scores[y].mean()),
            "mean_not": float(scores[~y].mean()),
            "pred_off_rate_at_zero": float((scores > 0).mean()),
        }
    with open("results/zero_shot_diagnostics.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print(f"{k:14s} F1@0={v['f1_at_zero']:.3f}  best={v['best_f1']:.3f}"
              f"  AUC={v['auc']:.3f}  mean(OFF)={v['mean_off']:+.2f}"
              f"  mean(NOT)={v['mean_not']:+.2f}  predOFF@0={v['pred_off_rate_at_zero']:.2f}")


if __name__ == "__main__":
    main()

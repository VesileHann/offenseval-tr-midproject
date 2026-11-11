#!/usr/bin/env python
"""Aggregate baseline, error-audit, LLM and fine-tuning runs into results/summary.json."""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from offenseval_tr.metrics import paired_bootstrap_runs


def main():
    root = "results"
    dev = json.load(open(f"{root}/baseline/dev_curve.json"))
    test = json.load(open(f"{root}/baseline/test_curve.json"))
    audit = json.load(open(f"{root}/errors/audit.json"))
    # Always "NOT": F1(OFF) = 0, so macro-F1 is F1(NOT) / 2.
    test_labels = pd.read_csv(f"{root}/baseline/test_ids.csv").label
    n_not, n_off = int((test_labels == "NOT").sum()), int((test_labels == "OFF").sum())
    majority_test = (2 * n_not / (2 * n_not + n_off)) / 2
    summary = {
        "sizes": dev["sizes"],
        "majority_dev": dev["majority_dev_macro_f1"],
        "majority_test": majority_test,
        "dev_curve": dev["curve"],
        "test_curve": test["curve"],
        "tfidf_30000_dev": dev["tfidf_lr_30000_dev"],
        "tfidf_30000_test": test["tfidf_30000"],
        "error_audit": audit,
        "llm": {},
        "finetune": {},
    }
    groups = defaultdict(list)
    for path in sorted(glob.glob(f"{root}/llm/*/*.json")):
        run = json.load(open(path))
        groups[f"{run['model']}_{run['mode']}"].append(run)
    for key, runs in groups.items():
        values = [r["dev_macro_f1"] for r in runs]
        summary["llm"][key] = {
            "runs": runs, "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "thresholds": [r["threshold"] for r in runs],
            "cal": [r["cal_macro_f1"] for r in runs],
        }
    ft = defaultdict(list)
    for path in sorted(glob.glob(f"{root}/finetune/*/n*_run*.json")):
        run = json.load(open(path))
        ft[(run["model"], run["size"])].append(run)
    for (model, size), runs in sorted(ft.items()):
        runs.sort(key=lambda r: r["run"])
        values = [r["dev_macro_f1"] for r in runs]
        summary["finetune"][f"{model}_{size}"] = {
            "model": model, "size": size, "n_runs": len(runs), "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "off_recall": float(np.mean([r["dev_off_recall"] for r in runs])),
            "single_class_runs": sum(r["single_class"] for r in runs), "dev_macro_f1": values,
        }
    # Ablation A on dev: BERTurk vs XLM-R-base on the same 750-label subsets.
    if ("berturk", 750) in ft and ("xlmr-base", 750) in ft:
        def preds(model):
            return [pd.read_csv(f"{root}/finetune/{model}/n750_run{r['run']}_dev_scores.csv").pred.to_numpy()
                    for r in ft[(model, 750)]]
        labels = pd.read_csv(f"{root}/finetune/berturk/n750_run1_dev_scores.csv").label.to_numpy()
        gap, lower, wins, interval = paired_bootstrap_runs(labels, preds("berturk"), preds("xlmr-base"))
        summary["ablation_a_dev"] = {"gap": gap, "one_sided_lower_95": lower, "win_rate": wins, "ci95": interval}
    with open(f"{root}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {root}/summary.json with {len(summary['llm'])} LLM and {len(summary['finetune'])} fine-tuning groups")


if __name__ == "__main__":
    main()

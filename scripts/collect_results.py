#!/usr/bin/env python
"""Aggregate baseline, error-audit and LLM runs into results/summary.json."""
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    root = "results"
    dev = json.load(open(f"{root}/baseline/dev_curve.json"))
    test = json.load(open(f"{root}/baseline/test_curve.json"))
    audit = json.load(open(f"{root}/errors/audit.json"))
    summary = {
        "sizes": dev["sizes"],
        "majority_dev": dev["majority_dev_macro_f1"],
        "majority_test": 0.5,
        "dev_curve": dev["curve"],
        "test_curve": test["curve"],
        "tfidf_30000_dev": dev["tfidf_lr_30000_dev"],
        "tfidf_30000_test": test["tfidf_30000"],
        "error_audit": audit,
        "llm": {},
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
    with open(f"{root}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {root}/summary.json with {len(summary['llm'])} LLM groups")


if __name__ == "__main__":
    main()

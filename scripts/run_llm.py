#!/usr/bin/env python
"""Run zero-shot or 8-example + 742-label threshold calibration."""
import argparse
import os

from offenseval_tr import data, llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(llm.MODEL_IDS))
    ap.add_argument("--mode", default="fewshot", choices=("fewshot", "zeroshot"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="results/llm")
    ap.add_argument("--zip", default="offenseval2020-turkish.zip")
    args = ap.parse_args()
    loaded = data.load(args.zip)
    train, dev = data.make_splits(loaded["train"])
    llm.run(args.model, train, dev, os.path.join(args.out, args.model), args.mode,
            args.runs, args.batch, token=os.environ.get("HF_TOKEN"))


if __name__ == "__main__":
    main()

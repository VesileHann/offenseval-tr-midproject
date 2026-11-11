#!/usr/bin/env python
"""Fine-tune BERTurk (learning curve) or XLM-R-base (Ablation A) on the shared subsets."""
import argparse
import os

from offenseval_tr import data, finetune

PLAN = {250: 10, 750: 10, 2000: 10, 6000: 3, 30000: 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="berturk", choices=sorted(finetune.MODEL_IDS))
    ap.add_argument("--sizes", type=int, nargs="+", default=[750])
    ap.add_argument("--start-run", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="20 updates on run 1, nothing saved")
    ap.add_argument("--out", default="results/finetune")
    ap.add_argument("--zip", default="offenseval2020-turkish.zip")
    args = ap.parse_args()
    loaded = data.load(args.zip)
    train, dev = data.make_splits(loaded["train"])
    for size in args.sizes:
        runs = [1] if args.smoke else list(range(args.start_run, PLAN[size] + 1))
        finetune.run(args.model, size, runs, train, dev, loaded["test"],
                     os.path.join(args.out, args.model), smoke=args.smoke)


if __name__ == "__main__":
    main()

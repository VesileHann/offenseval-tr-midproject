#!/usr/bin/env python
"""Dump TF-IDF dev errors and audit 30 per direction with fixed categories."""
import argparse
import json
import os

import numpy as np
import pandas as pd

from offenseval_tr import baselines, data

CATEGORIES = {
    "false_OFF": {
        "No insult / spurious cue": ["12420", "15670", "18845", "25321", "28821", "32349", "40559", "40582", "43440"],
        "Political or religious criticism": ["14272", "14585", "15098", "16809", "17089", "23656", "26192", "35124", "41296", "45176"],
        "Sports criticism or banter": ["13105", "13559", "23319", "33787", "41313"],
        "Personal criticism not judged offensive": ["22409", "29143", "30133"],
        "Sarcasm or irony": ["28737"],
        "Label noise or genuinely ambiguous": ["33218", "34166"],
    },
    "missed_OFF": {
        "Implicit insult, no profanity": ["12817", "16575", "23595", "24871", "28117", "31597", "43325", "45882"],
        "Masked or abbreviated profanity": ["11358", "15069", "24976", "32074", "34977"],
        "Veiled threat or violence": ["13285", "13464", "35789"],
        "Political or religious insult": ["18840", "25929", "31547", "37022"],
        "Institutional or sports criticism": ["20333", "20907", "36590"],
        "Label noise or not offensive": ["10001", "19272", "22248", "32434", "33592", "35932", "36488"],
    },
}

EXAMPLES = {
    "No insult / spurious cue": ("12420", "Fan poll; no insult. A character n-gram cue fired."),
    "Political or religious criticism": ("35124", "Political accusation (“like a thief”), labelled NOT."),
    "Sports criticism or banter": ("13105", "Club criticism without a slur."),
    "Personal criticism not judged offensive": ("22409", "Mild personal criticism."),
    "Sarcasm or irony": ("28737", "Sarcasm with no offensive word."),
    "Label noise or genuinely ambiguous": ("33218", "A curse; the NOT label is questionable."),
    "Implicit insult, no profanity": ("43325", "“No more parasitic species” — an insult with no swear word."),
    "Masked or abbreviated profanity": ("15069", "“amkk” is a masked swear; the token is unseen."),
    "Veiled threat or violence": ("35789", "An insult plus a threat."),
    "Political or religious insult": ("25929", "“Bigoted reactionaries” — a political insult."),
    "Institutional or sports criticism": ("20333", "Institutional criticism; the OFF label is a stretch."),
    "Label noise or not offensive": ("10001", "A joke; no offence."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/baseline")
    ap.add_argument("--out", default="results/errors")
    ap.add_argument("--zip", default="offenseval2020-turkish.zip")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    loaded = data.load(args.zip)
    train, dev = data.make_splits(loaded["train"])
    pred = baselines.tfidf_lr(train.text, train.label)(dev.text)
    frame = dev.assign(pred=pred)
    err = frame[frame.label != frame.pred].copy()
    err["kind"] = np.where((err.label == "OFF") & (err.pred == "NOT"), "missed_OFF", "false_OFF")
    err = err.sort_values(["kind", "id"]).reset_index(drop=True)
    err[["kind", "id", "label", "pred", "text"]].to_csv(f"{args.out}/dev_errors.tsv", sep="\t", index=False)
    sample = pd.concat([g.sample(n=30, random_state=42) for _, g in err.groupby("kind")])
    ids = set(sample.id)
    summary = []
    for direction, table in CATEGORIES.items():
        for category, members in table.items():
            assert set(members) <= ids
            ex_id, gloss = EXAMPLES[category]
            summary.append({"direction": direction, "category": category,
                            "count": len(members), "example_id": ex_id, "example": gloss})
    with open(f"{args.out}/audit.json", "w") as f:
        json.dump({"sample": 60, "false_OFF_total": int((err.kind == "false_OFF").sum()),
                   "missed_OFF_total": int((err.kind == "missed_OFF").sum()),
                   "categories": summary}, f, indent=2)
    print("errors:", len(err), "audited:", len(sample))


if __name__ == "__main__":
    main()

"""Fine-tuning arm: BERTurk (and XLM-R-base for Ablation A) on the shared subsets.

Settings are fixed before results (proposal box 4): 128 tokens, batch 16,
lr 2e-5 with 10% linear warm-up, class-weighted cross-entropy, the longer of
4 epochs or 400 updates, no early stopping. Run k trains on subset k with seed
42 + k. Dev is scored; test predictions are saved unscored until the freeze.
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import time

import numpy as np
import pandas as pd

from . import data
from .metrics import macro_f1

MODEL_IDS = {
    "berturk": "dbmdz/bert-base-turkish-cased",
    "xlmr-base": "FacebookAI/xlm-roberta-base",
}
MAX_LEN, BATCH, LR, WARMUP, EPOCHS, MIN_STEPS = 128, 16, 2e-5, 0.10, 4, 400


def _seed_all(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _loader(tok, texts, labels, shuffle, gen):
    import transformers
    from torch.utils.data import DataLoader
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN)
    rows = [{"input_ids": i, "attention_mask": a} for i, a in zip(enc["input_ids"], enc["attention_mask"])]
    if labels is not None:
        for row, label in zip(rows, labels):
            row["labels"] = int(label)
    return DataLoader(rows, batch_size=BATCH if shuffle else 128, shuffle=shuffle, generator=gen,
                      collate_fn=transformers.DataCollatorWithPadding(tok))


def _predict(model, tok, texts) -> np.ndarray:
    """P(OFF) for each text."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for batch in _loader(tok, texts, None, False, None):
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**{k: v.cuda() for k, v in batch.items()}).logits
            out.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def train_one(model_key: str, train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame,
              seed: int, smoke: bool = False) -> dict:
    """Train one run; return its record plus dev/test P(OFF)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    _seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_IDS[model_key])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_IDS[model_key], num_labels=2).cuda()
    y = (train.label == "OFF").astype(int).to_numpy()
    weight = torch.tensor(len(y) / (2 * np.bincount(y, minlength=2)), dtype=torch.float).cuda()
    loader = _loader(tok, train.text, y, True, gen)
    steps = 20 if smoke else max(EPOCHS * len(loader), MIN_STEPS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    start, step = time.time(), 0
    model.train()
    while step < steps:
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step >= steps:
                break
    train_s = time.time() - start
    p_dev, p_test = _predict(model, tok, dev.text), _predict(model, tok, test.text)
    del model
    torch.cuda.empty_cache()
    return {"steps": steps, "train_s": round(train_s, 1), "p_dev": p_dev, "p_test": p_test,
            "gpu": torch.cuda.get_device_name(0)}


def record(model_key: str, size: int, run: int, dev: pd.DataFrame, p_dev: np.ndarray, **extra) -> dict:
    """Run JSON in the same shape as the LLM runs (see docs/RESULT_SCHEMA.md)."""
    pred = p_dev >= 0.5
    y = (dev.label == "OFF").to_numpy()
    tp = int((y & pred).sum())
    return {
        "model": model_key, "hf_id": MODEL_IDS[model_key], "size": size, "run": run, "seed": data.SEED + run,
        "n_dev": len(dev), "dev_macro_f1": macro_f1(y, pred),
        "dev_off_precision": tp / max(int(pred.sum()), 1), "dev_off_recall": tp / int(y.sum()),
        "single_class": bool(pred.all() or not pred.any()), **extra,
    }


def save_run(out_dir: str, rec: dict, dev: pd.DataFrame, p_dev, test: pd.DataFrame, p_test) -> None:
    name = f"n{rec['size']}_run{rec['run']}"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"id": dev.id, "label": dev.label, "score": np.round(p_dev, 6),
                  "pred": np.where(p_dev >= 0.5, "OFF", "NOT")}).to_csv(f"{out_dir}/{name}_dev_scores.csv", index=False)
    pd.DataFrame({"id": test.id, "score": np.round(p_test, 6),
                  "pred": np.where(p_test >= 0.5, "OFF", "NOT")}).to_csv(f"{out_dir}/{name}_test_scores.csv", index=False)
    with open(f"{out_dir}/{name}.json", "w") as f:
        json.dump(rec, f, indent=2)


def run(model_key: str, size: int, runs: list[int], train: pd.DataFrame, dev: pd.DataFrame,
        test: pd.DataFrame, out_dir: str, smoke: bool = False) -> None:
    """Train each run that has no saved JSON yet, so a lost session costs one run."""
    import torch, transformers
    for k in runs:
        name = f"n{size}_run{k}"
        if not smoke and os.path.exists(f"{out_dir}/{name}.json"):
            print(f"skip {model_key} {name}", flush=True)
            continue
        # Same tweets as data.subset, kept in split order (the order the reported runs used).
        sample = train[train.id.isin(set(data.subset(train, size, k).id))]
        out = train_one(model_key, sample, dev, test, data.SEED + k, smoke)
        rec = record(model_key, size, k, dev, out["p_dev"], steps=out["steps"],
                     timing={"train_s": out["train_s"], "gpu": out["gpu"]},
                     versions={"python": platform.python_version(), "torch": torch.__version__,
                               "transformers": transformers.__version__})
        print(f"{model_key} {name}: steps={rec['steps']} dev macro-F1={rec['dev_macro_f1']:.3f}", flush=True)
        if not smoke:
            save_run(out_dir, rec, dev, out["p_dev"], test, out["p_test"])

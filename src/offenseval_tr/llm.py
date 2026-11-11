"""Few-shot scoring and threshold calibration for open decoder LLMs.

The model never writes a free answer. We read log P(A) - log P(B) at the first
answer token, where A = saldırgan (offensive) and B = saldırgan değil.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .metrics import best_threshold, macro_f1

MODEL_IDS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "gemma-4-12b": "google/gemma-4-12B-it",
    "gemma-4-26b-a4b": "google/gemma-4-26B-A4B-it",
}

INSTRUCTIONS = (
    "Aşağıdaki tweetleri saldırgan (A) veya saldırgan değil (B) olarak sınıflandır.\n"
    "Örnekler:\n"
)

SYSTEM = (
    "Sen bir Türkçe tweet sınıflandırıcısısın. Her tweet için iki seçenekten birini seç: "
    "A) saldırgan, B) saldırgan değil. Saldırgan: bir kişiye, gruba veya kuruma yönelik hakaret, "
    "aşağılama, tehdit veya nefret içeren ifade. Saldırgan değil: kimseye yönelmeyen, sadece bir "
    "duruma tepki gösteren veya bir sözü alıntılayan ifade. Yalnızca A veya B yaz."
)


def make_prompt(tok, tweet: str, examples: list[dict]) -> str:
    parts = [INSTRUCTIONS]
    for ex in examples:
        answer = "A" if ex["label"] == "OFF" else "B"
        parts.append(f"Tweet: {ex['text']}\nCevap: {answer}\n")
    parts.append(f"Tweet: {tweet}\nCevap:")
    messages = [{"role": "user", "content": "".join(parts)}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if tok.chat_template and "enable_thinking" in tok.chat_template:
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(messages, **kwargs)


def option_ids(tok) -> dict[str, list[int]]:
    return {opt: [tok.encode(opt, add_special_tokens=False)[0]] for opt in ("A", "B")}


def select_examples(train_subset, k: int) -> tuple[list[dict], list[str]]:
    """4 offensive + 4 non-offensive examples, deterministic per run k."""
    examples, ids = [], []
    for j, label in enumerate(("OFF", "NOT")):
        pool = train_subset[train_subset.label == label].sample(
            n=4, random_state=42 + 100 * k + j)
        ids += list(pool.id)
        examples += [{"text": t, "label": l} for t, l in zip(pool.text, pool.label)]
    return examples, ids


def load_model(key: str, token: str | None = None):
    """Load a 4-bit NF4 model. Gemma 4 uses the multimodal auto class."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoModelForMultimodalLM,
                              AutoProcessor, AutoTokenizer, BitsAndBytesConfig)

    model_id = MODEL_IDS[key]
    is_gemma4 = key.startswith("gemma-4")
    if is_gemma4:
        processor = AutoProcessor.from_pretrained(model_id, token=token)
        tok = processor.tokenizer
    else:
        tok = AutoTokenizer.from_pretrained(model_id, token=token)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    cls = AutoModelForMultimodalLM if is_gemma4 else AutoModelForCausalLM
    device_map = {"": 0} if is_gemma4 else "auto"
    start = time.time()
    model = cls.from_pretrained(model_id, quantization_config=quant,
                                device_map=device_map, token=token)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return model, tok, time.time() - start


def score_texts(model, tok, texts, ids, batch: int = 8):
    """Return scores, timing dict, with log P(A) - log P(B) per text."""
    import torch

    scores, n_tokens = [], 0
    start = time.time()
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        n_tokens += int(enc["input_ids"].numel())
        with torch.inference_mode():
            logits = model(**enc).logits[:, -1, :].float()
        a = torch.logsumexp(logits[:, ids["A"]], dim=-1)
        b = torch.logsumexp(logits[:, ids["B"]], dim=-1)
        scores.extend((a - b).cpu().tolist())
    seconds = time.time() - start
    return scores, {"seconds": seconds, "tokens": n_tokens}


def run(model_key: str, train, dev, out_dir: str, mode: str = "fewshot",
        runs: int = 3, batch: int = 8, token: str | None = None, start_run: int = 1):
    """Run zero-shot or few-shot calibration and write one JSON per run."""
    import json

    os.makedirs(out_dir, exist_ok=True)
    model, tok, load_s = load_model(model_key, token)
    ids = option_ids(tok)
    for k in range(start_run, runs + 1):
        if mode == "fewshot":
            from .data import subset
            tr = subset(train, 750, k).reset_index(drop=True)
            examples, ex_ids = select_examples(tr, k)
            cal = tr[~tr.id.isin(ex_ids)].reset_index(drop=True)
            assert len(cal) == 742
            cal_prompts = [make_prompt(tok, t, examples) for t in cal.text]
            dev_prompts = [make_prompt(tok, t, examples) for t in dev.text]
            cal_scores, cal_time = score_texts(model, tok, cal_prompts, ids, batch)
            threshold, cal_f1 = best_threshold(cal.label, cal_scores)
        else:
            cal, cal_scores, cal_time, cal_f1 = None, [], {"seconds": 0, "tokens": 0}, None
            threshold = 0.0
            dev_prompts = [make_prompt(tok, t, []) for t in dev.text]
        dev_scores, dev_time = score_texts(model, tok, dev_prompts, ids, batch)
        pred = ["OFF" if s > threshold else "NOT" for s in dev_scores]
        f1 = macro_f1(dev.label, pred)
        import torch
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        result = {
            "model": model_key, "mode": mode, "run": k, "n_dev": len(dev),
            "threshold": threshold, "cal_macro_f1": cal_f1, "dev_macro_f1": f1,
            "timing": {
                "load_s": load_s, "cal_s": cal_time["seconds"], "dev_s": dev_time["seconds"],
                "cal_tokens": cal_time["tokens"], "dev_tokens": dev_time["tokens"],
                "dev_tweets_per_s": len(dev) / dev_time["seconds"],
                "dev_tokens_per_s": dev_time["tokens"] / dev_time["seconds"],
                "peak_vram_gb": peak,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "batch": batch,
            },
            "dev_pred": dict(zip(dev.id, pred)),
        }
        stem = os.path.join(out_dir, f"{model_key}_{mode}_run{k}")
        with open(stem + ".json", "w") as f:
            json.dump(result, f, indent=2)
        import pandas as pd
        pd.DataFrame({"id": dev.id, "label": dev.label, "score": dev_scores,
                      "pred": pred}).to_csv(stem + "_dev_scores.csv", index=False)
        if cal is not None:
            pd.DataFrame({"id": cal.id, "label": cal.label, "score": cal_scores}).to_csv(
                stem + "_cal_scores.csv", index=False)
        print(f"{model_key} {mode} run {k}: dev={f1:.4f} threshold={threshold:.3f}", flush=True)

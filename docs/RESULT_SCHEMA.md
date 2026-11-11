# Result schema

Every LLM run writes one JSON and one score CSV under
`results/llm/<model>/`. The file name is `fewshot_run<k>.json` or
`zeroshot_run<k>.json`; score files add `_dev_scores.csv` or `_cal_scores.csv`.

```json
{
  "model": "gemma-4-12b",
  "mode": "fewshot",
  "run": 3,
  "n_dev": 1756,
  "threshold": -4.125,
  "cal_macro_f1": 0.8030,
  "dev_macro_f1": 0.7905,
  "timing": {
    "load_s": 92.0,
    "cal_s": 227.1,
    "dev_s": 531.0,
    "cal_tokens": 0,
    "dev_tokens": 0,
    "dev_tweets_per_s": 3.31,
    "dev_tokens_per_s": 0.0,
    "peak_vram_gb": 0.0,
    "gpu": "NVIDIA L4",
    "batch": 6
  },
  "dev_pred": {"<tweet id>": "OFF"}
}
```

Rules:

- `model` matches the directory name.
- `mode` is `fewshot` or `zeroshot`.
- `threshold` is chosen on the 742 non-example labels for `fewshot`; it is 0
  for `zeroshot`.
- `dev_macro_f1` is computed on all 1,756 dev tweets.
- score CSV columns: `id,label,score,pred`.
- Do not edit `results/summary.json` by hand; run `scripts/collect_results.py`.

## Fine-tuning runs

`results/finetune/<model>/n<size>_run<k>.json` with `model`, `hf_id`, `size`,
`run`, `seed`, `n_dev`, `dev_macro_f1`, `dev_off_precision`, `dev_off_recall`,
`single_class`, `steps`, `timing` and `versions`. Score files:
`n<size>_run<k>_dev_scores.csv` (`id,label,score,pred`) and
`n<size>_run<k>_test_scores.csv` (`id,score,pred`, no labels; scored only after
the freeze). `score` is P(OFF); `pred` is OFF when `score ≥ 0.5`.

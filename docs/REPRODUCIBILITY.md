# Reproducibility

## Fixed seeds and versions

| Item | Value |
|---|---|
| Train/dev split | seed 42, stratified, 1,756 dev |
| Subset *k* | seed 42 + k, stratified |
| Bootstrap | seed 42, 10,000 resamples |
| Python | 3.13.16 on Colab; 3.14.8 local |
| scikit-learn | 1.9.1 |
| torch / transformers | 2.14.1 / 5.22.1 |
| Quantization | 4-bit NF4, double quantization, bf16 compute |

## One-command CPU reproduction

```bash
uv venv
uv pip install -e ".[dev]"
bash scripts/reproduce.sh          # CPU
# or: bash scripts/reproduce.sh gpu   # + LLM runs
```

The individual steps are:

```bash
uv run python scripts/run_baseline.py --curve
uv run python scripts/run_test_baseline.py
uv run python scripts/extract_errors.py
uv run python scripts/collect_results.py
uv run pytest -q
```

This downloads the official zip (SHA-256 checked), rebuilds the split and
subsets, the dev/test TF-IDF curves and the 60-error audit.

## LLM reproduction

```bash
export HF_TOKEN=...
uv pip install -e ".[llm]"
uv run python scripts/run_llm.py --model gemma-4-12b --mode fewshot --runs 3 --batch 6
uv run python scripts/run_llm.py --model gemma-4-12b --mode zeroshot --runs 1 --batch 8
uv run python scripts/collect_results.py
```

Each run writes one JSON and one score CSV under `results/llm/<model>/`. A lost
Colab session therefore costs one run, not the experiment.

## Fine-tuning reproduction

```bash
uv pip install -e ".[finetune]"
uv run python scripts/run_finetune.py --model berturk --sizes 250 750 2000 6000 30000
uv run python scripts/run_finetune.py --model xlmr-base --sizes 750
uv run python scripts/collect_results.py
```

fp16 mixed precision on an NVIDIA L4 (Colab). Finished runs are skipped, so a
lost session costs one run. Retraining a saved run gives the same dev macro-F1
to three decimals.

## Expected numbers

`results/summary.json` must match `docs/RESULTS.md`. The baseline check is
`dev macro-F1 = 0.7634` at 30,000 labels and `test macro-F1 = 0.7626` with 95% CI
[0.7454, 0.7793].

## Known constraints

- Gemma-4-26B-A4B needs more than the 22 GB usable on an L4; it is included in
  the model registry for completeness but is not runnable on that device.
- All calibrated models use three runs.
- The test set was scored once for the deterministic baseline; method test runs
  happen only after methods freeze.

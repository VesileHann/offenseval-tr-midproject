#!/usr/bin/env bash
# Reproduce the mid-project numbers.
#   bash scripts/reproduce.sh        # CPU: data, split, baselines, error audit
#   bash scripts/reproduce.sh gpu    # + LLM and fine-tuning runs (needs HF_TOKEN and a GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

$PY scripts/run_baseline.py --curve
$PY scripts/run_test_baseline.py
$PY scripts/extract_errors.py

if [[ "${1:-}" == "gpu" ]]; then
  : "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token with access to the gated models}"
  for model in qwen3-8b gemma-2-9b; do
    $PY scripts/run_llm.py --model "$model" --mode fewshot  --runs 3 --batch 8
    $PY scripts/run_llm.py --model "$model" --mode zeroshot --runs 1 --batch 8
  done
  for model in qwen3-14b gemma-4-12b; do
    $PY scripts/run_llm.py --model "$model" --mode fewshot  --runs 3 --batch 6
    $PY scripts/run_llm.py --model "$model" --mode zeroshot --runs 1 --batch 8
  done
  $PY scripts/run_finetune.py --model berturk --sizes 250 750 2000 6000 30000
  $PY scripts/run_finetune.py --model xlmr-base --sizes 750
fi

$PY scripts/collect_results.py

PY ?= uv run python

.PHONY: install baseline test audit llm finetune collect test-all clean

install:
	uv venv
	uv pip install -e ".[dev]"

baseline:
	$(PY) scripts/run_baseline.py --curve
	$(PY) scripts/run_test_baseline.py

audit:
	$(PY) scripts/extract_errors.py
	$(PY) scripts/collect_results.py

llm:
	$(PY) scripts/run_llm.py --model qwen3-8b   --mode fewshot  --runs 3 --batch 8
	$(PY) scripts/run_llm.py --model qwen3-8b   --mode zeroshot --runs 1 --batch 8
	$(PY) scripts/run_llm.py --model gemma-2-9b --mode fewshot  --runs 3 --batch 8
	$(PY) scripts/run_llm.py --model gemma-2-9b --mode zeroshot --runs 1 --batch 8
	$(PY) scripts/collect_results.py

finetune:
	$(PY) scripts/run_finetune.py --model berturk --sizes 250 750 2000 6000 30000
	$(PY) scripts/run_finetune.py --model xlmr-base --sizes 750
	$(PY) scripts/collect_results.py

collect:
	$(PY) scripts/collect_results.py

test:
	uv run --extra dev python -m pytest -q

reproduce:
	bash scripts/reproduce.sh

reproduce-gpu:
	bash scripts/reproduce.sh gpu

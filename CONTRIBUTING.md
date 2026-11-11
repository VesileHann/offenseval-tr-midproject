# Contributing

This repository is the shared Team 10 workspace. Teammates and their agents
should work on a branch and open a pull request; do not push directly to `main`.

## Setup

```bash
git clone https://github.com/paysis/offenseval-tr-midproject.git
cd offenseval-tr-midproject
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
```

The data zip is downloaded automatically by the scripts and is **not** committed.
Never commit `secrets.env`, Hugging Face tokens, model weights (including
`*.safetensors`, `*.bin`, `*.gguf`) or the 2 MB data zip.

## Branches and commits

- Branch names: `feat/<topic>`, `fix/<topic>`, `results/<model>`, `docs/<topic>`.
- One concern per commit. Small, reviewable commits only.
- Do not force-push a branch that someone else is reviewing.
- Open a PR against `main`; CI runs the CPU test suite.

## What to run before a PR

```bash
uv run pytest -q
```

If you change the split, baselines or metrics, also run:

```bash
bash scripts/reproduce.sh
```

If you add LLM runs, use the existing protocol and write results in the standard
layout (see `docs/RESULT_SCHEMA.md`):

```bash
export HF_TOKEN=...
uv run python scripts/run_llm.py --model <model> --mode fewshot --runs 3 --batch 6
uv run python scripts/collect_results.py
```

`results/summary.json` is generated, never hand-edited.

## Review checklist

- [ ] One concern; no unrelated files.
- [ ] `uv run pytest -q` passes.
- [ ] New results follow `docs/RESULT_SCHEMA.md`.
- [ ] `results/summary.json` regenerated with `scripts/collect_results.py`.
- [ ] No secrets, data zip, model weights or large temporary files.
- [ ] Numbers in docs/README match `results/summary.json`.
- [ ] The fixed test set was not used for tuning.

## Agents

Agents must follow the same rules. In a PR description, include the exact
commands run and their final output. Do not rewrite `main`, do not amend someone
else's commits, and do not add generated files that the scripts can rebuild.

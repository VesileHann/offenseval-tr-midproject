## What changed

<!-- one or two sentences -->

## Why

<!-- link the issue or describe the gap -->

## Verification

- [ ] `uv run pytest -q`
- [ ] `bash scripts/reproduce.sh` (if baseline/metrics changed)
- [ ] `uv run python scripts/collect_results.py` (if results changed)
- [ ] No secrets, data zip or model weights committed
- [ ] Docs/README numbers match `results/summary.json`

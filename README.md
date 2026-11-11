# OffensEval-TR: fine-tuning vs threshold-calibrated open LLMs with 750 labels

Team 10 · CSE 484 / CSE 654 Natural Language Processing · Fall 2026

This repository contains the code, fixed data splits, saved predictions, error
audit and reproduction scripts for our mid-project study on Turkish offensive
language identification.

**Question.** With only 750 labelled tweets, does fine-tuning BERTurk beat an
open LLM that uses the same 750 labels — 8 as prompt examples and the other 742
to choose its decision threshold — on macro-F1 over the official OffensEval-TR
2020 test set?

**Status.** Mid-project. The baseline, the no-fine-tuning LLM arm and the
fine-tuning arm (BERTurk learning curve and Ablation A) are complete on dev.
Method test scores wait for the final freeze.

## Key results (dev, 1,756 tweets)

| System | Zero-shot | Calibrated | Calibration Δ | Runs |
|---|---:|---:|---:|---:|
| TF-IDF + LR, 750 labels | — | 0.613 ± 0.015 | — | 10 |
| Qwen3-8B | 0.162 | 0.628 ± 0.034 | +0.466 | 3 |
| Gemma-2-9B | 0.412 | 0.500 ± 0.048 | +0.088 | 3 |
| Qwen3-14B | 0.171 | **0.773 ± 0.010** | +0.602 | 3 |
| **Gemma-4-12B** | **0.763** | **0.790 ± 0.007** | +0.027 | 3 |

**Fine-tuning arm (dev):**

| System | 250 | 750 | 2,000 | 6,000 | 30,000 |
|---|---:|---:|---:|---:|---:|
| BERTurk | 0.680 ± 0.052 | **0.767 ± 0.014** | 0.793 ± 0.006 | 0.806 ± 0.004 | 0.832 ± 0.003 |
| XLM-R-base (Ablation A) | — | 0.670 ± 0.025 | — | — | — |
| TF-IDF + LR | 0.557 ± 0.020 | 0.613 ± 0.015 | 0.673 ± 0.009 | 0.727 ± 0.005 | 0.763 |

10 runs at 250/750/2,000 labels, 3 at 6,000/30,000. At 750 labels BERTurk is
above Qwen3-8B and Gemma-2-9B (the two LLMs named in the proposal) and below
Qwen3-14B and Gemma-4-12B on dev. Ablation A: BERTurk − XLM-R = +0.097 (paired
bootstrap over dev tweets, 95% CI [+0.080, +0.116]).

**Frozen test baseline:** TF-IDF with 30,000 labels reaches **0.763** macro-F1
(95% bootstrap CI [0.745, 0.779]); with 750 labels it reaches 0.620 ± 0.012.
The deterministic baseline was scored once on test; method results stay on dev
until the final freeze.

## Method

- **Data:** official OffensEval-TR 2020 release (Çöltekin, LREC 2020), annotations
  CC BY 2.0. 31,756 official training tweets split into 30,000 train / 1,756 dev
  (stratified, seed 42); 3,528 official test tweets.
- **Fine-tuning arm:** BERTurk with a classification head; 128 tokens, batch 16,
  lr 2e-5 with 10% warm-up, class-weighted loss, 4 epochs or 400 updates
  (whichever is longer), no early stopping. Run *k* uses subset *k* and seed 42 + k.
- **LLM arm:** 4-bit NF4 open decoders. Turkish instructions, 8 labelled examples
  (4 offensive, 4 not), answer options A/B. The model does not generate: we read
  `log P(A) − log P(B)` at the first answer token. The threshold maximises
  macro-F1 on the other 742 labelled tweets and is applied once to dev.
- **Protocol:** macro-F1; 10 BERTurk runs at 250/750/2,000 labels and 3 at
  6,000/30,000; 3 LLM runs per model; paired bootstrap with
  10,000 resamples; a gap counts only if it is at least 0.02 and BERTurk wins on
  at least 95% of resamples.

## Throughput (one L4, 4-bit)

| System | Dev pass | Throughput | Peak VRAM |
|---|---:|---:|---:|
| TF-IDF, 750 / 30,000 | 0.57 s | ~3,100 tweets/s | CPU |
| Qwen3-8B, few-shot | 355 s | 5.0 tweets/s | — |
| Gemma-2-9B, few-shot | 430 s | 4.1 tweets/s | — |
| Qwen3-14B, few-shot | 644 s | 2.7 tweets/s | — |
| Gemma-4-12B, few-shot | 579 s | 3.0 tweets/s | — |
| Qwen3-14B, zero-shot | 180 s | 9.8 tweets/s | 10.7 GB |

Few-shot scoring is 3–4× slower than zero-shot because each prompt repeats eight
examples (~600 tokens). Gemma-4-26B-A4B exceeds 22 GB on an L4 and is not
runnable on that device.

## Repository layout

```
src/offenseval_tr/        importable package
  data.py                 official zip loader, fixed split, subsets
  metrics.py              macro-F1, bootstrap CI, paired bootstrap, threshold
  baselines.py            majority and TF-IDF + logistic regression
  llm.py                  prompt construction, 4-bit scoring, calibration
  finetune.py             BERTurk / XLM-R fine-tuning on the shared subsets
scripts/                  thin command-line entry points
  run_baseline.py         dev curve and saved split/subsets
  run_test_baseline.py    frozen-test baseline with CI
  extract_errors.py       275 dev errors + reproducible 60-error audit
  run_llm.py              zero-shot / few-shot calibration
  run_finetune.py         fine-tuning runs (learning curve, Ablation A)
  collect_results.py      aggregate results/summary.json
results/
  baseline/               curves, ids, subsets, test predictions
  llm/<model>/            one JSON + score CSV per run
  finetune/<model>/       one JSON + dev/test score CSV per run
  errors/                 dev errors and audit
  figures/                deck figures
  summary.json            every number used in the report
docs/                     method, results, error analysis, Project Card, team
paper/                    Team10_MidProject.pdf and editable pptx
tests/                    CPU-only unit tests
```

## Quickstart

```bash
uv venv
uv pip install -e ".[dev]"

# One command for the CPU pipeline
bash scripts/reproduce.sh

# Or run the steps individually
uv run python scripts/run_baseline.py --curve
uv run python scripts/run_test_baseline.py
uv run python scripts/extract_errors.py
uv run python scripts/collect_results.py
uv run pytest -q
```

The official zip is downloaded automatically and checked against SHA-256
`7977e962…c9c2f2`. The zip is not committed.

### LLM arm (GPU, 4-bit)

```bash
export HF_TOKEN=...
uv pip install -e ".[llm]"
uv run python scripts/run_llm.py --model gemma-4-12b --mode fewshot --runs 3 --batch 6
uv run python scripts/run_llm.py --model gemma-4-12b --mode zeroshot --runs 1 --batch 8
uv run python scripts/collect_results.py
```

Models: `qwen3-8b`, `gemma-2-9b`, `qwen3-14b`, `gemma-4-12b`,
`gemma-4-26b-a4b` (does not fit a 24 GB L4). Runs write one JSON and one score
CSV per run, so a lost session costs one run, not the experiment.

### Fine-tuning arm (GPU)

```bash
uv pip install -e ".[finetune]"
uv run python scripts/run_finetune.py --model berturk --sizes 250 750 2000 6000 30000
uv run python scripts/run_finetune.py --model xlmr-base --sizes 750
uv run python scripts/collect_results.py
```

One run at 750 labels takes about 35 s of training on an L4; one at 30,000 about
11 minutes. Test predictions are saved in `*_test_scores.csv` but not scored.

## Contributing

See `CONTRIBUTING.md` for branch/PR rules, the result schema and the CI
commands. Teammates work on branches and open pull requests; `main` is kept
reproducible.

## Data and licence

- **Data:** [OffensEval-TR 2020](https://coltekin.github.io/offensive-turkish/),
  Çöltekin, LREC 2020. Annotations are **CC BY 2.0**; the tweets remain subject
  to Twitter/X terms. The data is not redistributed here.
- **Code:** MIT (see `LICENSE`).
- **Model licences:** Qwen3 (Apache 2.0), Gemma 2/4 (Apache 2.0; Gemma 2 access
  is gated on Hugging Face).

## Team

Team 10: Vesile Han, Gürkan Göztepeli, Mert Uğur, Utku Gökçek.
Task ownership will be finalized by the team before the final report.

**AI disclosure.** A coding assistant was used for code, analysis and drafting.
Every number in `results/summary.json` is generated by the scripts in this
repository and was checked by the team.

## Citation

```bibtex
@misc{team10_offensevaltr_2026,
  title  = {Fine-tuning vs threshold-calibrated open LLMs for Turkish offensive language with 750 labels},
  author = {Han, Vesile and G{\"o}ztepeli, G{\"u}rkan and U{\u{g}}ur, Mert and G{\"o}k{\c{c}}ek, Utku},
  year   = {2026},
  note   = {CSE 484 / CSE 654 mid-project, Team 10}
}
```

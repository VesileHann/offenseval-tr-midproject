# Results

All method results are on the 1,756-tweet dev set. The only test-set number is
the deterministic baseline, scored once because the mid-project rubric asks for
it. `results/summary.json` is the authoritative machine-readable table.

## Frozen-test baseline

| System | Test macro-F1 | Spread | OFF precision | OFF recall |
|---|---:|---|---:|---:|
| Always “not offensive” | 0.500 | deterministic | — | 0.000 |
| TF-IDF + LR, 750 labels | 0.620 | sd 0.012 over 10 samples | — | — |
| TF-IDF + LR, 30,000 labels | **0.763** | 95% CI [0.745, 0.779] | 0.589 | 0.673 |

Test curve means at 250 / 750 / 2,000 / 6,000 / 30,000: 0.549 / 0.620 / 0.677 /
0.727 / 0.763.

## Dev: no-fine-tuning arm

| System | Zero-shot | Run 1 | Run 2 | Run 3 | Mean ± sd | Δ calibration |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 0.162 | 0.637 | 0.658 | 0.591 | 0.628 ± 0.034 | +0.466 |
| Gemma-2-9B | 0.412 | 0.502 | 0.547 | 0.452 | 0.500 ± 0.048 | +0.088 |
| Qwen3-14B | 0.171 | 0.779 | 0.778 | 0.762 | 0.773 ± 0.010 | +0.602 |
| Gemma-4-12B | **0.763** | 0.797 | 0.791 | 0.784 | **0.790 ± 0.007** | +0.027 |

Thresholds chosen on 742 labels: Qwen3-8B 3.13/2.75/2.50; Gemma-2-9B
0.63/0.25/0.88; Qwen3-14B 4.31/4.69/7.50; Gemma-4-12B −5.00/−4.13/−5.13.

## Interpretation

- Threshold calibration is what makes the smaller Qwen3 arm usable: Qwen3-8B is
  worse than Gemma-2-9B zero-shot, then overtakes it after calibration.
- Gemma-4-12B is the strongest no-fine-tuning system: 0.763 zero-shot and 0.790
  calibrated, above the full-data TF-IDF baseline.
- Tokenizer fertility does not explain the Qwen-8B advantage. On 500 dev tweets:
  Qwen3 37.9, Gemma-2 32.5, Gemma-4 32.3 tokens/tweet. Gemma is more compact.
- Plausible remaining causes are pretraining language and score geometry; both
  are confounded with alignment. This is a hypothesis, not a controlled
  language-pretraining ablation.

## Throughput

See `README.md` and `results/timing_from_logs.json`. Few-shot scoring is 3–4×
slower than zero-shot because each prompt repeats 8 examples (~600 tokens).
Gemma-4-26B-A4B exceeds the 22 GB usable on an L4 and is not runnable there.

## Pending

BERTurk-750 (10 runs), BERTurk-30k (3 runs), Ablation A (XLM-R) and Ablation B
(English instructions). The final comparison uses the frozen test set and the
pre-registered paired bootstrap.

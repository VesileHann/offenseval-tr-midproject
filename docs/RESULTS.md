# Results

All method results are on the 1,756-tweet dev set. The only test-set number is
the deterministic baseline, scored once because the mid-project rubric asks for
it. `results/summary.json` is the authoritative machine-readable table.

## Frozen-test baseline

| System | Test macro-F1 | Spread | OFF precision | OFF recall |
|---|---:|---|---:|---:|
| Always “not offensive” | 0.444 | deterministic | — | 0.000 |
| TF-IDF + LR, 750 labels | 0.620 | sd 0.012 over 10 samples | — | — |
| TF-IDF + LR, 30,000 labels | **0.763** | 95% CI [0.745, 0.779] | 0.589 | 0.673 |

Test curve means at 250 / 750 / 2,000 / 6,000 / 30,000: 0.549 / 0.620 / 0.677 /
0.727 / 0.763.

## Dev: no-fine-tuning arm

| System | Zero-shot | Run 1 | Run 2 | Run 3 | Mean ± sd | Δ calibration |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 0.162 | 0.637 | 0.658 | 0.591 | 0.628 ± 0.034 | +0.466 |
| Gemma-2-9B | 0.412 | 0.502 | 0.547 | 0.452 | 0.500 ± 0.048 | +0.088 |
| Qwen3-14B | 0.171 | 0.779 | 0.778 | 0.762 | 0.773 ± 0.009 | +0.602 |
| Gemma-4-12B | **0.763** | 0.797 | 0.791 | 0.784 | **0.790 ± 0.007** | +0.027 |

Thresholds chosen on 742 labels: Qwen3-8B 3.13/2.75/2.50; Gemma-2-9B
0.63/0.25/0.88; Qwen3-14B 4.31/4.69/7.50; Gemma-4-12B −5.00/−4.13/−5.13.

## Dev: fine-tuning arm

| System | Labels | Runs | Mean ± sd | Min–max | OFF recall |
|---|---:|---:|---:|---:|---:|
| BERTurk | 250 | 10 | 0.680 ± 0.052 | 0.585–0.746 | 0.38 |
| BERTurk | 750 | 10 | **0.767 ± 0.014** | 0.739–0.790 | 0.57 |
| BERTurk | 2,000 | 10 | 0.793 ± 0.006 | 0.784–0.800 | 0.65 |
| BERTurk | 6,000 | 3 | 0.806 ± 0.004 | 0.802–0.810 | 0.62 |
| BERTurk | 30,000 | 3 | 0.832 ± 0.003 | 0.829–0.834 | 0.68 |
| XLM-R-base (Ablation A) | 750 | 10 | 0.670 ± 0.025 | 0.629–0.697 | 0.43 |

No run predicted a single class. **Ablation A** (same 10 subsets of 750, same
seeds and settings): BERTurk − XLM-R-base = +0.097; paired bootstrap over the
1,756 dev tweets, 10,000 resamples, run-averaged: 95% CI [+0.080, +0.116],
BERTurk wins on 100% of resamples. Training time per run on one L4: 35 s at 750
labels, about 29 min at 30,000 (1,657–1,724 s; `timing.train_s`) because the three
30,000-label runs shared one L4; alone a run takes about 11 min (7,500 updates at
the 0.087 s/update measured at 750 labels).

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

- At 750 labels, fine-tuned BERTurk (0.767) is above Qwen3-8B (0.628) and
  Gemma-2-9B (0.500), the two LLMs named in the proposal, and below Qwen3-14B
  (0.773) and Gemma-4-12B (0.790). These are dev numbers; the decision uses test.
- BERTurk at 750 labels matches TF-IDF at 30,000 (0.763). The curve flattens after
  2,000 labels, and run-to-run spread is largest at 250.

## Pending

Ablation B (English instructions), then the method freeze and one scoring pass
on the frozen test set with the pre-registered paired bootstrap. BERTurk and
XLM-R test predictions are already saved in `results/finetune/*/*_test_scores.csv`.

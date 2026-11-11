# Project Card — Team 10 (CSE 484 / CSE 654, Fall 2026)

## 1. Title and abstract

**Working title:** Fine-tuning BERTurk vs threshold-calibrated open LLMs for Turkish offensive
language with 750 labels.

Small Turkish platforms can label only a few hundred tweets. We ask whether fine-tuning BERTurk on 750 labelled OffensEval-TR tweets beats an open LLM that uses the same 750 labels as 8 prompt examples and 742 calibration tweets for its decision threshold. The TF-IDF + logistic-regression baseline scores 0.763 macro-F1 on the frozen 3,528-tweet test set, but 0.620 with 750 labels. The LLM arm scores each tweet as log P(A) − log P(B); the 742 labels set its threshold. On dev, calibration moves Qwen3-8B from 0.162 to 0.628 and Gemma-2-9B from 0.412 to 0.500; Qwen3-14B reaches 0.773 (zero-shot 0.171) and Gemma-4-12B 0.790 ± 0.007 (zero-shot 0.763). Fine-tuned BERTurk reaches 0.767 ± 0.014 on dev with 750 labels. **Gap:** the frozen-test comparison is pending; the final result uses the paired bootstrap. The final paper reports the paired bootstrap gap and the label count at which fine-tuning overtakes the LLM line.

## 2. Research question

Unchanged from the proposal. With 750 labelled tweets, does fine-tuning BERTurk beat prompting
Gemma-2-9B-it or Qwen3-8B that uses the same 750 tweets (8 examples, 742 threshold labels), on
macro-F1 over the official 3,528-tweet test set?

**“No” result:** against either LLM, BERTurk’s mean over 10 runs is not ≥0.02 higher than the
LLM’s mean over 3 runs, or the paired bootstrap says the gap could be chance (p ≥ 0.05).

**Change:** the question is unchanged, but two stronger LLMs (Qwen3-14B, Gemma-4-12B) were added
after the first dev results. Decided on 11 Nov 2026, before the test set opens: the formal test
(0.02 gap, paired bootstrap) compares BERTurk only with Gemma-2-9B and Qwen3-8B, the two LLMs in
the proposal; the new models are reported as extra analysis. Compute changed from free Colab T4
to paid Colab Pro L4, for speed. The mid-project rubric added a test-set baseline, reported
separately, which does not alter the question.

## 3. Contributions

- **C1.** At 750 labels, we test whether BERTurk’s mean test macro-F1 is at least 0.02 above each
  LLM’s and whether a paired bootstrap rejects chance. Test pending. On dev, BERTurk (0.767) is above
  Qwen3-8B (0.628) and Gemma-2-9B (0.500).
- **C2.** Choosing the threshold on the 742 non-example labels improves each LLM’s dev macro-F1
  over zero-shot by a measurable margin; we report the per-model delta and its run spread.
- **C3.** The learning curve tests where fine-tuning overtakes the LLM line between 250 and
  6,000 labels; if the lines never cross, the 750-label answer still stands.

## 4. Related work

1. **Zampieri et al. (2020).** SemEval-2020 Task 12: Multilingual Offensive Language
   Identification in Social Media (OffensEval 2020). Top Turkish systems fine-tune transformers
   on the full training data; no label-budget comparison.
2. **Çöltekin (2020).** A Corpus of Turkish Offensive Language. LREC 2020. The dataset we use;
   no fine-tuning-vs-prompting comparison.
3. **Zampieri et al. (2023).** OffensEval 2023: Offensive Language Identification in the Age of
   Large Language Models. NLE. Zero-shot LLMs; Turkish only Flan-T5-large (0.451); no examples
   or threshold calibration.
4. **Dehghan & Yanıkoğlu (2024).** Evaluating ChatGPT’s Ability to Detect Hate Speech in Turkish
   Tweets. CASE 2024, pp. 54–59. ChatGPT with 7 examples vs fine-tuned BERTurk, one dataset and
   one training size.
5. **Pecher, Srba & Bielikova (2025).** Comparing Specialised Small and General Large Language
   Models on Text Classification. EMNLP 2025, pp. 165–184. English break-even near 100 labels;
   we test a noisy Turkish two-class task with run variance.
6. **Schmitt & Schweter (2025).** SindBERT, arXiv:2510.21364. Turkish encoders on OffensEval-TR
   with full data only; no LLMs or small label budgets.
7. **Şeker, Uludoğan, Önal & Özgür (2025).** HATECAT-TR, Findings of EMNLP 2025. Turkish
   span-level hate; BERTurk 0.59 vs LLaMA-3.3-70B 0.53. We classify whole tweets and score
   probabilities.
8. **Karakaş & Şimşek (2026).** arXiv:2606.29614. Zero-shot LLMs vs BERTurk on Turkish sentiment
   at one size; we add examples, calibration and a label curve.

## 5. Data

OffensEval-TR 2020 (Çöltekin, LREC 2020), https://coltekin.github.io/offensive-turkish/.
Annotations CC BY 2.0. Turkish tweets; @USER/URL already masked in the release.

| Split | Tweets | NOT | OFF | OFF % |
|---|---:|---:|---:|---:|
| Official train | 31,756 | 25,625 | 6,131 | 19.3 |
| Our train (seed 42) | 30,000 | 24,208 | 5,792 | 19.3 |
| Dev | 1,756 | 1,417 | 339 | 19.3 |
| Test (frozen) | 3,528 | 2,812 | 716 | 20.3 |

After preprocessing: only the released @USER/URL masking; no lowercasing or deletion. 9 duplicate
texts in train; 1 test/train overlap. Example (OFF, id 33518): “Kırmızı ağır karar Allah belanı
versin” (“A red card is a harsh call, God damn you”). No labelled data of our own; no agreement
study needed.

## 6. Method and baseline

**Baseline as run:** TF-IDF word 1–2-grams + char_wb 2–5-grams, min_df=2, sublinear, Turkish-aware
lowercasing; logistic regression C=1.0, class_weight=balanced, max_iter=5000. Test 0.763
[0.745, 0.779].

**Method implemented:** Qwen3-8B and Gemma-2-9B in 4-bit; Turkish instructions, A/B answer
options, 8 examples (4+4), score log P(A) − log P(B); threshold maximises macro-F1 on the other
742 labels; dev evaluation. BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuned on every subset:
128 tokens, batch 16, lr 2e-5, 10% warm-up, class-weighted loss, 4 epochs or 400 updates, no early
stopping; 36 runs (10 at 250/750/2,000, 3 at 6,000/30,000) plus 10 XLM-R-base runs (Ablation A).

**One change vs baseline:** labels train weights vs choose prompt examples and a threshold.

**Compute used / needed:** about 2.5 L4-hours (9.4 Colab units) for 16 LLM passes and about 3 Colab
units on L4 for the 46 fine-tuning runs; about 18 GPU-hours needed
for the final (proposal estimate), free Colab only.

## 7. Evaluation protocol

Macro-F1; frozen 3,528-tweet test; 10 BERTurk runs at 250/750/2,000 and 3 at 6,000/30,000; 3 LLM
runs at 750. Paired bootstrap, 10,000 resamples: a gap is real only if BERTurk wins on ≥95% and
the mean gap is ≥0.02. Fixed before method results; dev is for code checks only.

## 8. Results so far

| System | Dev macro-F1 | Test macro-F1 |
|---|---:|---:|
| Always NOT | 0.447 | 0.444 |
| TF-IDF, 750 | 0.613 ± 0.015 | 0.620 ± 0.012 |
| TF-IDF, 30,000 | 0.763 [0.739, 0.787] | 0.763 [0.745, 0.779] |
| Qwen3-8B zero-shot | 0.162 | — |
| Qwen3-8B calibrated | 0.628 ± 0.034 | — |
| Gemma-2-9B zero-shot | 0.412 | — |
| Gemma-2-9B calibrated | 0.500 ± 0.048 | — |
| Qwen3-14B zero-shot | 0.171 | — |
| Qwen3-14B calibrated | 0.773 ± 0.009 | — |
| Gemma-4-12B zero-shot | 0.763 | — |
| Gemma-4-12B calibrated | 0.790 ± 0.007 | — |
| BERTurk, 250 | 0.680 ± 0.052 | — |
| BERTurk, 750 | 0.767 ± 0.014 | — |
| BERTurk, 2,000 | 0.793 ± 0.006 | — |
| BERTurk, 6,000 | 0.806 ± 0.004 | — |
| BERTurk, 30,000 | 0.832 ± 0.003 | — |
| XLM-R-base, 750 (Ablation A) | 0.670 ± 0.025 | — |

## 9. Error analysis

275 dev errors (166 false OFF, 109 missed OFF); 60 audited (30 per direction, seed 42).

**False alarms:** no insult / spurious cue 9; political or religious criticism 10; sports
criticism or banter 5; personal criticism 3; sarcasm 1; label noise 2.

**Misses:** implicit insult 8; masked profanity 5; veiled threat 3; political/religious insult 4;
institutional or sports criticism 3; label noise 7.

Examples: 35124 “Hırsız tabloyu çalarken…” (false alarm, political); 43325 “…parazit canlı türü
yok” (miss, implicit); 15069 “…amkk” (miss, masked). Lesson: conflict vocabulary drives false
alarms; implicit insults and masked profanity drive misses; about 15% is label noise, so the
ceiling is below 1.0.

## 10. Research lead section

**Vesile Han (CSE 654), fine-tuning side.** (1) Pecher, Srba & Bielikova, EMNLP 2025: small
fine-tuned models catch up with general LLMs after about 100 labels on average, but need more on
two-class tasks and once run variance is counted; all eight datasets are English. We test
Turkish, a suffix-heavy language, on a noisy two-class task with 19.3% offensive tweets.
(2) Schmitt & Schweter, SindBERT, arXiv:2510.21364 (2025): compare Turkish encoders on
OffensEval-TR only with full training data, no LLMs and no small sizes. We add both. **Ablation
A:** BERTurk vs multilingual XLM-R-base at 750 labels, 10 runs each; done on dev: +0.097, paired
bootstrap 95% CI [+0.080, +0.116]. **Published comparison:** KUISAIL reports plain fine-tuned
BERTurk at 0.816 on this test set; our 30,000-label runs score 0.832 ± 0.003 on dev, test after
the freeze.

**Gürkan Göztepeli (CSE 654), prompting side.** (1) Şeker, Uludoğan, Önal & Özgür, HATECAT-TR,
Findings of EMNLP 2025: fine-tuned BERTurk (0.59 F1) beat LLaMA-3.3-70B with a few examples
(0.53) on Turkish hate-span detection; one training set, span labels. We classify whole tweets,
vary the label budget, and read probabilities so every answer is valid. (2) Karakaş & Şimşek,
arXiv:2606.29614 (2026): zero-shot open LLMs vs fine-tuned BERTurk on Turkish review sentiment
at one size. We add examples, threshold calibration and a label curve. **Ablation B:**
Gemma-2-9B with English instructions, same examples and threshold method, 3 runs.

## 11. Plan and risk

Done: BERTurk learning curve (36 runs) and Ablation A XLM-R (10 runs) on dev. 21–24 Nov:
Ablation B English instructions. 25–27 Nov: zero-shot + contamination check. 28–30 Nov: freeze
methods, open test, final runs. 1–5 Dec: bootstrap, error analysis, paper. 8 Dec: final report.
Owners to be confirmed.

**Biggest risk:** a 26B+ model may not fit the L4; Gemma-4-26B-A4B OOM’d at 22 GB, so we
used the 12B dense model. **Fallback:** Qwen3-14B alone. **Result risk:** the lines may never cross; the 750-label
answer still stands. **Contamination:** give each LLM the first 100 test tweets and report exact
completions. **Proposal feedback:** the rubric asked for a test-set baseline; we scored the
deterministic baseline once and kept methods on dev.

## 12. Reproducibility

Repository: https://github.com/paysis/offenseval-tr-midproject. One command:
`bash scripts/reproduce.sh gpu`. Seeds: split 42, subsets 42 + k, bootstrap 42. Versions:
Python 3.13.16 on Colab (3.14.8 local), scikit-learn 1.9.1, transformers 5.22.1, torch 2.14.1,
bitsandbytes 4-bit NF4.

**Roles:** to be confirmed before the final report; the team will assign them.

**AI tools:** a coding assistant was used for code, analysis and drafting; every number in this
card is read from the saved result files and was checked by the team.

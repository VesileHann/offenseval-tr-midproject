# Method

## Research question

With only 750 labelled tweets, does fine-tuning BERTurk give a higher macro-F1
on the official OffensEval-TR 2020 test set (3,528 tweets) than prompting an open
LLM (Gemma-2-9B-it or Qwen3-8B) that uses the same 750 tweets: 8 as prompt
examples and the other 742 to choose a decision threshold?

**Falsification rule (fixed before method results).** The answer is “no” if,
against either LLM, BERTurk’s mean over 10 runs is not at least 0.02 higher than
the LLM’s mean over 3 runs, or a paired bootstrap test says the gap could be
chance (one-sided 95%; 10,000 resamples).

## Data

Official OffensEval-TR 2020 release. 31,756 training tweets are split,
stratified with seed 42, into 30,000 train and 1,756 dev. The 3,528 test tweets
are frozen. Subset *k* of size *n* uses seed `42 + k`; all systems use the same
saved subsets.

## Fine-tuning arm (pending)

BERTurk (`dbmdz/bert-base-turkish-cased`) with a classification head. Tweets are
cut to 128 tokens; batch 16; learning rate 2e-5 with 10% warm-up; class-weighted
cross-entropy; 4 epochs or 400 updates, whichever is longer; no early stopping.
A run that predicts one class is reported, not rerun.

## LLM arm (implemented)

The model never writes a free answer. The prompt contains Turkish instructions,
8 labelled examples (4 offensive, 4 not) and answer options `A) saldırgan`,
`B) saldırgan değil`. We read the logits at the first answer token and define

```
score(tweet) = log P(A) − log P(B)
```

A tweet is labelled offensive when `score > threshold`. The threshold maximises
macro-F1 on the 742 tweets of the same 750 that are not prompt examples. A
zero-shot run uses no examples and threshold 0.

Both sides receive the same 750 labels. The only difference is how the labels
are used: to update weights (BERTurk) or to choose examples and a threshold
(LLM).

## Controlled comparisons

- **Ablation A (planned):** BERTurk vs multilingual XLM-R-base, 750 labels,
  10 runs each, to test whether Turkish-only pretraining explains BERTurk.
- **Ablation B (planned):** Gemma-2-9B with English instructions, same examples
  and threshold method, 3 runs, to test instruction-language sensitivity.
- **Scale check (done):** Qwen3-8B/14B and Gemma-2-9B/Gemma-4-12B under the
  identical 8+742 protocol. This tests whether a stronger multilingual model
  closes the gap without fine-tuning.

## Statistics

- **Training luck:** different 750-tweet samples and seeds. Report mean ± sd.
- **Test-set luck:** paired bootstrap over the 3,528 test tweets, 10,000
  resamples. The gap is real only if the one-sided 95% lower bound is above zero
  and the observed gap is at least 0.02.
- No setting is chosen on dev; dev is for code checks and mid-project numbers.
  The test set opens only after methods are frozen.

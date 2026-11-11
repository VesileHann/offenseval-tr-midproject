# Error analysis

TF-IDF + logistic regression trained on 30,000 tweets makes **275 dev errors**:
166 false “offensive” alarms and 109 missed offensive tweets. We audited 30
errors per direction (seed 42, 60 total); counts below are counts in that audit.
The full 275 errors are in `results/errors/dev_errors.tsv`, the audit in
`results/errors/audit.json`.

## False alarms (166 total; 30 audited)

| Category | n | Example |
|---|---:|---|
| No insult / spurious cue | 9 | 12420 — a fan poll; no insult |
| Political or religious criticism | 10 | 35124 — “Hırsız…” (“like a thief”) |
| Sports criticism or banter | 5 | 13105 — “padişahlık zannediyor” |
| Personal criticism not judged offensive | 3 | 22409 — “fake hesapları sevmiyorum” |
| Sarcasm or irony | 1 | 28737 — “Ha bi sen eksiktin” |
| Label noise or genuinely ambiguous | 2 | 33218 — “Seni cehennem paklar” |

## Misses (109 total; 30 audited)

| Category | n | Example |
|---|---:|---|
| Implicit insult, no profanity | 8 | 43325 — “parazit canlı türü yok” |
| Masked or abbreviated profanity | 5 | 15069 — “…kaldı amkk” |
| Veiled threat or violence | 3 | 35789 — “idam edicem” |
| Political or religious insult | 4 | 25929 — “Gerici yobazlar” |
| Institutional or sports criticism | 3 | 20333 — “ÖSYM… güvenmeyebilir” |
| Label noise or not offensive | 7 | 10001 — a joke; no offence |

**What this suggests.** The model reacts to conflict vocabulary and character
n-grams rather than to whether a person is insulted. It misses implicit insults
and masked swearing. About 15% of the audit is label noise or genuinely
ambiguous, so the achievable ceiling is below 1.0. This motivates using
implicit-insult and masked-profanity examples in the prompt, not tuning on dev.

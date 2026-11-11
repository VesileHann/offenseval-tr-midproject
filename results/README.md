# Results

| Path | Content |
|---|---|
| `summary.json` | Every number used in the report: baseline curves, error audit, LLM means, thresholds, timing |
| `baseline/dev_curve.json` | TF-IDF dev curve and 30,000-label score |
| `baseline/test_curve.json` | Frozen-test baseline and 10,000-resample CI |
| `baseline/test_predictions.csv` | One prediction per official test tweet |
| `baseline/{train,dev,test}_ids.csv` | The fixed split |
| `baseline/subsets.csv` | Shared training subsets, size × subset × id |
| `errors/dev_errors.tsv` | All 275 dev errors |
| `errors/audit.json` | 60-error audit: categories, counts, one example each |
| `llm/<model>/` | One JSON and score CSV per run; JSON has threshold, F1 and timing |
| `figures/` | Figures used in the deck |
| `timing_tfidf.json`, `timing_from_logs.json` | CPU latency and per-pass GPU throughput |

The test set was scored once for the deterministic TF-IDF baseline. All method
results are on dev until the final freeze.

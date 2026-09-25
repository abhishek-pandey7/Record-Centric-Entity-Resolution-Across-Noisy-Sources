# Submission log

Public LB = subset of test; private LB decides ranking. Each upload should test one change against a logged baseline.

| # | Date (IST) | Name | Code state | Val F0.5 | Public LB | Notes |
|---|---|---|---|---|---|---|
| 00 | 2026-09-25 | all-empty probe | Task 1 (`src.probe_empty`) | 0.0558 (= val singleton rate) | _pending_ | Public score = public-test singleton rate; train rate is 0.0558. File: `submissions/sub00_all_empty/matching_results.tsv` |
| 01 | 2026-09-25 | baseline stage-1 LGBM | blocking k=5→rank≤3, 42 features, 4/17 train folds, exclusivity + thr 0.675 | 0.95825 (US 0.977, India 0.930; singletons 0.959) | _pending_ | File: `submissions/sub01_baseline/matching_results.tsv`. Blocking ceiling 0.982. Model hit 3000 rounds without early stop. |

# Submission log

Public LB = subset of test; private LB decides ranking. Each upload should test one change against a logged baseline.

| # | Date (IST) | Name | Code state | Val F0.5 | Public LB | Notes |
|---|---|---|---|---|---|---|
| 00 | 2026-09-25 | all-empty probe | Task 1 (`src.probe_empty`) | 0.0558 (= val singleton rate) | _pending_ | Public score = public-test singleton rate; train rate is 0.0558. File: `submissions/sub00_all_empty/matching_results.tsv` |
| 01 | 2026-09-25 | baseline stage-1 LGBM | blocking k=5→rank≤3, 42 features, 4/17 train folds, exclusivity + thr 0.675 | 0.95825 (US 0.977, India 0.930; singletons 0.959) | not submitted (inference stopped) | File: `submissions/sub01_baseline/matching_results.tsv`. Blocking ceiling 0.982. Model hit 3000 rounds without early stop. |
| 02 | 2026-09-26 | v2 translit + orphan world | commit d50c156, 6 folds, thr 0.725 | 0.97542 (orphan world) | **0.967** | Val overestimates LB by ~0.008. File: `submissions/sub02_v2/matching_results.tsv` |
| 03 | 2026-09-26 | v3 + number-edit + sibling features | commit 35efe84 | 0.97856 | not submitted | Local check: +574k links vs v2, mostly number-shifted decoys (10.7% incompatible numbers in US vs 2.0% true rate). Sibling features backfire on test. |
| (team) | 2026-09-26 | teammate best | teammate pipeline | n/a | 0.965 | `matching_results.tsv` in repo root |

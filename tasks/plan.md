# Implementation Plan: Business Entity Resolution (Amazon ML Challenge 2026)

## Overview

For every Source 1 (S1) entity, find all of its Source 2/3 records.
- **Metric:** macro F0.5 per S1 entity. Singletons count: they score 1.0 for an empty prediction and 0 for any prediction.
- **Deadline:** 27 Sep 2026, 23:59 IST.
- **Submissions:** 15 in total, 5 per day.

**Pipeline:**
1. Country-partitioned, record-centric multi-pass blocking.
2. LightGBM pairwise matcher.
3. Stage-2 context model: competition between S1s, and support from sibling records.
4. Decision layer: exclusivity, then choose each entity's subset to maximize expected F0.5.

Country is an open set of strings. France is 15% of test S1 and appears only in the test. It's handled with country-agnostic features, language dictionaries, and unsupervised statistics from the test inputs.

Task details, acceptance criteria and verification steps are in `tasks/todo.md`.

## Data facts that drive the design

Measured on the full train set unless marked "sample"; the sample is 2% of S1 entities, about 151k true pairs.

| Fact | Value | Consequence |
|---|---|---|
| S2/S3 records in more than one cluster | **0 of 7,638,365 links** | Each record belongs to at most one S1, so retrieve **per record** and enforce exclusivity |
| Records in no cluster (distractors) | S2 26.6%, S3 25.4% | The model must reject about a quarter of records outright |
| Singletons | 5.58% (US 5.6%, India 5.6%) | Predicting empty is right for only ~6% of entities; for the other ~94% it scores 0 |
| Cluster size | mean 3.46; 17% have 2, 24% have 3, 22% have 4; 4% have ≥ 7 | A per-S1 top-8 cap on candidates hurts large clusters |
| S1 entities sharing a normalized name with another S1 in the same country | **49.6%** (27% share even the raw name: "primary care group" ×76) | Name alone can't identify an entity; address is mandatory; add a name-commonness feature |
| Latin true pairs sharing a rare name token (DF < 5000) | US 76%, India 79% | A name-only index caps recall near 80% |
| True pairs sharing a number and a street word | US 81%, India 86% | The address anchor is the second pillar; recall comes from the union |
| First S1 number found in the record | US 78%, India 89% | Noise pads zeros (`0302`, `009111`), truncates (`8880`→`880`) and adds suffixes (`448-D`) |
| Zero name overlap (DBA / initials) | US 2.1%, India 2.5% of pairs, all with an address | e.g. `Ectoevo`, `Brixbelo`, `mc.com` ↔ Martinez Castillo… Recover via address + sibling support + initials |
| Indic-script names | India pairs 17.8%, across 9 scripts (Devanagari 56%) | **96% of those records have a Latin address body**; the address anchor covers them |
| Empty address | S1 0%; true-match records 4.4%; distractors 0.3% | An empty address is ambiguous when the name is shared |
| Postal codes | effectively absent (< 2%, mostly false hits) | No PIN/ZIP blocking |
| Train/test S1 overlap | 0 exact (name, address) | No lookup shortcut |
| **Test shift** | S2+S3 per S1: train 4.68 → test 5.75 (+23%) | Either more distractors or bigger clusters; monitor, and optionally probe (P0) |

## Architecture decisions

1. **Record-centric blocking.** Each S2/S3 record retrieves its top-K S1 entities in the same country. This follows from exclusivity, handles big clusters without a cap, and yields competition features. `candidate_pairs.tsv` is exactly this pool, inverted to one row per S1.
2. **Blocking is a union of passes**, because no single key clears ~86% recall:
   - IDF-weighted name tokens;
   - address anchor (number + street token);
   - concatenated name (domains);
   - typo key;
   - name × address conjunction keys for generic names.

   There are no city-letter or postal keys.
3. **Country is a partition key only**, compared as a string. It is never a model feature. IDF is computed per country from each split's own corpus, so French statistics come from French text.
4. **LightGBM, binary objective, trained on all candidates of the training entities.** There's no 1:4 negative sampling and no lambdarank, because the decision layer needs calibrated absolute probabilities, not a ranking.
5. **Two stages.** Stage 1 uses pairwise features. Stage 2 adds out-of-fold context:
   - the record's best alternative S1 and the margin to it;
   - how similar the record is to the entity's confident siblings.

   Sibling support is how DBA records get caught (e.g. `Deltafaye` shares `448-D Chism St` with its S3 siblings).
6. **The decision layer optimizes the metric directly.**
   - Each record goes only to its argmax S1.
   - Each entity gets the subset that maximizes E[1.25·TP / (0.25·T + k)].
   - k = 0 is scored by an entity-level singleton model.

   The global threshold stays as a fallback. The pasted plan's "singleton guardrail" is redundant with a global threshold; this replaces it.
7. **Indic names use back-transliteration learned from train pairs**, with a dependency-free romanizer built on Unicode character names as the fallback. Naive romanization turns `વન ઇન્ફ્રા પ્રા. લિ.` into `vana inphra pra. li.`, not `one infra pvt ltd`.
8. **Validation.**
   - Split 85/15 by S1 entity.
   - Run val blocking against the full train S2/S3 pool so val sees real distractors.
   - The scorer replicates the official definition exactly.
   - Val is the decision signal; the public LB is a sanity check. Final ranking uses the private split, so don't chase public LB noise.
9. **Compute.** Polars + Parquet with int ids, rapidfuzz, LightGBM and scipy sparse, with multiprocessing over country partitions and chunks.
   - Development runs locally on 16 threads.
   - Full runs go to Kaggle (30 GB RAM) or Lightning if local RAM is short.
   - The T4 GPUs are only needed for the optional Task 10.
   - All models are MIT or Apache-2.0 licensed (LightGBM is MIT).
10. **Version history** (the guidelines require it): a git repo with one tag per leaderboard submission, and `submissions/log.md` recording config, val F0.5 and LB F0.5.

## Dependency graph

```
T1 data + metric + split ─► T2 normalizer ─► T3 blocking ─► T4 features + LGBM + writers ─► Sub #1
        │                        │                                   │
        └─► P0 probe (opt.)      ├─► T6 Indic translit ──(re-run T3/T4)
                                 └─► T8 France pass ◄────────────────┤
                                                   T5 decision ◄─────┤
                                                   T7 stage-2 OOF ◄──┘
                      T5 + T6 + T7 + T8 ─► T9 full retrain ─► T11 package + docs
                                 T3 + T7 ─► T10 GPU (optional) ─► T11
```

## Task list

### Phase 0 — Foundation (Day 1 evening)
- [ ] Task 1: Data layer, exact metric, entity split, output writers
- [ ] Task P0 (optional): all-empty leaderboard probe → public-test singleton rate
- [ ] Task 2: Normalizer v1 (country-agnostic core + EN/IN/FR dictionaries)

### Phase 1 — Thin end-to-end slice (Day 1 night → Day 2 morning)
- [ ] Task 3: Blocking v1 (record-centric, multi-pass) + recall report
- [ ] Task 4: Pairwise features + LightGBM v1 + writers → **Submission #1**

### Checkpoint 1
- [ ] Tests green, validator PASS, Submission #1 scored, |LB − val| explained
- [ ] FP/FN error buckets decide the Phase 2 order; human review

### Phase 2 — Metric levers (Day 2)
- [ ] Task 5: Decision layer (calibration + singleton model + expected F0.5)
- [ ] Task 6: Indic back-transliteration
- [ ] Task 7: Stage-2 context model (OOF)
- [ ] Task 8: France robustness + unlabeled diagnostics

### Checkpoint 2
- [ ] Each change kept only if val improves; submissions 2–5 logged; France diagnostics OK; human review

### Phase 3 — Scale, optional GPU, finalize (Day 3)
- [ ] Task 9: Full-scale retrain + single-command reproducible run
- [ ] Task 10 (optional): GPU dense retrieval or a cross-encoder feature, only if Checkpoint 2 shows headroom
- [ ] Task 11: Final selection + submission zip + documentation

### Checkpoint 3
- [ ] Final upload before 22:00 IST on 27 Sep (buffer); zip submitted; `git tag final`

## Submission budget (15)

| Day | Slots | Planned use |
|---|---|---|
| 25 Sep | 5 | P0 probe (optional); Submission #1 if Phase 1 lands tonight |
| 26 Sep | 5 | Baseline (if not yet) → +decision layer → +Indic/stage-2 → +France fixes |
| 27 Sep | 5 | Full retrain → optional GPU variant → final; keep 1–2 slots in reserve |

Each upload must answer one question: one change against a logged baseline.

## France: will zero-shot transfer "just work"?

**Partly. The direction is right, but "models speak math, so it transfers" is overconfident.** The table below checks each claim from the pasted answer.

| Claim | Reality | What we do |
|---|---|---|
| Drop country features and similarity scores are language-neutral | Not quite. French addresses are full of function words (`de`, `la`, `des`, `du`) and a few street types (`rue`). These inflate unweighted overlap between *different* addresses. With only a handful of cities, city overlap carries little signal. | IDF-weighted overlaps with per-country IDF from the test corpus; strip particles; lean on number + street + name |
| The threshold learned on US/India transfers safely | The optimal threshold depends on calibration and on the match base rate. Both can differ by country, and between train and test (+23% S2/S3 per S1). | Use per-country diagnostics; decide with calibrated probabilities; optionally A/B France on the LB |
| "F0.5 safety net": low confidence → singleton → 1.0 | **Wrong.** About 94% of entities have matches, and an empty prediction scores 0 for them. If French confidence drops, up to 15% of the test score is at risk. | Target a French predicted-empty rate near the US/India rate (~6%); fix normalization until it gets there |
| The (country, number, street) key bypasses département vs région | True for blocking. But the model learned on US/India, where normalized states always agree for true pairs, so a state feature will penalize French S3 true pairs. | Mine state aliases from the data via city co-occurrence (`Gironde`→`Nouvelle-Aquitaine`), or leave state out of features |
| Strip SARL/SAS/SA/EURL/SCI | Correct and necessary | In Task 2 (plus SASU/SNC, `S.A.S` dotted forms) |
| "St" means Street | In French, `St`/`Ste` before a name means Saint (`RUE ST JEAN`) | Task 2 disambiguates by position |

**How we'll know it works without labels (Task 8):**
- per-country blocking coverage;
- the distribution of each entity's top-candidate p;
- the predicted-empty rate;
- predicted cluster size compared with the (S2+S3)/S1 ratio;
- adversarial validation of feature vectors;
- a manual audit of 50 French predictions.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Blocking recall ceiling too low (DBA, typos, generic names) | High | Union of 5 passes; per-pass recall report; optional dense pass (T10) |
| Shared names (49.6%) cause wrong-branch merges | High | Address-centric features; name-commonness feature; exclusivity; empty-address records assigned only when the name is unique |
| France zero-shot shift | High (15% of test) | Dictionaries, per-country IDF, alias mining, unlabeled diagnostics, optional LB A/B |
| Train→test shift (+23% S2/S3 per S1) | Med | P0 probe; per-country predicted rates; decision-layer prior adjustment |
| Stage-2 leakage | Med | Strict K-fold by S1 entity; shuffled-label check |
| Memory/time at 23M records | Med | Parquet + Polars, country partitions, chunking; Kaggle 30 GB for full runs |
| Format rejection wastes a slot | Med | Writers unit-tested; validator runs inside the pipeline |
| Overfitting to the public LB | Med | Val-driven selection; the private split decides ranking |
| Fair play / license | High | No external lookups; hand-written dictionaries documented; MIT/Apache models only |

## Decisions (25 Sep)
- P0 probe: **yes**, using today's slots.
- Test-side statistics (IDF, alias mining on test inputs, never labels): **yes**, documented in the methodology.
- Full-data runs: **Kaggle** (30 GB RAM, 4 CPU). Local machine is for development on samples (free RAM is ~1–3 GB).
- Team name/members: still needed for the docs.

## Open questions (original)

1. **P0 probe:** OK to spend one of today's otherwise idle slots on an all-empty submission? It measures the public-test singleton rate, which tells us whether the +23% shift is distractors or bigger clusters. Recommendation: yes.
2. **Test-side statistics:** per-country IDF and alias mining read test *inputs* (never labels). This is standard transductive practice, and the rules forbid only external lookups. Are you comfortable with it? Recommendation: yes, and say so in the docs.
3. **Team name and members** for the documentation, and does the team have more than one person? If so, Tasks 6 and 8 can run in parallel with Tasks 5 and 7.
4. **Where full runs happen:** local (16 threads, but only ~2 GB RAM free right now; close other apps) or Kaggle (30 GB RAM, 4 cores)?

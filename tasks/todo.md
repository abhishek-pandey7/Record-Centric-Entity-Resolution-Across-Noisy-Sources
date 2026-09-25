# Task List — Business Entity Resolution (Amazon ML Challenge 2026)

Plan and rationale: `tasks/plan.md`. Deadline **27 Sep 2026, 23:59 IST**. 15 leaderboard submissions total (5/day).

Conventions used below:
- Code lives in `code/business_entity_resolution/` (`src/`, `tests/`, `README.md`, `requirements.txt`), so packaging is a copy, not a port.
- Big intermediates (parquet, candidates, features, models, reports) go to `work/`, which is **not** shipped.
- `pytest` = `python -m pytest code/business_entity_resolution/tests -q`
- `validator` = run from `student_resource/`:
  `python utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test`
- "val" = the 15% of train S1 entities held out in Task 1, with blocking run against the **full** train S2/S3 pool (realistic distractors).

---

## Phase 0 — Foundation (Day 1 evening)

### Task 1: Data layer, exact metric, entity split, output writers

**Description:** Convert the 7 TSVs to Parquet with integer ids (source + int), explode the ground truth into a pairs table, create a deterministic 85/15 train/val split **by S1 entity** (stratified by country × cluster-size bucket), implement the official macro F0.5 exactly (singleton rules included), and write the two output files with a validator wrapper. `git init` + submissions log for the required version history.

**Acceptance criteria:**
- [x] Parquet row counts equal the TSVs (train 2,206,821 / 5,034,616 / 5,285,603; test 1,732,544 / 4,887,273 / 5,082,316) and GT links = 7,638,365.
- [x] Scorer: PDF example = 0.714; ground truth as prediction = 1.000; all-empty on val ≈ val singleton rate (~0.056).
- [x] An all-empty test submission written by our writer passes `validator`.

**Verification:**
- [x] `pytest` (test_metric.py: PDF example, singleton/empty edge cases, duplicate ids rejected)
- [x] `python -m src.data --check` prints the counts above
- [x] `validator` → PASS

**Dependencies:** None

**Files likely touched:** `src/data.py`, `src/metric.py`, `src/io_out.py`, `tests/test_metric.py`, `requirements.txt`

**Estimated scope:** M

### Task P0 (optional, needs your OK): Leaderboard probe

**Description:** Upload the all-empty file from Task 1. Its public score is exactly the public-test singleton rate. That tells us whether the test's +23% S2/S3-per-S1 shift comes from more distractors or bigger clusters. It costs one of today's otherwise-unused slots.

**Acceptance criteria:**
- [ ] Score recorded in `submissions/log.md` next to the train rate (5.58%).

**Verification:** Leaderboard shows SCORED.

**Dependencies:** Task 1

**Files likely touched:** `submissions/log.md`

**Estimated scope:** XS

### Task 2: Normalizer v1 (country-agnostic core + EN/IN/FR dictionaries)

**Description:** A single normalization path for all records, selected by the text itself, never by `country ∈ {US, India}`.
- **Names:**
  - NFKC, strip accents, case-fold, `&`→`and`.
  - Strip junk prefixes and brackets (`[Co]`, `>>`, `--`, `<<`).
  - Turn a domain into its concatenated form (`greatresources.com` → `greatresources`).
  - Collapse consecutive repeated tokens.
  - Extract the legal form into its own field (LLC/Inc/Corp/Co/Ltd/Pvt/LLP/PLC + SARL/SAS/SASU/SA/EURL/SCI/SNC).
- **Addresses:**
  - Canonicalize street types (EN + FR: R./RUE, AV/AVE, BD, PL, IMP, ALL, CHE, RTE, QUAI, CRS).
  - Tag particles (de/du/des/la/le/l'/d') and landmark words (near/nr/opp/b/h/behind).
  - Parse numbers: `#`, `##`, `H.NO`, `NO.`, `N°`, `(14)`, `23 -`, `BIS`/`TER`, `448-D`, `1/2`, `4-13-31/6`, `A-8/2B/8/3`. **Strip leading zeros**: the noise pads numbers (`0302`↔`302`, `009111`↔`9111`). Keep all number tokens plus a primary number.
  - Treat `NULL`, `N/A`, `NA`, `CDP` and `City Of` as noise tokens.
  - Build state alias tables for US codes and Indian codes/names. Mine native-script state names (महाराष्ट्र, తెలంగాణ, കേരളം) from train co-occurrence.

**Acceptance criteria:**
- [x] Golden tests from real rows pass:
  - `[Co] Leland and Runk Offshore` ≡ `Leland Leland and Runk Offshore Company` (core tokens {leland, runk, offshore}).
  - `greatresources.com` concat = `GREAT RESOURCES LLC` concat.
  - `30 BIS R. DE VERDUN` → number 30, street `rue verdun`.
  - `N° 17 RUE DES FAUVETTES` → 17, `rue fauvettes`.
  - `Ohio`≡`OH`, `GJ`≡`Gujarat`.
- [x] Normalizing all ~22.9M records finishes in < 30 min (multiprocess) and writes to `work/norm/*.parquet`.
- [ ] A spot check of 50 random rows per country (US, India, France) shows no systematic mis-parse.

**Verification:**
- [x] `pytest` (test_normalize.py)
- [ ] `python -m src.normalize --sample 50` prints before/after for manual review

**Dependencies:** Task 1

**Files likely touched:** `src/normalize.py`, `src/dicts.py`, `tests/test_normalize.py`

**Estimated scope:** M

---

## Phase 1 — Thin end-to-end slice → Submission #1 (Day 1 night → Day 2 morning)

### Task 3: Blocking v1 (record-centric, multi-pass) + recall report

**Description:** Within each country partition (string equality, open set), retrieve the top-K S1 entities for every S2/S3 record. Each record belongs to at most one S1, so retrieving per record handles large clusters without a per-S1 cap. The candidate pool is the union of four passes:
- **P1:** sparse TF-IDF over field-prefixed name + address tokens, using rare-token postings with a DF cap and per-country IDF computed from that split's own corpus.
- **P2:** address anchor (primary number + rarest street token). This catches DBA renames and Indic-script names.
- **P3:** exact match on the concatenated name, plus a char-gram key for domains and spacing.
- **P4:** typo key (deletion or phonetic key of the rarest name token).
- **P5:** conjunction keys (name token × address token) for generic names. On their own, `dental` and `group` are too common to block on; combined with a street token they are rare.

Why no single pass is enough: only 76–79% of Latin true pairs share a name token with DF < 5000, and only 81–86% share a number plus a street word. Recall has to come from the union.

Keep the pass flags, scores and ranks as features. `candidate_pairs.tsv` is this pool, inverted to one row per S1.

**Acceptance criteria:**
- [ ] On val, link recall is ≥ 97% (target 99%) — **got 95.5% @10 (Indic 70.6%, empty-address 74.9%); F0.5 ceiling 0.982 → Task 6 fixes Indic**, and the report covers:
  - the entity-level F0.5 ceiling;
  - recall broken down by pass, country, Indic vs Latin name, empty address and zero-name-overlap.
- [x] Mean candidates per S1 is ≤ 25, with the distribution and reduction ratio reported.
- [x] Full test blocking takes ≤ 60 min with peak RAM < 12 GB (local 16 threads or Kaggle).

**Verification:**
- [x] `pytest` (test_blocking.py: toy corpus where known pairs must be retrieved)
- [x] `python -m src.blocking --split val --report` → `work/reports/blocking_val.md`

**Dependencies:** Task 2

**Files likely touched:** `src/blocking.py`, `src/eval_blocking.py`, `tests/test_blocking.py`

**Estimated scope:** M

### Task 4: Pairwise features + LightGBM v1 + writers → Submission #1

**Description:** Build about 40 pair features:
- rapidfuzz ratios on names, addresses and concatenated names;
- IDF-weighted token overlaps;
- number features: exact, one-digit-truncation (8880↔880) and any-overlap;
- street and locality token matches, and legal-form equality;
- **name commonness** (how many S1 entities in the country share the core name; 49.6% of S1 share one);
- record flags: empty address, Indic, domain, source S2/S3;
- retrieval pass flags, scores and ranks;
- DBA signals:
  - initials match (`mc.com` ↔ Martinez Castillo…, `BA` ↔ Best Agro);
  - "brandable" name, i.e. tokens out of the S1 name vocabulary (`Ectoevo`, `Brixbelo`, `Avikelo`, `Deltafaye`);
  - empty-address flag (4.4% of true-match records but 0.3% of distractors).

No country or id features. Train LightGBM (binary) on **all** candidates of the training entities (no 1:4 sampling), with early stopping on val. Decision v0 uses exclusivity (a record stays only with its argmax S1), then a global threshold swept on val macro F0.5. Write both TSVs, run the validator, and submit.

**Acceptance criteria:**
- [x] Val macro F0.5 reported overall, by country, by cluster size and for singletons, with feature importances reviewed.
- [ ] Test outputs: validator PASS, every test S1 has exactly one row, matches ⊆ candidates.
- [ ] Submission #1 scored, and |LB − val| logged (investigate if > 0.03).

**Verification:**
- [x] `python -m src.train --stage 1` → val report
- [ ] `python -m src.predict --split test` → `output/*.tsv`
- [ ] `validator` → PASS

**Dependencies:** Task 3

**Files likely touched:** `src/features.py`, `src/train.py`, `src/predict.py`

**Estimated scope:** M

### Checkpoint 1 — end-to-end works
- [ ] `pytest` green; validator PASS; Submission #1 scored
- [ ] Error analysis: 100 val FPs + 100 val FNs bucketed (blocking miss, DBA/zero-overlap, Indic, empty address, shared name → wrong branch, number truncation, typo). Phase 2 order re-ranked by bucket size.
- [ ] Review with human before proceeding

---

## Phase 2 — Metric levers (Day 2)

### Task 5: Decision layer v1 (calibration, singleton model, expected-F0.5)

**Description:**
- Calibrate pair probabilities with isotonic regression on val.
- Train an entity-level singleton model (LightGBM on entity aggregates: max/sum p, number of candidates, name commonness, whether the top record is claimed by another S1).
- For each entity, choose the subset of its candidates that maximizes E[F0.5], where F0.5 = 1.25·TP / (0.25·T + k). k = 0 is scored by P(singleton).
- Keep the global threshold as the fallback.

**Acceptance criteria:**
- [ ] The expected-F implementation matches brute-force enumeration on random small cases, including the edge cases (no candidates, all p≈0, all p≈1).
- [ ] Val improves by ≥ +0.3 pt over the Task 4 decision; if it doesn't, keep the fallback and record why.
- [ ] Runs on 1.7M entities in < 5 min.

**Verification:**
- [ ] `pytest` (test_decide.py)
- [ ] `python -m src.decide --split val --compare`

**Dependencies:** Task 4

**Files likely touched:** `src/decide.py`, `tests/test_decide.py`

**Estimated scope:** S

### Task 6: Indic back-transliteration

**Description:**
- Learn an Indic-token → Latin-token dictionary from train pairs. Use positional alignment when the token counts match and similarity alignment otherwise.
- As a fallback, use a romanizer built on Unicode character names (works for every Indic script, no dependency) plus a consonant-skeleton key.
- Add a `name_latin` field; blocking P1/P4 and the name features use it.

**Acceptance criteria:**
- [ ] The dictionary covers ≥ 85% of Indic name tokens in the test data (reported).
- [ ] On the val Indic subset, name similarity, blocking recall and F0.5 all go up; non-Indic val stays within ±0.1 pt.

**Verification:**
- [ ] `pytest` (test_translit.py, e.g. `वन इन्फ्रा प्रा. लि.` / `વન ઇન્ફ્રા પ્રા. લિ.` → `one infra pvt ltd` on dictionary hits)
- [ ] Val report split by script

**Dependencies:** Task 2 (then re-run Tasks 3–4)

**Files likely touched:** `src/translit.py`, `src/normalize.py`, `tests/test_translit.py`

**Estimated scope:** M

### Task 7: Stage-2 context model (OOF)

**Description:** Produce 5-fold out-of-fold stage-1 predictions, with folds split by S1 entity. Then add context features:
- **Record side (competition):** this S1's rank among the record's candidates, margin to the best other S1, and how many S1s have p > 0.5.
- **Entity side:** this record's rank, number of confident siblings, and sum of p.
- **Sibling support:** max name and address similarity to the entity's confident siblings, and whether the address exactly equals a same-source sibling's. This covers cases like the `Deltafaye` DBA, which shares the `448-D Chism St` address with the other S3 records.

Retrain LightGBM as stage 2.

**Acceptance criteria:**
- [ ] OOF discipline verified: no training pair is scored by a model that saw it; a shuffled-label run gives val AUC ≈ 0.5.
- [ ] Val improves overall and in the zero-name-overlap, empty-address and shared-name buckets.

**Verification:**
- [ ] `python -m src.train --stage 2 --report`

**Dependencies:** Task 4 (uses Task 5 decision if done)

**Files likely touched:** `src/context.py`, `src/train.py`

**Estimated scope:** M

### Task 8: France robustness + unlabeled diagnostics

**Description:**
- **Normalization audit:** check 200 random French test rows, including the particles and the Saint/St/Ste vs Street ambiguity.
- **Country-agnostic mining on test inputs:**
  - state aliases via city co-occurrence (Gironde→Nouvelle-Aquitaine; also rediscovers Ohio→OH, MH→Maharashtra as a self-check);
  - abbreviation substitutions from high-confidence pairs.
- **Per-country diagnostics:**
  - blocking coverage (share of records with ≥ 1 candidate);
  - max-p distribution per S1;
  - predicted-empty rate;
  - predicted cluster size compared with the (S2+S3)/S1 ratio;
  - adversarial validation of the feature vectors (France test vs US/India val).
- **Manual audit:** 50 French predicted pairs and 50 French predicted-empty entities.

**Acceptance criteria:**
- [ ] France's predicted-empty rate and mean predicted cluster size are close to US/India's (target: empty rate within ±3 pt), or the gap is explained.
- [ ] The top shifted features from adversarial validation are normalized or dropped.
- [ ] Manual-audit precision on French predicted pairs is ≥ 95%.

**Verification:**
- [ ] `python -m src.diagnostics --split test` → `work/reports/france.md`

**Dependencies:** Task 4 (Task 2 dictionaries)

**Files likely touched:** `src/diagnostics.py`, `src/mine_aliases.py`, `src/dicts.py`, `src/normalize.py`

**Estimated scope:** M

### Checkpoint 2 — best config chosen
- [ ] Every Phase 2 change kept only if val improves; each submission logged (config, val, LB)
- [ ] France diagnostics acceptable
- [ ] Review with human before scaling / GPU work

---

## Phase 3 — Scale, optional GPU, finalize (Day 3)

### Task 9: Full-scale retrain + single-command reproducible run

**Description:** Retrain stages 1 and 2 on all train entities, fitting thresholds and calibration on OOF. Fix all seeds. A single entrypoint `python -m src.run --data-dir ... --out-dir ...` runs normalize → block → features → predict → decide → write → validate. Log runtime and peak RAM.

**Acceptance criteria:**
- [ ] A fresh run from the raw TSVs reproduces the submitted match sets exactly.
- [ ] The README documents runtime and hardware.

**Verification:**
- [ ] End-to-end run (local or Kaggle) + diff against the submitted files

**Dependencies:** Tasks 5–8

**Files likely touched:** `src/run.py`, `README.md`, `requirements.txt`

**Estimated scope:** S

### Task 10 (optional, GPU): Neural boost

**Description:** Do this only if the Checkpoint 2 buckets show recall headroom (Indic, typos, DBA). There are two options:
- **(a) Dense retrieval pass:** `intfloat/multilingual-e5-small` (MIT) or `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0), with FAISS on a T4.
- **(b) Cross-encoder feature:** fine-tune a small multilingual cross-encoder (MIT/Apache) and feed its score to stage 2, only for uncertain pairs (0.05 < p < 0.95).

**Acceptance criteria:**
- [ ] The recall ceiling or val F0.5 improves by ≥ 0.3 pt.
- [ ] Test inference takes < 2 h on a T4.
- [ ] The model license is recorded in the docs.

**Verification:**
- [ ] Val report + timing log (Kaggle/Lightning notebook)

**Dependencies:** Tasks 3, 7

**Files likely touched:** `src/dense.py`, `src/cross_encoder.py`, `notebooks/*.ipynb`

**Estimated scope:** M

### Task 11: Final selection + submission package + documentation

**Description:** Pick the final submission using val as the main signal and the public LB as a sanity check; ranking is decided on the private split, so don't chase public LB noise. Build the zip in the required layout. Fill in `Documentation_template.md` and write the 1–2 page approach doc from the guidelines. Pin the requirements, give exact README commands, and tag the git commit.

**Acceptance criteria:**
- [ ] The zip layout matches the spec exactly, the outputs pass the validator, and matches ⊆ candidates.
- [ ] The README steps work in a fresh venv on a small sample.
- [ ] The docs cover methodology, blocking, model/features, results, and FP/FN error analysis with examples.

**Verification:**
- [ ] `unzip -l <team>_submission.zip`; `validator`; fresh-venv dry run

**Dependencies:** Task 9 (Task 10 if done)

**Files likely touched:** `code/business_entity_resolution/README.md`, `Documentation_template.md`, `requirements.txt`, `scripts/package.py`

**Estimated scope:** S

### Checkpoint 3 — complete
- [ ] Final submission uploaded before 22:00 IST 27 Sep (buffer for portal issues)
- [ ] Zip submitted; git tag `final`
- [ ] All acceptance criteria met

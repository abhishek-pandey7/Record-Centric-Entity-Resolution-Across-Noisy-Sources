# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We flip the problem around: every Source 2/3 record belongs to at most one Source 1 entity, so each record retrieves its own likely owners with a multi-key blocking index, and a LightGBM classifier scores every (record, entity) pair. The main ideas are (1) record-centric blocking with an Indic-script transliteration dictionary learned from the training pairs, (2) training and validating in a simulated "orphan world" that matches the test set's higher share of unmatched records, and (3) features that expose how the synthetic decoys differ from genuine noise (legal-type switches, added descriptor words, number edits). Best validation macro F0.5 0.9798; public leaderboard 0.972.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploring the full training data (all numbers measured, not sampled unless stated):

- **One owner per record.** Of 7,638,365 ground-truth links, no S2/S3 record appears in more than one cluster. We therefore decide, for each record, which S1 entity it belongs to (or none).
- **Names are not unique.** 49.6% of S1 entities share their normalized name with another S1 entity in the same country (e.g. "primary care group" appears 76 times). The address has to decide; the name only supports.
- **No postal codes** (under 2% of addresses contain anything like a PIN/ZIP), so blocking cannot rely on them.
- **Indic scripts.** 17.8% of Indian matched records have names in one of 9 Indian scripts (Devanagari, Telugu, Kannada, Gujarati, Tamil, Bengali, Malayalam, Oriya, Gurmukhi). They are word-by-word transliterations of the English name (`शक्ति प्रोडक्ट्स एलएलपी` = "Shakti Products LLP"), and 96% of those records keep a Latin address.
- **Noise patterns.** Case, accents, leet (`Pr0duce`, `5ky`), shuffled words, repeated tokens, legal-suffix changes, website forms (`greatresources.com`), wrappers (`X doing business as Y`, `| www.x.com`), random trade names, zero-padded or truncated house numbers (`0302`, `8880` to `880`), state name/code/script variants, reordered address components.
- **Decoys.** About 26% of training records match nothing. Many are deliberate near-copies of a real S1 business with one targeted change. We found that decoys, unlike genuine noise, switch the legal type (`Inc` to `LLC`, `Limited` to `LLP`) and add business descriptor words that genuine noise never adds (`solutions`, `exports`, `global`, `finance`).
- **Test shift.** The test set has 5.75 S2/S3 records per S1 entity versus 4.68 in training, so it contains roughly 1.9x more unmatched records per entity. A model validated on the training distribution overestimates its leaderboard score.
- **France** (15% of test S1) has no training labels. Every feature is a relative similarity or a record flag, and country is used only as a blocking partition, never as a model input.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted classifier + decision layer (hybrid).
**Core Innovation:** record-centric retrieval combined with an "orphan world" training simulation, plus features that encode the difference between genuine noise and synthetic decoys.

Pipeline:

1. **Load** the TSVs into Parquet with integer ids; split train S1 entities into 20 hash folds (3 for validation, the rest for training).
2. **Normalize** names and addresses (section 3 inputs).
3. **Block**: every S2/S3 record retrieves its top 5 S1 entities in the same country; pairs ranked 1 to 3 are scored.
4. **Featurize** each pair (about 60 features) and score it with LightGBM.
5. **Decide**: each record goes only to its highest-scoring entity (exclusivity), then a probability threshold chosen on validation macro F0.5, plus stricter thresholds for legal-type conflicts and never-added descriptor words.

**Orphan world.** To make validation (and training) resemble the test set, we delete a random 18% of training S1 entities from the blocking index. Their records remain and can only match look-alikes, so the model learns to reject them and validation tracks the leaderboard (v2: validation 0.9754, leaderboard 0.967; v5: 0.9795 and 0.972).

---

## 3. Candidate Generation (Blocking)

**Normalization** (country-agnostic core, dictionaries selected by the text, not by a fixed country list):
- Names: Unicode NFKC, accents stripped, `&` to `and`, junk prefixes and brackets removed, domains turned into their concatenated form, consecutive duplicate words collapsed, legal forms (US, Indian and French: LLC, Inc, Ltd, Pvt, LLP, SARL, SAS, EURL, SCI, ...) moved to a separate field, honorifics (`M/s`, `Shri`, `Smt`) dropped.
- Indic names: a token dictionary (1,347 entries) learned from training pairs by positional alignment covers 96.4% of test Indic tokens; a romanizer built from Unicode character names handles the rest.
- Addresses: street types canonicalized (English and French), house numbers parsed from `#`, `H.NO`, `N°`, `BIS`, `448-D`, `1/2` and similar forms with leading zeros removed, state aliases resolved (US codes, Indian codes, French regions and departements, and 16 native-script state names learned from training pairs), noise tokens (`NULL`, `N/A`, `CDP`) removed.

**Blocking keys used** (hashed per country; only keys with a document frequency under a cap are indexed):
- name token; concatenated name; name 6-grams (for glued websites and typos);
- (house number, address word); (house number, name token); (house number, 4-letter name prefix);
- pair of name tokens; (name token, address word); pair of address words.

Each record scores its candidates by the summed log inverse document frequency of shared keys and keeps the top 5.

- **Candidate pairs generated:** about 50.8 million for the 10.3 million training records and about 29.7 million pairs (rank 3 and above) for the 10 million test records, roughly 17 candidates per test S1 entity.
- **How we ensured true matches were not lost:** recall was measured on validation after every change. Transliteration raised Indic-name recall from 70.6% to 96.1%; the address-word and 6-gram keys raised overall link recall at rank 10 from 95.5% to 98.0%. A perfect matcher on our candidates would score 0.993 macro F0.5 on validation.

---

## 4. Matching Model

**Features used:**
- Name features: rapidfuzz ratio, token-sort, token-set and partial ratios, Jaro-Winkler on the concatenated name, token Jaccard, IDF-weighted overlap (per-country IDF from each split's own S1 table), length difference, initials match, name commonness (how many S1 entities share the name), legal-type relation (same class, overlap, conflict), counts of added descriptor words that genuine records never add (table learned on held-out folds 15 and 16).
- Address features: rapidfuzz ratios, token Jaccard, IDF-weighted overlap, state agreement, first-number similarity, number overlap count, number-edit type (equal, digits deleted, digits inserted, one digit substituted, other) and minimum digit edit distance.
- Other: blocking score, rank among the record's candidates, score relative to the record's best candidate, number of candidates, source (S2 or S3), empty-address, website-name and Indic-script flags.

**Model type:** LightGBM binary classifier (MIT licence), 127 leaves, learning rate 0.08, up to 3,000 rounds with early stopping, trained on all candidate pairs of 12 of the 17 training folds (no negative subsampling, so probabilities stay calibrated).
**Threshold selection method:** threshold swept on validation macro F0.5 after exclusivity (best 0.7); stricter thresholds for legal-type conflicts and never-added words chosen the same way.

---

## 5. Results & Error Analysis

| Version | Main change | Validation F0.5 | Public LB |
|---|---|---|---|
| v2 | transliteration, new blocking keys, orphan world | 0.9754 | 0.967 |
| v4 | number-edit features | 0.9773 | |
| v5 | legal-type and added-word features | 0.9795 | |
| v5 + rules | stricter thresholds for conflicts | **0.9798** | **0.972** |

- **F0.5 Score (macro):** 0.9798 on validation (India 0.976, US 0.982, singletons 0.976).
- **Common false positives (wrong merges):** near-copy decoys with one targeted change (house number `311` to `31`, `430` to `43`; a random name at the exact address; a legal-type switch), and records of a different business at the same address (several companies in one building).
- **Common false negatives (missed matches):** records with an empty address and a common name (43% of rejected true pairs; several S1 entities share the name so the record is ambiguous), true records whose house number was substituted rather than truncated, and records outside the blocking candidates (2.4% of true links).

Things we tried that did not help: sibling-context features (they helped on validation but made the model accept groups of test decoys: 10.7% of US links had incompatible house numbers against a 2.0% true rate), expected-F0.5 per-entity decisions and per-segment thresholds (no gain because the probabilities were already well calibrated), and number-consensus rules (removed about ten true links per decoy).

---

## 6. Conclusion

Treating each record as a query for its single owner, then separating genuine noise from generated decoys, took the model from 0.958 to 0.980 on a deliberately test-like validation set and to 0.972 on the public leaderboard. The most useful lessons were to validate in a world that matches the test distribution, and to measure where errors come from before adding model capacity.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains the full pipeline (`src/`), unit tests (`tests/`), `README.md` and `requirements.txt`. One command reproduces both output files:

```
python -m src.run_all --data-dir <student_resource/dataset> --work-dir <scratch> --out-dir <output>
```

Stages: `src.data` (loading, folds), `src.normalize` and `src.translit` (normalization, transliteration), `src.blocking` (candidate generation), `src.features` and `src.context` (pair features), `src.train` (LightGBM, validation), `src.predict` (test scoring, output files, validator). Runtime on a Kaggle CPU notebook (4 cores, 30 GB RAM): about 5 to 6 hours.

### B. Additional Results

Blocking recall on validation (rank 10): overall 98.0%, US 98.6%, India 97.2%, Indic-script names 96.1%, empty-address records 74.9%.

# Business Entity Resolution: reproducible pipeline

Links every Source 1 entity to its Source 2/3 records. Output: `matching_results.tsv` (scored) and
`candidate_pairs.tsv` (the exact candidate set the model scores; matches are a subset).

## Pipeline

| Stage | Module | Output (under `--work-dir`) |
|---|---|---|
| 0. TSV → Parquet, GT pairs, entity folds | `src.data` | `raw/` |
| 1. Normalization (names, addresses, numbers, states) | `src.normalize` | `norm/` |
| 2. Record-centric blocking (7 hashed key types, top-K per record) | `src.blocking` | `cand/` |
| 3. Pair features (rapidfuzz, IDF overlaps, numbers, context) | `src.features` | `feat/` |
| 4. LightGBM pair classifier + val evaluation | `src.train` | `model/` |
| 5. Decision layer (exclusivity + threshold) | `src.decide` | (in memory) |
| 6. Test inference + writers + validator | `src.predict` | `--out-dir` |

## Run end-to-end

```bash
pip install -r requirements.txt
export BER_DATA_DIR=/path/to/student_resource/dataset   # contains train/ and test/
export BER_WORK_DIR=/path/to/work                        # ~15 GB of intermediates
export BER_OUT_DIR=/path/to/output

python -m src.data --check                 # ~2 min
python -m src.normalize                    # ~15 min on 8 cores
python -m src.blocking --split train --queries all
python -m src.train --cand $BER_WORK_DIR/cand/train_all.parquet
python -m src.predict                      # blocking + features + model on test, writes both TSVs, validates
```

The validator in `utils/validate_submission.py` must be reachable at `<data-dir>/../utils/`.

## Kaggle

1. Upload `student_resource` (dataset + utils) as a private Kaggle dataset, plus this folder.
2. In a notebook (CPU, 30 GB RAM): copy this folder to `/kaggle/working/ber`, `cd` into it, set
   `BER_DATA_DIR=/kaggle/input/<dataset>/student_resource/dataset`, `BER_WORK_DIR=/kaggle/working/work`,
   `BER_OUT_DIR=/kaggle/working/output`, and run the commands above.

## Tests

```bash
python -m pytest tests -q
```

## Rules compliance

No external data or lookups. Static dictionaries (US/Indian states, French regions/départements, street
types, legal forms) are general language/geography knowledge; native-script state names are mined from the
training pairs. Model: LightGBM (MIT).

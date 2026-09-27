"""Stage 8: blend LightGBM probabilities with cross-encoder logits on the uncertain band and write the submission.

The blender (logistic regression on [logit(p_lgbm), ce_logit]) is fitted on out-of-fold validation scores, then
applied to the test pairs the cross-encoder scored. Pairs outside the band keep their LightGBM probability.

Usage: python -m src.blend --val-oof <val_ce_oof.parquet> --test-scores <test_scores.parquet>
                           --test-ce <test_ce.parquet> --out <dir> [--threshold 0.7]
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from . import config, decide, io_out

KEYS = ["src", "rid", "s1"]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_blender(val_oof: pl.DataFrame) -> LogisticRegression:
    X = np.column_stack([_logit(val_oof["p"].to_numpy()), val_oof["ce"].to_numpy()])
    return LogisticRegression(C=1.0).fit(X, val_oof["y"].to_numpy())


def blend(scores: pl.DataFrame, ce: pl.DataFrame, lr: LogisticRegression) -> pl.DataFrame:
    X = np.column_stack([_logit(ce["p"].to_numpy()), ce["ce"].to_numpy()])
    new = ce.select(KEYS).with_columns(pl.Series("pb", lr.predict_proba(X)[:, 1]))
    return scores.join(new, on=KEYS, how="left").with_columns(pl.coalesce("pb", "p").alias("p")).drop("pb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir"); ap.add_argument("--val-oof", required=True); ap.add_argument("--test-scores", required=True)
    ap.add_argument("--test-ce", required=True); ap.add_argument("--out", required=True); ap.add_argument("--threshold", type=float, default=0.7)
    a = ap.parse_args()
    work, out = config.work_dir(a.work_dir), Path(a.out)
    lr = fit_blender(pl.read_parquet(a.val_oof))
    print("blender coefficients [lgbm logit, ce logit]:", lr.coef_.round(3).tolist(), "intercept", lr.intercept_.round(3).tolist())
    scores = pl.read_parquet(a.test_scores, columns=KEYS + ["p"])
    b = blend(scores, pl.read_parquet(a.test_ce), lr)
    matches = decide.apply_threshold(b, a.threshold)
    s1_ids = pl.read_parquet(work / "raw" / "test_s1.parquet", columns=["rid"])["rid"]
    io_out.write_id_lists(out / "matching_results.tsv", s1_ids, matches, io_out.MATCH_HEADER)
    io_out.write_id_lists(out / "candidate_pairs.tsv", s1_ids, scores.select("s1", "src", "rid"), io_out.CAND_HEADER)
    print(f"threshold {a.threshold}: {matches.height:,} links, {matches['s1'].n_unique():,}/{s1_ids.len():,} S1 non-empty")
    ok = io_out.validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", config.data_dir() / "test")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

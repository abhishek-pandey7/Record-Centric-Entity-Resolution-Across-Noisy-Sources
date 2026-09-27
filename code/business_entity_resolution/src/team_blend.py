"""Add our cross-encoder to the team's 0.981 pipeline (measured +0.0011 val F0.5 on 3.4M held-out pairs).

Inputs from the team pipeline:
  --val   validation pairs  (source1_entity_id, record_entity_id, score, predicted, is_true_match)
  --test  test pairs        (source1_entity_id, record_entity_id, score[, predicted])
  --match their test matching_results.tsv (decisions used outside the uncertain band)
Decision (exactly as measured on validation): keep the team's decisions for pairs outside the uncertain band
(LO < score < HI); inside the band, accept a pair when the blended probability >= --threshold (0.7).

Usage: python -m src.team_blend --val V.tsv --test T.tsv --match M.tsv --cand C.tsv --out DIR
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch

from . import config, io_out
from .blend import _logit
from .cross_encoder import HI, LO, pair_text, score

K = ["s1", "src", "rid"]


def read_pairs(path) -> pl.DataFrame:
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=10000,
                    schema_overrides={"source1_entity_id": pl.Utf8, "record_entity_id": pl.Utf8})
    out = d.select(pl.col("source1_entity_id").str.slice(3).cast(pl.Int64).alias("s1"),
                   pl.col("record_entity_id").str.slice(1, 1).cast(pl.Int8).alias("src"),
                   pl.col("record_entity_id").str.slice(3).cast(pl.Int64).alias("rid"),
                   pl.col("score").cast(pl.Float64).alias("p"))
    for c in ("predicted", "is_true_match"):
        if c in d.columns:
            out = out.with_columns(d[c].alias(c))
    return out


def read_lists(path) -> pl.DataFrame:
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    col = m.columns[1]
    return (m.select(pl.col("source1_entity_id").str.slice(3).cast(pl.Int64).alias("s1"), pl.col(col).str.split(",").alias("ids"))
            .explode("ids", empty_as_null=True).filter(pl.col("ids").is_not_null() & (pl.col("ids") != ""))
            .select("s1", pl.col("ids").str.slice(1, 1).cast(pl.Int8).alias("src"), pl.col("ids").str.slice(3).cast(pl.Int64).alias("rid")))


def main():
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir"); ap.add_argument("--val-ce", required=True, help="team val band pairs with CE (s1,src,rid,p,y,ce)")
    ap.add_argument("--test", required=True); ap.add_argument("--match", required=True); ap.add_argument("--cand", required=True)
    ap.add_argument("--ce-model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--threshold", type=float, default=0.7)
    a = ap.parse_args()
    work, out = config.work_dir(a.work_dir), Path(a.out)
    vc = pl.read_parquet(a.val_ce)
    lr = LogisticRegression().fit(np.column_stack([_logit(vc["p"].to_numpy()), vc["ce"].to_numpy()]), vc["y"].to_numpy())
    print("blender coefs", lr.coef_.round(3).tolist(), lr.intercept_.round(3).tolist(), flush=True)

    test = read_pairs(a.test)
    band = test.filter(pl.col("p").is_not_null() & (pl.col("p") > LO) & (pl.col("p") < HI))
    print(f"test pairs {test.height:,} | uncertain band {band.height:,}", flush=True)
    df = pair_text(work, "test", band)
    tok = AutoTokenizer.from_pretrained(a.ce_model)
    model = AutoModelForSequenceClassification.from_pretrained(a.ce_model, dtype=torch.float32).to("cuda")
    ce = score(tok, model, df)
    pb = lr.predict_proba(np.column_stack([_logit(df["p"].to_numpy()), ce]))[:, 1]
    inside = df.select(K).with_columns(pl.Series("pb", pb))
    inside.write_parquet(out / "team_test_blended_band.parquet")

    team = read_lists(a.match)
    outside = team.join(inside.select(K), on=K, how="anti")                 # team decisions outside the band
    accepted = inside.filter(pl.col("pb") >= a.threshold).select(K)
    final = pl.concat([outside, accepted]).unique()
    s1_ids = pl.read_parquet(work / "raw" / "test_s1.parquet", columns=["rid"])["rid"]
    io_out.write_id_lists(out / "matching_results.tsv", s1_ids, final, io_out.MATCH_HEADER)
    cand = read_lists(a.cand)
    io_out.write_id_lists(out / "candidate_pairs.tsv", s1_ids, pl.concat([cand, final]).unique(), io_out.CAND_HEADER)
    changed_add = final.join(team, on=K, how="anti").height
    changed_drop = team.join(final, on=K, how="anti").height
    print(f"final links {final.height:,} (team {team.height:,}) | added {changed_add:,} | dropped {changed_drop:,}", flush=True)
    ok = io_out.validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", config.data_dir() / "test")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

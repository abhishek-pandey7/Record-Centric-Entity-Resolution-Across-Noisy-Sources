"""Leaderboard probe: all-empty predictions. Public score == public-test singleton rate.

Usage: python -m src.probe_empty --out submissions/sub00_all_empty
"""
import argparse
from pathlib import Path

import polars as pl

from . import config, io_out, metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir"); ap.add_argument("--data-dir"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = config.work_dir(a.work_dir) / "raw"
    empty = pl.DataFrame(schema={"s1": pl.Int64, "src": pl.Int8, "rid": pl.Int64})

    folds = pl.read_parquet(raw / "train_s1_folds.parquet").filter("is_val")
    truth = pl.read_parquet(raw / "train_gt_pairs.parquet").join(folds.select("s1"), on="s1", how="semi")
    print("val, all-empty:", metric.report(metric.per_entity(empty, truth, folds.select("s1", "country"))))

    s1 = pl.read_parquet(raw / "test_s1.parquet", columns=["rid"])["rid"]
    out = Path(a.out)
    io_out.write_id_lists(out / "matching_results.tsv", s1, empty, io_out.MATCH_HEADER)
    io_out.write_id_lists(out / "candidate_pairs.tsv", s1, empty, io_out.CAND_HEADER)
    ok = io_out.validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", config.data_dir(a.data_dir) / "test")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

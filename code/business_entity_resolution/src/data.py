"""Stage 0: convert the raw TSVs to Parquet with integer ids, explode the ground truth, assign folds.

Records table (work/raw/{split}_s{src}.parquet): rid:Int64, src:Int8, name:Utf8, addr:Utf8, country:Utf8
  - `rid` is the integer after the "S{src}-" prefix; the original id is f"S{src}-{rid}" (checked lossless).
GT pairs (work/raw/train_gt_pairs.parquet): s1:Int64, src:Int8, rid:Int64
S1 entity table (work/raw/train_s1_folds.parquet): s1:Int64, country, n_true:Int32, fold20:Int8, is_val:bool

Usage: python -m src.data [--data-dir D] [--work-dir W] [--check]
"""
import argparse
import time

import numpy as np
import polars as pl

from . import config

TRAIN_COUNTS = {1: 2_206_821, 2: 5_034_616, 3: 5_285_603}
TEST_COUNTS = {1: 1_732_544, 2: 4_887_273, 3: 5_082_316}
GT_LINKS = 7_638_365


def _scan_tsv(path):
    # TSVs are unquoted: quote_char=None keeps apostrophes/quotes inside names literal.
    return pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False)


def stable_fold20(rid: np.ndarray) -> np.ndarray:
    """Deterministic across machines/library versions (plain uint64 arithmetic, wraps on overflow)."""
    x = rid.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = (x * np.uint64(0x9E3779B97F4A7C15)) ^ (x >> np.uint64(29))
        x = x * np.uint64(0xBF58476D1CE4E5B9)
    return ((x >> np.uint64(40)) % np.uint64(config.N_FOLDS20)).astype(np.int8)


def convert_records(ddir, raw):
    for split in config.SPLITS:
        for s in config.SOURCES:
            out = raw / f"{split}_s{s}.parquet"
            if out.exists():
                continue
            t = time.time()
            lf = _scan_tsv(ddir / split / f"{split}_source{s}.tsv")
            lf.select(
                pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("rid"),
                pl.lit(s, dtype=pl.Int8).alias("src"),
                pl.col("entity_id").str.slice(0, 3).alias("_prefix"),
                pl.col("business_name").fill_null("").alias("name"),
                pl.col("business_address").fill_null("").alias("addr"),
                pl.col("country").fill_null("").alias("country"),
                pl.col("entity_id").alias("_eid"),
            ).sink_parquet(tmp := out.with_suffix(".tmp.parquet"))
            # Verify the int id round-trips to the original string, then drop the helper columns.
            df = pl.read_parquet(tmp, memory_map=False)
            bad = df.filter((pl.col("_prefix") != f"S{s}-") |
                            (pl.format("S{}-{}", pl.col("src"), pl.col("rid")) != pl.col("_eid"))).height
            assert bad == 0, f"{out.name}: {bad} ids do not round-trip"
            df.drop("_prefix", "_eid").write_parquet(out)
            tmp.unlink()
            print(f"{out.name}: {df.height:,} rows ({time.time() - t:.0f}s)", flush=True)


def convert_gt(ddir, raw):
    out = raw / "train_gt_pairs.parquet"
    if out.exists():
        return
    gt = _scan_tsv(ddir / "train" / "train_ground_truth.tsv").collect()
    pairs = (gt.select(pl.col("source1_entity_id").str.slice(3).cast(pl.Int64).alias("s1"),
                       pl.col("matched_entity_ids").fill_null("").str.split(",").alias("ids"))
             .explode("ids", empty_as_null=True).filter(pl.col("ids") != "")
             .select("s1", pl.col("ids").str.slice(1, 1).cast(pl.Int8).alias("src"),
                     pl.col("ids").str.slice(3).cast(pl.Int64).alias("rid")))
    pairs.write_parquet(out)
    s1 = gt.select(pl.col("source1_entity_id").str.slice(3).cast(pl.Int64).alias("s1"))
    s1.write_parquet(raw / "train_gt_s1.parquet")
    print(f"GT: {s1.height:,} S1 rows, {pairs.height:,} links", flush=True)


def build_folds(raw):
    out = raw / "train_s1_folds.parquet"
    if out.exists():
        return
    s1 = pl.read_parquet(raw / "train_s1.parquet", columns=["rid", "country"]).rename({"rid": "s1"})
    n_true = pl.read_parquet(raw / "train_gt_pairs.parquet").group_by("s1").len("n_true")
    df = s1.join(n_true, on="s1", how="left").with_columns(pl.col("n_true").fill_null(0).cast(pl.Int32))
    df = df.with_columns(pl.Series("fold20", stable_fold20(df["s1"].to_numpy())))
    df = df.with_columns((pl.col("fold20") < config.VAL_FOLDS).alias("is_val"))
    df.write_parquet(out)


def check(raw):
    ok = True
    for split, counts in (("train", TRAIN_COUNTS), ("test", TEST_COUNTS)):
        for s, n in counts.items():
            got = pl.scan_parquet(raw / f"{split}_s{s}.parquet").select(pl.len()).collect().item()
            ok &= got == n
            print(f"{split}_s{s}: {got:,} (expected {n:,}) {'OK' if got == n else 'MISMATCH'}")
    links = pl.scan_parquet(raw / "train_gt_pairs.parquet").select(pl.len()).collect().item()
    dup = pl.read_parquet(raw / "train_gt_pairs.parquet").select(pl.struct("src", "rid").is_duplicated().sum()).item()
    ok &= links == GT_LINKS and dup == 0
    print(f"GT links: {links:,} (expected {GT_LINKS:,}); records in >1 cluster: {dup}")
    f = pl.read_parquet(raw / "train_s1_folds.parquet")
    gt_s1 = pl.read_parquet(raw / "train_gt_s1.parquet")
    ok &= f.height == gt_s1.height == TRAIN_COUNTS[1] and f.join(gt_s1, on="s1", how="anti").height == 0
    print(f.group_by("is_val", "country").agg(pl.len(), (pl.col("n_true") == 0).mean().alias("singleton_rate"),
                                              pl.col("n_true").mean().alias("mean_cluster")).sort("is_val", "country"))
    print("CHECK", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir"); ap.add_argument("--work-dir"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ddir, raw = config.data_dir(a.data_dir), config.work_dir(a.work_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    convert_records(ddir, raw)
    convert_gt(ddir, raw)
    build_folds(raw)
    if a.check and not check(raw):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

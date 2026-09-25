"""Stage 2: record-centric candidate generation.

Every S2/S3 record belongs to at most one S1 entity, so each record retrieves its own top-K S1 entities
(same country) and an S1's candidate set is the union of the records that retrieved it.

Key types (hashed to UInt64 with the country; only keys with S1 document frequency <= cap are kept):
  n  name token                      c  concatenated name (domains, spacing)
  a  number x address word           m  number x name token
  p  number x 4-char name prefix     q  pair of name tokens (generic names without numbers)
  x  name token x address word
Score of (record, S1) = sum over matched keys of log(N_country / df_key). Top-K per record by score.

Output: work/cand/{split}_{tag}.parquet with src, rid, s1, score, nkeys, types (bitmask), rank
Usage: python -m src.blocking --split train --queries val [--k 10]
"""
import argparse
import time

import polars as pl

from . import config

KEY_TYPES = {  # name: (bit, df cap)
    "n": (1, 300), "c": (2, 50), "a": (4, 200), "m": (8, 50), "p": (16, 50), "q": (32, 100), "x": (64, 50),
    "w": (128, 50),   # pair of address words (house number drifted / missing)
    "g": (256, 50),   # 6-gram of the concatenated name (glued domains, typos); query side only for 1-token names
}
GRAM = 6
MAX_NUMS = 3


def _prep(df: pl.DataFrame) -> pl.DataFrame:
    tok = pl.col("name_norm").str.split(" ").list.eval(
        pl.element().filter((pl.element().str.len_chars() >= 2) & ~pl.element().str.contains(r"^\d+$"))).list.unique()
    atok = pl.col("addr_norm").str.split(" ").list.eval(
        pl.element().filter((pl.element().str.len_chars() >= 3) & ~pl.element().str.contains(r"^\d+$"))).list.unique()
    nums = pl.col("nums").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.head(MAX_NUMS)
    ids = [c for c in ("s1", "src", "rid") if c in df.columns]
    return df.select(*ids, "country", "name_concat", tok.alias("nt"), atok.alias("at"), nums.alias("ns"))


def _keys(p: pl.DataFrame, kind: str, id_col: str) -> pl.DataFrame:
    """(key:UInt64, id) rows for one key type."""
    b = p.select(pl.col(id_col) if id_col in p.columns else pl.struct("src", "rid").alias(id_col),
                 "country", "nt", "at", "ns", "name_concat")
    if kind == "n":
        e = b.explode("nt").select(id_col, "country", pl.col("nt").alias("u"), pl.lit("").alias("v"))
    elif kind == "c":
        e = b.filter(pl.col("name_concat").str.len_chars() >= 4).select(id_col, "country", pl.col("name_concat").alias("u"), pl.lit("").alias("v"))
    elif kind == "a":
        e = b.explode("ns").explode("at").select(id_col, "country", pl.col("ns").alias("u"), pl.col("at").alias("v"))
    elif kind == "m":
        e = b.explode("ns").explode("nt").select(id_col, "country", pl.col("ns").alias("u"), pl.col("nt").alias("v"))
    elif kind == "p":
        e = (b.explode("ns").explode("nt").filter(pl.col("nt").str.len_chars() >= 5)
             .select(id_col, "country", pl.col("ns").alias("u"), pl.col("nt").str.slice(0, 4).alias("v")))
    elif kind == "q":
        e = (b.filter(pl.col("nt").list.len().is_between(2, 5))
             .with_columns(pl.col("nt").list.sort().alias("nt2")).explode("nt").explode("nt2")
             .filter(pl.col("nt") < pl.col("nt2")).select(id_col, "country", pl.col("nt").alias("u"), pl.col("nt2").alias("v")))
    elif kind == "x":
        e = b.explode("nt").explode("at").select(id_col, "country", pl.col("nt").alias("u"), pl.col("at").alias("v"))
    elif kind == "w":
        e = (b.filter(pl.col("at").list.len() >= 2).with_columns(pl.col("at").list.head(8).alias("at"))
             .with_columns(pl.col("at").alias("at2")).explode("at").explode("at2")
             .filter(pl.col("at") < pl.col("at2")).select(id_col, "country", pl.col("at").alias("u"), pl.col("at2").alias("v")))
    elif kind == "g":
        if id_col == "rec":
            b = b.filter(pl.col("nt").list.len() <= 1)
        b = b.filter(pl.col("name_concat").str.len_chars() >= GRAM)
        e = (b.with_columns(pl.int_ranges(0, pl.col("name_concat").str.len_chars() - GRAM + 1).alias("o")).explode("o")
             .select(id_col, "country", pl.col("name_concat").str.slice(pl.col("o"), GRAM).alias("u"), pl.lit("").alias("v")))
    else:
        raise ValueError(kind)
    e = e.drop_nulls(["u", "v"]).filter(pl.col("u") != "")
    key = pl.concat_str([pl.lit(kind), pl.col("country"), pl.col("u"), pl.col("v")], separator="\x1f").hash(seed=7)
    return e.select(key.alias("key"), id_col, "country").unique()


def build_index(s1: pl.DataFrame) -> dict:
    """{kind: DataFrame(key, s1, w)} restricted to keys with df <= cap."""
    p = _prep(s1.rename({"rid": "s1"}))
    n_country = p.group_by("country").len("N")
    index = {}
    for kind, (_, cap) in KEY_TYPES.items():
        k = _keys(p, kind, "s1")
        df = k.group_by("key").agg(pl.len().alias("df"), pl.col("country").first())
        df = df.filter(pl.col("df") <= cap).join(n_country, on="country")
        df = df.select("key", (pl.col("N") / pl.col("df")).log().cast(pl.Float32).alias("w"))
        index[kind] = k.join(df, on="key").select("key", "s1", "w")
    return index


def query(index: dict, recs: pl.DataFrame, k: int) -> pl.DataFrame:
    p = _prep(recs)
    parts = []
    for kind, (bit, _) in KEY_TYPES.items():
        q = _keys(p, kind, "rec").unnest("rec")
        parts.append(q.join(index[kind], on="key").select("src", "rid", "s1", "w", pl.lit(bit, pl.Int16).alias("bit")))
    hits = pl.concat(parts)
    agg = hits.group_by("src", "rid", "s1").agg(pl.col("w").sum().alias("score"), pl.len().alias("nkeys"),
                                                pl.col("bit").unique().sum().alias("types"))
    agg = agg.with_columns(pl.col("score").rank("ordinal", descending=True).over("src", "rid").alias("rank"))
    agg = agg.filter(pl.col("rank") <= k)
    # record-level retrieval context, computed over the full top-K before any downstream subsampling
    top = pl.col("score").max().over("src", "rid")
    return agg.with_columns(pl.len().over("src", "rid").cast(pl.Int16).alias("r_ncand"),
                            (pl.col("score") / top).cast(pl.Float32).alias("score_rel"),
                            (top - pl.col("score")).cast(pl.Float32).alias("score_gap"))


def run(work, split, queries, k, chunk=400_000):
    norm, out_dir = work / "norm", work / "cand"
    out_dir.mkdir(exist_ok=True)
    t = time.time()
    s1 = pl.read_parquet(norm / f"{split}_s1.parquet")
    if split == "train" and config.DELETE_FRAC > 0:  # orphan world, see config.DELETE_FRAC
        from .data import deleted_mask
        s1 = s1.filter(~pl.Series(deleted_mask(s1["rid"].to_numpy())))
        print(f"orphan world: S1 index {s1.height:,} after deleting {config.DELETE_FRAC:.0%}", flush=True)
    index = build_index(s1)
    print(f"index built: {sum(v.height for v in index.values()):,} postings ({time.time() - t:.0f}s)", flush=True)
    recs = pl.concat([pl.read_parquet(norm / f"{split}_s{s}.parquet") for s in (2, 3)])
    if queries == "val":  # true records of val entities + 5% of all other records (for size statistics)
        raw = work / "raw"
        val = pl.read_parquet(raw / "train_s1_folds.parquet").filter("is_val").select("s1")
        vr = pl.read_parquet(raw / "train_gt_pairs.parquet").join(val, on="s1", how="semi").select("src", "rid")
        mask = recs.select(pl.struct("src", "rid").is_in(vr.select(pl.struct("src", "rid")).to_series()) |
                           (pl.col("rid").hash(seed=3) % 20 == 0)).to_series()
        recs = recs.filter(mask)
    out = []
    for o in range(0, recs.height, chunk):
        out.append(query(index, recs.slice(o, chunk), k))
        print(f"  queried {min(o + chunk, recs.height):,}/{recs.height:,} ({time.time() - t:.0f}s)", flush=True)
    cand = pl.concat(out)
    path = out_dir / f"{split}_{queries}.parquet"
    cand.write_parquet(path)
    print(f"{path.name}: {cand.height:,} pairs for {recs.height:,} records ({time.time() - t:.0f}s)", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir"); ap.add_argument("--split", default="train")
    ap.add_argument("--queries", default="val", choices=["val", "all"]); ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    run(config.work_dir(a.work_dir), a.split, a.queries, a.k)


if __name__ == "__main__":
    main()

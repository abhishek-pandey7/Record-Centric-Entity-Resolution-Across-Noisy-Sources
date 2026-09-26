"""Stage 3: pairwise features for (S2/S3 record, S1 entity) candidates.

All features are relative similarities or record flags; no country / id columns are model inputs.
Country-level statistics (IDF, name commonness, S1 vocabulary) are computed from the same split's S1 table,
so they adapt to unseen countries (France) automatically.

Usage (library): build(work, split, cand_path) -> DataFrame with keys (src, rid, s1) + feature columns.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Indel, JaroWinkler

from . import context

KEYS = ["src", "rid", "s1"]
NORM_COLS = ["rid", "src", "country", "name_norm", "name_concat", "legal", "is_domain", "is_indic",
             "addr_norm", "nums", "state", "addr_empty"]


def _pd(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def s1_stats(s1: pl.DataFrame):
    """Per-country token IDF over S1 names and addresses, name commonness, S1 name vocabulary."""
    n = s1.group_by("country").len("N")
    def idf(col, name):
        t = (s1.select("country", pl.col(col).str.split(" ").list.unique().alias("t")).explode("t", empty_as_null=True)
             .filter(pl.col("t") != "").group_by("country", "t").len("df").join(n, on="country"))
        return t.select("country", "t", (pl.col("N") / pl.col("df")).log().cast(pl.Float32).alias(name))
    common = s1.group_by("country", "name_norm").len("name_common")
    return idf("name_norm", "idf_n"), idf("addr_norm", "idf_a"), common


def _set_feats(df, a, b, prefix, idf, idf_col):
    """Jaccard + IDF-weighted overlap (share of the S1 side's IDF mass that the record covers)."""
    la, lb = pl.col(a).str.split(" ").list.unique(), pl.col(b).str.split(" ").list.unique()
    df = df.with_columns(la.alias("_ta"), lb.alias("_tb"))
    df = df.with_columns(pl.col("_ta").list.set_intersection("_tb").alias("_ti"))
    inter, union = pl.col("_ti").list.len(), pl.col("_ta").list.set_union("_tb").list.len()
    df = df.with_columns((inter / union).fill_nan(0).cast(pl.Float32).alias(f"{prefix}_jacc"),
                         inter.cast(pl.Int16).alias(f"{prefix}_ninter"))
    rows = pl.int_range(pl.len()).alias("_row")
    df = df.with_columns(rows)
    def mass(col):
        return (df.select("_row", "country", pl.col(col).alias("t")).explode("t", empty_as_null=True)
                .join(idf, on=["country", "t"], how="left").group_by("_row").agg(pl.col(idf_col).fill_null(0).sum().alias("m")))
    m_i, m_b = mass("_ti"), mass("_tb")
    m_a = mass("_ta")
    df = (df.join(m_i.rename({"m": "_mi"}), on="_row", how="left").join(m_b.rename({"m": "_mb"}), on="_row", how="left")
          .join(m_a.rename({"m": "_ma"}), on="_row", how="left"))
    df = df.with_columns((pl.col("_mi") / pl.col("_mb")).fill_nan(0).fill_null(0).cast(pl.Float32).alias(f"{prefix}_idf_cov_s1"),
                         (pl.col("_mi") / pl.col("_ma")).fill_nan(0).fill_null(0).cast(pl.Float32).alias(f"{prefix}_idf_cov_rec"),
                         pl.col("_mi").fill_null(0).cast(pl.Float32).alias(f"{prefix}_idf_inter"))
    return df.drop("_ta", "_tb", "_ti", "_row", "_mi", "_mb", "_ma")


def pair_features(c: pl.DataFrame, rec: pl.DataFrame, s1: pl.DataFrame, stats) -> pl.DataFrame:
    """c: candidates (src, rid, s1, score, nkeys, types, rank). rec/s1: normalized tables."""
    idf_n, idf_a, common = stats
    df = (c.join(rec.select(NORM_COLS), on=["src", "rid"], how="left")
          .join(s1.select(NORM_COLS).rename({k: f"e_{k}" for k in NORM_COLS if k != "rid"}).rename({"rid": "s1"}), on="s1", how="left"))
    rn, en = df["name_norm"].to_list(), df["e_name_norm"].to_list()
    rc, ec = df["name_concat"].to_list(), df["e_name_concat"].to_list()
    ra, ea = df["addr_norm"].to_list(), df["e_addr_norm"].to_list()
    r1 = [x.split(" ", 1)[0] for x in df["nums"].to_list()]
    e1 = [x.split(" ", 1)[0] for x in df["e_nums"].to_list()]
    f = {
        "n_ratio": _pd(rn, en, fuzz.ratio), "n_tsort": _pd(rn, en, fuzz.token_sort_ratio),
        "n_tset": _pd(rn, en, fuzz.token_set_ratio), "n_partial": _pd(rn, en, fuzz.partial_ratio),
        "c_ratio": _pd(rc, ec, fuzz.ratio), "c_jw": _pd(rc, ec, JaroWinkler.normalized_similarity),
        "c_partial": _pd(rc, ec, fuzz.partial_ratio),
        "a_ratio": _pd(ra, ea, fuzz.ratio), "a_tsort": _pd(ra, ea, fuzz.token_sort_ratio),
        "a_tset": _pd(ra, ea, fuzz.token_set_ratio), "a_partial": _pd(ra, ea, fuzz.partial_ratio),
        "num1_sim": _pd(r1, e1, Indel.normalized_similarity),
    }
    df = df.with_columns([pl.Series(k, v) for k, v in f.items()])
    df = _set_feats(df, "name_norm", "e_name_norm", "n", idf_n, "idf_n")
    df = _set_feats(df, "addr_norm", "e_addr_norm", "a", idf_a, "idf_a")
    rnum, enum_ = pl.col("nums").str.split(" ").list.unique(), pl.col("e_nums").str.split(" ").list.unique()
    has_r, has_e = pl.col("nums") != "", pl.col("e_nums") != ""
    initials = pl.col("e_name_norm").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")
    df = df.with_columns(
        pl.when(has_r & has_e).then(rnum.list.set_intersection(enum_).list.len()).otherwise(-1).cast(pl.Int8).alias("num_inter"),
        pl.when(has_r & has_e).then((pl.col("nums").str.split(" ").list.first() == pl.col("e_nums").str.split(" ").list.first()).cast(pl.Int8)).otherwise(-1).alias("num1_eq"),
        has_r.cast(pl.Int8).alias("r_has_num"), has_e.cast(pl.Int8).alias("e_has_num"),
        pl.when((pl.col("state") != "") & (pl.col("e_state") != "")).then((pl.col("state") == pl.col("e_state")).cast(pl.Int8)).otherwise(-1).alias("state_eq"),
        (pl.col("legal") == pl.col("e_legal")).cast(pl.Int8).alias("legal_eq"),
        ((pl.col("name_concat") == initials) & (pl.col("name_concat").str.len_chars() >= 2)).cast(pl.Int8).alias("initials_match"),
        (pl.col("name_concat").str.len_chars().cast(pl.Int32) - pl.col("e_name_concat").str.len_chars().cast(pl.Int32)).abs().cast(pl.Int16).alias("n_len_diff"),
        pl.col("name_norm").str.count_matches(" ").add(1).cast(pl.Int8).alias("r_n_tokens"),
        pl.col("is_domain").cast(pl.Int8), pl.col("is_indic").cast(pl.Int8), pl.col("addr_empty").cast(pl.Int8),
        (pl.col("src") == 3).cast(pl.Int8).alias("is_s3"),
    )
    df = df.join(common.rename({"name_norm": "e_name_norm", "country": "e_country"}), on=["e_country", "e_name_norm"], how="left")
    df = pl.concat([df, context.number_features(df["nums"].to_list(), df["e_nums"].to_list())], how="horizontal")
    feats = [k for k in df.columns if k not in NORM_COLS and not k.startswith("e_") and not k.startswith("_")
             and k not in ("s1",)] + ["name_common", "is_domain", "is_indic", "addr_empty"]
    return df.select(KEYS + [k for k in dict.fromkeys(feats) if k not in KEYS])


def featurize(work, split: str, cand_path, out_path, full_cand_path=None, chunk=1_500_000):
    """Compute features for every candidate pair, chunked by record so per-record context stays intact.
    full_cand_path: the complete blocking output (all records), used to find each S1's siblings (rank-1 records)."""
    import time
    norm = work / "norm"
    t = time.time()
    cand = pl.read_parquet(cand_path).sort("src", "rid", "rank")
    s1 = pl.read_parquet(norm / f"{split}_s1.parquet")
    stats = s1_stats(s1)
    rank1 = (pl.read_parquet(full_cand_path or cand_path, columns=["src", "rid", "s1", "rank"])
             .filter(pl.col("rank") == 1).drop("rank").join(cand.select("s1").unique(), on="s1", how="semi"))
    need = pl.concat([cand.select("src", "rid"), rank1.select("src", "rid")]).unique()
    rec = pl.concat([pl.read_parquet(norm / f"{split}_s{s}.parquet").join(need, on=["src", "rid"], how="semi") for s in (2, 3)])
    s1 = s1.join(cand.select(pl.col("s1").alias("rid")).unique(), on="rid", how="semi")
    # chunk boundaries on record changes
    rec_id = cand.select((pl.col("src").cast(pl.Int64) * 10**10 + pl.col("rid")).alias("k"))["k"]
    bounds, start = [], 0
    while start < cand.height:
        end = min(start + chunk, cand.height)
        while end < cand.height and rec_id[end] == rec_id[end - 1]:
            end += 1
        bounds.append((start, end)); start = end
    parts = []
    for i, (a, b) in enumerate(bounds):
        parts.append(pair_features(cand.slice(a, b - a), rec, s1, stats))
        print(f"  features {b:,}/{cand.height:,} ({time.time() - t:.0f}s)", flush=True)
    out = pl.concat(parts)
    from . import config
    if config.USE_SIBLING:
        sib = context.sibling_features(out.select(KEYS), rank1, rec)
        out = out.join(sib, on=KEYS, how="left")
        print(f"  sibling features ({time.time() - t:.0f}s)", flush=True)
    out.write_parquet(out_path)
    return out_path

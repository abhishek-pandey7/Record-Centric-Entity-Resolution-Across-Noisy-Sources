"""Extra pair features: number-edit type and sibling context.

Number edits: the noise that keeps a record's identity deletes or pads digits (8880 -> 880, 302 -> 0302), while
near-copy distractors substitute digits (1920 -> 1924). We classify the relation between house numbers.

Siblings: for entity e, the siblings of record r are the other records whose top-ranked candidate is e.
A true record usually agrees with its siblings (same address even under a trade name); a near-copy
distractor disagrees on the number. Rank-based, so no labels or model outputs are involved.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

REL_EQ, REL_DEL, REL_INS, REL_SUB, REL_OTHER, REL_NONE = 0, 1, 2, 3, 4, -1


def _subseq(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


def num_relation(a: str, b: str) -> int:
    """Relation of record number a to S1 number b."""
    if not a or not b:
        return REL_NONE
    if a == b:
        return REL_EQ
    if len(a) < len(b) and _subseq(a, b):
        return REL_DEL
    if len(a) > len(b) and _subseq(b, a):
        return REL_INS
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
        return REL_SUB
    return REL_OTHER


def number_features(r_nums: list, e_nums: list) -> pl.DataFrame:
    rel1, best_rel, min_dist, n_extra = [], [], [], []
    for rn, en in zip(r_nums, e_nums):
        ra, ea = rn.split(), en.split()
        rel1.append(num_relation(ra[0] if ra else "", ea[0] if ea else ""))
        if ra and ea:
            rels = [num_relation(x, y) for x in ra for y in ea]
            best_rel.append(min(rels, key=lambda v: (v == REL_OTHER, v == REL_SUB, v)))
            min_dist.append(min(Levenshtein.distance(x, y) for x in ra for y in ea))
            es = set(ea)
            n_extra.append(sum(x not in es for x in ra))
        else:
            best_rel.append(REL_NONE); min_dist.append(-1); n_extra.append(-1)
    return pl.DataFrame({"num1_rel": pl.Series(rel1, dtype=pl.Int8), "num_best_rel": pl.Series(best_rel, dtype=pl.Int8),
                         "num_min_dist": pl.Series(min_dist, dtype=pl.Int16), "num_n_extra": pl.Series(n_extra, dtype=pl.Int16)})


LEGAL_CLASS = {"llc": "LLC", "inc": "INC", "corp": "INC", "co": "CO", "ltd": "LTD", "pvt": "LTD", "public": "PUB",
               "llp": "LLP", "lp": "LP", "plc": "PLC", "pllc": "PLLC", "pc": "PC", "pa": "PC"}


def legal_relation(r_legal: list, e_legal: list) -> pl.DataFrame:
    """Decoys switch the legal TYPE (Inc->LLC, Limited->LLP) while true noise keeps an equivalent form (Ltd/Limited).
    legal_rel: 0 same class / one side missing, 1 partial overlap, 2 conflicting classes. On validation, accepted
    pairs with a conflict are 28% false vs ~1% otherwise."""
    rel, r_n, e_n = [], [], []
    for a, b in zip(r_legal, e_legal):
        A = {LEGAL_CLASS.get(t, t) for t in a.split()} if a else set()
        B = {LEGAL_CLASS.get(t, t) for t in b.split()} if b else set()
        rel.append(0 if (not A or not B or A == B) else (1 if A & B else 2))
        r_n.append(len(A)); e_n.append(len(B))
    return pl.DataFrame({"legal_rel": pl.Series(rel, dtype=pl.Int8), "r_legal_n": pl.Series(r_n, dtype=pl.Int8),
                         "e_legal_n": pl.Series(e_n, dtype=pl.Int8)})


def sibling_features(pairs: pl.DataFrame, rank1: pl.DataFrame, rec: pl.DataFrame, n_chunks=8) -> pl.DataFrame:
    """pairs: src, rid, s1 (rows to featurize). rank1: src, rid, s1 for every record's top candidate
    (full pool). rec: normalized records (src, rid, name_norm, addr_norm, nums)."""
    info = rec.select("src", "rid", "name_norm", "addr_norm", pl.col("nums").str.split(" ").list.first().fill_null("").alias("n1"))
    sib = rank1.join(pairs.select("s1").unique(), on="s1", how="semi").join(info, on=["src", "rid"])
    sib = sib.rename({"src": "s_src", "rid": "s_rid", "name_norm": "s_name", "addr_norm": "s_addr", "n1": "s_n1"})
    out = []
    ids = pairs.select("src", "rid", "s1").with_columns((pl.col("s1").hash(seed=5) % n_chunks).alias("_c"))
    for c in range(n_chunks):
        p = ids.filter(pl.col("_c") == c).drop("_c").join(info, on=["src", "rid"], how="left")
        x = p.join(sib, on="s1", how="inner").filter(~((pl.col("s_src") == pl.col("src")) & (pl.col("s_rid") == pl.col("rid"))))
        both_addr = (pl.col("addr_norm") != "") & (pl.col("s_addr") != "")
        a_sim = process.cpdist(x["addr_norm"].to_list(), x["s_addr"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        n_sim = process.cpdist(x["name_norm"].to_list(), x["s_name"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32)
        x = x.with_columns(pl.when(both_addr).then(pl.Series(a_sim)).otherwise(None).alias("_as"), pl.Series("_ns", n_sim),
                           pl.when((pl.col("n1") != "") & (pl.col("s_n1") != "")).then((pl.col("n1") == pl.col("s_n1")).cast(pl.Float32)).otherwise(None).alias("_ne"),
                           ((pl.col("addr_norm") == pl.col("s_addr")) & both_addr).cast(pl.Int8).alias("_aeq"))
        g = x.group_by("src", "rid", "s1").agg(
            pl.len().cast(pl.Int16).alias("sib_n"), pl.col("_as").max().alias("sib_addr_best"), pl.col("_as").mean().alias("sib_addr_mean"),
            pl.col("_ns").max().alias("sib_name_best"), pl.col("_ne").mean().alias("sib_num1_agree"), pl.col("_aeq").max().alias("sib_addr_exact"))
        out.append(p.select("src", "rid", "s1").join(g, on=["src", "rid", "s1"], how="left"))
    res = pl.concat(out).with_columns(pl.col("sib_n").fill_null(0))
    return res.with_columns([pl.col(c).fill_null(-1) for c in ("sib_addr_best", "sib_addr_mean", "sib_name_best", "sib_num1_agree", "sib_addr_exact")])

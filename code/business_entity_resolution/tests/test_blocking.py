import polars as pl

from src.blocking import build_index, query
from src.normalize import normalize_frame


def _norm(rows, src):
    df = pl.DataFrame(rows, schema=["rid", "name", "addr", "country"], orient="row")
    return normalize_frame(df.with_columns(pl.lit(src, pl.Int8).alias("src")))


S1 = _norm([
    (1, "Leland and Runk Offshore Co", "448 Chism Street, Albany, TX", "US"),
    (2, "Great Resources LLC", "8880 Hawthorn Point, Westerville, OH", "US"),
    (3, "Dental Group", "12 Main Street, Springfield, IL", "US"),
    (4, "Dental Group", "77 Oak Road, Dayton, OH", "US"),
    (5, "Maison Mistral SNC", "20 Avenue Raymond Poincare, Merignac, Nouvelle-Aquitaine", "France"),
    (6, "Blue Harbor", "5 Quay Lane, Oakvale, Zedland", "Narnia"),
], 1)
RECS = _norm([
    (11, "Deltafaye", "448-D Chism St, Albany, Texas", "US"),                    # DBA -> 1 (address only)
    (12, "greatresources.com", "8880 Hawmhorn Point, Westerville, Ohio", "US"),  # domain + typo -> 2
    (13, "DENTAL GROUP", "77 OAK RD, DAYTON, OH", "US"),                          # generic name -> 4, not 3
    (14, "SNC Maison Mistral", "20 AV RAYMOND POINCARE, MERIGNAC, Gironde", "France"),  # -> 5
    (15, "Blue Harbour", "5 Quay Ln, Oakvale", "Narnia"),                          # unseen country -> 6
], 2)


def test_toy_retrieval_top1():
    cand = query(build_index(S1), RECS, k=3)
    top = dict(cand.filter(pl.col("rank") == 1).select("rid", "s1").iter_rows())
    assert top == {11: 1, 12: 2, 13: 4, 14: 5, 15: 6}


def test_no_cross_country_candidates():
    cand = query(build_index(S1), RECS, k=10)
    assert cand.filter((pl.col("rid") == 14) & (pl.col("s1") != 5)).height == 0

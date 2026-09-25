"""Blocking recall report on val: recall@k, breakdowns, entity-level F0.5 ceiling, candidate volume.

Usage: python -m src.eval_blocking [--cand work/cand/train_val.parquet]
"""
import argparse

import polars as pl

from . import config, metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir"); ap.add_argument("--cand")
    a = ap.parse_args()
    work = config.work_dir(a.work_dir)
    raw, norm = work / "raw", work / "norm"
    cand = pl.read_parquet(a.cand or work / "cand" / "train_val.parquet")
    from .data import load_folds
    folds = load_folds(raw)
    val = folds.filter(pl.col("is_val") & ~pl.col("deleted")).select("s1", "country")
    truth = pl.read_parquet(raw / "train_gt_pairs.parquet").join(val.select("s1"), on="s1", how="semi")
    recs = pl.concat([pl.read_parquet(norm / f"train_s{s}.parquet",
                                      columns=["src", "rid", "is_indic", "addr_empty", "is_domain"]) for s in (2, 3)])
    t = truth.join(recs, on=["src", "rid"], how="left").join(val, on="s1")
    hit = cand.select("src", "rid", "s1", "rank", "types")
    t = t.join(hit, on=["src", "rid", "s1"], how="left")

    print(f"val true links: {t.height:,}")
    for k in (1, 2, 3, 5, 10):
        print(f"  recall@{k:<2} = {(t['rank'] <= k).fill_null(False).mean():.4f}")
    found = t.with_columns((pl.col("rank") <= 10).fill_null(False).alias("found"))
    for col in ("country", "src", "is_indic", "addr_empty", "is_domain"):
        g = found.group_by(col).agg(pl.len().alias("n"), pl.col("found").mean().alias("recall")).sort(col)
        print(f"  by {col}: " + ", ".join(f"{r[col]}={r['recall']:.4f} (n={r['n']:,})" for r in g.iter_rows(named=True)))
    from .blocking import KEY_TYPES
    bits = {k: v[0] for k, v in KEY_TYPES.items()}
    f = found.filter("found")
    print("  share of found links hit by key type: " + ", ".join(
        f"{n}={(f['types'] & b > 0).mean():.3f}" for n, b in bits.items()))
    print("  found ONLY via one type: " + ", ".join(
        f"{n}={(f['types'] == b).mean():.3f}" for n, b in bits.items()))

    # Entity-level ceiling: perfect precision on whatever blocking found (k=10).
    pred = found.filter("found").select("s1", "src", "rid")
    pe = metric.per_entity(pred, truth, val)
    print("F0.5 ceiling @10 (perfect matcher):", metric.report(pe))

    # Volume: non-val records are a 5% sample of the pool -> extrapolate candidates per S1.
    vol = cand.join(truth.select("src", "rid"), on=["src", "rid"], how="anti")
    n_rec = vol.select(pl.struct("src", "rid").n_unique()).item()
    for k in (1, 3, 5, 10):
        per_rec = vol.filter(pl.col("rank") <= k).height / max(n_rec, 1)
        print(f"  k={k:<2}: {per_rec:.2f} candidates/record  ~ {per_rec * 10.32e6 / 2.2068e6:.1f} per S1 (full pool)")


if __name__ == "__main__":
    main()

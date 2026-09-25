"""Official metric: macro F0.5 over Source 1 entities, singletons included.

Per entity: both empty -> 1.0; any other case with TP == 0 -> 0.0;
otherwise F0.5 = 1.25*P*R / (0.25*P + R) = 1.25*TP / (0.25*|truth| + |pred|).
"""
import polars as pl


def f05(tp: int, n_true: int, n_pred: int) -> float:
    if n_true == 0 and n_pred == 0:
        return 1.0
    if tp == 0:
        return 0.0
    return 1.25 * tp / (0.25 * n_true + n_pred)


def score_sets(pred: dict, truth: dict) -> float:
    """Reference implementation. truth: {s1: set(ids)} over ALL evaluated entities; pred may omit entities."""
    tot = 0.0
    for e, t in truth.items():
        p = pred.get(e, set())
        tot += f05(len(p & t), len(t), len(p))
    return tot / len(truth)


def per_entity(pred_pairs: pl.DataFrame, truth_pairs: pl.DataFrame, entities: pl.DataFrame) -> pl.DataFrame:
    """Vectorized per-entity F0.5.

    pred_pairs / truth_pairs: columns s1, src, rid (one row per link; pred must be deduplicated).
    entities: column s1 (+ any grouping columns) = the full evaluation set.
    Returns entities with n_true, n_pred, tp, f05.
    """
    k = ["s1", "src", "rid"]
    pred_pairs = pred_pairs.select(k).unique()
    tp = pred_pairs.join(truth_pairs.select(k), on=k, how="inner").group_by("s1").len("tp")
    n_pred = pred_pairs.group_by("s1").len("n_pred")
    n_true = truth_pairs.group_by("s1").len("n_true")
    df = (entities.join(n_true, on="s1", how="left").join(n_pred, on="s1", how="left").join(tp, on="s1", how="left")
          .with_columns(pl.col("n_true", "n_pred", "tp").fill_null(0)))
    return df.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred"))).alias("f05"))


def report(pe: pl.DataFrame, by=("country",)) -> str:
    """Overall macro F0.5 plus breakdowns (by the given columns and by cluster-size bucket)."""
    pe = pe.with_columns(pl.col("n_true").clip(0, 7).alias("true_size"))
    lines = [f"macro F0.5 = {pe['f05'].mean():.5f}  (n={pe.height:,})"]
    for col in (*by, "true_size"):
        if col in pe.columns:
            g = pe.group_by(col).agg(pl.len().alias("n"), pl.col("f05").mean().alias("f05")).sort(col)
            lines.append(f"  by {col}: " + ", ".join(f"{r[col]}={r['f05']:.4f} (n={r['n']:,})" for r in g.iter_rows(named=True)))
    return "\n".join(lines)

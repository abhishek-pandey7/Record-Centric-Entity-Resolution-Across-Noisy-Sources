"""Stage 5: turn pair probabilities into per-S1 match lists.

v0 decision: exclusivity (a record keeps only its highest-p S1, since every record belongs to at most one
entity) followed by a global probability threshold chosen to maximize val macro F0.5.
"""
import numpy as np
import polars as pl

from . import metric


def exclusive(pred: pl.DataFrame) -> pl.DataFrame:
    """pred: src, rid, s1, p. Keep each record's argmax S1 only."""
    return pred.sort("p", descending=True).unique(subset=["src", "rid"], keep="first", maintain_order=True)


def apply_threshold(pred: pl.DataFrame, thr: float) -> pl.DataFrame:
    return exclusive(pred).filter(pl.col("p") >= thr).select("s1", "src", "rid")


def sweep(pred: pl.DataFrame, truth: pl.DataFrame, entities: pl.DataFrame, grid=None):
    """Returns (best_thr, best_score, table) for macro F0.5 on the given entities."""
    grid = grid if grid is not None else np.round(np.arange(0.20, 0.96, 0.025), 3)
    ex = exclusive(pred)
    rows = []
    for thr in grid:
        pe = metric.per_entity(ex.filter(pl.col("p") >= thr).select("s1", "src", "rid"), truth, entities)
        rows.append((float(thr), float(pe["f05"].mean())))
    table = pl.DataFrame(rows, schema=["thr", "f05"], orient="row")
    best = table.sort("f05", descending=True).row(0)
    return best[0], best[1], table

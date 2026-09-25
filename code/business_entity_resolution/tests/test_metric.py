import random

import polars as pl
import pytest

from src.metric import f05, per_entity, score_sets


def test_pdf_example():
    pred = {1: {"S2-00047", "S2-00193", "S3-00812"}}
    truth = {1: {"S2-00047", "S3-00812"}}
    assert score_sets(pred, truth) == pytest.approx(0.7142857, abs=1e-6)


@pytest.mark.parametrize("tp,nt,npred,want", [
    (0, 0, 0, 1.0),   # singleton, predicted empty
    (0, 0, 2, 0.0),   # singleton, predicted something
    (0, 3, 0, 0.0),   # missed everything
    (0, 3, 2, 0.0),   # all wrong
    (2, 2, 2, 1.0),   # perfect
    (1, 4, 1, 0.625),  # precise but incomplete
])
def test_edge_cases(tp, nt, npred, want):
    assert f05(tp, nt, npred) == pytest.approx(want)


def test_vectorized_matches_reference():
    rng = random.Random(0)
    truth, pred, tp_rows, pp_rows = {}, {}, [], []
    for e in range(500):
        t = {(rng.choice([2, 3]), rng.randrange(40)) for _ in range(rng.choice([0, 0, 1, 3, 5]))}
        p = {x for x in t if rng.random() < 0.7} | {(rng.choice([2, 3]), 100 + rng.randrange(40)) for _ in range(rng.choice([0, 0, 1, 2]))}
        truth[e], pred[e] = t, p
        tp_rows += [(e, s, r) for s, r in t]
        pp_rows += [(e, s, r) for s, r in p]
    schema = {"s1": pl.Int64, "src": pl.Int8, "rid": pl.Int64}
    pe = per_entity(pl.DataFrame(pp_rows, schema=schema, orient="row"),
                    pl.DataFrame(tp_rows, schema=schema, orient="row"),
                    pl.DataFrame({"s1": list(truth)}, schema={"s1": pl.Int64}))
    assert pe["f05"].mean() == pytest.approx(score_sets(pred, truth))

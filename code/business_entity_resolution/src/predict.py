"""Stage 6: test inference -> output/matching_results.tsv + output/candidate_pairs.tsv (+ validator).

candidate_pairs.tsv is exactly the set of pairs the model scores; matches are a subset of it.
Usage: python -m src.predict [--k 10] [--threshold T] [--out-dir D]
"""
import argparse
import json
import time

import lightgbm as lgb
import polars as pl

from . import blocking, config, decide, features, io_out


def predict_chunked(model, df: pl.DataFrame, cols, chunk=2_000_000):
    """Predict in float32 chunks with progress lines (avoids one huge silent float64 matrix)."""
    import numpy as np
    out, t = [], time.time()
    for o in range(0, df.height, chunk):
        x = df.slice(o, chunk).select(cols).to_numpy().astype(np.float32, copy=False)
        out.append(model.predict(x))
        print(f"  predicted {min(o + chunk, df.height):,}/{df.height:,} ({time.time() - t:.0f}s)", flush=True)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir"); ap.add_argument("--data-dir"); ap.add_argument("--out-dir")
    ap.add_argument("--block-k", type=int, default=5); ap.add_argument("--k", type=int, default=3); ap.add_argument("--threshold", type=float)
    a = ap.parse_args()
    work, out = config.work_dir(a.work_dir), config.out_dir(a.out_dir)
    t = time.time()
    cand_path = work / "cand" / "test_all.parquet"
    if not cand_path.exists():
        blocking.run(work, "test", "all", a.block_k)
    # Same candidate definition as training: top-block_k context features, pairs with rank <= k.
    fpath = work / "feat" / f"test_v3_k{a.k}.parquet"
    if not fpath.exists():
        (work / "feat").mkdir(exist_ok=True)
        sub_path = work / "cand" / f"test_k{a.k}.parquet"
        pl.read_parquet(cand_path).filter(pl.col("rank") <= a.k).write_parquet(sub_path)
        features.featurize(work, "test", sub_path, fpath, full_cand_path=cand_path)
    meta = json.load(open(work / "model" / "stage1_meta.json"))
    model = lgb.Booster(model_file=str(work / "model" / "stage1.txt"))
    feat = pl.read_parquet(fpath)
    feat = feat.with_columns(pl.Series("p", predict_chunked(model, feat, meta["features"])))
    thr = a.threshold if a.threshold is not None else meta["threshold"]
    matches = decide.apply_threshold(feat.select("src", "rid", "s1", "p"), thr)
    s1_ids = pl.read_parquet(work / "raw" / "test_s1.parquet", columns=["rid"])["rid"]
    io_out.write_id_lists(out / "matching_results.tsv", s1_ids, matches, io_out.MATCH_HEADER)
    io_out.write_id_lists(out / "candidate_pairs.tsv", s1_ids, feat.select("s1", "src", "rid"), io_out.CAND_HEADER)
    n_s1 = s1_ids.len()
    print(f"threshold {thr}: {matches.height:,} links, {matches['s1'].n_unique():,}/{n_s1:,} S1 non-empty "
          f"({time.time() - t:.0f}s)")
    ok = io_out.validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", config.data_dir(a.data_dir) / "test")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Stage 4: LightGBM pair classifier (stage 1) + val evaluation.

Pairs whose S1 is a val entity form the validation set; all other pairs train the model.
Every candidate is used (no negative subsampling) so probabilities stay calibratable.

Usage: python -m src.train --cand work/cand/train_all.parquet [--max-train-pairs N]
"""
import argparse
import json
import time

import lightgbm as lgb
import polars as pl

from . import config, data, decide, features, metric

KEYS = features.KEYS
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0, seed=7)


def label(feat: pl.DataFrame, raw) -> pl.DataFrame:
    gt = pl.read_parquet(raw / "train_gt_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    return feat.join(gt, on=KEYS, how="left").with_columns(pl.col("y").fill_null(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir"); ap.add_argument("--cand", required=True)
    ap.add_argument("--max-train-pairs", type=int, default=0); ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--k", type=int, default=3, help="use candidates with rank <= k")
    ap.add_argument("--train-folds", type=int, default=4, help="train entities used: N of the 17 train folds")
    a = ap.parse_args()
    work = config.work_dir(a.work_dir)
    raw, feat_dir, model_dir = work / "raw", work / "feat", work / "model"
    feat_dir.mkdir(exist_ok=True); model_dir.mkdir(exist_ok=True)
    t = time.time()
    fpath = feat_dir / f"train_v5_k{a.k}_tf{a.train_folds}.parquet"
    if not fpath.exists():
        # Keep val entities in full + a fraction of train entities (folds 3..3+train_folds-1 of 20).
        folds0 = pl.read_parquet(raw / "train_s1_folds.parquet")
        keep = folds0.filter(pl.col("is_val") | pl.col("fold20").is_between(config.VAL_FOLDS, config.VAL_FOLDS + a.train_folds - 1))
        sub = (pl.read_parquet(a.cand).filter(pl.col("rank") <= a.k).join(keep.select("s1"), on="s1", how="semi"))
        sub_path = work / "cand" / f"train_sub_k{a.k}_tf{a.train_folds}.parquet"
        sub.write_parquet(sub_path)
        features.featurize(work, "train", sub_path, fpath, full_cand_path=a.cand)
    df = label(pl.read_parquet(fpath), raw)
    folds = data.load_folds(raw)
    df = df.join(folds.select("s1", "is_val"), on="s1")
    fcols = [c for c in df.columns if c not in KEYS + ["y", "is_val"]]
    tr, va = df.filter(~pl.col("is_val")), df.filter("is_val")
    if a.max_train_pairs and tr.height > a.max_train_pairs:
        keep = tr.select("s1").unique().sample(fraction=a.max_train_pairs / tr.height, seed=1)
        tr = tr.join(keep, on="s1", how="semi")
    print(f"train pairs {tr.height:,} (pos {tr['y'].mean():.3f}) | val pairs {va.height:,} | {len(fcols)} features", flush=True)

    dtr = lgb.Dataset(tr.select(fcols).to_numpy().astype("float32"), tr["y"].to_numpy(), feature_name=fcols, free_raw_data=True)
    dva = lgb.Dataset(va.select(fcols).to_numpy().astype("float32"), va["y"].to_numpy(), reference=dtr)
    model = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    model.save_model(str(model_dir / "stage1.txt"))
    imp = sorted(zip(fcols, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(f, int(g)) for f, g in imp[:15]])

    from .predict import predict_chunked
    va = va.with_columns(pl.Series("p", predict_chunked(model, va, fcols)))
    va.select(KEYS + ["p", "y"]).write_parquet(model_dir / "val_pred_stage1.parquet")
    va.select(KEYS + ["p", "y"]).write_parquet(config.out_dir() / "val_scores.parquet")
    ents = folds.filter(pl.col("is_val") & ~pl.col("deleted")).select("s1", "country")
    truth = pl.read_parquet(raw / "train_gt_pairs.parquet").join(ents.select("s1"), on="s1", how="semi")
    thr, best, table = decide.sweep(va.select(KEYS + ["p"]), truth, ents)
    pe = metric.per_entity(decide.apply_threshold(va.select(KEYS + ["p"]), thr), truth, ents)
    print(f"best threshold {thr} -> val macro F0.5 {best:.5f}")
    print(metric.report(pe))
    json.dump({"threshold": thr, "val_f05": best, "features": fcols, "best_iter": model.best_iteration},
              open(model_dir / "stage1_meta.json", "w"), indent=1)
    print(f"done ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()

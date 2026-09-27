"""Stage 7 (optional): cross-encoder re-scoring of uncertain pairs.

A small transformer (microsoft/deberta-v3-small, MIT licence, ~140M params) reads both records' normalized text
together and learns the generator's noise-vs-decoy patterns directly. It is trained out-of-fold on the LightGBM
validation pairs (two halves of the val entities), then blended with the LightGBM probability.

Usage:
  python -m src.cross_encoder train --scores <val_scores.parquet> --out <dir>
  python -m src.cross_encoder score --split test --scores <test_scores.parquet> --out <dir>
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from . import config

MODEL = "microsoft/deberta-v3-small"
LO, HI = 0.02, 0.98          # uncertain band re-scored by the cross-encoder
MAX_LEN = 96


def pair_text(work: Path, split: str, pairs: pl.DataFrame) -> pl.DataFrame:
    norm = work / "norm"
    cols = ["name_norm", "legal", "addr_norm"]
    txt = (pl.concat_str([pl.col("name_norm"), pl.lit(" ["), pl.col("legal"), pl.lit("] | "), pl.col("addr_norm")]))
    # lazy scans + semi-joins: only the records these pairs need are materialized (memory-light)
    need_s1 = pairs.select("s1").unique().lazy()
    s1 = (pl.scan_parquet(norm / f"{split}_s1.parquet").select(["rid"] + cols).rename({"rid": "s1"})
          .join(need_s1, on="s1", how="semi").select("s1", txt.alias("t_e")).collect())
    recs = []
    for s in (2, 3):
        need = pairs.filter(pl.col("src") == s).select("rid").unique().lazy()
        recs.append(pl.scan_parquet(norm / f"{split}_s{s}.parquet").select(["src", "rid"] + cols)
                    .join(need, on="rid", how="semi").select("src", "rid", txt.alias("t_r")).collect())
    return pairs.join(s1, on="s1", how="left").join(pl.concat(recs), on=["src", "rid"], how="left")


class Batcher:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, rows):
        a, b, y = zip(*rows)
        enc = self.tok(list(a), list(b), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc


def build_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    # float32 master weights; autocast handles fp16 compute (fp16 weights break GradScaler)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, dtype=torch.float32)
    return tok, model


def train_one(df: pl.DataFrame, epochs=1, bs=64, lr=3e-5):
    tok, model = build_model()
    dev = "cuda"
    model.to(dev)
    rows = list(zip(df["t_r"].to_list(), df["t_e"].to_list(), df["y"].cast(pl.Float32).to_list()))
    dl = DataLoader(rows, batch_size=bs, shuffle=True, collate_fn=Batcher(tok), num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.06 * total)) * max(0.0, (total - s) / (total * 0.94)))
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train(); t = time.time(); step = 0
    for _ in range(epochs):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logit.float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 500 == 0:
                print(f"    step {step}/{total} loss {loss.item():.4f} ({time.time() - t:.0f}s)", flush=True)
    return tok, model


@torch.no_grad()
def score(tok, model, df: pl.DataFrame, bs=256) -> np.ndarray:
    model.eval(); out = []; t = time.time()
    a, b = df["t_r"].to_list(), df["t_e"].to_list()
    for i in range(0, len(a), bs):
        enc = tok(a[i:i + bs], b[i:i + bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
        if (i // bs) % 400 == 0:
            print(f"    scored {min(i + bs, len(a)):,}/{len(a):,} ({time.time() - t:.0f}s)", flush=True)
    return np.concatenate(out)


def cmd_train(a):
    work = config.work_dir(a.work_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    va = pl.read_parquet(a.scores)
    half = (pl.col("s1").hash(seed=13) % 2).alias("half")
    va = va.with_columns(half)
    band = (pl.col("p") > LO) & (pl.col("p") < HI)
    oof = []
    for h in (0, 1):
        tr = va.filter(pl.col("half") == h)
        # all uncertain pairs + a sample of confident ones so the model also sees easy cases
        tr = pl.concat([tr.filter(band), tr.filter(~band).sample(n=min(a.easy, tr.filter(~band).height), seed=h)])
        tr = pair_text(work, "train", tr).sample(fraction=1.0, shuffle=True, seed=h)
        print(f"half {h}: training on {tr.height:,} pairs (pos {tr['y'].mean():.3f})", flush=True)
        tok, model = train_one(tr, epochs=a.epochs)
        model.save_pretrained(out / f"ce_half{h}"); tok.save_pretrained(out / f"ce_half{h}")
        ev = pair_text(work, "train", va.filter((pl.col("half") != h) & band))
        oof.append(ev.select("src", "rid", "s1", "p", "y").with_columns(pl.Series("ce", score(tok, model, ev))))
        oof[-1].write_parquet(out / f"val_ce_oof_half{h}.parquet")   # checkpoint: survive a later crash
        del model, tr, ev; torch.cuda.empty_cache()
    pl.concat(oof).write_parquet(out / "val_ce_oof.parquet")
    print("saved", out / "val_ce_oof.parquet")


def cmd_oof(a):
    """Score the other half's uncertain val pairs with an already-trained half model (recovery / checking)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    work = config.work_dir(a.work_dir); out = Path(a.out); h = a.half
    va = pl.read_parquet(a.scores).with_columns((pl.col("s1").hash(seed=13) % 2).alias("half"))
    ev = pair_text(work, "train", va.filter((pl.col("half") != h) & (pl.col("p") > LO) & (pl.col("p") < HI)))
    tok = AutoTokenizer.from_pretrained(out / f"ce_half{h}")
    model = AutoModelForSequenceClassification.from_pretrained(out / f"ce_half{h}", dtype=torch.float32).to("cuda")
    ev.select("src", "rid", "s1", "p", "y").with_columns(pl.Series("ce", score(tok, model, ev))).write_parquet(out / f"val_ce_oof_half{h}.parquet")
    print("saved", out / f"val_ce_oof_half{h}.parquet")


def cmd_score(a):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    work = config.work_dir(a.work_dir); out = Path(a.out)
    s = pl.read_parquet(a.scores, columns=["src", "rid", "s1", "p"]).filter((pl.col("p") > LO) & (pl.col("p") < HI))
    df = pair_text(work, a.split, s)
    ces = []
    halves = [h for h in (0, 1) if (out / f"ce_half{h}" / "model.safetensors").exists()]
    print("cross-encoder models:", halves, "| pairs to score:", df.height, flush=True)
    for h in halves:
        tok = AutoTokenizer.from_pretrained(out / f"ce_half{h}")
        model = AutoModelForSequenceClassification.from_pretrained(out / f"ce_half{h}", dtype=torch.float32).to("cuda")
        ces.append(score(tok, model, df)); del model; torch.cuda.empty_cache()
    df.select("src", "rid", "s1", "p").with_columns(pl.Series("ce", sum(ces) / len(ces))).write_parquet(out / f"{a.split}_ce.parquet")
    print("saved", out / f"{a.split}_ce.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score", "oof"])
    ap.add_argument("--work-dir"); ap.add_argument("--scores", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--easy", type=int, default=120_000); ap.add_argument("--half", type=int, default=0)
    a = ap.parse_args()
    {"train": cmd_train, "score": cmd_score, "oof": cmd_oof}[a.cmd](a)


if __name__ == "__main__":
    main()

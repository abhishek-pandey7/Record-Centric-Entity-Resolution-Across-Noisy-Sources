"""Stage 1: text normalization shared by blocking and features.

Per record output (work/norm/{split}_s{src}.parquet):
  rid, src, country,
  name_norm   space-joined core name tokens (legal forms, honorifics, particles, repeats removed)
  name_concat core tokens without spaces (so "greatresources.com" == "Great Resources LLC")
  legal       sorted canonical legal-form tokens ("ltd pvt")
  is_domain   name was written as a web domain
  is_indic    name contains Indic-script characters
  addr_norm   space-joined canonical address tokens, admin components (state/region) removed
  nums        space-joined number tokens in order of appearance, leading zeros stripped
  state       canonical state/region ("" when absent or unrecognized)
  addr_empty  raw address was empty

Usage: python -m src.normalize [--work-dir W] [--sample N] [--workers K]
"""
import argparse
import json
import multiprocessing
import os
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from . import config, dicts

INDIC_RE = re.compile(r"[ऀ-෿]")
_COMBINING = re.compile(r"[̀-ͯ]")  # Latin diacritics only; Indic vowel signs are left intact
_DOMAIN = re.compile(r"\b([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|co\.in|in|biz|info|co|io|fr)\b")
_MS = re.compile(r"\bm\s*/\s*s\b")
_NAME_PUNCT = re.compile(r"[^\w\sऀ-෿]|_")
_ADDR_PUNCT = re.compile(r"[^\w\s/ऀ-෿]|_")
_ORDINAL = re.compile(r"\b(\d+)\s*(?:st|nd|rd|th)\b")
_FRACTION = re.compile(r"(?<![\d/])[1-3]/[2-4](?![\d/])")  # "1705 1/2" only; keeps "126/22", "45/2"
_NA = re.compile(r"\bn\s*/\s*a\b")
_NUMERO = re.compile(r"\bn\s*[°º]\s*")
_DIGITS = re.compile(r"\d+")
_SPLIT_ALNUM = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")

_ALIASES = None
LEARNED_ALIASES = {}  # {country: {alias: state}} mined from train pairs (native-script state names)


def strip_accents(s: str) -> str:
    return _COMBINING.sub("", unicodedata.normalize("NFKD", s))


def comp_key(s: str) -> str:
    """Normalization used for whole address components (state detection)."""
    return " ".join(re.sub(r"[^\wऀ-෿]+", " ", strip_accents(s).lower()).split())


def _dedupe(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


TRANSLIT = {}  # Indic token -> Latin token, learned from train pairs (src.translit)


def norm_name(raw: str):
    from .translit import to_latin
    indic = bool(INDIC_RE.search(raw))
    s = strip_accents(to_latin(raw, TRANSLIT) if indic else raw).lower().replace("&", " and ")
    s = _MS.sub(" ", s)
    domain = bool(_DOMAIN.search(s))
    if domain:
        s = _DOMAIN.sub(r" \1 ", s)
    s = s.replace(".", "").replace("'", "")        # p.c. -> pc, l.l.c. -> llc, ernesta's -> ernestas
    toks = _NAME_PUNCT.sub(" ", s).split()
    legal = sorted({dicts.LEGAL[t] for t in toks if t in dicts.LEGAL})
    core = _dedupe([t for t in toks if t not in dicts.LEGAL and t not in dicts.NAME_STOP])
    if not core:                                   # name made only of legal/stop words: keep what we have
        core = _dedupe(toks)
    return " ".join(core), "".join(core), " ".join(legal), domain, indic


def norm_addr(raw: str, country: str):
    global _ALIASES
    if _ALIASES is None:
        _ALIASES = dicts.state_aliases()
        for c, m in LEARNED_ALIASES.items():
            _ALIASES.setdefault(c, {}).update(m)
    aliases = _ALIASES.get(country, {})
    state, toks = "", []
    for comp in raw.split(","):
        key = comp_key(comp)
        if not key:
            continue
        if key in aliases:
            state = state or aliases[key]
            continue
        s = strip_accents(comp).lower()
        s = _NUMERO.sub(" no ", s)
        s = _ORDINAL.sub(r"\1", s)
        s = _NA.sub(" ", _FRACTION.sub(" ", s))
        s = s.replace("city of ", " ")
        s = _ADDR_PUNCT.sub(" ", s).replace("/", " ")
        s = _SPLIT_ALNUM.sub(" ", s)               # 448d -> 448 d, b104 -> b 104
        for t in s.split():
            if t.isdigit():
                toks.append(t.lstrip("0") or "0")
                continue
            t = dicts.STREET.get(t, t)
            if t not in dicts.ADDR_STOP:
                toks.append(t)
    toks = _dedupe(toks)
    nums = " ".join(t for t in toks if t.isdigit())
    return " ".join(toks), nums, state


def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    names = [norm_name(n) for n in df["name"].to_list()]
    addrs = [norm_addr(a, c) for a, c in zip(df["addr"].to_list(), df["country"].to_list())]
    return pl.DataFrame({
        "rid": df["rid"], "src": df["src"], "country": df["country"],
        "name_norm": [x[0] for x in names], "name_concat": [x[1] for x in names], "legal": [x[2] for x in names],
        "is_domain": [x[3] for x in names], "is_indic": [x[4] for x in names],
        "addr_norm": [x[0] for x in addrs], "nums": [x[1] for x in addrs], "state": [x[2] for x in addrs],
        "addr_empty": df["addr"].str.strip_chars() == "",
    })


def learn_state_aliases(work, min_support=20, min_purity=0.9):
    """Map address components the base tables don't know (e.g. native-script state names such as
    'दिल्ली', 'தமிழ்நாடு') to the state of the matched S1 record, using train ground-truth pairs."""
    raw = work / "raw"
    base = dicts.state_aliases()

    def comps(df):
        return (df.with_columns(pl.col("addr").str.split(",").alias("comp")).explode("comp", empty_as_null=True)
                .with_columns(pl.col("comp").map_elements(comp_key, return_dtype=pl.Utf8))
                .filter(pl.col("comp") != ""))

    s1 = comps(pl.read_parquet(raw / "train_s1.parquet", columns=["rid", "addr", "country"]))
    s1 = (s1.with_columns(pl.struct("country", "comp").map_elements(
              lambda r: base.get(r["country"], {}).get(r["comp"], ""), return_dtype=pl.Utf8).alias("s1_state"))
          .filter(pl.col("s1_state") != "").group_by("rid").agg(pl.col("s1_state").first()).rename({"rid": "s1"}))
    gt = pl.read_parquet(raw / "train_gt_pairs.parquet").join(s1, on="s1")
    rows = []
    for s in (2, 3):
        r = pl.read_parquet(raw / f"train_s{s}.parquet", columns=["rid", "src", "addr", "country"])
        r = comps(r.filter(pl.col("addr").str.contains(r"[ऀ-෿]")))
        r = r.filter(pl.col("comp").str.contains(r"[ऀ-෿]") & ~pl.col("comp").str.contains(r"[a-z0-9]"))
        rows.append(r.join(gt, on=["src", "rid"]).select("country", "comp", "s1_state"))
    c = pl.concat(rows).group_by("country", "comp", "s1_state").len()
    c = c.with_columns((pl.col("len") / pl.col("len").sum().over("country", "comp")).alias("purity"))
    c = c.filter((pl.col("len") >= min_support) & (pl.col("purity") >= min_purity))
    learned = {}
    for r in c.iter_rows(named=True):
        learned.setdefault(r["country"], {})[r["comp"]] = r["s1_state"]
    return learned


def _init(learned, translit_table):
    global LEARNED_ALIASES, _ALIASES, TRANSLIT
    LEARNED_ALIASES, _ALIASES, TRANSLIT = learned, None, translit_table


def _work(args):
    path, offset, length = args
    return normalize_frame(pl.scan_parquet(path).slice(offset, length).collect())


def run(work, workers, chunk=250_000):
    raw, out_dir = work / "raw", work / "norm"
    out_dir.mkdir(exist_ok=True)
    alias_path = out_dir / "learned_state_aliases.json"
    if not alias_path.exists():
        alias_path.write_text(json.dumps(learn_state_aliases(work), ensure_ascii=False, indent=1), encoding="utf-8")
    learned = json.loads(alias_path.read_text(encoding="utf-8"))
    print("learned state aliases:", {c: len(m) for c, m in learned.items()}, flush=True)
    tl_path = out_dir / "translit.json"
    if not tl_path.exists():
        from . import translit
        tl_path.write_text(json.dumps(translit.fit(work), ensure_ascii=False), encoding="utf-8")
    table = json.loads(tl_path.read_text(encoding="utf-8"))
    print("transliteration dictionary:", len(table), flush=True)
    # spawn, not fork: forking after Polars started its thread pool can deadlock (Linux/Kaggle)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init, initargs=(learned, table)) as ex:
        for split in config.SPLITS:
            for s in config.SOURCES:
                out = out_dir / f"{split}_s{s}.parquet"
                if out.exists():
                    continue
                t = time.time()
                src = raw / f"{split}_s{s}.parquet"
                n = pl.scan_parquet(src).select(pl.len()).collect().item()
                parts = list(ex.map(_work, [(src, o, chunk) for o in range(0, n, chunk)]))
                pl.concat(parts).write_parquet(out)
                print(f"{out.name}: {n:,} rows ({time.time() - t:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir"); ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    work = config.work_dir(a.work_dir)
    if a.sample:
        pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(a.sample * 6)
        for split, s in (("train", 2), ("train", 3), ("test", 3)):
            df = pl.read_parquet(work / "raw" / f"{split}_s{s}.parquet").sample(a.sample * 3, seed=1)
            n = normalize_frame(df)
            print(pl.concat([df.select("name", "addr"), n.select("name_norm", "legal", "addr_norm", "nums", "state")],
                            how="horizontal_extend"))
        return
    run(work, a.workers)


if __name__ == "__main__":
    main()

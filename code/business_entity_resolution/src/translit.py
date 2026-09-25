"""Indic-script names -> Latin tokens.

S2/S3 Indic names are word-by-word transliterations of the English name
("शक्ति प्रोडक्ट्स एलएलपी" == "Shakti Products LLP"). We learn a token dictionary from train ground-truth
pairs (positional alignment when token counts match) and fall back to a rule-based romanizer built from
Unicode character names, which works for every Indic script without external data.
"""
import re
import unicodedata
from collections import Counter, defaultdict

import polars as pl

INDIC = re.compile(r"[ऀ-෿]")
_TOKEN = re.compile(r"[^\s]+")
_STRIP = re.compile(r"[^\wऀ-෿]")

_VOWEL = {"AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u", "E": "e", "EE": "e", "AI": "ai", "O": "o",
          "OO": "o", "AU": "au", "VOCALIC R": "ri", "A": "a", "SHORT E": "e", "SHORT O": "o", "CANDRA E": "e",
          "CANDRA O": "o"}


def clean_token(t: str) -> str:
    return _STRIP.sub("", t.lower())


def romanize(tok: str) -> str:
    """Rough phonetic romanization from Unicode names (consonant + inherent 'a', vowel signs, virama)."""
    out = []
    pending_a = False
    for ch in tok:
        name = unicodedata.name(ch, "")
        if " LETTER " in name:
            if pending_a:
                out.append("a")
            base = name.split(" LETTER ", 1)[1]
            if base in _VOWEL:                      # independent vowel
                out.append(_VOWEL[base]); pending_a = False
            else:
                cons = base.lower().replace(" ", "")
                out.append(cons[:-1] if cons.endswith("a") and len(cons) > 1 else cons); pending_a = True
        elif " VOWEL SIGN " in name:
            out.append(_VOWEL.get(name.split(" VOWEL SIGN ", 1)[1], "")); pending_a = False
        elif name.endswith("SIGN VIRAMA") or name.endswith("SIGN VIRAMA "):
            pending_a = False
        elif name.endswith("SIGN ANUSVARA") or name.endswith("SIGN CANDRABINDU"):
            if pending_a:
                out.append("a")
            out.append("n"); pending_a = False
        elif ch.isascii() and ch.isalnum():
            if pending_a:
                out.append("a")
            out.append(ch.lower()); pending_a = False
    if pending_a and len(out) > 1:
        pass                                        # drop final inherent 'a' (schwa deletion)
    return "".join(out)


def fit(work, min_support=2, min_purity=0.5) -> dict:
    """Learn {indic_token: latin_token} from train pairs whose record name is Indic."""
    raw = work / "raw"
    gt = pl.read_parquet(raw / "train_gt_pairs.parquet")
    s1 = pl.read_parquet(raw / "train_s1.parquet", columns=["rid", "name"]).rename({"rid": "s1", "name": "s1_name"})
    counts = defaultdict(Counter)
    for s in (2, 3):
        r = pl.read_parquet(raw / f"train_s{s}.parquet", columns=["rid", "src", "name"])
        r = r.filter(pl.col("name").str.contains(r"[ऀ-෿]"))
        pairs = r.join(gt, on=["src", "rid"]).join(s1, on="s1").select("name", "s1_name")
        for ind, lat in pairs.iter_rows():
            it = [clean_token(t) for t in _TOKEN.findall(ind)]
            lt = [clean_token(t) for t in _TOKEN.findall(lat.replace("&", " and "))]
            it, lt = [t for t in it if t], [t for t in lt if t]
            if len(it) == len(lt):
                for a, b in zip(it, lt):
                    if INDIC.search(a) and not INDIC.search(b):
                        counts[a][b] += 1
    out = {}
    for tok, c in counts.items():
        best, n = c.most_common(1)[0]
        if n >= min_support and n / sum(c.values()) >= min_purity:
            out[tok] = best
    return out


def to_latin(name: str, table: dict) -> str:
    """Replace Indic tokens by dictionary hits, else by romanization. Latin tokens are kept."""
    if not INDIC.search(name):
        return name
    toks = []
    for t in _TOKEN.findall(name):
        k = clean_token(t)
        if INDIC.search(k):
            toks.append(table.get(k) or romanize(k))
        else:
            toks.append(t)
    return " ".join(toks)

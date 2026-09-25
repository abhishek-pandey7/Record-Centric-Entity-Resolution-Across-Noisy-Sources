"""Paths and constants shared by all pipeline stages.

Paths resolve from CLI args first, then env vars (BER_DATA_DIR, BER_WORK_DIR, BER_OUT_DIR),
then defaults relative to the repo layout (<root>/student_resource/dataset, <root>/work, <root>/output).
"""
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]          # code/business_entity_resolution
REPO_ROOT = PKG_DIR.parents[1]                          # project root in local dev

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)

# Entity-level split: fold20 = stable_hash(s1 rid) % 20; folds < VAL_FOLDS are validation (15%).
N_FOLDS20 = 20
VAL_FOLDS = 3
N_OOF = 5


def data_dir(arg=None) -> Path:
    return Path(arg or os.environ.get("BER_DATA_DIR") or REPO_ROOT / "student_resource" / "dataset")


def work_dir(arg=None) -> Path:
    p = Path(arg or os.environ.get("BER_WORK_DIR") or REPO_ROOT / "work")
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_dir(arg=None) -> Path:
    p = Path(arg or os.environ.get("BER_OUT_DIR") or REPO_ROOT / "output")
    p.mkdir(parents=True, exist_ok=True)
    return p

"""Writers for matching_results.tsv / candidate_pairs.tsv, and a wrapper around the official validator.

Both files: one row per test S1 entity (all of them), ids as "S{src}-{rid}" joined by commas, empty when none.
"""
import subprocess
import sys
from pathlib import Path

import polars as pl

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def write_id_lists(path, s1_ids: pl.Series, pairs: pl.DataFrame, header):
    """s1_ids: every S1 rid in the evaluated split; pairs: columns s1, src, rid (duplicates removed here)."""
    lists = (pairs.select("s1", "src", "rid").unique()
             .sort("s1", "src", "rid")
             .group_by("s1", maintain_order=True)
             .agg(pl.format("S{}-{}", pl.col("src"), pl.col("rid")).str.join(",").alias(header[1])))
    out = (pl.DataFrame({"s1": s1_ids}).unique()
           .join(lists, on="s1", how="left")
           .select(pl.format("S1-{}", pl.col("s1")).alias(header[0]), pl.col(header[1]).fill_null("")))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
    return out.height


def validate(matching, candidate, test_dir) -> bool:
    validator = Path(test_dir).parents[1] / "utils" / "validate_submission.py"
    cmd = [sys.executable, str(validator), "--matching", str(matching), "--test-dir", str(test_dir)]
    if candidate:
        cmd += ["--candidate", str(candidate)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip()[-2000:])
    return r.returncode == 0

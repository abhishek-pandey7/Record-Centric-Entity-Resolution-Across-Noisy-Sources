"""End-to-end run: data -> normalize -> train blocking -> train model -> test inference + validation.

Usage:
  python -m src.run_all --data-dir <.../student_resource/dataset> --work-dir <scratch> --out-dir <output>
Each stage skips work whose outputs already exist, so a crashed run can be resumed with the same command.
"""
import argparse
import os
import subprocess
import sys
import time


def sh(args, env):
    print(f"\n=== {' '.join(args)}", flush=True)
    t = time.time()
    subprocess.run([sys.executable, "-m", *args], check=True, env=env)
    print(f"=== done in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True); ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-folds", type=int, default=6, help="train entities used: N of the 17 train folds")
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--block-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    a = ap.parse_args()
    env = dict(os.environ, BER_DATA_DIR=a.data_dir, BER_WORK_DIR=a.work_dir, BER_OUT_DIR=a.out_dir,
               PYTHONIOENCODING="utf-8")
    cand = os.path.join(a.work_dir, "cand", "train_all.parquet")
    sh(["src.data", "--check"], env)
    sh(["src.normalize", "--workers", str(a.workers)], env)
    if not os.path.exists(cand):
        sh(["src.blocking", "--split", "train", "--queries", "all", "--k", str(a.block_k)], env)
    if not os.path.exists(os.path.join(a.work_dir, "model", "stage1.txt")):
        sh(["src.train", "--cand", cand, "--k", str(a.k), "--train-folds", str(a.train_folds)], env)
    sh(["src.predict", "--block-k", str(a.block_k), "--k", str(a.k)], env)


if __name__ == "__main__":
    main()

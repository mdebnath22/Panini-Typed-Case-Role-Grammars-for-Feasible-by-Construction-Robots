"""Merges per-body result shards (written by experiments.rl_train_mjx when
run with --body_id/--body_index, i.e. one SLURM array task per body) into
the single rl_results.json that experiments.rank_correlation_analysis
expects.

Usage:
    python3 -m experiments.merge_rl_shards --out_dir results_refit/rank_corr

Safe to re-run at any time, including while array tasks are still running
-- it just merges whatever shards exist so far and reports how many of the
manifest's feasible bodies are still missing.
"""
from __future__ import annotations
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results_refit/rank_corr")
    args = ap.parse_args()

    shard_dir = os.path.join(args.out_dir, "rl_results_shards")
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.json")))
    if not shard_paths:
        print(f"No shards found in {shard_dir}. Nothing to merge.")
        return

    results_path = os.path.join(args.out_dir, "rl_results.json")
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            for row in json.load(f):
                existing[row["body_id"]] = row

    n_new = 0
    for p in shard_paths:
        with open(p) as f:
            row = json.load(f)
        if row["body_id"] not in existing:
            n_new += 1
        existing[row["body_id"]] = row

    merged = list(existing.values())
    with open(results_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(shard_paths)} shards -> {len(merged)} total results "
         f"({n_new} new) written to {results_path}")

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        n_feasible = sum(1 for b in manifest["bodies"] if b.get("geometric_feasible"))
        missing = n_feasible - len(merged)
        if missing > 0:
            print(f"{missing} of {n_feasible} feasible bodies still have no shard "
                 "(array job may still be running, or some tasks failed).")
        else:
            print(f"All {n_feasible} feasible bodies accounted for.")
    print(f"Next: python3 -m experiments.rank_correlation_analysis --out_dir {args.out_dir}")


if __name__ == "__main__":
    main()

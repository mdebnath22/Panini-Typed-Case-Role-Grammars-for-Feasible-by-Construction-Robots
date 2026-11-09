"""Analysis half of the proxy-vs-RL rank-correlation experiment.

Reads manifest.json (written by rank_correlation_harness.py, proxy fitness
for each body) and rl_results.json (written by your GPU-side training run,
per the schema in manifest['gpu_side_contract']['results_file_schema']),
joins them on body_id, and reports Spearman and Kendall-tau rank
correlation between the cheap CPG proxy and the trained-policy result --
the number that tells you whether the paper's proxy-fitness search
(everything in Sections "Task-Directed Morphology Search" onward) is a
reasonable stand-in for what RL training would actually find, or whether
it's systematically misranking bodies.

Runnable before the GPU run exists: it will tell you the manifest is ready
and print the exact file it's waiting for, so you can validate the harness
end-to-end before spending GPU time.

Usage:
    python3 -m experiments.rank_correlation_analysis --out_dir results_refit/rank_corr
"""
from __future__ import annotations
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results_refit/rank_corr")
    ap.add_argument("--rl_results", type=str, default=None,
                    help="Defaults to <out_dir>/rl_results.json")
    args = ap.parse_args()

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"No manifest at {manifest_path}. Run rank_correlation_harness.py first.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)

    rl_path = args.rl_results or os.path.join(args.out_dir, "rl_results.json")
    if not os.path.exists(rl_path):
        schema = manifest["gpu_side_contract"]["results_file_schema"]
        print(f"Manifest found ({len(manifest['bodies'])} bodies, "
             f"{manifest['n_geometrically_feasible']} geometrically feasible) but no RL "
             f"results yet at {rl_path}.")
        print("Waiting on the GPU-side training run. Expected file:")
        print(json.dumps(schema, indent=2))
        return

    with open(rl_path) as f:
        rl_rows = json.load(f)
    rl_by_id = {r["body_id"]: r["rl_fitness"] for r in rl_rows}

    proxy_vals, rl_vals, ids = [], [], []
    skipped = []
    for b in manifest["bodies"]:
        bid = b["body_id"]
        if not b.get("geometric_feasible", False):
            continue
        if bid not in rl_by_id:
            skipped.append(bid)
            continue
        proxy_vals.append(b["proxy_fitness"])
        rl_vals.append(rl_by_id[bid])
        ids.append(bid)

    if skipped:
        print(f"Note: {len(skipped)} feasible bodies in the manifest have no matching "
             f"RL result yet (e.g. training still running / failed): {skipped[:10]}"
             f"{' ...' if len(skipped) > 10 else ''}")

    n = len(proxy_vals)
    if n < 4:
        print(f"Only {n} matched (body, proxy, rl) triples -- too few for a meaningful "
             "correlation. Need at least a handful of completed RL runs.")
        return

    try:
        from scipy.stats import spearmanr, kendalltau
    except ImportError:
        print("scipy not available; install it to compute Spearman/Kendall correlation. "
             f"Raw matched pairs (n={n}) written to {args.out_dir}/matched_pairs.json instead.")
        with open(os.path.join(args.out_dir, "matched_pairs.json"), "w") as f:
            json.dump({"body_id": ids, "proxy_fitness": proxy_vals, "rl_fitness": rl_vals}, f, indent=2)
        return

    rho, p_spearman = spearmanr(proxy_vals, rl_vals)
    tau, p_kendall = kendalltau(proxy_vals, rl_vals)

    print(f"n = {n} matched bodies")
    print(f"Spearman rho = {rho:.4f}  (p = {p_spearman:.4g})")
    print(f"Kendall tau  = {tau:.4f}  (p = {p_kendall:.4g})")

    out = {"n": n, "spearman_rho": float(rho), "spearman_p": float(p_spearman),
          "kendall_tau": float(tau), "kendall_p": float(p_kendall),
          "body_id": ids, "proxy_fitness": proxy_vals, "rl_fitness": rl_vals}
    out_path = os.path.join(args.out_dir, "rank_correlation_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Full result written to {out_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(proxy_vals, rl_vals, alpha=0.7)
        ax.set_xlabel("CPG-proxy fitness")
        ax.set_ylabel("RL-trained fitness")
        ax.set_title(f"Proxy vs. RL fitness (Spearman rho={rho:.2f}, n={n})")
        fig_path = os.path.join(args.out_dir, "rank_correlation_scatter.png")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        print(f"Scatter plot written to {fig_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()

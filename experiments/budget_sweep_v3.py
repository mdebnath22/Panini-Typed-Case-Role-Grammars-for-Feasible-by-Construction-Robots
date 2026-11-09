"""
Budget sweep (60 / 120 / 200) for PĀṇini (panini_search, formerly
grammar_guided_cem_v3) vs. the current top cluster + the original
grammar_guided_evolution.
=============================================================================

Restricted method set (7 methods) so the sweep is tractable:
  - PĀṇini (new)                     -- our new candidate (rank 1/30 @ budget 60)
  - cross_entropy_method             -- rank 1 baseline (before v3)
  - ipop_cma_es
  - turbo
  - coarse_to_fine
  - lhs_plus_refinement
  - grammar_guided_evolution (ours)  -- the paper's original method

Same harness conventions as experiments/full28_with_v3.py: every method
spends exactly `budget` calls to evaluate_genome per seed, matched fairness
discipline carried over unchanged from the rest of the project.

Run:
    python3 -m experiments.budget_sweep_v3 --n_seeds 24 \
        --budgets 60,120,200 --cache_dir /path/to/cache/dir
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np

from experiments.expanded_baseline_table import ALL_METHODS
from experiments.recent_sota_baselines import NEW_METHODS as _SOTA_METHODS
from experiments.m_search_loop import evolutionary_search
from experiments.m_search_loop_improved import panini_search

OURS_NAME = "grammar_guided_evolution (ours)"

SWEEP_METHODS: Dict[str, callable] = {
    "PĀṇini (new)": lambda budget, seed: panini_search(
        budget, seed=seed, verbose=False)["best_fitness"],
    "cross_entropy_method": ALL_METHODS["cross_entropy_method"],
    "ipop_cma_es": ALL_METHODS["ipop_cma_es"],
    "turbo": ALL_METHODS["turbo"],
    "coarse_to_fine": lambda budget, seed: _SOTA_METHODS["coarse_to_fine"](
        budget, seed=seed, verbose=False)["best_fitness"],
    "lhs_plus_refinement": ALL_METHODS["lhs_plus_refinement"],
    OURS_NAME: lambda budget, seed: evolutionary_search(
        budget, seed=seed, verbose=False)["best_fitness"],
}


def run_one_budget(budget: int, n_seeds: int, cache_path: str = None,
                   verbose: bool = True) -> Dict:
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            results = json.load(f)
        if verbose:
            print(f"[budget={budget}] loaded cached results from {cache_path}")
    else:
        results = {name: [] for name in SWEEP_METHODS}
        for s in range(n_seeds):
            for name, fn in SWEEP_METHODS.items():
                fit = fn(budget, s)
                results[name].append(fit)
            if verbose:
                print(f"[budget={budget}] seed {s} done")
        if cache_path:
            with open(cache_path, "w") as f:
                json.dump({k: list(map(float, v)) for k, v in results.items()}, f)

    from scipy.stats import rankdata, friedmanchisquare
    names = list(results.keys())
    data = np.array([results[n] for n in names])
    ranks = np.array([rankdata(-data[:, s]) for s in range(data.shape[1])]).T
    mean_ranks = ranks.mean(axis=1)
    chi2, p = friedmanchisquare(*[data[i] for i in range(len(names))])

    nemenyi = {}
    try:
        import scikit_posthocs as sp
        import pandas as pd
        wide = pd.DataFrame(data.T, columns=names)
        nem = sp.posthoc_nemenyi_friedman(wide)
        v3 = "PĀṇini (new)"
        nemenyi = {other: float(nem.loc[v3, other]) for other in names if other != v3}
    except Exception as e:
        if verbose:
            print(f"Nemenyi unavailable: {e}")

    order = np.argsort(mean_ranks)
    if verbose:
        print(f"\n--- budget={budget}, n_seeds={data.shape[1]} "
             f"({len(names)} methods) ---")
        print(f"{'method':<36}{'mean rank':>12}{'mean fitness':>14}{'std':>10}")
        for idx in order:
            name = names[idx]
            tag = ("  <== NEW" if "(new)" in name
                  else ("  <== OURS(orig)" if name == OURS_NAME else ""))
            print(f"{name:<36}{mean_ranks[idx]:>12.2f}{data[idx].mean():>14.3f}"
                 f"{data[idx].std():>10.3f}{tag}")
        print(f"Friedman chi2={chi2:.2f} p={p:.2e}")
        v3_idx = names.index("PĀṇini (new)")
        v3_rank = 1 + int(np.sum(mean_ranks < mean_ranks[v3_idx]))
        print(f"PĀṇini (new): rank {v3_rank} of {len(names)}  "
             f"(mean rank {mean_ranks[v3_idx]:.2f}, mean fitness {data[v3_idx].mean():.3f})")
        if nemenyi:
            print("Nemenyi p-values, PĀṇini vs others:")
            for other, pv in nemenyi.items():
                print(f"  vs {other:<32} p={pv:.4f}")

    return {"results": results, "mean_ranks": dict(zip(names, mean_ranks)),
           "chi2": float(chi2), "p_friedman": float(p), "nemenyi": nemenyi}


def sweep(n_seeds: int, budgets, cache_dir: str = None, verbose: bool = True) -> Dict:
    out = {}
    for b in budgets:
        cp = os.path.join(cache_dir, f"sweep_b{b}_n{n_seeds}.json") if cache_dir else None
        out[b] = run_one_budget(b, n_seeds, cache_path=cp, verbose=verbose)
    if verbose:
        print("\n=== budget-sweep summary (PĀṇini rank-of-7) ===")
        for b, r in out.items():
            names = list(r["mean_ranks"].keys())
            mr = r["mean_ranks"]
            v3_name = "PĀṇini (new)"
            rank = 1 + sum(1 for n in names if mr[n] < mr[v3_name])
            print(f"budget={b:<4} v3 rank {rank}/{len(names)}  "
                 f"mean_rank={mr[v3_name]:.2f}  "
                 f"mean_fitness={np.mean(r['results'][v3_name]):.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=24)
    ap.add_argument("--budgets", type=str, default="60,120,200")
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    sweep(n_seeds=args.n_seeds, budgets=budgets, cache_dir=args.cache_dir)

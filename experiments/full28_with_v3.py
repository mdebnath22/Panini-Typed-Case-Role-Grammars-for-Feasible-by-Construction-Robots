"""
Combined 28-method comparison, PLUS our new PĀṇini (panini_search,
formerly grammar_guided_cem_v3) variant.
=============================================================================

Combines, unmodified:
  - expanded_baseline_table.ALL_METHODS (21 methods: the original 6 from
    main_baseline_table.py + the 15 from expanded_baseline_table.py)
  - recent_sota_baselines.NEW_METHODS (5 methods)
  - robogrammar_glso_baselines.NEW_METHODS (2 methods)
  = 28 methods total, exactly the set experiments/search21_final_designs.py
  documents as the full comparison (DIRECT_METHODS + ALL_METHODS pattern).

Adds ONE new method, grammar_guided_evolution_v2 (m_search_loop_improved.py)
and/or PĀṇini (panini_search, formerly grammar_guided_cem_v3; also
m_search_loop_improved.py), WITHOUT touching the original
grammar_guided_evolution or any of the 28 existing methods. Reports the
same style of table as search21_scaled_stats.py: mean rank, mean fitness,
std, Friedman chi2/p, Nemenyi vs ours.

Run:
    python3 -m experiments.full28_with_v3 --n_seeds 16 --budget 60
"""
from __future__ import annotations

import argparse
import json
from typing import Dict

import numpy as np

from experiments.expanded_baseline_table import ALL_METHODS
from experiments.recent_sota_baselines import NEW_METHODS as _SOTA_METHODS
from experiments.robogrammar_glso_baselines import NEW_METHODS as _GLSO_METHODS
from experiments.m_search_loop_improved import grammar_guided_evolution_v2, panini_search

OURS_NAME = "grammar_guided_evolution (ours)"


def _wrap_dict_returning(fn):
    return lambda budget, seed: fn(budget, seed=seed, verbose=False)["best_fitness"]


ALL_28 = dict(ALL_METHODS)
for name, fn in _SOTA_METHODS.items():
    ALL_28[name] = _wrap_dict_returning(fn)
for name, fn in _GLSO_METHODS.items():
    ALL_28[name] = _wrap_dict_returning(fn)

ALL_29 = dict(ALL_28)
ALL_29["grammar_guided_evolution_v2 (new)"] = lambda budget, seed: grammar_guided_evolution_v2(budget, seed=seed, verbose=False)["best_fitness"]
ALL_30 = dict(ALL_29)
ALL_30["PĀṇini (new)"] = lambda budget, seed: panini_search(budget, seed=seed, verbose=False)["best_fitness"]


def run(n_seeds: int = 16, budget: int = 60, verbose: bool = True,
       cache_path: str = None, methods: Dict = None) -> Dict:
    import os
    methods = methods or ALL_30
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            results = json.load(f)
        if verbose:
            print(f"Loaded cached raw results from {cache_path}")
    else:
        results = {name: [] for name in methods}
        for s in range(n_seeds):
            for name, fn in methods.items():
                try:
                    fit = fn(budget, s)
                except ImportError as e:
                    if verbose:
                        print(f"skipping {name}: {e}")
                    fit = float("nan")
                results[name].append(fit)
            if verbose:
                print(f"seed {s} done")
        if cache_path:
            with open(cache_path, "w") as f:
                json.dump({k: list(map(float, v)) for k, v in results.items()}, f)

    names = list(results.keys())
    usable = [n for n in names if not np.all(np.isnan(results[n]))]
    data = np.array([results[n] for n in usable])

    data_for_rank = data.copy()
    if np.isnan(data_for_rank).any():
        for i in range(data_for_rank.shape[0]):
            row = data_for_rank[i]
            if np.isnan(row).any():
                row_mean = np.nanmean(row)
                row[np.isnan(row)] = row_mean

    from scipy.stats import rankdata, friedmanchisquare
    ranks = np.array([rankdata(-data_for_rank[:, s]) for s in range(n_seeds)]).T
    mean_ranks = ranks.mean(axis=1)
    chi2, p_friedman = friedmanchisquare(*[data_for_rank[i] for i in range(len(usable))])

    nemenyi = {}
    try:
        import scikit_posthocs as sp
        import pandas as pd
        wide = pd.DataFrame(data_for_rank.T, columns=usable)
        nem = sp.posthoc_nemenyi_friedman(wide)
        for new_name in [n for n in usable if "(new)" in n]:
            nemenyi[new_name] = {other: float(nem.loc[new_name, other])
                                 for other in usable if other != new_name}
    except Exception as e:
        if verbose:
            print(f"Nemenyi unavailable: {e}")

    order = np.argsort(mean_ranks)
    if verbose:
        print(f"\n{len(usable)}-method comparison, n_seeds={n_seeds}, budget={budget}\n")
        print(f"{'method':<38}{'mean rank':>12}{'mean fitness':>14}{'std':>10}")
        for idx in order:
            name = usable[idx]
            tag = "  <== NEW" if "(new)" in name else ("  <== OURS" if name == OURS_NAME else "")
            print(f"{name:<38}{mean_ranks[idx]:>12.2f}{np.nanmean(data[idx]):>14.3f}"
                 f"{np.nanstd(data[idx]):>10.3f}{tag}")
        print(f"\nFriedman chi2={chi2:.2f} p={p_friedman:.2e} ({len(usable)} methods, {n_seeds} seeds)")
        for new_name in [n for n in usable if "(new)" in n]:
            rank_order = 1 + int(np.sum(mean_ranks < mean_ranks[usable.index(new_name)]))
            print(f"\n{new_name}: rank {rank_order} of {len(usable)}  "
                 f"(mean rank {mean_ranks[usable.index(new_name)]:.2f}, "
                 f"mean fitness {np.nanmean(data[usable.index(new_name)]):.3f})")

    return {"results": results, "usable": usable,
           "mean_ranks": dict(zip(usable, mean_ranks)),
           "chi2": float(chi2), "p_friedman": float(p_friedman),
           "nemenyi": nemenyi}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=16)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--cache_path", type=str, default=None)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds, budget=args.budget, cache_path=args.cache_path)

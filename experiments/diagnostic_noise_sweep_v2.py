"""
Diagnostic v2 — Isolating Averaging Quality from Search Budget
==================================================================

The first sweep (diagnostic_noise_sweep.py) held TOTAL PHYSICS CALLS fixed
and traded off genomes-tried against rollouts-per-genome. It found a
monotonic trend in the predicted direction (evo-rnd difference: -0.293 ->
-0.207 -> -0.197 -> +0.195 as n_gait_seeds increased 1->3->6->10), but this
is confounded: fewer genomes tried ALSO shrinks the best-of-N order
statistic for both conditions, independent of noise.

This version holds GENOMES TRIED fixed (at n_genomes) and only increases
n_gait_seeds on top -- i.e., TOTAL COST increases with n_gait_seeds here.
This isolates the specific question: given the SAME exploration breadth,
does a tighter per-genome fitness estimate specifically help evolutionary
search relative to random search (which should be far less sensitive to
this, since it never re-selects based on a past noisy estimate)?

Run (from your existing repo root; this is MORE expensive than v1 since
cost now grows with n_gait_seeds rather than staying fixed):
    python3 -m experiments.diagnostic_noise_sweep_v2 --n_seeds 16 --n_genomes 40
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from experiments.m_search_loop import (Genome, random_genome, mutate,
                                       crossover, evaluate_genome)


def _search_fixed_genomes(n_genomes: int, n_gait_seeds: int, seed: int,
                          use_gate: bool = True, pop_size: int = 8) -> float:
    rng = np.random.default_rng(seed)
    population: List = []

    def evaluate(genome: Genome) -> float:
        fit, _ = evaluate_genome(genome, use_gate=use_gate,
                                 n_gait_seeds=n_gait_seeds)
        return fit

    tried = 0
    while len(population) < pop_size and tried < n_genomes:
        g = random_genome(rng)
        population.append((evaluate(g), g))
        tried += 1

    while tried < n_genomes:
        population.sort(key=lambda x: -x[0])
        survivors = [g for _, g in population[:max(2, pop_size // 2)]]
        if rng.random() < 0.5 and len(survivors) >= 2:
            i, j = rng.choice(len(survivors), 2, replace=False)
            child = crossover(survivors[i], survivors[j], rng)
        else:
            child = mutate(survivors[rng.integers(len(survivors))], rng)
        population.append((evaluate(child), child))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
        tried += 1

    return max(f for f, _ in population)


def _random_fixed_genomes(n_genomes: int, n_gait_seeds: int, seed: int,
                          use_gate: bool = True) -> float:
    rng = np.random.default_rng(seed)
    best = -1e9
    for _ in range(n_genomes):
        g = random_genome(rng)
        fit, _ = evaluate_genome(g, use_gate=use_gate, n_gait_seeds=n_gait_seeds)
        best = max(best, fit)
    return best


def run(n_seeds: int = 16, n_genomes: int = 40,
       gait_seed_settings: List[int] = (1, 3, 6, 10),
       verbose: bool = True) -> Dict:
    results = {}
    if verbose:
        print(f"Genomes tried is FIXED at {n_genomes} across all settings; "
             f"total physics cost = {n_genomes} x n_gait_seeds (grows "
             f"left to right, unlike v1's fixed-budget design).\n")
    for ngs in gait_seed_settings:
        evo_fits, rnd_fits = [], []
        for s in range(n_seeds):
            evo_fits.append(_search_fixed_genomes(n_genomes, ngs, seed=s))
            rnd_fits.append(_random_fixed_genomes(n_genomes, ngs, seed=s + 1000))
        evo_arr, rnd_arr = np.array(evo_fits), np.array(rnd_fits)
        wins = int((evo_arr > rnd_arr).sum())
        try:
            from scipy import stats
            t_stat, p_val = stats.ttest_rel(evo_arr, rnd_arr)
        except Exception:
            t_stat, p_val = float("nan"), float("nan")
        diff = float(evo_arr.mean() - rnd_arr.mean())
        results[ngs] = {"evo_mean": float(evo_arr.mean()),
                        "evo_std": float(evo_arr.std()),
                        "rnd_mean": float(rnd_arr.mean()),
                        "rnd_std": float(rnd_arr.std()),
                        "diff": diff, "wins": wins,
                        "t_stat": float(t_stat), "p_val": float(p_val)}
        if verbose:
            total_cost = n_genomes * ngs
            print(f"n_gait_seeds={ngs:2d}  (total cost={total_cost:4d})  "
                 f"evo={evo_arr.mean():.3f}+/-{evo_arr.std():.3f}  "
                 f"rnd={rnd_arr.mean():.3f}+/-{rnd_arr.std():.3f}  "
                 f"diff={diff:+.3f}  wins={wins}/{n_seeds}  p={p_val:.3f}")

    if verbose:
        diffs = [results[k]["diff"] for k in gait_seed_settings]
        monotonic = all(diffs[i] <= diffs[i+1] for i in range(len(diffs)-1))
        print(f"\nDifference sequence (evo-rnd): "
             f"{[round(d,3) for d in diffs]}")
        print(f"Monotonically non-decreasing (consistent with noise-"
             f"selection hypothesis, now WITHOUT the fixed-budget "
             f"confound): {monotonic}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=16)
    ap.add_argument("--n_genomes", type=int, default=40)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds, n_genomes=args.n_genomes)

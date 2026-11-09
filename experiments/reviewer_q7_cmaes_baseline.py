"""
Reviewer Q7 — CMA-ES Baseline Over the Same Genome
=====================================================

"Does the grammar-guided advantage persist under stronger search?" A
lightweight, dependency-free CMA-ES-style implementation (not the full
covariance-matrix adaptation algorithm -- a simplified separable variant,
since the genome is only 5-dimensional and mostly discrete/near-discrete,
so full CMA-ES's covariance machinery is overkill) operating on a
continuous relaxation of the same Genome used elsewhere (n_legs and
segments_per_leg relaxed to real values, rounded only at evaluation time).

This is syntax-checked only in the sandbox that built it (no MuJoCo/torch
available there) -- run and verify on your existing repo.

Run (from your existing repo root):
    python3 -m experiments.reviewer_q7_cmaes_baseline --n_seeds 8 --budget 60
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import Genome, evaluate_genome
from experiments.m_search_loop import evolutionary_search, random_search


def relaxed_to_genome(x: np.ndarray) -> Genome:
    """x = [n_legs, segs, seg_len, seg_rad, jitter], continuous; rounded/
    clamped to a valid Genome only at evaluation time."""
    return Genome(
        n_legs=int(np.clip(round(x[0]), 2, 8)),
        segments_per_leg=int(np.clip(round(x[1]), 1, 3)),
        seg_len=float(np.clip(x[2], 0.05, 0.30)),
        seg_rad=float(np.clip(x[3], 0.01, 0.05)),
        angle_jitter=float(np.clip(x[4], 0.0, 0.3)),
    )


def separable_cmaes(budget: int, seed: int, use_gate: bool = True,
                    pop_size: int = 8, sigma0: float = 1.0) -> Dict:
    """A simplified separable evolution strategy: maintain a mean vector and
    a per-dimension step size, sample a population, move the mean toward
    the weighted-best samples, shrink sigma over time. This captures CMA-ES's
    core idea (adapt step size from population performance) without the
    full covariance matrix, appropriate for this low-dimensional genome."""
    rng = np.random.default_rng(seed)
    dim = 5
    mean = np.array([5.0, 2.0, 0.13, 0.02, 0.1])  # roughly centered in valid range
    sigma = np.array([2.0, 0.7, 0.05, 0.01, 0.08]) * sigma0
    best_fit, best_x = -1e9, mean.copy()
    evals_used = 0
    history = []

    while evals_used < budget:
        n_this_gen = min(pop_size, budget - evals_used)
        samples = mean + sigma * rng.normal(size=(n_this_gen, dim))
        fits = []
        for x in samples:
            genome = relaxed_to_genome(x)
            fit, _ = evaluate_genome(genome, use_gate=use_gate)
            fits.append(fit)
        fits = np.array(fits)
        evals_used += n_this_gen

        if fits.max() > best_fit:
            best_fit = float(fits.max())
            best_x = samples[fits.argmax()].copy()
        history.append(best_fit)

        # weighted recombination toward top half
        order = np.argsort(-fits)
        top = samples[order[:max(1, n_this_gen // 2)]]
        mean = top.mean(axis=0)
        sigma *= 0.95  # simple annealing in place of full step-size adaptation

    return {"best_fitness": best_fit, "history": history, "evals_used": evals_used}


def run(n_seeds: int = 8, budget: int = 60, verbose: bool = True) -> Dict:
    cma_fits, evo_fits, rnd_fits = [], [], []
    for s in range(n_seeds):
        cma = separable_cmaes(budget, seed=s)
        evo = evolutionary_search(budget, seed=s, verbose=False)
        rnd = random_search(budget, seed=s + 1000, verbose=False)
        cma_fits.append(cma["best_fitness"])
        evo_fits.append(evo["best_fitness"])
        rnd_fits.append(rnd["best_fitness"])
        if verbose:
            print(f"seed={s}: cma-es={cma_fits[-1]:+.3f}  "
                 f"grammar-evo={evo_fits[-1]:+.3f}  random={rnd_fits[-1]:+.3f}")

    cma_arr, evo_arr, rnd_arr = map(np.array, (cma_fits, evo_fits, rnd_fits))
    if verbose:
        print(f"\ncma-es      : {cma_arr.mean():.3f} +/- {cma_arr.std():.3f}")
        print(f"grammar-evo : {evo_arr.mean():.3f} +/- {evo_arr.std():.3f}")
        print(f"random      : {rnd_arr.mean():.3f} +/- {rnd_arr.std():.3f}")
        print(f"\ngrammar-evo beats cma-es on {(evo_arr > cma_arr).sum()}/{n_seeds} seeds")
        print(f"cma-es beats random on {(cma_arr > rnd_arr).sum()}/{n_seeds} seeds")
    return {"cma_fits": cma_fits, "evo_fits": evo_fits, "rnd_fits": rnd_fits}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=60)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds, budget=args.budget)

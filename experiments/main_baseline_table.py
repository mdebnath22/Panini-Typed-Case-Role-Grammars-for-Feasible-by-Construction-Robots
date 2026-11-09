"""
Main Comparison Table — Every Search Baseline, Same Budget, Same Accounting
==============================================================================

This is the "as big as possible, apples-to-apples" table the paper needs
for its central optimization claim. Every method below is given EXACTLY
`budget` physics-evaluation calls (evaluate_genome invocations) per seed --
not genome tries, not iterations, not wall-clock time -- so the comparison
is fair regardless of each method's internal population size or update
frequency. This is the same accounting discipline established in
mh6b_gate_ablation.py after an earlier version conflated genome-tries with
physics-calls and produced a misleading result.

Methods, and why each is here (the "reviewer's five-minute alternative"
standard):

  1. Random search            -- the simplest possible baseline; already
                                  in the paper.
  2. Grammar-guided evolution  -- OUR method; already in the paper.
  3. CMA-ES-lite               -- "why not a smarter classical optimizer?"
                                  A separable step-size-adaptive ES,
                                  reviewer-requested (Q7).
  4. PSO                       -- already cited in the paper's own related
                                  work (Kennedy & Eberhart 1995); the
                                  simplest classical POPULATION method with
                                  social/cognitive structure, distinct from
                                  a plain generational GA.
  5. NES (separable)           -- a natural-gradient method; structurally
                                  different from both GA-style (grammar-
                                  evo) and covariance-adaptive (CMA-ES)
                                  search, so its inclusion tests whether
                                  the grammar's advantage (if any) is
                                  specific to evolutionary-style search or
                                  holds against gradient-flavored black-box
                                  methods too.
  6. Hill-climbing             -- the trivial "why not just greedy local
                                  search" baseline; if grammar-guided
                                  search cannot beat this, no other
                                  comparison in this table matters.

All methods search the SAME 5-dimensional relaxed genome space
(n_legs, segments_per_leg, seg_len, seg_rad, angle_jitter), continuous,
rounded/clamped to a valid Genome only at evaluation time -- except
grammar-guided evolution, which operates on the discrete/typed Genome
directly (its native representation). This means grammar-guided evolution
is the only method whose mutations are legal-by-construction at every
step; every other method's proposals are validated only when evaluate_genome
calls the feasibility gate / grammar re-derivation, which is itself part
of the comparison's point.

NOT included, with reasons (documented so the table's coverage limits are
explicit, not silently absent):
  - RoboGrammar's Graph Heuristic Search: would require reimplementing
    their graph grammar and learned GNN heuristic; a literature comparison
    is given in the paper's related work instead (see Section 2).
  - A trained per-body policy (e.g. SAC): would require a full RL training
    loop per candidate body (hundreds of separate training runs at this
    search budget); a deliberate compute-budget decision, not run here.
  - Bayesian Optimization: omitted from this pass for scope; a natural
    next addition given a GP library, not fundamentally harder than the
    methods here.

Run (from your existing repo root):
    python3 -m experiments.main_baseline_table --n_seeds 8 --budget 60
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import (Genome, random_genome, evaluate_genome,
                                       evolutionary_search, random_search)


def relaxed_to_genome(x: np.ndarray) -> Genome:
    return Genome(
        n_legs=int(np.clip(round(x[0]), 2, 8)),
        segments_per_leg=int(np.clip(round(x[1]), 1, 3)),
        seg_len=float(np.clip(x[2], 0.05, 0.30)),
        seg_rad=float(np.clip(x[3], 0.01, 0.05)),
        angle_jitter=float(np.clip(x[4], 0.0, 0.3)),
    )


def genome_to_relaxed(g: Genome) -> np.ndarray:
    return np.array([g.n_legs, g.segments_per_leg, g.seg_len, g.seg_rad,
                     g.angle_jitter])


_BOUNDS_LO = np.array([2.0, 1.0, 0.05, 0.01, 0.0])
_BOUNDS_HI = np.array([8.0, 3.0, 0.30, 0.05, 0.3])


# ---------------------------------------------------------------------------
# 3. CMA-ES-lite (separable step-size-adaptive ES)
# ---------------------------------------------------------------------------

def run_cmaes_lite(budget: int, seed: int, pop_size: int = 8) -> float:
    rng = np.random.default_rng(seed)
    dim = 5
    mean = np.array([5.0, 2.0, 0.13, 0.02, 0.1])
    sigma = np.array([2.0, 0.7, 0.05, 0.01, 0.08])
    best_fit = -1e9
    evals_used = 0
    while evals_used < budget:
        n_this = min(pop_size, budget - evals_used)
        samples = mean + sigma * rng.normal(size=(n_this, dim))
        fits = np.array([evaluate_genome(relaxed_to_genome(x))[0] for x in samples])
        evals_used += n_this
        if fits.max() > best_fit:
            best_fit = float(fits.max())
        order = np.argsort(-fits)
        top = samples[order[:max(1, n_this // 2)]]
        mean = top.mean(axis=0)
        sigma *= 0.95
    return best_fit


# ---------------------------------------------------------------------------
# 4. PSO (particle swarm) -- already cited in the paper's related work
# ---------------------------------------------------------------------------

def run_pso(budget: int, seed: int, n_particles: int = 8, w: float = 0.6,
           c1: float = 1.5, c2: float = 1.5) -> float:
    rng = np.random.default_rng(seed)
    dim = 5
    x = _BOUNDS_LO + rng.random((n_particles, dim)) * (_BOUNDS_HI - _BOUNDS_LO)
    v = np.zeros_like(x)
    pbest = x.copy()
    pbest_fit = np.full(n_particles, -1e9)
    gbest = x[0].copy()
    gbest_fit = -1e9
    evals_used = 0

    while evals_used < budget:
        n_this = min(n_particles, budget - evals_used)
        for i in range(n_this):
            fit, _ = evaluate_genome(relaxed_to_genome(x[i]))
            evals_used += 1
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest[i] = x[i].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest = x[i].copy()
        r1, r2 = rng.random((n_particles, dim)), rng.random((n_particles, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = np.clip(x + v, _BOUNDS_LO, _BOUNDS_HI)

    return gbest_fit


# ---------------------------------------------------------------------------
# 5. NES (separable natural evolution strategy, Wierstra et al. 2014 style)
# ---------------------------------------------------------------------------

def _utility_weights(n: int) -> np.ndarray:
    """Standard rank-based NES utility transform: robust to fitness scale,
    zero-sum (doesn't bias the mean update toward absolute fitness level)."""
    ranks = np.arange(1, n + 1)
    raw = np.maximum(0.0, np.log(n / 2.0 + 1.0) - np.log(ranks))
    u = raw / raw.sum() - 1.0 / n
    return u


def run_nes(budget: int, seed: int, pop_size: int = 8,
           eta_mu: float = 1.0, eta_sigma: float = 0.5) -> float:
    rng = np.random.default_rng(seed)
    dim = 5
    mean = np.array([5.0, 2.0, 0.13, 0.02, 0.1])
    log_sigma = np.log(np.array([2.0, 0.7, 0.05, 0.01, 0.08]))
    best_fit = -1e9
    evals_used = 0
    weights = _utility_weights(pop_size)

    while evals_used < budget:
        n_this = min(pop_size, budget - evals_used)
        sigma = np.exp(log_sigma)
        z = rng.normal(size=(n_this, dim))
        samples = mean + sigma * z
        fits = np.array([evaluate_genome(relaxed_to_genome(s))[0] for s in samples])
        evals_used += n_this
        if fits.max() > best_fit:
            best_fit = float(fits.max())

        order = np.argsort(-fits)
        u = _utility_weights(n_this)[np.argsort(order)]  # utility by original index
        # natural gradient steps (separable/diagonal approximation)
        grad_mu = (u[:, None] * z).sum(axis=0)
        grad_log_sigma = (u[:, None] * (z ** 2 - 1.0)).sum(axis=0) * 0.5
        mean = mean + eta_mu * sigma * grad_mu
        log_sigma = log_sigma + eta_sigma * grad_log_sigma
        mean = np.clip(mean, _BOUNDS_LO, _BOUNDS_HI)

    return best_fit


# ---------------------------------------------------------------------------
# 6. Hill-climbing (greedy local search) -- the trivial baseline
# ---------------------------------------------------------------------------

def run_hill_climbing(budget: int, seed: int, step_size: float = 0.3) -> float:
    rng = np.random.default_rng(seed)
    dim = 5
    x = _BOUNDS_LO + rng.random(dim) * (_BOUNDS_HI - _BOUNDS_LO)
    fit, _ = evaluate_genome(relaxed_to_genome(x))
    best_x, best_fit = x.copy(), fit
    evals_used = 1

    scale = (_BOUNDS_HI - _BOUNDS_LO) * step_size
    while evals_used < budget:
        candidate = np.clip(best_x + rng.normal(size=dim) * scale,
                            _BOUNDS_LO, _BOUNDS_HI)
        cfit, _ = evaluate_genome(relaxed_to_genome(candidate))
        evals_used += 1
        if cfit > best_fit:
            best_fit, best_x = cfit, candidate.copy()

    return best_fit


# ---------------------------------------------------------------------------
# Consolidated runner
# ---------------------------------------------------------------------------

METHODS = {
    "random_search": lambda budget, seed: random_search(budget, seed=seed, verbose=False)["best_fitness"],
    "grammar_guided_evolution (ours)": lambda budget, seed: evolutionary_search(budget, seed=seed, verbose=False)["best_fitness"],
    "cma_es_lite": run_cmaes_lite,
    "pso": run_pso,
    "nes": run_nes,
    "hill_climbing": run_hill_climbing,
}


def run(n_seeds: int = 8, budget: int = 60, verbose: bool = True) -> Dict:
    results = {name: [] for name in METHODS}
    for s in range(n_seeds):
        row = {}
        for name, fn in METHODS.items():
            fit = fn(budget, s)
            results[name].append(fit)
            row[name] = fit
        if verbose:
            print(f"seed={s}: " + "  ".join(f"{k}={v:+.3f}" for k, v in row.items()))

    if verbose:
        print(f"\n{'method':<34}{'mean':>10}{'std':>10}{'wins vs ours':>14}")
        ours = np.array(results["grammar_guided_evolution (ours)"])
        for name, vals in results.items():
            arr = np.array(vals)
            wins_vs_ours = int((ours > arr).sum()) if name != "grammar_guided_evolution (ours)" else "--"
            print(f"{name:<34}{arr.mean():>10.3f}{arr.std():>10.3f}"
                 f"{str(wins_vs_ours):>14}")

        try:
            from scipy import stats
            print(f"\nPaired t-tests (ours vs each baseline):")
            for name, vals in results.items():
                if name == "grammar_guided_evolution (ours)":
                    continue
                t_stat, p_val = stats.ttest_rel(ours, np.array(vals))
                print(f"  ours vs {name:<32} t={t_stat:+.3f}  p={p_val:.3f}")
        except ImportError:
            pass
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=60)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds, budget=args.budget)

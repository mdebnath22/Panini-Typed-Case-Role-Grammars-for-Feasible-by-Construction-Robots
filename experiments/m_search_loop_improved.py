"""
grammar_guided_evolution_v2 -- an improved grammar-guided search
===================================================================

Goal: push "grammar_guided_evolution (ours)" up the 28-method rank table
(currently rank 6/28 by mean fitness, per results_refit/search21_n24_b60.json
-era numbers) WITHOUT abandoning the grammar-guided framing: every genome
here is still constructed/mutated through the typed Genome representation
and derived through the SAME BodyGrammar/PhiBody pipeline as the original
`evolutionary_search` -- this file imports Genome, random_genome, mutate,
crossover, evaluate_genome from experiments.m_search_loop UNCHANGED, and
does not modify that file or Genome.mount_plan()/mutate() in place.

What's borrowed from the top-5 methods that beat the original GA, and why:

  - cross_entropy_method (rank 1): importance-weighted resampling -- refit
    a Gaussian to the top elite_frac of the population each generation and
    sample the NEXT generation from that Gaussian, rather than from
    independent per-field coin-flip mutation of a single parent. This is
    the single biggest mechanical difference vs. the original GA.
  - ipop_cma_es (rank 2): adaptive step size (here: a per-field sigma
    vector updated by a 1/5-success rule, the separable analogue of
    CMA-ES's covariance adaptation) + restart-on-stagnation with a larger
    population.
  - pso (rank 3): a small momentum term pulling each new sample toward the
    running global-best, blended with the CEM-style resampling.
  - coarse_to_fine (rank 4): start with wide step sizes (coarse) and anneal
    them down over the run (fine), plus a final pure local-refinement
    phase appended after the main generational loop.
  - turbo (rank 5): a shrinking "trust region" around the incumbent best
    that the final refinement phase samples within, shrinking on failure
    streaks and growing on success streaks.

The discrete fields (n_legs, segments_per_leg) stay under grammar-level
mutation/crossover (exactly `mutate`/`crossover` from m_search_loop) --
only the 3 continuous fields (seg_len, seg_rad, angle_jitter) get the
adaptive/CEM/trust-region treatment, applied by constructing a new Genome
(still going through Genome -> BodyGrammar -> PhiBody, same as every other
method in this comparison).

Run:
    python3 -m experiments.m_search_loop_improved --n_seeds 16 --budget 60
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import (
    Genome, random_genome, mutate, crossover, evaluate_genome,
)

_CONT_FIELDS = ("seg_len", "seg_rad", "angle_jitter")
_CONT_LO = np.array([0.05, 0.01, 0.0])
_CONT_HI = np.array([0.30, 0.05, 0.3])


def _cont_vec(g: Genome) -> np.ndarray:
    return np.array([g.seg_len, g.seg_rad, g.angle_jitter])


def _with_cont(g: Genome, v: np.ndarray) -> Genome:
    h = copy.copy(g)
    h.seg_len, h.seg_rad, h.angle_jitter = float(v[0]), float(v[1]), float(v[2])
    return h.clamp()


def _mutate_discrete(genome: Genome, rng: np.random.Generator,
                     rate: float = 0.5) -> Genome:
    """Grammar-level mutation restricted to the DISCRETE fields (n_legs,
    segments_per_leg) only -- the continuous fields are left untouched here
    so the adaptive-sigma step (applied separately, once, by the caller)
    doesn't compound with `mutate`'s own +-uniform(0.7,1.4)/+-0.1 continuous
    jitter. Each discrete field mutates independently w.p. `rate`; at least
    one always mutates."""
    g = copy.copy(genome)
    mask = rng.random(2) < rate
    if not mask.any():
        mask[rng.integers(0, 2)] = True
    if mask[0]:
        g.n_legs += int(rng.choice([-2, -1, 1, 2]))
    if mask[1]:
        g.segments_per_leg += int(rng.choice([-1, 1]))
    return g.clamp()


def grammar_guided_evolution_v2(
    budget: int, seed: int = 0, pop_size: int = 8,
    sigma0_frac: float = 0.25,        # coarse: fraction of field range, initial
    sigma_min_frac: float = 0.02,     # fine floor
    sigma_max_frac: float = 0.5,
    success_window: int = 8,
    restart_patience: int = 18,       # evals with no improvement -> restart
    refine_frac: float = 0.15,        # fraction of budget spent on final local polish
    use_gate: bool = True, spec=None, verbose: bool = False,
) -> Dict:
    """(mu+lambda)-style GA, same ONE-OFFSPRING-PER-EVAL granularity as the
    original `evolutionary_search` (so it gets just as many generations to
    adapt within the same budget -- batching many children per generation,
    as a naive CEM port would, was tried and tested WORSE: fewer, coarser
    generations under this noisy fitness proxy). Discrete fields
    (n_legs, segments_per_leg) still go through exactly `mutate`/`crossover`
    from m_search_loop. Continuous fields (seg_len, seg_rad, angle_jitter)
    are instead perturbed by a per-field Gaussian step whose size is
    adapted online by a 1/5-success rule (CMA-ES-lite's separable analogue),
    seeded coarse and annealed fine by that same rule -- this replaces the
    original's fixed +-uniform(0.7,1.4)/+-0.1 jitter for those 3 fields only."""
    rng = np.random.default_rng(seed)
    evals_used = 0
    history: List[float] = []

    def eval_g(g: Genome) -> float:
        nonlocal evals_used
        fit, _ = evaluate_genome(g, use_gate=use_gate, spec=spec)
        evals_used += 1
        return fit

    refine_budget = max(1, int(round(budget * refine_frac)))
    main_budget = budget - refine_budget

    scale = _CONT_HI - _CONT_LO
    sigma = sigma0_frac * scale

    population: List[Tuple[float, Genome]] = []
    while len(population) < pop_size and evals_used < main_budget:
        g = random_genome(rng)
        f = eval_g(g)
        population.append((f, g))
        history.append(max(x[0] for x in population))

    best_fit, best_genome = max(population, key=lambda x: x[0])
    evals_since_improve = 0
    successes: List[bool] = []

    while evals_used < main_budget:
        population.sort(key=lambda x: -x[0])
        survivors = [g for _, g in population[:max(2, pop_size // 2)]]
        worst_fit = population[-1][0]

        if rng.random() < 0.5 and len(survivors) >= 2:
            i, j = rng.choice(len(survivors), 2, replace=False)
            base = crossover(survivors[i], survivors[j], rng)
        else:
            base = _mutate_discrete(survivors[rng.integers(len(survivors))], rng, rate=0.5)

        # continuous-field adaptive Gaussian step, centered on `base`'s own
        # continuous values (so the discrete-field edit from mutate/crossover
        # above is preserved, and only seg_len/seg_rad/angle_jitter get this
        # extra adaptive-step treatment)
        cont = np.clip(_cont_vec(base) + sigma * rng.normal(size=3),
                       _CONT_LO, _CONT_HI)
        child = _with_cont(base, cont)
        f = eval_g(child)

        success = f > worst_fit
        successes.append(success)
        if len(successes) > success_window:
            successes.pop(0)
        if len(successes) >= success_window:
            rate = np.mean(successes)
            if rate > 0.2:
                sigma = np.minimum(sigma * 1.22, sigma_max_frac * scale)
            else:
                sigma = np.maximum(sigma * 0.85, sigma_min_frac * scale)

        population.append((f, child))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
        history.append(population[0][0])

        if population[0][0] > best_fit + 1e-9:
            best_fit, best_genome = population[0]
            evals_since_improve = 0
        else:
            evals_since_improve += 1

        # ipop-style restart: long stagnation -> inject fresh random blood
        # and widen sigma back toward coarse, like a population-size-doubling
        # restart's effect of re-exploring the space
        if evals_since_improve >= restart_patience and evals_used < main_budget:
            evals_since_improve = 0
            sigma = np.minimum(sigma * 2.5, sigma_max_frac * scale)
            n_fresh = min(max(2, pop_size // 2), main_budget - evals_used)
            for _ in range(n_fresh):
                g = random_genome(rng)
                f = eval_g(g)
                population.append((f, g))
                if f > best_fit:
                    best_fit, best_genome = f, g
                if evals_used >= main_budget:
                    break
            population.sort(key=lambda x: -x[0])
            population = population[:pop_size]

        if verbose and evals_used % 10 == 0:
            print(f"  [v2] evals {evals_used:3d}/{budget}  best={best_fit:+.4f}"
                 f"  sigma={sigma}")

    # --- final local-refinement phase: TuRBO-style shrinking trust region ------
    tr = 0.5 * sigma0_frac  # start half as wide as the main loop's coarse sigma
    tr_lo, tr_hi = sigma_min_frac, 0.6
    succ_streak = fail_streak = 0
    center = best_genome
    while evals_used < budget:
        cont = np.clip(_cont_vec(center) + tr * scale * rng.normal(size=3),
                       _CONT_LO, _CONT_HI)
        # keep discrete fields fixed during polish (small local step, as
        # TuRBO/hill-climbing-style refinement phases do)
        candidate = _with_cont(center, cont)
        f = eval_g(candidate)
        if f > best_fit:
            best_fit, best_genome = f, candidate
            center = candidate
            succ_streak += 1; fail_streak = 0
        else:
            fail_streak += 1; succ_streak = 0
        if succ_streak >= 3:
            tr = min(tr * 2, tr_hi); succ_streak = 0
        if fail_streak >= 5:
            tr = max(tr / 2, tr_lo); fail_streak = 0
        history.append(best_fit)

    return {"history": history, "best_fitness": best_fit,
           "best_genome": best_genome, "evals_used": evals_used}


_ALL_LO = np.array([2.0, 1.0, 0.05, 0.01, 0.0])
_ALL_HI = np.array([8.0, 3.0, 0.30, 0.05, 0.3])

# --- discrete-field std floor (bug fix, see panini_search docstring) -------
# segments_per_leg has only 3 valid integer values (1,2,3) in a [1,3] range,
# so a continuous-Gaussian-then-round CEM step that lets this field's std
# shrink toward 0 (the old code only added `+ 1e-3`, negligible against a
# field range of 2.0) permanently kills any sampling probability on the
# neighboring integers once the fitted mean settles near one value -- and
# because the initial mean is exactly the domain midpoint (2.0) for every
# seed, "2" is the value the sampler is already densest on before fitness
# has any say, so EVERY seed ratchets onto segments_per_leg=2 regardless of
# the actual fitness landscape (verified: 24/24 seeds at budget=60, 0
# exploration of segments_per_leg in {1,3}, vs. 7-10 unique (n_legs,
# segments_per_leg) combos for the other CEM-family methods in this
# comparison, which use the exact same round-to-int discretization but
# never add an elitism carryover that re-feeds frozen past samples back
# into the next generation's elite-refit, compounding the same bias).
# Fix: floor this field's resampling std at 0.6 (comparable to its own
# domain range of 2.0), so the Gaussian always keeps real mass on
# neighboring integers no matter how confident the CEM fit becomes --
# n_legs and the 3 continuous fields are left on the original floor since
# they were not observed to collapse this way.
# Verified at n=24 seeds, budget=60 (scripts in scratchpad/panini_diff_axes.py
# and scratchpad/test_floor_fix.py):
#   before: 3 unique (n_legs,segs) combos, segs stuck at {2} in 24/24 seeds,
#           mean fitness +0.403 (full30_n24.json cache)
#   after:  6 unique (n_legs,segs) combos, segs in {2,3} (24 seeds),
#           mean fitness +0.411 (slightly BETTER, not worse)
_STD_FLOOR = np.array([1e-3, 0.6, 1e-3, 1e-3, 1e-3])


def _genome_to_full(g: Genome) -> np.ndarray:
    return np.array([g.n_legs, g.segments_per_leg, g.seg_len, g.seg_rad, g.angle_jitter])


def _full_to_genome(v: np.ndarray) -> Genome:
    return Genome(
        n_legs=int(np.clip(round(v[0]), 2, 8)),
        segments_per_leg=int(np.clip(round(v[1]), 1, 3)),
        seg_len=float(np.clip(v[2], 0.05, 0.30)),
        seg_rad=float(np.clip(v[3], 0.01, 0.05)),
        angle_jitter=float(np.clip(v[4], 0.0, 0.3)),
    )


def panini_search(
    budget: int, seed: int = 0, pop_size: int = 10, elite_frac: float = 0.3,
    elitism: int = 2, refine_frac: float = 0.1,
    use_gate: bool = True, spec=None, verbose: bool = False,
) -> Dict:
    """PĀṇini -- "Production-rule And role-typed compositioN for
    Intelligent, Novel robot desIgn" (internal identifier: formerly
    `grammar_guided_cem_v3`, kept as `panini_search` in code).

    CEM-style resampling over the FULL 5-field relaxed genome vector
    (same mechanism that wins outright as `cross_entropy_method`), but kept
    grammar-guided/Genome-typed throughout: every sample is built as a real
    Genome (discrete fields rounded at construction, exactly like
    `relaxed_to_genome` elsewhere in this project) and evaluated through
    the unmodified BodyGrammar/PhiBody pipeline via `evaluate_genome` --
    nothing here touches raw XML or bypasses the grammar. Two additions
    on top of plain CEM, borrowed from the other top-5 methods: (1) a
    small elitism carryover (the top `elitism` genomes survive into the
    next generation's sample pool, like a GA's survivors, rather than
    being discarded the way vanilla CEM discards its whole population each
    round) and (2) a short final local-refinement phase (coarse_to_fine/
    turbo-style) that polishes the incumbent best with small local steps
    once the main distribution-fitting budget is spent."""
    rng = np.random.default_rng(seed)
    evals_used = 0
    history: List[float] = []

    def eval_v(v: np.ndarray) -> float:
        nonlocal evals_used
        fit, _ = evaluate_genome(_full_to_genome(v), use_gate=use_gate, spec=spec)
        evals_used += 1
        return fit

    refine_budget = max(1, int(round(budget * refine_frac)))
    main_budget = budget - refine_budget

    mean = (_ALL_LO + _ALL_HI) / 2
    std = (_ALL_HI - _ALL_LO) / 3
    n_elite = max(1, int(round(pop_size * elite_frac)))

    best_fit, best_v = -1e9, mean.copy()
    carry: List[Tuple[float, np.ndarray]] = []

    while evals_used < main_budget:
        n_this = min(pop_size, main_budget - evals_used)
        # guarantee at least 1 new eval per generation so evals_used always
        # advances -- when the elitism carryover is >= n_this (can happen
        # near the end of the budget, when remaining < elitism) the old
        # `max(0, ...)` made n_new hit 0 and the loop spun forever without
        # ever consuming budget.
        n_new = max(1, n_this - len(carry))
        samples = np.clip(mean + std * rng.normal(size=(n_new, 5)), _ALL_LO, _ALL_HI)
        fits = np.array([eval_v(s) for s in samples])
        pool = carry + list(zip(fits.tolist(), list(samples)))
        pool.sort(key=lambda x: -x[0])
        if pool[0][0] > best_fit:
            best_fit, best_v = pool[0]
        history.append(best_fit)

        elites = pool[:min(n_elite, len(pool))]
        elite_arr = np.array([e[1] for e in elites])
        mean = elite_arr.mean(axis=0)
        std = np.maximum(elite_arr.std(axis=0), _STD_FLOOR)
        carry = pool[:elitism]  # small elitism carryover into next generation

        if verbose and evals_used % 10 == 0:
            print(f"  [PĀṇini] evals {evals_used:3d}/{budget}  best={best_fit:+.4f}")

    # final local-refinement phase: small Gaussian steps around the
    # incumbent best, shrinking trust region on failure (TuRBO-lite)
    tr = 0.15
    scale = _ALL_HI - _ALL_LO
    fail_streak = succ_streak = 0
    while evals_used < budget:
        cand = np.clip(best_v + tr * scale * rng.normal(size=5), _ALL_LO, _ALL_HI)
        f = eval_v(cand)
        if f > best_fit:
            best_fit, best_v = f, cand
            succ_streak += 1; fail_streak = 0
        else:
            fail_streak += 1; succ_streak = 0
        if succ_streak >= 3:
            tr = min(tr * 2, 0.5); succ_streak = 0
        if fail_streak >= 5:
            tr = max(tr / 2, 0.02); fail_streak = 0
        history.append(best_fit)

    return {"history": history, "best_fitness": best_fit,
           "best_genome": _full_to_genome(best_v), "evals_used": evals_used}


def benchmark_v2(n_seeds: int = 8, budget: int = 60, verbose: bool = True) -> Dict:
    from experiments.m_search_loop import evolutionary_search
    from experiments.expanded_baseline_table import run_cem
    fits_v2, fits_v1, fits_v3, fits_cem = [], [], [], []
    for s in range(n_seeds):
        v2 = grammar_guided_evolution_v2(budget, seed=s, verbose=False)
        v1 = evolutionary_search(budget, seed=s, verbose=False)
        v3 = panini_search(budget, seed=s, verbose=False)
        cem = run_cem(budget, s)
        fits_v2.append(v2["best_fitness"])
        fits_v1.append(v1["best_fitness"])
        fits_v3.append(v3["best_fitness"])
        fits_cem.append(cem)
        if verbose:
            print(f"  seed={s}: v3={fits_v3[-1]:+.4f}  cem={fits_cem[-1]:+.4f}"
                 f"  v2={fits_v2[-1]:+.4f}  v1(orig)={fits_v1[-1]:+.4f}")
    v2a, v1a, v3a, cema = map(np.array, (fits_v2, fits_v1, fits_v3, fits_cem))
    if verbose:
        print(f"\nPĀṇini (CEM-hybrid): {v3a.mean():+.4f} +/- {v3a.std():.4f}")
        print(f"cem (baseline)     : {cema.mean():+.4f} +/- {cema.std():.4f}")
        print(f"v2                 : {v2a.mean():+.4f} +/- {v2a.std():.4f}")
        print(f"v1 (orig)          : {v1a.mean():+.4f} +/- {v1a.std():.4f}")
        print(f"PĀṇini wins {int((v3a > v1a).sum())}/{n_seeds} seeds vs original")
        print(f"PĀṇini wins {int((v3a > cema).sum())}/{n_seeds} seeds vs plain CEM")
    return {"v2": fits_v2, "v1": fits_v1, "v3": fits_v3, "cem": fits_cem}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=60)
    args = ap.parse_args()
    benchmark_v2(n_seeds=args.n_seeds, budget=args.budget)

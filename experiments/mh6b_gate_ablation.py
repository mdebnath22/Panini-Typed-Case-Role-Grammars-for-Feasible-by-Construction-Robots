"""
M-H6b — Śulba Gate Ablation INSIDE the Search Loop (corrected accounting)
==========================================================================

M-H6 (mh6_feasibility.py) showed the gate saves physics evaluations when
searching for ANY feasible body. This asks the sharper question: does the
gate help when searching for a GOOD (task-fit) body under a FIXED PHYSICS
BUDGET?

IMPORTANT CORRECTNESS NOTE (found while building this): a first version of
this ablation counted every genome TRIED against the budget, including
gate-rejections. Since a rejection costs near-zero physics time (that is
the entire M-H6 point), that unfairly throttled gate-on to the same tiny
number of tries as gate-off — at a demanding spec it collapsed to a
constant -1000 penalty because the search never got enough tries to even
fill its initial population. The fix below counts budget as PHYSICS
EVALUATIONS ONLY: gate-on may try (and reject) as many genomes as it needs
for free, and only spends budget when it actually runs MuJoCo. This is the
fair comparison — "for N physics rollouts you can afford, which condition
finds a better body" — and the one that matters for a paper's efficiency
claim, not a mechanically equal genome-try count.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.m_search_loop import Genome, random_genome, mutate, crossover, evaluate_genome
from sulba.feasibility import FeasibilitySpec

# Calibrated against THIS search space's own genome distribution (see
# README_SEARCH.md's calibration note) — always recalibrate a feasibility
# spec against the actual distribution it gates, never reuse another
# experiment's spec blindly. ~7% pass rate here.
#
# RECALIBRATED (post mounting+EXTEND geometry fix): the old absolute
# thresholds (required_reach=0.6, mass_budget=3.6, min_support_area=0.5)
# were calibrated against the pre-fix geometry's much larger achievable
# scale and are now entirely unreachable (0/20000 tries at every seed —
# min_support_area=0.5 alone is impossible: max achievable area under the
# corrected geometry is ~0.22). Recalibrated by a joint percentile
# bisection over 4000 fresh grammar genomes (reach/area thresholds at the
# SAME upper percentile p, mass threshold at the matching lower
# percentile p, min_tip_separation left untouched since it was already
# easily met and isn't the blocker) to hit the SAME ~7% joint pass rate
# this spec was originally designed to represent: converged at p=0.515,
# giving required_reach=0.247, mass_budget=3.4, min_support_area=0.061
# (measured joint pass rate 7.03% on the calibration sample); rounded to
# the values below, which measure 6.2% pass rate on an independent
# verification sample of 3000 genomes.
DEMANDING_SPEC = FeasibilitySpec(required_reach=0.25, mass_budget=3.4,
                                 min_support_area=0.06, min_tip_separation=0.05)


def _search_with_physics_budget(physics_budget: int, seed: int, use_gate: bool,
                                spec, pop_size: int = 8,
                                max_tries: int = 20000) -> Dict:
    """Evolutionary search where `physics_budget` counts only genomes that
    actually reached a MuJoCo rollout (gate-rejections are free and don't
    count). `max_tries` is a safety cap on total genome attempts."""
    rng = np.random.default_rng(seed)
    population: List[Tuple[float, Genome]] = []
    physics_used = 0
    tries = 0
    history: List[float] = []

    def try_genome(genome: Genome) -> Optional[float]:
        nonlocal physics_used, tries
        tries += 1
        fit, info = evaluate_genome(genome, use_gate=use_gate, spec=spec)
        if use_gate and info.get("error") == "infeasible":
            return None  # free rejection, doesn't consume physics budget
        physics_used += 1
        return fit

    while len(population) < pop_size and physics_used < physics_budget \
            and tries < max_tries:
        genome = random_genome(rng)
        fit = try_genome(genome)
        if fit is not None:
            population.append((fit, genome))
            history.append(max(f for f, _ in population))

    while physics_used < physics_budget and tries < max_tries:
        if not population:
            break
        population.sort(key=lambda x: -x[0])
        survivors = [g for _, g in population[:max(2, pop_size // 2)]]
        if rng.random() < 0.5 and len(survivors) >= 2:
            i, j = rng.choice(len(survivors), 2, replace=False)
            child = crossover(survivors[i], survivors[j], rng)
        else:
            parent = survivors[rng.integers(len(survivors))]
            child = mutate(parent, rng)
        fit = try_genome(child)
        if fit is not None:
            population.append((fit, child))
            population.sort(key=lambda x: -x[0])
            population = population[:pop_size]
            history.append(population[0][0])

    if not population:
        return {"best_fitness": -1e3, "history": [-1e3],
               "physics_used": physics_used, "tries": tries,
               "note": "never found a feasible genome"}
    best_fit, best_genome = max(population, key=lambda x: x[0])
    return {"best_fitness": best_fit, "history": history,
           "physics_used": physics_used, "tries": tries,
           "best_genome": best_genome}


def run(n_seeds: int = 8, physics_budget: int = 40, spec=None,
       verbose: bool = True) -> Dict:
    spec = spec or DEMANDING_SPEC
    with_gate, without_gate = [], []
    for s in range(n_seeds):
        r_on = _search_with_physics_budget(physics_budget, s, True, spec)
        r_off = _search_with_physics_budget(physics_budget, s, False, spec)
        with_gate.append(r_on["best_fitness"])
        without_gate.append(r_off["best_fitness"])
        if verbose:
            print(f"  seed={s}: gate-on={with_gate[-1]:+.3f} "
                 f"(tries={r_on['tries']})  gate-off={without_gate[-1]:+.3f} "
                 f"(tries={r_off['tries']})  diff={with_gate[-1]-without_gate[-1]:+.3f}")

    on, off = np.array(with_gate), np.array(without_gate)
    wins = int((on > off).sum())
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_rel(on, off)
    except Exception:
        t_stat, p_val = float("nan"), float("nan")

    if verbose:
        print(f"\ngate-on : {on.mean():+.3f} ± {on.std():.3f}")
        print(f"gate-off: {off.mean():+.3f} ± {off.std():.3f}")
        print(f"gate-on wins {wins}/{n_seeds} seeds")
        print(f"paired t-test: t={t_stat:.3f}  p={p_val:.3f}")
    return {"with_gate": with_gate, "without_gate": without_gate,
           "wins": wins, "t_stat": t_stat, "p_val": p_val}


if __name__ == "__main__":
    run()

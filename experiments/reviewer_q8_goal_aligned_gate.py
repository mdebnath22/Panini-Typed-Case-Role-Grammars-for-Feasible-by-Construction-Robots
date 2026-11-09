"""
Reviewer Q8 — Goal-Aligned Feasibility Gate
==============================================

The paper's original gate (reach/mass/support/clearance) is geometrically
sensible but ORTHOGONAL to the locomotion fitness metric -- which is
exactly why it didn't help task fitness under a fixed budget (Section 4.6).
This tests the reviewer's specific proposed fix: a gate criterion actually
correlated with gait quality (a coarse stride-length/leg-swing-clearance
proxy computable WITHOUT running physics), to see if goal-alignment
reverses the earlier negative result.

Goal-aligned proxy (cheap, geometric, no simulation): reject bodies whose
LEG-LENGTH-TO-TORSO-CLEARANCE RATIO falls outside a plausible
walking-gait range -- legs that are too short relative to torso height
cannot achieve ground clearance during a swing phase; legs that are
disproportionately long relative to mount spacing are prone to
self-collision during a stride. Both are geometric facts computable from
the genome alone, and both plausibly correlate with whether a CPG gait can
actually produce forward locomotion (unlike reach/mass/support, which
constrain the body without saying anything about whether it can WALK).

Run (from your existing repo root):
    python3 -m experiments.reviewer_q8_goal_aligned_gate --n_seeds 8 --budget 40
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np

from experiments.m_search_loop import (Genome, evaluate_genome,
                                       evolutionary_search)
from ptg.body_grammar import BodyGrammar
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody


def goal_aligned_feasible(genome: Genome, min_ratio: float = 0.6,
                          max_ratio: float = 2.5,
                          torso_height: float = 0.30) -> bool:
    """Cheap, physics-free geometric proxy: total leg reach (seg_len x
    segments_per_leg) relative to torso_height must fall in a plausible
    walking-gait range. Too short -> can't clear the ground during swing;
    too long relative to mount spacing -> self-collision risk during
    stride (approximated here via the same ratio, since both failure
    modes scale with leg-length-to-torso ratio in this simplified model)."""
    total_leg_len = genome.seg_len * genome.segments_per_leg
    ratio = total_leg_len / torso_height
    return min_ratio <= ratio <= max_ratio


def evaluate_with_goal_gate(genome: Genome, n_gait_seeds: int = 3) -> Tuple[float, Dict]:
    if not goal_aligned_feasible(genome):
        return -1e3, {"error": "goal-gate-rejected"}
    return evaluate_genome(genome, use_gate=False, n_gait_seeds=n_gait_seeds)


def search_with_goal_gate(budget: int, seed: int, pop_size: int = 8) -> Dict:
    import copy
    from experiments.m_search_loop import random_genome, mutate, crossover
    rng = np.random.default_rng(seed)
    population = []
    evals_used = 0
    while len(population) < pop_size and evals_used < budget:
        g = random_genome(rng)
        fit, _ = evaluate_with_goal_gate(g)
        evals_used += 1
        population.append((fit, g))
    while evals_used < budget:
        population.sort(key=lambda x: -x[0])
        survivors = [g for _, g in population[:max(2, pop_size // 2)]]
        if rng.random() < 0.5 and len(survivors) >= 2:
            i, j = rng.choice(len(survivors), 2, replace=False)
            child = crossover(survivors[i], survivors[j], rng)
        else:
            child = mutate(survivors[rng.integers(len(survivors))], rng)
        fit, _ = evaluate_with_goal_gate(child)
        evals_used += 1
        population.append((fit, child))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
    return {"best_fitness": max(f for f, _ in population)}


def run(n_seeds: int = 8, budget: int = 40, verbose: bool = True) -> Dict:
    goal_gate_fits, orthogonal_gate_fits, no_gate_fits = [], [], []
    for s in range(n_seeds):
        goal = search_with_goal_gate(budget, seed=s)
        orthogonal = evolutionary_search(budget, seed=s, use_gate=True, verbose=False)
        none = evolutionary_search(budget, seed=s, use_gate=False, verbose=False)
        goal_gate_fits.append(goal["best_fitness"])
        orthogonal_gate_fits.append(orthogonal["best_fitness"])
        no_gate_fits.append(none["best_fitness"])
        if verbose:
            print(f"seed={s}: goal-aligned-gate={goal_gate_fits[-1]:+.3f}  "
                 f"orthogonal-gate(paper)={orthogonal_gate_fits[-1]:+.3f}  "
                 f"no-gate={no_gate_fits[-1]:+.3f}")

    g_arr, o_arr, n_arr = map(np.array, (goal_gate_fits, orthogonal_gate_fits, no_gate_fits))
    if verbose:
        print(f"\ngoal-aligned gate : {g_arr.mean():.3f} +/- {g_arr.std():.3f}")
        print(f"orthogonal gate (paper's original) : {o_arr.mean():.3f} +/- {o_arr.std():.3f}")
        print(f"no gate            : {n_arr.mean():.3f} +/- {n_arr.std():.3f}")
        print(f"\ngoal-aligned gate beats no-gate on "
             f"{(g_arr > n_arr).sum()}/{n_seeds} seeds "
             f"(paper's orthogonal gate beat no-gate on "
             f"{(o_arr > n_arr).sum()}/{n_seeds})")
        reversed_result = g_arr.mean() > n_arr.mean()
        print(f"\nDoes goal-alignment reverse the paper's negative result?: "
             f"{reversed_result}")
    return {"goal_gate": goal_gate_fits, "orthogonal_gate": orthogonal_gate_fits,
           "no_gate": no_gate_fits}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=40)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds, budget=args.budget)

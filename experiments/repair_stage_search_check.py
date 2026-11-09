"""
Does enabling the penetration-threshold repair/rejection stage
(experiments/contact_sensitivity.py) change the paper's main search
conclusion for grammar-guided evolution?

Section~sec:budget-sweep and Table~tab:search21 report grammar-guided
evolution's mean fitness as $0.422\\pm0.141$ at budget 60, $n=8$ seeds,
measured with NO penetration check at all -- every candidate that loads
and steps is evaluated as-is, however deep its self-contact penetration.
Appendix~app:pitfalls's contact-sensitivity study shows this penetration
can reach several centimeters and proposes an explicit threshold with a
repair stage as a deployment option, but deliberately does not wire it
into the main search loop, to avoid confounding the already-reported
numbers with a change to the generator. This script asks the natural
follow-up directly: if that repair stage HAD been wired in, would the
main number change enough to matter?

Every genome the search loop evaluates is first checked against a
penetration threshold (deepest_penetration, as in contact_sensitivity.py);
a genome that fails is repaired by shrinking its segment radius (same
shrink_factor=0.85, same max 5 attempts) and re-checked, or rejected
(fitness -1e3, same convention evaluate_genome already uses for
grammar-invalid or gate-rejected genomes) if still over threshold after
5 attempts. Everything else about the search loop -- the (mu+lambda)
evolutionary search, the feasibility gate, the CPG fitness proxy -- is
untouched; experiments/m_search_loop.py is not edited.

Run:
    python3 -m experiments.repair_stage_search_check --n_seeds 8 --budget 60
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from experiments.contact_sensitivity import deepest_penetration
from experiments.m_search_loop import (
    Genome, crossover, evaluate_genome, mutate, random_genome)
from experiments.mh1_xml_baseline import _tree_of
from sulba.feasibility import FeasibilitySpec
from gmm.mjcf_compiler import MJCFCompiler


@dataclass
class RepairStats:
    accepted_as_drawn: int = 0
    repaired: int = 0
    rejected: int = 0


def _shrunk_genome(genome: Genome, scale: float) -> Genome:
    return Genome(n_legs=genome.n_legs, segments_per_leg=genome.segments_per_leg,
                 seg_len=genome.seg_len, seg_rad=max(0.005, genome.seg_rad * scale),
                 angle_jitter=genome.angle_jitter).clamp()


def evaluate_genome_with_repair(genome: Genome, *, use_gate: bool = True,
                                spec: Optional[FeasibilitySpec] = None,
                                n_gait_seeds: int = 3, horizon_steps: int = 500,
                                grammar_seed: int = 0, threshold: float = 0.02,
                                max_repair_attempts: int = 5, shrink_factor: float = 0.85,
                                settle_steps: int = 200,
                                stats_out: Optional[RepairStats] = None) -> Tuple[float, Dict]:
    """Same contract as m_search_loop.evaluate_genome, with one additional
    stage inserted before the feasibility gate: a penetration-threshold
    check with repair (radius shrink, re-derived through the grammar each
    time) and hard rejection, exactly as contact_sensitivity.repair_body
    implements it, but operating on the SAME genome object the search
    loop is already mutating, so the rest of evaluate_genome's contract
    (fitness convention, -1e3 on rejection) is unaffected."""
    comp = MJCFCompiler()
    g = genome
    scale = 1.0
    accepted = False
    for attempt in range(max_repair_attempts + 1):
        try:
            tree = _tree_of(g, grammar_seed=grammar_seed)
            xml = comp.compile(tree)
        except Exception:
            scale *= shrink_factor
            g = _shrunk_genome(genome, scale)
            continue
        pen = deepest_penetration(xml, settle_steps=settle_steps)
        if pen is not None and pen <= threshold:
            accepted = True
            break
        scale *= shrink_factor
        g = _shrunk_genome(genome, scale)

    if stats_out is not None:
        if not accepted:
            stats_out.rejected += 1
        elif scale < 1.0:
            stats_out.repaired += 1
        else:
            stats_out.accepted_as_drawn += 1

    if not accepted:
        return -1e3, {"error": "repair-rejected"}

    return evaluate_genome(g, use_gate=use_gate, spec=spec, n_gait_seeds=n_gait_seeds,
                           horizon_steps=horizon_steps, grammar_seed=grammar_seed)


def evolutionary_search_with_repair(budget: int, seed: int = 0, pop_size: int = 8,
                                    use_gate: bool = True, spec=None,
                                    threshold: float = 0.02,
                                    stats_out: Optional[RepairStats] = None,
                                    verbose: bool = False) -> Dict:
    """Line-for-line the same (mu+lambda) loop as m_search_loop's own
    evolutionary_search, with evaluate_genome_with_repair in place of
    evaluate_genome. m_search_loop.py is not edited."""
    rng = np.random.default_rng(seed)
    population: List[Tuple[float, Genome]] = []
    history: List[float] = []
    evals_used = 0

    while len(population) < pop_size and evals_used < budget:
        genome = random_genome(rng)
        fit, _ = evaluate_genome_with_repair(genome, use_gate=use_gate, spec=spec,
                                             threshold=threshold, stats_out=stats_out)
        evals_used += 1
        population.append((fit, genome))
        history.append(max(f for f, _ in population))

    while evals_used < budget:
        population.sort(key=lambda x: -x[0])
        survivors = [g for _, g in population[:max(2, pop_size // 2)]]
        if rng.random() < 0.5 and len(survivors) >= 2:
            i, j = rng.choice(len(survivors), 2, replace=False)
            child = crossover(survivors[i], survivors[j], rng)
        else:
            parent = survivors[rng.integers(len(survivors))]
            child = mutate(parent, rng)
        fit, _ = evaluate_genome_with_repair(child, use_gate=use_gate, spec=spec,
                                             threshold=threshold, stats_out=stats_out)
        evals_used += 1
        population.append((fit, child))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
        history.append(population[0][0])
        if verbose and evals_used % 10 == 0:
            print(f"  [evo+repair] eval {evals_used:3d}/{budget}  best={population[0][0]:+.4f}")

    best_fit, best_genome = max(population, key=lambda x: x[0])
    return {"history": history, "best_fitness": best_fit,
           "best_genome": best_genome, "evals_used": evals_used}


# The paper's already-reported, no-repair-stage number for grammar-guided
# evolution at budget=60, n=8 seeds (Table~tab:search21 / tab:recent-sota).
REPORTED_NO_REPAIR_MEAN = 0.422
REPORTED_NO_REPAIR_STD = 0.141


def run(n_seeds: int = 8, budget: int = 60, threshold: float = 0.02,
       verbose: bool = True) -> Dict:
    fits = []
    stats_out = RepairStats()
    for seed in range(n_seeds):
        result = evolutionary_search_with_repair(budget=budget, seed=seed,
                                                 threshold=threshold, stats_out=stats_out)
        fits.append(result["best_fitness"])
        if verbose:
            print(f"  seed={seed}: best_fitness_with_repair={result['best_fitness']:+.4f}")

    fits = np.array(fits)
    mean, std = float(np.mean(fits)), float(np.std(fits))
    t, p = stats.ttest_1samp(fits, REPORTED_NO_REPAIR_MEAN)
    n_total = stats_out.accepted_as_drawn + stats_out.repaired + stats_out.rejected

    if verbose:
        print(f"\nWith repair/rejection stage (threshold={threshold*100:.1f}cm), "
             f"n={n_seeds} seeds, budget={budget}:")
        print(f"  mean +/- std = {mean:+.4f} +/- {std:.4f}")
        print(f"  reported (no repair stage) = {REPORTED_NO_REPAIR_MEAN:+.4f} +/- "
             f"{REPORTED_NO_REPAIR_STD:.4f}")
        print(f"  one-sample t-test vs. reported mean: t={t:.3f}  p={p:.4f}")
        print(f"  genomes checked: {n_total}  accepted as drawn: "
             f"{stats_out.accepted_as_drawn}  repaired: {stats_out.repaired}  "
             f"rejected: {stats_out.rejected}  "
             f"(rejection rate {stats_out.rejected/max(1,n_total)*100:.1f}%)")
    return dict(mean=mean, std=std, fits=fits.tolist(), t=float(t), p=float(p),
               accepted_as_drawn=stats_out.accepted_as_drawn, repaired=stats_out.repaired,
               rejected=stats_out.rejected, n_checked=n_total)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.02)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds, budget=args.budget, threshold=args.threshold)

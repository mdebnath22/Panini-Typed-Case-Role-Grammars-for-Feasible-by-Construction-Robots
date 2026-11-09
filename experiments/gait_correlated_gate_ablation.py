"""
Does a gait-correlated feasibility gate reverse the negative result in
mh6b_gate_ablation.py / Section 4.3 (``Feasibility Gate versus Task
Optimality'')?
=============================================================================

The paper's own Limitations section raised this directly: the feasibility
gate (sulba/feasibility.py's check_feasibility) only checks static
geometry -- workspace reach, ground clearance, mass budget, support-
polygon area, minimum inter-terminal separation -- all orthogonal to the
locomotion task metric itself. Under a fixed physics-evaluation budget,
gate-active search does not improve, and directionally reduces, best task
fitness (Table tab:gate: $1.137\\pm0.745$ gate-on vs.\\ $1.672\\pm0.661$
gate-off, $p=0.254$), because the gate rejects geometrically unconventional
but effective bodies the locomotion fitness itself has no objection to.
The question this script tests, stated before running it: does adding a
check that IS correlated with gait quality -- not geometry for its own
sake -- close or reverse this gap?

The new check, sulba.feasibility.check_gait_capability, runs a SHORT CPG
rollout (100 steps vs.\\ the full 1000-step evaluation horizon used
elsewhere) and rejects a body that cannot produce at least 1cm of forward
motion in that short window. Unlike every check in check_feasibility, this
one is not free -- it costs a real physics rollout -- so we charge it a
FRACTIONAL budget cost (100/1000 = 0.1 of a full evaluation) against the
same shared currency every other method in this paper uses, consistent
with the accounting discipline mh6b_gate_ablation.py itself established
("a rejection costs near-zero physics time" no longer holds for this
check, so pretending it is free would unfairly subsidize this condition).

Three conditions, same fixed PHYSICS budget (in full-evaluation-equivalent
units), same search loop, same demanding specification as
mh6b_gate_ablation.DEMANDING_SPEC:
  1. No gate (baseline, already reported in Table tab:gate's "gate
     inactive" column).
  2. Geometric gate only (already reported in Table tab:gate's "gate
     active" column).
  3. Geometric gate + gait-capability pre-check (the new condition this
     script adds): a genome must pass BOTH the geometric gate AND the
     short-rollout gait-capability check before it is charged a full
     physics evaluation; failing either is a free-or-fractional rejection
     as appropriate.

Run:
    python3 -m experiments.gait_correlated_gate_ablation --n_seeds 8 --physics_budget 40
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.m_search_loop import Genome, random_genome, mutate, crossover
from experiments.mh6b_gate_ablation import DEMANDING_SPEC
from ptg.body_grammar import BodyGrammar
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody
from gmm.task_fitness import evaluate_fitness
from sulba.feasibility import FeasibilitySpec, check_feasibility, check_gait_capability

# the short gait-capability rollout's horizon as a fraction of the full
# 1000-step evaluation horizon used everywhere else -- charged against the
# shared physics budget at this fraction, since it is a real (if short)
# MuJoCo rollout, not a free geometric check.
_GAIT_CHECK_HORIZON = 100
_FULL_HORIZON = 1000
_GAIT_CHECK_COST = _GAIT_CHECK_HORIZON / _FULL_HORIZON  # 0.1


def _build_tree(genome: Genome, grammar_seed: int = 0):
    g = BodyGrammar(seed_body_rules(), seed=grammar_seed)
    deriv = g.derive(genome.to_plan())
    return PhiBody().build_tree(deriv)


def _evaluate_with_condition(genome: Genome, spec: FeasibilitySpec, condition: str,
                             n_gait_seeds: int = 3, horizon_steps: int = 500,
                             ) -> Tuple[Optional[float], float]:
    """Returns (fitness_or_None, physics_cost_charged). fitness is None for
    a rejected genome (gate or gait-capability), in which case the caller
    must NOT count this as a used population slot but MUST still charge
    `physics_cost_charged` against the budget if it is nonzero (the gait-
    capability check's short rollout)."""
    try:
        tree = _build_tree(genome)
    except Exception:
        return None, 0.0

    if condition in ("geometric_gate", "gait_correlated_gate"):
        rep = check_feasibility(tree, spec)
        if not rep.ok:
            return None, 0.0  # free rejection, same as mh6b_gate_ablation

    if condition == "gait_correlated_gate":
        gcap = check_gait_capability(tree, horizon_steps=_GAIT_CHECK_HORIZON)
        if not gcap.ok:
            return None, _GAIT_CHECK_COST  # charged: this was a real rollout

    # survived whatever gating this condition applies -- full evaluation
    fits = []
    for s in range(n_gait_seeds):
        r = evaluate_fitness(tree, horizon_steps=horizon_steps, seed=s)
        if r.ok:
            fits.append(r.fitness)
    extra_cost = _GAIT_CHECK_COST if condition == "gait_correlated_gate" else 0.0
    if not fits:
        return -1e3, 1.0 + extra_cost
    return float(np.mean(fits)), 1.0 + extra_cost


def _search_with_physics_budget(physics_budget: float, seed: int, condition: str,
                                spec, pop_size: int = 8, max_tries: int = 20000
                                ) -> Dict:
    rng = np.random.default_rng(seed)
    population: List[Tuple[float, Genome]] = []
    physics_used = 0.0
    tries = 0
    history: List[float] = []

    def try_genome(genome: Genome) -> Optional[float]:
        nonlocal physics_used, tries
        tries += 1
        fit, cost = _evaluate_with_condition(genome, spec, condition)
        physics_used += cost
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
           "physics_used": physics_used, "tries": tries, "best_genome": best_genome}


def run(n_seeds: int = 8, physics_budget: float = 40, spec=None,
       verbose: bool = True) -> Dict:
    spec = spec or DEMANDING_SPEC
    conditions = ["no_gate", "geometric_gate", "gait_correlated_gate"]
    results: Dict[str, List[float]] = {c: [] for c in conditions}

    for s in range(n_seeds):
        row = {}
        for c in conditions:
            r = _search_with_physics_budget(physics_budget, s, c, spec)
            results[c].append(r["best_fitness"])
            row[c] = r["best_fitness"]
        if verbose:
            print(f"  seed={s}: no_gate={row['no_gate']:+.3f}  "
                 f"geometric_gate={row['geometric_gate']:+.3f}  "
                 f"gait_correlated_gate={row['gait_correlated_gate']:+.3f}")

    arrs = {c: np.array(results[c]) for c in conditions}
    try:
        from scipy import stats
        t_geo, p_geo = stats.ttest_rel(arrs["geometric_gate"], arrs["no_gate"])
        t_gait, p_gait = stats.ttest_rel(arrs["gait_correlated_gate"], arrs["no_gate"])
        t_gait_vs_geo, p_gait_vs_geo = stats.ttest_rel(arrs["gait_correlated_gate"], arrs["geometric_gate"])
    except Exception:
        t_geo = p_geo = t_gait = p_gait = t_gait_vs_geo = p_gait_vs_geo = float("nan")

    wins_geo = int((arrs["geometric_gate"] > arrs["no_gate"]).sum())
    wins_gait = int((arrs["gait_correlated_gate"] > arrs["no_gate"]).sum())

    if verbose:
        print()
        for c in conditions:
            print(f"{c:<24}: {arrs[c].mean():+.3f} +/- {arrs[c].std():.3f}")
        print(f"\ngeometric_gate      vs no_gate: wins {wins_geo}/{n_seeds}, "
             f"t={t_geo:.3f}, p={p_geo:.3f}")
        print(f"gait_correlated_gate vs no_gate: wins {wins_gait}/{n_seeds}, "
             f"t={t_gait:.3f}, p={p_gait:.3f}")
        print(f"gait_correlated_gate vs geometric_gate: "
             f"t={t_gait_vs_geo:.3f}, p={p_gait_vs_geo:.3f}")
    return {"results": results, "wins_geo": wins_geo, "wins_gait": wins_gait,
           "p_geo": float(p_geo), "p_gait": float(p_gait),
           "p_gait_vs_geo": float(p_gait_vs_geo)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--physics_budget", type=float, default=40)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds, physics_budget=args.physics_budget)

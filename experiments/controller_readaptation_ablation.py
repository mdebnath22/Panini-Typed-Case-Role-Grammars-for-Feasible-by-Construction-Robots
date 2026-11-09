"""
Controller-Readaptation Ablation
=================================

Motivating question (raised in review, citing Cheney et al. 2018 "Scalable
co-optimization of morphology and control in embodied machines" and
Strgar & Kriegman 2026 "Accelerated co-design of robots through
morphological pretraining"): the search loop in `m_search_loop.py` scores
every genome -- including freshly-mutated children -- with a CPG gait whose
parameters (freq=1.5, amp=0.8, phase_offset=0.0; see
`gmm/task_fitness.py::evaluate_fitness`'s defaults, which `evaluate_genome`
never overrides) are IDENTICAL for every body, never re-tuned per-morphology.
That is a real instance of the morphology-evolution literature's "unfair
immediate evaluation" confound: a structural mutation that would be good
under a gait tuned for the new body can look bad when scored with a gait
tuned (implicitly, by being fixed at all) for nothing in particular, which
could systematically penalize exactly the structural diversity grammar-guided
search is supposed to produce.

VERIFIED BEFORE building this: `evaluate_genome` in `m_search_loop.py` calls
`evaluate_fitness(tree, horizon_steps=..., seed=s)` with NO freq/amp/
phase_offset arguments for every genome in every search method (evolutionary,
random). The controller is case (a) in the task framing: fixed/shared across
all bodies, never re-optimized per-body. So the confound is live here, not a
non-issue -- this script measures how much it matters.

Three evaluation conditions, equal TOTAL physics-rollout budget throughout
(a "rollout" = one `gmm.task_fitness.evaluate_fitness` call; this is more
granular than `evaluate_genome`'s own accounting, which always spends exactly
`n_gait_seeds=3` rollouts per genome -- here conditions B/C spend their
per-genome rollouts differently, so the finer unit is required for a fair
comparison):

  A. IMMEDIATE (control, matches current m_search_loop.py behaviour exactly):
     every genome scored by averaging 3 rollouts, all at the fixed default
     controller (freq=1.5, amp=0.8, phase_offset=0.0). Cost: 3 rollouts/genome.

  B. LIMITED READAPTATION: same 3-rollout-per-genome budget, but spent on a
     short local search over controller params instead of pure noise
     averaging -- rollout 1 at the default controller, then up to 2 more
     rollouts trying a locally-perturbed controller (freq/amp/phase_offset
     jittered around the current best), keeping whichever controller setting
     scored best. Same cost as A (3 rollouts/genome) so the SAME number of
     genomes get evaluated per unit budget -- this isolates "what you do with
     a fixed per-genome rollout budget", not "how much extra compute you get".

  C. INNOVATION PROTECTION (closer to the Cheney et al. mechanism): a new
     genome is first scored cheaply (2 rollouts averaged, not 3 -- saving a
     rollout relative to A/B). If, at the next truncation step, that genome
     would be the one eliminated AND it has not yet had its one grace
     re-evaluation, it gets one more rollout (the 3rd) averaged into its
     score before the elimination decision is finalized, rather than being
     discarded on a noisier, cheaper first look. A genome that is clearly
     good enough never spends the 3rd rollout at all, so the saved budget is
     available to evaluate more genomes elsewhere -- still bounded by the
     SAME total rollout budget as A/B (the loop simply stops once the total
     is spent, exactly like A/B).

Gate rejections (Śulba feasibility check) cost 0 rollouts in every
condition, matching `evaluate_genome`'s / `mh6b_gate_ablation.py`'s existing
"budget counts physics evaluations only" convention.

Does NOT touch gmm/phi_body.py, m_search_loop.py's Genome.mount_plan()/
mutate(), or any gap/feasible_parent code -- those are read-only imports
here (BodyGrammar/PhiBody/seed_body_rules/check_feasibility/Genome/
random_genome/mutate/crossover come straight from the existing modules).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ptg.body_grammar import BodyGrammar
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody
from gmm.task_fitness import evaluate_fitness
from sulba.feasibility import FeasibilitySpec, check_feasibility

from experiments.m_search_loop import Genome, random_genome, mutate, crossover

DEFAULT_SPEC = FeasibilitySpec(required_reach=0.15, mass_budget=6.0,
                               min_support_area=0.01)
DEFAULT_CTRL = dict(freq=1.5, amp=0.8, phase_offset=0.0)


# ---------------------------------------------------------------------------
# Low-level: build once per genome, roll out as many times as a condition
# needs, at whatever controller params that condition chooses.
# ---------------------------------------------------------------------------

def _build_tree(genome: Genome, spec: FeasibilitySpec, use_gate: bool,
                grammar_seed: int = 0):
    """Returns tree, or None if grammar-invalid / gate-rejected (free)."""
    g = BodyGrammar(seed_body_rules(), seed=grammar_seed)
    try:
        deriv = g.derive(genome.to_plan())
    except Exception:
        return None
    tree = PhiBody().build_tree(deriv)
    if use_gate:
        rep = check_feasibility(tree, spec)
        if not rep.ok:
            return None
    return tree


def _rollout(tree, seed: int, ctrl: Dict[str, float],
            horizon_steps: int = 500) -> Optional[float]:
    r = evaluate_fitness(tree, horizon_steps=horizon_steps, seed=seed, **ctrl)
    return r.fitness if r.ok else None


# ---------------------------------------------------------------------------
# Condition-specific per-genome scorers. Each returns (fitness, rollouts_spent).
# fitness == -1e3 and rollouts_spent == 0 means grammar-invalid/gate-rejected.
# ---------------------------------------------------------------------------

def score_A(genome: Genome, spec, use_gate, rng, horizon_steps=500) -> Tuple[float, int]:
    tree = _build_tree(genome, spec, use_gate)
    if tree is None:
        return -1e3, 0
    fits = [f for f in (_rollout(tree, s, DEFAULT_CTRL, horizon_steps) for s in range(3))
            if f is not None]
    if not fits:
        return -1e3, 3
    return float(np.mean(fits)), 3


def score_B(genome: Genome, spec, use_gate, rng, horizon_steps=500) -> Tuple[float, int]:
    """3-rollout budget spent on a short local controller search instead of
    pure seed-averaging: rollout 1 at the default controller, then up to 2
    local perturbations, keeping the best controller setting found."""
    tree = _build_tree(genome, spec, use_gate)
    if tree is None:
        return -1e3, 0
    best_ctrl = dict(DEFAULT_CTRL)
    best_fit = _rollout(tree, 0, best_ctrl, horizon_steps)
    spent = 1
    if best_fit is None:
        best_fit = -1e3
    for step in range(2):
        cand_ctrl = dict(
            freq=float(np.clip(best_ctrl["freq"] * rng.uniform(0.85, 1.15), 0.5, 3.0)),
            amp=float(np.clip(best_ctrl["amp"] * rng.uniform(0.85, 1.15), 0.2, 1.0)),
            phase_offset=float(best_ctrl["phase_offset"] + rng.uniform(-0.3, 0.3)),
        )
        f = _rollout(tree, step + 1, cand_ctrl, horizon_steps)
        spent += 1
        if f is not None and f > best_fit:
            best_fit, best_ctrl = f, cand_ctrl
    return best_fit, spent


def score_C_initial(genome: Genome, spec, use_gate, rng, horizon_steps=500):
    """Cheap first look: 2 rollouts averaged (not 3), default controller.
    Returns (fitness, rollouts_spent, tree_or_None, grace_used=False)."""
    tree = _build_tree(genome, spec, use_gate)
    if tree is None:
        return -1e3, 0, None, True  # nothing to protect; grace moot
    fits = [f for f in (_rollout(tree, s, DEFAULT_CTRL, horizon_steps) for s in range(2))
            if f is not None]
    spent = 2
    fit = float(np.mean(fits)) if fits else -1e3
    return fit, spent, tree, False


def score_C_grace(tree, old_fit: float, horizon_steps=500) -> Tuple[float, int]:
    """One more rollout (the 3rd seed), averaged into the existing score."""
    f = _rollout(tree, 2, DEFAULT_CTRL, horizon_steps)
    if f is None:
        return old_fit, 1
    new_fit = (old_fit * 2 + f) / 3.0
    return new_fit, 1


# ---------------------------------------------------------------------------
# Generic (mu+lambda)-style evolutionary loop, parameterized by condition.
# Mirrors m_search_loop.evolutionary_search's structure/selection logic
# exactly (same crossover/mutate probabilities, same truncation-to-pop_size)
# so condition is the ONLY thing that differs from the paper's existing
# grammar_guided_evolution.
# ---------------------------------------------------------------------------

@dataclass
class Indiv:
    fitness: float
    genome: Genome
    age: int = 0            # generations survived
    tree: Optional[object] = None
    grace_used: bool = True  # condition C only; True = not eligible for grace


def run_evolution(condition: str, budget: int, seed: int, pop_size: int = 8,
                  spec=None, use_gate: bool = True,
                  horizon_steps: int = 500) -> Dict:
    assert condition in ("A", "B", "C")
    rng = np.random.default_rng(seed)
    spec = spec or DEFAULT_SPEC
    population: List[Indiv] = []
    history: List[Tuple[int, float]] = []  # (rollouts_used, best_fitness_so_far)
    rollouts_used = 0
    best_so_far = -1e9

    def _record():
        nonlocal best_so_far
        cur_best = max((p.fitness for p in population), default=-1e9)
        best_so_far = max(best_so_far, cur_best)
        history.append((rollouts_used, best_so_far))

    def _score_new(genome: Genome) -> Indiv:
        nonlocal rollouts_used
        if condition == "A":
            fit, cost = score_A(genome, spec, use_gate, rng, horizon_steps)
            rollouts_used += cost
            return Indiv(fitness=fit, genome=genome, age=0, grace_used=True)
        if condition == "B":
            fit, cost = score_B(genome, spec, use_gate, rng, horizon_steps)
            rollouts_used += cost
            return Indiv(fitness=fit, genome=genome, age=0, grace_used=True)
        # condition C
        fit, cost, tree, grace_used = score_C_initial(genome, spec, use_gate, rng, horizon_steps)
        rollouts_used += cost
        return Indiv(fitness=fit, genome=genome, age=0, tree=tree, grace_used=grace_used)

    # seed initial population
    while len(population) < pop_size and rollouts_used < budget:
        population.append(_score_new(random_genome(rng)))
        _record()

    while rollouts_used < budget:
        population.sort(key=lambda p: -p.fitness)
        survivors = [p.genome for p in population[:max(2, pop_size // 2)]]
        if rng.random() < 0.5 and len(survivors) >= 2:
            i, j = rng.choice(len(survivors), 2, replace=False)
            child_genome = crossover(survivors[i], survivors[j], rng)
        else:
            parent = survivors[rng.integers(len(survivors))]
            child_genome = mutate(parent, rng)

        child = _score_new(child_genome)
        candidates = population + [child]
        candidates.sort(key=lambda p: -p.fitness)

        if condition == "C" and len(candidates) > pop_size:
            # would the weakest-ranked be cut a "fresh" (age==0), un-graced
            # individual? If so and budget allows, give it one grace
            # re-evaluation before finalizing the cut.
            cutoff = candidates[pop_size:]
            kept = candidates[:pop_size]
            still_cut = []
            for p in cutoff:
                if (not p.grace_used) and p.tree is not None and rollouts_used < budget:
                    new_fit, cost = score_C_grace(p.tree, p.fitness, horizon_steps)
                    rollouts_used += cost
                    p.fitness = new_fit
                    p.grace_used = True
                still_cut.append(p)
            candidates = kept + still_cut
            candidates.sort(key=lambda p: -p.fitness)

        for p in candidates:
            p.age += 1
        population = candidates[:pop_size]
        _record()

    best = max(population, key=lambda p: p.fitness)
    return {"history": history, "best_fitness": best.fitness,
           "best_genome": best.genome, "rollouts_used": rollouts_used}


def run_random_search(condition: str, budget: int, seed: int, spec=None,
                      use_gate: bool = True, horizon_steps: int = 500) -> Dict:
    """Same structure as m_search_loop.random_search (no selection pressure,
    so condition C's elimination-protection mechanism is a no-op here by
    construction -- only run A/B for random search)."""
    assert condition in ("A", "B")
    rng = np.random.default_rng(seed)
    spec = spec or DEFAULT_SPEC
    scorer = score_A if condition == "A" else score_B
    best_fit, best_genome = -1e9, None
    history: List[Tuple[int, float]] = []
    rollouts_used = 0
    while rollouts_used < budget:
        genome = random_genome(rng)
        fit, cost = scorer(genome, spec, use_gate, rng, horizon_steps)
        rollouts_used += cost
        if fit > best_fit:
            best_fit, best_genome = fit, genome
        history.append((rollouts_used, best_fit))
    return {"history": history, "best_fitness": best_fit,
           "best_genome": best_genome, "rollouts_used": rollouts_used}


def run_hill_climbing(condition: str, budget: int, seed: int, spec=None,
                      use_gate: bool = True, horizon_steps: int = 500) -> Dict:
    """(1+1)-style hill climber over the same genome space, for the
    relative-rank check (this method does not exist in m_search_loop.py for
    this genome encoding, so it is implemented fresh here, minimally)."""
    assert condition == "A", "hill_climbing rank-check is run under condition A only"
    rng = np.random.default_rng(seed)
    spec = spec or DEFAULT_SPEC
    current = random_genome(rng)
    cur_fit, cost = score_A(current, spec, use_gate, rng, horizon_steps)
    rollouts_used = cost
    best_fit = cur_fit
    history: List[Tuple[int, float]] = [(rollouts_used, best_fit)]
    while rollouts_used < budget:
        child = mutate(current, rng)
        fit, cost = score_A(child, spec, use_gate, rng, horizon_steps)
        rollouts_used += cost
        if fit >= cur_fit:
            current, cur_fit = child, fit
        best_fit = max(best_fit, fit)
        history.append((rollouts_used, best_fit))
    return {"history": history, "best_fitness": best_fit,
           "best_genome": current, "rollouts_used": rollouts_used}

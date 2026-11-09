"""
M2/M3 — The Outer Search Loop: Optimizing Morphology for a Task
==================================================================

This is the experiment that was missing from every prior result. H5/M-H5
prove the grammar's geometry is internally consistent; H6/M-H3 prove it's
compact; H-BOX proves paribhāṣā is geometrically recoverable. NONE of that
shows the grammar can be *searched* to find a body that is actually good at
a task — that is the thesis's central claim, and this is the first
experiment that tests it directly.

Design (grammar-level evolutionary search):

  A "genome" here is not raw MuJoCo XML — it's a small set of GRAMMAR
  PARAMETERS: leg count, segments per leg, per-limb seg_len/seg_rad, and
  mount-angle spacing. Mutating a genome edits these parameters and
  re-derives the body through the SAME grammar (anuvṛtti still applies:
  changing "seg_len" changes every inheriting limb consistently). This is
  the central comparison against raw-XML mutation (RoboMorph/LASeR-style):
  every mutation here is legal-by-construction, because it's still a
  grammar derivation, not an edited XML tree.

Fitness proxy: the CPG-driven locomotion evaluator in gmm/task_fitness.py.
This is a coarse, cheap proxy (deterministic sinusoidal gait, not a trained
policy) — standard practice in morphology-search literature (Sims-style) so
morphology search isn't confounded with control-policy search in this first
pass. A trained-policy fitness (Swarm-JEPA per body) is the natural, more
expensive refinement (Phase M2's later stage), not what's tested here.
Because the CPG signal is noisy (a lucky/unlucky initial phase can dominate
short rollouts — see this file's companion note in task_fitness.py), every
genome's fitness is AVERAGED over several gait seeds.

Three conditions compared:
  1. Grammar-guided evolutionary search (mutate genome, re-derive via grammar)
  2. Random search (same genome space, no selection pressure)
  3. (Optional) raw-parameter random search with NO Śulba pre-filter, to
     isolate the gate's contribution to search efficiency (ties to M-H6)

This is M-H2-adjacent (task-fit search) and directly operationalizes the
roadmap's Phase M3.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ptg.body_grammar import BodyGrammar, legged_plan
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody, leg_mount_plan
from gmm.task_fitness import evaluate_fitness
from sulba.feasibility import FeasibilitySpec, check_feasibility


@dataclass
class Genome:
    n_legs: int
    segments_per_leg: int
    seg_len: float
    seg_rad: float
    angle_jitter: float = 0.0  # perturbation added to even leg spacing

    def clamp(self) -> "Genome":
        self.n_legs = int(np.clip(self.n_legs, 2, 8))
        self.segments_per_leg = int(np.clip(self.segments_per_leg, 1, 3))
        self.seg_len = float(np.clip(self.seg_len, 0.05, 0.30))
        self.seg_rad = float(np.clip(self.seg_rad, 0.01, 0.05))
        self.angle_jitter = float(np.clip(self.angle_jitter, 0.0, 0.3))
        return self

    def mount_plan(self) -> Tuple[List[float], List[float]]:
        base_angles, xs = leg_mount_plan(self.n_legs)
        base_angles = np.array(base_angles)
        # BUG FIX (2026-10-24): the jitter RNG used to be seeded from
        # hash(self) over ALL FIVE fields (n_legs, segments_per_leg,
        # seg_len, seg_rad, angle_jitter, each rounded to 4dp) -- so a
        # change of as little as 1e-4 in seg_len or seg_rad (far smaller
        # than any mutation/refinement step any search method takes, and
        # with no causal effect on leg angle whatsoever) silently
        # re-sampled a COMPLETELY DIFFERENT per-leg jitter direction
        # vector, and changing angle_jitter itself did the same instead of
        # smoothly scaling a fixed direction. This made EVERY continuous
        # dimension of the genome space discontinuous/non-smooth in mount
        # angle (confirmed empirically: seg_len+1e-4 moved every leg angle
        # by ~0.1-0.2 rad, the same order as angle_jitter's own range) --
        # death for any local-refinement/hill-climbing-style search step
        # (evolutionary_search's mutate(), lhs_plus_refinement's local
        # phase, hill_climbing, cma_es_lite's mean-shift, NES), which all
        # rely on small parameter changes producing correspondingly small
        # fitness changes. The jitter DIRECTION pattern must depend only
        # on n_legs (the only thing that determines how many angles there
        # are to jitter and their base layout) and nothing else, so it
        # stays fixed across seg_len/seg_rad/angle_jitter changes; the
        # jitter MAGNITUDE then scales that fixed direction continuously
        # by angle_jitter, so angle_jitter=0 is exactly "no jitter" and
        # small angle_jitter changes produce small angle changes.
        rng = np.random.default_rng(int(self.n_legs) * 1_000_003 + 17)
        unit_dir = rng.uniform(-1.0, 1.0, self.n_legs)
        jittered = base_angles + unit_dir * self.angle_jitter
        # Clip to [pi, 2*pi] so jitter can never push a leg out of the lower
        # hemisphere (see gmm.phi_body.lower_hemisphere_mount_angles) -- the
        # feasibility gate's support-polygon check assumes every leg has a
        # non-positive Z component. x offsets (leg_mount_plan's longitudinal
        # placement, which is what gives the support-polygon real area) are
        # left unjittered -- only the circumferential angle is perturbed.
        return list(np.clip(jittered, np.pi, 2 * np.pi)), xs

    def mount_angles(self) -> List[float]:
        return self.mount_plan()[0]

    def __hash__(self):
        return hash((self.n_legs, self.segments_per_leg,
                    round(self.seg_len, 4), round(self.seg_rad, 4),
                    round(self.angle_jitter, 4)))

    def to_plan(self) -> List[Dict]:
        angles, xs = self.mount_plan()
        return legged_plan(n_legs=self.n_legs,
                           segments_per_leg=self.segments_per_leg,
                           mount_angles=angles, mount_x_offsets=xs,
                           seg_len=self.seg_len, seg_rad=self.seg_rad)


def random_genome(rng: np.random.Generator) -> Genome:
    return Genome(
        n_legs=int(rng.integers(2, 9)),
        segments_per_leg=int(rng.integers(1, 4)),
        seg_len=float(rng.uniform(0.06, 0.22)),
        seg_rad=float(rng.uniform(0.015, 0.035)),
        angle_jitter=float(rng.uniform(0.0, 0.2)),
    ).clamp()


def mutate(genome: Genome, rng: np.random.Generator, rate: float = 0.5) -> Genome:
    """Grammar-level mutation: edit genome parameters, re-derive through
    the grammar. Every result is legal-by-construction — anuvṛtti guarantees
    every inheriting limb stays consistent after a seg_len/seg_rad edit.

    BUG FIX (2026-10-24): `rate` used to be an unused parameter -- every
    call mutated exactly ONE of the 5 fields (rng.integers(0, 5) picked
    one branch), regardless of `rate`. That makes evolutionary_search's
    exploitation phase a pure coordinate-wise search over a 5-D space,
    far weaker per-offspring than the other methods it's compared against
    (CMA-ES-lite/NES/PSO/lhs_plus_refinement's local phase all perturb
    every dimension jointly every step). `rate` now does what its name
    says: each field mutates independently with probability `rate`
    (at least one field always mutates, so a call never returns a no-op
    copy)."""
    g = copy.copy(genome)
    mask = rng.random(5) < rate
    if not mask.any():
        mask[rng.integers(0, 5)] = True
    if mask[0]:
        g.n_legs += int(rng.choice([-2, -1, 1, 2]))
    if mask[1]:
        g.segments_per_leg += int(rng.choice([-1, 1]))
    if mask[2]:
        g.seg_len *= float(rng.uniform(0.7, 1.4))
    if mask[3]:
        g.seg_rad *= float(rng.uniform(0.7, 1.4))
    if mask[4]:
        g.angle_jitter += float(rng.uniform(-0.1, 0.1))
    return g.clamp()


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    return Genome(
        n_legs=a.n_legs if rng.random() < 0.5 else b.n_legs,
        segments_per_leg=a.segments_per_leg if rng.random() < 0.5 else b.segments_per_leg,
        seg_len=(a.seg_len + b.seg_len) / 2,
        seg_rad=(a.seg_rad + b.seg_rad) / 2,
        angle_jitter=(a.angle_jitter + b.angle_jitter) / 2,
    ).clamp()


# ---------------------------------------------------------------------------
# Evaluation: grammar -> tree -> (Śulba gate) -> physics fitness
# ---------------------------------------------------------------------------

def evaluate_genome(genome: Genome, *, use_gate: bool = True,
                    spec: Optional[FeasibilitySpec] = None,
                    n_gait_seeds: int = 3, horizon_steps: int = 500,
                    grammar_seed: int = 0) -> Tuple[float, Dict]:
    """Returns (fitness, info). Fitness is -1e3 for grammar-invalid or
    feasibility-gate-rejected genomes (so the search learns to avoid them
    without an expensive physics call in the rejected case)."""
    spec = spec or FeasibilitySpec(required_reach=0.15, mass_budget=6.0,
                                   min_support_area=0.01)
    g = BodyGrammar(seed_body_rules(), seed=grammar_seed)
    try:
        deriv = g.derive(genome.to_plan())
    except Exception as e:
        return -1e3, {"error": f"grammar: {e}"}

    tree = PhiBody().build_tree(deriv)

    if use_gate:
        rep = check_feasibility(tree, spec)
        if not rep.ok:
            return -1e3, {"error": "infeasible", "checks": rep.checks}

    fits = []
    for s in range(n_gait_seeds):
        r = evaluate_fitness(tree, horizon_steps=horizon_steps, seed=s)
        if r.ok:
            fits.append(r.fitness)
    if not fits:
        return -1e3, {"error": "all rollouts failed"}
    return float(np.mean(fits)), {"n_ok": len(fits), "std": float(np.std(fits))}


# ---------------------------------------------------------------------------
# Search loops
# ---------------------------------------------------------------------------

def evolutionary_search(budget: int, seed: int = 0, pop_size: int = 8,
                        use_gate: bool = True, spec=None,
                        verbose: bool = True) -> Dict:
    """(mu+lambda)-style: keep the best `pop_size` genomes seen so far,
    generate the next batch by mutating/crossing-over survivors."""
    rng = np.random.default_rng(seed)
    population: List[Tuple[float, Genome]] = []
    history: List[float] = []
    evals_used = 0

    # seed initial population
    while len(population) < pop_size and evals_used < budget:
        genome = random_genome(rng)
        fit, _ = evaluate_genome(genome, use_gate=use_gate, spec=spec)
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
        fit, _ = evaluate_genome(child, use_gate=use_gate, spec=spec)
        evals_used += 1
        population.append((fit, child))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
        history.append(population[0][0])
        if verbose and evals_used % 10 == 0:
            print(f"  [evo] eval {evals_used:3d}/{budget}  best={population[0][0]:+.4f}")

    best_fit, best_genome = max(population, key=lambda x: x[0])
    return {"history": history, "best_fitness": best_fit,
           "best_genome": best_genome, "evals_used": evals_used}


def random_search(budget: int, seed: int = 0, use_gate: bool = True,
                  spec=None, verbose: bool = True) -> Dict:
    rng = np.random.default_rng(seed)
    best_fit, best_genome = -1e9, None
    history = []
    for i in range(budget):
        genome = random_genome(rng)
        fit, _ = evaluate_genome(genome, use_gate=use_gate, spec=spec)
        if fit > best_fit:
            best_fit, best_genome = fit, genome
        history.append(best_fit)
        if verbose and (i + 1) % 10 == 0:
            print(f"  [rand] eval {i+1:3d}/{budget}  best={best_fit:+.4f}")
    return {"history": history, "best_fitness": best_fit,
           "best_genome": best_genome, "evals_used": budget}


def benchmark(n_seeds: int = 8, budget: int = 60, verbose: bool = True) -> Dict:
    """The honest report: run both searches across several seeds and compare
    means with a paired test, rather than quoting one seed's result. Fitness
    here is a noisy CPG-driven proxy (see task_fitness.py), so single-seed
    comparisons can mislead — one seed can show evolutionary search LOSING
    to random due to variance alone."""
    evo_fits, rnd_fits = [], []
    for s in range(n_seeds):
        evo = evolutionary_search(budget, seed=s, verbose=False)
        rnd = random_search(budget, seed=s + 1000, verbose=False)
        evo_fits.append(evo["best_fitness"])
        rnd_fits.append(rnd["best_fitness"])
        if verbose:
            diff = evo_fits[-1] - rnd_fits[-1]
            print(f"  seed={s}: evo={evo_fits[-1]:+.3f}  rnd={rnd_fits[-1]:+.3f}  "
                 f"diff={diff:+.3f}")

    evo_arr, rnd_arr = np.array(evo_fits), np.array(rnd_fits)
    wins = int((evo_arr > rnd_arr).sum())
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_rel(evo_arr, rnd_arr)
    except Exception:
        t_stat, p_val = float("nan"), float("nan")

    if verbose:
        print(f"\nevolutionary: {evo_arr.mean():+.3f} ± {evo_arr.std():.3f}")
        print(f"random      : {rnd_arr.mean():+.3f} ± {rnd_arr.std():.3f}")
        print(f"evolutionary wins {wins}/{n_seeds} seeds")
        print(f"paired t-test: t={t_stat:.3f}  p={p_val:.3f}")
        sig = p_val < 0.05 if p_val == p_val else False  # nan check
        verdict = "YES" if sig else "NOT YET (increase n_seeds/budget)"
        print(f"significant at p<0.05: {verdict}")
    return {"evo_fits": evo_fits, "rnd_fits": rnd_fits, "wins": wins,
           "n_seeds": n_seeds, "t_stat": t_stat, "p_val": p_val}


def run(budget: int = 60, seed: int = 0, verbose: bool = True) -> Dict:
    random.seed(seed)
    if verbose:
        print("M3 — Grammar-guided evolutionary search vs random search")
        print(f"(budget={budget} physics evaluations each, task=locomotion)\n")
        print("Evolutionary search:")
    evo = evolutionary_search(budget, seed=seed, verbose=verbose)
    if verbose:
        print("\nRandom search:")
    rnd = random_search(budget, seed=seed + 1, verbose=verbose)

    if verbose:
        print(f"\nbest fitness — evolutionary: {evo['best_fitness']:+.4f}  "
              f"({evo['best_genome']})")
        print(f"best fitness — random      : {rnd['best_fitness']:+.4f}  "
              f"({rnd['best_genome']})")
        improvement = evo["best_fitness"] - rnd["best_fitness"]
        print(f"\nimprovement of grammar-guided search over random: "
              f"{improvement:+.4f}")
        go = evo["best_fitness"] > rnd["best_fitness"]
        print(f"GO/NO-GO (evolutionary search beats random search): "
              f"{'GO' if go else 'NO-GO'}")
    return {"evo": evo, "rnd": rnd}


if __name__ == "__main__":
    print("Single-seed illustration (seed=0):\n")
    run(budget=60, seed=0)
    print("\n" + "=" * 70)
    print("Multi-seed benchmark (the honest number):")
    print("=" * 70)
    benchmark(n_seeds=8, budget=60)

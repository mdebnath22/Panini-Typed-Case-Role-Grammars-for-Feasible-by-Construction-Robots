"""
Budget-matched analogues of recent LLM/co-design morphology-search papers
==========================================================================

The papers below all train a real RL policy (SAC, ~5e5-1e6 steps) per
candidate design on Gym/Brax Ant-family tasks. Our pipeline uses a fixed
CPG gait as a cheap fitness proxy on grammar-synthesized legged bodies,
under a 40-60 physics-evaluation budget. Those are different tasks,
different simulators, and a ~10,000x different cost per candidate --
reporting their published numbers next to ours in one table would be
apples-to-oranges.

So this file does NOT reproduce their numbers. It reimplements each
paper's CORE SEARCH MECHANISM inside *our* pipeline -- same Genome, same
CPG fitness proxy, same physics-evaluation budget as every other method in
expanded_baseline_table.py -- so the comparison is fair (same task, same
budget) even though it is "inspired by X", not "X reproduced".

Sources (read their actual abstracts/methods before citing numbers from
them -- this file only borrows the mechanism, not their results):
  - RoboMorph        (Qiu et al. 2024, arXiv:2407.08626) -- LLM proposes
    designs in a grammar, conditioned on the archive; RL-trained; compared
    to RoboGrammar's graph search.
  - LASeR            (Song et al., ICLR 2025) -- LLM-driven diversified
    design search.
  - RoboMoRe         (arXiv:2506.00276) -- LLM coarse-to-fine joint
    morphology+reward search: stage 1 broad grid + keep top-k%, stage 2
    iterative local refinement; baselines include BO and Eureka.
  - Debate2Create    (arXiv:2510.25850) -- multi-agent LLM
    thesis-antithesis-synthesis loop; a round-based hall-of-fame archive
    with criterion-specific judges (speed/stability/efficiency/novelty)
    steering each round.
  - Strgar & Kriegman (arXiv:2502.10862, 2025) -- "accelerated co-design
    via morphological pretraining": one controller evaluates the evolving
    population zero-shot; an ablation fine-tunes the controller on the
    current population instead of retraining per-candidate from scratch.

Five new methods, each a numeric/LLM analogue matched to our existing
Genome + CPG-fitness + physics-eval budget accounting:

  1. llm_proposer_search          (RoboMorph / LASeR analogue)
  2. coarse_to_fine_search        (RoboMoRe analogue)
  3. archive_criterion_rotation_search   (Debate2Create analogue)
  4. diversity_archive_es         (RoboMorph "Diversity Reflection" /
                                    DiRect reflection-operator analogue --
                                    explicit genome-parameter-distance
                                    archive, distinct from the existing
                                    21-method table's behavior-distance
                                    Novelty Search)
  5. finetuned_controller_search  (Strgar & Kriegman analogue -- same
                                    evolutionary search, but each genome's
                                    CPG controller (freq, amp, phase) gets
                                    a small per-candidate local search
                                    instead of staying frozen; this also
                                    exercises gmm/task_fitness.py's new
                                    freq/amp/phase_offset args, which is
                                    what reviewerFGH's part H needed)

Every one of these counts a physics evaluation (a call to evaluate_genome
/ evaluate_fitness) against the SAME shared budget as the other 21
methods -- an LLM call, a grid point, or a controller-tuning trial is free
bookkeeping; only a physics rollout costs budget. This mirrors the
accounting fix already applied to reviewerFGH/expanded_baseline_table.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.m_search_loop import (
    Genome, random_genome, mutate, crossover, evaluate_genome,
)
from gmm.task_fitness import evaluate_fitness
from ptg.body_grammar import BodyGrammar
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody


# ---------------------------------------------------------------------------
# Shared budget-accounting helper
# ---------------------------------------------------------------------------

class Budget:
    """Tiny counter so every method below spends the same currency: a
    completed physics evaluation. Mirrors the accounting fix in
    reviewerFGH_cmaes_budget_controller.py / expanded_baseline_table.py --
    only evaluate_genome() calls (or direct evaluate_fitness() calls, for
    the controller analogue) decrement it."""

    def __init__(self, total: int):
        self.total = total
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.total - self.used

    def spend(self, n: int = 1) -> None:
        self.used += n


# Genome.clamp()'s actual bounds (ptg/.. via experiments/m_search_loop.py):
# n_legs in [2,8], segments_per_leg in [1,3], seg_len in [0.05,0.30],
# seg_rad in [0.01,0.05], angle_jitter in [0.0,0.3]. Each entry below is
# (min, max) so _genome_distance can do real per-dimension min-max scaling
# to [0,1] -- NOT divide-by-max, which was the original (buggy) version
# here: dividing n_legs by 8 alone leaves its usable range at [0.25, 1.0]
# (span 0.75) while angle_jitter/0.3 spans the full [0, 1] (span 1.0), so
# angle_jitter silently dominated the distance and biased which parameter
# diversity_archive_es actually diversified. Fixed to true min-max below.
_GENOME_BOUNDS = {
    "n_legs": (2, 8),
    "segments_per_leg": (1, 3),
    "seg_len": (0.05, 0.30),
    "seg_rad": (0.01, 0.05),
    "angle_jitter": (0.0, 0.3),
}


def _minmax(value: float, name: str) -> float:
    lo, hi = _GENOME_BOUNDS[name]
    return (value - lo) / (hi - lo)


def _genome_distance(a: Genome, b: Genome) -> float:
    """Min-max-normalized L2 distance over the 5 genome parameters, each
    scaled to [0,1] using Genome.clamp()'s actual bounds (_GENOME_BOUNDS),
    so every dimension contributes on an equal footing regardless of its
    own range or floor. Used by diversity_archive_es's archive-rejection
    test."""
    va = np.array([_minmax(a.n_legs, "n_legs"),
                   _minmax(a.segments_per_leg, "segments_per_leg"),
                   _minmax(a.seg_len, "seg_len"),
                   _minmax(a.seg_rad, "seg_rad"),
                   _minmax(a.angle_jitter, "angle_jitter")])
    vb = np.array([_minmax(b.n_legs, "n_legs"),
                   _minmax(b.segments_per_leg, "segments_per_leg"),
                   _minmax(b.seg_len, "seg_len"),
                   _minmax(b.seg_rad, "seg_rad"),
                   _minmax(b.angle_jitter, "angle_jitter")])
    return float(np.linalg.norm(va - vb))


# ---------------------------------------------------------------------------
# 1. LLM-as-mutation-operator (RoboMorph / LASeR analogue)
# ---------------------------------------------------------------------------

def _llm_propose(archive: List[Tuple[float, Genome]], rng: np.random.Generator) -> Genome:
    """Ask an LLM to propose a new genome conditioned on the current
    archive (RoboMorph's and LASeR's mechanism). Falls back to a
    structured heuristic proposer -- perturb the best entry, biased away
    from the archive's mean, which is the non-LLM analogue of "propose
    something plausible but different" -- if no API key is configured, so
    this is runnable offline for smoke-testing.

    Set ANTHROPIC_API_KEY to use a real LLM call.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return _llm_propose_anthropic(archive, api_key)
        except Exception:
            pass  # fall through to heuristic on any API/parsing failure
    best = max(archive, key=lambda x: x[0])[1]
    g = copy.copy(best)
    archive_mean = np.mean([[g2.n_legs, g2.segments_per_leg, g2.seg_len,
                             g2.seg_rad, g2.angle_jitter]
                            for _, g2 in archive], axis=0)
    g.n_legs = int(np.clip(round(2 * g.n_legs - archive_mean[0]
                                  + rng.integers(-1, 2)), 2, 8))
    g.segments_per_leg = int(np.clip(round(2 * g.segments_per_leg - archive_mean[1]
                                           + rng.integers(-1, 2)), 1, 3))
    g.seg_len = float(g.seg_len * rng.uniform(0.8, 1.3))
    g.seg_rad = float(g.seg_rad * rng.uniform(0.8, 1.3))
    g.angle_jitter = float(np.clip(g.angle_jitter + rng.uniform(-0.08, 0.08), 0, 0.3))
    return g.clamp()


def _llm_propose_anthropic(archive: List[Tuple[float, Genome]], api_key: str) -> Genome:
    import urllib.request
    top = sorted(archive, key=lambda x: -x[0])[:5]
    history = [{"fitness": round(f, 4), "n_legs": g.n_legs,
               "segments_per_leg": g.segments_per_leg,
               "seg_len": round(g.seg_len, 4), "seg_rad": round(g.seg_rad, 4),
               "angle_jitter": round(g.angle_jitter, 4)} for f, g in top]
    prompt = (
        "You are designing a legged robot body for forward locomotion. "
        "Here are the top candidates tried so far, as JSON "
        "(n_legs in [2,8], segments_per_leg in [1,3], seg_len in [0.05,0.30], "
        "seg_rad in [0.01,0.05], angle_jitter in [0.0,0.3]):\n"
        f"{json.dumps(history)}\n"
        "Propose ONE new candidate, different from all of these, that you "
        "think will do better. Reply with ONLY a JSON object with keys "
        "n_legs, segments_per_leg, seg_len, seg_rad, angle_jitter."
    )
    body = json.dumps({
        "model": "claude-haiku-4-5",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read())
    text = out["content"][0]["text"]
    params = json.loads(text[text.index("{"):text.rindex("}") + 1])
    return Genome(n_legs=int(params["n_legs"]),
                 segments_per_leg=int(params["segments_per_leg"]),
                 seg_len=float(params["seg_len"]),
                 seg_rad=float(params["seg_rad"]),
                 angle_jitter=float(params["angle_jitter"])).clamp()


def llm_proposer_search(budget: int, seed: int = 0, archive_size: int = 8,
                        use_gate: bool = True, verbose: bool = False) -> Dict:
    rng = np.random.default_rng(seed)
    archive: List[Tuple[float, Genome]] = []
    history: List[float] = []
    b = Budget(budget)
    while b.remaining > 0 and len(archive) < archive_size:
        g = random_genome(rng)
        fit, _ = evaluate_genome(g, use_gate=use_gate)
        b.spend()
        archive.append((fit, g))
        history.append(max(f for f, _ in archive))
    while b.remaining > 0:
        g = _llm_propose(archive, rng)
        fit, _ = evaluate_genome(g, use_gate=use_gate)
        b.spend()
        archive.append((fit, g))
        archive.sort(key=lambda x: -x[0])
        archive = archive[:archive_size]
        history.append(archive[0][0])
    best_fit, best_genome = max(archive, key=lambda x: x[0])
    return {"history": history, "best_fitness": best_fit,
           "best_genome": best_genome, "evals_used": b.used}


# ---------------------------------------------------------------------------
# 2. Coarse-to-fine (RoboMoRe analogue)
# ---------------------------------------------------------------------------

def coarse_to_fine_search(budget: int, seed: int = 0, coarse_frac: float = 0.5,
                          keep_frac: float = 0.2, use_gate: bool = True,
                          verbose: bool = False) -> Dict:
    """Stage 1 (coarse): random-sample a batch, keep the top keep_frac.
    Stage 2 (fine): round-robin local hill-climbing refinement of the kept
    candidates until the budget runs out. Mirrors RoboMoRe's
    grid-then-alternating-refinement structure, with our CPG fitness
    standing in for their per-candidate RL training."""
    rng = np.random.default_rng(seed)
    b = Budget(budget)
    coarse_n = max(1, int(round(budget * coarse_frac)))
    history: List[float] = []

    coarse: List[Tuple[float, Genome]] = []
    for _ in range(coarse_n):
        if b.remaining <= 0:
            break
        g = random_genome(rng)
        fit, _ = evaluate_genome(g, use_gate=use_gate)
        b.spend()
        coarse.append((fit, g))
        history.append(max(f for f, _ in coarse))

    coarse.sort(key=lambda x: -x[0])
    keep_n = max(1, int(round(len(coarse) * keep_frac)))
    kept = [list(x) for x in coarse[:keep_n]]  # mutable [fit, genome] pairs

    idx = 0
    while b.remaining > 0 and kept:
        fit_i, genome_i = kept[idx % len(kept)]
        candidate = mutate(genome_i, rng, rate=0.3)  # smaller local step than full mutate
        cand_fit, _ = evaluate_genome(candidate, use_gate=use_gate)
        b.spend()
        if cand_fit > fit_i:
            kept[idx % len(kept)] = [cand_fit, candidate]
        history.append(max(f for f, _ in kept))
        idx += 1

    best_fit, best_genome = max(kept, key=lambda x: x[0])
    return {"history": history, "best_fitness": best_fit,
           "best_genome": best_genome, "evals_used": b.used,
           "coarse_n": coarse_n, "keep_n": keep_n}


# ---------------------------------------------------------------------------
# 3. Archive + rotating criterion (Debate2Create analogue)
# ---------------------------------------------------------------------------

_CRITERIA = ["speed", "stability", "efficiency", "novelty"]


def _criterion_mutate(genome: Genome, criterion: str, rng: np.random.Generator,
                      archive: List[Genome]) -> Genome:
    g = copy.copy(genome)
    if criterion == "speed":
        g.seg_len *= float(rng.uniform(1.0, 1.3))  # longer limbs, bigger stride
        g.angle_jitter *= float(rng.uniform(0.8, 1.0))
    elif criterion == "stability":
        g.seg_rad *= float(rng.uniform(1.05, 1.3))  # thicker limbs
        g.n_legs = int(np.clip(g.n_legs + rng.choice([0, 1]), 2, 8))
    elif criterion == "efficiency":
        g.seg_len *= float(rng.uniform(0.85, 1.0))
        g.seg_rad *= float(rng.uniform(0.85, 1.0))
    else:  # novelty: explicit push away from the archive mean
        if archive:
            mean = np.mean([[a.n_legs, a.segments_per_leg, a.seg_len,
                             a.seg_rad, a.angle_jitter] for a in archive], axis=0)
            g.n_legs = int(np.clip(round(2 * g.n_legs - mean[0]), 2, 8))
            g.seg_len = float(np.clip(2 * g.seg_len - mean[2], 0.05, 0.30))
        g.angle_jitter = float(np.clip(g.angle_jitter + rng.uniform(-0.15, 0.15), 0, 0.3))
    return g.clamp()


def archive_criterion_rotation_search(budget: int, seed: int = 0,
                                      round_size: int = 4, use_gate: bool = True,
                                      verbose: bool = False) -> Dict:
    """Each round proposes `round_size` candidates, one per rotating
    criterion (speed/stability/efficiency/novelty), all derived from the
    current best. The best of the round joins a hall-of-fame archive;
    the next round's base genome is the archive's current best. Analogue
    of Debate2Create's round-based, criterion-judged archive loop, without
    the LLM debate text."""
    rng = np.random.default_rng(seed)
    b = Budget(budget)
    history: List[float] = []

    g0 = random_genome(rng)
    fit0, _ = evaluate_genome(g0, use_gate=use_gate)
    b.spend()
    hall_of_fame: List[Tuple[float, Genome]] = [(fit0, g0)]
    history.append(fit0)
    archive_genomes = [g0]

    round_idx = 0
    while b.remaining > 0:
        base = max(hall_of_fame, key=lambda x: x[0])[1]
        round_best: Optional[Tuple[float, Genome]] = None
        for i in range(round_size):
            if b.remaining <= 0:
                break
            criterion = _CRITERIA[(round_idx * round_size + i) % len(_CRITERIA)]
            cand = _criterion_mutate(base, criterion, rng, archive_genomes)
            fit, _ = evaluate_genome(cand, use_gate=use_gate)
            b.spend()
            archive_genomes.append(cand)
            if round_best is None or fit > round_best[0]:
                round_best = (fit, cand)
            history.append(max(history[-1] if history else fit, fit))
        if round_best is not None:
            hall_of_fame.append(round_best)
        round_idx += 1

    best_fit, best_genome = max(hall_of_fame, key=lambda x: x[0])
    return {"history": history, "best_fitness": best_fit,
           "best_genome": best_genome, "evals_used": b.used,
           "n_rounds": round_idx}


# ---------------------------------------------------------------------------
# 4. Diversity-archive evolutionary search (RoboMorph/DiRect analogue)
# ---------------------------------------------------------------------------

def diversity_archive_es(budget: int, seed: int = 0, pop_size: int = 8,
                         min_archive_dist: float = 0.15, max_resample: int = 3,
                         use_gate: bool = True, verbose: bool = False) -> Dict:
    """Same (mu+lambda) structure as the paper's evolutionary_search(), but
    a proposed mutation is resampled (up to max_resample times) if it is
    closer than min_archive_dist (genome-parameter L2, see
    _genome_distance) to ANY previously-evaluated genome. This is an
    explicit, literal 'reflect on the full history and stay different'
    operator -- closer to RoboMorph's Diversity Reflection prompt and
    DiRect's reflection mechanism than the existing 21-method table's
    Novelty Search, which rewards behavioral (fitness-space) distance
    rather than resampling in parameter space."""
    rng = np.random.default_rng(seed)
    b = Budget(budget)
    population: List[Tuple[float, Genome]] = []
    all_genomes: List[Genome] = []
    history: List[float] = []

    while len(population) < pop_size and b.remaining > 0:
        g = random_genome(rng)
        fit, _ = evaluate_genome(g, use_gate=use_gate)
        b.spend()
        population.append((fit, g))
        all_genomes.append(g)
        history.append(max(f for f, _ in population))

    while b.remaining > 0:
        population.sort(key=lambda x: -x[0])
        survivors = [g for _, g in population[:max(2, pop_size // 2)]]
        parent = survivors[rng.integers(len(survivors))]
        child = mutate(parent, rng)
        tries = 0
        while (tries < max_resample and all_genomes and
               min(_genome_distance(child, h) for h in all_genomes) < min_archive_dist):
            child = mutate(parent, rng)
            tries += 1
        fit, _ = evaluate_genome(child, use_gate=use_gate)
        b.spend()
        population.append((fit, child))
        all_genomes.append(child)
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
        history.append(population[0][0])

    best_fit, best_genome = max(population, key=lambda x: x[0])
    return {"history": history, "best_fitness": best_fit,
           "best_genome": best_genome, "evals_used": b.used}


# ---------------------------------------------------------------------------
# 5. Frozen vs. fine-tuned controller (Strgar & Kriegman analogue)
# ---------------------------------------------------------------------------

def _evaluate_genome_tuned_controller(genome: Genome, b: Budget, use_gate: bool,
                                      tuning_trials: int, rng: np.random.Generator,
                                      n_gait_seeds: int = 3) -> Tuple[float, Dict]:
    """Re-derive and compile the body once (free), then spend
    `tuning_trials` controller candidates searching the CPG's (freq, amp,
    phase_offset) for THIS candidate before reporting its best score --
    the per-candidate controller fine-tuning step in Strgar & Kriegman,
    cheapened from gradient-based differentiable-sim fine-tuning to a
    small random/local search, since we don't have a differentiable
    simulator here.

    Each (freq, amp, phase_offset) candidate is itself scored by averaging
    over n_gait_seeds rollouts (matching evaluate_genome's own noise
    averaging elsewhere in this file and in m_search_loop.py) -- an
    earlier version of this function scored each candidate on a single
    fixed seed=0 rollout, which was noisier than every other method here
    and NOT apples-to-apples; that was a bug, not a design choice, and is
    fixed here. Budget is spent per rollout, so a controller candidate now
    costs n_gait_seeds evaluations, not 1."""
    from sulba.feasibility import FeasibilitySpec, check_feasibility
    spec = FeasibilitySpec(required_reach=0.15, mass_budget=6.0, min_support_area=0.01)
    g = BodyGrammar(seed_body_rules(), seed=0)
    try:
        deriv = g.derive(genome.to_plan())
    except Exception as e:
        return -1e3, {"error": f"grammar: {e}"}
    tree = PhiBody().build_tree(deriv)
    if use_gate:
        rep = check_feasibility(tree, spec)
        if not rep.ok:
            return -1e3, {"error": "infeasible"}

    best = -1e9
    for _ in range(max(1, tuning_trials)):
        if b.remaining <= 0:
            break
        freq = float(rng.uniform(0.8, 2.5))
        amp = float(rng.uniform(0.4, 1.0))
        phase_offset = float(rng.uniform(0, 2 * np.pi))
        fits = []
        for gait_seed in range(n_gait_seeds):
            if b.remaining <= 0:
                break
            r = evaluate_fitness(tree, horizon_steps=500, seed=gait_seed,
                                 freq=freq, amp=amp, phase_offset=phase_offset)
            b.spend()
            if r.ok:
                fits.append(r.fitness)
        if fits and np.mean(fits) > best:
            best = float(np.mean(fits))
    if best == -1e9:
        return -1e3, {"error": "no successful controller trial"}
    return best, {}


def finetuned_controller_search(budget: int, seed: int = 0, pop_size: int = 8,
                                tuning_trials: int = 3, use_gate: bool = True,
                                verbose: bool = False) -> Dict:
    """Identical (mu+lambda) search to evolutionary_search(), but every
    genome's fitness comes from _evaluate_genome_tuned_controller instead
    of the frozen default-CPG evaluate_genome -- each candidate "spends"
    tuning_trials physics evals finding its own best gait, same budget
    currency as every other method here. Compare this against
    m_search_loop.evolutionary_search() (the frozen-controller baseline
    every other result in the paper already reports) to get the
    zero-shot-vs-fine-tuned-controller analogue of Strgar & Kriegman's
    ablation."""
    rng = np.random.default_rng(seed)
    b = Budget(budget)
    population: List[Tuple[float, Genome]] = []
    history: List[float] = []

    while len(population) < pop_size and b.remaining > 0:
        g = random_genome(rng)
        fit, _ = _evaluate_genome_tuned_controller(g, b, use_gate, tuning_trials, rng)
        population.append((fit, g))
        history.append(max(f for f, _ in population))

    while b.remaining > 0:
        population.sort(key=lambda x: -x[0])
        survivors = [g for _, g in population[:max(2, pop_size // 2)]]
        parent = survivors[rng.integers(len(survivors))]
        child = mutate(parent, rng)
        fit, _ = _evaluate_genome_tuned_controller(child, b, use_gate, tuning_trials, rng)
        population.append((fit, child))
        population.sort(key=lambda x: -x[0])
        population = population[:pop_size]
        history.append(population[0][0])

    best_fit, best_genome = max(population, key=lambda x: x[0])
    return {"history": history, "best_fitness": best_fit,
           "best_genome": best_genome, "evals_used": b.used}


# ---------------------------------------------------------------------------
# Benchmark harness (same style as m_search_loop.benchmark / expanded_baseline_table)
# ---------------------------------------------------------------------------

NEW_METHODS = {
    "llm_proposer": llm_proposer_search,
    "coarse_to_fine": coarse_to_fine_search,
    "archive_criterion_rotation": archive_criterion_rotation_search,
    "diversity_archive_es": diversity_archive_es,
    "finetuned_controller": finetuned_controller_search,
}


def benchmark(n_seeds: int = 8, budget: int = 60, verbose: bool = True) -> Dict:
    """Runs every method in NEW_METHODS plus the existing grammar-guided
    evolutionary_search baseline (frozen controller) across n_seeds, same
    budget, same CPG task. Reports mean +/- std and a paired t-test of
    each new method against our evolutionary_search."""
    from experiments.m_search_loop import evolutionary_search
    from scipy import stats

    results: Dict[str, List[float]] = {name: [] for name in NEW_METHODS}
    results["ours_evolutionary"] = []

    for s in range(n_seeds):
        results["ours_evolutionary"].append(
            evolutionary_search(budget, seed=s, verbose=False)["best_fitness"])
        for name, fn in NEW_METHODS.items():
            results[name].append(fn(budget, seed=s, verbose=False)["best_fitness"])
        if verbose:
            print(f"  seed {s} done")

    ours = np.array(results["ours_evolutionary"])
    summary = {}
    for name in NEW_METHODS:
        arr = np.array(results[name])
        wins = int((ours > arr).sum())
        try:
            t_stat, p_val = stats.ttest_rel(ours, arr)
        except Exception:
            t_stat, p_val = float("nan"), float("nan")
        summary[name] = {"mean": float(arr.mean()), "std": float(arr.std()),
                         "ours_wins": wins, "n_seeds": n_seeds,
                         "t_stat": float(t_stat), "p_val": float(p_val)}
        if verbose:
            print(f"{name:28s} {arr.mean():+.3f} +/- {arr.std():.3f}   "
                 f"ours wins {wins}/{n_seeds}   p={p_val:.3f}")

    if verbose:
        print(f"{'ours_evolutionary':28s} {ours.mean():+.3f} +/- {ours.std():.3f}  (reference)")
    return {"raw": results, "summary": summary}


if __name__ == "__main__":
    print("Budget-matched analogues of recent LLM/co-design morphology papers")
    print("(RoboMorph/LASeR/RoboMoRe/Debate2Create/Strgar&Kriegman -- see")
    print(" module docstring for exactly what is and isn't reproduced)\n")
    benchmark(n_seeds=8, budget=60)

"""
Constraint-aware production selection (lit-review item #14)
=============================================================================

The existing feasibility mechanism in this project (sulba.feasibility.
check_feasibility, wired into evaluate_genome via `use_gate`) is a PURE
GEOMETRY check -- reach, ground clearance, mass budget, support-polygon
area, minimum inter-terminal (tip) separation -- computed from the
derived BodyGrammar/PhiBody tree BEFORE any MuJoCo call. It is genuinely
cheap: ~36ms/call measured here vs. ~200ms/call for a real physics
evaluation (~5.5x cheaper, not free). But as currently used throughout
this project (evaluate_genome, m_search_loop.py, m_search_loop_improved.py,
main_baseline_table.py, expanded_baseline_table.py), it is a POST-HOC GATE:
a genome is fully sampled from the search distribution FIRST, THEN checked,
and a reject just gets a -1e3 fitness slapped on a design that was already
finished -- the search budget's SLOT for that sample is still spent
(every one of these call sites charges exactly 1 unit of its own `budget`
per evaluate_genome call, gate-rejected or not; this is the "ACCOUNTING
DISCIPLINE" documented in expanded_baseline_table.py and main_baseline_table.py).
mh6b_gate_ablation.py/feasibility_rate_scaling.py additionally tested a
PHYSICS-only-budget variant of this same gate (free blind resampling until
something passes) and found it still doesn't help -- and, at a demanding
spec, directionally HURTS best task fitness, because rejecting geometrically
unconventional-but-effective bodies throws away information the locomotion
objective itself doesn't care about.

Lit-review item #14 asks for something mechanically different from BOTH of
those: use the cheap geometric estimate to influence WHICH PRODUCTION
(genome field) gets perturbed and how, before a physics call is ever
spent -- not just accept/reject-and-resample blindly. This file implements
that, as a NEW, ADDITIVE variant (`panini_search_constraint_aware`) plus
two matched baselines sharing the exact same physics-budget accounting
(`panini_search_gate` = blind free-resample gate, `hand_coded_sampler` =
a non-adaptive, domain-informed sampler), so the three "ways to spend a
cheap geometric estimate" the lit-review item asks to compare
(gate / random / hand-coded) are all here under one roof, measured against
plain `panini_search` (unmodified, imported) and three of the "tied
cluster" baselines (ipop_cma_es, cross_entropy_method, pso -- their REAL
implementations, imported and run unmodified).

Nothing in experiments/m_search_loop.py, experiments/m_search_loop_improved.py,
experiments/main_baseline_table.py or experiments/expanded_baseline_table.py
is edited. The only "interference" with those modules is a strictly
temporary, restored-in-a-`finally`, monkeypatch of the module-level name
`evaluate_genome` purely for INSTRUMENTATION (counting how many of a
baseline's own budget-consuming calls were geometrically feasible) --
it never changes what evaluate_genome computes or returns, only observes it.

BUDGET-ACCOUNTING DECISION (read this before interpreting any number below):
  Two different, INCOMPATIBLE conventions coexist in this project:
    (a) "slot" accounting (plain panini_search, ipop_cma_es,
        cross_entropy_method, pso, as already implemented): budget =
        number of evaluate_genome() CALLS, whether or not the internal
        gate rejects the sample for free. A gate-rejected sample still
        burns a slot of search budget even though it burns ~0 physics time.
    (b) "physics" accounting (mh6b_gate_ablation.py's own fix, and this
        file's three new methods): budget = number of samples that
        actually reach a MuJoCo rollout. Cheap geometric checks and any
        number of resamples/repairs triggered by them are free and do NOT
        count against budget.
  This file uses (b) -- PHYSICS EVALUATIONS ONLY -- for its own three new
  methods, for the same reason mh6b_gate_ablation.py adopted it: the
  question lit-review item #14 asks ("at EQUAL PHYSICS CALLS, does
  pre-filtering help?") is only answerable under (b); under (a) a
  gate/constraint-aware method would be unfairly throttled to fewer real
  physics tries than a no-filter method at the same nominal `budget`,
  which would hide exactly the effect being tested. Plain panini_search
  and the three imported tied-cluster baselines keep THEIR OWN native (a)
  accounting unmodified (touching their internals is out of scope here) --
  so for every seed we run plain panini_search FIRST under (a) with
  budget=60, measure how many of those 60 slots actually reached physics
  (via instrumentation, not a code change), and then give the three new
  (b)-accounted methods EXACTLY THAT MANY physics calls for that seed --
  i.e. per-seed-matched real physics budget, not just matched nominal
  budget. ipop_cma_es/cross_entropy_method/pso are run at the same nominal
  budget=60 under their own native (a) accounting for context (their
  realized physics-call counts are reported alongside, since their own
  files don't expose a way to pin an exact physics-call target without
  editing them, which is out of scope).

Run:
    python3 -m experiments.panini_constraint_aware --mode smoke
    python3 -m experiments.panini_constraint_aware --mode full --n_seeds 24 --budget 60
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from experiments.m_search_loop import Genome, random_genome, evaluate_genome
from experiments.m_search_loop_improved import (
    _ALL_LO, _ALL_HI, _STD_FLOOR, _full_to_genome, _genome_to_full,
    panini_search,
)
from ptg.body_grammar import BodyGrammar
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody
from gmm.task_fitness import evaluate_fitness
from sulba.feasibility import FeasibilitySpec, check_feasibility

# Same "easy" default used throughout evaluate_genome (~84% of random
# genomes pass it -- measured below, see measure_random_feasibility_rate);
# DEMANDING_SPEC mirrors mh6b_gate_ablation.py's own calibrated spec
# (~6-7% pass rate), reused here (not copy-pasted logic, just the same
# dataclass values) so this file's "is there headroom for pre-filtering to
# matter" question is tested at both ends.
DEFAULT_SPEC = FeasibilitySpec(required_reach=0.15, mass_budget=6.0,
                               min_support_area=0.01)
DEMANDING_SPEC = FeasibilitySpec(required_reach=0.25, mass_budget=3.4,
                                 min_support_area=0.06, min_tip_separation=0.05)


# ---------------------------------------------------------------------------
# The cheap geometric pre-check itself (same check_feasibility used by the
# existing gate, just invoked BEFORE a sample is finalized rather than
# after -- not a new check, a new place to call it)
# ---------------------------------------------------------------------------

def cheap_check(genome: Genome, spec: FeasibilitySpec,
                grammar_seed: int = 0) -> Tuple[bool, Dict[str, bool], Dict[str, float]]:
    """Pure geometry, no MuJoCo. Returns (ok, per-check pass/fail, margins)."""
    g = BodyGrammar(seed_body_rules(), seed=grammar_seed)
    try:
        deriv = g.derive(genome.to_plan())
    except Exception:
        return False, {"grammar": False}, {"grammar": -1.0}
    tree = PhiBody().build_tree(deriv)
    rep = check_feasibility(tree, spec)
    return rep.ok, rep.checks, rep.margins


def _repair(genome: Genome, checks: Dict[str, bool], rng: np.random.Generator,
           step: float = 0.18) -> Genome:
    """THE constraint-aware step: map a violated geometric check to the
    genome field(s) that actually drive it, and nudge those fields toward
    satisfying it. This is what makes production SELECTION constraint-
    aware -- a blind gate would instead throw this sample away and draw a
    fresh independent one, learning nothing from WHY it failed."""
    g = copy.copy(genome)
    if not checks.get("reach", True):
        # reach = max tip distance from torso -- driven by limb length
        g.seg_len = float(np.clip(g.seg_len * (1 + step), 0.05, 0.30))
        if rng.random() < 0.35:
            g.segments_per_leg = int(np.clip(g.segments_per_leg + 1, 1, 3))
    if not checks.get("support", True):
        # support-polygon area -- more legs spread wider, more jitter
        # spreads them further off the symmetric layout, longer legs widen
        # the polygon too
        if rng.random() < 0.6:
            g.n_legs = int(np.clip(g.n_legs + 2, 2, 8))
        g.angle_jitter = float(np.clip(g.angle_jitter * (1 + step) + 0.02, 0.0, 0.3))
        g.seg_len = float(np.clip(g.seg_len * (1 + 0.5 * step), 0.05, 0.30))
    if not checks.get("mass_budget", True):
        # total mass -- thinner limbs, fewer segments, fewer legs
        g.seg_rad = float(np.clip(g.seg_rad * (1 - step), 0.01, 0.05))
        if rng.random() < 0.4:
            g.segments_per_leg = int(np.clip(g.segments_per_leg - 1, 1, 3))
        if rng.random() < 0.25:
            g.n_legs = int(np.clip(g.n_legs - 1, 2, 8))
    if not checks.get("self_clear", True):
        # tips too close together -- spread angularly, lengthen limbs
        g.angle_jitter = float(np.clip(g.angle_jitter * (1 + step) + 0.02, 0.0, 0.3))
        g.seg_len = float(np.clip(g.seg_len * (1 + 0.5 * step), 0.05, 0.30))
    if not checks.get("clearance", True):
        # torso clearance isn't under genome control at all in this grammar
        # (no genome field maps to torso mount height) -- nothing to repair
        pass
    return g.clamp()


def hand_coded_genome(rng: np.random.Generator) -> Genome:
    """A non-adaptive, domain-informed sampler: draws from a sub-region of
    genome space a human would pick AFTER looking at check_feasibility's
    checks once (many legs -> support area; mid-length limbs -> reach
    without excess mass; moderate jitter -> tip separation) -- competent,
    but fixed, never updated by search feedback. The lit-review item's
    third comparison point alongside "a gate" and "random sampling"."""
    return Genome(
        n_legs=int(rng.integers(5, 9)),
        segments_per_leg=int(rng.integers(1, 3)),
        seg_len=float(rng.uniform(0.12, 0.22)),
        seg_rad=float(rng.uniform(0.015, 0.025)),
        angle_jitter=float(rng.uniform(0.05, 0.15)),
    ).clamp()


# ---------------------------------------------------------------------------
# Generic "propose, optionally filter/repair using the cheap check" step,
# shared by all three new methods so the SEARCH MECHANICS (CEM) are held
# fixed and only the SELECTION mechanism varies between conditions.
# ---------------------------------------------------------------------------

def _propose(propose_fn: Callable[[], Genome], mode: str,
            rng: np.random.Generator, spec: FeasibilitySpec,
            max_cycles: int = 24) -> Tuple[Genome, int, bool]:
    """Returns (genome to spend a physics call on, cheap-check calls used,
    whether it is geometrically feasible). `mode`:
      - "none"            : no filtering at all (plain blind sampling);
                             cheap check is still RUN once, for bookkeeping
                             only (it does not change what is returned).
      - "gate"             : blind reject-and-resample (the lit-review
                             item's "a gate" comparison point) -- draws a
                             fresh independent sample from propose_fn each
                             try, no use of WHY a sample failed.
      - "constraint_aware" : repair toward the violated check(s) first
                             (uses WHY it failed to pick which production/
                             field to perturb); falls back to blind
                             resampling only if repair alone doesn't
                             converge within max_cycles.
    """
    cheap_calls = 0
    g = propose_fn()
    ok, checks, _ = cheap_check(g, spec)
    cheap_calls += 1
    if mode == "none" or ok:
        return g, cheap_calls, ok
    if mode == "gate":
        for _ in range(max_cycles):
            g = propose_fn()
            ok, checks, _ = cheap_check(g, spec)
            cheap_calls += 1
            if ok:
                return g, cheap_calls, True
        return g, cheap_calls, False
    if mode == "constraint_aware":
        for _ in range(max_cycles):
            g = _repair(g, checks, rng)
            ok, checks, _ = cheap_check(g, spec)
            cheap_calls += 1
            if ok:
                return g, cheap_calls, True
        # repair plateaued (e.g. conflicting checks) -- fall back to a few
        # blind resamples rather than loop forever on a stuck repair
        for _ in range(max_cycles):
            g = propose_fn()
            ok, checks, _ = cheap_check(g, spec)
            cheap_calls += 1
            if ok:
                return g, cheap_calls, True
        return g, cheap_calls, False
    raise ValueError(mode)


def _cem_with_mode(budget_physics: int, seed: int, mode: str,
                   spec: FeasibilitySpec, pop_size: int = 10,
                   elite_frac: float = 0.3, elitism: int = 2,
                   refine_frac: float = 0.1, max_cycles: int = 24,
                   sampler: Optional[Callable[[np.random.Generator], Genome]] = None
                   ) -> Dict:
    """Same CEM mechanics as panini_search (full 5-field relaxed vector,
    elitism carryover, final local-refinement phase, same _STD_FLOOR fix),
    reused here as the shared search skeleton so `mode` is the ONLY thing
    that differs between the gate/constraint_aware/plain-CEM conditions.
    `budget_physics` counts ONLY calls that reach evaluate_fitness (see
    module docstring's accounting decision). If `sampler` is given, it
    replaces the CEM proposal entirely (used for hand_coded_sampler, which
    is non-adaptive and ignores the CEM mean/std)."""
    rng = np.random.default_rng(seed)
    physics_used = 0
    cheap_calls_total = 0
    feasible_evals = 0
    history: List[float] = []

    def eval_physics(g: Genome) -> float:
        nonlocal physics_used
        fit, _ = evaluate_genome(g, use_gate=False, spec=spec)
        physics_used += 1
        return fit

    refine_budget = max(1, int(round(budget_physics * refine_frac)))
    main_budget = budget_physics - refine_budget

    mean = (_ALL_LO + _ALL_HI) / 2
    std = (_ALL_HI - _ALL_LO) / 3
    n_elite = max(1, int(round(pop_size * elite_frac)))
    best_fit, best_v = -1e9, mean.copy()
    carry: List[Tuple[float, np.ndarray]] = []

    while physics_used < main_budget:
        n_this = min(pop_size, main_budget - physics_used)
        n_new = max(1, n_this - len(carry))
        batch: List[Tuple[float, np.ndarray]] = []
        for _ in range(n_new):
            if sampler is not None:
                propose_fn = lambda: sampler(rng)
            else:
                _mean, _std = mean, std  # capture current values
                propose_fn = lambda: _full_to_genome(
                    np.clip(_mean + _std * rng.normal(size=5), _ALL_LO, _ALL_HI))
            g, cc, feas = _propose(propose_fn, mode, rng, spec, max_cycles)
            cheap_calls_total += cc
            f = eval_physics(g)
            if feas:
                feasible_evals += 1
            batch.append((f, _genome_to_full(g)))
            if physics_used >= main_budget:
                break
        pool = carry + batch
        pool.sort(key=lambda x: -x[0])
        if pool[0][0] > best_fit:
            best_fit, best_v = pool[0]
        history.append(best_fit)
        if sampler is None:
            elites = pool[:min(n_elite, len(pool))]
            elite_arr = np.array([e[1] for e in elites])
            mean = elite_arr.mean(axis=0)
            std = np.maximum(elite_arr.std(axis=0), _STD_FLOOR)
        carry = pool[:elitism]

    tr = 0.15
    scale = _ALL_HI - _ALL_LO
    fail_streak = succ_streak = 0
    while physics_used < budget_physics:
        if sampler is not None:
            propose_fn = lambda: sampler(rng)
        else:
            _best_v, _tr, _scale = best_v, tr, scale
            propose_fn = lambda: _full_to_genome(
                np.clip(_best_v + _tr * _scale * rng.normal(size=5), _ALL_LO, _ALL_HI))
        g, cc, feas = _propose(propose_fn, mode, rng, spec, max_cycles)
        cheap_calls_total += cc
        f = eval_physics(g)
        if feas:
            feasible_evals += 1
        v = _genome_to_full(g)
        if f > best_fit:
            best_fit, best_v = f, v
            succ_streak += 1; fail_streak = 0
        else:
            fail_streak += 1; succ_streak = 0
        if sampler is None:
            if succ_streak >= 3:
                tr = min(tr * 2, 0.5); succ_streak = 0
            if fail_streak >= 5:
                tr = max(tr / 2, 0.02); fail_streak = 0
        history.append(best_fit)

    return {
        "history": history, "best_fitness": best_fit,
        "best_genome": _full_to_genome(best_v), "evals_used": physics_used,
        "physics_used": physics_used, "cheap_calls": cheap_calls_total,
        "feasible_evals": feasible_evals,
        "feasibility_rate": feasible_evals / physics_used if physics_used else 0.0,
    }


def panini_search_gate(budget_physics: int, seed: int = 0,
                       spec: FeasibilitySpec = DEFAULT_SPEC, **kw) -> Dict:
    """Blind free-resample gate (mh6b_gate_ablation.py's mechanism, same
    physics-only accounting), re-implemented here as the "generic gate"
    comparison point sharing this file's CEM skeleton."""
    return _cem_with_mode(budget_physics, seed, "gate", spec, **kw)


def panini_search_constraint_aware(budget_physics: int, seed: int = 0,
                                   spec: FeasibilitySpec = DEFAULT_SPEC, **kw) -> Dict:
    """THE new method: cheap geometric check feeds a directional repair of
    the violated production/field BEFORE a physics call is spent, instead
    of blind reject-and-resample."""
    return _cem_with_mode(budget_physics, seed, "constraint_aware", spec, **kw)


def hand_coded_search(budget_physics: int, seed: int = 0,
                      spec: FeasibilitySpec = DEFAULT_SPEC, **kw) -> Dict:
    """Competent hand-coded sampler, still filtered by the cheap check
    (mode="gate") so it too only spends physics on cheap-check-passing
    designs -- isolates "how good is a fixed, non-adaptive, domain-
    informed prior" from "how good is adaptation" (CEM mean/std)."""
    kw.pop("sampler", None)
    return _cem_with_mode(budget_physics, seed, "gate", spec,
                          sampler=hand_coded_genome, **kw)


def panini_search_no_filter(budget_physics: int, seed: int = 0,
                            spec: FeasibilitySpec = DEFAULT_SPEC, **kw) -> Dict:
    """Same CEM skeleton, mode="none": no pre-check influence at all, but
    under the SAME physics-only accounting as the three methods above
    (unlike plain `panini_search`, which uses slot accounting). Useful as
    an apples-to-apples "does filtering/repair help AT ALL, holding the
    accounting convention fixed" control that isn't confounded by the two
    different budget conventions."""
    return _cem_with_mode(budget_physics, seed, "none", spec, **kw)


# ---------------------------------------------------------------------------
# Instrumentation: count feasibility of every budget-consuming call a
# slot-accounted method (plain panini_search, or an imported tied-cluster
# baseline) makes, WITHOUT changing its behavior. Strictly a temporary,
# restored monkeypatch of the module-level name the target module's own
# code resolves `evaluate_genome` through.
# ---------------------------------------------------------------------------

def _instrumented_call(fn: Callable, budget: int, seed: int,
                       patch_modules: List, spec: Optional[FeasibilitySpec] = None
                       ) -> Tuple[Dict, int, int]:
    """Runs fn(budget, seed=seed, ...) (or fn(budget, seed) for the plain
    baseline wrappers) with `evaluate_genome` patched, in every module in
    `patch_modules`, to a counting wrapper; restores the original in a
    `finally` no matter what. Returns (raw_result, total_calls, feasible_calls)."""
    counters = {"total": 0, "feasible": 0}
    originals = {}

    def make_wrapper(orig):
        def _wrapped(genome, **kwargs):
            fit, info = orig(genome, **kwargs)
            counters["total"] += 1
            if info.get("error") != "infeasible":
                counters["feasible"] += 1
            return fit, info
        return _wrapped

    for mod in patch_modules:
        if hasattr(mod, "evaluate_genome"):
            originals[mod] = mod.evaluate_genome
            mod.evaluate_genome = make_wrapper(mod.evaluate_genome)
    try:
        if spec is not None:
            result = fn(budget, seed=seed, spec=spec, verbose=False)
        else:
            result = fn(budget, seed=seed, verbose=False)
    finally:
        for mod, orig in originals.items():
            mod.evaluate_genome = orig
    if isinstance(result, dict):
        fitness = result["best_fitness"]
    else:
        fitness = result
    return fitness, counters["total"], counters["feasible"]


def _instrumented_baseline_call(fn: Callable, budget: int, seed: int,
                                patch_modules: List) -> Tuple[float, int, int]:
    counters = {"total": 0, "feasible": 0}
    originals = {}

    def make_wrapper(orig):
        def _wrapped(genome, **kwargs):
            fit, info = orig(genome, **kwargs)
            counters["total"] += 1
            if info.get("error") != "infeasible":
                counters["feasible"] += 1
            return fit, info
        return _wrapped

    for mod in patch_modules:
        if hasattr(mod, "evaluate_genome"):
            originals[mod] = mod.evaluate_genome
            mod.evaluate_genome = make_wrapper(mod.evaluate_genome)
    try:
        fitness = fn(budget, seed)
    finally:
        for mod, orig in originals.items():
            mod.evaluate_genome = orig
    return float(fitness), counters["total"], counters["feasible"]


# ---------------------------------------------------------------------------
# Ground-truth FP/FN of the cheap geometric check against MuJoCo-validated
# physical feasibility. "Ground truth" here = does a real rollout (1 gait
# seed is enough: we're testing whether physics RUNS cleanly -- no NaN/
# blowup, actuators present -- not whether the gait is any good) actually
# succeed, independent of the geometric check's own criteria (which never
# touch dynamics/contacts at all).
# ---------------------------------------------------------------------------

def measure_cheap_check_accuracy(n: int = 200, seed: int = 0,
                                 spec: FeasibilitySpec = DEFAULT_SPEC) -> Dict:
    rng = np.random.default_rng(seed)
    tp = fp = tn = fn_ = 0
    for _ in range(n):
        g = random_genome(rng)
        pred_ok, _, _ = cheap_check(g, spec)
        _, info = evaluate_genome(g, use_gate=False, spec=spec, n_gait_seeds=1)
        # evaluate_genome (called with use_gate=False, so "infeasible" can
        # never appear here) returns info WITHOUT an "error" key exactly
        # when >=1 gait seed produced r.ok=True -- i.e. physics ran
        # cleanly (no NaN/blowup, actuators present). The only possible
        # "error" values at this call site are "grammar: ..." (derivation
        # failed) or "all rollouts failed" (the single gait seed's physics
        # call raised or blew up) -- both mean ground-truth PHYSICALLY
        # INFEASIBLE, independent of locomotion quality.
        truth_ok = "error" not in info
        if pred_ok and truth_ok:
            tp += 1
        elif pred_ok and not truth_ok:
            fp += 1
        elif not pred_ok and truth_ok:
            fn_ += 1
        else:
            tn += 1
    total = tp + fp + tn + fn_
    pred_pos = tp + fp
    pred_neg = tn + fn_
    return {
        "n": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn_,
        "ground_truth_feasible_rate": (tp + fn_) / total,
        "cheap_check_pass_rate": pred_pos / total,
        "false_positive_rate": fp / pred_pos if pred_pos else 0.0,  # of cheap-passes, frac physically broken
        "false_negative_rate": fn_ / pred_neg if pred_neg else 0.0,  # of cheap-rejects, frac actually fine
        "accuracy": (tp + tn) / total,
    }


def measure_random_feasibility_rate(n: int = 500, seed: int = 0,
                                    spec: FeasibilitySpec = DEFAULT_SPEC) -> float:
    rng = np.random.default_rng(seed)
    ok_count = 0
    for _ in range(n):
        g = random_genome(rng)
        ok, _, _ = cheap_check(g, spec)
        ok_count += int(ok)
    return ok_count / n


# ---------------------------------------------------------------------------
# n_seeds x budget comparison harness
# ---------------------------------------------------------------------------

def run_comparison(n_seeds: int = 24, budget: int = 60,
                   spec: FeasibilitySpec = DEFAULT_SPEC,
                   include_baselines: bool = True,
                   out_path: Optional[str] = None,
                   verbose: bool = True) -> Dict:
    import experiments.m_search_loop_improved as msli
    import experiments.main_baseline_table as mbt
    import experiments.expanded_baseline_table as ebt

    fitness: Dict[str, List[float]] = {
        "panini_search (slot-acc.)": [],
        "panini_search_gate": [],
        "panini_search_constraint_aware": [],
        "hand_coded_sampler": [],
    }
    feas_rate: Dict[str, List[float]] = {k: [] for k in fitness}
    physics_calls: Dict[str, List[int]] = {k: [] for k in fitness}

    baseline_fns = {}
    if include_baselines:
        baseline_fns = {
            "ipop_cma_es": ebt.run_ipop_cmaes,
            "cross_entropy_method": ebt.run_cem,
            "pso": mbt.run_pso,
        }
        for name in baseline_fns:
            fitness[name] = []
            feas_rate[name] = []
            physics_calls[name] = []

    t0 = time.time()
    for s in range(n_seeds):
        # 1. plain panini_search, native slot accounting, instrumented
        fit_p, total_p, feas_p = _instrumented_call(
            panini_search, budget, s, [msli], spec=spec)
        fitness["panini_search (slot-acc.)"].append(fit_p)
        feas_rate["panini_search (slot-acc.)"].append(feas_p / total_p if total_p else 0.0)
        physics_calls["panini_search (slot-acc.)"].append(feas_p)
        # match the three new methods' physics budget EXACTLY to however
        # many of plain panini_search's `budget` slots actually reached
        # physics this seed (see module docstring's accounting decision).
        # `feas_p` from the instrumentation counts exactly the calls whose
        # info did NOT carry error=="infeasible" -- i.e. exactly the calls
        # that reached evaluate_fitness (a real MuJoCo rollout), whether or
        # not that rollout itself then produced a usable fitness.
        matched_physics_budget = max(1, feas_p)

        for name, fn in [
            ("panini_search_gate", panini_search_gate),
            ("panini_search_constraint_aware", panini_search_constraint_aware),
            ("hand_coded_sampler", hand_coded_search),
        ]:
            res = fn(matched_physics_budget, seed=s, spec=spec)
            fitness[name].append(res["best_fitness"])
            feas_rate[name].append(res["feasibility_rate"])
            physics_calls[name].append(res["physics_used"])

        for name, fn in baseline_fns.items():
            fit_b, total_b, feas_b = _instrumented_baseline_call(
                fn, budget, s, [mbt, ebt])
            fitness[name].append(fit_b)
            feas_rate[name].append(feas_b / total_b if total_b else 0.0)
            physics_calls[name].append(feas_b)

        if verbose:
            print(f"seed {s} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    names = list(fitness.keys())
    fit_arr = np.array([fitness[n] for n in names])
    feas_arr = np.array([feas_rate[n] for n in names])

    from scipy.stats import friedmanchisquare
    chi2_fit, p_fit = friedmanchisquare(*[fit_arr[i] for i in range(len(names))])
    chi2_feas, p_feas = friedmanchisquare(*[feas_arr[i] for i in range(len(names))])

    nemenyi_fit = nemenyi_feas = None
    try:
        import scikit_posthocs as sp
        import pandas as pd
        nemenyi_fit = sp.posthoc_nemenyi_friedman(pd.DataFrame(fit_arr.T, columns=names)).to_dict()
        nemenyi_feas = sp.posthoc_nemenyi_friedman(pd.DataFrame(feas_arr.T, columns=names)).to_dict()
    except Exception as e:
        if verbose:
            print(f"Nemenyi unavailable: {e}")

    out = {
        "n_seeds": n_seeds, "nominal_budget": budget,
        "spec": spec.__dict__,
        "fitness": fitness, "feasibility_rate": feas_rate,
        "physics_calls_used": physics_calls,
        "friedman_fitness": {"chi2": float(chi2_fit), "p": float(p_fit)},
        "friedman_feasibility": {"chi2": float(chi2_feas), "p": float(p_feas)},
        "nemenyi_fitness": nemenyi_fit, "nemenyi_feasibility": nemenyi_feas,
        "mean_fitness": {n: float(np.mean(fitness[n])) for n in names},
        "mean_feasibility_rate": {n: float(np.mean(feas_rate[n])) for n in names},
        "mean_physics_calls": {n: float(np.mean(physics_calls[n])) for n in names},
    }
    if verbose:
        print(f"\n{'method':<34}{'mean fit':>12}{'mean feas.rate':>16}{'mean physics calls':>20}")
        for n in names:
            print(f"{n:<34}{out['mean_fitness'][n]:>12.4f}"
                 f"{out['mean_feasibility_rate'][n]:>16.3f}"
                 f"{out['mean_physics_calls'][n]:>20.1f}")
        print(f"\nFriedman (fitness):     chi2={chi2_fit:.2f} p={p_fit:.2e}")
        print(f"Friedman (feas. rate):  chi2={chi2_feas:.2f} p={p_feas:.2e}")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full", "accuracy", "rate"], default="smoke")
    ap.add_argument("--n_seeds", type=int, default=24)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--spec", choices=["default", "demanding"], default="default")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no_baselines", action="store_true",
                    help="skip ipop_cma_es/cross_entropy_method/pso -- use "
                         "this for --spec demanding, since those baselines' "
                         "own files hardcode calls to evaluate_genome with "
                         "no spec argument, so they would silently run "
                         "under DEFAULT_SPEC instead of the demanding one, "
                         "which would not be a fair same-spec comparison.")
    args = ap.parse_args()
    spec = DEFAULT_SPEC if args.spec == "default" else DEMANDING_SPEC

    if args.mode == "rate":
        print("random-genome cheap-check pass rate:",
             measure_random_feasibility_rate(spec=spec))
    elif args.mode == "accuracy":
        print(json.dumps(measure_cheap_check_accuracy(n=200, spec=spec), indent=1))
    elif args.mode == "smoke":
        run_comparison(n_seeds=3, budget=20, spec=spec, out_path=args.out)
    else:
        run_comparison(n_seeds=args.n_seeds, budget=args.budget, spec=spec,
                       include_baselines=not args.no_baselines,
                       out_path=args.out)

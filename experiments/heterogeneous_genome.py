"""
Heterogeneous (per-limb) genome + search-space extension
===========================================================

Context / motivation (see the module docstring of m_search_loop.py for the
original, uniform-genome setup): the paper's existing `Genome` (n_legs,
segments_per_leg, seg_len, seg_rad, angle_jitter) forces every limb of a
body to share the SAME segment count/length/radius. Nothing in the grammar
itself requires this -- `Extend`/`Articulate`/`Terminate` are applied
per-limb, and `BodySutra.apply()` lets an explicit per-step `overrides`
dict win over both the rule's own defaults and anuvrtti-inherited values
(see ptg/body_sutra.py:BodySutra.apply, "resolved.update(overrides)"). The
ONLY reason every limb after the first ends up uniform in the existing
`legged_plan()` (ptg/body_grammar.py) is that only leg 0's ADD_LIMB step
passes an explicit seg_len/seg_rad override -- every subsequent leg's
ADD_LIMB step passes no override, so it falls back to whatever leg 0's
override exported into `state.inherited_params` via anuvrtti_export. This
module changes nothing about the grammar's rules or resolution order; it
only builds plans that pass a DIFFERENT explicit override for every leg's
ADD_LIMB (and a different per-leg segment count), which the existing
override-wins-over-inheritance machinery already supports.

This file is STRICTLY ADDITIVE: it does not import-and-edit, monkeypatch,
or modify `experiments/m_search_loop.py`, `experiments/m_search_loop_improved.py`,
`ptg/body_grammar.py`, `ptg/body_sutra.py`, or `gmm/phi_body.py`. It only
imports read-only helpers from them (seed_body_rules, BodyGrammar, PhiBody,
leg_mount_plan, torso_frame, the private-but-reusable `_limb_role_frame`)
and builds new, parallel types/functions around them.

Design choices (see task description for the brief):
  - n_legs stays a single top-level discrete count, domain [2, 8], exactly
    like the existing Genome -- this preserves comparability.
  - Each leg independently gets its own (segments, seg_len, seg_rad):
    segments in [1, 3] (same domain as the existing uniform field),
    seg_len in [0.05, 0.30], seg_rad in [0.01, 0.05] (same ranges as the
    existing uniform fields).
  - angle_jitter is kept GLOBAL (one scalar), not per-limb. Rationale:
    angle_jitter perturbs MOUNT ANGLE (where a limb attaches to the
    torso), not limb body-shape -- it is a layout/placement knob, not a
    structural one, and the literature's "combinatorially large design
    space" framing (RoboGrammar/GLSO) is about structural composition,
    which per-leg (segments, seg_len, seg_rad) already captures. Making
    it per-limb too would add dimensionality without adding a
    qualitatively different kind of heterogeneity, so it is left global
    exactly as in the original Genome.
  - Effective dimensionality: 1 (n_legs) + 3*n_legs (per-leg fields) + 1
    (angle_jitter) = 2 + 3*n_legs, ranging from 8 (n_legs=2) to 26
    (n_legs=8) -- matches the brief's "~8 to ~26" target. The number of
    distinct (segments-per-leg) PATTERNS alone grows as 3**n_legs (9 to
    6561 across the n_legs domain), vs. a fixed 3 possible values in the
    uniform genome (or ~7 once combined with n_legs choices) -- genuinely
    combinatorial, not just higher-dimensional-continuous.
  - Variable dimensionality is handled with a FIXED-size, zero-padded
    vector encoding (dim D = 26 = 1 + 8*3 + 1, one slot per possible leg
    up to MAX_LEGS=8): `encode_padded`/`decode_padded` below. `decode_padded`
    reads n_legs from slot 0 and simply IGNORES the per-leg slots for legs
    beyond that count -- so a search method operating on the full 26-D
    vector transparently handles bodies with fewer legs; the unused slots
    carry no gradient/fitness signal and are free to drift, exactly as
    intended by "padded representation, unused legs' parameters ignored"
    in the task brief.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ptg.body_grammar import BodyGrammar, torso_frame, _limb_role_frame
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody, leg_mount_plan
from gmm.mjcf_compiler import MJCFCompiler
from gmm.task_fitness import evaluate_fitness
from sulba.feasibility import FeasibilitySpec, check_feasibility

MAX_LEGS = 8


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class LegSpec:
    segments: int
    seg_len: float
    seg_rad: float

    def clamp(self) -> "LegSpec":
        self.segments = int(np.clip(round(self.segments), 1, 3))
        self.seg_len = float(np.clip(self.seg_len, 0.05, 0.30))
        self.seg_rad = float(np.clip(self.seg_rad, 0.01, 0.05))
        return self


@dataclass
class HeteroGenome:
    n_legs: int
    legs: List[LegSpec] = field(default_factory=list)
    angle_jitter: float = 0.0

    def clamp(self) -> "HeteroGenome":
        self.n_legs = int(np.clip(round(self.n_legs), 2, 8))
        while len(self.legs) < self.n_legs:
            self.legs.append(LegSpec(2, 0.12, 0.02))
        self.legs = [leg.clamp() for leg in self.legs[: self.n_legs]]
        self.angle_jitter = float(np.clip(self.angle_jitter, 0.0, 0.3))
        return self

    def mount_plan(self) -> Tuple[List[float], List[float]]:
        # Identical jitter-RNG scheme to Genome.mount_plan() in
        # m_search_loop.py (see that file's 2026-10-24 bug-fix note): the
        # jitter DIRECTION depends only on n_legs, so small continuous
        # changes elsewhere never discontinuously reshuffle leg angles.
        base_angles, xs = leg_mount_plan(self.n_legs)
        base_angles = np.array(base_angles)
        rng = np.random.default_rng(int(self.n_legs) * 1_000_003 + 17)
        unit_dir = rng.uniform(-1.0, 1.0, self.n_legs)
        jittered = base_angles + unit_dir * self.angle_jitter
        return list(np.clip(jittered, np.pi, 2 * np.pi)), xs

    def to_plan(self) -> List[Dict]:
        angles, xs = self.mount_plan()
        return hetero_plan(self.legs, mount_angles=angles, mount_x_offsets=xs)

    def summary(self) -> str:
        legs_s = ", ".join(f"(s={l.segments},len={l.seg_len:.3f},rad={l.seg_rad:.3f})"
                           for l in self.legs)
        return f"n_legs={self.n_legs} jitter={self.angle_jitter:.3f} legs=[{legs_s}]"


def hetero_plan(legs: List[LegSpec], mount_angles: List[float],
                mount_x_offsets: List[float]) -> List[Dict]:
    """Like ptg.body_grammar.legged_plan, but gives EVERY leg its own
    explicit ADD_LIMB override (seg_len, seg_rad) and its own segment
    count, instead of overriding only leg 0 and letting anuvrtti copy
    that one value to every other leg. An explicit per-step override
    always wins over anuvrtti-inherited values (BodySutra.apply:
    `resolved.update(overrides)` runs after the inherited-params merge),
    so this is legal under the EXISTING grammar rules -- no rule, no
    resolve order, nothing in ptg/body_sutra.py or ptg/body_grammar.py is
    touched."""
    rf = _limb_role_frame()
    plan: List[Dict] = [{"op": "ADD_TORSO", "frame": torso_frame()}]
    for ang, x, leg in zip(mount_angles, mount_x_offsets, legs):
        plan.append({"op": "ADD_LIMB", "frame": rf,
                     "params": {"mount_angle": float(ang), "mount_x": float(x),
                                "seg_len": leg.seg_len, "seg_rad": leg.seg_rad}})
        for _ in range(1, leg.segments):
            plan.append({"op": "ARTICULATE", "frame": rf})
            plan.append({"op": "EXTEND", "frame": rf})
        plan.append({"op": "TERMINATE", "frame": rf})
    return plan


# ---------------------------------------------------------------------------
# Random construction
# ---------------------------------------------------------------------------

def random_leg(rng: np.random.Generator) -> LegSpec:
    return LegSpec(segments=int(rng.integers(1, 4)),
                  seg_len=float(rng.uniform(0.06, 0.22)),
                  seg_rad=float(rng.uniform(0.015, 0.035))).clamp()


def random_hetero_genome(rng: np.random.Generator) -> HeteroGenome:
    n_legs = int(rng.integers(2, 9))
    legs = [random_leg(rng) for _ in range(n_legs)]
    return HeteroGenome(n_legs=n_legs, legs=legs,
                        angle_jitter=float(rng.uniform(0.0, 0.2))).clamp()


# ---------------------------------------------------------------------------
# Evaluation: grammar -> tree -> (Śulba gate) -> physics fitness
# (byte-for-byte the same pipeline/logic as m_search_loop.evaluate_genome,
#  just retargeted at HeteroGenome.to_plan())
# ---------------------------------------------------------------------------

def evaluate_hetero_genome(genome: HeteroGenome, *, use_gate: bool = True,
                           spec: Optional[FeasibilitySpec] = None,
                           n_gait_seeds: int = 3, horizon_steps: int = 500,
                           grammar_seed: int = 0) -> Tuple[float, Dict]:
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


def hetero_genome_to_tree(genome: HeteroGenome, grammar_seed: int = 0):
    g = BodyGrammar(seed_body_rules(), seed=grammar_seed)
    deriv = g.derive(genome.to_plan())
    return PhiBody().build_tree(deriv)


# ---------------------------------------------------------------------------
# Fixed-size padded vector encoding (handles n_legs-dependent dimensionality)
# ---------------------------------------------------------------------------
# Layout (D = 1 + MAX_LEGS*3 + 1 = 26):
#   v[0]                        n_legs
#   v[1 + 3*i : 1 + 3*i + 3]    (segments, seg_len, seg_rad) for leg i,
#                               i = 0 .. MAX_LEGS-1 (slots for i >= decoded
#                               n_legs are PRESENT but IGNORED by decode)
#   v[-1]                       angle_jitter

D = 1 + MAX_LEGS * 3 + 1

_ALL_LO = np.array([2.0] + [1.0, 0.05, 0.01] * MAX_LEGS + [0.0])
_ALL_HI = np.array([8.0] + [3.0, 0.30, 0.05] * MAX_LEGS + [0.3])

# Discrete-field std floor, same rationale/fix as m_search_loop_improved.py's
# _STD_FLOOR (see that file's long comment): a continuous-Gaussian-then-round
# CEM-family sampler can let a discrete field's resampling std collapse to
# ~0 once the fitted mean settles near one integer, permanently killing
# exploration of the neighboring integers. In the uniform genome this hit
# ONE field (segments_per_leg); here it would hit it independently in EVERY
# one of up to 8 per-leg "segments" slots, which would be a far more severe
# version of the same artifact. The exact same floor (0.6, comparable to
# the field's own domain range of 2.0) is applied here, to every per-leg
# segments slot, and given to every CEM-family method below (panini_search_
# hetero AND its competitors) so the comparison is not confounded by this
# orthogonal implementation bug landing differently on different methods.
_STD_FLOOR = np.array([1e-3] + [0.6, 1e-3, 1e-3] * MAX_LEGS + [1e-3])


def decode_padded(v: np.ndarray) -> HeteroGenome:
    n_legs = int(np.clip(round(v[0]), 2, 8))
    legs = []
    for i in range(n_legs):
        base = 1 + 3 * i
        legs.append(LegSpec(
            segments=int(np.clip(round(v[base]), 1, 3)),
            seg_len=float(np.clip(v[base + 1], 0.05, 0.30)),
            seg_rad=float(np.clip(v[base + 2], 0.01, 0.05)),
        ))
    angle_jitter = float(np.clip(v[-1], 0.0, 0.3))
    return HeteroGenome(n_legs=n_legs, legs=legs, angle_jitter=angle_jitter)


def encode_padded(g: HeteroGenome) -> np.ndarray:
    v = np.zeros(D)
    v[0] = g.n_legs
    for i in range(MAX_LEGS):
        base = 1 + 3 * i
        leg = g.legs[i] if i < len(g.legs) else LegSpec(2, 0.12, 0.02)
        v[base], v[base + 1], v[base + 2] = leg.segments, leg.seg_len, leg.seg_rad
    v[-1] = g.angle_jitter
    return v


def _eval_padded(v: np.ndarray, use_gate: bool = True, spec=None) -> float:
    fit, _ = evaluate_hetero_genome(decode_padded(v), use_gate=use_gate, spec=spec)
    return fit


# ---------------------------------------------------------------------------
# Search methods over the padded vector
# ---------------------------------------------------------------------------

def panini_search_hetero(
    budget: int, seed: int = 0, pop_size: int = 10, elite_frac: float = 0.3,
    elitism: int = 2, refine_frac: float = 0.1,
    use_gate: bool = True, spec=None, verbose: bool = False,
) -> Dict:
    """PĀṇini, generalized to the padded heterogeneous vector. Identical
    mechanism to m_search_loop_improved.panini_search (CEM resampling +
    small elitism carryover + final local-refinement phase), just over the
    D=26-dim padded space instead of the original 5-dim relaxed genome."""
    rng = np.random.default_rng(seed)
    evals_used = 0
    history: List[float] = []

    def eval_v(v):
        nonlocal evals_used
        f = _eval_padded(v, use_gate=use_gate, spec=spec)
        evals_used += 1
        return f

    refine_budget = max(1, int(round(budget * refine_frac)))
    main_budget = budget - refine_budget

    mean = (_ALL_LO + _ALL_HI) / 2
    std = (_ALL_HI - _ALL_LO) / 3
    n_elite = max(1, int(round(pop_size * elite_frac)))

    best_fit, best_v = -1e9, mean.copy()
    carry: List[Tuple[float, np.ndarray]] = []

    while evals_used < main_budget:
        n_this = min(pop_size, main_budget - evals_used)
        n_new = max(1, n_this - len(carry))
        samples = np.clip(mean + std * rng.normal(size=(n_new, D)), _ALL_LO, _ALL_HI)
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
        carry = pool[:elitism]

        if verbose and evals_used % 10 == 0:
            print(f"  [PĀṇini-hetero] evals {evals_used:3d}/{budget}  best={best_fit:+.4f}")

    tr = 0.15
    scale = _ALL_HI - _ALL_LO
    fail_streak = succ_streak = 0
    while evals_used < budget:
        cand = np.clip(best_v + tr * scale * rng.normal(size=D), _ALL_LO, _ALL_HI)
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
           "best_genome": decode_padded(best_v), "evals_used": evals_used}


def cross_entropy_method_hetero(budget: int, seed: int = 0, pop_size: int = 10,
                                elite_frac: float = 0.3, use_gate: bool = True,
                                spec=None, verbose: bool = False) -> Dict:
    """Plain CEM over the padded vector (no elitism carryover, no
    refinement phase) -- the generic analogue of expanded_baseline_table
    .run_cem, one of the 6 methods PĀṇini was found statistically tied
    with in the uniform genome. Gets the same discrete-field std floor as
    panini_search_hetero (see _STD_FLOOR's docstring) so the comparison
    isn't confounded by that orthogonal bug."""
    rng = np.random.default_rng(seed)
    mean = (_ALL_LO + _ALL_HI) / 2
    std = (_ALL_HI - _ALL_LO) / 3
    best_fit, best_v = -1e9, mean.copy()
    evals_used = 0
    n_elite = max(1, int(pop_size * elite_frac))
    history: List[float] = []

    while evals_used < budget:
        n_this = min(pop_size, budget - evals_used)
        samples = np.clip(mean + std * rng.normal(size=(n_this, D)), _ALL_LO, _ALL_HI)
        fits = np.array([_eval_padded(s, use_gate=use_gate, spec=spec) for s in samples])
        evals_used += n_this
        if fits.max() > best_fit:
            best_fit, best_v = float(fits.max()), samples[int(np.argmax(fits))]
        elite_idx = np.argsort(-fits)[:min(n_elite, n_this)]
        elites = samples[elite_idx]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), _STD_FLOOR)
        history.append(best_fit)
        if verbose and evals_used % 10 == 0:
            print(f"  [cem-hetero] evals {evals_used:3d}/{budget}  best={best_fit:+.4f}")

    return {"history": history, "best_fitness": best_fit,
           "best_genome": decode_padded(best_v), "evals_used": evals_used}


def ipop_cma_es_hetero(budget: int, seed: int = 0, use_gate: bool = True,
                       spec=None, verbose: bool = False) -> Dict:
    """Genuine IPOP-CMA-ES (the `cma` package's own restart mechanism),
    generalized to the D=26 padded vector -- same structure as
    main_baseline_table.run_ipop_cmaes, including its `effective_cap`
    fix for cma.fmin's between-generation-only budget check."""
    import cma
    rng_seed = seed
    best_holder = {"fit": -1e9, "v": (_ALL_LO + _ALL_HI) / 2}

    def objective(x):
        f = _eval_padded(np.array(x), use_gate=use_gate, spec=spec)
        if f > best_holder["fit"]:
            best_holder["fit"], best_holder["v"] = f, np.array(x)
        return -f

    x0 = ((_ALL_LO + _ALL_HI) / 2).tolist()
    sigma0 = 1.0
    bounds = [_ALL_LO.tolist(), _ALL_HI.tolist()]
    safety_margin = 10
    effective_cap = max(1, budget - safety_margin)

    res = cma.fmin(objective, x0, sigma0,
                   options={"bounds": bounds, "maxfevals": effective_cap,
                           "seed": rng_seed, "verbose": -9},
                   restarts=3, incpopsize=2, bipop=False)
    best_fit = float(best_holder["fit"])
    best_v = best_holder["v"]
    if verbose:
        print(f"  [ipop-cma-hetero] best={best_fit:+.4f}")
    return {"history": [best_fit], "best_fitness": best_fit,
           "best_genome": decode_padded(best_v), "evals_used": budget}


def pso_hetero(budget: int, seed: int = 0, n_particles: int = 8, w: float = 0.6,
               c1: float = 1.5, c2: float = 1.5, use_gate: bool = True,
               spec=None, verbose: bool = False) -> Dict:
    """PSO over the padded vector -- generic analogue of
    main_baseline_table.run_pso."""
    rng = np.random.default_rng(seed)
    x = _ALL_LO + rng.random((n_particles, D)) * (_ALL_HI - _ALL_LO)
    v = np.zeros_like(x)
    pbest = x.copy()
    pbest_fit = np.full(n_particles, -1e9)
    gbest = x[0].copy()
    gbest_fit = -1e9
    evals_used = 0
    history: List[float] = []

    while evals_used < budget:
        n_this = min(n_particles, budget - evals_used)
        for i in range(n_this):
            fit = _eval_padded(x[i], use_gate=use_gate, spec=spec)
            evals_used += 1
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit; pbest[i] = x[i].copy()
            if fit > gbest_fit:
                gbest_fit = fit; gbest = x[i].copy()
            history.append(gbest_fit)
        r1, r2 = rng.random((n_particles, D)), rng.random((n_particles, D))
        vv = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = vv
        x = np.clip(x + v, _ALL_LO, _ALL_HI)

    return {"history": history, "best_fitness": float(gbest_fit),
           "best_genome": decode_padded(gbest), "evals_used": evals_used}


METHODS_HETERO = {
    "panini_search (hetero)": panini_search_hetero,
    "cross_entropy_method (hetero)": cross_entropy_method_hetero,
    "ipop_cma_es (hetero)": ipop_cma_es_hetero,
    "pso (hetero)": pso_hetero,
}


# ---------------------------------------------------------------------------
# Phase 1 validation
# ---------------------------------------------------------------------------

def validate_mjcf_string(xml: str, settle_steps: int = 50) -> Dict:
    report = {"loaded": False, "stepped": False, "error": None, "n_actuators": None}
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_string(xml)
        report["loaded"] = True
        report["n_actuators"] = int(model.nu)
        data = mujoco.MjData(model)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        if np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)):
            report["stepped"] = True
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
    return report


def leg_lengths_from_tree(tree) -> List[List[float]]:
    """Returns, for each top-level limb child of the torso, the ordered
    list of its own-plus-descendant 'link' segment lengths -- used to
    confirm two legs of a deliberately-heterogeneous genome really do
    differ in geometry after compilation (not just in the Genome object)."""
    out = []
    for limb_root in tree.children:
        lens = [n.geom.get("length") for n in limb_root.walk() if n.kind == "link"]
        out.append(lens)
    return out


def run_phase1_validation(verbose: bool = True) -> Dict:
    rng = np.random.default_rng(0)
    results = []

    # (a) a deliberately, strongly heterogeneous genome: 4 legs, each with
    # a different segment count and very different seg_len/seg_rad.
    deliberate = HeteroGenome(
        n_legs=4,
        legs=[
            LegSpec(segments=1, seg_len=0.08, seg_rad=0.015),
            LegSpec(segments=2, seg_len=0.14, seg_rad=0.02),
            LegSpec(segments=3, seg_len=0.20, seg_rad=0.03),
            LegSpec(segments=1, seg_len=0.28, seg_rad=0.045),
        ],
        angle_jitter=0.1,
    ).clamp()

    cases = [("deliberate_heterogeneous", deliberate)]
    for i in range(4):
        cases.append((f"random_{i}", random_hetero_genome(rng)))

    for name, genome in cases:
        entry = {"name": name, "genome_summary": genome.summary()}
        try:
            tree = hetero_genome_to_tree(genome)
            leg_lens = leg_lengths_from_tree(tree)
            entry["leg_lengths"] = leg_lens
            entry["legs_distinct"] = len({tuple(l) for l in leg_lens}) > 1
            xml = MJCFCompiler().compile(tree, name=f"hetero_{name}")
            mj = validate_mjcf_string(xml)
            entry["mjcf"] = mj
            fit, info = evaluate_hetero_genome(genome, use_gate=False)
            entry["fitness"] = fit
            entry["fitness_info"] = info
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        results.append(entry)
        if verbose:
            ok = results[-1].get("mjcf", {})
            print(f"[{name}] {genome.summary()}")
            print(f"    legs_distinct={entry.get('legs_distinct')}  "
                 f"leg_lengths={entry.get('leg_lengths')}")
            print(f"    mjcf loaded={ok.get('loaded')} stepped={ok.get('stepped')} "
                 f"error={ok.get('error')}")
            print(f"    fitness={entry.get('fitness')}  error={entry.get('error')}")

    n_ok = sum(1 for e in results if e.get("mjcf", {}).get("loaded")
              and e.get("mjcf", {}).get("stepped") and not e.get("error"))
    n_distinct = sum(1 for e in results if e.get("legs_distinct"))
    summary = {"n_cases": len(results), "n_mjcf_ok": n_ok,
              "n_legs_distinct": n_distinct, "results": results}
    if verbose:
        print(f"\nPhase 1 summary: {n_ok}/{len(results)} compiled+loaded+stepped OK, "
             f"{n_distinct}/{len(results)} genuinely heterogeneous across legs.")
    return summary


# ---------------------------------------------------------------------------
# Phase 2 screening / confirmation
# ---------------------------------------------------------------------------

def run_screening(n_seeds: int = 10, budget: int = 80, verbose: bool = True) -> Dict:
    results = {name: [] for name in METHODS_HETERO}
    for s in range(n_seeds):
        for name, fn in METHODS_HETERO.items():
            try:
                res = fn(budget, seed=s, verbose=False)
                fit = res["best_fitness"]
            except Exception as e:
                fit = float("nan")
                if verbose:
                    print(f"  [{name} seed={s}] FAILED: {e}")
            results[name].append(fit)
        if verbose:
            row = "  ".join(f"{n}={results[n][-1]:+.3f}" for n in METHODS_HETERO)
            print(f"seed={s}: {row}")

    return _summarize(results, verbose=verbose, label=f"SCREENING (n_seeds={n_seeds}, budget={budget})")


def run_confirmation(n_seeds: int = 24, budget: int = 80, verbose: bool = True) -> Dict:
    results = {name: [] for name in METHODS_HETERO}
    for s in range(n_seeds):
        for name, fn in METHODS_HETERO.items():
            res = fn(budget, seed=s, verbose=False)
            results[name].append(res["best_fitness"])
        if verbose and (s + 1) % 4 == 0:
            print(f"  ...seed {s + 1}/{n_seeds} done")
    return _summarize(results, verbose=verbose, label=f"CONFIRMATION (n_seeds={n_seeds}, budget={budget})")


def _summarize(results: Dict[str, List[float]], verbose: bool, label: str) -> Dict:
    names = list(results.keys())
    data = np.array([results[n] for n in names])

    from scipy.stats import rankdata, friedmanchisquare
    ranks = np.array([rankdata(-data[:, s]) for s in range(data.shape[1])]).T
    mean_ranks = ranks.mean(axis=1)
    chi2, p_friedman = friedmanchisquare(*[data[i] for i in range(len(names))])

    nemenyi = {}
    try:
        import scikit_posthocs as sp
        import pandas as pd
        wide = pd.DataFrame(data.T, columns=names)
        nem = sp.posthoc_nemenyi_friedman(wide)
        ours = "panini_search (hetero)"
        nemenyi = {other: float(nem.loc[ours, other]) for other in names if other != ours}
    except Exception as e:
        if verbose:
            print(f"Nemenyi unavailable: {e}")

    order = np.argsort(mean_ranks)
    ours_idx = names.index("panini_search (hetero)")
    ours_rank = 1 + int(np.sum(mean_ranks < mean_ranks[ours_idx]))

    if verbose:
        print(f"\n--- {label} ---")
        print(f"{'method':<34}{'mean rank':>12}{'mean fitness':>14}{'std':>10}")
        for idx in order:
            tag = "  <== PĀṇini" if names[idx] == "panini_search (hetero)" else ""
            print(f"{names[idx]:<34}{mean_ranks[idx]:>12.2f}{data[idx].mean():>14.3f}"
                 f"{data[idx].std():>10.3f}{tag}")
        print(f"Friedman chi2={chi2:.3f} p={p_friedman:.4f}")
        print(f"PĀṇini (hetero): rank {ours_rank} of {len(names)}  "
             f"(mean rank {mean_ranks[ours_idx]:.2f}, "
             f"mean fitness {data[ours_idx].mean():.3f})")
        if nemenyi:
            print("Nemenyi p-values, PĀṇini(hetero) vs others:")
            for other, pv in nemenyi.items():
                sig = "SIGNIFICANT" if pv < 0.05 else "not significant"
                print(f"  vs {other:<30} p={pv:.4f}  ({sig})")

    return {"results": results, "mean_ranks": dict(zip(names, mean_ranks.tolist())),
           "chi2": float(chi2), "p_friedman": float(p_friedman), "nemenyi": nemenyi,
           "ours_rank": ours_rank, "n_methods": len(names)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["validate", "screen", "confirm"], default="validate")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=80)
    args = ap.parse_args()
    if args.phase == "validate":
        run_phase1_validation()
    elif args.phase == "screen":
        run_screening(n_seeds=args.n_seeds, budget=args.budget)
    else:
        run_confirmation(n_seeds=args.n_seeds, budget=args.budget)

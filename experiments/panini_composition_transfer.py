"""
Held-out structural-composition transfer: does PA-ṇini's compositional,
per-leg-typed representation let it carry useful structural preferences
from a "seen" region of design space into a disjoint "held-out" region,
when flat-vector baselines cannot?
=================================================================================

Context
-------
Ten prior investigations in this project found PA-ṇini (`panini_search` /
`panini_search_hetero`) statistically TIED with a cluster of ~6-7 competitor
methods on raw final-fitness, within a single fixed design-space region (no
train/test split of the space itself). This file asks a DIFFERENT question,
literature-review item #23 ("held-out-composition generalization"): if a
method is allowed to spend part of its budget exploring a "seen" sub-region
of the design space and must then search a disjoint "held-out" sub-region
with the REST of its budget, can PA-ṇini's compositional structure make that
transfer more sample-efficient than starting the held-out phase cold --
something the flat-vector baselines have no natural mechanism to do?

Substrate: the HETEROGENEOUS per-limb genome
----------------------------------------------
Uses `experiments/heterogeneous_genome.py` (HeteroGenome/LegSpec, the padded
D=26 vector: v[0]=n_legs, v[1+3i:1+3i+3]=(segments,seg_len,seg_rad) for leg
i=0..7, v[-1]=angle_jitter), NOT the original flat 5-field Genome. Rationale
(per a prior agent's finding on a related motif-reuse idea): the base
Genome's structural vocabulary -- one scalar segments_per_leg shared by every
leg -- is too thin to define an interesting notion of "a structural
combination"; there is essentially only one axis (n_legs) to split on, and
no module-level substructure for a compositional method to have an edge on.
The hetero genome instead gives each leg an independent (segments, seg_len,
seg_rad) triple, so "a combination of modules" has real content: a per-leg
STRUCTURAL PATTERN (how many segments each leg has, and how thick/long), and
the grammar's own repeated-production structure (ADD_LIMB;[ARTICULATE;EXTEND]*
;TERMINATE applied once per leg, same production family regardless of which
leg index) is exactly what licenses a LEG-POSITION-INDEPENDENT structural
prior -- a flat per-dimension vector has no such thing (see "Why this is a
fair, honest test of asymmetry" below).

The held-out split
-------------------
Along the ONE discrete axis this genome shares with the task brief's own
example ("new limb counts combined with previously seen segment patterns"):
  SEEN   n_legs in {2, 3, 4, 5}   (v[0] in [2, 5] during the seen phase)
  HELD   n_legs in {6, 7, 8}      (v[0] in [6, 8] during the held-out phase)
Both phases use the SAME per-leg segment/seg_len/seg_rad domains (segments in
{1,2,3}, seg_len in [0.05,0.30], seg_rad in [0.01,0.05]) -- it is specifically
the LEG-COUNT composition that is held out, i.e. "new limb counts combined
with previously seen segment patterns," exactly as item #23 frames it. Note
leg SLOTS 5, 6, 7 of the padded vector (0-indexed) are *structurally unseen*
during the seen phase: decode_padded only reads slots < n_legs, so for every
seen-phase genome (n_legs <= 5) those three slots are pure padding, never
read by evaluation, never selected on by fitness. Any per-DIMENSION (i.e.
per-slot) statistic a flat-vector method's search state holds for slots 5-7
after the seen phase is therefore pure noise -- it reflects nothing about
fitness. This is the concrete, falsifiable form of "the flat encoding has no
module-level preference to transfer": it is not merely under-explored there,
it is UNINFORMED there by construction, whereas a LEG-POSITION-INDEPENDENT
statistic (pooled over all real legs seen, regardless of index) is just as
meaningful for slot 7 as for slot 0, because the grammar treats every leg as
an instance of the same repeated production.

The transfer mechanism
-----------------------
`build_leg_prior`: after a seen-phase CEM run, pool (segments, seg_len,
seg_rad) across EVERY real leg (i < that genome's own n_legs) of every elite
genome retained across all seen-phase generations, IGNORING leg index --
giving one pooled (mean, std) triple per field, plus a pooled (mean, std)
for the global angle_jitter. `panini_transfer_hetero` then warm-starts the
held-out-phase CEM's initial mean/std by broadcasting this SAME pooled prior
to every one of the 8 leg slots (including slots 5-7, never informative
during the seen phase) instead of the generic domain-midpoint/one-third-range
initialization every cold method uses. Nothing about the grammar, the Genome
classes, or any existing file is touched; this file only adds new, bounded
variants of the existing CEM/PSO/IPOP-CMA-ES mechanics (reimplemented here,
not imported, specifically so they can take configurable [lo, hi] bounds and
an optional warm-start mean/std -- the originals in heterogeneous_genome.py
hard-code the full [2,8] n_legs domain at module scope and cannot be
parameterized without editing that file, which is disallowed).

Why this is a fair, honest test of the asymmetry (read before trusting the result)
------------------------------------------------------------------------------------
The comparison is NOT "give PA-ṇini extra information no baseline could ever
have." Every baseline (cem_cold_hetero, ipop_cma_cold_hetero, pso_cold_hetero)
gets the SAME TOTAL evaluation budget as panini_transfer_hetero (budget_seen
+ budget_held), and in fact `panini_cold_hetero` -- plain PA-ṇini's own
mechanism, cold, spending that entire combined budget directly on the
held-out region with no seen phase at all -- is included as the critical
control. If panini_transfer beats panini_cold, that isolates the prior's
effect from the method family. The baselines are run cold, spending their
FULL matched budget on the held-out region, rather than being forced to
"waste" budget on a seen phase they have no way to use -- this is the
*most generous* fair comparison for them, not a rigged one: giving a flat
method a seen phase would only ever cost it evals for free (for slots 5-7)
or duplicate what warm-starting its OWN per-dimension mean from seen-phase
elites would do for slots 0-4 only (tested separately below as
`cem_naive_transfer_hetero`, to quantify -- not merely assert -- that
naive per-dimension warm-starting is useless here, since it carries
informative values for slots 0-4 but nothing for 5-7, which is most of
what distinguishes a held-out n_legs>=6 body).

Metric: best fitness reached on the HELD-OUT region at a matched total
budget (sample efficiency), plus feasibility rate (fraction of seeds whose
final genome passes the SA-ulba feasibility gate, fitness > -999).
"""
from __future__ import annotations

import copy
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.heterogeneous_genome import (
    LegSpec, HeteroGenome, decode_padded, encode_padded,
    evaluate_hetero_genome, hetero_genome_to_tree, validate_mjcf_string,
    D, MAX_LEGS,
)

# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------

_FULL_LO = np.array([2.0] + [1.0, 0.05, 0.01] * MAX_LEGS + [0.0])
_FULL_HI = np.array([8.0] + [3.0, 0.30, 0.05] * MAX_LEGS + [0.3])

SEEN_LO = _FULL_LO.copy(); SEEN_LO[0] = 2.0
SEEN_HI = _FULL_HI.copy(); SEEN_HI[0] = 5.0   # n_legs in {2,3,4,5}

HELD_LO = _FULL_LO.copy(); HELD_LO[0] = 6.0
HELD_HI = _FULL_HI.copy(); HELD_HI[0] = 8.0   # n_legs in {6,7,8}

# same discrete-field std floor rationale/value as heterogeneous_genome.py's
# _STD_FLOOR (segments collapses without it) -- applied identically to every
# method below so the comparison isn't confounded by this orthogonal bug.
_STD_FLOOR = np.array([1e-3] + [0.6, 1e-3, 1e-3] * MAX_LEGS + [1e-3])

FAILED_FIT = -1e3


def _eval_v(v: np.ndarray, use_gate: bool = True, spec=None) -> float:
    fit, _ = evaluate_hetero_genome(decode_padded(v), use_gate=use_gate, spec=spec)
    return fit


# ---------------------------------------------------------------------------
# Generic, bounds-parameterized CEM (covers both "PA-ṇini" style -- elitism
# carryover + final local-refinement -- and "plain CEM" style -- neither --
# depending on elitism/refine_frac, and optionally a warm-started mean/std)
# ---------------------------------------------------------------------------

def _cem(budget: int, seed: int, lo: np.ndarray, hi: np.ndarray,
         pop_size: int = 10, elite_frac: float = 0.3, elitism: int = 0,
         refine_frac: float = 0.0, mean0: Optional[np.ndarray] = None,
         std0: Optional[np.ndarray] = None, use_gate: bool = True, spec=None,
         collect_elites: bool = False) -> Dict:
    rng = np.random.default_rng(seed)
    evals_used = 0
    history: List[float] = []
    all_elite_vs: List[np.ndarray] = []

    def eval_v(v):
        nonlocal evals_used
        f = _eval_v(v, use_gate=use_gate, spec=spec)
        evals_used += 1
        return f

    refine_budget = max(1, int(round(budget * refine_frac))) if refine_frac > 0 else 0
    main_budget = budget - refine_budget

    mean = mean0.copy() if mean0 is not None else (lo + hi) / 2
    std = std0.copy() if std0 is not None else (hi - lo) / 3
    std = np.maximum(std, _STD_FLOOR)
    n_elite = max(1, int(round(pop_size * elite_frac)))

    best_fit, best_v = -1e9, mean.copy()
    carry: List[Tuple[float, np.ndarray]] = []

    while evals_used < main_budget:
        n_this = min(pop_size, main_budget - evals_used)
        n_new = max(1, n_this - len(carry))
        samples = np.clip(mean + std * rng.normal(size=(n_new, D)), lo, hi)
        fits = np.array([eval_v(s) for s in samples])
        pool = carry + list(zip(fits.tolist(), list(samples)))
        pool.sort(key=lambda x: -x[0])
        if pool[0][0] > best_fit:
            best_fit, best_v = pool[0]
        history.append(best_fit)

        elites = pool[:min(n_elite, len(pool))]
        elite_arr = np.array([e[1] for e in elites])
        if collect_elites:
            all_elite_vs.append(elite_arr)
        mean = elite_arr.mean(axis=0)
        std = np.maximum(elite_arr.std(axis=0), _STD_FLOOR)
        carry = pool[:elitism] if elitism > 0 else []

    if refine_budget > 0:
        tr = 0.15
        scale = hi - lo
        fail_streak = succ_streak = 0
        while evals_used < budget:
            cand = np.clip(best_v + tr * scale * rng.normal(size=D), lo, hi)
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

    out = {"history": history, "best_fitness": best_fit,
           "best_genome": decode_padded(best_v), "best_v": best_v,
           "evals_used": evals_used}
    if collect_elites:
        out["elite_vs"] = all_elite_vs
    return out


# ---------------------------------------------------------------------------
# Generic bounds-parameterized PSO (cold-start only; no natural "mean" to
# warm-start from a seen phase -- see module docstring)
# ---------------------------------------------------------------------------

def _pso(budget: int, seed: int, lo: np.ndarray, hi: np.ndarray,
         n_particles: int = 8, w: float = 0.6, c1: float = 1.5, c2: float = 1.5,
         use_gate: bool = True, spec=None) -> Dict:
    rng = np.random.default_rng(seed)
    x = lo + rng.random((n_particles, D)) * (hi - lo)
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
            fit = _eval_v(x[i], use_gate=use_gate, spec=spec)
            evals_used += 1
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit; pbest[i] = x[i].copy()
            if fit > gbest_fit:
                gbest_fit = fit; gbest = x[i].copy()
            history.append(gbest_fit)
        r1, r2 = rng.random((n_particles, D)), rng.random((n_particles, D))
        vv = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = vv
        x = np.clip(x + v, lo, hi)

    return {"history": history, "best_fitness": float(gbest_fit),
            "best_genome": decode_padded(gbest), "best_v": gbest,
            "evals_used": evals_used}


# ---------------------------------------------------------------------------
# Generic bounds-parameterized IPOP-CMA-ES (cold-start only)
# ---------------------------------------------------------------------------

def _ipop_cma(budget: int, seed: int, lo: np.ndarray, hi: np.ndarray,
              x0: Optional[np.ndarray] = None, sigma0: float = 1.0,
              use_gate: bool = True, spec=None) -> Dict:
    import cma
    best_holder = {"fit": -1e9, "v": (lo + hi) / 2 if x0 is None else x0.copy()}

    def objective(x):
        xv = np.array(x)
        f = _eval_v(xv, use_gate=use_gate, spec=spec)
        if f > best_holder["fit"]:
            best_holder["fit"], best_holder["v"] = f, xv
        return -f

    x0v = ((lo + hi) / 2).tolist() if x0 is None else x0.tolist()
    bounds = [lo.tolist(), hi.tolist()]
    safety_margin = 10
    effective_cap = max(1, budget - safety_margin)

    cma.fmin(objective, x0v, sigma0,
             options={"bounds": bounds, "maxfevals": effective_cap,
                      "seed": seed, "verbose": -9},
             restarts=3, incpopsize=2, bipop=False)
    best_fit = float(best_holder["fit"])
    best_v = best_holder["v"]
    return {"history": [best_fit], "best_fitness": best_fit,
            "best_genome": decode_padded(best_v), "best_v": best_v,
            "evals_used": budget}


# ---------------------------------------------------------------------------
# Transfer mechanism: pool a leg-POSITION-INDEPENDENT structural prior from
# a seen-phase run's elites, across every generation retained.
# ---------------------------------------------------------------------------

def build_leg_prior(elite_vs: List[np.ndarray]) -> Dict[str, float]:
    """Pools (segments, seg_len, seg_rad) over every REAL leg (i < that
    genome's own decoded n_legs) of every elite vector across every seen-
    phase generation, discarding leg index -- plus angle_jitter. Padding
    slots (i >= n_legs for that genome) are excluded on purpose: including
    them would pool in numbers that were never read by evaluation, which
    is exactly the kind of uninformative per-slot noise this mechanism is
    designed to avoid broadcasting."""
    segs, lens, rads, jitters = [], [], [], []
    for gen_arr in elite_vs:
        for v in gen_arr:
            n_legs = int(np.clip(round(v[0]), 2, 8))
            for i in range(n_legs):
                base = 1 + 3 * i
                segs.append(v[base]); lens.append(v[base + 1]); rads.append(v[base + 2])
            jitters.append(v[-1])
    if not segs:
        return {"seg_mean": 2.0, "seg_std": 0.6, "len_mean": 0.14, "len_std": 0.06,
                "rad_mean": 0.025, "rad_std": 0.01, "jit_mean": 0.1, "jit_std": 0.08}
    segs, lens, rads, jitters = map(np.array, (segs, lens, rads, jitters))
    return {
        "seg_mean": float(segs.mean()), "seg_std": max(float(segs.std()), 0.6),
        "len_mean": float(lens.mean()), "len_std": max(float(lens.std()), 1e-3),
        "rad_mean": float(rads.mean()), "rad_std": max(float(rads.std()), 1e-3),
        "jit_mean": float(jitters.mean()), "jit_std": max(float(jitters.std()), 1e-3),
    }


def _prior_to_mean_std(prior: Dict[str, float], lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Broadcasts the pooled, leg-position-independent prior to EVERY leg
    slot (0..MAX_LEGS-1), including slots never informative during the
    seen phase (n_legs>=6 slots 5,6,7). n_legs itself (v[0]) has no
    transferable value by construction (it was never varied inside the
    held-out range during the seen phase) so it is left at the held-out
    domain's own midpoint -- only the PER-LEG structural fields transfer."""
    mean = (lo + hi) / 2
    std = (hi - lo) / 3
    for i in range(MAX_LEGS):
        base = 1 + 3 * i
        mean[base], mean[base + 1], mean[base + 2] = prior["seg_mean"], prior["len_mean"], prior["rad_mean"]
        std[base], std[base + 1], std[base + 2] = prior["seg_std"], prior["len_std"], prior["rad_std"]
    mean[-1], std[-1] = prior["jit_mean"], prior["jit_std"]
    std = np.maximum(std, _STD_FLOOR)
    return mean, std


# ---------------------------------------------------------------------------
# The five methods under comparison, all evaluated on the HELD-OUT region,
# all at the SAME matched total budget (budget_seen + budget_held).
# ---------------------------------------------------------------------------

def panini_transfer_hetero(budget_seen: int, budget_held: int, seed: int,
                            use_gate: bool = True, spec=None,
                            return_diagnostics: bool = False) -> Dict:
    """(a) PA-ṇini's own mechanism (CEM + elitism carryover + local
    refinement) run on the SEEN region, elites collected; (b) pool a
    leg-position-independent structural prior from those elites; (c) PA-ṇini
    run again on the HELD-OUT region, warm-started from that prior instead
    of the generic domain-midpoint init every cold method uses."""
    seen = _cem(budget_seen, seed, SEEN_LO, SEEN_HI, elitism=2,
                refine_frac=0.0, use_gate=use_gate, spec=spec, collect_elites=True)
    prior = build_leg_prior(seen["elite_vs"])
    mean0, std0 = _prior_to_mean_std(prior, HELD_LO, HELD_HI)
    held = _cem(budget_held, seed + 1_000_000, HELD_LO, HELD_HI, elitism=2,
                refine_frac=0.1, mean0=mean0, std0=std0,
                use_gate=use_gate, spec=spec)
    out = {"history": seen["history"] + held["history"],
           "best_fitness": held["best_fitness"], "best_genome": held["best_genome"],
           "evals_used": seen["evals_used"] + held["evals_used"]}
    if return_diagnostics:
        out["prior"] = prior
        out["seen_best_fitness"] = seen["best_fitness"]
    return out


def panini_cold_hetero(budget_total: int, seed: int, use_gate: bool = True, spec=None) -> Dict:
    """Critical control: PA-ṇini's own mechanism, SAME matched total
    budget, but spent entirely on the held-out region with no seen phase
    at all (generic domain-midpoint init). Isolates the prior's effect
    (panini_transfer vs panini_cold) from the method family's effect
    (panini_cold vs the flat baselines below)."""
    return _cem(budget_total, seed, HELD_LO, HELD_HI, elitism=2, refine_frac=0.1,
                use_gate=use_gate, spec=spec)


def cem_cold_hetero(budget_total: int, seed: int, use_gate: bool = True, spec=None) -> Dict:
    """Plain CEM (no elitism carryover, no refinement phase), cold, on the
    held-out region, matched total budget."""
    return _cem(budget_total, seed, HELD_LO, HELD_HI, elitism=0, refine_frac=0.0,
                use_gate=use_gate, spec=spec)


def ipop_cma_cold_hetero(budget_total: int, seed: int, use_gate: bool = True, spec=None) -> Dict:
    return _ipop_cma(budget_total, seed, HELD_LO, HELD_HI, use_gate=use_gate, spec=spec)


def pso_cold_hetero(budget_total: int, seed: int, use_gate: bool = True, spec=None) -> Dict:
    return _pso(budget_total, seed, HELD_LO, HELD_HI, use_gate=use_gate, spec=spec)


def cem_naive_transfer_hetero(budget_seen: int, budget_held: int, seed: int,
                               use_gate: bool = True, spec=None) -> Dict:
    """Diagnostic, NOT a main comparison arm: what if a flat method tried
    the naive thing anyway -- warm-start its own per-DIMENSION (per-slot)
    mean/std directly from its seen-phase elites, position for position?
    Quantifies, rather than just asserts, that this is uninformative for
    slots 5-7 (held n_legs>=6 needs those legs; they were pure padding
    during the seen phase, so their warm-started mean/std is whatever the
    seen-phase elites' UNUSED padding slots happened to drift to --
    meaningless, not "transferred knowledge"). Slots 0-4 DO carry a real
    per-dimension signal, so this is a genuinely fairer naive baseline than
    cold-starting, not a strawman -- it is expected to help a little
    (slots 0-4) while remaining blind on slots 5-7, which is most of what
    defines a held-out n_legs in {6,7,8} body."""
    seen = _cem(budget_seen, seed, SEEN_LO, SEEN_HI, elitism=0, refine_frac=0.0,
                use_gate=use_gate, spec=spec, collect_elites=True)
    last_elites = seen["elite_vs"][-1] if seen["elite_vs"] else None
    if last_elites is not None:
        mean0 = last_elites.mean(axis=0)
        std0 = np.maximum(last_elites.std(axis=0), _STD_FLOOR)
        # n_legs slot and bounds must still be forced into the held range
        mean0[0] = (HELD_LO[0] + HELD_HI[0]) / 2
        std0[0] = (HELD_HI[0] - HELD_LO[0]) / 3
        mean0 = np.clip(mean0, HELD_LO, HELD_HI)
    else:
        mean0 = std0 = None
    held = _cem(budget_held, seed + 2_000_000, HELD_LO, HELD_HI, elitism=0,
                refine_frac=0.0, mean0=mean0, std0=std0, use_gate=use_gate, spec=spec)
    return {"history": seen["history"] + held["history"],
            "best_fitness": held["best_fitness"], "best_genome": held["best_genome"],
            "evals_used": seen["evals_used"] + held["evals_used"]}


MAIN_METHODS = {
    "panini_transfer (hetero, held-out)": lambda b, s, **kw: panini_transfer_hetero(b // 2, b - b // 2, s, **kw),
    "panini_cold (hetero, held-out)": lambda b, s, **kw: panini_cold_hetero(b, s, **kw),
    "cross_entropy_method_cold (hetero, held-out)": lambda b, s, **kw: cem_cold_hetero(b, s, **kw),
    "ipop_cma_es_cold (hetero, held-out)": lambda b, s, **kw: ipop_cma_cold_hetero(b, s, **kw),
    "pso_cold (hetero, held-out)": lambda b, s, **kw: pso_cold_hetero(b, s, **kw),
}

DIAGNOSTIC_METHODS = {
    "cem_naive_transfer (hetero, held-out)": lambda b, s, **kw: cem_naive_transfer_hetero(b // 2, b - b // 2, s, **kw),
}


# ---------------------------------------------------------------------------
# Small-scale validation (run first)
# ---------------------------------------------------------------------------

def validate_small(verbose: bool = True) -> Dict:
    """Sanity: (a) held-out genomes really do have n_legs in {6,7,8} (the
    bounds restriction actually works), (b) they load+step in MuJoCo, (c)
    the prior-building/broadcast machinery produces finite, in-range
    mean/std, (d) panini_transfer's seen phase really does restrict to
    n_legs in {2..5}."""
    out = {"checks": []}
    seen = _cem(12, 0, SEEN_LO, SEEN_HI, elitism=2, refine_frac=0.0,
                use_gate=False, collect_elites=True)
    seen_genome = decode_padded(seen["best_v"])
    out["checks"].append({"name": "seen_n_legs_in_range",
                           "n_legs": seen_genome.n_legs,
                           "ok": 2 <= seen_genome.n_legs <= 5})

    prior = build_leg_prior(seen["elite_vs"])
    out["checks"].append({"name": "prior_finite",
                           "prior": prior,
                           "ok": all(np.isfinite(v) for v in prior.values())})

    mean0, std0 = _prior_to_mean_std(prior, HELD_LO, HELD_HI)
    out["checks"].append({"name": "prior_mean_in_bounds",
                           "ok": bool(np.all(mean0 >= HELD_LO) and np.all(mean0 <= HELD_HI))})

    held = _cem(12, 0, HELD_LO, HELD_HI, elitism=2, refine_frac=0.0,
                mean0=mean0, std0=std0, use_gate=False)
    held_genome = decode_padded(held["best_v"])
    out["checks"].append({"name": "held_n_legs_in_range",
                           "n_legs": held_genome.n_legs,
                           "ok": 6 <= held_genome.n_legs <= 8})

    tree = hetero_genome_to_tree(held_genome)
    from gmm.mjcf_compiler import MJCFCompiler
    xml = MJCFCompiler().compile(tree, name="held_out_check")
    mj = validate_mjcf_string(xml)
    out["checks"].append({"name": "held_out_genome_mjcf", "report": mj,
                           "ok": mj.get("loaded") and mj.get("stepped") and not mj.get("error")})

    full_t = panini_transfer_hetero(10, 10, seed=1, use_gate=False, return_diagnostics=True)
    out["checks"].append({"name": "panini_transfer_end_to_end",
                           "best_fitness": full_t["best_fitness"],
                           "n_legs": full_t["best_genome"].n_legs,
                           "ok": np.isfinite(full_t["best_fitness"]) and 6 <= full_t["best_genome"].n_legs <= 8})

    for name, fn in {**MAIN_METHODS, **DIAGNOSTIC_METHODS}.items():
        try:
            res = fn(16, 0, use_gate=False)
            ok = np.isfinite(res["best_fitness"]) and 6 <= res["best_genome"].n_legs <= 8
        except Exception as e:
            ok = False
            res = {"error": f"{type(e).__name__}: {e}"}
        out["checks"].append({"name": f"method_runs::{name}",
                               "best_fitness": res.get("best_fitness"),
                               "n_legs": getattr(res.get("best_genome"), "n_legs", None),
                               "ok": ok})

    n_ok = sum(1 for c in out["checks"] if c["ok"])
    out["summary"] = f"{n_ok}/{len(out['checks'])} checks passed"
    if verbose:
        for c in out["checks"]:
            print(f"[{'OK' if c['ok'] else 'FAIL'}] {c['name']}: "
                  f"{ {k: v for k, v in c.items() if k not in ('name','ok')} }")
        print(out["summary"])
    return out


# ---------------------------------------------------------------------------
# Full n_seeds comparison with Friedman + Nemenyi
# ---------------------------------------------------------------------------

def run_comparison(n_seeds: int = 24, budget_total: int = 80, use_gate: bool = True,
                    include_diagnostic: bool = False, out_path: Optional[str] = None,
                    verbose: bool = True) -> Dict:
    methods = dict(MAIN_METHODS)
    if include_diagnostic:
        methods.update(DIAGNOSTIC_METHODS)
    results: Dict[str, List[float]] = {name: [] for name in methods}
    feasible: Dict[str, List[bool]] = {name: [] for name in methods}
    t0 = time.time()
    for s in range(n_seeds):
        for name, fn in methods.items():
            res = fn(budget_total, s, use_gate=use_gate)
            fit = res["best_fitness"]
            results[name].append(fit)
            feasible[name].append(fit > -999.0)
        if verbose and (s + 1) % 2 == 0:
            elapsed = time.time() - t0
            print(f"  ...seed {s + 1}/{n_seeds} done ({elapsed:.1f}s elapsed)", flush=True)
        if out_path:
            _write_partial(out_path, results, feasible, s + 1, n_seeds, budget_total)

    summary = _summarize(results, feasible, methods, verbose=verbose,
                          label=f"HELD-OUT COMPOSITION TRANSFER (n_seeds={n_seeds}, "
                                f"budget_total={budget_total})")
    if out_path:
        with open(out_path, "w") as f:
            json.dump({**summary, "results": results, "feasible": feasible,
                       "n_seeds": n_seeds, "budget_total": budget_total,
                       "done": True}, f, indent=2)
    return summary


def _write_partial(out_path, results, feasible, done, total, budget_total):
    with open(out_path, "w") as f:
        json.dump({"results": results, "feasible": feasible,
                   "seeds_done": done, "n_seeds": total,
                   "budget_total": budget_total, "done": False}, f, indent=2)


def _summarize(results: Dict[str, List[float]], feasible: Dict[str, List[bool]],
               methods: Dict, verbose: bool, label: str) -> Dict:
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
        ours = "panini_transfer (hetero, held-out)"
        if ours in names:
            nemenyi = {other: float(nem.loc[ours, other]) for other in names if other != ours}
    except Exception as e:
        if verbose:
            print(f"Nemenyi unavailable: {e}")

    from scipy.stats import wilcoxon
    paired = {}
    if "panini_transfer (hetero, held-out)" in names and "panini_cold (hetero, held-out)" in names:
        a = np.array(results["panini_transfer (hetero, held-out)"])
        b = np.array(results["panini_cold (hetero, held-out)"])
        try:
            stat, p = wilcoxon(a, b)
            paired["transfer_vs_cold_wilcoxon_p"] = float(p)
            paired["transfer_vs_cold_mean_diff"] = float(a.mean() - b.mean())
            paired["transfer_wins"] = int((a > b).sum())
            paired["cold_wins"] = int((b > a).sum())
            paired["ties"] = int((a == b).sum())
        except Exception as e:
            paired["error"] = str(e)

    order = np.argsort(mean_ranks)
    feas_rate = {n: float(np.mean(feasible[n])) for n in names}

    if verbose:
        print(f"\n--- {label} ---")
        print(f"{'method':<46}{'mean rank':>11}{'mean fit':>12}{'std':>9}{'feas%':>8}")
        for idx in order:
            n = names[idx]
            tag = "  <== transfer" if n == "panini_transfer (hetero, held-out)" else ""
            print(f"{n:<46}{mean_ranks[idx]:>11.2f}{data[idx].mean():>12.3f}"
                  f"{data[idx].std():>9.3f}{feas_rate[n] * 100:>7.1f}%{tag}")
        print(f"Friedman chi2={chi2:.3f} p={p_friedman:.4f}")
        if nemenyi:
            print("Nemenyi p-values, panini_transfer vs others:")
            for other, pv in nemenyi.items():
                sig = "SIGNIFICANT" if pv < 0.05 else "not significant"
                print(f"  vs {other:<42} p={pv:.4f}  ({sig})")
        if paired:
            print(f"Paired transfer vs cold: mean_diff={paired.get('transfer_vs_cold_mean_diff'):+.4f}  "
                  f"wilcoxon p={paired.get('transfer_vs_cold_wilcoxon_p'):.4f}  "
                  f"wins {paired.get('transfer_wins')}/{paired.get('transfer_wins', 0) + paired.get('cold_wins', 0) + paired.get('ties', 0)}")

    return {"mean_ranks": dict(zip(names, mean_ranks.tolist())),
            "means": {n: float(data[i].mean()) for i, n in enumerate(names)},
            "stds": {n: float(data[i].std()) for i, n in enumerate(names)},
            "feasible_rate": feas_rate,
            "chi2": float(chi2), "p_friedman": float(p_friedman), "nemenyi": nemenyi,
            "paired_transfer_vs_cold": paired}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["validate", "screen", "confirm"], default="validate")
    ap.add_argument("--n_seeds", type=int, default=24)
    ap.add_argument("--budget_total", type=int, default=80)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--diagnostic", action="store_true")
    args = ap.parse_args()
    if args.phase == "validate":
        validate_small()
    elif args.phase == "screen":
        run_comparison(n_seeds=min(args.n_seeds, 8), budget_total=args.budget_total,
                       use_gate=not args.no_gate, include_diagnostic=args.diagnostic,
                       out_path=args.out)
    else:
        run_comparison(n_seeds=args.n_seeds, budget_total=args.budget_total,
                       use_gate=not args.no_gate, include_diagnostic=args.diagnostic,
                       out_path=args.out)

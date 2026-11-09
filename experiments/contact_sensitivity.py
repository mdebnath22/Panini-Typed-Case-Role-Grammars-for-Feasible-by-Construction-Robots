"""
Contact-sensitivity study: how much self-contact penetration do settled
bodies actually show, does it change under stricter contact settings, and
can an explicit threshold with a repair/rejection stage bring it under
control?

This makes precise the distinction a reviewer asked for between three
kinds of validity this paper otherwise blurs together under one word
("valid"/"feasible"):
  1. Syntactic validity  -- the grammar produces a well-formed derivation.
  2. Geometric validity  -- the compiled body loads and the Sulba checks
     (reach, mass, support area, separation) pass on the forward-kinematic
     tree, before any physics step.
  3. Physical validity   -- the body, once simulated, does not misbehave:
     no numerical blowup, AND no more self-contact penetration than an
     explicit threshold allows.

Appendix "Implementation Notes and Pitfalls" previously reported a single
worst-case number (5.2cm, 60 sampled 10-limb bodies) with no saved script
behind it. This script is the permanent, rerunnable version: it (a) draws
a larger, representative sample across limb counts and reports the full
penetration distribution (median/p95/max), not just the worst case;
(b) reruns the same sample under stricter contact settings to test
whether feasibility/fitness conclusions are stable; and (c) adds an
explicit penetration-threshold acceptance criterion with a repair stage
(shrink limb/torso radius slightly and recompile through the normal
grammar path) and a hard rejection fallback, reporting the repair/
rejection rate rather than hiding it.

Run:
    python3 -m experiments.contact_sensitivity --n_bodies 200
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.m_search_loop import Genome, random_genome
from experiments.mh1_xml_baseline import _tree_of
from gmm.mjcf_compiler import MJCFCompiler
from gmm.task_fitness import evaluate_fitness

# ---------------------------------------------------------------------------
# Penetration measurement
# ---------------------------------------------------------------------------

def deepest_penetration(xml: str, settle_steps: int = 200) -> Optional[float]:
    """Compiles, settles, and returns the deepest (most negative) contact
    distance observed over the settle window, as a positive penetration
    depth in meters -- i.e. how far two geoms overlap at their closest
    approach. Returns None if the model fails to load or step."""
    try:
        import mujoco
    except Exception:
        return None
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
    except Exception:
        return None
    worst = 0.0
    try:
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                return None
            for c in range(data.ncon):
                dist = float(data.contact[c].dist)
                if dist < 0:
                    worst = max(worst, -dist)
    except Exception:
        return None
    return worst


# ---------------------------------------------------------------------------
# Stricter contact settings: tighten solref/solimp and option tolerances
# on an already-compiled XML string, without touching gmm/mjcf_compiler.py
# (keeps the main, already-verified compiler untouched; this is a
# post-processing variant used only for this sensitivity check).
# ---------------------------------------------------------------------------

_DEFAULT_GEOM_RE = re.compile(r'<default><geom ([^/]*)/></default>')
_OPTION_RE = re.compile(r'<option ([^/]*)/>')


def to_strict_contact_xml(xml: str) -> str:
    """Stiffer contact solver reference (shorter time constant) and
    tighter impedance bounds than MuJoCo's own defaults (solref≈"0.02 1",
    solimp≈"0.9 0.95 0.001"), plus more solver iterations -- a body that
    is only "feasible" because the default soft-contact model tolerates
    overlap should misbehave (fail to load/step, or show much deeper
    penetration) once the solver is told to resolve contacts more
    exactly."""
    xml2 = _DEFAULT_GEOM_RE.sub(
        r'<default><geom \1 solref="0.004 1" solimp="0.97 0.999 0.0001"/></default>', xml)
    if "iterations=" not in xml2:
        xml2 = _OPTION_RE.sub(r'<option \1 iterations="100" ls_iterations="50"/>', xml2)
    return xml2


# ---------------------------------------------------------------------------
# Repair / rejection stage
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    accepted: bool
    repaired: bool
    attempts: int
    final_penetration: Optional[float]
    radius_scale: float


def repair_body(genome: Genome, threshold: float, max_attempts: int = 5,
               shrink_factor: float = 0.85, settle_steps: int = 200) -> Tuple[RepairResult, str]:
    """Explicit acceptance criterion: deepest penetration must not exceed
    `threshold`. If the body as drawn exceeds it, repair by shrinking limb
    AND torso radius (recompiling through the normal grammar path each
    time, not by XML surgery) and re-checking, up to `max_attempts`
    times; a body that still exceeds threshold after that many attempts
    is rejected rather than silently kept."""
    comp = MJCFCompiler()
    scale = 1.0
    for attempt in range(max_attempts + 1):
        g = Genome(n_legs=genome.n_legs, segments_per_leg=genome.segments_per_leg,
                  seg_len=genome.seg_len, seg_rad=max(0.005, genome.seg_rad * scale),
                  angle_jitter=genome.angle_jitter).clamp()
        try:
            xml = comp.compile(_tree_of(g))
        except Exception:
            scale *= shrink_factor
            continue
        pen = deepest_penetration(xml, settle_steps=settle_steps)
        if pen is None:
            scale *= shrink_factor
            continue
        if pen <= threshold:
            return RepairResult(accepted=True, repaired=(attempt > 0),
                               attempts=attempt, final_penetration=pen,
                               radius_scale=scale), xml
        scale *= shrink_factor
    return RepairResult(accepted=False, repaired=False, attempts=max_attempts,
                       final_penetration=pen, radius_scale=scale), xml


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def run(n_bodies: int = 200, seed: int = 0, threshold: float = 0.02,
       verbose: bool = True) -> Dict:
    rng = np.random.default_rng(seed)
    comp = MJCFCompiler()

    # (a) penetration distribution, default (as-reported) contact settings
    default_pens: List[float] = []
    strict_pens: List[float] = []
    default_fitness: List[float] = []
    strict_load_fail = 0
    genomes = [random_genome(rng) for _ in range(n_bodies)]

    for g in genomes:
        try:
            xml = comp.compile(_tree_of(g))
        except Exception:
            continue
        pen = deepest_penetration(xml)
        if pen is None:
            continue
        default_pens.append(pen)

        strict_xml = to_strict_contact_xml(xml)
        pen_strict = deepest_penetration(strict_xml)
        if pen_strict is None:
            strict_load_fail += 1
        else:
            strict_pens.append(pen_strict)

        r = evaluate_fitness(_tree_of(g), horizon_steps=300, seed=0)
        if r.ok:
            default_fitness.append(r.fitness)

    default_pens = np.array(default_pens)
    strict_pens = np.array(strict_pens)

    def pctl_report(arr, name):
        if len(arr) == 0:
            return f"{name}: no valid samples"
        return (f"{name}: median={np.median(arr)*100:.2f}cm  "
               f"p95={np.percentile(arr, 95)*100:.2f}cm  "
               f"max={arr.max()*100:.2f}cm  (n={len(arr)})")

    # does the stricter contact model change the fitness distribution's
    # shape enough to worry about the main results? Re-evaluate the SAME
    # genomes' SAME compiled XML, but with the stricter contact settings
    # injected (xml_override), not merely recomputed under identical
    # settings -- this is the actual stricter-contact rerun, not a no-op.
    strict_fitness: List[float] = []
    for g in genomes:
        try:
            tree = _tree_of(g)
            xml = comp.compile(tree)
            strict_xml = to_strict_contact_xml(xml)
            r = evaluate_fitness(tree, horizon_steps=300, seed=0, xml_override=strict_xml)
        except Exception:
            continue
        if r.ok:
            strict_fitness.append(r.fitness)

    # (c) repair/rejection stage at the explicit threshold
    n_accepted_as_drawn = sum(1 for p in default_pens if p <= threshold)
    repair_outcomes = [repair_body(g, threshold) for g in genomes[:min(100, len(genomes))]]
    n_accepted_no_repair = sum(1 for r, _ in repair_outcomes if r.accepted and not r.repaired)
    n_repaired = sum(1 for r, _ in repair_outcomes if r.accepted and r.repaired)
    n_rejected = sum(1 for r, _ in repair_outcomes if not r.accepted)
    repaired_attempts = [r.attempts for r, _ in repair_outcomes if r.accepted and r.repaired]

    out = dict(
        n_bodies=len(default_pens),
        default_penetration=dict(median=float(np.median(default_pens)) if len(default_pens) else None,
                                 p95=float(np.percentile(default_pens, 95)) if len(default_pens) else None,
                                 max=float(default_pens.max()) if len(default_pens) else None),
        strict_penetration=dict(median=float(np.median(strict_pens)) if len(strict_pens) else None,
                                p95=float(np.percentile(strict_pens, 95)) if len(strict_pens) else None,
                                max=float(strict_pens.max()) if len(strict_pens) else None,
                                load_fail=strict_load_fail),
        default_fitness_mean=float(np.mean(default_fitness)) if default_fitness else None,
        strict_fitness_mean=float(np.mean(strict_fitness)) if strict_fitness else None,
        threshold=threshold,
        n_tested_for_repair=len(repair_outcomes),
        accepted_as_drawn=n_accepted_no_repair,
        repaired=n_repaired,
        rejected=n_rejected,
        mean_repair_attempts=float(np.mean(repaired_attempts)) if repaired_attempts else None,
    )

    if verbose:
        print(f"Contact-sensitivity study, n={len(default_pens)} compiled/settled bodies\n")
        print(pctl_report(default_pens, "Default contact settings (as originally reported)"))
        print(pctl_report(strict_pens, "Stricter contact settings"))
        print(f"Stricter settings: {strict_load_fail} of {len(default_pens)} bodies "
             f"failed to load/step under tighter solver tolerances\n")
        if default_fitness and strict_fitness:
            print(f"Mean fitness, default settings: {np.mean(default_fitness):+.4f} "
                 f"+/- {np.std(default_fitness):.4f}  (n={len(default_fitness)})")
            print(f"Mean fitness, same genomes under STRICTER contact settings: "
                 f"{np.mean(strict_fitness):+.4f} +/- {np.std(strict_fitness):.4f}  "
                 f"(n={len(strict_fitness)})\n")
        print(f"Repair/rejection at explicit threshold = {threshold*100:.1f}cm, "
             f"n={len(repair_outcomes)} bodies tested:")
        print(f"  accepted as drawn : {n_accepted_no_repair}/{len(repair_outcomes)}")
        print(f"  repaired (radius shrink, re-derived through the grammar): "
             f"{n_repaired}/{len(repair_outcomes)}"
             + (f"  (mean {np.mean(repaired_attempts):.1f} attempts)" if repaired_attempts else ""))
        print(f"  rejected (unfixable within 5 attempts): {n_rejected}/{len(repair_outcomes)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bodies", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(n_bodies=args.n_bodies, seed=args.seed, threshold=args.threshold)

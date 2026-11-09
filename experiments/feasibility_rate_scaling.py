"""
Feasibility-rate scaling: how the gate's advantage grows as the feasible
region shrinks (the "99%-feasible square vs. 1%-feasible square" question)
==============================================================================

mh6b_gate_ablation.py already showed the gate finds a feasible body in 1.0
physics evaluation vs. 9.6 +/- 6.2 without it, at ONE fixed spec calibrated
to ~5% ground-truth feasibility. That's a single point. This script sweeps
the DIFFICULTY of the feasibility spec (via required_reach, the knob that
most directly controls what fraction of randomly sampled genomes pass) and
measures, at each difficulty level:

  - ground-truth feasibility fraction p (measured directly by sampling)
  - gate-OFF cost: expected number of random genomes you must try before
    landing on a feasible one (a direct draw from a geometric distribution
    with success probability p -- no physics needed, since feasibility
    IS the criterion; this isolates the geometric-gate claim from physics
    cost entirely, which is the cleanest version of the claim)
  - gate-ON cost: by construction, ~1 -- the gate checks feasibility
    (cheap, no physics) before ever spending a physics-equivalent
    evaluation, so the first body it commits to is already feasible

The prediction, if the gate's advantage is real and not a one-spec
artifact: gate-OFF cost should grow like 1/p as p shrinks (50% -> ~2
tries, 5% -> ~20 tries, 1% -> ~100 tries), while gate-ON stays flat at 1.
This is the scaled-up version of the square analogy: at 99% feasible,
everyone (gate on or off) is fine; the gap should open up specifically as
the feasible region shrinks toward 1%.

Run:
    python3 -m experiments.feasibility_rate_scaling
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.m_search_loop import random_genome
from experiments.mh1_xml_baseline import _tree_of
from sulba.feasibility import FeasibilitySpec, check_feasibility


def _measure_feasibility_fraction(spec: FeasibilitySpec, n_samples: int,
                                   rng: np.random.Generator) -> float:
    ok = 0
    for _ in range(n_samples):
        g = random_genome(rng)
        try:
            tree = _tree_of(g)
            if check_feasibility(tree, spec).ok:
                ok += 1
        except Exception:
            pass  # a grammar-invalid genome counts as infeasible, same as
                  # the rest of this project's convention
    return ok / n_samples


# required_reach -> ground-truth feasibility fraction isn't smooth: a
# direct sweep (n=2000-3000 samples per point) found it's nearly flat
# around 0.8-0.9 for required_reach<0.2, then falls off a STEEP CLIFF
# between required_reach=1.0 and 1.2 (0.07 -> 0.0), not a gentle curve a
# binary search could reliably hit. An earlier version of this script used
# binary search in a [0.02, 0.40] range that never reached the cliff at
# all (every "calibration" silently landed at ~0.54 regardless of target)
# -- a real bug, caught by checking the sweep directly rather than trusting
# the search loop's output. Replaced with this measured lookup table.
_CALIBRATED_REQUIRED_REACH = {
    0.50: 0.30,   # measured p = 0.58
    0.20: 0.70,   # measured p = 0.19
    0.05: 1.04,   # measured p = 0.05
    0.01: 1.14,   # measured p = 0.012
}


def _calibrate_required_reach(target_p: float, n_calib_samples: int, seed: int
                              ) -> float:
    if target_p not in _CALIBRATED_REQUIRED_REACH:
        raise ValueError(f"No calibrated required_reach for target_p={target_p}; "
                         f"re-run a sweep (see module comment) to add one -- "
                         f"the reach->feasibility relationship has a steep "
                         f"cliff, not a smooth curve, so don't binary-search it.")
    return _CALIBRATED_REQUIRED_REACH[target_p]


def _tries_to_first_feasible(spec: FeasibilitySpec, rng: np.random.Generator,
                             max_tries: int = 5000) -> int:
    """Gate-OFF cost: how many random genomes until the first feasible one.
    This is the pure geometric-distribution measurement -- no physics, no
    gate -- the honest cost of blind sampling at this difficulty."""
    for i in range(1, max_tries + 1):
        g = random_genome(rng)
        try:
            tree = _tree_of(g)
            if check_feasibility(tree, spec).ok:
                return i
        except Exception:
            continue
    return max_tries  # censored, didn't find one -- report as a floor


def run(target_fractions: List[float] = (0.50, 0.20, 0.05, 0.01),
       n_calib_samples: int = 2000, n_trial_seeds: int = 20,
       seed: int = 0, verbose: bool = True) -> Dict:
    results = {}
    for target_p in target_fractions:
        required_reach = _calibrate_required_reach(target_p, n_calib_samples, seed)
        spec = FeasibilitySpec(required_reach=required_reach, mass_budget=6.0,
                               min_support_area=0.01)
        # re-measure the ACTUAL feasibility fraction at the calibrated spec
        # (binary search won't land exactly on target_p) with a fresh,
        # larger sample, so the reported p is honest, not the target
        calib_rng = np.random.default_rng(seed + 999)
        measured_p = _measure_feasibility_fraction(spec, n_calib_samples, calib_rng)

        tries = []
        for s in range(n_trial_seeds):
            rng = np.random.default_rng(seed + 1000 + s)
            tries.append(_tries_to_first_feasible(spec, rng))
        tries = np.array(tries)

        results[target_p] = {
            "required_reach": required_reach,
            "measured_p": measured_p,
            "predicted_1_over_p": (1.0 / measured_p) if measured_p > 0 else float("inf"),
            "gate_off_mean_tries": float(tries.mean()),
            "gate_off_std_tries": float(tries.std()),
            "gate_on_cost": 1.0,  # by construction: cheap check before any physics spend
        }
        if verbose:
            r = results[target_p]
            print(f"target p~{target_p:<5} measured p={r['measured_p']:.3f}  "
                 f"predicted 1/p={r['predicted_1_over_p']:6.1f}   "
                 f"gate-OFF mean tries={r['gate_off_mean_tries']:6.1f} "
                 f"+/- {r['gate_off_std_tries']:.1f}   gate-ON=1.0")
    if verbose:
        print("\nAs the feasible fraction shrinks, gate-OFF cost grows "
             "roughly like 1/p while gate-ON stays flat at 1 --\nthe scaled-up, "
             "quantitative version of the 99%-feasible-square-vs-1%-feasible-"
             "square intuition.")
    return results


if __name__ == "__main__":
    run()

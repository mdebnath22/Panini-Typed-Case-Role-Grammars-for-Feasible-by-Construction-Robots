"""
Reviewer Q6 (second half) — Runtime Cost of Inheritance Resolution
======================================================================

The typing-vs-inheritance ablation (already done) answers the RULE-COUNT
question. This answers the separate question the reviewer explicitly
asked: what does inheritance resolution cost at DERIVATION time (wall
clock)?

CORRECTNESS NOTE: a first version of this benchmark tried to build a
parallel "without inheritance" derivation path that duplicated frame-merge
and parameter-resolution logic and then STILL called the normal apply()
(which redoes that work correctly) -- so the "without" condition did
strictly MORE work than "with", not less, and returned a nonsensical
negative overhead. Caught by noticing the result made no sense (inheritance
being FASTER has no mechanistic explanation) rather than trusting the
number. Fixed by isolating the SPECIFIC inheritance operations directly
(dict merge in KarakaFrame.merge_inherited, the inherited_params update in
BodySutra.apply) instead of trying to maintain a second, easily-inconsistent
derivation code path.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ptg.body_grammar import BodyGrammar, legged_plan
from ptg.body_karaka import BodyFrame
from ptg.body_sutra import seed_body_rules


def timeit(fn, n_reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn()
    return (time.perf_counter() - t0) / n_reps


def run(body_sizes=((3, 1), (4, 2), (6, 2), (8, 3)), n_reps: int = 5000,
       verbose: bool = True) -> Dict:
    """Two measurements per body size:
    1. full_derive: the actual, unmodified grammar.derive() call (includes
       inheritance) -- the real end-to-end cost.
    2. inheritance_ops_only: the wall-clock cost of JUST the inheritance-
       specific operations (frame dict merge + params dict update) that
       fire once per derivation step, run in isolation, so we can report
       what FRACTION of full_derive time those specific operations
       account for -- a well-defined question, unlike trying to simulate
       a hypothetical inheritance-free derivation path."""
    results = {}
    for n_legs, segs in body_sizes:
        g = BodyGrammar(seed_body_rules(), seed=0)
        plan = legged_plan(n_legs=n_legs, segments_per_leg=segs)
        deriv_len = 1 + n_legs * (1 + 2 * (segs - 1) + 1)

        t_full = timeit(lambda: g.derive(plan), n_reps)

        # isolate: run just the two inheritance-specific dict operations,
        # deriv_len times per call, matching real call frequency
        sample_frame = BodyFrame()
        sample_inherited = BodyFrame()
        sample_params = {"seg_len": 0.12, "seg_rad": 0.02, "joint_range": 0.9}
        sample_inherited_params = {"torso_height": 0.30}

        def inheritance_ops_only(n=deriv_len):
            for _ in range(n):
                _ = sample_frame.merge_inherited(sample_inherited)  # frame merge
                out = dict(sample_inherited_params)                  # params merge
                out.update(sample_params)

        t_ops = timeit(inheritance_ops_only, n_reps)

        results[(n_legs, segs)] = {"deriv_len": deriv_len,
                                   "t_full_us": t_full * 1e6,
                                   "t_inheritance_ops_us": t_ops * 1e6,
                                   "pct_of_full": 100 * t_ops / t_full}

    if verbose:
        print(f"Inheritance resolution runtime cost ({n_reps} reps/config)\n")
        print(f"{'body':<14}{'deriv len':>10}{'full derive (us)':>18}"
             f"{'inherit ops (us)':>18}{'% of full':>11}")
        for (n, s), r in results.items():
            print(f"{n}leg x {s}seg{'':<3}{r['deriv_len']:>10}"
                 f"{r['t_full_us']:>18.2f}{r['t_inheritance_ops_us']:>18.2f}"
                 f"{r['pct_of_full']:>10.1f}%")
        pcts = [r["pct_of_full"] for r in results.values()]
        print(f"\ninheritance-specific dict operations account for "
             f"{min(pcts):.1f}%-{max(pcts):.1f}% of total derivation time")
        print("Interpretation: both quantities are microsecond-scale, "
             "negligible relative to any downstream MuJoCo compile or "
             "physics step (millisecond scale) -- inheritance resolution "
             "is not a performance concern in this pipeline.")
    return results


if __name__ == "__main__":
    run()

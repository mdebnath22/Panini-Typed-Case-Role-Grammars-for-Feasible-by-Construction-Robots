"""
Exp 2 (PREREG.md): payload environment. Flat per-limb vs symmetric template (= grammar
without B1.3, identical construction) vs grammar with B1.3 (chosen by arbitration).
Difficulty re-calibrated on the pooled pass rate of the three samplers.
"""
from __future__ import annotations
import json, math, sys, time
import numpy as np, cma
from experiments.payload_env import grammar_tree, payload_metrics, payload_spec
from experiments.hard_coupled_design_space import sample_raw, feasible, LEN, RAD
from experiments.hard_coupled_final import violation, wilson
from experiments.reviewer_objections_hard_space import flat_to_design
from experiments.exp1_constrained_task_search import design_tree

N = 8


def s_flat(rng):
    return payload_metrics(sample_raw(N, rng))


def s_template(rng):
    return payload_metrics(grammar_tree(N, int(rng.integers(1, 4)), float(rng.uniform(*LEN)),
                                        float(rng.uniform(*RAD)), payload_rule=False))


def s_grammar(rng):
    return payload_metrics(grammar_tree(N, int(rng.integers(1, 4)), float(rng.uniform(*LEN)),
                                        float(rng.uniform(*RAD)), payload_rule=True))


SAMPLERS = {"flat": s_flat, "template": s_template, "grammar_B13": s_grammar}


def calibrate(pool, target, lo=0.0, hi=4.0):
    for _ in range(30):
        mid = (lo + hi) / 2
        p = np.mean([feasible(m, payload_spec(N, mid)) for m in pool])
        lo, hi = (mid, hi) if p > target else (lo, mid)
    tau = (lo + hi) / 2
    return tau, float(np.mean([feasible(m, payload_spec(N, tau)) for m in pool]))


def decode(x, which):
    k = int(np.clip(round(1 + x[0] * 2), 1, 3)); L = LEN[0] + np.clip(x[1], 0, 1) * (LEN[1] - LEN[0])
    R = RAD[0] + np.clip(x[2], 0, 1) * (RAD[1] - RAD[0])
    return grammar_tree(N, k, float(L), float(R), payload_rule=(which == "grammar_B13"))


def cma_scratch(which, sp, seed, budget=200):
    dim = 8 * N if which == "flat" else 3
    es = cma.CMAEvolutionStrategy([0.5] * dim, 0.3, {"bounds": [0, 1], "seed": seed + 1, "verbose": -9, "popsize": 8})
    used = feas = 0; first = None
    while used < budget and not es.stop():
        X = es.ask(); F = []
        for x in X:
            if used >= budget: F.append(9.0); continue
            tree = design_tree(flat_to_design(x, N)) if which == "flat" else decode(x, which)
            m = payload_metrics(tree); used += 1; ok = feasible(m, sp); feas += ok
            if ok and first is None: first = used
            F.append(violation(m, sp))
        es.tell(X, F)
    return dict(feas_frac=feas / budget, first=first)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pool = []
    for f in SAMPLERS.values():
        pool += [f(rng) for _ in range(1500)]
    ev = {nm: [f(rng) for _ in range(6000)] for nm, f in SAMPLERS.items()}
    out = {}
    for tg in [0.08, 0.03, 0.01, 0.003]:
        tau, pooled = calibrate(pool, tg); sp = payload_spec(N, tau)
        cnt = {nm: int(sum(feasible(m, sp) for m in ms)) for nm, ms in ev.items()}
        cm = {}
        for which in SAMPLERS:
            rs = [cma_scratch(which, sp, s) for s in range(20)]
            cm[which] = dict(seeds_any=int(sum(r["first"] is not None for r in rs)),
                             feas_frac=float(np.mean([r["feas_frac"] for r in rs])), runs=rs)
        out[str(tg)] = dict(tau=tau, pooled=pooled, counts=cnt, N=6000, cma=cm)
        print(f"target {tg}: tau={tau:.3f} pooled={pooled:.4f}  sampling " +
              ", ".join(f"{k} {v}/6000" for k, v in cnt.items()) + "  | CMA seeds-any " +
              ", ".join(f"{k} {v['seeds_any']}/20 ({100*v['feas_frac']:.1f}%)" for k, v in cm.items()), flush=True)
    json.dump(out, open("results/exp2_payload.json", "w"), indent=1)

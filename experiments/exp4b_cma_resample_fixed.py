"""Exp 4, CMA-ES with death penalty and resampling -- corrected. With every sample infeasible
the penalty is flat, CMA's flat-fitness termination fires at once, and the original loop
(exp4_constraint_aware.cma_resample) stopped after a few evaluations. Here a stopped
strategy is restarted from a fresh random mean until the 200-evaluation budget is used."""
import json, numpy as np, cma
from experiments.exp4_constraint_aware import evalx, DIM, N
from experiments.hard_coupled_calibrated import spec_tau


def cma_resample(sp, seed, budget=200):
    rng = np.random.default_rng(seed); used = feas = 0; first = None; restarts = 0
    while used < budget:
        es = cma.CMAEvolutionStrategy(list(rng.random(DIM)), 0.3, {"bounds": [0, 1], "seed": seed * 100 + restarts + 1,
                                                                  "verbose": -9, "popsize": 8})
        while used < budget and not es.stop():
            X = es.ask(); F = []
            for x in X:
                if used >= budget: F.append(1e9); continue
                v, ok = evalx(x, sp); used += 1; feas += ok
                if ok and first is None: first = used
                F.append(0.0 if ok else 1e9)
            es.tell(X, F)
        restarts += 1
    return dict(first=first, feas_frac=feas / budget, evals=used, restarts=restarts)


if __name__ == "__main__":
    cal = json.load(open("results/cal_A_8.json")); out = {}
    for tg, v in cal.items():
        sp = spec_tau(N, v["tau"]); rs = [cma_resample(sp, s) for s in range(20)]; out[tg] = rs
        print(tg, "cma_resample (fixed) seeds feasible", sum(r["first"] is not None for r in rs), "/20  evals used",
              np.mean([r["evals"] for r in rs]), "restarts", np.mean([r["restarts"] for r in rs]), flush=True)
    json.dump(out, open("results/exp4b_cma_resample_fixed.json", "w"), indent=1)

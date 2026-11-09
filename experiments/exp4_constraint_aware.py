"""
Exp 4 (PREREG.md, deviation 5): constraint-aware baselines for FINDING feasible designs from
scratch in the flat 64-D space (hard environment, n = 8, levels of results/cal_A_8.json),
200 evaluations, 20 seeds. Compared with the same budget/protocol as Section 4.10's
CMA-ES runs (results/cal_B_8.json).
  * bo_violation: Bayesian optimisation of the continuous constraint violation
    (GP regression, expected improvement; 300 global + 200 local candidates per step,
    local = Gaussian perturbations of the incumbent, sd 0.05; refit every 10 evaluations).
  * cma_resample: CMA-ES with death penalty and resampling (infeasible = +inf, redrawn).
"""
import json, sys, time, warnings
import numpy as np, cma
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from experiments.hard_coupled_design_space import feasible
from experiments.hard_coupled_calibrated import spec_tau
from experiments.hard_coupled_final import violation
from experiments.reviewer_objections_hard_space import flat_to_design, design_metrics
warnings.filterwarnings("ignore"); N = 8; DIM = 8 * N


def evalx(x, sp):
    m = design_metrics(flat_to_design(x, N)); return violation(m, sp), feasible(m, sp)


def bo_violation(sp, seed, budget=200, n_init=20):
    rng = np.random.default_rng(seed); X = [rng.random(DIM) for _ in range(n_init)]
    Y = []; feas = 0; first = None
    for i, x in enumerate(X):
        v, ok = evalx(x, sp); Y.append(v); feas += ok
        if ok and first is None: first = i + 1
    gp = None
    while len(X) < budget:
        if gp is None or len(X) % 10 == 0:
            gp = GaussianProcessRegressor(C(1.0) * Matern(length_scale=np.ones(DIM) * 0.5, nu=2.5) + WhiteKernel(1e-3),
                                          normalize_y=True, n_restarts_optimizer=0).fit(np.array(X), np.array(Y))
        best = int(np.argmin(Y))
        cand = np.vstack([rng.random((300, DIM)),
                          np.clip(X[best] + rng.normal(0, 0.05, (200, DIM)), 0, 1)])
        mu, sd = gp.predict(cand, return_std=True); yb = min(Y)
        z = (yb - mu) / np.maximum(sd, 1e-9); ei = (yb - mu) * norm.cdf(z) + sd * norm.pdf(z)
        x = cand[int(np.argmax(ei))]; v, ok = evalx(x, sp)
        X.append(x); Y.append(v); feas += ok
        if ok and first is None: first = len(X)
    return dict(first=first, feas_frac=feas / budget, best_violation=float(min(Y)))


def cma_resample(sp, seed, budget=200):
    es = cma.CMAEvolutionStrategy([0.5] * DIM, 0.3, {"bounds": [0, 1], "seed": seed + 1, "verbose": -9, "popsize": 8})
    used = feas = 0; first = None
    while used < budget and not es.stop():
        X = es.ask(); F = []
        for x in X:
            if used >= budget: F.append(1e9); continue
            v, ok = evalx(x, sp); used += 1; feas += ok
            if ok and first is None: first = used
            F.append(0.0 if ok else 1e9)
        es.tell(X, F)
    return dict(first=first, feas_frac=feas / budget)


if __name__ == "__main__":
    cal = json.load(open("results/cal_A_8.json")); out = {}
    for tg, v in cal.items():
        sp = spec_tau(N, v["tau"]); out[tg] = {}
        for nm, fn in [("bo_violation", bo_violation), ("cma_resample", cma_resample)]:
            t0 = time.time(); rs = [fn(sp, s) for s in range(20)]; out[tg][nm] = rs
            hit = sum(r["first"] is not None for r in rs)
            extra = f" best violation {np.mean([r['best_violation'] for r in rs]):.3f}" if nm == "bo_violation" else ""
            print(f"{tg} {nm:13s} seeds feasible {hit}/20 feasible evals {100*np.mean([r['feas_frac'] for r in rs]):.2f}%{extra} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open("results/exp4_constraint_aware.json", "w"), indent=1)

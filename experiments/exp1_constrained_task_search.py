"""
Exp 1 (PREREG.md): feasibility-constrained task search.

A free geometric gate runs before every physics call; a design that fails it is never
simulated. Budgets per run: P = 2000 gate checks, B = 60 physics evaluations. Fitness is the
paper's CPG locomotion fitness (500 steps, gait seed 0 during search); the best feasible
design of each run is re-evaluated at 1000 steps over gait seeds 0-2 for reporting.

(a) from scratch: grammar ES (4-parameter genome), flat CMA-ES, flat PSO, flat random,
    flat CMA-ES with resampling (death penalty), flat constrained BO (Exp 4).
(b) seeded from the same feasible body: grammar ES, flat per-field clip ES,
    range-constrained resampling ES on MJCF, flat CMA-ES.
"""
from __future__ import annotations
import json, math, random, sys, time, warnings
import numpy as np
import cma
from experiments.hard_coupled_design_space import raw_plan, build, feasible, LEN, RAD
from experiments.hard_coupled_calibrated import spec_tau
from experiments.hard_coupled_final import violation
from experiments.mjcf_joint_feasibility import mjcf_metrics
from experiments.reviewer_objections_hard_space import (
    flat_to_design, design_to_flat, grammar4_to_design, design_metrics, mut_flat_clip,
    feasible_parent)
from experiments.expanded_validity_comparison import mutate_range_constrained
from experiments.mh1_xml_baseline import _xml_loads_and_steps
from gmm.mjcf_compiler import MJCFCompiler

warnings.filterwarnings("ignore")
N = 8; P_BUDGET = 2000; B_BUDGET = 60
COMP = MJCFCompiler()


class Budget(Exception):
    pass


def design_tree(d):
    k = d["nseg"]
    return build(raw_plan(d["ang"], k, [l[:kk] for l, kk in zip(d["lens"], k)],
                          [r[:kk] for r, kk in zip(d["rads"], k)]))


def fitness_xml(xml, horizon=500, seed=0, dt_control=4, min_torso_height=0.03,
                energy_penalty=0.01, freq=1.5, amp=0.8):
    """gmm.task_fitness.evaluate_fitness, but on a compiled MJCF string (n_act = model.nu).
    Verified identical to the tree version on grammar bodies (see __main__ check)."""
    import mujoco
    try:
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    except Exception:
        return -1e3
    n_act = model.nu
    if n_act == 0:
        return -1e3
    rng = np.random.default_rng(seed)
    phases = np.linspace(0, 2 * np.pi, n_act, endpoint=False) + rng.normal(0, 0.05, n_act)
    x0 = float(data.qpos[0]); energies = []; fell = False; t = 0.0
    try:
        for step in range(horizon):
            if step % dt_control == 0:
                ctrl = amp * np.sin(2 * np.pi * freq * t + phases)
                data.ctrl[:n_act] = np.clip(ctrl, -1, 1); energies.append(float(np.mean(np.abs(ctrl))))
            mujoco.mj_step(model, data); t += model.opt.timestep
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                fell = True; break
            if float(data.qpos[2]) < min_torso_height:
                fell = True
    except Exception:
        return -1e3
    disp = float(data.qpos[0]) - x0
    if abs(disp) > 3.0 or not np.isfinite(disp):
        return -1e3
    return disp - energy_penalty * float(np.mean(energies)) - (0.3 if fell else 0.0)


def final_score(xml):
    """PREREG deviation 1: a failed rollout (-1000 sentinel) counts as -0.3."""
    v = [fitness_xml(xml, horizon=1000, seed=s) for s in range(3)]
    return float(np.mean([-0.3 if f <= -1000 else f for f in v]))


class Run:
    """Holds the budgets, the gate, and the best feasible design found."""

    def __init__(self, sp):
        self.sp = sp; self.p = 0; self.b = 0; self.best = (-np.inf, None); self.trace = []

    def score_xml(self, xml):
        if self.p >= P_BUDGET or self.b >= B_BUDGET:
            raise Budget
        self.p += 1
        m = mjcf_metrics(xml) if xml is not None and _xml_loads_and_steps(xml) else None
        if m is None:
            return -2000.0
        if not feasible(m, self.sp):
            return -1000.0 - violation(m, self.sp)
        self.b += 1
        f = fitness_xml(xml)
        if f > self.best[0]:
            self.best = (f, xml)
        self.trace.append((self.p, self.b, f))
        return f

    def score_design(self, d):
        try:
            xml = COMP.compile(design_tree(d))
        except Exception:
            xml = None
        return self.score_xml(xml)

# ------------------------------------------------------------------ methods

def es(run, x0s, decode, mutate, rng, mu=4, lam=4):
    pop = [(run.score_design(decode(x)), x) for x in x0s]
    while True:
        kids = []
        for _ in range(lam):
            p = pop[rng.integers(len(pop))][1]; c = mutate(p, rng)
            kids.append((run.score_design(decode(c)), c))
        pop = sorted(pop + kids, key=lambda t: -t[0])[:mu]


def es_xml(run, xml0, rng, pr, mu=4, lam=4):
    pop = [(run.score_xml(xml0), xml0)] * mu
    while True:
        kids = []
        for _ in range(lam):
            p = pop[rng.integers(len(pop))][1]
            try:
                c = mutate_range_constrained(p, pr)
            except Exception:
                c = p
            kids.append((run.score_xml(c), c))
        pop = sorted(pop + kids, key=lambda t: -t[0])[:mu]


def cma_run(run, x0, sigma, decode, seed, resample=False):
    es_ = cma.CMAEvolutionStrategy(list(x0), sigma, {"bounds": [0, 1], "seed": seed + 1,
                                                      "verbose": -9, "popsize": 8})
    while True:
        X = es_.ask(); F = []
        for i, x in enumerate(X):
            f = run.score_design(decode(x))
            if resample:
                while f <= -1000.0:            # death penalty with resampling
                    x = es_.ask(1)[0]; X[i] = x; f = run.score_design(decode(x))
            F.append(-f)
        es_.tell(X, F)


def pso_run(run, dim, decode, rng, n=8, w=0.7, c1=1.5, c2=1.5):
    X = rng.random((n, dim)); V = np.zeros((n, dim))
    F = np.array([run.score_design(decode(x)) for x in X]); PB = X.copy(); PF = F.copy()
    g = PB[np.argmax(PF)].copy()
    while True:
        r1, r2 = rng.random((n, dim)), rng.random((n, dim))
        V = w * V + c1 * r1 * (PB - X) + c2 * r2 * (g - X); X = np.clip(X + V, 0, 1)
        for i in range(n):
            f = run.score_design(decode(X[i]))
            if f > PF[i]: PF[i] = f; PB[i] = X[i].copy()
        g = PB[np.argmax(PF)].copy()


def random_run(run, dim, decode, rng):
    while True:
        run.score_design(decode(rng.random(dim)))


def cbo_run(run, dim, decode, rng, batch=20, cand=500):
    """Constrained BO: GP classifier for feasibility, GP regressor for fitness of
    feasible points, acquisition EI x P(feasible) over random candidates."""
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from scipy.stats import norm
    X, y, feas = [], [], []
    for _ in range(batch):
        x = rng.random(dim); f = run.score_design(decode(x)); X.append(x); y.append(f); feas.append(f > -1000)
    while True:
        Xa = np.array(X); fa = np.array(feas); Cn = rng.random((cand, dim))
        if fa.any() and (~fa).any():
            clf = GaussianProcessClassifier(RBF(1.0)).fit(Xa, fa); pf = clf.predict_proba(Cn)[:, 1]
        else:
            pf = np.full(cand, 1.0 if fa.all() else 0.5)
        if fa.sum() >= 2:
            gp = GaussianProcessRegressor(C(1.0) * RBF(1.0), normalize_y=True).fit(Xa[fa], np.array(y)[fa])
            mu_, sd = gp.predict(Cn, return_std=True); best = np.array(y)[fa].max()
            z = (mu_ - best) / np.maximum(sd, 1e-9); ei = (mu_ - best) * norm.cdf(z) + sd * norm.pdf(z)
            acq = np.maximum(ei, 1e-12) * pf
        else:
            acq = pf + 1e-6 * rng.random(cand)
        for j in np.argsort(-acq)[:batch]:
            f = run.score_design(decode(Cn[j])); X.append(Cn[j]); y.append(f); feas.append(f > -1000)


def g4_from_design(d):
    k = d["nseg"][0]; L = d["lens"][0][0]; R = d["rads"][0][0]
    return np.clip(np.array([(k - 1) / 2, (L - LEN[0]) / (LEN[1] - LEN[0]),
                             (R - RAD[0]) / (RAD[1] - RAD[0]), 0.0]), 0, 1)


def mut_g4(x, rng, sigma=0.1):
    y = x.copy(); j = rng.integers(len(y)); y[j] = np.clip(y[j] + rng.normal(0, sigma), 0, 1); return y


def one(method, mode, sp, seed):
    rng = np.random.default_rng(seed); pr = random.Random(seed); run = Run(sp)
    dec_f = lambda x: flat_to_design(x, N); dec_g = lambda x: grammar4_to_design(x, N)
    try:
        if mode == "scratch":
            if method == "grammar_es": es(run, [rng.random(4) for _ in range(4)], dec_g, mut_g4, rng)
            elif method == "flat_cma": cma_run(run, rng.random(8 * N), 0.3, dec_f, seed)
            elif method == "flat_cma_resample": cma_run(run, rng.random(8 * N), 0.3, dec_f, seed, resample=True)
            elif method == "flat_pso": pso_run(run, 8 * N, dec_f, rng)
            elif method == "flat_random": random_run(run, 8 * N, dec_f, rng)
            elif method == "flat_cbo": cbo_run(run, 8 * N, dec_f, rng)
        else:
            d0, _ = feasible_parent(N, sp, np.random.default_rng(10_000 + seed))
            if method == "grammar_es": es(run, [g4_from_design(d0)] * 4, dec_g, mut_g4, rng)
            elif method == "flat_clip_es": es(run, [design_to_flat(d0)] * 4, dec_f, mut_flat_clip, rng)
            elif method == "range_xml_es": es_xml(run, COMP.compile(design_tree(d0)), rng, pr)
            elif method == "flat_cma": cma_run(run, design_to_flat(d0), 0.05, dec_f, seed)
            elif method == "flat_cma_s001": cma_run(run, design_to_flat(d0), 0.01, dec_f, seed)
    except Budget:
        pass
    best = final_score(run.best[1]) if run.best[1] is not None else None
    return dict(best=best, search_best=None if run.best[1] is None else float(run.best[0]),
                proposals=run.p, physics=run.b)


SCRATCH = ["grammar_es", "flat_cma", "flat_pso", "flat_random", "flat_cma_resample", "flat_cbo"]
SEEDED = ["grammar_es", "flat_clip_es", "range_xml_es", "flat_cma", "flat_cma_s001"]

if __name__ == "__main__":
    mode = sys.argv[1]; levels = sys.argv[2].split(","); seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    methods = SCRATCH if mode == "scratch" else SEEDED
    if len(sys.argv) > 4: methods = sys.argv[4].split(",")
    cal = json.load(open("results/cal_A_8.json")); out = {}
    for tg in levels:
        sp = spec_tau(N, cal[tg]["tau"]); out[tg] = {}
        for m in methods:
            t0 = time.time(); rs = [one(m, mode, sp, s) for s in range(seeds)]
            out[tg][m] = rs
            b = [r["best"] for r in rs if r["best"] is not None]
            print(f"{mode} {tg} {m:18s} found {len(b)}/{seeds}  best-fitness mean "
                  f"{np.mean(b) if b else float('nan'):.3f}  physics used "
                  f"{np.mean([r['physics'] for r in rs]):.1f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(f"results/exp1_{mode}_{'_'.join(levels)}_{'-'.join(methods) if len(sys.argv) > 4 else 'all'}.json", "w"), indent=1)

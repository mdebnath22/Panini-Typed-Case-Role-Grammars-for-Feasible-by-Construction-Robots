"""
Four reviewer objections, each tested in the hard, jointly constrained design space
(experiments/hard_coupled_design_space.py; difficulty levels from
results/cal_A_<n>.json, calibrated on the pooled pass rate of all samplers).

E1  "Once a valid body exists, per-field np.clip mutation suffices."
    Every method starts from the SAME feasible body. (a) Random-walk retention:
    feasibility after k unselected mutations, k = 1..50. (b) Budgeted search
    (200 evals, 20 seeds): minimise body mass subject to joint feasibility.
E2  "The 5-parameter genome is too simple; the boundary is separable."
    (a) Separability test: fit the tightest per-field interval box around known
    feasible designs, sample uniformly inside it, measure feasibility. Done in the
    original 5-D genome and in the flat 8n-D space (with and without the rotation
    symmetry every legged body has). (b) A richer grammar genome (mirror-pair
    lengths/radii, 12-D at n=8) to check the advantage is not an artefact of a
    tiny grammar genome.
E3  "Hand-coded editors match the grammar."
    Edit suite on a feasible parent: add limb, remove limb, add segment, lengthen.
    Structure-aware hand-coded editors vs. hand-coded + re-space (reimplements the
    grammar's layout rule) vs. grammar re-derivation. Lines of code reported.
E4  "The gate is an accept/reject check, not the grammar."
    2x2 factorial generator {flat, grammar} x gate {off, on}; count draws and
    physics calls per feasible design.
"""
from __future__ import annotations
import inspect, json, math, sys, time
import numpy as np
import cma
from experiments.hard_coupled_design_space import (
    raw_plan, build, metrics, feasible, sample_grammar, LEN, RAD)
from experiments.hard_coupled_calibrated import spec_tau
from experiments.hard_coupled_final import violation
from ptg.body_grammar import legged_plan

# ---------------------------------------------------------------- representations
# A "design" is the flat, grammar-free description: per-limb angle, segment count,
# per-segment lengths and radii. Normalised flat vector: 8 numbers per limb in [0,1].

def flat_to_design(x, n):
    x = np.clip(np.asarray(x, float), 0, 1)
    d = dict(ang=[], nseg=[], lens=[], rads=[])
    for i in range(n):
        b = x[i * 8:(i + 1) * 8]
        d["ang"].append(b[0] * 2 * math.pi)
        d["nseg"].append(int(np.clip(round(1 + b[1] * 2), 1, 3)))
        d["lens"].append(list(LEN[0] + b[2:5] * (LEN[1] - LEN[0])))
        d["rads"].append(list(RAD[0] + b[5:8] * (RAD[1] - RAD[0])))
    return d


def design_to_flat(d):
    out = []
    for i in range(len(d["ang"])):
        L = list(d["lens"][i]) + [d["lens"][i][-1]] * (3 - len(d["lens"][i]))
        R = list(d["rads"][i]) + [d["rads"][i][-1]] * (3 - len(d["rads"][i]))
        out += [(d["ang"][i] % (2 * math.pi)) / (2 * math.pi), (d["nseg"][i] - 1) / 2]
        out += [(l - LEN[0]) / (LEN[1] - LEN[0]) for l in L[:3]]
        out += [(r - RAD[0]) / (RAD[1] - RAD[0]) for r in R[:3]]
    return np.clip(np.array(out), 0, 1)


def design_metrics(d):
    k = d["nseg"]
    return metrics(build(raw_plan(d["ang"], k, [l[:kk] for l, kk in zip(d["lens"], k)],
                                  [r[:kk] for r, kk in zip(d["rads"], k)])))

# Rich grammar genome (E2b, E1): layout rule (even spacing + mirrored jitter),
# mirror rule (limb i and limb n-i share geometry), inheritance within a limb
# (all segments of a limb share one length/radius), one global segment count.
# Normalised vector: [k, jitter, L_g..., R_g...] with G = n//2 + 1 mirror groups.

def n_groups(n):
    return n // 2 + 1


def group_of(i, n):
    return min(i, n - i) if i else 0


def rich_to_design(x, n):
    x = np.clip(np.asarray(x, float), 0, 1)
    G = n_groups(n)
    k = int(np.clip(round(1 + x[0] * 2), 1, 3)); jit = x[1] * 0.3
    L = LEN[0] + x[2:2 + G] * (LEN[1] - LEN[0]); R = RAD[0] + x[2 + G:2 + 2 * G] * (RAD[1] - RAD[0])
    base = np.linspace(0, 2 * math.pi, n, endpoint=False)
    # mirrored jitter keeps the layout symmetric about the x-axis
    jv = np.array([jit * math.sin(3.0 * min(i, n - i)) if i else 0.0 for i in range(n)])
    jv = np.array([jv[i] if i <= n // 2 else -jv[n - i] for i in range(n)])
    d = dict(ang=list(base + jv), nseg=[k] * n, lens=[], rads=[])
    for i in range(n):
        g = group_of(i, n); d["lens"].append([L[g]] * 3); d["rads"].append([R[g]] * 3)
    return d


def rich_dim(n):
    return 2 + 2 * n_groups(n)


def grammar4_to_design(x, n):
    """The paper's original grammar genome (k, seg_len, seg_rad, jitter), n fixed."""
    x = np.clip(np.asarray(x, float), 0, 1)
    k = int(np.clip(round(1 + x[0] * 2), 1, 3)); L = LEN[0] + x[1] * (LEN[1] - LEN[0])
    R = RAD[0] + x[2] * (RAD[1] - RAD[0]); jit = x[3] * 0.3
    rr = np.random.default_rng(int(1e6 * x[3]) % 2 ** 31)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False) + rr.uniform(-jit, jit, n)
    return dict(ang=list(ang), nseg=[k] * n, lens=[[L] * 3] * n, rads=[[R] * 3] * n)


def design_to_rich(d, n):
    """Map a symmetric grammar design (shared L,R, zero jitter) into the rich genome."""
    G = n_groups(n); k = d["nseg"][0]
    L = (d["lens"][0][0] - LEN[0]) / (LEN[1] - LEN[0]); R = (d["rads"][0][0] - RAD[0]) / (RAD[1] - RAD[0])
    return np.clip(np.array([(k - 1) / 2, 0.0] + [L] * G + [R] * G), 0, 1)


def mass_of(m):
    return m["mass"]

# ---------------------------------------------------------------- feasible parents

def feasible_parent(n, sp, rng, max_draws=200000):
    """Feasible, perfectly symmetric grammar body (zero jitter) -- the same parent is
    handed to every method, so no method starts with an advantage."""
    for t in range(max_draws):
        k = int(rng.integers(1, 4)); L = float(rng.uniform(*LEN)); R = float(rng.uniform(*RAD))
        d = dict(ang=list(np.linspace(0, 2 * math.pi, n, endpoint=False)), nseg=[k] * n,
                 lens=[[L] * 3] * n, rads=[[R] * 3] * n)
        m = design_metrics(d)
        if feasible(m, sp):
            return d, m
    raise RuntimeError("no feasible parent")

# ---------------------------------------------------------------- mutation operators

def mut_flat_clip(x, rng, sigma=0.1):
    """Range-constrained mutation: perturb ONE field, np.clip to its own range."""
    y = x.copy(); j = rng.integers(len(y)); y[j] = np.clip(y[j] + rng.normal(0, sigma), 0, 1); return y


def mut_flat_resample(x, rng):
    """Range-constrained resampling: redraw ONE field uniformly in its own range."""
    y = x.copy(); j = rng.integers(len(y)); y[j] = rng.random(); return y


def mut_rich(x, rng, sigma=0.1):
    """Grammar mutation: perturb ONE grammar parameter (also clipped), re-derive."""
    y = x.copy(); j = rng.integers(len(y)); y[j] = np.clip(y[j] + rng.normal(0, sigma), 0, 1); return y

# ---------------------------------------------------------------- E1

def e1_walks(n, sp, rng, n_walks=150, K=50):
    ks = [1, 2, 5, 10, 20, 50]
    res = {nm: np.zeros(K) for nm in ["flat_clip", "flat_resample", "grammar"]}
    for w in range(n_walks):
        d, _ = feasible_parent(n, sp, rng)
        xf = design_to_flat(d); xr = design_to_rich(d, n)
        a, b, c = xf.copy(), xf.copy(), xr.copy()
        for k in range(K):
            a = mut_flat_clip(a, rng); b = mut_flat_resample(b, rng); c = mut_rich(c, rng)
            res["flat_clip"][k] += feasible(design_metrics(flat_to_design(a, n)), sp)
            res["flat_resample"][k] += feasible(design_metrics(flat_to_design(b, n)), sp)
            res["grammar"][k] += feasible(design_metrics(rich_to_design(c, n)), sp)
    return {nm: {str(k): float(v[k - 1] / n_walks) for k in ks} for nm, v in res.items()}


def material(d):
    """Total leg material, sum of pi r^2 l over every segment actually built (continuous;
    body mass in this simulator depends only on segment count, so it is not used)."""
    return float(sum(math.pi * r * r * l for L, R, k in zip(d["lens"], d["rads"], d["nseg"])
                     for l, r in zip(L[:k], R[:k])))


def penalised_mass(m, sp, d=None):
    obj = material(d) * 1e3 if d is not None else m["mass"]
    return obj + (0.0 if feasible(m, sp) else 100.0 * (1.0 + violation(m, sp)))


def mu_plus_lambda(x0, decode, mutate, n, sp, rng, budget=200, mu=4, lam=4):
    d0 = decode(x0, n); pop = [(penalised_mass(design_metrics(d0), sp, d0), x0.copy())] * mu
    used = 1; feas = 1; best = pop[0][0]
    while used < budget:
        kids = []
        for _ in range(lam):
            if used >= budget: break
            p = pop[rng.integers(len(pop))][1]; c = mutate(p, rng)
            dc = decode(c, n); m = design_metrics(dc); f = penalised_mass(m, sp, dc); used += 1
            ok = feasible(m, sp); feas += ok
            if ok: best = min(best, f)
            kids.append((f, c))
        pop = sorted(pop + kids, key=lambda t: t[0])[:mu]
    return feas / budget, best


def cma_from(x0, decode, n, sp, seed, budget=200, sigma=0.1):
    es = cma.CMAEvolutionStrategy(list(x0), sigma, {"bounds": [0, 1], "seed": seed + 1, "verbose": -9,
                                                   "popsize": 8})
    d0 = decode(x0, n); m0 = design_metrics(d0); best = penalised_mass(m0, sp, d0); used = 1; feas = 1
    while used < budget and not es.stop():
        X = es.ask(); F = []
        for x in X:
            if used >= budget: F.append(1e9); continue
            dx = decode(x, n); m = design_metrics(dx); f = penalised_mass(m, sp, dx); used += 1
            ok = feasible(m, sp); feas += ok
            if ok: best = min(best, f)
            F.append(f)
        es.tell(X, F)
    return feas / budget, best


def e1_search(n, sp, seeds=20, budget=200):
    out = {k: dict(feas=[], best_rel=[]) for k in
           ["flat_clip_es", "flat_resample_es", "flat_cma", "grammar_es", "grammar_cma"]}
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        d, m = feasible_parent(n, sp, rng); m0 = material(d) * 1e3
        xf = design_to_flat(d); xr = design_to_rich(d, n)
        runs = {
            "flat_clip_es": lambda: mu_plus_lambda(xf, flat_to_design, mut_flat_clip, n, sp, np.random.default_rng(s)),
            "flat_resample_es": lambda: mu_plus_lambda(xf, flat_to_design, mut_flat_resample, n, sp, np.random.default_rng(s)),
            "flat_cma": lambda: cma_from(xf, flat_to_design, n, sp, s),
            "grammar_es": lambda: mu_plus_lambda(xr, rich_to_design, mut_rich, n, sp, np.random.default_rng(s)),
            "grammar_cma": lambda: cma_from(xr, rich_to_design, n, sp, s),
        }
        for nm, fn in runs.items():
            fr, best = fn(); out[nm]["feas"].append(fr); out[nm]["best_rel"].append(best / m0)
    return out

# ---------------------------------------------------------------- E2

def box_test(points, sample_fn, check_fn, rng, N=3000):
    P = np.array(points); lo, hi = P.min(0), P.max(0)
    ok = 0
    for _ in range(N):
        ok += check_fn(sample_fn(lo + rng.random(len(lo)) * (hi - lo)))
    return ok / N, float(np.prod(np.maximum(hi - lo, 1e-12)))


def e2_separability(n, sp, rng, n_feas=400, N=3000):
    # flat space: feasible designs = feasible grammar bodies (incl. jitter variants)
    pts, pts_rot = [], []
    while len(pts) < n_feas:
        x = rng.random(4); d = grammar4_to_design(x, n); m = design_metrics(d)
        if feasible(m, sp):
            pts.append(design_to_flat(d))
            phi = rng.uniform(0, 2 * math.pi)   # rotating a body never changes feasibility
            dr = dict(d); dr["ang"] = [a + phi for a in d["ang"]]
            assert feasible(design_metrics(dr), sp)
            pts_rot.append(design_to_flat(dr))
    chk = lambda x: feasible(design_metrics(flat_to_design(x, n)), sp)
    r_fixed, _ = box_test(pts, lambda x: x, chk, rng, N)
    r_rot, _ = box_test(pts_rot, lambda x: x, chk, rng, N)
    return dict(flat_box_fixed_orientation=r_fixed, flat_box_rotation_augmented=r_rot,
                n_feasible_points=n_feas)


def e2_separability_5d(rng, tau, N=3000, n_feas=400):
    """The paper's original 5-D genome (n_legs free in [2,8]), same family of joint
    constraints at the same tau (spec rescaled to each body's own n)."""
    def dec(v):
        nl = int(np.clip(round(2 + v[0] * 6), 2, 8)); k = int(np.clip(round(1 + v[1] * 2), 1, 3))
        L = LEN[0] + v[2] * (LEN[1] - LEN[0]); R = RAD[0] + v[3] * (RAD[1] - RAD[0]); jit = v[4] * 0.3
        rr = np.random.default_rng(int(1e6 * v[4]) % 2 ** 31)
        ang = np.linspace(0, 2 * math.pi, nl, endpoint=False) + rr.uniform(-jit, jit, nl)
        return nl, dict(ang=list(ang), nseg=[k] * nl, lens=[[L] * 3] * nl, rads=[[R] * 3] * nl)
    def chk(v):
        nl, d = dec(v); return feasible(design_metrics(d), spec_tau(nl, tau))
    pts = []; draws = 0
    while len(pts) < n_feas and draws < 200000:
        v = rng.random(5); draws += 1
        if chk(v): pts.append(v)
    base = len(pts) / draws
    r, _ = box_test(pts, lambda v: v, chk, rng, N)
    return dict(genome5d_uniform=base, genome5d_box=r)


def e2_rich_sampling(n, sp, rng, N=6000):
    dims = rich_dim(n)
    ok_rich = sum(feasible(design_metrics(rich_to_design(rng.random(dims), n)), sp) for _ in range(N))
    return dict(rich_dim=dims, rich_uniform=ok_rich / N)


def e2_rich_cma(n, sp, seeds=20, budget=200):
    hits, fr = [], []
    for s in range(seeds):
        es = cma.CMAEvolutionStrategy([0.5] * rich_dim(n), 0.3, {"bounds": [0, 1], "seed": s + 1,
                                                                 "verbose": -9, "popsize": 8})
        used = 0; feas = 0
        while used < budget and not es.stop():
            X = es.ask(); F = []
            for x in X:
                if used >= budget: F.append(9.0); continue
                m = design_metrics(rich_to_design(x, n)); used += 1; ok = feasible(m, sp); feas += ok
                F.append(violation(m, sp))
            es.tell(X, F)
        hits.append(feas > 0); fr.append(feas / budget)
    return dict(seeds_any=int(sum(hits)), feas_frac=float(np.mean(fr)))

# ---------------------------------------------------------------- E3 editors

def ed_add_limb_careful(d, rng):
    """Hand-coded, structure-aware: new limb in the middle of the largest angular gap,
    geometry copied from its neighbour; every other limb untouched."""
    a = np.mod(np.array(d["ang"]), 2 * math.pi); o = np.argsort(a); s = a[o]
    gaps = np.diff(np.r_[s, s[0] + 2 * math.pi]); g = int(np.argmax(gaps))
    new = (s[g] + gaps[g] / 2) % (2 * math.pi); j = int(o[g])
    e = {k: list(v) for k, v in d.items()}
    e["ang"].append(new); e["nseg"].append(d["nseg"][j])
    e["lens"].append(list(d["lens"][j])); e["rads"].append(list(d["rads"][j]))
    return e


def ed_remove_limb_careful(d, rng):
    """Hand-coded: delete one limb; every other limb untouched."""
    j = int(rng.integers(len(d["ang"])))
    return {k: [v for i, v in enumerate(vv) if i != j] for k, vv in d.items()}


def ed_add_segment_careful(d, rng):
    """Hand-coded: add one segment to one limb, copying its last segment."""
    e = {k: [list(x) if isinstance(x, list) else x for x in v] for k, v in d.items()}
    cand = [i for i, k in enumerate(e["nseg"]) if k < 3]
    if not cand: return e
    j = int(rng.choice(cand)); e["nseg"][j] += 1
    return e


def ed_lengthen_careful(d, rng):
    """Hand-coded: lengthen one limb's segments by U(1.1,1.4), clipped to range."""
    e = {k: [list(x) if isinstance(x, list) else x for x in v] for k, v in d.items()}
    j = int(rng.integers(len(e["ang"]))); f = rng.uniform(1.1, 1.4)
    e["lens"][j] = [float(np.clip(l * f, *LEN)) for l in e["lens"][j]]
    return e


def respace(e):
    """Hand-coded re-spacing: reimplements the grammar's layout rule (linspace)."""
    e = {k: list(v) for k, v in e.items()}
    o = np.argsort(np.mod(np.array(e["ang"]), 2 * math.pi))
    for k in e: e[k] = [e[k][i] for i in o]
    e["ang"] = list(np.linspace(0, 2 * math.pi, len(e["ang"]), endpoint=False))
    return e


def ed_add_limb_respace(d, rng):
    return respace(ed_add_limb_careful(d, rng))


def ed_remove_limb_respace(d, rng):
    return respace(ed_remove_limb_careful(d, rng))


def gr_edit(d, kind, rng):
    """Grammar edit: change one grammar parameter and re-derive the whole body."""
    n = len(d["ang"]); k = d["nseg"][0]; L = d["lens"][0][0]; R = d["rads"][0][0]
    if kind == "add_limb": n += 1
    elif kind == "remove_limb": n -= 1
    elif kind == "add_segment": k = min(3, k + 1)
    elif kind == "lengthen": L = float(np.clip(L * rng.uniform(1.1, 1.4), *LEN))
    return dict(ang=list(np.linspace(0, 2 * math.pi, n, endpoint=False)), nseg=[k] * n,
                lens=[[L] * 3] * n, rads=[[R] * 3] * n)


def e3_editors(n, tau, rng, trials=400):
    eds = {
        "add_limb": [("careful", ed_add_limb_careful), ("careful+respace", ed_add_limb_respace)],
        "remove_limb": [("careful", ed_remove_limb_careful), ("careful+respace", ed_remove_limb_respace)],
        "add_segment": [("careful", ed_add_segment_careful)],
        "lengthen": [("careful", ed_lengthen_careful)],
    }
    sp = spec_tau(n, tau); out = {}
    for kind, lst in eds.items():
        cnt = {nm: 0 for nm, _ in lst}; cnt["grammar"] = 0
        for t in range(trials):
            d, _ = feasible_parent(n, sp, rng)
            for nm, fn in lst:
                e = fn(d, rng); cnt[nm] += feasible(design_metrics(e), spec_tau(len(e["ang"]), tau))
            e = gr_edit(d, kind, rng); cnt["grammar"] += feasible(design_metrics(e), spec_tau(len(e["ang"]), tau))
        out[kind] = {k: v / trials for k, v in cnt.items()}
    loc = lambda f: len([l for l in inspect.getsource(f).splitlines() if l.strip() and not l.strip().startswith(('"""', '#'))])
    out["loc"] = {f.__name__: loc(f) for f in [ed_add_limb_careful, ed_remove_limb_careful,
                                                ed_add_segment_careful, ed_lengthen_careful, respace]}
    return out

# ---------------------------------------------------------------- E4

def e4_factorial(n, sp, rng, D=6000):
    out = {}
    for gen in ["flat", "grammar"]:
        feas = 0
        for _ in range(D):
            if gen == "flat":
                d = flat_to_design(rng.random(8 * n), n)
            else:
                d = grammar4_to_design(rng.random(4), n)
            feas += feasible(design_metrics(d), sp)
        for gate in ["off", "on"]:
            physics = D if gate == "off" else feas
            out[f"{gen}|gate_{gate}"] = dict(draws=D, physics_calls=physics, feasible=feas,
                                            physics_per_feasible=(physics / feas) if feas else None,
                                            draws_per_feasible=(D / feas) if feas else None)
    return out

# ---------------------------------------------------------------- main

if __name__ == "__main__":
    which = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cal = json.load(open(f"results/cal_A_{n}.json")); taus = {k: v["tau"] for k, v in cal.items()}
    res = {}; t0 = time.time()
    for tg, tau in taus.items():
        sp = spec_tau(n, tau); rng = np.random.default_rng(7)
        if which == "E1walk": res[tg] = e1_walks(n, sp, rng)
        elif which == "E1search": res[tg] = e1_search(n, sp)
        elif which == "E2":
            r = e2_separability(n, sp, rng); r.update(e2_separability_5d(rng, tau))
            r.update(e2_rich_sampling(n, sp, rng)); r.update(e2_rich_cma(n, sp)); res[tg] = r
        elif which == "E3": res[tg] = e3_editors(n, tau, rng)
        elif which == "E4": res[tg] = e4_factorial(n, sp, rng)
        print(which, n, tg, f"{time.time() - t0:.0f}s", json.dumps(res[tg], default=float)[:600], flush=True)
    json.dump(res, open(f"results/obj_{which}_{n}.json", "w"), default=float, indent=1)

"""Hard coupled design space, final protocol (a-priori, n-scaled thresholds).
Part A: from-scratch sampling feasibility.  Part B: budgeted search."""
import numpy as np, json, sys, math
from scipy import stats
from experiments.hard_coupled_design_space import *

TIERS = {  # (area, stab, gap_mult, sep_scale, mass_slack)
 "moderate": dict(area=0.50, stab=0.15, gm=2.2, ss=0.6, ms=0.55),
 "strict":   dict(area=0.70, stab=0.25, gm=1.8, ss=1.0, ms=0.40),
 "severe":   dict(area=0.80, stab=0.30, gm=1.5, ss=1.4, ms=0.30),
}
def spec_for(n, tier):
    t = TIERS[tier]
    return dict(area=t["area"], stab=t["stab"], gap=t["gm"]*2*math.pi/n,
                sep=t["ss"]/n, mass=2.0 + t["ms"]*n)
def violation(mt, sp):
    v = 0.0
    v += max(0, sp["area"]-mt["area"])/sp["area"]
    v += max(0, sp["stab"]-mt["stab"])/sp["stab"]
    v += max(0, mt["gap"]-sp["gap"])/sp["gap"]
    v += max(0, sp["sep"]-mt["sep"])/sp["sep"]
    v += max(0, mt["mass"]-sp["mass"])/sp["mass"]
    return v
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return (c-h)/d, (c+h)/d

def partA(ns=(6,8,10), N=3000, seed=0):
    rng = np.random.default_rng(seed); out = {}
    for n in ns:
        M = {nm: [metrics(fn(n, rng)) for _ in range(N)] for nm, fn in
             [("raw", sample_raw), ("grammar", sample_grammar), ("raw_sym", sample_raw_sym)]}
        for tier in TIERS:
            sp = spec_for(n, tier); row = {}
            for nm, ms in M.items():
                row[nm] = sum(feasible(m, sp) for m in ms)
            out[f"{n}|{tier}"] = row
            lo, hi = wilson(row["raw"], N)
            print(f"n={n:2d} {tier:9s} raw {row['raw']:4d}/{N} [{100*lo:.2f},{100*hi:.2f}]%  "
                  f"grammar {row['grammar']:4d}/{N}  raw_sym {row['raw_sym']:4d}/{N}", flush=True)
    return out

# ---- Part B: budgeted search (CMA-ES on continuous relaxation; violation objective) ----
import cma
from gmm.phi_body import leg_mount_plan
def raw_decode(x, n):
    x = np.asarray(x); lens=[]; rads=[]; nseg=[]; ang=[]
    for i in range(n):
        b = x[i*8:(i+1)*8]
        ang.append(b[0]*2*math.pi); nseg.append(int(np.clip(round(1+b[1]*2),1,3)))
        lens.append(LEN[0]+np.clip(b[2:5],0,1)*(LEN[1]-LEN[0]))
        rads.append(RAD[0]+np.clip(b[5:8],0,1)*(RAD[1]-RAD[0]))
    # same longitudinal-offset treatment as sample_raw: coordinated x, free angle.
    _, x_offsets = leg_mount_plan(n)
    return build(raw_plan(ang, nseg, lens, rads, x_offsets))
def gram_decode(x, n):
    k = int(np.clip(round(1+x[0]*2),1,3)); L = LEN[0]+np.clip(x[1],0,1)*(LEN[1]-LEN[0])
    R = RAD[0]+np.clip(x[2],0,1)*(RAD[1]-RAD[0]); jit = np.clip(x[3],0,1)*0.3
    # base angles/x now come from leg_mount_plan (lower-hemisphere row pairs),
    # not a free 0..2pi circle -- jitter perturbs around that coordinated base,
    # same convention as every other grammar-side call site. NOTE: passing
    # mount_angles without mount_x_offsets to legged_plan makes it recompute
    # BOTH from scratch (silently discarding the jitter), so x_offsets must be
    # passed explicitly too.
    #
    # BUG FIX (2026-10-24): the jitter RNG used to be seeded from
    # int(1e6*abs(x[3])) -- i.e. from the jitter MAGNITUDE itself, the very
    # search coordinate CMA-ES is trying to locally vary. Any infinitesimal
    # step in x[3] (even one that leaves `jit` numerically unchanged after
    # clipping/rounding) re-seeds the RNG and resamples a brand-new random
    # jitter DIRECTION, so the objective is discontinuous in x[3] everywhere
    # -- the same antipattern fixed in m_search_loop.Genome.mount_plan().
    # Fixed the same way: the jitter direction is now seeded only from `n`
    # (a structural property that CMA-ES never varies within a single call),
    # and jit (x[3]'s decoded magnitude) scales that fixed unit direction
    # continuously, so jit=0 means exactly "no jitter" and small changes in
    # x[3] produce small, smooth changes in the mount angles.
    base, x_offsets = leg_mount_plan(n)
    rr = np.random.default_rng(int(n) * 1_000_003 + 17)
    unit_dir = rr.uniform(-1.0, 1.0, n)
    ang = np.asarray(base) + unit_dir * jit
    return build(legged_plan(n_legs=n, segments_per_leg=k, mount_angles=list(ang),
                             mount_x_offsets=list(x_offsets), seg_len=float(L), seg_rad=float(R)))
def search(method, n, tier, budget, seed):
    sp = spec_for(n, tier); rng = np.random.default_rng(seed)
    if method == "random_raw":
        dim = 8*n; ask = lambda: rng.random(dim); dec = raw_decode
    elif method == "random_grammar":
        dim = 4; ask = lambda: rng.random(dim); dec = gram_decode
    if method.startswith("random"):
        feas = []; first = None
        for e in range(budget):
            ok = feasible(metrics(dec(ask(), n)), sp); feas.append(ok)
            if ok and first is None: first = e+1
        return sum(feas), first
    dim = 8*n if method == "cma_raw" else 4; dec = raw_decode if method == "cma_raw" else gram_decode
    es = cma.CMAEvolutionStrategy([0.5]*dim, 0.3, {"bounds":[0,1],"seed":seed+1,"verbose":-9,
                                  "popsize": 8, "maxfevals": budget})
    feas = 0; first = None; used = 0
    while used < budget and not es.stop():
        X = es.ask(); F = []
        for x in X:
            if used >= budget: F.append(9.0); continue
            mt = metrics(dec(x, n)); ok = feasible(mt, sp); used += 1
            feas += ok
            if ok and first is None: first = used
            F.append(violation(mt, sp))
        es.tell(X, F)
    return feas, first
def partB(n=8, tiers=("moderate","strict","severe"), budget=200, seeds=20):
    methods = ["random_raw","cma_raw","random_grammar","cma_grammar"]; out = {}
    for tier in tiers:
        for m in methods:
            r = [search(m, n, tier, budget, s) for s in range(seeds)]
            f = np.array([a for a,_ in r]); fr = [b for _,b in r]
            out[f"{tier}|{m}"] = dict(feasible_evals=f.tolist(), first=fr)
            hit = sum(b is not None for b in fr)
            print(f"{tier:9s} {m:15s} feasible-evals {f.mean():6.1f}/{budget} ({100*f.mean()/budget:5.1f}%)"
                  f"  found-any {hit}/{seeds}  median-first {np.median([b for b in fr if b]) if hit else float('nan')}", flush=True)
    return out
if __name__ == "__main__":
    which = sys.argv[1]
    res = partA() if which == "A" else partB()
    json.dump(res, open(f"/tmp/hard_{which}.json","w"))

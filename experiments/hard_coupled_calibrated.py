"""Pooled-calibrated difficulty: one scalar knob tau tightens ALL thresholds; tau is
bisected so the POOLED pass rate over {raw, grammar, raw_sym} samples hits a target.
No threshold is tuned to any single method."""
import numpy as np, math, json, sys
from experiments.hard_coupled_design_space import *
from experiments.hard_coupled_final import wilson, violation, raw_decode, gram_decode, search
import experiments.hard_coupled_final as hf

def spec_tau(n, tau):
    lerp = lambda a, b: a + (b-a)*tau
    # mass budget held fixed (it is a per-limb-count cliff, not a coupling constraint);
    # tau tightens only the continuous constraints that couple limbs
    #
    # AREA/STAB/SEP RESCALE (post mounting+EXTEND geometry fix): all three
    # endpoints below were calibrated against the pre-fix geometry's much
    # larger achievable scale (grammar bodies at n=8 were still passing
    # feasibility at area>=2.74 at the most extreme calibrated tau=2.04
    # used historically, see results/cal_A_8.json.pre_fix_bak). Under the
    # corrected geometry (legs restricted to the lower hemisphere with real
    # front/middle/back longitudinal spread), fresh measurement (n=900
    # pooled across raw/grammar/raw_sym samplers, n_legs=8) gives:
    #   area: min=0.023 max=0.188 mean=0.090
    #   stab: max=0.090 (grammar/raw_sym: EXACTLY 0.090, no variance at all
    #         -- see note below)
    #   sep:  max=0.303 (grammar/raw_sym: EXACTLY 0.060, no variance at all)
    # Each endpoint pair is rescaled by the same factor (new pooled max /
    # old endpoint b), preserving the original a:b ratio and the original
    # tau->value mapping shape, so the SAME tau values historically needed
    # to hit a given pooled target (tau ~1.4-2.0) land on a proportionally
    # equivalent point of the new achievable range instead of being
    # recalibrated from scratch:
    #   area: *0.1177 (0.188/1.60)  ->  (0.0589, 0.1883)
    #   stab: *0.15   (0.090/0.60) ->  (0.0225, 0.0900)
    #   sep:  *0.1011 (0.303/3.00) ->  (0.0607, 0.3034)
    # gap/mass are left as-is (angular gap and total mass budget are not
    # driven by the longitudinal x-offset/EXTEND changes).
    #
    # LIMITATION, not a bug: for the grammar and raw_sym samplers specifically
    # (which give every leg on a body the SAME length/radius), stab and sep
    # come out as an exact constant across resampling, independent of leg
    # length -- the binding term is the fixed row-to-row longitudinal
    # spacing (torso_len*x_span_frac/(n_rows-1), e.g. 0.06 at n_legs=8), not
    # anything length-dependent. So for grammar/raw_sym, whether a given tau
    # passes stab/sep is a step function of n alone, not a smoothly
    # calibratable quantity; only `raw` (independent per-limb lengths)
    # avoids this degeneracy. This was already true before any of today's
    # fixes -- it was simply swamped by a much larger absolute scale, so it
    # never bound. It is reported here as-is, per design, rather than
    # smoothed over by changing x_span_frac/row spacing.
    # GAP REDEFINITION (user-directed fix, not just a rescale): the old `gap`
    # was the max angular gap between tip angles measured from the fixed
    # BODY ORIGIN, normalized as a multiple of the "ideal even spacing"
    # 2*pi/n -- a sound proxy only when legs are spread all the way around
    # that origin. Under lower-hemisphere, row-based mounting, legs occupy a
    # narrow angular band by design (nothing is ever mounted "above" the
    # torso), so that normalization made the metric spuriously fail every
    # grammar/raw_sym body (0/N joint-feasible at every tau -- see
    # results/cal_A_8.json.pre_fix_bak-era runs). `_support_gap` in
    # hard_coupled_design_space.py now measures the same "largest undefended
    # arc" quantity from the support footprint's OWN convex-hull centroid
    # instead, which is shape-agnostic (circular or rectangular) and keeps
    # the same radian units/[0,2*pi] scale -- but its *typical magnitude* is
    # a different, roughly n-INDEPENDENT quantity now (fresh pooled
    # raw/grammar/raw_sym measurement, n=3..12, 1800 samples/n: median
    # ~2.3-2.7 rad across the whole range, NOT ~2*pi/n), so the old
    # "multiple of 2*pi/n" formula no longer means anything and is replaced
    # by a direct empirical percentile lerp against the new metric's own
    # pooled distribution (n=8/10, N=4500 each): p90=2.70 (loose/tau=0,
    # ~90% pass on gap alone) to p20=2.06 (tight/tau=1, ~20% pass on gap
    # alone), the same easy<->hard interpolation philosophy as area/stab/sep.
    # (n_legs<3 still degenerates to gap=2*pi, i.e. always infeasible --
    # unchanged, pre-existing, and correct: you cannot have a real support
    # polygon with fewer than 3 contact points.)
    _AREA_SCALE, _STAB_SCALE, _SEP_SCALE = 0.1177, 0.15, 0.1011
    return dict(area=lerp(0.50 * _AREA_SCALE, 1.60 * _AREA_SCALE),
                stab=lerp(0.15 * _STAB_SCALE, 0.60 * _STAB_SCALE),
                gap=lerp(2.70, 2.06),
                sep=lerp(0.6 * _SEP_SCALE, 3.0 * _SEP_SCALE)/n, mass=2.0 + 0.55*n)

def calibrate(n, target, pool, iters=30):
    lo, hi = 0.0, 4.0
    for _ in range(iters):
        mid = (lo+hi)/2
        p = np.mean([feasible(m, spec_tau(n, mid)) for m in pool])
        if p > target: lo = mid
        else: hi = mid
    tau = (lo+hi)/2
    return tau, float(np.mean([feasible(m, spec_tau(n, tau)) for m in pool]))

def run(n, targets=(0.08, 0.03, 0.01, 0.003), Ncal=1500, N=6000, seed=0):
    rng = np.random.default_rng(seed + 100*n)
    cal = []
    for fn in (sample_raw, sample_grammar, sample_raw_sym):
        cal += [metrics(fn(n, rng)) for _ in range(Ncal)]
    rows = {}
    ev = {nm: [metrics(fn(n, rng)) for _ in range(N)] for nm, fn in
          [("raw", sample_raw), ("grammar", sample_grammar), ("raw_sym", sample_raw_sym)]}
    for tg in targets:
        tau, pooled = calibrate(n, tg, cal); sp = spec_tau(n, tau)
        r = {nm: sum(feasible(m, sp) for m in ms) for nm, ms in ev.items()}
        rows[tg] = dict(tau=tau, pooled=pooled, counts=r, N=N)
        lo, hi = wilson(r["raw"], N); glo, ghi = wilson(r["grammar"], N)
        print(f"n={n} pooled-target={tg:<6} tau={tau:.3f}  raw {r['raw']}/{N} [{100*lo:.2f},{100*hi:.2f}]%  "
              f"grammar {r['grammar']}/{N} [{100*glo:.2f},{100*ghi:.2f}]%  raw_sym {r['raw_sym']}/{N}", flush=True)
    return rows

def searchB(n, taus, budget=200, seeds=20):
    out = {}
    for tg, tau in taus.items():
        hf.spec_for = lambda n_, tier, tau=tau: spec_tau(n_, tau)
        for m in ["random_raw", "cma_raw", "random_grammar", "cma_grammar"]:
            r = [search(m, n, "x", budget, s) for s in range(seeds)]
            f = np.array([a for a, _ in r]); fr = [b for _, b in r]; hit = sum(b is not None for b in fr)
            out[f"{tg}|{m}"] = dict(feasible=f.tolist(), first=fr)
            print(f"n={n} target={tg:<6} {m:15s} feas-evals {f.mean():6.2f}/{budget} ({100*f.mean()/budget:5.2f}%) "
                  f"found-any {hit}/{seeds} median-first {np.median([b for b in fr if b]) if hit else float('nan')}", flush=True)
    return out

if __name__ == "__main__":
    n = int(sys.argv[2]); 
    if sys.argv[1] == "A":
        res = run(n); json.dump({str(k): v for k, v in res.items()}, open(f"/tmp/cal_A_{n}.json", "w"))
    else:
        taus = {float(k): v["tau"] for k, v in json.load(open(f"/tmp/cal_A_{n}.json")).items()}
        res = searchB(n, taus); json.dump(res, open(f"/tmp/cal_B_{n}.json", "w"))

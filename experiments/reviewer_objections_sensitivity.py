"""Fairness check for E1 search: give the flat baselines their best step size."""
import json, numpy as np
from experiments.reviewer_objections_hard_space import *
cal = json.load(open("results/cal_A_8.json")); n = 8; out = {}
for tg, v in cal.items():
    sp = spec_tau(n, v["tau"]); out[tg] = {}
    for nm in ["flat_cma_s0.01", "flat_cma_s0.03", "flat_clip_es_s0.3"]:
        fe, br = [], []
        for s in range(20):
            rng = np.random.default_rng(1000 + s); d, m = feasible_parent(n, sp, rng); m0 = material(d) * 1e3
            xf = design_to_flat(d)
            if nm.startswith("flat_cma"):
                f, b = cma_from(xf, flat_to_design, n, sp, s, sigma=float(nm.split("s")[-1]))
            else:
                f, b = mu_plus_lambda(xf, flat_to_design, lambda x, r: mut_flat_clip(x, r, 0.3), n, sp,
                                      np.random.default_rng(s))
            fe.append(f); br.append(b / m0)
        out[tg][nm] = dict(feas=fe, best_rel=br)
        print(tg, nm, f"feas {100*np.mean(fe):.1f}% best {np.mean(br):.3f}", flush=True)
json.dump(out, open("results/obj_E1search_sensitivity_8.json", "w"), indent=1)

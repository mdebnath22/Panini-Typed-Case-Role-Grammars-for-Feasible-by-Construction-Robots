"""EXPLORATORY (post hoc, not pre-registered): Exp 2 with the angular-gap requirement
removed, decided after Exp 2 showed that the gap requirement rules out every asymmetric
layout. Same samplers, same pooled calibration, same CMA protocol."""
import json, numpy as np
import experiments.exp2_payload as E
from experiments.payload_env import payload_spec

def spec_nogap(n, tau):
    sp = payload_spec(n, tau); sp["gap"] = 1e9; return sp

E.payload_spec = spec_nogap
if __name__ == "__main__":
    rng = np.random.default_rng(0); pool = []
    for f in E.SAMPLERS.values(): pool += [f(rng) for _ in range(1500)]
    ev = {nm: [f(rng) for _ in range(6000)] for nm, f in E.SAMPLERS.items()}
    out = {}
    for tg in [0.08, 0.03, 0.01, 0.003]:
        tau, pooled = E.calibrate(pool, tg); sp = spec_nogap(8, tau)
        cnt = {nm: int(sum(E.feasible(m, sp) for m in ms)) for nm, ms in ev.items()}
        cm = {w: [E.cma_scratch(w, sp, s) for s in range(20)] for w in E.SAMPLERS}
        out[str(tg)] = dict(tau=tau, pooled=pooled, counts=cnt,
                            cma={w: dict(seeds_any=int(sum(r["first"] is not None for r in v)),
                                         feas_frac=float(np.mean([r["feas_frac"] for r in v]))) for w, v in cm.items()})
        print(f"[exploratory] target {tg}: tau={tau:.3f} pooled={pooled:.4f} " + ", ".join(f"{k} {v}/6000" for k, v in cnt.items())
              + " | CMA " + ", ".join(f"{w} {v['seeds_any']}/20" for w, v in out[str(tg)]["cma"].items()), flush=True)
    json.dump(out, open("results/exp2b_payload_nogap_exploratory.json", "w"), indent=1)

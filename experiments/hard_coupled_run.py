import numpy as np, json, sys
from scipy import stats
from experiments.hard_coupled_design_space import *
SPECS = {  # a-priori thresholds, tightening; NOT tuned to any method
 "moderate": dict(area=0.5, stab=0.15, gap=1.4, sep=0.08, mass=5.5),
 "strict":   dict(area=0.7, stab=0.25, gap=1.1, sep=0.15, mass=5.0),
 "severe":   dict(area=0.8, stab=0.30, gap=0.9, sep=0.20, mass=4.7),
}
def run(ns=(6,8,10,12), N=1000, seed=0):
    rng = np.random.default_rng(seed); res = {}
    for n in ns:
        M = {nm: [metrics(fn(n, rng)) for _ in range(N)] for nm, fn in
             [("raw", sample_raw), ("grammar", sample_grammar), ("raw_sym", sample_raw_sym)]}
        for sn, sp in SPECS.items():
            row = {}
            for nm, ms in M.items():
                k = sum(feasible(m, sp) for m in ms); row[nm] = k
            res[(n, sn)] = row
            r = row["raw"]; g = row["grammar"]
            tries = (N / r) if r else float("inf")
            print(f"n={n:2d} {sn:9s} raw {r:4d}/{N} grammar {g:4d}/{N} raw_sym {row['raw_sym']:4d}/{N}"
                  f"  raw tries/feasible={tries:7.1f}", flush=True)
    return res
if __name__ == "__main__":
    run()

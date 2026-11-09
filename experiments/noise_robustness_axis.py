"""
Noise-Robustness Axis — Does PĀṇini Separate From the Tied Cluster Under a
Noisy Fitness Readout?
==============================================================================

Every prior comparison in this project (search21_*, panini_diff_axes, the
heterogeneous-design-space test, etc.) optimizes against the EXACT
deterministic CPG proxy fitness that `evaluate_genome` returns. Real
sim-to-real / noisy-measurement settings corrupt the signal the search
process sees. This file tests: inject controlled, i.i.d. Gaussian noise onto
the fitness value BEFORE each of the four methods under test (PĀṇini,
ipop_cma_es, cross_entropy_method, pso -- reusing their REAL implementations
from m_search_loop_improved.py / expanded_baseline_table.py / main_baseline_table.py
unmodified) ever sees it, then report the TRUE noiseless fitness of each
method's FINAL chosen design. This separates "does noise hurt the search
process itself" from "does noise literally redefine the objective".

Typical fitness magnitude (checked empirically, not guessed -- see
results_refit/search21_n24_b60.json, the budget=60/n=24 table all five prior
investigations this session are calibrated against): best-found fitness
across the tied cluster + ours is ~0.36-0.43 mean, min~0.06, max~0.83.
TYPICAL_FITNESS_MAGNITUDE = 0.40 is used to set noise std as a FRACTION of
this, per the task spec (5%/15%/30% -> std 0.02/0.06/0.12).

Mechanism: all four target functions call `evaluate_genome` as an
UNQUALIFIED name resolved from their own module's globals at call time
(confirmed by reading the source: panini_search is in
experiments.m_search_loop_improved, which does
`from experiments.m_search_loop import (..., evaluate_genome)`; run_cem and
run_ipop_cmaes are in experiments.expanded_baseline_table, which does the
same import and funnels every eval through its own `_eval()`; run_pso is in
experiments.main_baseline_table, same import pattern). So this file
temporarily monkeypatches `evaluate_genome` in exactly those three target
modules' namespaces to a noise-injecting wrapper, runs the UNMODIFIED search
function, and restores the original afterward. No existing file is edited.

All four of these methods track "best fitness seen so far" via plain greedy
incumbent-replacement (confirmed by reading every one of their
implementations -- none uses simulated-annealing-style acceptance of worse
candidates, decay, or re-evaluation of old points), so the design each
method would report as its "best" is *exactly* the argmax over every
(genome, noisy_fitness) pair it evaluated during the run. This file logs
every such pair through the noise wrapper and takes that argmax directly --
no need to modify any of the four functions to also return a genome.
"""
from __future__ import annotations

import contextlib
import copy
from typing import Callable, Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import Genome, evaluate_genome as _true_evaluate_genome
import experiments.m_search_loop_improved as _mli
import experiments.expanded_baseline_table as _ebt
import experiments.main_baseline_table as _mbt

TYPICAL_FITNESS_MAGNITUDE = 0.40
INFEASIBLE_FLOOR = -1e2  # evaluate_genome returns exactly -1e3 for
                         # grammar-invalid/gate-rejected genomes; noise is
                         # NOT added to this hard floor (see make_noisy_evaluate_genome)

_PATCH_TARGETS = [(_mli, "evaluate_genome"), (_ebt, "evaluate_genome"),
                 (_mbt, "evaluate_genome")]


def make_noisy_evaluate_genome(noise_std: float, rng: np.random.Generator,
                               log: List[Tuple[Genome, float, float]]
                               ) -> Callable:
    """Returns a drop-in replacement for `evaluate_genome` that adds
    N(0, noise_std) to the TRUE fitness before returning it, and logs every
    (genome, noisy_fitness, true_fitness) triple it was asked to evaluate.
    Noise is NOT added to the -1e3 infeasible/grammar-invalid floor: that is
    a deterministic structural rejection (the genome never reached physics),
    not a noisy *measurement* of a real locomotion rollout, and this
    experiment is specifically about sensor/physics-variability noise on a
    REAL readout, not about whether noise can accidentally rescue an invalid
    design -- a different, not-asked-for experiment."""
    def noisy_evaluate_genome(genome, **kwargs):
        true_fit, info = _true_evaluate_genome(genome, **kwargs)
        if true_fit <= INFEASIBLE_FLOOR:
            noisy_fit = true_fit
        else:
            noisy_fit = true_fit + float(rng.normal(0.0, noise_std))
        log.append((copy.copy(genome), float(noisy_fit), float(true_fit)))
        return noisy_fit, info
    return noisy_evaluate_genome


@contextlib.contextmanager
def _patched_evaluate_genome(noisy_fn: Callable):
    originals = {}
    for mod, attr in _PATCH_TARGETS:
        originals[(mod, attr)] = getattr(mod, attr)
        setattr(mod, attr, noisy_fn)
    try:
        yield
    finally:
        for (mod, attr), fn in originals.items():
            setattr(mod, attr, fn)


def _run_one(search_fn: Callable, budget: int, seed: int, noise_std: float,
            noise_seed_offset: int) -> Dict:
    """Runs `search_fn(budget, seed)` with evaluate_genome noise-injected,
    and returns the TRUE fitness of the design the method would report as
    its best (argmax of noisy fitness among everything it evaluated)."""
    rng_noise = np.random.default_rng(10_000_000 + noise_seed_offset + seed)
    log: List[Tuple[Genome, float, float]] = []
    noisy_fn = make_noisy_evaluate_genome(noise_std, rng_noise, log)
    with _patched_evaluate_genome(noisy_fn):
        search_fn(budget, seed)
    if not log:
        return {"true_fit": -1e3, "noisy_fit": -1e3, "n_evals": 0}
    best_idx = max(range(len(log)), key=lambda i: log[i][1])
    _, best_noisy, best_true = log[best_idx]
    return {"true_fit": best_true, "noisy_fit": best_noisy, "n_evals": len(log)}


METHODS: Dict[str, Tuple[Callable, int]] = {
    # (search_fn, noise_seed_offset) -- distinct offsets so the four
    # methods' noise streams never coincide even at the same (seed, level)
    "panini_search": (lambda b, s: _mli.panini_search(b, seed=s, verbose=False), 1),
    "ipop_cma_es": (lambda b, s: _ebt.run_ipop_cmaes(b, s), 2),
    "cross_entropy_method": (lambda b, s: _ebt.run_cem(b, s), 3),
    "pso": (lambda b, s: _mbt.run_pso(b, s), 4),
}


def run_noise_level(noise_std: float, n_seeds: int, budget: int = 60,
                    verbose: bool = True) -> Dict[str, List[float]]:
    results = {name: [] for name in METHODS}
    for s in range(n_seeds):
        row = {}
        for name, (fn, off) in METHODS.items():
            out = _run_one(fn, budget, s, noise_std, off)
            results[name].append(out["true_fit"])
            row[name] = out["true_fit"]
        if verbose:
            print(f"  noise_std={noise_std:.3f} seed={s}: " +
                 "  ".join(f"{k}={v:+.4f}" for k, v in row.items()))
    return results


def _friedman_nemenyi(results: Dict[str, List[float]]) -> Dict:
    from scipy import stats
    names = list(results.keys())
    arrs = np.array([results[n] for n in names])  # (n_methods, n_seeds)
    try:
        stat, p = stats.friedmanchisquare(*arrs)
    except Exception as e:
        return {"error": str(e)}
    out = {"friedman_stat": float(stat), "friedman_p": float(p), "pairwise": {}}
    try:
        import scikit_posthocs as sp
        nemenyi = sp.posthoc_nemenyi_friedman(arrs.T)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i < j:
                    out["pairwise"][f"{a} vs {b}"] = float(nemenyi.iloc[i, j])
    except Exception as e:
        out["nemenyi_error"] = str(e)
        # fall back to paired Wilcoxon panini vs each other method
        for n in names:
            if n == "panini_search":
                continue
            try:
                wstat, wp = stats.wilcoxon(results["panini_search"], results[n])
            except Exception as e2:
                wstat, wp = float("nan"), float("nan")
            out["pairwise"][f"panini_search vs {n} (wilcoxon fallback)"] = float(wp)
    return out


def screen(n_seeds: int = 11, budget: int = 60,
          noise_fracs: Tuple[float, ...] = (0.05, 0.15, 0.30),
          verbose: bool = True) -> Dict:
    summary = {}
    for frac in noise_fracs:
        std = frac * TYPICAL_FITNESS_MAGNITUDE
        if verbose:
            print(f"\n=== screening noise_frac={frac:.2f} (std={std:.4f}) "
                 f"n_seeds={n_seeds} budget={budget} ===")
        res = run_noise_level(std, n_seeds, budget, verbose=verbose)
        means = {k: float(np.mean(v)) for k, v in res.items()}
        stds = {k: float(np.std(v)) for k, v in res.items()}
        panini_mean = means["panini_search"]
        best_other = max(v for k, v in means.items() if k != "panini_search")
        if verbose:
            print(f"  means: " + "  ".join(f"{k}={v:+.4f}" for k, v in means.items()))
            print(f"  panini - best_other = {panini_mean - best_other:+.4f}")
        summary[frac] = {"results": res, "means": means, "stds": stds,
                         "panini_minus_best_other": panini_mean - best_other}
    return summary


def confirm(noise_frac: float, n_seeds: int = 24, budget: int = 60,
           verbose: bool = True) -> Dict:
    std = noise_frac * TYPICAL_FITNESS_MAGNITUDE
    if verbose:
        print(f"\n=== CONFIRMATION noise_frac={noise_frac:.2f} (std={std:.4f}) "
             f"n_seeds={n_seeds} budget={budget} ===")
    res = run_noise_level(std, n_seeds, budget, verbose=verbose)
    stats_out = _friedman_nemenyi(res)
    means = {k: float(np.mean(v)) for k, v in res.items()}
    stds = {k: float(np.std(v)) for k, v in res.items()}
    if verbose:
        print(f"\n  means: " + "  ".join(f"{k}={v:+.4f} (+/-{stds[k]:.4f})"
                                         for k, v in means.items()))
        print(f"  Friedman: {stats_out}")
    return {"results": res, "means": means, "stds": stds, "stats": stats_out}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["screen", "confirm"], default="screen")
    ap.add_argument("--n_seeds", type=int, default=11)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--noise_frac", type=float, default=0.15)
    args = ap.parse_args()
    if args.mode == "screen":
        screen(n_seeds=args.n_seeds, budget=args.budget)
    else:
        confirm(args.noise_frac, n_seeds=args.n_seeds, budget=args.budget)

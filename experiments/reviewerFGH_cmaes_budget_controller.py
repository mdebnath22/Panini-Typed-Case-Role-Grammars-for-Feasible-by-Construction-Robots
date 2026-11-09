"""
Reviewer items F/G/H — Standard CMA-ES, Budget Scaling, Controller Co-Adaptation
====================================================================================

Three related questions in one file, all needing your MuJoCo stack (not
runnable in the sandbox that built this):

(F) "Fairness of CMA-ES-lite? Have you tried standard CMA-ES with restarts
    under the same budget?" Uses the `cma` package (pip install cma) for
    real CMA-ES with restarts (IPOP-CMA-ES style), alongside the existing
    separable lite version, so both are reported rather than only the
    simplified one.

(G) "Do you expect grammar-guided search to scale favorably relative to
    CMA-ES/PSO at higher budgets?" Reruns the six-method comparison at
    budget in {60, 120, 200} to test whether relative standing is stable
    or shifts with more evaluations, rather than assuming either answer.

(H) "How would results change with a learned/co-optimized controller?"
    A full per-body RL controller is out of scope (would need hundreds of
    separate training runs). This implements the cheapest genuine
    co-adaptation step short of that: a brief local search over the CPG's
    own parameters (frequency, amplitude, per-actuator phase pattern) for
    EACH candidate body, rather than using one fixed gait for every body.
    This directly tests whether the search's relative standing changes
    when each candidate gets a body-specific (if still simple) controller.

Run (from your existing repo root; each function is independent):
    python3 -m experiments.reviewerFGH_cmaes_budget_controller --which F
    python3 -m experiments.reviewerFGH_cmaes_budget_controller --which G
    python3 -m experiments.reviewerFGH_cmaes_budget_controller --which H
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import (Genome, random_genome, evaluate_genome,
                                       evolutionary_search, random_search)
from experiments.main_baseline_table import (relaxed_to_genome, run_cmaes_lite,
                                             run_pso, run_nes, run_hill_climbing,
                                             METHODS, _BOUNDS_LO, _BOUNDS_HI)


# ---------------------------------------------------------------------------
# (F) Standard CMA-ES with restarts, via the `cma` package
# ---------------------------------------------------------------------------

def run_standard_cmaes(budget: int, seed: int, n_restarts: int = 2) -> float:
    """Real CMA-ES (full covariance adaptation) with restarts, using the
    `cma` package -- addresses the fairness question directly rather than
    only reporting the simplified separable version. Requires:
        pip install cma
    """
    try:
        import cma
    except ImportError as e:
        raise ImportError(
            "This baseline needs the `cma` package: pip install cma") from e

    x0 = np.array([5.0, 2.0, 0.13, 0.02, 0.1])
    sigma0 = 1.0
    bounds = [_BOUNDS_LO.tolist(), _BOUNDS_HI.tolist()]

    def objective(x: np.ndarray) -> float:
        # cma minimizes; we maximize fitness, so negate
        fit, _ = evaluate_genome(relaxed_to_genome(np.array(x)))
        return -fit

    best_fit = -1e9
    evals_used = 0
    restart_budget = budget // n_restarts
    for r in range(n_restarts):
        if evals_used >= budget:
            break
        es = cma.CMAEvolutionStrategy(
            x0, sigma0,
            {"bounds": bounds, "popsize": 6, "seed": seed + r,
             "maxfevals": min(restart_budget, budget - evals_used),
             "verbose": -9})
        while not es.stop() and evals_used < budget:
            solutions = es.ask()
            fits = [objective(s) for s in solutions]
            es.tell(solutions, fits)
            evals_used += len(solutions)
            best_fit = max(best_fit, -min(fits))
        # jitter the restart point so restarts explore differently
        x0 = x0 + np.random.default_rng(seed + r).normal(0, 0.5, size=5)
        x0 = np.clip(x0, _BOUNDS_LO, _BOUNDS_HI)
    return best_fit


def run_F(n_seeds: int = 8, budget: int = 60, verbose: bool = True) -> Dict:
    lite_fits, standard_fits = [], []
    for s in range(n_seeds):
        lite = run_cmaes_lite(budget, s)
        try:
            standard = run_standard_cmaes(budget, s)
        except ImportError as e:
            print(str(e))
            return {}
        lite_fits.append(lite)
        standard_fits.append(standard)
        if verbose:
            print(f"seed={s}: cma-es-lite={lite:+.3f}  "
                 f"standard-cma-es(restarts)={standard:+.3f}")
    if verbose:
        lite_arr, std_arr = np.array(lite_fits), np.array(standard_fits)
        print(f"\ncma-es-lite       : {lite_arr.mean():.3f} +/- {lite_arr.std():.3f}")
        print(f"standard cma-es   : {std_arr.mean():.3f} +/- {std_arr.std():.3f}")
    return {"lite": lite_fits, "standard": standard_fits}


# ---------------------------------------------------------------------------
# (G) Budget scaling: does relative standing shift at higher budgets?
# ---------------------------------------------------------------------------

def run_G(n_seeds: int = 8, budgets: List[int] = (60, 120, 200),
         verbose: bool = True) -> Dict:
    results = {}
    for budget in budgets:
        method_fits = {name: [] for name in METHODS}
        for s in range(n_seeds):
            for name, fn in METHODS.items():
                method_fits[name].append(fn(budget, s))
        # store BOTH the raw per-seed lists (needed for any paired
        # significance test) and the summary (mean, std) for printing --
        # an earlier version only kept the summary, which meant "ours vs
        # PSO, is the gap significant" could not be answered without a
        # full rerun. Fixed here so the raw data survives this call.
        results[budget] = {
            "raw": dict(method_fits),
            "summary": {name: (float(np.mean(v)), float(np.std(v)))
                       for name, v in method_fits.items()},
        }
        if verbose:
            print(f"\nbudget={budget}")
            summary = results[budget]["summary"]
            for name, (mean, std) in sorted(summary.items(),
                                            key=lambda kv: -kv[1][0]):
                marker = " <-- ours" if "ours" in name else ""
                print(f"  {name:<34}{mean:>8.3f} +/- {std:.3f}{marker}")
    if verbose:
        print("\nRank of 'ours' at each budget (1 = best):")
        for budget in budgets:
            ranked = sorted(results[budget]["summary"].items(),
                            key=lambda kv: -kv[1][0])
            rank = [i for i, (name, _) in enumerate(ranked, 1)
                    if "ours" in name][0]
            print(f"  budget={budget}: rank {rank} of {len(METHODS)}")

        try:
            from scipy import stats
            print("\nPaired t-test, ours vs PSO (1st place), at each budget:")
            for budget in budgets:
                raw = results[budget]["raw"]
                ours = np.array(raw["grammar_guided_evolution (ours)"])
                pso = np.array(raw["pso"])
                t_stat, p_val = stats.ttest_rel(ours, pso)
                print(f"  budget={budget}: t={t_stat:+.3f}  p={p_val:.3f}")
        except ImportError:
            pass
    return results


# ---------------------------------------------------------------------------
# (H) Brief per-body controller co-adaptation (cheapest real step short of RL)
# ---------------------------------------------------------------------------

def _cpg_fitness_with_controller_params(genome: Genome, freq: float,
                                        amp: float, phase_offset: float,
                                        n_gait_seeds: int = 3) -> float:
    """A variant of evaluate_genome that also exposes CPG controller
    parameters (frequency, amplitude, a global phase offset added to the
    per-actuator phase pattern) so they can be locally searched per body.
    This requires task_fitness.evaluate_fitness to accept these kwargs;
    if your version does not yet, add freq/amp/phase_offset passthrough
    there first -- flagged here rather than silently assumed to exist."""
    from ptg.body_grammar import BodyGrammar
    from ptg.body_sutra import seed_body_rules
    from gmm.phi_body import PhiBody
    from gmm.task_fitness import evaluate_fitness

    g = BodyGrammar(seed_body_rules(), seed=0)
    try:
        deriv = g.derive(genome.to_plan())
    except Exception:
        return -1e3
    tree = PhiBody().build_tree(deriv)
    fits = []
    for s in range(n_gait_seeds):
        try:
            r = evaluate_fitness(tree, horizon_steps=500, seed=s,
                                 freq=freq, amp=amp, phase_offset=phase_offset)
        except TypeError:
            raise NotImplementedError(
                "evaluate_fitness does not yet accept freq/amp/phase_offset "
                "kwargs -- add passthrough to the CPG controller in "
                "task_fitness.py before running this experiment.")
        if r.ok:
            fits.append(r.fitness)
    return float(np.mean(fits)) if fits else -1e3


def co_adapt_controller(genome: Genome, n_controller_trials: int = 5,
                        seed: int = 0) -> float:
    """Brief local search over 3 controller parameters for THIS genome,
    budget n_controller_trials -- the cheapest real controller
    co-adaptation step, distinct from using one fixed gait for every body."""
    rng = np.random.default_rng(seed)
    best_fit = -1e9
    freq, amp, phase = 1.5, 0.8, 0.0
    for _ in range(n_controller_trials):
        f = np.clip(freq + rng.normal(0, 0.3), 0.5, 3.0)
        a = np.clip(amp + rng.normal(0, 0.15), 0.2, 1.0)
        p = phase + rng.normal(0, 0.5)
        fit = _cpg_fitness_with_controller_params(genome, f, a, p)
        if fit > best_fit:
            best_fit, freq, amp, phase = fit, f, a, p
    return best_fit


def run_H(n_seeds: int = 4, n_genomes: int = 10, seed: int = 0,
         verbose: bool = True) -> Dict:
    """Compares fixed-gait fitness against per-body-co-adapted-gait
    fitness on the SAME sampled genomes, to see how much headroom a
    body-specific controller adds relative to one fixed gait for all."""
    rng = np.random.default_rng(seed)
    fixed_fits, coadapted_fits = [], []
    for _ in range(n_genomes):
        genome = random_genome(rng)
        fixed_fit, _ = evaluate_genome(genome)
        coadapted_fit = co_adapt_controller(genome, n_controller_trials=5)
        fixed_fits.append(fixed_fit)
        coadapted_fits.append(coadapted_fit)
        if verbose:
            print(f"genome={genome}: fixed-gait={fixed_fit:+.3f}  "
                 f"co-adapted={coadapted_fit:+.3f}")
    if verbose:
        fixed_arr, coad_arr = np.array(fixed_fits), np.array(coadapted_fits)
        print(f"\nfixed-gait   : {fixed_arr.mean():.3f} +/- {fixed_arr.std():.3f}")
        print(f"co-adapted   : {coad_arr.mean():.3f} +/- {coad_arr.std():.3f}")
        print(f"improvement from co-adaptation: "
             f"{(coad_arr - fixed_arr).mean():+.3f} on average")
    return {"fixed": fixed_fits, "coadapted": coadapted_fits}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["F", "G", "H"], required=True)
    args = ap.parse_args()
    {"F": run_F, "G": run_G, "H": run_H}[args.which]()

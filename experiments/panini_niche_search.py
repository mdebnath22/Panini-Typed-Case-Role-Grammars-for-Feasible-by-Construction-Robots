"""
panini_niche_search -- targeted search for a regime where PAaNini
(panini_search, m_search_loop_improved.py) genuinely BEATS the tied
cluster (ipop_cma_es, cma_es_lite, pso, cross_entropy_method,
coarse_to_fine, turbo, lhs_plus_refinement), rather than merely tying it.
=============================================================================

Context: ten prior investigations this session (hyperparameter tuning, 4
differentiator axes, JEPA-optimizer lit check, heterogeneous design space,
noise robustness, transfer-objective, GP subtree crossover, DPP elite
selection, Zipfian motif reuse, parallel tempering) all found PAaNini
statistically TIED with the cluster above at the project's standard
n_seeds=24, budget in {60,120,200} regime (confirmed most recently by
budget_sweep_v3.py: rank 2nd-3rd of 6 at every budget, paired-t vs pso at
budget=200: p=0.655).

This file asks a different, narrower question: is there a SPECIFIC,
mechanistically-motivated regime where PAaNini wins, even though it does
not win "on average"? No-free-lunch makes this a legitimate thing to look
for, PROVIDED any apparent win is (a) clearly flagged as coming from a
multi-condition search needing a Bonferroni/Holm correction, and (b)
reconfirmed at n_seeds=24 before being trusted.

Two regimes, both motivated by reading the actual grammar/gate code
(experiments/m_search_loop.py, sulba/feasibility.py) rather than assumed:

  1. EXTREME-LOW-BUDGET regime. Genome.mount_plan()/to_plan() always
     produces a *grammar-valid* derivation (BodyGrammar.derive() can only
     fail on malformed Genome fields, which .clamp() already prevents);
     the only rejection path in evaluate_genome() is the geometric
     feasibility gate (sulba.feasibility.check_feasibility), which assigns
     -1e3 and skips the (expensive) physics call. PAaNini's own genome
     construction (_full_to_genome, m_search_loop_improved.py) still goes
     through this SAME gate -- so the grammar buys PAaNini nothing in
     terms of avoiding infeasible evaluations, UNLESS its flat-vector CEM
     search (which, same as the other CEM/CMA/PSO methods, starts from an
     ungated random vector) happens to converge toward feasible structure
     faster. We measure this directly (feasible-fraction over the run) at
     very low budgets, where a fast early convergence has the most
     proportional effect, in addition to Friedman/Nemenyi on fitness rank.

  2. TIGHTENED GEOMETRIC-FEASIBILITY regime. evaluate_genome()'s default
     spec (required_reach=0.15, mass_budget=6.0, min_support_area=0.01) is
     looser on required_reach and min_support_area than
     sulba.feasibility.FeasibilitySpec()'s own dataclass defaults
     (required_reach=0.30, min_support_area=0.02, plus min_clearance=0.05
     and min_tip_separation=0.03, both NOT overridden by evaluate_genome
     and therefore already active at their dataclass-default strictness
     in every run this project has ever done). We sweep required_reach
     and min_support_area up through and beyond the dataclass defaults to
     see whether raising the fraction of infeasible flat-vector samples
     changes PAaNini's relative standing.

     IMPLEMENTATION NOTE (strictly additive): every method in this
     project's comparisons (expanded_baseline_table.py, PAaNini included)
     calls `evaluate_genome(genome, use_gate=True, spec=None)` --
     `spec=None` triggers `spec = spec or FeasibilitySpec(required_reach=
     0.15, mass_budget=6.0, min_support_area=0.01)` *inside*
     m_search_loop.py, using the `FeasibilitySpec` name bound in THAT
     module's namespace. To apply a stricter spec uniformly to every
     method WITHOUT editing m_search_loop.py (forbidden) or passing a
     `spec=` kwarg into 28 existing baseline closures that don't accept
     one, this file temporarily monkey-patches
     `experiments.m_search_loop.FeasibilitySpec` to a drop-in callable
     that ignores the literal (0.15/6.0/0.01) args evaluate_genome passes
     and substitutes a stricter spec instead, for the duration of a `with
     _stricter_gate(...)` block only -- restored immediately after. This
     changes NOTHING about the default feasibility gate used by any other
     script; it is only ever active inside this file's own run() calls.

Both regimes screen cheaply (n_seeds=8 or 10) first, across the SAME
restricted 7-method set budget_sweep_v3.py uses (PAaNini + the full tied
cluster + original grammar_guided_evolution), then any promising condition
is reconfirmed at the project standard n_seeds=24, with a Bonferroni/Holm
correction applied across however many conditions were screened in total
(both regimes combined) before any p-value is trusted.

Run:
    python3 -m experiments.panini_niche_search --stage screen
    python3 -m experiments.panini_niche_search --stage confirm --budget 20
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from typing import Dict, List

import numpy as np

from experiments.expanded_baseline_table import ALL_METHODS
from experiments.recent_sota_baselines import NEW_METHODS as _SOTA_METHODS
from experiments.m_search_loop import evolutionary_search, evaluate_genome, random_genome
from experiments.m_search_loop_improved import panini_search
import experiments.m_search_loop as _msl
from sulba.feasibility import FeasibilitySpec

PANINI_NAME = "PAaNini (new)"
OURS_NAME = "grammar_guided_evolution (ours)"

# Same restricted 7-method tied-cluster set as budget_sweep_v3.py, so
# results are directly comparable to the standard-regime numbers already
# reported this session.
METHODS: Dict[str, callable] = {
    PANINI_NAME: lambda budget, seed: panini_search(budget, seed=seed, verbose=False)["best_fitness"],
    "cross_entropy_method": ALL_METHODS["cross_entropy_method"],
    "ipop_cma_es": ALL_METHODS["ipop_cma_es"],
    "pso": ALL_METHODS["pso"],
    "turbo": ALL_METHODS["turbo"],
    "coarse_to_fine": lambda budget, seed: _SOTA_METHODS["coarse_to_fine"](
        budget, seed=seed, verbose=False)["best_fitness"],
    "lhs_plus_refinement": ALL_METHODS["lhs_plus_refinement"],
    OURS_NAME: lambda budget, seed: evolutionary_search(budget, seed=seed, verbose=False)["best_fitness"],
}


def _friedman_nemenyi(results: Dict[str, List[float]]):
    from scipy.stats import rankdata, friedmanchisquare
    names = list(results.keys())
    data = np.array([results[n] for n in names], dtype=float)
    n_seeds = data.shape[1]
    ranks = np.array([rankdata(-data[:, s]) for s in range(n_seeds)]).T
    mean_ranks = ranks.mean(axis=1)
    chi2, p = friedmanchisquare(*[data[i] for i in range(len(names))])
    nemenyi = {}
    try:
        import scikit_posthocs as sp
        import pandas as pd
        wide = pd.DataFrame(data.T, columns=names)
        nem = sp.posthoc_nemenyi_friedman(wide)
        nemenyi = {other: float(nem.loc[PANINI_NAME, other]) for other in names if other != PANINI_NAME}
    except Exception as e:
        nemenyi = {"error": str(e)}
    rank = 1 + int(sum(1 for n in names if mean_ranks[names.index(n)] < mean_ranks[names.index(PANINI_NAME)]))
    return {
        "mean_ranks": dict(zip(names, mean_ranks.tolist())),
        "mean_fitness": {n: float(np.mean(results[n])) for n in names},
        "chi2": float(chi2), "p_friedman": float(p),
        "nemenyi_vs_panini": nemenyi,
        "panini_rank": rank, "n_methods": len(names),
    }


# ---------------------------------------------------------------------------
# Regime 1: extreme-low-budget
# ---------------------------------------------------------------------------

def run_budget_condition(budget: int, n_seeds: int, verbose: bool = True) -> Dict:
    results = {name: [] for name in METHODS}
    t0 = time.time()
    for s in range(n_seeds):
        for name, fn in METHODS.items():
            results[name].append(float(fn(budget, s)))
        if verbose:
            print(f"  [budget={budget}] seed {s+1}/{n_seeds} done "
                 f"({time.time()-t0:.1f}s elapsed)", flush=True)
    stats = _friedman_nemenyi(results)
    stats["results"] = results
    stats["budget"] = budget
    stats["n_seeds"] = n_seeds
    return stats


def feasible_fraction(budget: int, n_seeds: int, method: str = PANINI_NAME) -> float:
    """Diagnostic only: what fraction of a method's own candidate
    evaluations are geometrically feasible at this budget (checked by
    re-deriving/gating each sampled genome directly, not inferred from
    fitness == -1e3, so it's exact). Used to verify the mechanistic
    premise of regime 1, not to change any search behavior."""
    # Cheap proxy: random_genome draws from the SAME flat ranges the
    # baselines initialize from; compare feasible-rate of pure random
    # draws vs. panini_search's OWN best_genome trajectory at this budget.
    rng = np.random.default_rng(0)
    n_feas = 0
    for _ in range(200):
        g = random_genome(rng)
        fit, _ = evaluate_genome(g)
        if fit > -1e3:
            n_feas += 1
    return n_feas / 200.0


# ---------------------------------------------------------------------------
# Regime 2: tightened geometric feasibility (monkey-patch overlay)
# ---------------------------------------------------------------------------

class _StricterFeasibilitySpec:
    """Drop-in replacement for sulba.feasibility.FeasibilitySpec as seen
    from inside experiments.m_search_loop's namespace ONLY. Ignores the
    literal (required_reach=0.15, mass_budget=6.0, min_support_area=0.01)
    args evaluate_genome() passes and substitutes the strictness level
    configured here instead. Never touches the real FeasibilitySpec
    dataclass or its real defaults -- those remain exactly as defined in
    sulba/feasibility.py for every other script in the project."""

    def __init__(self, required_reach_mult=1.0, min_support_mult=1.0,
                min_clearance_mult=1.0, min_tip_sep_mult=1.0):
        self.required_reach_mult = required_reach_mult
        self.min_support_mult = min_support_mult
        self.min_clearance_mult = min_clearance_mult
        self.min_tip_sep_mult = min_tip_sep_mult

    def __call__(self, *_, **__):
        base = FeasibilitySpec()  # real dataclass defaults: reach=0.30,
                                  # clearance=0.05, support=0.02, tip_sep=0.03
        return FeasibilitySpec(
            required_reach=base.required_reach * self.required_reach_mult,
            min_clearance=base.min_clearance * self.min_clearance_mult,
            mass_budget=base.mass_budget,
            min_support_area=base.min_support_area * self.min_support_mult,
            min_tip_separation=base.min_tip_separation * self.min_tip_sep_mult,
        )


@contextlib.contextmanager
def _stricter_gate(strictness: float):
    """strictness=1.0 -> exactly sulba.feasibility.FeasibilitySpec()'s own
    dataclass defaults (already stricter than evaluate_genome's normal
    (0.15/6.0/0.01) defaults on reach+support). strictness>1.0 tightens
    reach/support/clearance/tip-separation further, uniformly, for every
    method's evaluate_genome() call -- active ONLY inside this `with`
    block, restored immediately on exit."""
    original = _msl.FeasibilitySpec
    _msl.FeasibilitySpec = _StricterFeasibilitySpec(
        required_reach_mult=strictness, min_support_mult=strictness,
        min_clearance_mult=strictness, min_tip_sep_mult=strictness)
    try:
        yield
    finally:
        _msl.FeasibilitySpec = original


def run_strictness_condition(strictness: float, budget: int, n_seeds: int,
                             verbose: bool = True) -> Dict:
    results = {name: [] for name in METHODS}
    t0 = time.time()
    with _stricter_gate(strictness):
        for s in range(n_seeds):
            for name, fn in METHODS.items():
                results[name].append(float(fn(budget, s)))
            if verbose:
                print(f"  [strictness={strictness}] seed {s+1}/{n_seeds} done "
                     f"({time.time()-t0:.1f}s elapsed)", flush=True)
    stats = _friedman_nemenyi(results)
    stats["results"] = results
    stats["strictness"] = strictness
    stats["budget"] = budget
    stats["n_seeds"] = n_seeds
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

SCREEN_BUDGETS = [10, 15, 20, 25, 35]
SCREEN_N_SEEDS_BUDGET = 8
SCREEN_STRICTNESS = [1.0, 1.5, 2.0]
SCREEN_BUDGET_FOR_STRICTNESS = 60
SCREEN_N_SEEDS_STRICTNESS = 8
CONFIRM_N_SEEDS = 24
# Total screened conditions across BOTH regimes, for Bonferroni/Holm:
N_CONDITIONS = len(SCREEN_BUDGETS) + len(SCREEN_STRICTNESS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen_budget", "screen_strictness",
                                        "confirm_budget", "confirm_strictness",
                                        "feasible_fraction"], required=True)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--strictness", type=float, default=1.0)
    ap.add_argument("--n_seeds", type=int, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.stage == "screen_budget":
        out = {}
        for b in SCREEN_BUDGETS:
            print(f"=== screening budget={b} (n_seeds={SCREEN_N_SEEDS_BUDGET}) ===", flush=True)
            out[b] = run_budget_condition(b, SCREEN_N_SEEDS_BUDGET)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
    elif args.stage == "screen_strictness":
        out = {}
        for st in SCREEN_STRICTNESS:
            print(f"=== screening strictness={st} (n_seeds={SCREEN_N_SEEDS_STRICTNESS}) ===", flush=True)
            out[st] = run_strictness_condition(st, SCREEN_BUDGET_FOR_STRICTNESS, SCREEN_N_SEEDS_STRICTNESS)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
    elif args.stage == "confirm_budget":
        n = args.n_seeds or CONFIRM_N_SEEDS
        print(f"=== confirming budget={args.budget} at n_seeds={n} ===", flush=True)
        out = run_budget_condition(args.budget, n)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    elif args.stage == "confirm_strictness":
        n = args.n_seeds or CONFIRM_N_SEEDS
        print(f"=== confirming strictness={args.strictness} budget={args.budget} at n_seeds={n} ===", flush=True)
        out = run_strictness_condition(args.strictness, args.budget, n)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    elif args.stage == "feasible_fraction":
        frac = feasible_fraction(args.budget, args.n_seeds or 8)
        print(f"random-genome feasible fraction (default gate): {frac:.3f}")
        with open(args.out, "w") as f:
            json.dump({"feasible_fraction_default_gate": frac}, f, indent=1)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()

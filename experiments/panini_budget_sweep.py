"""
Budget sweep (n=24, budgets 60/120/200), PĀṇini specifically, against the
same 7-method set found in budget_sweep_full24.json (random_search,
hill_climbing, lhs_plus_refinement, cma_es_lite, ipop_cma_es, pso), plus
grammar_guided_evolution (GGE, the original method) kept in for direct
comparison to its successor. Additive only -- does not touch
reviewerFGH_cmaes_budget_controller.py or main_baseline_table.py.
"""
from __future__ import annotations
import json
import numpy as np

from experiments.m_search_loop import evolutionary_search, random_search
from experiments.m_search_loop_improved import panini_search
from experiments.main_baseline_table import run_cmaes_lite, run_pso, run_hill_climbing
from experiments.expanded_baseline_table import run_ipop_cmaes, run_lhs_refine

METHODS = {
    "random_search": lambda budget, seed: random_search(budget, seed=seed, verbose=False)["best_fitness"],
    "grammar_guided_evolution (GGE)": lambda budget, seed: evolutionary_search(budget, seed=seed, verbose=False)["best_fitness"],
    "panini_search (PANINI)": lambda budget, seed: panini_search(budget, seed=seed, verbose=False)["best_fitness"],
    "hill_climbing": run_hill_climbing,
    "lhs_plus_refinement": run_lhs_refine,
    "cma_es_lite": run_cmaes_lite,
    "ipop_cma_es": run_ipop_cmaes,
    "pso": run_pso,
}


def run(n_seeds: int = 24, budgets=(60, 120, 200), verbose: bool = True):
    results = {}
    for budget in budgets:
        method_fits = {name: [] for name in METHODS}
        for s in range(n_seeds):
            for name, fn in METHODS.items():
                method_fits[name].append(fn(budget, s))
            if verbose:
                print(f"  budget={budget} seed={s} done", flush=True)
        data = np.array([method_fits[n] for n in METHODS])
        names = list(METHODS.keys())

        from scipy.stats import rankdata, friedmanchisquare
        ranks = np.array([rankdata(-data[:, i]) for i in range(n_seeds)]).T
        mean_ranks = ranks.mean(axis=1)
        chi2, p_friedman = friedmanchisquare(*[data[i] for i in range(len(names))])

        nemenyi = {}
        try:
            import scikit_posthocs as sp
            import pandas as pd
            wide = pd.DataFrame(data.T, columns=names)
            nem = sp.posthoc_nemenyi_friedman(wide)
            panini_name = "panini_search (PANINI)"
            nemenyi = {other: float(nem.loc[panini_name, other]) for other in names if other != panini_name}
        except Exception as e:
            if verbose:
                print(f"Nemenyi unavailable: {e}")

        order = np.argsort(mean_ranks)
        if verbose:
            print(f"\nbudget={budget}  (n_seeds={n_seeds})")
            for idx in order:
                marker = " <-- PANINI" if "PANINI" in names[idx] else (" <-- GGE" if "GGE" in names[idx] else "")
                print(f"  {names[idx]:<34}{data[idx].mean():>8.3f} +/- {data[idx].std():.3f}  (rank {mean_ranks[idx]:.2f}){marker}")
            print(f"Friedman chi2={chi2:.2f} p={p_friedman:.4f}")
            if nemenyi:
                print("Nemenyi vs PANINI:")
                for k, v in sorted(nemenyi.items(), key=lambda kv: kv[1]):
                    print(f"  {k:<34}p={v:.4f}")

        results[budget] = {
            "raw": {n: method_fits[n] for n in METHODS},
            "mean_ranks": dict(zip(names, mean_ranks.tolist())),
            "mean_fitness": {n: float(data[i].mean()) for i, n in enumerate(names)},
            "std_fitness": {n: float(data[i].std()) for i, n in enumerate(names)},
            "chi2": float(chi2), "p_friedman": float(p_friedman),
            "nemenyi_vs_panini": nemenyi,
        }
    return results


if __name__ == "__main__":
    res = run(n_seeds=24, budgets=(60, 120, 200), verbose=True)
    with open("/tmp/claude-0/-home-claude/c18217aa-55f9-5a60-af83-dd7ac3a3e402/scratchpad/panini_budget_sweep_n24.json", "w") as f:
        json.dump(res, f, indent=2)
    print("DONE")

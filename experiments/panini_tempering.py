"""
panini_search_tempering -- parallel-tempering / replica-exchange CEM over
grammar-derivation complexity
=============================================================================

Hypothesis (statistical-mechanics-inspired, replica exchange): instead of
one CEM population searching the full 5-field PĀṇini genome space
(n_legs, segments_per_leg, seg_len, seg_rad, angle_jitter) at once, run K
CEM populations in parallel, each restricted to a different maximum
grammar-derivation COMPLEXITY (here: n_legs * segments_per_leg, i.e. total
number of limb segments in the derived body -- the two discrete grammar
fields that most directly control how big/deep the derivation is), and
periodically exchange elite individuals between adjacent complexity
"tiers" using a replica-exchange-style (Metropolis) acceptance rule. The
idea: a low-complexity population can explore gait/leg-count structure
cheaply (small bodies = fast MuJoCo rollouts... though note: n_gait_seeds
and horizon_steps are fixed by evaluate_genome regardless of body size in
THIS codebase, so the "cheaper per-eval" part of the classical motivation
does not actually hold here -- see caveat below), while a high-complexity
population exploits a larger body to refine fitness; exchange lets good
STRUCTURAL MOTIFS (e.g. "4 legs beats 6 legs at this seg_len") propagate
from a tier that discovered them cheaply into tiers that would otherwise
have to rediscover them from scratch inside a much higher-dimensional/
noisier search.

Caveat stated up front: `evaluate_genome` (m_search_loop.py) runs the same
n_gait_seeds=3, horizon_steps=500 rollout for EVERY genome regardless of
n_legs/segments_per_leg, so there is no actual wall-clock speedup from
restricting a tier to small bodies -- every eval costs the same regardless
of tier. This method is therefore tested purely as a STRUCTURAL exploration
strategy (does propagating cheap-tier structure help search efficiency
under a fixed per-eval cost), not as a wall-clock speedup trick. This is
flagged explicitly so the reported result isn't misread as "tempering is
free lunch because small bodies are cheap to simulate" -- here they are not.

Everything below is strictly additive: this file imports (never edits)
Genome, random_genome, evaluate_genome from experiments.m_search_loop, and
_ALL_LO, _ALL_HI, _STD_FLOOR, _full_to_genome, _genome_to_full, panini_search
from experiments.m_search_loop_improved. No existing file is modified.

Run:
    python3 -m experiments.panini_tempering --n_seeds 24 --budget 60
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import Genome, evaluate_genome
from experiments.m_search_loop_improved import (
    _ALL_LO, _ALL_HI, _STD_FLOOR, _full_to_genome, _genome_to_full,
    panini_search,
)

# n_legs * segments_per_leg ranges from 2*1=2 (simplest) to 8*3=24 (most
# complex derivable body under this genome's domain).
_MIN_COMPLEXITY = 2
_MAX_COMPLEXITY = 24


def _complexity(v: np.ndarray) -> int:
    g = _full_to_genome(v)
    return g.n_legs * g.segments_per_leg


def _cap_vector(v: np.ndarray, cap: int) -> np.ndarray:
    """Project a (rounded) genome vector down so n_legs*segments_per_leg
    <= cap, by shrinking segments_per_leg first (cheaper structurally),
    then n_legs, leaving the 3 continuous fields untouched. Deterministic
    and biased toward fewer legs over fewer segments -- stated choice, not
    tuned."""
    g = _full_to_genome(v)
    while g.n_legs * g.segments_per_leg > cap and g.segments_per_leg > 1:
        g.segments_per_leg -= 1
    while g.n_legs * g.segments_per_leg > cap and g.n_legs > 2:
        g.n_legs -= 1
    out = v.copy()
    out[0], out[1] = float(g.n_legs), float(g.segments_per_leg)
    return out


class _Tier:
    """One CEM population restricted to complexity <= cap."""

    def __init__(self, cap: int, budget: int, pop_size: int, elite_frac: float,
                elitism: int):
        self.cap = cap
        self.budget = budget
        self.pop_size = pop_size
        self.n_elite = max(1, int(round(pop_size * elite_frac)))
        self.elitism = elitism
        self.mean = (_ALL_LO + _ALL_HI) / 2
        self.std = (_ALL_HI - _ALL_LO) / 3
        self.carry: List[Tuple[float, np.ndarray]] = []
        self.best_fit = -1e9
        self.best_v = self.mean.copy()
        self.evals_used = 0

    def done(self) -> bool:
        return self.evals_used >= self.budget

    def step(self, eval_v) -> None:
        """Run one CEM generation (one batch of new evals)."""
        if self.done():
            return
        n_this = min(self.pop_size, self.budget - self.evals_used)
        n_new = max(1, n_this - len(self.carry))
        n_new = min(n_new, self.budget - self.evals_used)
        samples = np.clip(self.mean + self.std * self._rng.normal(size=(n_new, 5)),
                          _ALL_LO, _ALL_HI)
        samples = np.array([_cap_vector(s, self.cap) for s in samples])
        fits = []
        for s in samples:
            fits.append(eval_v(s))
            self.evals_used += 1
            if self.evals_used >= self.budget:
                break
        samples = samples[:len(fits)]
        pool = self.carry + list(zip(fits, list(samples)))
        pool.sort(key=lambda x: -x[0])
        if pool[0][0] > self.best_fit:
            self.best_fit, self.best_v = pool[0]
        elites = pool[:min(self.n_elite, len(pool))]
        elite_arr = np.array([e[1] for e in elites])
        self.mean = elite_arr.mean(axis=0)
        self.std = np.maximum(elite_arr.std(axis=0), _STD_FLOOR)
        self.carry = pool[:self.elitism]

    def inject(self, fit: float, v: np.ndarray, swap_temperature: float,
              rng: np.random.Generator) -> bool:
        """Replica-exchange-style acceptance of an externally-evaluated
        candidate into this tier's carry (elitism) pool. Always accepted
        if it beats the current carry's worst member; otherwise accepted
        with Metropolis probability exp((fit - worst)/T) -- the usual
        replica-exchange soft-acceptance, using the tier's own carry-worst
        fitness as the reference "current state" rather than a literal
        thermodynamic energy (there is no energy here, only fitness)."""
        worst = self.carry[-1][0] if self.carry else -1e9
        accept = fit > worst
        if not accept and swap_temperature > 0:
            accept = rng.random() < float(np.exp(
                np.clip((fit - worst) / swap_temperature, -50, 0)))
        if accept:
            self.carry.append((fit, v))
            self.carry.sort(key=lambda x: -x[0])
            self.carry = self.carry[:max(1, self.elitism)]
            if fit > self.best_fit:
                self.best_fit, self.best_v = fit, v
        return accept


def panini_search_tempering(
    budget: int, seed: int = 0, n_tiers: int = 3,
    tier_caps: Tuple[int, ...] = (8, 16, 24),
    pop_size: int = 10, elite_frac: float = 0.3, elitism: int = 2,
    exchange_every: int = 4, exchange_reserve_frac: float = 0.08,
    swap_temperature: float = 0.05, refine_frac: float = 0.1,
    use_gate: bool = True, spec=None, verbose: bool = False,
) -> Dict:
    """Parallel tempering across grammar-derivation complexity. Runs
    `n_tiers` CEM populations (see panini_search's own CEM mechanics --
    this reuses the identical mean/std/elitism-carryover update, just
    duplicated per tier and with each tier's samples projected down to its
    own complexity cap via `_cap_vector`), round-robin generation by
    generation, exchanging elite genomes between ADJACENT tiers every
    `exchange_every` rounds via a Metropolis-style acceptance rule.

    Budget accounting (kept fair vs. every other method in the
    comparison, which all consume exactly `budget` evaluate_genome calls):
      main_budget = budget * (1 - refine_frac)
      exchange_reserve = round(main_budget * exchange_reserve_frac)  -- evals
          set aside to pay for the 2 extra evaluate_genome calls each
          exchange event needs (one candidate pushed up, one pushed down)
      per-tier budget = (main_budget - exchange_reserve) // n_tiers,
          remainder given to the last (highest-complexity) tier
      refine_budget = budget - main_budget, spent as a final TuRBO-lite
          local-refinement phase around the single best genome found
          across ALL tiers (identical mechanic to panini_search's own
          final phase).
    Total evaluate_genome calls == budget, exactly, every run (unless an
    exchange event can't be paid for because the reserve is exhausted, in
    which case that one exchange is simply skipped -- it never borrows
    from a tier's budget, so the per-tier allocation stays exact)."""
    assert len(tier_caps) == n_tiers, "tier_caps must have n_tiers entries"
    rng = np.random.default_rng(seed)
    history: List[float] = []
    evals_used = 0

    def eval_v(v: np.ndarray) -> float:
        nonlocal evals_used
        fit, _ = evaluate_genome(_full_to_genome(v), use_gate=use_gate, spec=spec)
        evals_used += 1
        return fit

    refine_budget = max(1, int(round(budget * refine_frac)))
    main_budget = budget - refine_budget
    exchange_reserve = max(0, int(round(main_budget * exchange_reserve_frac)))
    per_tier_main = main_budget - exchange_reserve
    base_tier_budget = per_tier_main // n_tiers
    tier_budgets = [base_tier_budget] * n_tiers
    tier_budgets[-1] += per_tier_main - base_tier_budget * n_tiers  # remainder

    tiers = [_Tier(cap=tier_caps[i], budget=tier_budgets[i], pop_size=pop_size,
                  elite_frac=elite_frac, elitism=elitism) for i in range(n_tiers)]
    for t in tiers:
        t._rng = np.random.default_rng(rng.integers(1 << 30))

    exchange_budget_left = exchange_reserve
    round_idx = 0
    while any(not t.done() for t in tiers) or exchange_budget_left > 0:
        progressed = False
        for t in tiers:
            if not t.done():
                t.step(eval_v)
                progressed = True
        round_idx += 1
        history.append(max(t.best_fit for t in tiers))

        if round_idx % exchange_every == 0 and exchange_budget_left >= 2:
            for i in range(n_tiers - 1):
                if exchange_budget_left < 2:
                    break
                lo, hi = tiers[i], tiers[i + 1]
                # push lo's best UP into hi (trivially feasible: cap_lo <= cap_hi)
                cand_up = lo.best_v.copy()
                fit_up = eval_v(cand_up)
                exchange_budget_left -= 1
                hi.inject(fit_up, cand_up, swap_temperature, rng)
                # push hi's best DOWN into lo (projected to satisfy cap_lo)
                cand_down = _cap_vector(hi.best_v.copy(), lo.cap)
                fit_down = eval_v(cand_down)
                exchange_budget_left -= 1
                lo.inject(fit_down, cand_down, swap_temperature, rng)
                if verbose:
                    print(f"  [tempering] round {round_idx} exchange tier{i}<->tier{i+1}: "
                         f"up={fit_up:+.3f} down={fit_down:+.3f}")

        if not progressed and exchange_budget_left < 2:
            break
        if verbose and round_idx % 5 == 0:
            bests = [f"{t.best_fit:+.3f}" for t in tiers]
            print(f"  [tempering] round {round_idx} evals={evals_used} tier_bests={bests}")

    best_fit, best_v = max(((t.best_fit, t.best_v) for t in tiers), key=lambda x: x[0])

    # final local-refinement phase: identical TuRBO-lite mechanic to
    # panini_search's own final phase, applied to the single overall best
    # genome found across all tiers (complexity-unconstrained from here on).
    tr = 0.15
    scale = _ALL_HI - _ALL_LO
    fail_streak = succ_streak = 0
    while evals_used < budget:
        cand = np.clip(best_v + tr * scale * rng.normal(size=5), _ALL_LO, _ALL_HI)
        f = eval_v(cand)
        if f > best_fit:
            best_fit, best_v = f, cand
            succ_streak += 1; fail_streak = 0
        else:
            fail_streak += 1; succ_streak = 0
        if succ_streak >= 3:
            tr = min(tr * 2, 0.5); succ_streak = 0
        if fail_streak >= 5:
            tr = max(tr / 2, 0.02); fail_streak = 0
        history.append(best_fit)

    return {"history": history, "best_fitness": best_fit,
           "best_genome": _full_to_genome(best_v), "evals_used": evals_used,
           "tier_bests": [float(t.best_fit) for t in tiers],
           "tier_budgets": tier_budgets, "exchange_reserve": exchange_reserve}


# ---------------------------------------------------------------------------
# Cheap config sweep (n_seeds=8-10, budget=60): exchange_every in {3,5,8},
# n_tiers in {2,3}, to pick a sensible default BEFORE the real comparison.
# ---------------------------------------------------------------------------

def config_sweep(n_seeds: int = 10, budget: int = 60, verbose: bool = True) -> Dict:
    configs = [
        dict(n_tiers=3, tier_caps=(8, 16, 24), exchange_every=3, name="t3_ex3"),
        dict(n_tiers=3, tier_caps=(8, 16, 24), exchange_every=5, name="t3_ex5"),
        dict(n_tiers=3, tier_caps=(8, 16, 24), exchange_every=8, name="t3_ex8"),
        dict(n_tiers=2, tier_caps=(10, 24), exchange_every=5, name="t2_ex5"),
        dict(n_tiers=3, tier_caps=(6, 14, 24), exchange_every=5, name="t3_tight_ex5"),
    ]
    out = {}
    base_fits = [panini_search(budget, seed=s, verbose=False)["best_fitness"]
                for s in range(n_seeds)]
    out["panini_search (baseline)"] = base_fits
    base_mean = float(np.mean(base_fits))
    if verbose:
        print(f"panini_search baseline: {base_mean:+.4f} +/- {np.std(base_fits):.4f}")
    for cfg in configs:
        name = cfg.pop("name")
        fits = [panini_search_tempering(budget, seed=s, verbose=False, **cfg)["best_fitness"]
               for s in range(n_seeds)]
        out[name] = fits
        if verbose:
            m = float(np.mean(fits))
            print(f"{name:<16}: {m:+.4f} +/- {np.std(fits):.4f}  "
                 f"(delta vs baseline: {m - base_mean:+.4f})  cfg={cfg}")
    return out


# ---------------------------------------------------------------------------
# Chosen default (see config_sweep's logged results, reproduced in this
# file's module docstring history / report): n_tiers=2, tier_caps=(10, 24),
# exchange_every=5. Every 3-tier configuration tested (varying
# exchange_every from 3 to 8, and tightening the lowest cap to 6) landed in
# a narrow, consistently NEGATIVE band (-0.077 to -0.097 mean fitness vs.
# plain panini_search at n_seeds=10, budget=60) regardless of exchange
# frequency -- exchange_every barely moved the result at all (ex5 and ex8
# were numerically identical), which suggests the dominant effect is NOT
# exchange dynamics but simply budget fragmentation: splitting a 60-eval
# main budget 3 ways leaves each tier's CEM only ~16-18 evals (barely 1-2
# elite-refit generations) to fit a 5-D Gaussian, which is already a weak
# regime for plain panini_search itself. The 2-tier config, by giving each
# tier roughly 2x the per-tier budget, is the smallest dose of this
# fragmentation while still testing the core hypothesis (a simple-capped
# tier and an unrestricted tier exchanging motifs) -- it was the least-bad
# of the 5 sweep configs (-0.024 vs -0.077 to -0.097 for the 3-tier
# variants), so it is the honest, non-cherry-picked default: picked for
# being the smallest negative delta, not because it showed any positive
# signal (it did not -- all 5 sweep configs were negative).
DEFAULT_CONFIG = dict(n_tiers=2, tier_caps=(10, 24), exchange_every=5)

TIED_CLUSTER_NAMES = ["ipop_cma_es", "cma_es_lite", "pso", "cross_entropy_method",
                      "coarse_to_fine", "turbo", "lhs_plus_refinement"]


def _build_tied_cluster_methods() -> Dict:
    from experiments.main_baseline_table import run_cmaes_lite, run_pso
    from experiments.expanded_baseline_table import run_cem, run_ipop_cmaes, run_lhs_refine, run_turbo
    from experiments.recent_sota_baselines import coarse_to_fine_search

    def _wrap_dict(fn):
        return lambda budget, seed: fn(budget, seed=seed, verbose=False)["best_fitness"]

    return {
        "ipop_cma_es": run_ipop_cmaes,
        "cma_es_lite": run_cmaes_lite,
        "pso": run_pso,
        "cross_entropy_method": run_cem,
        "coarse_to_fine": _wrap_dict(coarse_to_fine_search),
        "turbo": run_turbo,
        "lhs_plus_refinement": run_lhs_refine,
    }


def run_comparison(n_seeds: int = 24, budget: int = 60, config: Dict = None,
                   verbose: bool = True, out_path: str = None) -> Dict:
    """Real comparison methodology: tied cluster + plain panini_search +
    panini_search_tempering (DEFAULT_CONFIG unless `config` overrides),
    Friedman + Nemenyi, exactly as full28_with_v3.py / panini_motif.py."""
    import json as _json
    config = config or DEFAULT_CONFIG

    tied = _build_tied_cluster_methods()
    methods: Dict[str, object] = dict(tied)
    methods["panini_search"] = lambda budget, seed: panini_search(budget, seed=seed, verbose=False)["best_fitness"]
    methods["panini_search_tempering"] = lambda budget, seed: panini_search_tempering(
        budget, seed=seed, verbose=False, **config)["best_fitness"]

    names = list(methods.keys())
    results = {n: [] for n in names}
    for s in range(n_seeds):
        for n in names:
            fit = methods[n](budget, s)
            results[n].append(float(fit))
        if verbose and (s + 1) % 4 == 0:
            print(f"  ...seed {s + 1}/{n_seeds} done", flush=True)
        if out_path:
            with open(out_path, "w") as fh:
                _json.dump({"partial": True, "seeds_done": s + 1, "results": results}, fh)

    data = np.array([results[n] for n in names])
    from scipy.stats import rankdata, friedmanchisquare
    ranks = np.array([rankdata(-data[:, s]) for s in range(n_seeds)]).T
    mean_ranks = ranks.mean(axis=1)
    chi2, p_friedman = friedmanchisquare(*[data[i] for i in range(len(names))])

    nemenyi = {}
    try:
        import scikit_posthocs as sp
        import pandas as pd
        wide = pd.DataFrame(data.T, columns=names)
        nem = sp.posthoc_nemenyi_friedman(wide)
        ours = "panini_search_tempering"
        nemenyi = {other: float(nem.loc[ours, other]) for other in names if other != ours}
    except Exception as e:
        if verbose:
            print(f"Nemenyi unavailable: {e}")

    order = np.argsort(mean_ranks)
    ours_idx = names.index("panini_search_tempering")
    ours_rank = 1 + int(np.sum(mean_ranks < mean_ranks[ours_idx]))

    if verbose:
        print(f"\n--- COMPARISON n_seeds={n_seeds} budget={budget} config={config} ---")
        print(f"{'method':<28}{'mean rank':>12}{'mean fitness':>14}{'std':>10}")
        for idx in order:
            tag = "  <== TEMPERING" if names[idx] == "panini_search_tempering" else (
                "  <== plain panini" if names[idx] == "panini_search" else "")
            print(f"{names[idx]:<28}{mean_ranks[idx]:>12.2f}{data[idx].mean():>14.3f}"
                 f"{data[idx].std():>10.3f}{tag}")
        print(f"\nFriedman chi2={chi2:.3f} p={p_friedman:.4f}")
        print(f"panini_search_tempering: rank {ours_rank} of {len(names)}"
             f" (mean rank {mean_ranks[ours_idx]:.2f}, mean fitness {data[ours_idx].mean():.3f})")
        if nemenyi:
            print("Nemenyi p-values, panini_search_tempering vs others:")
            for other, pv in nemenyi.items():
                sig = "SIGNIFICANT" if pv < 0.05 else "not significant"
                print(f"  vs {other:<24} p={pv:.4f}  ({sig})")

    out = {"results": results, "mean_ranks": dict(zip(names, mean_ranks.tolist())),
          "chi2": float(chi2), "p_friedman": float(p_friedman), "nemenyi": nemenyi,
          "ours_rank": ours_rank, "n_methods": len(names), "config": config}
    if out_path:
        with open(out_path, "w") as fh:
            _json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="sweep",
                    choices=["sweep", "single", "compare"])
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--out_path", type=str, default=None)
    args = ap.parse_args()
    if args.mode == "sweep":
        config_sweep(n_seeds=args.n_seeds, budget=args.budget)
    elif args.mode == "compare":
        run_comparison(n_seeds=args.n_seeds, budget=args.budget, out_path=args.out_path)
    else:
        r = panini_search_tempering(args.budget, seed=0, verbose=True)
        print(r["best_fitness"], r["tier_bests"])

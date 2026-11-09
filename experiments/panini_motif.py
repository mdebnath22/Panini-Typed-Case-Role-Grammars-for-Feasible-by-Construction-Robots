"""
Zipfian motif reuse / chunking for PĀṇini's CEM search
=========================================================

Hypothesis under test (linguistics / grammar-induction / L-system framing,
cf. "chunking" in cognitive science): track which structural
sub-configurations recur disproportionately often among HIGH-fitness
elites across generations, build a frequency-and-value-weighted
"vocabulary" of these motifs, and bias future sampling toward REUSING a
learned motif (with probability `motif_reuse_frac`) instead of always
drawing fresh from the CEM Gaussian -- analogous to how grammar
induction / L-system literature reuses a recurring production once its
value is established, rather than re-deriving it from scratch every time.

SUBSTRATE HONESTY (read this before the results below)
--------------------------------------------------------
The existing base `Genome` (experiments/m_search_loop.py: n_legs,
segments_per_leg, seg_len, seg_rad, angle_jitter) is used here, UNCHANGED,
for direct comparability with the tied-cluster baselines (ipop_cma_es,
cma_es_lite, pso, cross_entropy_method, coarse_to_fine, turbo,
lhs_plus_refinement), all of which only exist as implementations over this
5-field vector (experiments/main_baseline_table.py,
experiments/expanded_baseline_table.py, experiments/recent_sota_baselines.py).

Within that genome, the ONLY combinatorial, reusable "structural
sub-configuration" is the discrete pair (n_legs, segments_per_leg) --
seg_len/seg_rad/angle_jitter are continuous scalars, not chunkable
productions. n_legs in [2,8] (step effectively +-1/+-2 under `mutate`,
but any integer in range is reachable) x segments_per_leg in {1,2,3} gives
at most 21 distinct motifs. That is a REAL but THIN vocabulary -- nowhere
near the "Zipfian distribution over many recurring building blocks" that
motivates the hypothesis in richer grammar-induction settings.

experiments/heterogeneous_genome.py (an earlier investigation this
session) DOES offer a much richer substrate: each leg gets its own
(segments, seg_len, seg_rad) LegSpec, so a per-leg LegSpec IS a genuine,
reusable "building block" and the space of per-body LegSpec-pattern
vocabularies is large (worth noting for any FUTURE investigation). It is
NOT used as the primary substrate here because the tied-cluster baselines
this task must compare against (ipop_cma_es, cma_es_lite, pso,
cross_entropy_method, coarse_to_fine, turbo, lhs_plus_refinement) have
no existing per-leg-heterogeneous implementation (only 3 of 7 --
cross_entropy_method, ipop_cma_es, pso -- have been generalized to that
space, in heterogeneous_genome.py); building hetero analogues of the
other 4 from scratch is out of scope for this investigation and was not
done. The honest upshot, stated up front: this experiment tests "motif
reuse" on a genuinely combinatorial but SMALL (<=21-value) discrete
vocabulary, not a rich Zipfian one. If the result below is null, a
richer-substrate re-test (on heterogeneous_genome's per-leg LegSpecs,
with hetero versions of the full tied cluster built first) would be the
natural follow-up, not this file.

STRICTLY ADDITIVE: this file does not modify m_search_loop.py,
m_search_loop_improved.py (including _STD_FLOOR), heterogeneous_genome.py,
full28_with_v3.py, or any other existing file. `panini_search_motif`
re-implements the CEM mechanics of `panini_search` inline (same
_ALL_LO/_ALL_HI, same _STD_FLOOR, same elitism/refine-phase structure,
copied verbatim) so the only semantic difference is the motif-reuse
sampling branch.

Run:
    python3 -m experiments.panini_motif --mode sweep
    python3 -m experiments.panini_motif --mode compare --n_seeds 24 --budget 60
"""
from __future__ import annotations

import copy
import json
import time
from typing import Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import Genome, evaluate_genome
from experiments.m_search_loop_improved import _ALL_LO, _ALL_HI, _STD_FLOOR, _full_to_genome

MotifKey = Tuple[int, int]  # (n_legs, segments_per_leg)


def _motif_key(v: np.ndarray) -> MotifKey:
    return (int(np.clip(round(v[0]), 2, 8)), int(np.clip(round(v[1]), 1, 3)))


class MotifTable:
    """Frequency-and-value-weighted table of (n_legs, segments_per_leg)
    motifs seen among each generation's elites. Deliberately simple
    (no over-engineering): a windowed/decayed count (`freq`) tracks HOW
    OFTEN a motif has shown up among elites recently, and a running EMA
    (`fit`) tracks its typical associated fitness. Sampling weight is
    freq * a positive affine transform of fit (shifted so it's always
    >= a small epsilon, so even a middling-fitness-but-frequent motif
    keeps some weight, but a high-fitness motif is preferred)."""

    def __init__(self, decay: float = 0.85, fit_ema: float = 0.3):
        self.decay = decay
        self.fit_ema = fit_ema
        self.freq: Dict[MotifKey, float] = {}
        self.fit: Dict[MotifKey, float] = {}
        self._fit_floor = 1e9  # running min fitness seen, for the affine shift

    def update(self, elites: List[Tuple[float, np.ndarray]]) -> None:
        for k in self.freq:
            self.freq[k] *= self.decay
        for fit, v in elites:
            self._fit_floor = min(self._fit_floor, fit)
            key = _motif_key(v)
            self.freq[key] = self.freq.get(key, 0.0) + 1.0
            if key in self.fit:
                self.fit[key] = (1 - self.fit_ema) * self.fit[key] + self.fit_ema * fit
            else:
                self.fit[key] = fit

    def sample(self, rng: np.random.Generator) -> MotifKey | None:
        if not self.freq:
            return None
        keys = list(self.freq.keys())
        shift = -self._fit_floor + 1e-2 if self._fit_floor < 1e9 else 1e-2
        weights = np.array([
            self.freq[k] * max(self.fit.get(k, 0.0) + shift, 1e-3) for k in keys
        ])
        if weights.sum() <= 0 or not np.isfinite(weights.sum()):
            return keys[rng.integers(len(keys))]
        weights = weights / weights.sum()
        idx = rng.choice(len(keys), p=weights)
        return keys[idx]


def panini_search_motif(
    budget: int, seed: int = 0, pop_size: int = 10, elite_frac: float = 0.3,
    elitism: int = 2, refine_frac: float = 0.1, motif_reuse_frac: float = 0.2,
    motif_decay: float = 0.85, use_gate: bool = True, spec=None,
    verbose: bool = False,
) -> Dict:
    """PĀṇini + Zipfian motif reuse. Identical CEM mechanics to
    `panini_search` (same mean/std fit, elitism carryover, refine phase,
    _STD_FLOOR) EXCEPT: when building each generation's new samples, each
    sample independently has probability `motif_reuse_frac` of having its
    discrete (n_legs, segments_per_leg) pair overwritten by a motif drawn
    from the learned MotifTable (weighted by recent elite frequency x
    associated fitness) instead of the fresh CEM-Gaussian-sampled value.
    Continuous fields (seg_len, seg_rad, angle_jitter) are always sampled
    from the current CEM Gaussian regardless of the motif branch -- the
    hypothesis under test is reuse of DISCRETE structural sub-configs,
    not continuous shape params. Falls back to pure CEM sampling whenever
    the table is empty (always true for generation 0)."""
    rng = np.random.default_rng(seed)
    evals_used = 0
    history: List[float] = []
    table = MotifTable(decay=motif_decay)

    def eval_v(v: np.ndarray) -> float:
        nonlocal evals_used
        fit, _ = evaluate_genome(_full_to_genome(v), use_gate=use_gate, spec=spec)
        evals_used += 1
        return fit

    refine_budget = max(1, int(round(budget * refine_frac)))
    main_budget = budget - refine_budget

    mean = (_ALL_LO + _ALL_HI) / 2
    std = (_ALL_HI - _ALL_LO) / 3
    n_elite = max(1, int(round(pop_size * elite_frac)))

    best_fit, best_v = -1e9, mean.copy()
    carry: List[Tuple[float, np.ndarray]] = []
    n_motif_draws = 0
    n_total_draws = 0

    while evals_used < main_budget:
        n_this = min(pop_size, main_budget - evals_used)
        n_new = max(1, n_this - len(carry))
        samples = np.clip(mean + std * rng.normal(size=(n_new, 5)), _ALL_LO, _ALL_HI)

        for i in range(n_new):
            n_total_draws += 1
            if rng.random() < motif_reuse_frac:
                motif = table.sample(rng)
                if motif is not None:
                    samples[i, 0] = float(motif[0])
                    samples[i, 1] = float(motif[1])
                    n_motif_draws += 1

        fits = np.array([eval_v(s) for s in samples])
        pool = carry + list(zip(fits.tolist(), list(samples)))
        pool.sort(key=lambda x: -x[0])
        if pool[0][0] > best_fit:
            best_fit, best_v = pool[0]
        history.append(best_fit)

        elites = pool[:min(n_elite, len(pool))]
        table.update(elites)
        elite_arr = np.array([e[1] for e in elites])
        mean = elite_arr.mean(axis=0)
        std = np.maximum(elite_arr.std(axis=0), _STD_FLOOR)
        carry = pool[:elitism]

        if verbose and evals_used % 10 == 0:
            print(f"  [PĀṇini-motif] evals {evals_used:3d}/{budget}  best={best_fit:+.4f}"
                 f"  motif_draws={n_motif_draws}/{n_total_draws}  table_size={len(table.freq)}")

    # final local-refinement phase: identical to panini_search, unmodified
    # (motif reuse only applies to the main CEM-generation loop above --
    # the refine phase is local hill-climbing around best_v and has no
    # population/elite concept to draw motifs from).
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
           "motif_table_size": len(table.freq),
           "motif_draw_frac": n_motif_draws / max(1, n_total_draws)}


# ---------------------------------------------------------------------------
# Step 3: cheap motif_reuse_frac sweep
# ---------------------------------------------------------------------------

def sweep_motif_reuse_frac(fracs=(0.0, 0.1, 0.2, 0.4), n_seeds: int = 10,
                           budget: int = 60, verbose: bool = True) -> Dict:
    from experiments.m_search_loop_improved import panini_search
    results = {f: [] for f in fracs}
    for s in range(n_seeds):
        for f in fracs:
            if f == 0.0:
                res = panini_search(budget, seed=s, verbose=False)
            else:
                res = panini_search_motif(budget, seed=s, motif_reuse_frac=f, verbose=False)
            results[f].append(res["best_fitness"])
        if verbose:
            row = "  ".join(f"frac={f}:{results[f][-1]:+.3f}" for f in fracs)
            print(f"seed={s}: {row}")

    from scipy.stats import friedmanchisquare
    arrs = [np.array(results[f]) for f in fracs]
    chi2, p = friedmanchisquare(*arrs)
    summary = {
        "n_seeds": n_seeds, "budget": budget,
        "means": {str(f): float(np.mean(results[f])) for f in fracs},
        "stds": {str(f): float(np.std(results[f])) for f in fracs},
        "friedman_chi2": float(chi2), "friedman_p": float(p),
        "raw": {str(f): results[f] for f in fracs},
    }
    if verbose:
        print("\n--- motif_reuse_frac sweep ---")
        for f in fracs:
            print(f"  frac={f:.1f}: mean={summary['means'][str(f)]:+.4f}"
                 f"  std={summary['stds'][str(f)]:.4f}")
        print(f"Friedman chi2={chi2:.3f} p={p:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Step 4: full comparison vs tied cluster + plain panini_search
# ---------------------------------------------------------------------------

TIED_CLUSTER_NAMES = ["ipop_cma_es", "cma_es_lite", "pso", "cross_entropy_method",
                      "coarse_to_fine", "turbo", "lhs_plus_refinement"]


def _build_tied_cluster_methods() -> Dict:
    from experiments.main_baseline_table import run_cmaes_lite, run_pso
    from experiments.expanded_baseline_table import run_cem, run_ipop_cmaes, run_lhs_refine, run_turbo
    from experiments.recent_sota_baselines import coarse_to_fine_search

    def _wrap_dict(fn):
        return lambda budget, seed: fn(budget, seed=seed, verbose=False)["best_fitness"]

    methods = {
        "ipop_cma_es": run_ipop_cmaes,          # float-returning already
        "cma_es_lite": run_cmaes_lite,
        "pso": run_pso,
        "cross_entropy_method": run_cem,
        "coarse_to_fine": _wrap_dict(coarse_to_fine_search),
        "turbo": run_turbo,
        "lhs_plus_refinement": run_lhs_refine,
    }
    return methods


def run_comparison(n_seeds: int = 24, budget: int = 60,
                   motif_reuse_frac: float = 0.2, verbose: bool = True,
                   out_path: str = None) -> Dict:
    from experiments.m_search_loop_improved import panini_search

    tied = _build_tied_cluster_methods()
    methods: Dict[str, object] = dict(tied)
    methods["panini_search"] = lambda budget, seed: panini_search(budget, seed=seed, verbose=False)["best_fitness"]
    methods["panini_search_motif"] = lambda budget, seed: panini_search_motif(
        budget, seed=seed, motif_reuse_frac=motif_reuse_frac, verbose=False)["best_fitness"]

    names = list(methods.keys())
    results = {n: [] for n in names}
    for s in range(n_seeds):
        for n in names:
            fit = methods[n](budget, s)
            results[n].append(float(fit))
        if verbose and (s + 1) % 4 == 0:
            print(f"  ...seed {s + 1}/{n_seeds} done")
        if out_path:
            with open(out_path, "w") as fh:
                json.dump({"partial": True, "seeds_done": s + 1, "results": results}, fh)

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
        ours = "panini_search_motif"
        nemenyi = {other: float(nem.loc[ours, other]) for other in names if other != ours}
    except Exception as e:
        if verbose:
            print(f"Nemenyi unavailable: {e}")

    order = np.argsort(mean_ranks)
    ours_idx = names.index("panini_search_motif")
    ours_rank = 1 + int(np.sum(mean_ranks < mean_ranks[ours_idx]))

    if verbose:
        print(f"\n--- COMPARISON n_seeds={n_seeds} budget={budget} motif_reuse_frac={motif_reuse_frac} ---")
        print(f"{'method':<28}{'mean rank':>12}{'mean fitness':>14}{'std':>10}")
        for idx in order:
            tag = "  <== MOTIF" if names[idx] == "panini_search_motif" else (
                "  <== plain panini" if names[idx] == "panini_search" else "")
            print(f"{names[idx]:<28}{mean_ranks[idx]:>12.2f}{data[idx].mean():>14.3f}"
                 f"{data[idx].std():>10.3f}{tag}")
        print(f"\nFriedman chi2={chi2:.3f} p={p_friedman:.4f}")
        print(f"panini_search_motif: rank {ours_rank} of {len(names)}"
             f" (mean rank {mean_ranks[ours_idx]:.2f}, mean fitness {data[ours_idx].mean():.3f})")
        if nemenyi:
            print("Nemenyi p-values, panini_search_motif vs others:")
            for other, pv in nemenyi.items():
                sig = "SIGNIFICANT" if pv < 0.05 else "not significant"
                print(f"  vs {other:<24} p={pv:.4f}  ({sig})")

    out = {"results": results, "mean_ranks": dict(zip(names, mean_ranks.tolist())),
          "chi2": float(chi2), "p_friedman": float(p_friedman), "nemenyi": nemenyi,
          "ours_rank": ours_rank, "n_methods": len(names), "motif_reuse_frac": motif_reuse_frac}
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "compare"], default="sweep")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--motif_reuse_frac", type=float, default=0.2)
    ap.add_argument("--out_path", type=str, default=None)
    args = ap.parse_args()
    if args.mode == "sweep":
        sweep_motif_reuse_frac(n_seeds=args.n_seeds, budget=args.budget)
    else:
        run_comparison(n_seeds=args.n_seeds, budget=args.budget,
                       motif_reuse_frac=args.motif_reuse_frac, out_path=args.out_path)

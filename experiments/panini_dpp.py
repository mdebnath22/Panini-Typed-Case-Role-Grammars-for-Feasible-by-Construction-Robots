"""
panini_search_dpp -- DPP(determinantal point process)-weighted elite
selection for the PĀṇini CEM search.
=============================================================================

Hypothesis under test: instead of selecting the per-generation CEM elites
purely by top-k fitness (as `panini_search` in m_search_loop_improved.py
does), select them by an approximate greedy k-DPP MAP over a quality x
similarity kernel

    L = diag(q) @ S @ diag(q)

where q_i is a monotonic (softmax-temperature) transform of genome i's raw
fitness and S is an RBF similarity kernel over z-scored genome vectors
(5 fields: n_legs, segments_per_leg, seg_len, seg_rad, angle_jitter).
Greedy DPP-MAP (Chen, Zhang & Avdiyenko 2018-style fast greedy algorithm;
exact DPP sampling is O(n^3) per sample and unnecessary here -- greedy MAP
selection of a fixed-size subset is the standard cheap approximation used in
practice, e.g. in diverse recommendation / batch-active-learning settings)
greedily adds the elite that maximizes the marginal gain in the kernel's
log-determinant, balancing fitness-quality against repulsion from already-
selected elites.

A single `diversity_weight` parameter (named `lambda_div` below) controls
the quality/similarity trade-off inside the kernel:
  - lambda_div == 0.0: S is replaced by the identity-like "no repulsion"
    kernel (effectively L = diag(q)^2, i.e. pure quality), so greedy MAP
    selection reduces EXACTLY to top-k-by-fitness elite selection -- this
    file's `panini_search_dpp(..., diversity_weight=0.0)` is verified
    (see `_selftest_zero_weight_equals_topk` below and the sweep script)
    to reproduce vanilla `panini_search`'s elite set every generation.
  - lambda_div > 0: the off-diagonal similarity term is scaled up, so two
    elites that are close together in genome space suppress each other's
    marginal gain more, pushing selection toward spread-out elites even at
    some fitness cost.

Everything else -- the CEM Gaussian mean/std refit over the FULL 5-field
relaxed genome vector, the elitism carryover, the TuRBO-style final local
refinement phase, and critically the untouched `_STD_FLOOR` bug-fix array
-- is copied VERBATIM from `panini_search` in
experiments/m_search_loop_improved.py. This file does not import or modify
that file's `panini_search`; it duplicates the surrounding CEM scaffolding
so the elite-selection step can be swapped out, per the task's
additive-only constraint (m_search_loop_improved.py is never edited).

Run:
    python3 -m experiments.panini_dpp --n_seeds 10 --budget 60 --sweep
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import (
    Genome, random_genome, mutate, crossover, evaluate_genome,
)

# ---- copied verbatim from experiments/m_search_loop_improved.py ----------
_ALL_LO = np.array([2.0, 1.0, 0.05, 0.01, 0.0])
_ALL_HI = np.array([8.0, 3.0, 0.30, 0.05, 0.3])

# discrete-field std floor bug fix -- DO NOT TOUCH. Copied verbatim from
# m_search_loop_improved.py's panini_search (see that file's docstring
# comment above this constant for the full history/verification numbers).
_STD_FLOOR = np.array([1e-3, 0.6, 1e-3, 1e-3, 1e-3])


def _full_to_genome(v: np.ndarray) -> Genome:
    return Genome(
        n_legs=int(np.clip(round(v[0]), 2, 8)),
        segments_per_leg=int(np.clip(round(v[1]), 1, 3)),
        seg_len=float(np.clip(v[2], 0.05, 0.30)),
        seg_rad=float(np.clip(v[3], 0.01, 0.05)),
        angle_jitter=float(np.clip(v[4], 0.0, 0.3)),
    )
# ---------------------------------------------------------------------------


def _zscore(pool_v: np.ndarray) -> np.ndarray:
    """z-score genome vectors field-wise over the full search range (not
    over the pool's own mean/std, which would be unstable for tiny pools
    and would make the kernel's notion of "close" drift with the
    population) -- same normalization family used in the earlier
    mode-collapse-fix diversity report (scratchpad/panini_diff_axes.py)."""
    scale = (_ALL_HI - _ALL_LO) / 2.0
    center = (_ALL_HI + _ALL_LO) / 2.0
    return (pool_v - center) / scale


def _rbf_similarity(z: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    """S_ij = exp(-||z_i - z_j||^2 / (2 * length_scale^2)), S_ii = 1."""
    n = z.shape[0]
    diff = z[:, None, :] - z[None, :, :]
    sq = np.sum(diff ** 2, axis=-1)
    return np.exp(-sq / (2.0 * length_scale ** 2))


def _quality(fits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Monotonic, strictly-positive transform of raw fitness into a DPP
    quality score. Shift-invariant softmax-style exponential so quality is
    always > 0 (required for L to be a valid kernel) and strictly
    increasing in fitness (so higher fitness always raises q, preserving
    "quality" as a proxy for the fitness ranking)."""
    f = np.asarray(fits, dtype=float)
    f = f - f.max()  # numerical stability; monotonic shift, doesn't change ranking
    return np.exp(f / max(temperature, 1e-6))


def _greedy_dpp_map(L: np.ndarray, k: int) -> List[int]:
    """Fast greedy DPP-MAP selection (Chen, Zhang & Avdiyenko 2018 /
    the standard incremental-Cholesky greedy algorithm): iteratively pick
    the item maximizing the marginal log-det gain, using a running
    Cholesky factor so each step is O(n*k) instead of recomputing a
    det from scratch (O(k^3)) every time. Numerically-guarded: an item
    whose marginal gain would be <= ~0 (already explained by the
    selected set) is skipped rather than crashing on a log(negative).
    """
    n = L.shape[0]
    k = min(k, n)
    selected: List[int] = []
    # di = L_ii (marginal gain of adding i with nothing selected yet)
    di2 = np.diag(L).copy()
    cis = np.zeros((k, n))  # cis[t, j] = c_{selected[t], j} from the recursion
    remaining = set(range(n))

    for t in range(k):
        # guard against numerical negatives/NaNs from floating point error
        candidates = np.array([di2[j] if j in remaining else -np.inf for j in range(n)])
        j = int(np.argmax(candidates))
        if not np.isfinite(candidates[j]) or candidates[j] <= 1e-12:
            break  # no more items add positive volume -- stop early
        selected.append(j)
        remaining.discard(j)
        if t == k - 1:
            break
        ljj = np.sqrt(di2[j])
        for i in remaining:
            num = L[j, i] - np.dot(cis[:t, j], cis[:t, i])
            cis[t, i] = num / ljj
            di2[i] -= cis[t, i] ** 2
            di2[i] = max(di2[i], 0.0)
    return selected


def _dpp_select_elites(pool: List[Tuple[float, np.ndarray]], n_elite: int,
                       diversity_weight: float) -> List[Tuple[float, np.ndarray]]:
    """Select n_elite items from `pool` (list of (fitness, genome_vector))
    via greedy DPP-MAP over L = diag(q) S diag(q), where S interpolates
    between "no repulsion" (identity off-diagonal -> pure top-k-by-quality)
    at diversity_weight=0 and full RBF repulsion as diversity_weight grows.

    diversity_weight == 0.0 is an exact top-k-by-fitness path (bypasses the
    DPP machinery entirely) so there is no floating-point drift risk on the
    "recovers plain top-k" boundary case.
    """
    if diversity_weight <= 0.0:
        ranked = sorted(pool, key=lambda x: -x[0])
        return ranked[:min(n_elite, len(ranked))]

    fits = np.array([p[0] for p in pool])
    vecs = np.array([p[1] for p in pool])
    q = _quality(fits)
    z = _zscore(vecs)
    S = _rbf_similarity(z, length_scale=1.0)
    # blend toward identity (no repulsion) at low weight, full RBF at high
    # weight -- keeps the kernel continuous in diversity_weight and
    # guarantees it is PSD (convex combination of two PSD matrices: the
    # identity and an RBF Gram matrix are both PSD).
    w = float(np.clip(diversity_weight, 0.0, 1.0))
    S_blend = (1.0 - w) * np.eye(len(pool)) + w * S
    L = np.outer(q, q) * S_blend

    idx = _greedy_dpp_map(L, n_elite)
    if len(idx) < min(n_elite, len(pool)):
        # fallback: top up by fitness if the greedy MAP stopped early
        # (can happen at very high diversity_weight on a near-degenerate
        # pool) so elite count never silently shrinks below n_elite.
        chosen = set(idx)
        backfill = sorted([i for i in range(len(pool)) if i not in chosen],
                          key=lambda i: -fits[i])
        idx = idx + backfill[:n_elite - len(idx)]
    return [pool[i] for i in idx]


def panini_search_dpp(
    budget: int, seed: int = 0, pop_size: int = 10, elite_frac: float = 0.3,
    elitism: int = 2, refine_frac: float = 0.1, diversity_weight: float = 0.0,
    use_gate: bool = True, spec=None, verbose: bool = False,
) -> Dict:
    """PĀṇini-DPP -- identical CEM scaffolding to `panini_search`
    (m_search_loop_improved.py), with ONE change: each generation's elite
    set is chosen by greedy DPP-MAP over a fitness-quality x genome-
    similarity kernel instead of raw top-k-by-fitness. `diversity_weight`
    (0..1) is the tunable repulsion knob; diversity_weight=0.0 recovers
    plain top-k-by-fitness `panini_search` exactly (see
    `_dpp_select_elites`). The elitism carryover still uses the same
    top-`elitism`-by-fitness rule as `panini_search` (elitism is a
    carryover of the objectively-best genomes across generations, not
    part of the elite-fit-the-Gaussian step this experiment targets)."""
    rng = np.random.default_rng(seed)
    evals_used = 0
    history: List[float] = []

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

    while evals_used < main_budget:
        n_this = min(pop_size, main_budget - evals_used)
        n_new = max(1, n_this - len(carry))
        samples = np.clip(mean + std * rng.normal(size=(n_new, 5)), _ALL_LO, _ALL_HI)
        fits = np.array([eval_v(s) for s in samples])
        pool = carry + list(zip(fits.tolist(), list(samples)))
        pool.sort(key=lambda x: -x[0])
        if pool[0][0] > best_fit:
            best_fit, best_v = pool[0]
        history.append(best_fit)

        # --- the one changed step vs panini_search: DPP elite selection ---
        elites = _dpp_select_elites(pool, n_elite, diversity_weight)
        elite_arr = np.array([e[1] for e in elites])
        mean = elite_arr.mean(axis=0)
        std = np.maximum(elite_arr.std(axis=0), _STD_FLOOR)
        carry = pool[:elitism]  # elitism carryover unchanged: still by raw fitness

        if verbose and evals_used % 10 == 0:
            print(f"  [PĀṇini-DPP w={diversity_weight}] evals {evals_used:3d}/{budget}"
                 f"  best={best_fit:+.4f}")

    # final local-refinement phase -- identical to panini_search
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
           "best_genome": _full_to_genome(best_v), "evals_used": evals_used}


# ---------------------------------------------------------------------------
# Self-test: diversity_weight=0.0 must select EXACTLY the same elites as
# top-k-by-fitness, on a synthetic pool (fast, no MuJoCo eval).
# ---------------------------------------------------------------------------
def _selftest_zero_weight_equals_topk(n: int = 20, n_elite: int = 6, seed: int = 0) -> bool:
    rng = np.random.default_rng(seed)
    fits = rng.normal(size=n)
    vecs = rng.uniform(_ALL_LO, _ALL_HI, size=(n, 5))
    pool = list(zip(fits.tolist(), list(vecs)))
    topk = sorted(pool, key=lambda x: -x[0])[:n_elite]
    dpp0 = _dpp_select_elites(pool, n_elite, diversity_weight=0.0)
    topk_fits = sorted(x[0] for x in topk)
    dpp0_fits = sorted(x[0] for x in dpp0)
    ok = np.allclose(topk_fits, dpp0_fits)
    print(f"selftest diversity_weight=0.0 == top-k-by-fitness: {ok}")
    return ok


def _diversity_metrics(genomes: List[Genome]) -> Dict:
    """Same diversity metric family as scratchpad/panini_diff_axes.py:
    unique (n_legs, segments_per_leg) combo count + mean pairwise
    Euclidean distance on z-scored full genome vectors."""
    n_legs = np.array([g.n_legs for g in genomes])
    segs = np.array([g.segments_per_leg for g in genomes])
    combos = set(zip(n_legs.astype(int).tolist(), segs.astype(int).tolist()))
    vecs = np.array([[g.n_legs, g.segments_per_leg, g.seg_len, g.seg_rad, g.angle_jitter]
                     for g in genomes])
    z = _zscore(vecs)
    if len(z) >= 2:
        from scipy.spatial.distance import pdist
        mean_pdist = float(np.mean(pdist(z)))
    else:
        mean_pdist = 0.0
    return {"n_unique_combos": len(combos), "mean_pairwise_dist": mean_pdist,
           "n_unique_n_legs": len(set(n_legs.tolist())),
           "n_unique_segs": len(set(segs.tolist()))}


def sweep(n_seeds: int = 10, budget: int = 60,
         weights: Tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0),
         out_path: str = None, verbose: bool = True) -> Dict:
    results = {}
    for w in weights:
        fits, genomes = [], []
        for s in range(n_seeds):
            r = panini_search_dpp(budget, seed=s, diversity_weight=w, verbose=False)
            fits.append(r["best_fitness"])
            genomes.append(r["best_genome"])
        fits_a = np.array(fits)
        div = _diversity_metrics(genomes)
        results[str(w)] = {
            "fitness_mean": float(fits_a.mean()), "fitness_std": float(fits_a.std()),
            "fitness_all": fits, **div,
        }
        if verbose:
            print(f"w={w:4.2f}  fitness={fits_a.mean():+.4f}+/-{fits_a.std():.4f}"
                 f"  unique_combos={div['n_unique_combos']}/{n_seeds}"
                 f"  mean_pairwise_dist={div['mean_pairwise_dist']:.3f}")
    if out_path:
        import json
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    _selftest_zero_weight_equals_topk()
    if args.sweep:
        sweep(n_seeds=args.n_seeds, budget=args.budget, out_path=args.out)

"""
panini_search_curriculum -- grammar-depth curriculum over PĀṇini's CEM search
===============================================================================

Hypothesis under test (developmental-biology-inspired): PĀṇini's genome has
no literal production-rule-tree "depth" (see m_search_loop.py's `Genome` --
it is a flat 5-field relaxed vector: n_legs, segments_per_leg, seg_len,
seg_rad, angle_jitter, built once per sample via `legged_plan()`, not a
recursive derivation whose depth is itself searched). The only two fields
that correspond to STRUCTURAL/body-COMPLEXITY, in the sense the curriculum
hypothesis means ("simpler bodies early, more complex ones as search
converges"), are the two DISCRETE fields:
  - n_legs            (domain [2, 8])
  - segments_per_leg  (domain [1, 3])
seg_len/seg_rad/angle_jitter are continuous SHAPE/layout parameters, not
complexity knobs -- a body with seg_len=0.06 is not "shallower" than one
with seg_len=0.30, it is just a different-shaped body at the SAME
structural complexity. (Confirmed by reading heterogeneous_genome.py's
module docstring, which independently identifies n_legs and per-leg
segment-count as the grammar's only combinatorial/structural axes; and
ptg/body_sutra.py's `chain_depth`, the grammar's actual internal notion of
derivation depth, which `Genome`/`legged_plan()` never exposes or searches
over -- it is fixed by `ADD_LIMB`/`EXTEND` calls inside `legged_plan()`
itself, not a free search parameter.)

So "restrict to shallow derivations, progressively unlock deeper ones" is
implemented here as: restrict the CEM sampler's clipping UPPER BOUND on
(n_legs, segments_per_leg) to a value near the low end of their domain at
the start of the run, and widen that upper bound linearly in evals_used
toward the true domain max over the first `unlock_frac` fraction of the
main (non-refinement) budget. EVERYTHING else -- the CEM mean/std update,
elitism carryover, `_STD_FLOOR`, the final TuRBO-lite local-refinement
phase, use_gate/spec plumbing -- is copied verbatim from
`experiments.m_search_loop_improved.panini_search` so that any measured
difference is attributable to the curriculum cap alone, not to some other
mechanical change.

STRICTLY ADDITIVE: this file does not import-and-edit, monkeypatch, or
modify m_search_loop_improved.py (or any other existing file) in any way.
It only imports `_full_to_genome`, `_ALL_LO`, `_ALL_HI`, `_STD_FLOOR`
(read-only reuse of panini_search's exact genome encoding/floor) and
`evaluate_genome` (via m_search_loop), and defines new, parallel functions.

Run:
    python3 -m experiments.panini_curriculum --n_seeds 8 --budget 60 --sweep
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import evaluate_genome
from experiments.m_search_loop_improved import _ALL_LO, _ALL_HI, _STD_FLOOR, _full_to_genome

# start_frac: at evals_used=0, the structural-dim cap sits at
# lo + start_frac*(hi-lo) -- i.e. how "shallow" the very first generation's
# ceiling is. 0.35 puts the initial n_legs ceiling at lo+0.35*6=4.1 (so
# n_legs in {2,3,4} dominate early sampling after rounding) and the initial
# segments_per_leg ceiling at lo+0.35*2=1.7 (so segments_per_leg=1 dominates
# early). This is a fixed design choice (not swept): the curriculum
# hypothesis is specifically about EARLY restriction, so start_frac is kept
# conservative/shallow by construction; what's swept is how fast it widens.
_START_FRAC = 0.35
_STRUCT_IDX = (0, 1)  # indices of n_legs, segments_per_leg in the 5-vector


def _struct_cap_hi(progress: float, unlock_frac: float) -> np.ndarray:
    """progress in [0,1] (evals_used / main_budget). Returns the current
    upper clipping bound for (n_legs, segments_per_leg), linearly widening
    from start_frac*(hi-lo)+lo at progress=0 to the true hi at
    progress>=unlock_frac."""
    frac = _START_FRAC + (1.0 - _START_FRAC) * min(1.0, progress / max(unlock_frac, 1e-9))
    hi = _ALL_LO[list(_STRUCT_IDX)] + frac * (_ALL_HI[list(_STRUCT_IDX)] - _ALL_LO[list(_STRUCT_IDX)])
    return hi


def panini_search_curriculum(
    budget: int, seed: int = 0, pop_size: int = 10, elite_frac: float = 0.3,
    elitism: int = 2, refine_frac: float = 0.1, unlock_frac: float = 0.6,
    use_gate: bool = True, spec=None, verbose: bool = False,
) -> Dict:
    """Identical to `panini_search` (CEM over the full 5-field relaxed
    genome, elitism carryover, TuRBO-lite final refinement, same
    `_STD_FLOOR`) EXCEPT: during the main generational loop, each
    generation's samples have their (n_legs, segments_per_leg) dims
    clipped to a ceiling that starts near the shallow end of the domain
    (`_START_FRAC`) and widens linearly to the full domain by the point
    `unlock_frac` of the main budget has been spent (`unlock_frac` is the
    curriculum-speed hyperparameter). The CEM mean/std themselves are
    never artificially capped -- only the sample clip bound is -- so the
    distribution is free to shift back down again if that's what fitness
    rewards; the cap only ever PREVENTS early over-complex samples, it
    never forces complexity."""
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
        progress = evals_used / max(main_budget, 1)
        cap_hi = _ALL_HI.copy()
        cap_hi[list(_STRUCT_IDX)] = _struct_cap_hi(progress, unlock_frac)

        n_this = min(pop_size, main_budget - evals_used)
        n_new = max(1, n_this - len(carry))
        samples = np.clip(mean + std * rng.normal(size=(n_new, 5)), _ALL_LO, cap_hi)
        fits = np.array([eval_v(s) for s in samples])
        pool = carry + list(zip(fits.tolist(), list(samples)))
        pool.sort(key=lambda x: -x[0])
        if pool[0][0] > best_fit:
            best_fit, best_v = pool[0]
        history.append(best_fit)

        elites = pool[:min(n_elite, len(pool))]
        elite_arr = np.array([e[1] for e in elites])
        mean = elite_arr.mean(axis=0)
        std = np.maximum(elite_arr.std(axis=0), _STD_FLOOR)
        carry = pool[:elitism]

        if verbose and evals_used % 10 == 0:
            print(f"  [PĀṇini-curric] evals {evals_used:3d}/{budget}  best={best_fit:+.4f}"
                 f"  cap_hi(struct)={cap_hi[list(_STRUCT_IDX)]}")

    # final local-refinement phase: IDENTICAL to panini_search, full domain
    # (curriculum only governs the main CEM loop above)
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
# Schedule sweep (cheap, small n_seeds/budget) to pick a default unlock_frac
# ---------------------------------------------------------------------------

def sweep_schedules(n_seeds: int = 10, budget: int = 60,
                    unlock_fracs=(0.3, 0.6, 1.0), verbose: bool = True) -> Dict:
    from experiments.m_search_loop_improved import panini_search
    out = {"plain": []}
    for uf in unlock_fracs:
        out[f"uf={uf}"] = []
    for s in range(n_seeds):
        plain = panini_search(budget, seed=s, verbose=False)["best_fitness"]
        out["plain"].append(plain)
        row = [plain]
        for uf in unlock_fracs:
            r = panini_search_curriculum(budget, seed=s, unlock_frac=uf, verbose=False)["best_fitness"]
            out[f"uf={uf}"].append(r)
            row.append(r)
        if verbose:
            print(f"seed={s}: plain={row[0]:+.4f}  " +
                 "  ".join(f"uf={uf}:{v:+.4f}" for uf, v in zip(unlock_fracs, row[1:])))
    if verbose:
        print()
        for k, v in out.items():
            arr = np.array(v)
            print(f"{k:<12} mean={arr.mean():+.4f}  std={arr.std():.4f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.sweep:
        res = sweep_schedules(n_seeds=args.n_seeds, budget=args.budget)
        if args.out:
            with open(args.out, "w") as f:
                json.dump({k: list(map(float, v)) for k, v in res.items()}, f, indent=2)

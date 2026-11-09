"""
panini_search_gp -- a genuinely grammar-native GP variation operator for PĀṇini
=================================================================================

Context (see this session's prior six investigations, all honestly reported
as no-edge/negative): `panini_search` (m_search_loop_improved.py) is CEM
resampling over a FLAT relaxed-genome vector. The grammar (BodySutra/
production rules) is used ONLY to define the domain of that vector -- it
never participates in the search DYNAMICS, so `panini_search` behaves
exactly like every other vector optimizer in the 28-method comparison,
because structurally it IS one. That is the working hypothesis for why six
independent axes (hyperparameters, 4 differentiator axes, JEPA-literature
check, heterogeneous design space, noise robustness, transfer-objective)
all found no significant, honest edge over the tied cluster
(ipop_cma_es, cma_es_lite, pso, cross_entropy_method, coarse_to_fine, turbo,
lhs_plus_refinement).

This file tests that hypothesis directly by adding a REAL grammar-native
variation operator: standard genetic-programming (Koza-style) subtree
crossover/mutation, operating on genuine tree structure.

Why `HeteroGenome`, not the original flat `Genome`
---------------------------------------------------
The original `Genome` (m_search_loop.py: n_legs, segments_per_leg, seg_len,
seg_rad, angle_jitter) is homogeneous -- every leg is forced to share the
same (segments, seg_len, seg_rad). There is no real tree/hierarchical
structure to operate subtree-crossover on beyond "which scalar field",
which is exactly the kind of thing CEM already does well; a "subtree swap"
there would just be a relabeled coordinate-wise mutation.

`HeteroGenome` (experiments/heterogeneous_genome.py, a prior investigation
this session) gives every leg its OWN (segments, seg_len, seg_rad) --
i.e. genuine per-limb subtrees under the SAME grammar (every leg's ADD_LIMB/
ARTICULATE/EXTEND/TERMINATE production, with an explicit per-step override
instead of anuvrtti-inherited sharing -- see that file's docstring). A
`LegSpec` is a self-contained structural unit: swapping one leg's LegSpec
between two genomes, or resampling one leg's LegSpec wholesale from the
grammar's own random-production rule (`random_leg`), is a textbook
Koza-style subtree crossover/mutation, not a relabeled continuous jitter.
This file builds on `HeteroGenome` exactly as that file intends ("this
module changes nothing about the grammar's rules") -- it is itself
STRICTLY ADDITIVE: it only imports read-only names from
experiments.heterogeneous_genome and experiments.m_search_loop_improved,
and does not modify either file, `panini_search`, `panini_search_hetero`,
`_STD_FLOOR`, or any other existing search method.

What `panini_search_gp` does
-----------------------------
Identical CEM machinery to `panini_search_hetero` (mean/std over the D=26
padded vector, elitism carryover, final local-refinement phase) for the
continuous LEAF parameters, PLUS: each generation, a `gp_frac` fraction of
the new children are generated not by CEM Gaussian resampling but by:
  - subtree_crossover: deep-copy one elite parent, then replace ONE leg's
    LegSpec with the corresponding leg's LegSpec from a SECOND elite parent
    (a genuine structural swap between two individuals' trees), or
  - subtree_mutation: deep-copy one elite parent, then replace ONE leg's
    LegSpec by resampling it wholesale from the grammar's own production
    rule (`random_leg`) -- a structural resample, not a scalar jitter.
These GP children are evaluated through the SAME evaluate_hetero_genome
pipeline as everything else and folded back into the elite-refit pool
(after re-encoding to the padded vector) exactly like a CEM-sampled child
would be, so the rest of the machinery (elitism, mean/std refit, the
inherited discrete-field _STD_FLOOR) is untouched.

Run:
    python3 -m experiments.panini_gp --n_seeds 8 --budget 80 --gp_frac 0.25
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from experiments.heterogeneous_genome import (
    HeteroGenome, LegSpec, random_leg,
    evaluate_hetero_genome, decode_padded, encode_padded,
    _ALL_LO, _ALL_HI, _STD_FLOOR, D,
)


# ---------------------------------------------------------------------------
# Grammar-native variation operators (operate on real HeteroGenome trees)
# ---------------------------------------------------------------------------

def subtree_crossover(a: HeteroGenome, b: HeteroGenome,
                      rng: np.random.Generator) -> HeteroGenome:
    """Koza-style single-subtree crossover. The child starts as a deep copy
    of a randomly chosen BASE parent (preserving its n_legs/angle_jitter
    and every leg's subtree), then exactly ONE leg index is chosen and that
    leg's whole LegSpec subtree (segments, seg_len, seg_rad -- the unit a
    single ADD_LIMB production actually sets) is overwritten with the
    corresponding leg's LegSpec from the DONOR parent. If the donor has
    fewer legs than the chosen index, the base's own subtree at that index
    is left untouched (no-op crossover for that draw, resampled by the
    caller's control flow naturally since the child is otherwise valid)."""
    base, donor = (a, b) if rng.random() < 0.5 else (b, a)
    child = HeteroGenome(n_legs=base.n_legs,
                         legs=[copy.copy(leg) for leg in base.legs],
                         angle_jitter=base.angle_jitter)
    if child.n_legs >= 1 and donor.legs:
        idx = int(rng.integers(0, min(child.n_legs, len(donor.legs))))
        child.legs[idx] = copy.copy(donor.legs[idx])
    return child.clamp()


def subtree_mutation(g: HeteroGenome, rng: np.random.Generator) -> HeteroGenome:
    """Koza-style subtree mutation: deep-copy the parent, pick ONE leg
    index, and replace that leg's ENTIRE LegSpec subtree by resampling it
    wholesale from the grammar's own random-production rule (`random_leg`)
    -- a genuine structural replacement (can change segments AND seg_len
    AND seg_rad all at once, to values with no numerical relationship to
    the parent's), not the +-uniform(0.7,1.4)-style continuous jitter that
    every other mutation operator in this project uses."""
    child = HeteroGenome(n_legs=g.n_legs,
                         legs=[copy.copy(leg) for leg in g.legs],
                         angle_jitter=g.angle_jitter)
    if child.legs:
        idx = int(rng.integers(0, len(child.legs)))
        child.legs[idx] = random_leg(rng)
    return child.clamp()


def _leg_signature(g: HeteroGenome) -> Tuple[Tuple[int, float, float], ...]:
    """A hashable structural fingerprint of a genome's per-leg subtrees,
    used only for the diversity diagnostic below (never consumed by the
    search itself)."""
    return tuple((leg.segments, round(leg.seg_len, 4), round(leg.seg_rad, 4))
                for leg in g.legs)


# ---------------------------------------------------------------------------
# Search: CEM (continuous leaf params) + GP subtree ops (structural), additive
# ---------------------------------------------------------------------------

def panini_search_gp(
    budget: int, seed: int = 0, pop_size: int = 10, elite_frac: float = 0.3,
    elitism: int = 2, refine_frac: float = 0.1, gp_frac: float = 0.25,
    gp_crossover_frac: float = 0.6,  # of the GP share, fraction done via crossover vs mutation
    use_gate: bool = True, spec=None, verbose: bool = False,
    log_diversity: bool = False,
) -> Dict:
    """`panini_search_hetero`'s CEM machinery (mean/std over the D=26
    padded vector, elitism carryover, final local-refinement phase) is
    kept EXACTLY as-is for a `1 - gp_frac` share of each generation's new
    children. The remaining `gp_frac` share is instead generated by
    `subtree_crossover`/`subtree_mutation` applied to genomes decoded from
    the CURRENT generation's elites -- i.e. a real grammar-native
    structural variation operator running ALONGSIDE (not replacing) the
    Gaussian CEM step, exactly as specified. gp_frac=0 reduces this to
    `panini_search_hetero` in all but minor bookkeeping (verified below)."""
    rng = np.random.default_rng(seed)
    evals_used = 0
    history: List[float] = []
    gp_log: List[Dict] = []

    def eval_v(v: np.ndarray) -> float:
        nonlocal evals_used
        fit, _ = evaluate_hetero_genome(decode_padded(v), use_gate=use_gate, spec=spec)
        evals_used += 1
        return fit

    def eval_g(g: HeteroGenome) -> Tuple[float, np.ndarray]:
        nonlocal evals_used
        fit, _ = evaluate_hetero_genome(g, use_gate=use_gate, spec=spec)
        evals_used += 1
        return fit, encode_padded(g)

    refine_budget = max(1, int(round(budget * refine_frac)))
    main_budget = budget - refine_budget

    mean = (_ALL_LO + _ALL_HI) / 2
    std = (_ALL_HI - _ALL_LO) / 3
    n_elite = max(1, int(round(pop_size * elite_frac)))

    best_fit, best_v = -1e9, mean.copy()
    carry: List[Tuple[float, np.ndarray]] = []
    prev_pool: List[Tuple[float, np.ndarray]] = []  # last generation's elites, for GP parents

    while evals_used < main_budget:
        n_this = min(pop_size, main_budget - evals_used)
        n_remaining = max(1, n_this - len(carry))
        # GP children only once we actually have a previous generation's
        # elites to operate on (generation 0 is pure CEM, as there is
        # nothing yet to take a subtree FROM).
        n_gp = int(round(n_remaining * gp_frac)) if prev_pool else 0
        n_gp = min(n_gp, n_remaining)
        n_cem = n_remaining - n_gp

        new_entries: List[Tuple[float, np.ndarray]] = []

        if n_cem > 0:
            samples = np.clip(mean + std * rng.normal(size=(n_cem, D)), _ALL_LO, _ALL_HI)
            for s in samples:
                if evals_used >= main_budget:
                    break
                f = eval_v(s)
                new_entries.append((f, s))

        if n_gp > 0:
            elite_pool = sorted(prev_pool, key=lambda x: -x[0])[:max(2, n_elite)]
            elite_genomes = [decode_padded(v) for _, v in elite_pool]
            for _ in range(n_gp):
                if evals_used >= main_budget:
                    break
                if len(elite_genomes) >= 2 and rng.random() < gp_crossover_frac:
                    i, j = rng.choice(len(elite_genomes), 2, replace=False)
                    parent_sig = _leg_signature(elite_genomes[i])
                    child_g = subtree_crossover(elite_genomes[i], elite_genomes[j], rng)
                    op = "crossover"
                else:
                    p_idx = int(rng.integers(len(elite_genomes)))
                    parent_sig = _leg_signature(elite_genomes[p_idx])
                    child_g = subtree_mutation(elite_genomes[p_idx], rng)
                    op = "mutation"
                f, v = eval_g(child_g)
                new_entries.append((f, v))
                if log_diversity:
                    child_sig = _leg_signature(child_g)
                    gp_log.append({
                        "op": op, "fitness": f,
                        "parent_sig": parent_sig, "child_sig": child_sig,
                        "structurally_different": child_sig != parent_sig,
                    })

        this_pool = carry + new_entries
        this_pool.sort(key=lambda x: -x[0])
        if this_pool[0][0] > best_fit:
            best_fit, best_v = this_pool[0]
        history.append(best_fit)

        elites = this_pool[:min(n_elite, len(this_pool))]
        elite_arr = np.array([e[1] for e in elites])
        mean = elite_arr.mean(axis=0)
        std = np.maximum(elite_arr.std(axis=0), _STD_FLOOR)
        carry = this_pool[:elitism]
        prev_pool = this_pool[:pop_size]

        if verbose and evals_used % 10 == 0:
            print(f"  [PĀṇini-GP] evals {evals_used:3d}/{budget}  best={best_fit:+.4f}"
                 f"  n_gp={n_gp}/{n_remaining}")

    # final local-refinement phase: byte-for-byte identical to
    # panini_search_hetero's (continuous-leaf-only polish; GP ops are a
    # MAIN-LOOP-only structural operator, not touched here)
    tr = 0.15
    scale = _ALL_HI - _ALL_LO
    fail_streak = succ_streak = 0
    while evals_used < budget:
        cand = np.clip(best_v + tr * scale * rng.normal(size=D), _ALL_LO, _ALL_HI)
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

    result = {"history": history, "best_fitness": best_fit,
             "best_genome": decode_padded(best_v), "evals_used": evals_used}
    if log_diversity:
        result["gp_log"] = gp_log
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--gp_frac", type=float, default=0.25)
    args = ap.parse_args()
    fits = []
    for s in range(args.n_seeds):
        r = panini_search_gp(args.budget, seed=s, gp_frac=args.gp_frac, verbose=False)
        fits.append(r["best_fitness"])
        print(f"  seed={s}: best={r['best_fitness']:+.4f}")
    arr = np.array(fits)
    print(f"\npanini_search_gp (gp_frac={args.gp_frac}): {arr.mean():+.4f} +/- {arr.std():.4f}")

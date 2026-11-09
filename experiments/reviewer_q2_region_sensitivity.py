"""
Reviewer Q2 — Region-Family Sensitivity + Order-Embedding Baseline
======================================================================

Two things in one file:

1. A CORRECTED synthetic rule set. Last session's deeper-hierarchy test
   confounded specificity with raw feature-count overlap (override rules
   were literal feature supersets of their general rule, so Euclidean
   nearest-neighbor trivially tracked specificity by counting shared bits).
   Fixed here: general rules carry IRRELEVANT extra features their
   overrides don't share, so feature-count and specificity are no longer
   perfectly correlated -- a fairer test of whether containment structure
   is doing real work.

2. An order-embedding baseline (Vendrov et al. 2015, already in the
   paper's citations): each rule is a POINT x in R^d (not a box), and
   generality is encoded as coordinatewise domination -- x_general <=
   x_specific in every coordinate means "general subsumes specific". This
   is trained with a margin loss enforcing that relation for known pairs,
   exactly the "learned point-embedding + margin-based ordering" baseline
   the reviewer explicitly asked for, compared against boxes on the SAME
   corrected rule set and the SAME held-out transitive pairs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .box_embed_np import BoxBankNP, softplus, softplus_grad

FEATURE_DIM = 10

# CORRECTED rule set: general rules carry an extra IRRELEVANT feature
# (index 8 or 9) that their overrides do NOT share, decoupling raw
# feature-count/overlap from the specificity relation.
RULES: List[Tuple[str, List[int], int]] = [
    ("r0_generic", [0, 8], 0),                  # extra irrelevant feature 8
    ("r1_tool", [0, 1], 1),
    ("r2_tool_fragile", [0, 1, 2], 2),
    ("r3_tool_fragile_precision", [0, 1, 2, 3], 3),
    ("s0_generic", [4, 9], 0),                   # extra irrelevant feature 9
    ("s1_specific", [4, 5], 1),
    ("s2_more_specific", [4, 5, 6], 2),
]

TRAIN_PAIRS = [
    ("r1_tool", "r0_generic"),
    ("r2_tool_fragile", "r1_tool"),
    ("r3_tool_fragile_precision", "r2_tool_fragile"),
    ("s1_specific", "s0_generic"),
    ("s2_more_specific", "s1_specific"),
]
ALL_PAIRS = TRAIN_PAIRS + [
    ("r2_tool_fragile", "r0_generic"),
    ("r3_tool_fragile_precision", "r1_tool"),
    ("r3_tool_fragile_precision", "r0_generic"),
    ("s2_more_specific", "s0_generic"),
]
HELD_OUT = [p for p in ALL_PAIRS if p not in TRAIN_PAIRS]


def rule_feature_vector(feats: List[int]) -> np.ndarray:
    v = np.zeros(FEATURE_DIM)
    v[feats] = 1.0
    return v


def sample_states(rng: np.random.Generator, n: int, note_generic_extra: bool = True
                  ) -> List[Tuple[str, np.ndarray]]:
    fires = []
    for _ in range(n):
        active = (rng.random(FEATURE_DIM) < 0.3).astype(float)
        satisfied = [(rid, spec) for rid, feats, spec in RULES
                    if np.all(active[feats] == 1.0)]
        if not satisfied:
            continue
        winner = max(satisfied, key=lambda x: x[1])[0]
        fires.append((winner, active.copy()))
    return fires


# ---------------------------------------------------------------------------
# Order embeddings (Vendrov et al. 2015): point x, generality = coordinatewise
# domination. Penalize violations of x_general <= x_specific componentwise.
# ---------------------------------------------------------------------------

class OrderEmbeddingBank:
    def __init__(self, rule_ids: List[str], dim: int, seed: int = 0):
        self.rule_ids = list(rule_ids)
        self.index = {r: i for i, r in enumerate(rule_ids)}
        rng = np.random.default_rng(seed)
        self.x = np.abs(rng.normal(0, 1.0, size=(len(rule_ids), dim)))  # >=0 convention
        self._m = np.zeros_like(self.x); self._v = np.zeros_like(self.x); self._t = 0

    def point(self, rid: str) -> np.ndarray:
        return self.x[self.index[rid]]

    def order_violation(self, specific_id: str, general_id: str) -> float:
        """Penalize where x_general > x_specific (general should be <=
        specific in every coordinate -- general subsumes/lies 'below')."""
        xs, xg = self.point(specific_id), self.point(general_id)
        return float(np.sum(np.maximum(xg - xs, 0.0) ** 2))

    def train_step(self, fires, pairs, lr=0.05, margin_weight=5.0):
        grad = np.zeros_like(self.x)
        n = max(1, len(fires))
        for rid, point in fires:
            i = self.index[rid]
            # pull embedding toward representing states it fires on: encode
            # the STATE's active features as a lower bound the rule's point
            # must dominate (order-embedding "reversed image" convention)
            diff = np.maximum(point[:self.x.shape[1]] - self.x[i], 0.0)
            grad[i] += -2 * diff / n
        for spec_id, gen_id in pairs:
            si, gi = self.index[spec_id], self.index[gen_id]
            viol = np.maximum(self.x[gi] - self.x[si], 0.0)
            grad[si] += -2 * margin_weight * viol
            grad[gi] += 2 * margin_weight * viol
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        self._m = b1 * self._m + (1 - b1) * grad
        self._v = b2 * self._v + (1 - b2) * (grad ** 2)
        mhat = self._m / (1 - b1 ** self._t)
        vhat = self._v / (1 - b2 ** self._t)
        self.x = np.maximum(self.x - lr * mhat / (np.sqrt(vhat) + eps), 0.0)


def order_embedding_pick(bank: OrderEmbeddingBank, candidates: List[str]) -> str:
    """Among candidates, the order-embedding analogue of 'most specific':
    the point with the SMALLEST L1 norm dominates the fewest others, i.e.
    lies furthest 'up' the order -- by convention here, largest norm =
    most specific (accumulates more coordinatewise constraints)."""
    return max(candidates, key=lambda c: float(np.sum(bank.point(c))))


def run(n_states: int = 500, epochs: int = 400, seed: int = 0,
       width_floors: List[float] = (0.01, 0.05, 0.1, 0.2),
       verbose: bool = True) -> Dict:
    rng = np.random.default_rng(seed)
    rule_ids = [r[0] for r in RULES]
    fires = sample_states(rng, n_states)
    eval_fires = sample_states(np.random.default_rng(seed + 1), 300)

    # --- Part A: box embeddings at several regularization (width-floor) settings ---
    box_results = {}
    for wf in width_floors:
        bank = BoxBankNP(rule_ids, dim=FEATURE_DIM, width_floor=wf, seed=seed)
        for rid, feats, _ in RULES:
            bank.center[bank.index[rid]] = rule_feature_vector(feats)
        for _ in range(epochs):
            bank.train_step(fires, TRAIN_PAIRS, lr=0.05)
        contain_ok = sum(
            bool(np.all(bank.bounds(s)[0] >= bank.bounds(g)[0] - 1e-4)
                and np.all(bank.bounds(s)[1] <= bank.bounds(g)[1] + 1e-4))
            for s, g in ALL_PAIRS) / len(ALL_PAIRS)
        held_out_ok = sum(
            bool(np.all(bank.bounds(s)[0] >= bank.bounds(g)[0] - 1e-4)
                and np.all(bank.bounds(s)[1] <= bank.bounds(g)[1] + 1e-4))
            for s, g in HELD_OUT) / max(1, len(HELD_OUT))
        box_results[wf] = {"contain_acc": contain_ok, "held_out_acc": held_out_ok}

    # --- Part B: order-embedding baseline, same rule set, same held-out pairs ---
    oe = OrderEmbeddingBank(rule_ids, dim=FEATURE_DIM, seed=seed)
    for _ in range(epochs):
        oe.train_step(fires, TRAIN_PAIRS, lr=0.05)
    oe_contain_ok = sum(oe.order_violation(s, g) < 1e-3 for s, g in ALL_PAIRS) / len(ALL_PAIRS)
    oe_held_out_ok = sum(oe.order_violation(s, g) < 1e-3 for s, g in HELD_OUT) / max(1, len(HELD_OUT))

    # decision agreement, corrected rule set, box vs order-embedding vs
    # Euclidean-on-corrected-features
    bank_best = BoxBankNP(rule_ids, dim=FEATURE_DIM, width_floor=0.05, seed=seed)
    for rid, feats, _ in RULES:
        bank_best.center[bank_best.index[rid]] = rule_feature_vector(feats)
    for _ in range(epochs):
        bank_best.train_step(fires, TRAIN_PAIRS, lr=0.05)

    prototypes = {rid: rule_feature_vector(feats) for rid, feats, _ in RULES}
    box_agree, oe_agree, euclid_agree, n_conflict = 0, 0, 0, 0
    for winner_true, point in eval_fires:
        satisfied = [rid for rid, feats, _ in RULES if np.all(point[feats] == 1.0)]
        if len(satisfied) < 2:
            continue
        n_conflict += 1
        box_pick = min(satisfied, key=lambda c: bank_best.log_volume(c))
        oe_pick = order_embedding_pick(oe, satisfied)
        euclid_pick = min(satisfied, key=lambda c: float(np.linalg.norm(prototypes[c] - point)))
        box_agree += int(box_pick == winner_true)
        oe_agree += int(oe_pick == winner_true)
        euclid_agree += int(euclid_pick == winner_true)

    if verbose:
        print("Part A -- box-embedding regularization sensitivity "
             "(width_floor sweep)\n")
        print(f"{'width_floor':<14}{'containment acc':>18}{'held-out acc':>15}")
        for wf, r in box_results.items():
            print(f"{wf:<14}{r['contain_acc']:>18.3f}{r['held_out_acc']:>15.3f}")

        print(f"\nPart B -- order-embedding baseline (Vendrov-style), "
             f"same corrected rule set:")
        print(f"  containment (order-violation) accuracy: {oe_contain_ok:.3f}")
        print(f"  held-out transitive accuracy:            {oe_held_out_ok:.3f}")

        print(f"\nDecision agreement on {n_conflict} held-out conflicts "
             f"(CORRECTED rule set -- feature-count confound removed):")
        print(f"  box-containment pick : {box_agree/n_conflict:.3f}")
        print(f"  order-embedding pick : {oe_agree/n_conflict:.3f}")
        print(f"  Euclidean-nearest pick: {euclid_agree/n_conflict:.3f}  "
             f"(simpler-alternative control)")
    return {"box_by_width_floor": box_results,
           "order_embedding": {"contain_acc": oe_contain_ok, "held_out_acc": oe_held_out_ok},
           "agreement": {"box": box_agree/max(1,n_conflict),
                        "order_embedding": oe_agree/max(1,n_conflict),
                        "euclidean": euclid_agree/max(1,n_conflict)}}


if __name__ == "__main__":
    run()

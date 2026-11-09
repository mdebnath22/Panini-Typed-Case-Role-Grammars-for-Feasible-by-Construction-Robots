"""
Multi-seed robustness check for the containment-agreement claim in
Table~tab:boxes / Section~sec:boxes.

The original result (box agreement = 1.000 vs. a Euclidean control at
0.719) was reported from a single training seed (seed=0, the default used
throughout box_embed_np.py and reviewer_q2_region_sensitivity.py). A
single favorable construction does not establish robustness, so this
script reruns the identical decision-agreement protocol
(experiments.reviewer_q2_region_sensitivity.run) at 10 independent
training seeds and reports mean +/- std instead of one number.

Run:
    python3 -m experiments.multiseed_containment
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from experiments.reviewer_q2_region_sensitivity import run


def run_multiseed(seeds: List[int] = list(range(10)), verbose: bool = True) -> Dict:
    box, oe, euc, held_out_box = [], [], [], []
    for s in seeds:
        r = run(seed=s, verbose=False)
        box.append(r["agreement"]["box"])
        oe.append(r["agreement"]["order_embedding"])
        euc.append(r["agreement"]["euclidean"])
        held_out_box.append(r["box_by_width_floor"][0.05]["held_out_acc"])

    def stat(arr):
        a = np.array(arr)
        return dict(mean=float(a.mean()), std=float(a.std()),
                   min=float(a.min()), max=float(a.max()))

    out = dict(seeds=list(seeds),
               box_decision=stat(box), order_embedding_decision=stat(oe),
               euclidean_decision=stat(euc), box_held_out_containment=stat(held_out_box))
    if verbose:
        print(f"{len(seeds)} independent training seeds: {list(seeds)}\n")
        for name, s in [("box-containment decision", out["box_decision"]),
                       ("order-embedding decision", out["order_embedding_decision"]),
                       ("Euclidean-nearest (control)", out["euclidean_decision"]),
                       ("box held-out containment", out["box_held_out_containment"])]:
            print(f"{name:<28} mean={s['mean']:.3f} std={s['std']:.3f} "
                 f"min={s['min']:.3f} max={s['max']:.3f}  (n={len(seeds)} seeds)")
    return out


if __name__ == "__main__":
    run_multiseed()

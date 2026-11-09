"""
Reviewer Q3 — Diagnosing "Lower MSE on Held-Out Than In-Distribution"
========================================================================

The paper's Table 4 shows twist MSE of 0.0436 in-distribution vs 0.0120
held-out -- backwards from what generalization usually looks like, and the
reviewer is right to ask whether this is a data-complexity artifact, a
train/validation mismatch, or a labeling error. This runs the SAME
manipulation-side compositional-generalization experiment across multiple
seeds and adds the specific diagnostic the anomaly calls for: a per-
template and per-derivation-length breakdown, since the most likely
innocent explanation is that held-out derivations happen to be
SHORTER/SIMPLER on average (fewer composed twists means less opportunity
for error to accumulate), which is a real confound, not a labeling bug --
but it needs to be checked, not assumed.

This depends on the manipulation-side ptg/sutra.py, ptg/grammar.py,
gmm/phi_seed.py, gmm/phi_neural.py from your ORIGINAL panini-motion repo
(not the pm/ morphology-only package) -- run it there.

Run:
    python3 -m experiments.reviewer_q3_ood_diagnostic --n_seeds 5
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

from gmm.phi_neural import PhiNeural, T2I, jepa_phi_loss, tokenize_sequence
from gmm.phi_seed import PhiSeed
from gmm.se3 import compose, exp_se3, geodesic_distance
from experiments.scenes import sample_derivations
from experiments.h1_compgen import (TRAIN_TEMPLATES, TEST_TEMPLATES,
                                    build_batch)


def evaluate_per_derivation(model: PhiNeural, pairs, phi: PhiSeed) -> List[Dict]:
    """Same as h1_compgen.evaluate() but returns PER-DERIVATION records
    (length, twist MSE, endpoint error) instead of only the aggregate mean
    -- needed to check whether the anomaly is explained by derivation
    length rather than by a genuine ID/OOD difference in difficulty."""
    model.eval()
    records = []
    with torch.no_grad():
        for deriv, ee in pairs:
            x = torch.tensor(tokenize_sequence(deriv, ee)).unsqueeze(0)
            out = model(x)
            xi_hat = out["twists"][0].numpy().astype(np.float64)
            res = phi.map_derivation(deriv, ee)
            xi_true = res["twists"]
            twist_mse = float(np.mean((xi_hat - xi_true) ** 2))
            T_hat = compose(*[exp_se3(x_) for x_ in xi_hat])
            endpoint_err = geodesic_distance(ee @ T_hat, res["final"])
            records.append({"length": len(deriv), "twist_mse": twist_mse,
                           "endpoint_err": endpoint_err})
    return records


def run(n_seeds: int = 5, n_train: int = 600, n_test: int = 100,
       epochs: int = 60, verbose: bool = True) -> Dict:
    phi = PhiSeed()
    all_id_mse, all_ood_mse = [], []
    all_id_records, all_ood_records = [], []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        train_pairs = sample_derivations(n_train, seed=seed,
                                         goal_templates=TRAIN_TEMPLATES)
        test_id = sample_derivations(n_test, seed=seed + 100,
                                     goal_templates=TRAIN_TEMPLATES)
        test_ood = sample_derivations(n_test, seed=seed + 200,
                                      goal_templates=TEST_TEMPLATES)

        model = PhiNeural()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        bs = 64
        for ep in range(epochs):
            perm = np.random.permutation(len(train_pairs))
            for i in range(0, len(train_pairs), bs):
                batch = [train_pairs[j] for j in perm[i:i + bs]]
                X, XI, ID, M = build_batch(batch, phi)
                loss, _ = jepa_phi_loss(model, X, XI, ID)
                opt.zero_grad(); loss.backward(); opt.step()
                model.ema_update()

        id_records = evaluate_per_derivation(model, test_id, phi)
        ood_records = evaluate_per_derivation(model, test_ood, phi)
        id_mse = float(np.mean([r["twist_mse"] for r in id_records]))
        ood_mse = float(np.mean([r["twist_mse"] for r in ood_records]))
        all_id_mse.append(id_mse); all_ood_mse.append(ood_mse)
        all_id_records += id_records; all_ood_records += ood_records

        if verbose:
            id_len = np.mean([r["length"] for r in id_records])
            ood_len = np.mean([r["length"] for r in ood_records])
            print(f"seed={seed}: ID mse={id_mse:.4f} (mean len={id_len:.1f})  "
                 f"OOD mse={ood_mse:.4f} (mean len={ood_len:.1f})")

    # length-controlled comparison: bucket both ID and OOD records by
    # derivation length and compare MSE within matched-length buckets --
    # this isolates whether the anomaly survives once length is controlled
    if verbose:
        print(f"\nAcross {n_seeds} seeds: ID={np.mean(all_id_mse):.4f}"
             f"+/-{np.std(all_id_mse):.4f}  "
             f"OOD={np.mean(all_ood_mse):.4f}+/-{np.std(all_ood_mse):.4f}")
        anomaly_persists = np.mean(all_ood_mse) < np.mean(all_id_mse)
        print(f"anomaly (OOD < ID) persists across seeds: {anomaly_persists}")

        print("\nLength-controlled breakdown (mean twist MSE by derivation "
             "length, pooled across seeds):")
        lengths = sorted(set(r["length"] for r in all_id_records + all_ood_records))
        print(f"{'length':<10}{'ID mse (n)':>18}{'OOD mse (n)':>18}")
        for L in lengths:
            id_at_L = [r["twist_mse"] for r in all_id_records if r["length"] == L]
            ood_at_L = [r["twist_mse"] for r in all_ood_records if r["length"] == L]
            id_str = f"{np.mean(id_at_L):.4f} ({len(id_at_L)})" if id_at_L else "--"
            ood_str = f"{np.mean(ood_at_L):.4f} ({len(ood_at_L)})" if ood_at_L else "--"
            print(f"{L:<10}{id_str:>18}{ood_str:>18}")
        print("\nInterpretation: if OOD derivations are systematically "
             "SHORTER on average, lower OOD MSE is expected (less twist-"
             "composition error to accumulate) and is NOT evidence against "
             "compositional generalization -- it would instead mean the "
             "ID/OOD comparison needs to control for length before drawing "
             "any conclusion. If OOD MSE is still lower WITHIN matched-"
             "length buckets, that points to a genuine template-difficulty "
             "or data artifact requiring further investigation.")
    return {"id_mse": all_id_mse, "ood_mse": all_ood_mse}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    args = ap.parse_args()
    run(n_seeds=args.n_seeds)

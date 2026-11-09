"""
Neural Φ Training + H1 Compositional Generalization
===================================================

Phase 2 (learning Φ) and H1 in one harness:

  1. Distill the closed-form seed Φ into the JEPA-style PhiNeural on
     TRAIN rule combinations only.
  2. Evaluate twist prediction + composed-endpoint geodesic error on
     HELD-OUT combinations of the same primitive rules (H1), plus the
     Φ⁻¹ symbolic-readout accuracy.

Train templates cover pick-place and tool-use orderings; the held-out
template is a *novel composition* (e.g. re-grasp / place-then-retract)
never seen during training.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from gmm.phi_neural import (PhiNeural, T2I, jepa_phi_loss, tokenize_sequence)
from gmm.phi_seed import PhiSeed
from gmm.se3 import compose, exp_se3, geodesic_distance
from .scenes import sample_derivations

TRAIN_TEMPLATES = [
    ["APPROACH", "GRASP", "LIFT", "TRANSPORT", "PLACE"],
    ["APPROACH", "GRASP", "LIFT"],
    ["APPROACH", "GRASP", "LIFT", "ROTATE_TOOL"],
]
# held-out compositions of the SAME primitives (H1)
TEST_TEMPLATES = [
    ["APPROACH", "GRASP", "LIFT", "TRANSPORT", "PLACE",
     "APPROACH", "GRASP", "LIFT"],                       # sequential re-pick
    ["APPROACH", "GRASP", "LIFT", "PLACE"],              # skip transport
]


def build_batch(pairs, phi: PhiSeed, max_len: int = 16):
    xs, xis, ids, lens = [], [], [], []
    for deriv, ee in pairs:
        x = tokenize_sequence(deriv, ee)
        res = phi.map_derivation(deriv, ee)
        xs.append(torch.tensor(x))
        xis.append(torch.tensor(res["twists"], dtype=torch.float32))
        ids.append(torch.tensor(
            [T2I[a.sutra.transformation] for a in deriv]))
        lens.append(len(deriv))
    L = max(lens)
    B = len(pairs)
    X = torch.zeros(B, L, xs[0].shape[-1])
    XI = torch.zeros(B, L, 6)
    ID = torch.zeros(B, L, dtype=torch.long)
    M = torch.zeros(B, L, dtype=torch.bool)
    for i, (x, xi, rid, l) in enumerate(zip(xs, xis, ids, lens)):
        X[i, :l], XI[i, :l], ID[i, :l], M[i, :l] = x, xi, rid, True
    return X, XI, ID, M


def evaluate(model: PhiNeural, pairs, phi: PhiSeed):
    model.eval()
    twist_err, endpoint_err, inv_acc = [], [], []
    with torch.no_grad():
        for deriv, ee in pairs:
            x = torch.tensor(tokenize_sequence(deriv, ee)).unsqueeze(0)
            out = model(x)
            xi_hat = out["twists"][0].numpy().astype(np.float64)
            res = phi.map_derivation(deriv, ee)
            xi_true = res["twists"]
            twist_err.append(float(np.mean((xi_hat - xi_true) ** 2)))
            T_hat = compose(*[exp_se3(x_) for x_ in xi_hat])
            endpoint_err.append(geodesic_distance(ee @ T_hat, res["final"]))
            pred = out["rule_logits"][0].argmax(-1).numpy()
            true = np.array([T2I[a.sutra.transformation] for a in deriv])
            inv_acc.append(float((pred == true).mean()))
    return (float(np.mean(twist_err)), float(np.mean(endpoint_err)),
            float(np.mean(inv_acc)))


def run(n_train: int = 600, n_test: int = 100, epochs: int = 60,
        seed: int = 0, verbose: bool = True):
    torch.manual_seed(seed)
    phi = PhiSeed()
    train_pairs = sample_derivations(n_train, seed=seed,
                                     goal_templates=TRAIN_TEMPLATES)
    test_id = sample_derivations(n_test, seed=seed + 1,
                                 goal_templates=TRAIN_TEMPLATES)
    test_ood = sample_derivations(n_test, seed=seed + 2,
                                  goal_templates=TEST_TEMPLATES)

    model = PhiNeural()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 64
    for ep in range(epochs):
        perm = np.random.permutation(len(train_pairs))
        tot = 0.0
        for i in range(0, len(train_pairs), bs):
            batch = [train_pairs[j] for j in perm[i:i + bs]]
            X, XI, ID, M = build_batch(batch, phi)
            loss, parts = jepa_phi_loss(model, X, XI, ID)
            opt.zero_grad(); loss.backward(); opt.step()
            model.ema_update()
            tot += loss.item()
        if verbose and (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1:3d}  loss {tot / max(1, len(train_pairs)//bs):.4f}"
                  f"  (jepa {parts['jepa']:.4f} twist {parts['twist']:.4f}"
                  f" inv {parts['inv']:.4f})")

    tw_id, ep_id, inv_id = evaluate(model, test_id, phi)
    tw_ood, ep_ood, inv_ood = evaluate(model, test_ood, phi)
    if verbose:
        print("\nH1 — Compositional generalization (neural Φ)")
        print(f"{'split':<28}{'twist MSE':>12}{'endpoint d_geo':>16}"
              f"{'Φ⁻¹ acc':>10}")
        print(f"{'in-dist (train templates)':<28}{tw_id:>12.5f}"
              f"{ep_id:>16.4f}{inv_id:>10.3f}")
        print(f"{'OOD (held-out compositions)':<28}{tw_ood:>12.5f}"
              f"{ep_ood:>16.4f}{inv_ood:>10.3f}")
        print(f"generalization gap (endpoint): {ep_ood - ep_id:+.4f}")
    return model, (tw_id, ep_id, inv_id), (tw_ood, ep_ood, inv_ood)


if __name__ == "__main__":
    run()

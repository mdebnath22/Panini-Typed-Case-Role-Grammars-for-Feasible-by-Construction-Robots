"""CPU-side half of the "does the cheap CPG proxy rank bodies the way a
trained RL policy would" experiment -- the single most valuable experiment
in the GPU-training follow-up roadmap. This script does everything that
needs no GPU:

  1. Samples a population of grammar-derived bodies (same genome space as
     the paper's main search, Section "Task-Directed Morphology Search").
  2. Computes each body's cheap CPG-proxy fitness (gmm.task_fitness,
     unmodified -- this is exactly the proxy used throughout the paper).
  3. Compiles each body to MJCF (+ URDF, best-effort) with real catalog
     parts bound (ptg.catalog_binding), so the GPU side gets a buildable
     robot, not an abstract genome.
  4. Writes a self-contained job manifest describing exactly what the
     GPU-side training run must produce and in what format, so the
     analysis script (rank_correlation_analysis.py) can consume it with no
     back-and-forth about schema.

Usage:
    python3 -m experiments.rank_correlation_harness --n_bodies 40 --out_dir results_refit/rank_corr

What this does NOT do: train anything. That half (morphology-conditioned
policy learning on an GPU-parallel simulator -- MJX, Isaac Lab, or Genesis
-- with a terrain curriculum) is the GPU-side counterpart; see the manifest
this script writes for the exact handoff contract.
"""
from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import asdict

import numpy as np

from ptg.body_grammar import BodyGrammar, legged_plan
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody, leg_mount_plan
from gmm.mjcf_compiler import compile_with_bom, compile_urdf
from gmm.task_fitness import evaluate_fitness
from ptg.catalog_binding import bind_catalog
from sulba.feasibility import check_feasibility, FeasibilitySpec


def sample_genome(rng: random.Random) -> dict:
    return {
        "n_legs": rng.randint(4, 8),
        "segments_per_leg": rng.randint(1, 3),
        "seg_len": rng.uniform(0.08, 0.20),
        "seg_rad": rng.uniform(0.012, 0.030),
        "mount_jitter": rng.uniform(0.0, 0.3),
    }


def build_body(genome: dict, seed: int):
    g = BodyGrammar(seed_body_rules(), seed=seed)
    base_angles, xs = leg_mount_plan(genome["n_legs"])
    rng = np.random.RandomState(seed)
    angles = [float(np.clip(a + rng.uniform(-genome["mount_jitter"], genome["mount_jitter"]),
                            np.pi, 2 * np.pi))
             for a in base_angles]
    plan = legged_plan(n_legs=genome["n_legs"], segments_per_leg=genome["segments_per_leg"],
                       mount_angles=angles, mount_x_offsets=xs,
                       seg_len=genome["seg_len"], seg_rad=genome["seg_rad"])
    deriv = g.derive(plan)
    tree = PhiBody().build_tree(deriv)
    return tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bodies", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results_refit/rank_corr")
    ap.add_argument("--task", type=str, default="walk", choices=["walk", "parkour", "waiter"],
                    help="Which curriculum this job targets -- see docs/task_curriculum_schema.md. "
                         "Only 'walk' has a defined GPU-side reward in this handoff today; "
                         "'parkour'/'waiter' are included so the manifest schema is ready, but "
                         "their reward/observation spec in the manifest is a placeholder pending "
                         "the task_spec work (experiments/task_spec_waiter.py).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bodies_dir = os.path.join(args.out_dir, "bodies")
    os.makedirs(bodies_dir, exist_ok=True)

    rng = random.Random(args.seed)
    manifest_bodies = []
    n_feasible = 0

    for i in range(args.n_bodies):
        genome = sample_genome(rng)
        body_id = f"body_{i:04d}"
        try:
            tree = build_body(genome, seed=args.seed * 10_000 + i)
        except Exception as e:
            manifest_bodies.append({"body_id": body_id, "genome": genome, "error": str(e)})
            continue

        geo = check_feasibility(tree, FeasibilitySpec())
        proxy = evaluate_fitness(tree, seed=args.seed)
        bom = bind_catalog(tree)  # mutates tree in place with real parts/masses

        compiled = compile_with_bom(tree, name=body_id, bind_if_unbound=False)
        mjcf_path = os.path.join(bodies_dir, f"{body_id}.xml")
        with open(mjcf_path, "w") as f:
            f.write(compiled["xml"])
        urdf_path = os.path.join(bodies_dir, f"{body_id}.urdf")
        try:
            with open(urdf_path, "w") as f:
                f.write(compile_urdf(tree, name=body_id))
        except Exception:
            urdf_path = None

        if geo.ok:
            n_feasible += 1

        manifest_bodies.append({
            "body_id": body_id,
            "genome": genome,
            "geometric_feasible": geo.ok,
            "proxy_fitness": proxy.fitness,
            "proxy_displacement": proxy.displacement,
            "proxy_fell": proxy.fell,
            "n_actuators": compiled["n_actuators"],
            "bom_total_cost": bom.total_cost,
            "bom_lead_time_days": bom.lead_time_days,
            "mjcf_path": os.path.relpath(mjcf_path, args.out_dir),
            "urdf_path": os.path.relpath(urdf_path, args.out_dir) if urdf_path else None,
        })

    manifest = {
        "task": args.task,
        "n_bodies": args.n_bodies,
        "n_geometrically_feasible": n_feasible,
        "proxy_controller": "fixed sinusoidal CPG (gmm.task_fitness.evaluate_fitness, horizon_steps=1000)",
        "bodies": manifest_bodies,
        "gpu_side_contract": {
            "what_to_produce": (
                "For each body_id in this manifest with geometric_feasible=true, train a "
                "morphology-specific (or morphology-conditioned) locomotion policy on the "
                "body's MJCF (preferred; loads directly in MuJoCo/MJX) or URDF (best-effort "
                "export, validate against your importer first), then write back one scalar "
                "per body -- the metric defined below -- to a results file."
            ),
            "observation_space": "qpos, qvel of the compiled model (mujoco.MjModel from the "
                                 "given MJCF gives exact dims via model.nq, model.nv); add a "
                                 "contact/foot-state vector if your terrain curriculum needs it.",
            "action_space": "one normalized actuator command in [-1, 1] per entry in the "
                            "model's <actuator> block (n_actuators given per body above); each "
                            "maps to a position-servo target, ctrlrange='-1 1' already set in "
                            "the MJCF, forcerange set from the bound actuator's real stall "
                            "torque (ptg.catalog_binding) so sim-exploited infinite-torque "
                            "policies aren't possible.",
            "reward_walk": "forward (+x) CoM displacement per step, minus a small control-"
                          "energy penalty and a fall penalty -- i.e. the same shape as "
                          "gmm.task_fitness.evaluate_fitness's proxy reward, just replacing the "
                          "fixed CPG with a trained policy. Keeping the reward shape identical "
                          "to the proxy is what makes the correlation test meaningful: a "
                          "divergence would otherwise confound 'different reward' with "
                          "'different controller'.",
            "metric_to_report": "mean episode return over the final 10% of training (or an "
                                "evaluation rollout average over >=20 episodes with the frozen "
                                "final policy) -- a single scalar per body_id, matching units "
                                "with proxy_displacement/proxy_fitness above as closely as "
                                "possible (forward displacement in meters, same horizon).",
            "results_file_schema": {
                "path": "rl_results.json, placed in the same --out_dir as this manifest",
                "format": {"body_id": "<str, matches a body_id in this manifest>",
                          "rl_fitness": "<float>",
                          "optional": {"train_env_steps": "<int>", "wall_clock_s": "<float>"}},
                "example": [{"body_id": "body_0000", "rl_fitness": 0.812,
                            "train_env_steps": 2_000_000, "wall_clock_s": 340.5}],
            },
            "then_run": "python3 -m experiments.rank_correlation_analysis --out_dir <same --out_dir>",
        },
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Sampled {args.n_bodies} bodies ({n_feasible} geometrically feasible).")
    print(f"Manifest written to {manifest_path}")
    print(f"MJCF/URDF files written to {bodies_dir}/")
    print("Hand the manifest + bodies/ directory to the GPU-side training run; "
         "see manifest['gpu_side_contract'] for the exact handoff format.")


if __name__ == "__main__":
    main()

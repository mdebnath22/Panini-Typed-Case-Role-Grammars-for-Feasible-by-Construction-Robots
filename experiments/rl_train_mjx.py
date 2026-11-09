"""GPU-parallel RL training for the rank-correlation experiment, using
MuJoCo MJX (JAX) -- the simplest path to "morphology-conditioned policy
learning on a GPU-parallel simulator" that needs no extra simulator
install (Isaac Lab / Genesis are heavier installs; MJX just needs
`mujoco`, `jax`, and `optax`, and loads the exact MJCF files the CPU-side
harness already wrote).

Trains one small MLP policy per body with an evolution-strategy (ES)
update (OpenAI-ES / antithetic-sampling style): no backprop through
contact dynamics (notoriously stiff/discontinuous for policy gradients),
instead the whole population of perturbed policies is rolled out in
parallel via jax.vmap + mjx, which is exactly where a GPU helps -- this
scales with population size and horizon, not with wall-clock per
rollout, the way a sequential CPU loop would.

The reward is deliberately the SAME shape as gmm.task_fitness's CPG-proxy
fitness (displacement - energy_penalty*mean|ctrl| - fall_penalty), so a
correlation gap between this and the proxy isn't confounded by also
having changed the objective -- see rank_correlation_harness.py's
docstring for why that matters.

Usage (small smoke test, works on CPU too, just slow):
    python3 -m experiments.rl_train_mjx --out_dir results_refit/rank_corr \
        --generations 5 --population 32 --horizon_steps 200 --n_bodies_limit 2

Usage (real run, GPU):
    python3 -m experiments.rl_train_mjx --out_dir results_refit/rank_corr \
        --generations 150 --population 256 --horizon_steps 1000

Writes results_refit/rank_corr/rl_results.json in the schema
rank_correlation_analysis.py expects, incrementally (one line of progress
per body, one JSON flush after each body so a crash partway through
doesn't lose completed bodies).
"""
from __future__ import annotations
import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import jax
import jax.numpy as jnp
import optax


def init_policy_params(key, obs_dim: int, act_dim: int, hidden: int = 32):
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (obs_dim, hidden)) * jnp.sqrt(1.0 / obs_dim)
    b1 = jnp.zeros((hidden,))
    w2 = jax.random.normal(k2, (hidden, act_dim)) * jnp.sqrt(1.0 / hidden)
    b2 = jnp.zeros((act_dim,))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def policy_forward(params, obs):
    h = jnp.tanh(obs @ params["w1"] + params["b1"])
    return jnp.tanh(h @ params["w2"] + params["b2"])


def make_rollout_fn(mx_model, horizon_steps: int, dt_control: int,
                    min_torso_height: float, energy_penalty: float):
    import mujoco.mjx as mjx

    def rollout(params, init_data):
        def control_step(carry, _):
            data, energy_sum, min_height, fell, step_idx = carry
            obs = jnp.concatenate([data.qpos, data.qvel])
            ctrl = policy_forward(params, obs)

            def inner_step(d, _):
                d = d.replace(ctrl=ctrl)
                d = mjx.step(mx_model, d)
                return d, None

            data, _ = jax.lax.scan(inner_step, data, None, length=dt_control)
            finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
            torso_z = data.qpos[2]
            min_height = jnp.minimum(min_height, torso_z)
            fell = fell | (torso_z < min_torso_height) | (~finite)
            energy_sum = energy_sum + jnp.mean(jnp.abs(ctrl))
            return (data, energy_sum, min_height, fell, step_idx + 1), None

        n_ctrl_steps = horizon_steps // dt_control
        x0 = init_data.qpos[0]
        carry0 = (init_data, 0.0, jnp.inf, False, 0)
        (final_data, energy_sum, min_height, fell, _), _ = jax.lax.scan(
            control_step, carry0, None, length=n_ctrl_steps)
        x1 = final_data.qpos[0]
        displacement = x1 - x0
        energy = energy_sum / jnp.maximum(n_ctrl_steps, 1)
        fall_penalty = jnp.where(fell, 0.3, 0.0)
        fitness = displacement - energy_penalty * energy - fall_penalty
        # guard against a physics blowup dominating the ES gradient estimate
        fitness = jnp.where(jnp.isfinite(fitness), fitness, -1e3)
        return fitness, displacement, energy, fell

    return rollout


def train_one_body(xml_path: str, *, generations: int, population: int, sigma: float,
                   lr: float, horizon_steps: int, dt_control: int, hidden: int,
                   energy_penalty: float, min_torso_height: float, seed: int):
    import mujoco
    import mujoco.mjx as mjx

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mx_model = mjx.put_model(mj_model)
    init_data = mjx.make_data(mx_model)
    obs_dim = mj_model.nq + mj_model.nv
    act_dim = mj_model.nu
    if act_dim == 0:
        return {"ok": False, "error": "no actuators"}

    rollout = make_rollout_fn(mx_model, horizon_steps, dt_control, min_torso_height, energy_penalty)
    rollout_batched = jax.jit(jax.vmap(lambda p: rollout(p, init_data)))

    key = jax.random.PRNGKey(seed)
    key, pkey = jax.random.split(key)
    params = init_policy_params(pkey, obs_dim, act_dim, hidden=hidden)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)
    n_params = flat_params.shape[0]

    opt = optax.adam(lr)
    opt_state = opt.init(flat_params)

    half_pop = max(population // 2, 1)
    best_fitness_seen = -1e9

    for gen in range(generations):
        key, nkey = jax.random.split(key)
        eps = jax.random.normal(nkey, (half_pop, n_params))
        eps_full = jnp.concatenate([eps, -eps], axis=0)  # antithetic pairs
        candidates = flat_params[None, :] + sigma * eps_full
        cand_params = jax.vmap(unravel)(candidates)
        fitness, displacement, energy, fell = rollout_batched(cand_params)
        fitness_np = np.asarray(fitness)
        best_fitness_seen = max(best_fitness_seen, float(fitness_np.max()))

        # standardize fitness for a stable ES gradient estimate (OpenAI-ES)
        f_std = (fitness - jnp.mean(fitness)) / (jnp.std(fitness) + 1e-8)
        grad_est = (eps_full * f_std[:, None]).mean(axis=0) / sigma
        updates, opt_state = opt.update(-grad_est, opt_state)  # ascend on fitness
        flat_params = optax.apply_updates(flat_params, updates)

    final_params = unravel(flat_params)
    final_fitness, final_disp, final_energy, final_fell = rollout(final_params, init_data)
    return {
        "ok": True,
        "rl_fitness": float(final_fitness),
        "rl_displacement": float(final_disp),
        "rl_energy": float(final_energy),
        "rl_fell": bool(final_fell),
        "best_fitness_during_training": best_fitness_seen,
        "n_params": int(n_params),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results_refit/rank_corr")
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--population", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--horizon_steps", type=int, default=1000)
    ap.add_argument("--dt_control", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--energy_penalty", type=float, default=0.01)
    ap.add_argument("--min_torso_height", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_bodies_limit", type=int, default=None,
                    help="Cap the number of bodies trained (for a quick smoke test "
                         "before committing a full GPU run). Ignored if --body_id or "
                         "--body_index is given.")
    ap.add_argument("--body_id", type=str, default=None,
                    help="Train only this one body_id (for a per-body SLURM job/array). "
                         "Writes a standalone shard file to "
                         "<out_dir>/rl_results_shards/<body_id>.json instead of the shared "
                         "rl_results.json, so concurrent array tasks never write the same "
                         "file. Run experiments.merge_rl_shards afterward to combine shards "
                         "into rl_results.json.")
    ap.add_argument("--body_index", type=int, default=None,
                    help="Like --body_id, but select the Nth geometrically-feasible body in "
                         "manifest order (0-indexed) -- convenient for a SLURM array job "
                         "driven by $SLURM_ARRAY_TASK_ID without having to look up IDs first.")
    args = ap.parse_args()

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    feasible = [b for b in manifest["bodies"] if b.get("geometric_feasible")]

    if args.body_id is not None or args.body_index is not None:
        if args.body_id is not None:
            matches = [b for b in feasible if b["body_id"] == args.body_id]
            if not matches:
                raise SystemExit(f"body_id {args.body_id!r} not found among feasible bodies "
                                 f"in {manifest_path}")
            target = matches[0]
        else:
            if not (0 <= args.body_index < len(feasible)):
                raise SystemExit(f"--body_index {args.body_index} out of range "
                                 f"(0..{len(feasible)-1})")
            target = feasible[args.body_index]
        _train_and_write_shard(target, args)
        return

    if args.n_bodies_limit:
        feasible = feasible[: args.n_bodies_limit]

    results_path = os.path.join(args.out_dir, "rl_results.json")
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    done_ids = {r["body_id"] for r in results}

    print(f"Training {len(feasible)} bodies ({len(done_ids)} already done, resuming)...")
    for b in feasible:
        bid = b["body_id"]
        if bid in done_ids:
            continue
        row = _run_training(b, args)
        if row is None:
            continue
        results.append(row)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  {bid}: rl_fitness={row['rl_fitness']:.4f} "
             f"(proxy_fitness={b['proxy_fitness']:.4f})  [{row['wall_clock_s']:.1f}s]")

    print(f"Wrote {len(results)} results to {results_path}")
    print("Next: python3 -m experiments.rank_correlation_analysis "
         f"--out_dir {args.out_dir}")


def _run_training(body_entry: dict, args) -> dict | None:
    """Shared single-body training call used by both sequential mode and
    the per-body shard mode below."""
    xml_path = os.path.join(args.out_dir, body_entry["mjcf_path"])
    t0 = time.time()
    out = train_one_body(
        xml_path, generations=args.generations, population=args.population,
        sigma=args.sigma, lr=args.lr, horizon_steps=args.horizon_steps,
        dt_control=args.dt_control, hidden=args.hidden,
        energy_penalty=args.energy_penalty, min_torso_height=args.min_torso_height,
        seed=args.seed,
    )
    wall = time.time() - t0
    if not out.get("ok"):
        print(f"  {body_entry['body_id']}: SKIPPED ({out.get('error')})")
        return None
    return {
        "body_id": body_entry["body_id"],
        "rl_fitness": out["rl_fitness"],
        "train_env_steps": args.generations * args.population * args.horizon_steps,
        "wall_clock_s": wall,
        "rl_displacement": out["rl_displacement"],
        "rl_fell": out["rl_fell"],
    }


def _train_and_write_shard(body_entry: dict, args):
    bid = body_entry["body_id"]
    shard_dir = os.path.join(args.out_dir, "rl_results_shards")
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(shard_dir, f"{bid}.json")
    if os.path.exists(shard_path):
        print(f"{bid}: shard already exists at {shard_path}, skipping (delete it to redo).")
        return
    print(f"Training single body {bid} -> {shard_path}")
    row = _run_training(body_entry, args)
    if row is None:
        return
    with open(shard_path, "w") as f:
        json.dump(row, f, indent=2)
    print(f"  {bid}: rl_fitness={row['rl_fitness']:.4f} "
         f"(proxy_fitness={body_entry['proxy_fitness']:.4f})  [{row['wall_clock_s']:.1f}s]")
    print(f"Wrote shard {shard_path}")
    print(f"Once all bodies' shards exist: python3 -m experiments.merge_rl_shards "
         f"--out_dir {args.out_dir}")


if __name__ == "__main__":
    main()

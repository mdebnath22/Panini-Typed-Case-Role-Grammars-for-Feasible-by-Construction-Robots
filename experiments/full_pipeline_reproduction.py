"""
Real Training-Pipeline Reproduction of the Five Recent-Method Analogues
=========================================================================

`recent_sota_baselines.py` and `robogrammar_glso_baselines.py` reimplement
each recent paper's SEARCH MECHANISM (RoboMorph/LASeR's LLM-conditioned
archive proposer, RoboMoRe's coarse-to-fine search, Debate2Create's
criterion-rotation archive, a quality-diversity archive-rejection ES, a
controller-fine-tuning variant, RoboGrammar's graph heuristic search, and
GLSO's latent-space Bayesian optimization) but evaluate every candidate
with this project's cheap CPG-proxy fitness (`gmm.task_fitness.
evaluate_fitness`), NOT a real trained policy. The paper states this
explicitly as a scope limitation: the published methods train a real
SAC policy (5e5-1e6 environment steps) per candidate on Gym/Brax
Ant-family tasks, a different simulator and ~10,000x the cost per
candidate, so quoting their numbers next to ours would be apples-to-
oranges, and training that many real policies here was judged
infeasible without a GPU.

THIS SCRIPT removes that specific gap by actually doing the real
per-candidate training, on Brax (not MuJoCo), using the SAME MJCF body
this project's grammar already generates. The key fact that makes this
possible without hand-building a new simulator per body: Brax can load
an arbitrary MJCF file directly (`brax.io.mjcf.load`), so the exact XML
`gmm.mjcf_compiler.MJCFCompiler` already produces for a genome is a
valid Brax system with no translation step. `GenomeBraxEnv` below wraps
that system in a standard Brax `PipelineEnv` (observation = joint
positions/velocities, reward = forward-x velocity + a survival bonus -
a control-effort cost, termination = torso height below a floor -- the
same reward/termination SHAPE Brax's own `Ant`/`Humanoid` envs use, and
the same floor-height convention `task_fitness.evaluate_fitness` already
uses for this project's CPG-proxy fitness, so this is a drop-in
REPLACEMENT fitness, not a different objective). `train_real_fitness`
then runs Brax's own SAC trainer on that env for `num_timesteps` real
environment steps and returns a `FitnessResult` with the exact fields
`evaluate_fitness` returns, so it is a drop-in replacement: swapping it
in for `evaluate_fitness` inside the already-written search mechanisms
is a two-line monkeypatch (`run_full_pipeline_benchmark` below), not a
rewrite of each baseline.

HARDWARE: this needs a real GPU to be worth running at paper-faithful
scale. On this project's own cloud container (2 CPU cores, no GPU),
SAC on a small Brax body gets ~140 steps/sec; at that rate 5e5 steps
(the LOW end of the 5e5-1e6 range the paper's related-work section
cites for these methods) is ~1 hour PER CANDIDATE, and the search
mechanisms above each evaluate dozens of candidates per seed across 8
seeds -- multiple CPU-weeks to CPU-months for one method, let alone
five. A GPU with a batched/vmapped Brax rollout (which this script does
NOT implement -- see `n_envs` below) should cut this by 1-2 orders of
magnitude, but still expect real per-method wall-clock in the hours-to-
a-day range at paper-faithful settings; budget accordingly before
launching the full `--n_seeds 8 --budget 60 --num_timesteps 500000` run.

Run (on a GPU machine, after `pip install brax`; JAX will use the GPU
automatically if `jax[cuda12]` or equivalent is installed instead of the
CPU-only `jax` this project's own container has):

    # quick feasibility check (NOT comparable to the paper's numbers --
    # tiny step count, 1 seed, 1 method, just confirms the pipeline runs
    # end to end on your hardware and reports real steps/sec so you can
    # extrapolate your own wall-clock budget):
    python3 -m experiments.full_pipeline_reproduction --smoke_test

    # paper-faithful run (one method at a time is recommended so a crash
    # or a timeout doesn't lose every method's progress):
    python3 -m experiments.full_pipeline_reproduction \\
        --method graph_heuristic_search --n_seeds 8 --budget 60 \\
        --num_timesteps 500000 --out results_ghs.json

Repeat `--method` for: llm_proposer_search, coarse_to_fine_search,
archive_criterion_rotation_search, diversity_archive_es,
graph_heuristic_search, latent_space_bo_search. (finetuned_controller_
search is intentionally excluded: it already fine-tunes a CPG
controller per candidate under this project's own fitness, so swapping
in a trained RL policy on top of that would conflate two different
"extra training per candidate" mechanisms rather than cleanly replace
one with the other.)
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, Optional

import numpy as np

from gmm.task_fitness import FitnessResult
from gmm.mjcf_compiler import MJCFCompiler

_JAX_SHIMMED = False


def _ensure_jax_brax_compat():
    """This project's own container has brax 0.14.2 + jax 0.11.2, a
    combination where brax's SAC trainer calls `jax.device_put_replicated`,
    an API jax 0.11 removed. Newer jax[cuda] installs on a GPU machine may
    or may not hit this depending on the exact pinned versions; the shim
    is a no-op if the real API is still present, so it is safe to always
    apply."""
    global _JAX_SHIMMED
    if _JAX_SHIMMED:
        return
    import jax
    import jax.numpy as jnp
    if not hasattr(jax, "device_put_replicated"):
        def _device_put_replicated(x, devices):
            return jax.tree_util.tree_map(
                lambda y: jax.device_put(
                    jnp.broadcast_to(y, (len(devices),) + jnp.shape(y))),
                x)
        jax.device_put_replicated = _device_put_replicated
    _JAX_SHIMMED = True


def tree_to_brax_system(tree):
    """Compiles `tree` (a PhiBody-built morphology tree, exactly what
    `evaluate_fitness` already consumes) to MJCF and loads it directly as
    a Brax system -- no translation beyond what MJCFCompiler already does
    for the MuJoCo-based CPG-proxy path."""
    from brax.io import mjcf
    comp = MJCFCompiler()
    xml = comp.compile(tree, name="full_pipeline_body")
    n_act = len(comp.actuators)
    if n_act == 0:
        return None, 0
    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(xml)
        path = f.name
    try:
        sys_ = mjcf.load(path)
    finally:
        os.unlink(path)
    return sys_, n_act


class GenomeBraxEnv:
    """A Brax PipelineEnv for an arbitrary genome-derived body, built only
    when first used (so importing this module doesn't require brax to be
    installed, matching the rest of this project's optional-dependency
    convention). Reward/termination mirror Brax's own Ant env (forward-x
    velocity + survival bonus - control cost; terminate when the torso
    drops below `min_torso_height`) and this project's own
    `evaluate_fitness`'s `min_torso_height` convention, so this is a
    like-for-like replacement fitness, not a differently-shaped one."""

    def __new__(cls, sys_, min_torso_height: float = 0.03,
                ctrl_cost_weight: float = 0.01, survive_reward: float = 1.0,
                reset_noise_scale: float = 0.05):
        _ensure_jax_brax_compat()
        import jax
        import jax.numpy as jp
        from brax.envs.base import PipelineEnv, State

        class _Env(PipelineEnv):
            def __init__(self):
                # "positional" backend: much cheaper to JIT-compile and
                # step than "generalized" for an arbitrary custom body
                # (the "generalized" backend took >10 CPU-minutes just to
                # compile for one genome body on this project's own
                # container and never finished a single training step
                # inside a 2-minute-granularity check -- killed rather
                # than assumed stuck); "positional" is also what Brax's
                # own Ant/Humanoid envs default much of their published
                # benchmarking to. A smaller fixed timestep (0.005s) with
                # more substeps per control step is required for the
                # positional backend's stability, mirroring how Brax's
                # own Ant.__init__ adjusts opt.timestep/n_frames when
                # backend in ("spring", "positional").
                sys_local = sys_.tree_replace({"opt.timestep": 0.005})
                super().__init__(sys=sys_local, backend="positional", n_frames=10)
                self._min_z = min_torso_height
                self._ctrl_cost_weight = ctrl_cost_weight
                self._survive_reward = survive_reward
                self._reset_noise_scale = reset_noise_scale

            def reset(self, rng):
                rng, rng1, rng2 = jax.random.split(rng, 3)
                lo, hi = -self._reset_noise_scale, self._reset_noise_scale
                q = self.sys.init_q + jax.random.uniform(
                    rng1, (self.sys.q_size(),), minval=lo, maxval=hi)
                qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
                pipeline_state = self.pipeline_init(q, qd)
                obs = self._get_obs(pipeline_state)
                reward, done, zero = jp.zeros(3)
                metrics = {"x_velocity": zero, "distance": zero}
                return State(pipeline_state, obs, reward, done, metrics)

            def step(self, state, action):
                p0 = state.pipeline_state
                p1 = self.pipeline_step(p0, action)
                vel = (p1.x.pos[0] - p0.x.pos[0]) / self.dt
                forward_reward = vel[0]
                torso_z = p1.x.pos[0, 2]
                is_healthy = jp.where(torso_z < self._min_z, 0.0, 1.0)
                ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
                reward = forward_reward + self._survive_reward * is_healthy - ctrl_cost
                done = 1.0 - is_healthy
                obs = self._get_obs(p1)
                state.metrics.update(x_velocity=vel[0],
                                     distance=p1.x.pos[0, 0] - 0.0)
                return state.replace(pipeline_state=p1, obs=obs,
                                     reward=reward, done=done)

            def _get_obs(self, pipeline_state):
                return jp.concatenate([pipeline_state.q, pipeline_state.qd])

        return _Env()


def train_real_fitness(tree, *, num_timesteps: int = 500_000,
                       episode_length: int = 1000, seed: int = 0,
                       num_envs: int = 4, num_evals: int = 1,
                       verbose: bool = False) -> FitnessResult:
    """Drop-in replacement for `gmm.task_fitness.evaluate_fitness`: trains
    a REAL SAC policy for `num_timesteps` environment steps on this body
    (via Brax, loading the same MJCF this project's CPG-proxy path
    compiles) and returns the trained policy's evaluation performance as
    the fitness, in the same `FitnessResult` shape so every search
    mechanism in recent_sota_baselines.py / robogrammar_glso_baselines.py
    that calls `evaluate_fitness(tree, ...)` works unmodified once this
    function is monkeypatched in (see `run_full_pipeline_benchmark`).
    `num_timesteps=500_000` is the LOW end of the 5e5-1e6 range the
    paper's related-work section cites for the actual published methods;
    pass a smaller value only for a disclosed feasibility check, never
    for a number reported next to the paper's other results."""
    _ensure_jax_brax_compat()
    from brax.training.agents.sac import train as sac

    sys_, n_act = tree_to_brax_system(tree)
    if sys_ is None:
        return FitnessResult(False, 0.0, True, 0.0, -1e3,
                             error="no actuators (degenerate body)")
    try:
        env = GenomeBraxEnv(sys_)
        make_inference_fn, params, metrics = sac.train(
            environment=env,
            num_timesteps=num_timesteps,
            episode_length=episode_length,
            num_envs=num_envs,
            num_evals=num_evals,
            learning_rate=3e-4,
            batch_size=64,
            grad_updates_per_step=num_envs,
            max_devices_per_host=1,
            seed=seed,
        )
    except Exception as e:
        return FitnessResult(False, 0.0, True, 0.0, -1e3, error=str(e))

    displacement = float(metrics.get("eval/episode_distance", 0.0))
    reward = float(metrics.get("eval/episode_reward", -1e3))
    fell = displacement < 0.02
    if verbose:
        print(f"  trained {num_timesteps} steps: eval reward={reward:.3f}, "
             f"displacement={displacement:.3f}")
    return FitnessResult(ok=True, displacement=displacement, fell=fell,
                         energy=0.0, fitness=reward)


def run_full_pipeline_benchmark(method: str, n_seeds: int = 8, budget: int = 60,
                                num_timesteps: int = 500_000,
                                verbose: bool = True) -> Dict:
    """Monkeypatches `evaluate_fitness` inside recent_sota_baselines.py /
    robogrammar_glso_baselines.py (whichever module defines `method`) to
    `train_real_fitness`, then runs that one method's search loop for
    `n_seeds` seeds under `budget` evaluations -- same accounting
    convention as every other table in this paper, except each
    "evaluation" is now a full `num_timesteps`-step SAC training run
    instead of a CPG-proxy rollout, so treat `budget` x `num_timesteps`
    as the real cost before launching this at paper-faithful settings."""
    import experiments.recent_sota_baselines as rsb
    import experiments.robogrammar_glso_baselines as rgb

    def _patched(tree, **kwargs):
        s = kwargs.get("seed", 0)
        return train_real_fitness(tree, num_timesteps=num_timesteps, seed=s,
                                  verbose=verbose)

    rsb.evaluate_fitness = _patched
    rgb.evaluate_fitness = _patched

    if hasattr(rsb, method):
        search_fn = getattr(rsb, method)
    elif hasattr(rgb, method):
        search_fn = getattr(rgb, method)
    else:
        raise ValueError(f"no such method '{method}' in recent_sota_baselines "
                         f"or robogrammar_glso_baselines")

    results = []
    for s in range(n_seeds):
        r = search_fn(budget, seed=s)
        best = r["best_fitness"] if isinstance(r, dict) else r
        results.append(float(best))
        if verbose:
            print(f"seed={s}: best_fitness={best:+.3f}")
    arr = np.array(results)
    if verbose:
        print(f"\n{method} (REAL trained-policy fitness, "
             f"num_timesteps={num_timesteps}): {arr.mean():.3f} +/- {arr.std():.3f}")
    return {"method": method, "results": results, "mean": float(arr.mean()),
           "std": float(arr.std()), "num_timesteps": num_timesteps}


def smoke_test(num_timesteps: int = 2000, episode_length: int = 200,
               verbose: bool = True) -> Dict:
    """Confirms the whole pipeline (genome -> MJCF -> Brax system -> SAC
    training -> FitnessResult) runs end to end on your hardware and
    reports real steps/sec, so you can extrapolate wall-clock for your
    own GPU before committing to a full run. NOT a result --
    num_timesteps/episode_length here are far below anything reportable
    (the paper-faithful default is episode_length=1000, matching
    `evaluate_fitness`'s horizon_steps=1000; the shorter length here is
    purely to keep this correctness check's JIT-compile fast on whatever
    hardware you're running it on, CPU included)."""
    from experiments.m_search_loop import random_genome
    from ptg.body_grammar import BodyGrammar
    from ptg.body_sutra import seed_body_rules
    from gmm.phi_body import PhiBody
    import time

    rng = np.random.default_rng(0)
    genome = random_genome(rng)
    g = BodyGrammar(seed_body_rules(), seed=0)
    deriv = g.derive(genome.to_plan())
    tree = PhiBody().build_tree(deriv)

    t0 = time.time()
    result = train_real_fitness(tree, num_timesteps=num_timesteps, seed=0,
                                episode_length=episode_length, verbose=verbose)
    elapsed = time.time() - t0
    sps = num_timesteps / max(elapsed, 1e-6)
    if verbose:
        print(f"\nsmoke test: {num_timesteps} steps in {elapsed:.1f}s "
             f"({sps:.1f} steps/sec on this hardware)")
        print(f"result: {result}")
        for target, label in [(500_000, "low end (5e5)"), (1_000_000, "high end (1e6)")]:
            est_hours = target / max(sps, 1e-6) / 3600
            print(f"  extrapolated time for one {label} candidate: "
                 f"{est_hours:.2f} hours")
    return {"elapsed": elapsed, "steps_per_sec": sps, "result": result}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--method", type=str, default=None)
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--num_timesteps", type=int, default=500_000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        if args.method is None:
            raise SystemExit("pass --method <name> or --smoke_test")
        res = run_full_pipeline_benchmark(args.method, n_seeds=args.n_seeds,
                                          budget=args.budget,
                                          num_timesteps=args.num_timesteps)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)

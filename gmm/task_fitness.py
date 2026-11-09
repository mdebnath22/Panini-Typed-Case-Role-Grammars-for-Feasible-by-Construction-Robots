"""
Task Fitness Evaluator (Phase M2/M3 core)
==========================================

This is the piece that turns "the grammar generates valid bodies" into
"the grammar can be searched to generate GOOD bodies" — the actual claim
of the thesis. Without this, every prior result (H5/M-H5 homomorphism,
H6/M-H3 compactness, M-H6 feasibility) is a structural/validity result,
not an optimization result.

Since building a full Swarm-JEPA policy-training loop per candidate body is
expensive, this uses a cheap, deterministic **central pattern generator**
(CPG) controller — a fixed sinusoidal gait with per-actuator phase offsets
derived from limb mount angle — as the fitness proxy. This is standard
practice in morphology-search literature (e.g. Sims 1994, robogrammar-style
work): body quality is measured under a simple universal controller, not a
fully-trained policy, precisely so morphology and control search don't
confound each other in this first pass. A real trained-policy fitness
(Swarm-JEPA per body) is Phase M2's later, more expensive refinement — see
`experiments/m_search_loop.py`'s docstring for where that plugs in.

Fitness = forward CoM displacement over a fixed horizon, penalized for
falling (torso height dropping below a floor) and for excessive control
energy (encourages efficient gaits, not thrashing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from gmm.mjcf_compiler import MJCFCompiler


@dataclass
class FitnessResult:
    ok: bool
    displacement: float      # forward (+x) CoM displacement, meters
    fell: bool
    energy: float            # mean |ctrl| over the rollout
    fitness: float           # scalar objective (higher is better)
    error: Optional[str] = None


def _cpg_controls(t: float, n_act: int, phases: np.ndarray,
                  freq: float = 1.5, amp: float = 0.8) -> np.ndarray:
    """Simple sinusoidal CPG: every actuator gets the same frequency, its
    own phase (so limbs move out of sync — a walking-gait prior), and a
    shared amplitude. Deterministic, no learning — this is the point."""
    return amp * np.sin(2 * np.pi * freq * t + phases)


def evaluate_fitness(tree, *, horizon_steps: int = 1000, dt_control: int = 4,
                     min_torso_height: float = 0.03,
                     energy_penalty: float = 0.01,
                     seed: int = 0,
                     freq: float = 1.5, amp: float = 0.8,
                     phase_offset: float = 0.0,
                     xml_override: Optional[str] = None) -> FitnessResult:
    """Compile `tree`, run it under a CPG gait, return locomotion fitness.

    horizon_steps : number of *physics* steps (at model dt, typically 5ms)
    dt_control    : hold each control sample for this many physics steps
    freq, amp     : CPG gait parameters (previously hard-coded in
                    _cpg_controls; exposed so a search loop can treat the
                    controller itself as tunable — see
                    experiments/recent_sota_baselines.py's
                    frozen_vs_finetuned_controller analogue, and
                    reviewerFGH_cmaes_budget_controller.py part H, which
                    was blocked on this exact signature not existing).
    phase_offset  : added to every actuator's phase before the per-actuator
                    linspace jitter below; a uniform shift, not per-limb.
    xml_override  : if given, simulate THIS xml string instead of
                    recompiling `tree` (e.g. the same body under stricter
                    contact settings, experiments/contact_sensitivity.py).
                    `tree` is still compiled once to get the actuator
                    count, which contact settings never change.
    """
    comp = MJCFCompiler()
    xml = comp.compile(tree, name="fitness_eval")
    n_act = len(comp.actuators)
    if xml_override is not None:
        xml = xml_override
    if n_act == 0:
        return FitnessResult(False, 0.0, True, 0.0, -1e3,
                             error="no actuators (degenerate body)")

    try:
        import mujoco
    except Exception as e:
        return FitnessResult(False, 0.0, True, 0.0, -1e3, error=str(e))

    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        rng = np.random.default_rng(seed)
        # phase offset per actuator: derived from actuator index so limbs at
        # different mount angles are out of phase (walking-gait prior),
        # plus a small random jitter so the CPG isn't perfectly symmetric
        phases = (np.linspace(0, 2 * np.pi, n_act, endpoint=False)
                 + phase_offset
                 + rng.normal(0, 0.05, n_act))

        x0 = float(data.qpos[0]) if model.nq > 0 else 0.0
        energies: List[float] = []
        ever_fell = False
        min_height_seen = float("inf")
        t = 0.0
        for step in range(horizon_steps):
            if step % dt_control == 0:
                ctrl = _cpg_controls(t, n_act, phases, freq=freq, amp=amp)
                data.ctrl[:n_act] = np.clip(ctrl, -1, 1)
                energies.append(float(np.mean(np.abs(ctrl))))
            mujoco.mj_step(model, data)
            t += model.opt.timestep

            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                ever_fell = True
                break  # numerical blowup: must stop, unlike a soft fall
            torso_z = float(data.qpos[2]) if model.nq > 2 else 1.0
            min_height_seen = min(min_height_seen, torso_z)
            if torso_z < min_torso_height:
                ever_fell = True
                # don't break: let a fallen body keep crawling — this is a
                # softer, more informative signal than truncating the episode

        x1 = float(data.qpos[0]) if model.nq > 0 else 0.0
        displacement = x1 - x0
        # sanity guard: a legged body under this gait cannot plausibly cover
        # more than ~3m in this horizon; anything beyond that is a physics
        # blowup (unstable contact/actuator interaction), not locomotion —
        # MuJoCo can emit huge-but-finite QACC before actually going NaN.
        max_plausible = 3.0
        if abs(displacement) > max_plausible:
            return FitnessResult(False, displacement, True, 0.0, -1e3,
                                 error="implausible displacement (blowup)")
        energy = float(np.mean(energies)) if energies else 0.0
        fall_penalty = 0.3 if ever_fell else 0.0
        fitness = displacement - energy_penalty * energy - fall_penalty

        return FitnessResult(True, displacement, ever_fell, energy, fitness)
    except Exception as e:  # noqa
        return FitnessResult(False, 0.0, True, 0.0, -1e3, error=str(e))

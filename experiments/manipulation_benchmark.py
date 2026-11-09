"""
3-task manipulation benchmark: does the grammar's staged derivation
(APPROACH -> GRASP -> LIFT -> TRANSPORT -> PLACE, with the k_it "holding"
marker enforcing grasp-before-lift) produce more reliable physical
manipulation than an unstructured baseline planner that goes straight
from the end-effector's start pose to the final goal and attempts to
grasp only opportunistically, with no explicit GRASP stage gating LIFT?

Until this script, every manipulation rule in ptg/sutra.py (APPROACH,
GRASP, LIFT, TRANSPORT, PLACE, ROTATE_TOOL, RETRACT) was exercised only
symbolically -- as an SE(3) twist consumed by Φ for compositional-
generalization/homomorphism testing (experiments/h1_compgen.py,
experiments/scenes.py) -- never against an actual physics simulation.
This closes that gap with a genuinely physical (if deliberately minimal)
MuJoCo benchmark:

  * The end-effector is a mocap body (kinematically driven, no dynamics
    of its own -- this isolates "does the STAGED PLAN succeed" from
    "can a real controller track it", which is a separate question).
  * The object is a real dynamic free body under gravity and contact.
  * "Grasping" is modeled as a rigid kinematic attachment: while held,
    the object's pose is pinned to the end-effector pose (at the
    relative offset recorded at the moment of grasp) each step, and
    released back to free rigid-body dynamics at PLACE -- a simplified-
    grasp proxy (no finger/contact dynamics) used so the question under
    test is the STAGED STRUCTURE (grasp-before-lift), not finger
    mechanics or contact-force grasp stability. (An explicit MuJoCo
    <weld> equality constraint was tried first and found to ring/
    destabilize under the mocap end-effector's discontinuous-looking
    position updates; the direct kinematic pin is the robust choice for
    isolating the question this benchmark asks.)

Three tasks of increasing length, run on the SAME random scenes for
both the grammar-derived plan and the baseline:
  1. Reach       -- does the end-effector's trajectory ever come within
                    grasp tolerance of the object?
  2. Grasp+Lift   -- is the object lifted above a height threshold AND
                    still attached (not dropped) at the end?
  3. Full place   -- does the object end up resting within tolerance of
                    the target support location after release?

Two baselines, of deliberately different strength:

  * "baseline" -- a single straight-line motion from the end-effector's
    start pose directly to the FINAL target (skipping APPROACH/GRASP/
    LIFT/TRANSPORT as separate staged waypoints), over the same total
    step budget as the grammar's segments combined, with NO knowledge of
    the object's location. It attempts a grasp only opportunistically --
    the first time it happens to pass within grasp tolerance of the
    object en route, which may never happen. This is a weak planner kept
    for continuity with an earlier version of this study; by itself it
    mostly shows the task is non-trivial, not that the grammar's STAGING
    specifically matters, since it also never looked at the object.
  * "informed" -- a competent, OBJECT-AWARE planner handed the exact same
    ground-truth object and target positions the grammar's own waypoints
    are computed from, that deliberately routes through the object
    before the target (start-to-object, then object-to-target; two
    segments, same total step budget). What it still lacks, relative to
    the grammar, is the staged ORDERING itself -- no separate hover-above
    waypoint, no separate grasp-depth waypoint, no separate lift-to-
    height waypoint before transporting. This is the fairer test: once
    "doesn't know where the object is" is no longer available as the
    reason a weaker planner fails, does the grammar's explicit staging
    still buy anything over a planner that is merely competent and
    object-aware?

Both baselines attempt a grasp via the same tolerance-triggered
mechanism as the grammar path, so any remaining gap is attributable to
the staged ordering (or its absence), not to a different grasp mechanism
or a different time budget. This operationalizes the grammar's
structural guarantee (GRASP's k_it marker must fire, licensing LIFT,
before TRANSPORT/PLACE can proceed) as a measurable physical-outcome
difference against both a weak and a strong competitor, rather than a
claim about symbolic well-formedness alone.

Run:
    python3 -m experiments.manipulation_benchmark --n_trials 150
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.scenes import random_scene
from gmm.phi_seed import PhiSeed
from ptg.grammar import PTG
from ptg.karaka import Karaka, KarakaFrame
from ptg.sutra import seed_rules

STEPS_PER_SEGMENT = 120
GRASP_TOL = 0.05           # m: opportunistic/engage grasp radius (covers the
                           # ~4cm gap between the grammar's GRASP target --
                           # symbolic block height + grasp_depth -- and the
                           # physical block's actual resting height on the
                           # ground plane, which the symbolic scene doesn't
                           # model)
REACH_TOL = 0.12           # m: "reach" success -- within hover range of block
                           # (APPROACH deliberately hovers ~0.10m above the
                           # object; that is success for "reach", not a miss)
LIFT_THRESHOLD = 0.08      # m: object height counted as "lifted"
PLACE_TOL = 0.05           # m: final xy tolerance for a successful place


def _scene_mjcf(ee0: np.ndarray, block0: np.ndarray) -> str:
    return f"""
<mujoco model="manip_bench">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default><geom friction="0.8 0.01 0.001" solref="0.01 1" solimp="0.9 0.95 0.001"/></default>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.5 0.5 0.5 1"/>
    <body name="gripper" mocap="true" pos="{ee0[0]} {ee0[1]} {ee0[2]}">
      <geom type="sphere" size="0.02" rgba="0.2 0.4 0.9 0.6" contype="0" conaffinity="0"/>
    </body>
    <body name="block" pos="{block0[0]} {block0[1]} {block0[2]}">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.05" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass
class TrialOutcome:
    task: str
    planner: str
    success: bool
    ever_grasped: bool
    final_object_dist: Optional[float]  # distance to target (place) or None
    min_ee_to_block: float
    object_height_max: float
    sim_failure: bool


def _derive_waypoints(scene) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Returns (poses[6,4,4] world poses after each grammar step incl.
    the initial pose, support_target_xyz). poses[0]=start, [1]=post-
    APPROACH, [2]=post-GRASP, [3]=post-LIFT, [4]=post-TRANSPORT,
    [5]=post-PLACE."""
    # A plain pick-place frame (no KARANA/tool role bound) so the resolver's
    # only candidate for the first step is plain 1.1.1_APPROACH (from
    # directly above), not the more-specific 1.1.2_APPROACH_TOOL side-
    # approach apavada (which exists for tool-handling tasks and adds a
    # lateral offset -- a deliberate grammar behavior, just not this
    # benchmark's scenario: picking up a block, not a tool).
    frame = (KarakaFrame()
             .bind(Karaka.KARTA, scene["ee"])
             .bind(Karaka.KARMA, scene["block"])
             .bind(Karaka.SAMPRADANA, scene["bin"])
             .bind(Karaka.ADHIKARANA, scene["table"]))
    template = ["APPROACH", "GRASP", "LIFT", "TRANSPORT", "PLACE"]
    ptg = PTG(seed_rules())
    plan = [{"goal": g, "frame": frame if i == 0 else KarakaFrame()}
            for i, g in enumerate(template)]
    plan[0]["params"] = {"approach_height": 0.10, "lift_height": 0.15}
    deriv = ptg.derive(plan)
    out = PhiSeed().map_derivation(deriv, scene["ee"].pose.copy())
    return out["poses"], scene["table"].position.copy()


def _run_segments(xml: str, waypoints: List[np.ndarray], grasp_at: Optional[int],
                  release_at: Optional[int], block0: np.ndarray,
                  steps_per_seg: int = STEPS_PER_SEGMENT) -> TrialOutcome:
    """waypoints: list of 3-vectors the gripper mocap body is linearly
    interpolated through, one segment per consecutive pair.
    grasp_at / release_at: waypoint INDEX at which to toggle the weld on
    arrival (None = never)."""
    try:
        import mujoco
    except Exception:
        return TrialOutcome("", "", False, False, None, 1e9, 0.0, True)
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
    except Exception:
        return TrialOutcome("", "", False, False, None, 1e9, 0.0, True)

    gripper_mocap_id = 0  # only one mocap body
    grasped = False
    ever_grasped = False
    min_ee_to_block = 1e9
    object_height_max = 0.0
    sim_failure = False
    attach_offset = np.zeros(3)  # block_pos - ee_pos, recorded at grasp time

    for seg in range(len(waypoints) - 1):
        p0, p1 = waypoints[seg], waypoints[seg + 1]
        for k in range(steps_per_seg):
            alpha = (k + 1) / steps_per_seg
            data.mocap_pos[gripper_mocap_id] = p0 * (1 - alpha) + p1 * alpha
            block_pos = data.qpos[0:3].copy()
            ee_pos = data.mocap_pos[gripper_mocap_id].copy()
            d = float(np.linalg.norm(ee_pos - block_pos))
            min_ee_to_block = min(min_ee_to_block, d)
            if (not grasped) and d <= GRASP_TOL:
                if grasp_at is None or seg >= grasp_at:
                    attach_offset = block_pos - ee_pos
                    grasped = True
                    ever_grasped = True
            if grasped:
                data.qpos[0:3] = ee_pos + attach_offset
                data.qvel[0:6] = 0.0
            try:
                mujoco.mj_step(model, data)
            except Exception:
                sim_failure = True
                break
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                sim_failure = True
                break
            object_height_max = max(object_height_max, float(data.qpos[2]))
        if sim_failure:
            break
        if release_at is not None and seg + 1 == release_at and grasped:
            grasped = False

    # let the object settle briefly after the final segment (post-release)
    if not sim_failure:
        for _ in range(100):
            try:
                mujoco.mj_step(model, data)
            except Exception:
                sim_failure = True
                break
            if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                sim_failure = True
                break

    final_block_xyz = data.qpos[0:3].copy() if not sim_failure else None
    return TrialOutcome("", "", False, ever_grasped, None, min_ee_to_block,
                       object_height_max, sim_failure), final_block_xyz, grasped


def run_trial(seed: int, task: str, planner: str) -> TrialOutcome:
    rng = np.random.default_rng(seed)
    scene = random_scene(rng)
    poses, table_xyz = _derive_waypoints(scene)
    grammar_wps = [poses[i][:3, 3] for i in range(poses.shape[0])]  # 6 points
    block0 = scene["block"].position.copy()
    block0[2] = 0.025  # rest exactly on the ground at the start

    xml = _scene_mjcf(grammar_wps[0], block0)

    if planner == "grammar":
        if task == "reach":
            wps = grammar_wps[0:2]
            grasp_at, release_at = None, None
        elif task == "grasp_lift":
            wps = grammar_wps[0:4]
            grasp_at, release_at = 1, None
        else:  # place
            wps = grammar_wps[0:6]
            grasp_at, release_at = 1, 5
    elif planner == "baseline":  # unstructured: one straight shot to the FINAL target,
        # not even aware of the object's location -- a weak baseline, kept
        # for continuity with the earlier result.
        final_targets = {"reach": grammar_wps[1], "grasp_lift": grammar_wps[3],
                         "place": grammar_wps[5]}
        target = final_targets[task]
        n_seg = {"reach": 1, "grasp_lift": 3, "place": 5}[task]
        wps = [grammar_wps[0]] + [grammar_wps[0] * (1 - (i + 1) / n_seg)
                                  + target * ((i + 1) / n_seg) for i in range(n_seg)]
        grasp_at = 0 if task != "reach" else None
        release_at = (len(wps) - 1) if task != "reach" else None
    else:  # "informed": a competent, OBJECT-AWARE planner -- it knows exactly
        # where the object and the final target are (the same ground-truth
        # positions the grammar's own waypoints are computed from), and
        # deliberately routes through the object before the target. What it
        # skips, relative to the grammar, is the STAGED ordering: no separate
        # hover-above waypoint, no separate grasp-depth waypoint, no separate
        # lift-to-height waypoint before transporting -- it goes directly to
        # the object (one segment), then directly to the final target (one
        # segment), attempting the grasp as soon as it is within tolerance of
        # the object en route. This isolates what the staging itself buys,
        # once "doesn't know where the object is" is no longer the reason a
        # weaker planner fails.
        object_xyz = grammar_wps[2]  # the grammar's own GRASP-depth point,
        # used only as ground truth for an informed planner, not as a staged
        # waypoint this planner is told to visit in any particular manner.
        final_targets = {"reach": grammar_wps[1], "grasp_lift": grammar_wps[3],
                         "place": grammar_wps[5]}
        target = final_targets[task]
        if task == "reach":
            wps = [grammar_wps[0], object_xyz]
            grasp_at, release_at = None, None
        else:
            n_total = {"grasp_lift": 3, "place": 5}[task]
            n_to_object = max(1, n_total // 2)
            n_to_target = n_total - n_to_object
            wps = ([grammar_wps[0] * (1 - (i + 1) / n_to_object) + object_xyz * ((i + 1) / n_to_object)
                   for i in range(n_to_object)]
                  + [object_xyz * (1 - (i + 1) / n_to_target) + target * ((i + 1) / n_to_target)
                     for i in range(n_to_target)])
            wps = [grammar_wps[0]] + wps
            grasp_at = 0
            release_at = len(wps) - 1

    outcome, final_block_xyz, still_grasped = _run_segments(
        xml, wps, grasp_at, release_at, block0)
    outcome.task, outcome.planner = task, planner

    if task == "reach":
        outcome.success = outcome.min_ee_to_block <= REACH_TOL
    elif task == "grasp_lift":
        outcome.success = (outcome.ever_grasped and
                          outcome.object_height_max >= LIFT_THRESHOLD and
                          not outcome.sim_failure)
    else:  # place
        if outcome.sim_failure or final_block_xyz is None:
            outcome.success = False
        else:
            dist = float(np.linalg.norm(final_block_xyz[:2] - table_xyz[:2]))
            outcome.final_object_dist = dist
            outcome.success = (outcome.ever_grasped and dist <= PLACE_TOL
                              and not still_grasped)
    return outcome


def run_benchmark(n_trials: int = 150, seed0: int = 0, verbose: bool = True) -> Dict:
    tasks = ["reach", "grasp_lift", "place"]
    planners = ["grammar", "baseline", "informed"]
    results: Dict[str, Dict[str, List[TrialOutcome]]] = {
        t: {p: [] for p in planners} for t in tasks}

    for i in range(n_trials):
        seed = seed0 + i
        for task in tasks:
            for planner in planners:
                results[task][planner].append(run_trial(seed, task, planner))

    summary = {}
    for task in tasks:
        summary[task] = {}
        for planner in planners:
            outs = results[task][planner]
            n = len(outs)
            succ = sum(o.success for o in outs) / n
            fails = sum(o.sim_failure for o in outs)
            grasp_rate = sum(o.ever_grasped for o in outs) / n
            dists = [o.final_object_dist for o in outs if o.final_object_dist is not None]
            summary[task][planner] = dict(
                n=n, success_rate=succ, sim_failures=fails, grasp_rate=grasp_rate,
                mean_place_error=float(np.mean(dists)) if dists else None,
                steps=STEPS_PER_SEGMENT * {"reach": 1, "grasp_lift": 3, "place": 5}[task],
            )

    if verbose:
        print(f"Manipulation benchmark, n={n_trials} scenes per (task, planner) cell\n")
        print("Note: the end-effector is a kinematically-driven mocap point (no "
              "actuators, no joint limits), so actuator/joint-limit violations are "
              "not a meaningful metric here and are omitted rather than fabricated. "
              "'Simulation failures' and grasp/success rates are measured directly.\n")
        for task in tasks:
            print(f"== {task} (fixed horizon: {summary[task]['grammar']['steps']} steps) ==")
            for planner in planners:
                s = summary[task][planner]
                line = (f"  {planner:10s} success={s['success_rate']*100:5.1f}%  "
                       f"grasp_rate={s['grasp_rate']*100:5.1f}%  "
                       f"sim_failures={s['sim_failures']}/{s['n']}")
                if s["mean_place_error"] is not None:
                    line += f"  mean_place_error={s['mean_place_error']*100:.2f}cm"
                print(line)
            print()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=150)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    run_benchmark(n_trials=args.n_trials, seed0=args.seed0)

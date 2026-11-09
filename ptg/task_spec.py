"""Task-to-curriculum specification layer.

Turns a natural-language task description ("carry drinks across a crowded
restaurant") into something a GPU-side RL training pipeline can actually
consume: a typed list of success metrics, hard constraints, and curriculum
stages with increasing difficulty. This is deliberately NOT a simulator or
a trainer -- it has no GPU dependency, it is the missing specification
layer the roadmap flagged as absent between "task in English" and "reward
function + domain-randomization schedule in Isaac Lab/MJX/Genesis."

Three task families are given as worked examples: `walk` (the paper's own
task, included as the calibration case so the schema isn't invented in a
vacuum), `parkour` (perception+terrain curriculum), and `waiter` (contact +
navigation + human-aware safety). Each is a plain, serializable dataclass
tree; dump any of them with `to_json` and hand the file to whatever
GPU-side training code consumes task configs.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


@dataclass
class SuccessMetric:
    name: str
    description: str
    kind: str              # "maximize" | "minimize" | "threshold"
    units: str
    target: Optional[float] = None   # for "threshold": pass/fail cutoff


@dataclass
class Constraint:
    name: str
    description: str
    kind: str              # "hard" (episode-ending violation) | "soft" (penalty)
    limit: float
    units: str
    penalty_weight: Optional[float] = None  # for "soft"


@dataclass
class CurriculumStage:
    stage_id: int
    name: str
    description: str
    env_params: Dict[str, float] = field(default_factory=dict)
    advance_condition: str = ""   # plain-language criterion for moving to next stage
    advance_threshold: Optional[float] = None


@dataclass
class TaskSpec:
    task_id: str
    description: str
    success_metrics: List[SuccessMetric]
    constraints: List[Constraint]
    curriculum: List[CurriculumStage]
    required_sim_features: List[str] = field(default_factory=list)
    required_controller_class: str = "trained_policy"  # "fixed_cpg" | "trained_policy" | "mpc" | "vla"
    notes: str = ""

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "TaskSpec":
        with open(path) as f:
            d = json.load(f)
        d["success_metrics"] = [SuccessMetric(**m) for m in d["success_metrics"]]
        d["constraints"] = [Constraint(**c) for c in d["constraints"]]
        d["curriculum"] = [CurriculumStage(**s) for s in d["curriculum"]]
        return TaskSpec(**d)


def walk_task() -> TaskSpec:
    """The paper's own task, expressed in this schema -- included so the
    schema is calibrated against a task whose answer (a fixed CPG is an
    adequate proxy controller) is already known, rather than invented with
    no reference point."""
    return TaskSpec(
        task_id="walk",
        description="Forward locomotion over flat ground under a fixed time horizon.",
        success_metrics=[
            SuccessMetric("forward_displacement", "CoM displacement along +x over the horizon",
                          "maximize", "m"),
        ],
        constraints=[
            Constraint("torso_height_floor", "torso must not drop below this height (a fall)",
                      "hard", limit=0.03, units="m"),
            Constraint("control_energy", "mean |ctrl| over the rollout", "soft",
                      limit=1.0, units="dimensionless", penalty_weight=0.01),
        ],
        curriculum=[
            CurriculumStage(0, "flat_ground", "Flat, rigid, infinite ground plane.",
                            env_params={"terrain_roughness": 0.0}),
        ],
        required_sim_features=["rigid_body_dynamics", "floating_base"],
        required_controller_class="fixed_cpg",
        notes="This is the paper's existing proxy task; a trained policy on this same task is "
             "exactly the GPU-side half of the rank-correlation experiment "
             "(experiments/rank_correlation_harness.py).",
    )


def parkour_task() -> TaskSpec:
    return TaskSpec(
        task_id="parkour",
        description="Traverse a sequence of terrain obstacles (gaps, steps, ramps, narrow "
                   "beams) that require perceiving terrain ahead and timing footfalls, not "
                   "just a fixed gait.",
        success_metrics=[
            SuccessMetric("course_completion_fraction", "fraction of the obstacle course "
                          "traversed before falling or timing out", "maximize", "fraction"),
            SuccessMetric("completion_time", "time to finish the course, for runs that finish",
                          "minimize", "s"),
        ],
        constraints=[
            Constraint("torso_height_floor", "torso must not drop below this height (a fall)",
                      "hard", limit=0.03, units="m"),
            Constraint("max_impact_force", "peak foot-ground contact force on landing (actuator/"
                      "structural safety; see sulba.engineering_feasibility's torque/stress "
                      "checks for the static-design-time version of this same limit)",
                      "soft", limit=400.0, units="N", penalty_weight=0.002),
        ],
        curriculum=[
            CurriculumStage(0, "flat_with_single_step", "One small step-up obstacle, known height.",
                            env_params={"obstacle_density": 0.1, "max_step_height": 0.05},
                            advance_condition="course_completion_fraction >= 0.9 over 100 episodes",
                            advance_threshold=0.9),
            CurriculumStage(1, "mixed_steps_and_gaps", "Steps + gaps, randomized height/width "
                            "within a moderate range.",
                            env_params={"obstacle_density": 0.3, "max_step_height": 0.12,
                                       "max_gap_width": 0.15},
                            advance_condition="course_completion_fraction >= 0.8 over 100 episodes",
                            advance_threshold=0.8),
            CurriculumStage(2, "narrow_beams_and_ramps", "Adds narrow-beam balance sections and "
                            "variable-angle ramps.",
                            env_params={"obstacle_density": 0.5, "min_beam_width": 0.06,
                                       "max_ramp_angle_deg": 25.0},
                            advance_condition="course_completion_fraction >= 0.7 over 100 episodes",
                            advance_threshold=0.7),
            CurriculumStage(3, "full_randomized_course", "Full terrain randomization, hardest "
                            "difficulty, held out from earlier stages.",
                            env_params={"obstacle_density": 0.8},
                            advance_condition="terminal stage"),
        ],
        required_sim_features=["rigid_body_dynamics", "floating_base", "terrain_heightfield",
                               "exteroceptive_sensing"],
        required_controller_class="trained_policy",
        notes="Perception-plus-RL, not a morphology-only problem -- the grammar's job here is "
             "limited to proposing bodies whose actuators/structure clear the "
             "sulba.engineering_feasibility checks (peak torque for a landing impact, "
             "structural stress) before any GPU time is spent training on them.",
    )


def waiter_task() -> TaskSpec:
    """The worked example the roadmap asked for: 'carry drinks across a
    crowded restaurant' decomposed into metrics, constraints, and a
    curriculum."""
    return TaskSpec(
        task_id="waiter",
        description="Carry a tray of drinks from a pickup point to a table across a room "
                   "containing static furniture and moving people, without spilling, "
                   "colliding, or entering anyone's safety margin.",
        success_metrics=[
            SuccessMetric("delivery_success_rate", "fraction of episodes where the tray "
                          "reaches the target table with no spill", "maximize", "fraction"),
            SuccessMetric("delivery_time", "time from pickup to successful delivery",
                          "minimize", "s"),
            SuccessMetric("spill_volume", "total liquid displaced from its container over the "
                          "episode (a physically simulated proxy, not a literal fluid sim: "
                          "e.g. container-tilt-angle integrated over time past a threshold)",
                          "minimize", "proxy units (tilt-seconds)"),
        ],
        constraints=[
            Constraint("tray_tilt", "instantaneous tray tilt from horizontal", "soft",
                      limit=8.0, units="deg", penalty_weight=0.05),
            Constraint("human_safety_margin", "minimum distance maintained to any person in "
                      "the scene", "hard", limit=0.4, units="m"),
            Constraint("max_speed_near_humans", "speed cap when within 1.5m of a person",
                      "hard", limit=0.6, units="m/s"),
            Constraint("furniture_collision", "any contact with static furniture outside the "
                      "designated path", "hard", limit=0.0, units="count"),
        ],
        curriculum=[
            CurriculumStage(0, "static_empty_room", "Pickup-to-table navigation, no obstacles, "
                            "no people, tray held level by a scripted grasp.",
                            env_params={"n_obstacles": 0, "n_people": 0, "tray_dynamics": "fixed"},
                            advance_condition="delivery_success_rate >= 0.95 over 100 episodes",
                            advance_threshold=0.95),
            CurriculumStage(1, "static_furniture", "Adds fixed tables/chairs; still no moving "
                            "people; tray now has free dynamics (can tip).",
                            env_params={"n_obstacles": 6, "n_people": 0, "tray_dynamics": "free"},
                            advance_condition="delivery_success_rate >= 0.9 over 100 episodes",
                            advance_threshold=0.9),
            CurriculumStage(2, "sparse_moving_people", "Adds 1-3 people walking scripted paths "
                            "at low density.",
                            env_params={"n_obstacles": 6, "n_people": 3, "person_speed": 0.8,
                                       "tray_dynamics": "free"},
                            advance_condition="delivery_success_rate >= 0.85 over 100 episodes "
                                              "AND zero human_safety_margin violations",
                            advance_threshold=0.85),
            CurriculumStage(3, "dense_dynamic_crowd", "High person density, unscripted/reactive "
                            "pedestrian motion, randomized furniture layout.",
                            env_params={"n_obstacles": 10, "n_people": 10, "person_speed": 1.2,
                                       "tray_dynamics": "free", "layout": "randomized"},
                            advance_condition="terminal stage"),
        ],
        required_sim_features=["rigid_body_dynamics", "floating_base", "contact_rich_grasping",
                               "free_dynamic_payload", "pedestrian_simulation", "RGB_or_depth_perception"],
        required_controller_class="vla",
        notes=(
            "This is the hardest of the three tasks in this file and the manipulation benchmark "
            "in the paper (a kinematic grasp pin, no tray-balance dynamics, no navigation, no "
            "perception, no people) is not a reduced version of it -- it answers a narrower "
            "question (does the grammar's staged manipulation plan help at all) and was never "
            "meant to scale to this. Reaching this task for real needs: real contact simulation "
            "of the tray/grasp (not a kinematic pin), a pedestrian-simulation layer, and most "
            "likely a vision-language-action model rather than a single fixed reward -- 'carry "
            "drinks across a crowded restaurant' is a family of qualitatively different skills "
            "(navigation, tray balance, human-aware safety) stitched together, not one RL reward "
            "shaped once. The curriculum above exists to make that stitching incremental rather "
            "than attempted in one jump."
        ),
    )


ALL_TASKS = {"walk": walk_task, "parkour": parkour_task, "waiter": waiter_task}


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(ALL_TASKS), default="waiter")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    spec = ALL_TASKS[args.task]()
    out = args.out or f"{args.task}_task_spec.json"
    spec.to_json(out)
    print(f"Wrote {out}")

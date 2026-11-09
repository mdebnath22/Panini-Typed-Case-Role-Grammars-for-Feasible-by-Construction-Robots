"""
Closed-Form Seed Φ : PTG -> GMM
===============================

Hand-specified correspondence for the seed rule set: each applied sūtra maps
to a *screw motion* — a twist ξ ∈ se(3) executed from the current end-effector
frame. Sequential rule application (sūtra composition) therefore maps to
composition of SE(3) actions:

    Φ(r₁ ∘ r₂ ∘ ... ∘ rₙ) = exp(ξ̂₁) · exp(ξ̂₂) · ... · exp(ξ̂ₙ)

Role -> subgroup assignments (Phase 2 closed-form initialization):

    KARMA displacement      -> translation subgroup  (APPROACH, TRANSPORT)
    KARANA orientation      -> rotation subgroup     (ROTATE_TOOL)
    ADHIKARANA contact      -> -z translation        (PLACE)
    APADANA return          -> translation to source (RETRACT)
    grasp/lift verticality  -> z-axis translations   (GRASP, LIFT)

The *homomorphism defect* arises exactly where anuvṛtti changes downstream
bindings — measured in experiments/h5_homomorphism.py.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ptg.karaka import Karaka
from ptg.sutra import AppliedRule
from .se3 import compose, exp_se3


class PhiSeed:
    """Stateful evaluator: maps a derivation to a twist sequence and the
    composed SE(3) motion, given the initial end-effector pose."""

    def __init__(self, trans_weight: float = 1.0):
        self.trans_weight = trans_weight

    # -- per-rule twist ---------------------------------------------------
    def rule_twist(self, applied: AppliedRule, ee_pose: np.ndarray) -> np.ndarray:
        t = applied.sutra.transformation
        f = applied.frame
        p = applied.params
        pos = ee_pose[:3, 3]
        R = ee_pose[:3, :3]

        def to_body(v_world: np.ndarray) -> np.ndarray:
            # twists are executed in the body (end-effector) frame
            return R.T @ v_world

        if t in ("APPROACH", "APPROACH_SIDE"):
            karma = f.get(Karaka.KARMA)
            target = karma.position + np.array([0, 0, p.get("approach_height", 0.1)])
            if t == "APPROACH_SIDE":
                target = target + np.array([p.get("lateral_offset", 0.08), 0, 0])
            return np.concatenate([to_body(target - pos), np.zeros(3)])

        if t == "GRASP":
            dz = -(p.get("approach_height", 0.1) - p.get("grasp_depth", 0.02))
            return np.concatenate([to_body(np.array([0, 0, dz])), np.zeros(3)])

        if t == "LIFT":
            return np.concatenate(
                [to_body(np.array([0, 0, p.get("lift_height", 0.15)])), np.zeros(3)])

        if t == "TRANSPORT":
            goal = f.get(Karaka.SAMPRADANA)
            target = goal.position + np.array(
                [0, 0, p.get("lift_height", 0.15)])
            return np.concatenate([to_body(target - pos), np.zeros(3)])

        if t == "PLACE":
            support = f.get(Karaka.ADHIKARANA)
            target = support.position + np.array(
                [0, 0, p.get("release_height", 0.02)])
            return np.concatenate([to_body(target - pos), np.zeros(3)])

        if t == "ROTATE_TOOL":
            tool = f.get(Karaka.KARANA)
            axis = np.array(tool.attrs.get("axis", (0.0, 0.0, 1.0)))
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            return np.concatenate(
                [np.zeros(3), p.get("rotation_angle", np.pi / 2) * (R.T @ axis)])

        if t == "RETRACT":
            src = f.get(Karaka.APADANA)
            return np.concatenate([to_body(src.position - pos), np.zeros(3)])

        raise KeyError(f"Φ_seed has no assignment for transformation '{t}'")

    # -- derivation -> motion --------------------------------------------
    def map_derivation(self, derivation: List[AppliedRule],
                       ee_init: np.ndarray) -> Dict:
        pose = ee_init.copy()
        twists, poses = [], [pose.copy()]
        for applied in derivation:
            xi = self.rule_twist(applied, pose)
            twists.append(xi)
            pose = pose @ exp_se3(xi)
            poses.append(pose.copy())
        return {
            "twists": np.stack(twists) if twists else np.zeros((0, 6)),
            "poses": np.stack(poses),
            "final": pose,
            "composed": compose(*[exp_se3(x) for x in twists]),
        }

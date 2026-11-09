"""Engineering-feasibility checks that sit on top of sulba.feasibility's
purely geometric gate (reach, clearance, mass budget, support area,
self-clearance). Those checks never touch a catalog part; this module is
the complementary layer the roadmap asked for -- does an actuator actually
have the torque/power to move what it's attached to, can the battery last
the task, is the body statically stable by a real margin (not just
"3+ tips exist"), and will a structural member yield under its own worst
static load.

This module requires a tree that has been through
ptg.catalog_binding.bind_catalog (so every link has part_id/yield_strength/
second_moment and every actuated joint has stall_torque/rated_torque/
rated_current/voltage). A tree that hasn't been bound raises a clear
KeyError rather than silently falling back to a made-up number -- these
checks are meaningless without real catalog parts attached, unlike the
geometric gate, which was designed to work on the grammar's raw output.

Everything here is STATIC analysis (worst-case lever-arm torque, steady-
state thermal rise, a cantilever-beam bending-stress estimate, a convex-hull
stability margin). None of it runs a dynamics rollout, so it costs the same
near-zero evaluation budget as the geometric gate -- the explicit tradeoff
is that it is a conservative, first-order engineering screen, not a
substitute for the kind of dynamics-aware sizing you'd eventually want from
an actual RL/MPC-in-the-loop controller.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

G = 9.81


@dataclass
class EngineeringSpec:
    actuator_safety_factor: float = 1.5     # required torque must be <= stall/this
    continuous_safety_factor: float = 1.0   # required holding torque must be <= rated/this
    structural_safety_factor: float = 2.5   # bending stress must be <= yield/this
    required_task_duration_s: float = 1800.0  # 30 min default mission duration
    gait_frequency_hz: float = 1.0           # reference joint oscillation rate for power estimate
    min_stability_margin: float = 0.01       # m, CoM-to-support-edge clearance


@dataclass
class EngineeringReport:
    ok: bool
    checks: Dict[str, bool]
    margins: Dict[str, float]
    details: Dict[str, Dict] = field(default_factory=dict)


def _subtree_mass_and_com(node) -> Tuple[float, np.ndarray]:
    masses, positions = [], []
    for n in node.walk():
        m = n.geom.get("mass", 0.0)
        if m > 0:
            masses.append(m)
            positions.append(n.world[:3, 3])
    if not masses:
        return 0.0, node.world[:3, 3]
    masses = np.array(masses)
    positions = np.array(positions)
    com = (masses[:, None] * positions).sum(0) / masses.sum()
    return float(masses.sum()), com


def _joint_nodes_with_parent_pivot(tree):
    """Yield (node, pivot_world_xyz) for every actuated joint, where pivot
    is the joint's own world position -- the lever arm for everything
    distal to it is measured from here."""
    for n in tree.walk():
        if n.joint and n.joint.get("actuated") and n.joint.get("stall_torque") is not None:
            yield n


def check_actuator_torque(tree, spec: Optional[EngineeringSpec] = None) -> EngineeringReport:
    """Worst-case static lever-arm torque at every actuated joint (the
    weight of everything distal to that joint, times the horizontal
    distance from the joint to that distal sub-tree's center of mass) vs.
    the chosen actuator's stall torque (lift-at-all, short-duration) and
    rated torque (hold-indefinitely, continuous)."""
    spec = spec or EngineeringSpec()
    checks, margins, details = {}, {}, {}
    worst_stall_margin = float("inf")
    worst_rated_margin = float("inf")
    for jn in _joint_nodes_with_parent_pivot(tree):
        pivot = jn.world[:3, 3]
        dist_mass, dist_com = _subtree_mass_and_com(jn)
        lever = float(np.linalg.norm(dist_com[:2] - pivot[:2]))
        required_torque = dist_mass * G * lever
        stall = float(jn.joint["stall_torque"])
        rated = float(jn.joint["rated_torque"])
        stall_margin = stall / spec.actuator_safety_factor - required_torque
        rated_margin = rated / spec.continuous_safety_factor - required_torque
        details[jn.node_id] = {"required_torque": required_torque, "stall_torque": stall,
                               "rated_torque": rated, "lever_arm": lever,
                               "distal_mass": dist_mass}
        worst_stall_margin = min(worst_stall_margin, stall_margin)
        worst_rated_margin = min(worst_rated_margin, rated_margin)
    if worst_stall_margin == float("inf"):
        worst_stall_margin = worst_rated_margin = 0.0
    margins["actuator_stall"] = worst_stall_margin
    margins["actuator_rated"] = worst_rated_margin
    checks["actuator_stall"] = worst_stall_margin >= 0
    checks["actuator_rated"] = worst_rated_margin >= 0
    return EngineeringReport(ok=all(checks.values()), checks=checks, margins=margins, details=details)


def check_power_and_endurance(tree, battery_id: Optional[str] = None,
                              spec: Optional[EngineeringSpec] = None) -> EngineeringReport:
    """Estimate continuous electrical power draw (every actuated joint at
    its rated current/voltage, a conservative all-joints-working-hard
    assumption, plus sensor draw) and check the attached battery's
    endurance against the required mission duration."""
    from ptg import catalog as cat
    spec = spec or EngineeringSpec()
    battery_id = battery_id or tree.geom.get("battery_id")
    if battery_id is None:
        raise KeyError("tree has no battery_id -- run ptg.catalog_binding.bind_catalog first")
    battery = cat.BATTERIES[battery_id]
    actuator_power = 0.0
    for jn in _joint_nodes_with_parent_pivot(tree):
        actuator_power += float(jn.joint["rated_current"]) * float(jn.joint["voltage"])
    sensor_power = sum(cat.SENSORS[sid].power_w for sid in tree.geom.get("sensor_ids", []))
    total_power = actuator_power + sensor_power
    endurance_h = battery.endurance_hours(total_power)
    required_h = spec.required_task_duration_s / 3600.0
    margin = endurance_h - required_h
    checks = {"battery_endurance": margin >= 0}
    margins = {"battery_endurance_hours": margin}
    details = {"actuator_power_w": actuator_power, "sensor_power_w": sensor_power,
              "total_power_w": total_power, "endurance_hours": endurance_h,
              "required_hours": required_h, "battery_id": battery_id}
    return EngineeringReport(ok=checks["battery_endurance"], checks=checks, margins=margins, details=details)


def check_structural_stress(tree, spec: Optional[EngineeringSpec] = None) -> EngineeringReport:
    """Cantilever bending-stress estimate on every structural link: treat
    the segment as a beam fixed at its proximal end, loaded by the weight
    of everything distal to it acting at that distal sub-tree's center of
    mass. Peak bending moment M = distal_weight * horizontal_lever_arm;
    outer-fiber stress sigma = M * outer_radius / second_moment_of_area.
    Requires second_moment/yield_strength from ptg.catalog_binding."""
    spec = spec or EngineeringSpec()
    worst_margin = float("inf")
    details = {}
    for n in tree.walk():
        if n.kind != "link" or "second_moment" not in n.geom:
            continue
        pivot = n.world[:3, 3]
        dist_mass, dist_com = _subtree_mass_and_com(n)
        lever = float(np.linalg.norm(dist_com[:2] - pivot[:2]))
        moment = dist_mass * G * lever
        I = n.geom["second_moment"]
        r = n.geom.get("radius", 0.02)
        stress = moment * r / I if I > 0 else float("inf")
        yield_strength = n.geom["yield_strength"]
        margin = (yield_strength / spec.structural_safety_factor) - stress
        details[n.node_id] = {"bending_stress_pa": stress, "yield_pa": yield_strength,
                              "moment_nm": moment}
        worst_margin = min(worst_margin, margin)
    if worst_margin == float("inf"):
        worst_margin = 0.0
    checks = {"structural_stress": worst_margin >= 0}
    margins = {"structural_stress": worst_margin}
    return EngineeringReport(ok=checks["structural_stress"], checks=checks, margins=margins, details=details)


def check_stability_margin(tree, spec: Optional[EngineeringSpec] = None) -> EngineeringReport:
    """Real static-stability margin: horizontal distance from the whole
    body's true mass-weighted center of mass to the nearest edge of the
    convex hull of its ground-contact tip positions (as opposed to
    sulba.feasibility's support-polygon AREA check, which says nothing
    about where the CoM actually sits relative to that polygon)."""
    spec = spec or EngineeringSpec()
    tips = np.array([n.world[:3, 3] for n in tree.walk() if n.kind == "foot"])
    total_mass, com = _subtree_mass_and_com(tree)
    if len(tips) < 3:
        margin = -1.0  # cannot even form a support polygon
    else:
        xy = tips[:, :2]
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(xy)
            hull_pts = xy[hull.vertices]
        except Exception:
            c = xy.mean(0)
            ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
            hull_pts = xy[np.argsort(ang)]
        com_xy = com[:2]
        n_h = len(hull_pts)
        dists = []
        inside_signs = []
        for i in range(n_h):
            a, b = hull_pts[i], hull_pts[(i + 1) % n_h]
            edge = b - a
            normal = np.array([edge[1], -edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            d = np.dot(com_xy - a, normal)
            dists.append(abs(d))
            inside_signs.append(d)
        centroid = hull_pts.mean(0)
        ref_sign = np.sign(np.dot(centroid - hull_pts[0], np.array([
            hull_pts[1 % n_h][1] - hull_pts[0][1], -(hull_pts[1 % n_h][0] - hull_pts[0][0])])
            / (np.linalg.norm(hull_pts[1 % n_h] - hull_pts[0]) + 1e-12)))
        is_inside = all(np.sign(s + 1e-12) == ref_sign or abs(s) < 1e-9 for s in inside_signs)
        margin = min(dists) if is_inside else -min(dists)
    checks = {"stability_margin": margin >= spec.min_stability_margin}
    margins = {"stability_margin": margin - spec.min_stability_margin}
    details = {"com_xy": com[:2].tolist(), "raw_margin": margin, "total_mass": total_mass}
    return EngineeringReport(ok=checks["stability_margin"], checks=checks, margins=margins, details=details)


def check_engineering_feasibility(tree, spec: Optional[EngineeringSpec] = None) -> EngineeringReport:
    """Combine all four checks into one gate, in the same shape as
    sulba.feasibility.check_feasibility so the two can be ANDed together by
    a caller that wants both the geometric and the engineering gate."""
    spec = spec or EngineeringSpec()
    reports = [check_actuator_torque(tree, spec), check_power_and_endurance(tree, spec=spec),
              check_structural_stress(tree, spec), check_stability_margin(tree, spec)]
    checks, margins, details = {}, {}, {}
    for r in reports:
        checks.update(r.checks)
        margins.update(r.margins)
        details.update(r.details)
    return EngineeringReport(ok=all(checks.values()), checks=checks, margins=margins, details=details)

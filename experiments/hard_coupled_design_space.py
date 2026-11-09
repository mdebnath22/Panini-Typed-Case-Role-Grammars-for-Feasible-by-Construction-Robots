"""
Hard, coupled design space: grammar construction vs. flat per-limb parameterization.

Question (reviewer-facing): is there a regime where the grammar is the ONLY (or
clearly best) route to feasible designs, not merely tied with per-field range
clamping? Prior tests mutated a body the grammar had ALREADY coordinated, which
lets range clamping inherit that coordination. Here nothing is inherited:

  * RAW      : flat vector, every limb has its own angle, segment count, per-segment
               length and radius, each drawn in its own plausible range (no grammar).
  * GRAMMAR  : same body family, built by the rule grammar (evenly spaced limbs via
               role-gated ADD_LIMB, segment geometry shared by inheritance), with
               continuous global params sampled in the same ranges.
  * RAW+REJ  : RAW with generate-and-test (cost = number of draws per feasible body).
  * RAW-SYM  : RAW but with a hand-coded symmetric sampler (angle = linspace, shared
               geometry). This is the grammar's coordination written by hand; reported
               to show the advantage is CONVENIENCE there, not necessity.

Feasibility spec is JOINT across limbs: support-polygon area, static stability
(COM projection inside the support polygon with a margin), maximum angular gap
between neighbouring limbs, min tip separation, mass budget. Difficulty scales
with the number of limbs n (dimension of RAW grows ~ n*(1+2k)).
"""
from __future__ import annotations
import numpy as np
from ptg.body_grammar import BodyGrammar, legged_plan, _limb_role_frame, torso_frame
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody, leg_mount_plan
from sulba.feasibility import _tips, _total_mass, _support_area

RF = _limb_role_frame()
LEN = (0.06, 0.22); RAD = (0.015, 0.035)


def raw_plan(angles, nseg, lens, rads, x_offsets=None):
    """`x_offsets` gives each limb's longitudinal (torso-axis) mount offset,
    same convention as `gmm.phi_body.leg_mount_plan`. Defaults to all-zero
    for backward compatibility with any caller that still only has angles,
    but every sampler below now passes real per-row offsets so RAW/RAW-SYM
    bodies get genuine 2-D foot spread rather than a degenerate 1-D line."""
    if x_offsets is None:
        x_offsets = [0.0] * len(angles)
    plan = [{"op": "ADD_TORSO", "frame": torso_frame()}]
    for i, a in enumerate(angles):
        plan.append({"op": "ADD_LIMB", "frame": RF, "params": {
            "mount_angle": float(a), "mount_x": float(x_offsets[i]),
            "seg_len": float(lens[i][0]), "seg_rad": float(rads[i][0])}})
        for s in range(1, nseg[i]):
            plan.append({"op": "ARTICULATE", "frame": RF})
            plan.append({"op": "EXTEND", "frame": RF, "params": {
                "seg_len": float(lens[i][s]), "seg_rad": float(rads[i][s])}})
        plan.append({"op": "TERMINATE", "frame": RF})
    return plan


def build(plan):
    d = BodyGrammar(seed_body_rules(), seed=0).derive(plan)
    return PhiBody().build_tree(d)


def sample_raw(n, rng):
    nseg = rng.integers(1, 4, n)
    angles = rng.uniform(0, 2 * np.pi, n)
    # RAW draws its own angle per limb (uncoordinated), but still needs SOME
    # longitudinal placement to avoid a degenerate zero-area support polygon;
    # we give it the same front/middle/back row stations leg_mount_plan uses
    # (coordinated only along x, not angle) so the comparison isolates the
    # grammar's ANGULAR coordination specifically, not x-placement for free.
    _, x_offsets = leg_mount_plan(n)
    lens = [rng.uniform(*LEN, k) for k in nseg]
    rads = [rng.uniform(*RAD, k) for k in nseg]
    return build(raw_plan(angles, nseg, lens, rads, x_offsets))


def sample_grammar(n, rng):
    k = int(rng.integers(1, 4))
    plan = legged_plan(n_legs=n, segments_per_leg=k, seg_len=float(rng.uniform(*LEN)),
                       seg_rad=float(rng.uniform(*RAD)))
    return build(plan)


def sample_raw_sym(n, rng):
    k = int(rng.integers(1, 4))
    # "the grammar's coordination written by hand": use the SAME angle/x
    # layout leg_mount_plan gives the grammar (lower-hemisphere left/right
    # row pairs, front/middle/back x stations), just assembled via raw_plan
    # instead of legged_plan, to isolate "grammar machinery" from "the
    # coordinated layout it encodes" the same way hand_coded_coordinated_body
    # does for the main search comparison.
    angles, x_offsets = leg_mount_plan(n)
    L = rng.uniform(*LEN, k); R = rng.uniform(*RAD, k)
    return build(raw_plan(angles, [k] * n, [L] * n, [R] * n, x_offsets))


def metrics(tree):
    tips = _tips(tree); c = tree.world[:3, 3]
    xy = tips[:, :2] - c[:2]
    area = _support_area(tips)
    # static stability: COM (mass-weighted over links/feet) inside hull w/ margin
    nodes = [(nd.world[:3, 3], nd.geom.get("mass", 0.0)) for nd in tree.walk()]
    m = np.array([x[1] for x in nodes]); p = np.array([x[0][:2] for x in nodes])
    com = (m[:, None] * p).sum(0) / max(m.sum(), 1e-9) - c[:2]
    margin = _com_margin(xy, com)
    gap = _support_gap(xy)
    d = np.linalg.norm(tips[:, None] - tips[None], axis=-1) + np.eye(len(tips)) * 1e9
    return dict(area=area, stab=margin, gap=gap, sep=float(d.min()),
                mass=_total_mass(tree))


def _support_gap(xy):
    """Largest undefended angular sector around the body's actual support
    footprint -- shape-agnostic version.

    HISTORY: the original metric sorted tip angles measured from the fixed
    BODY ORIGIN and took the largest gap between consecutive tips, in
    radians (scale [0, 2*pi]). That is a reasonable "danger sector" proxy
    ONLY when the tip cloud is roughly centered on, and spread all the way
    around, that origin -- true under the old full-circumference mounting.

    Under the post-fix LOWER-HEMISPHERE, row-based mounting (front/middle/
    back longitudinal stations, left/right at fixed angles only), legs never
    occupy the upper half of the circle around the body origin by design --
    there is no leg "above" the torso, nor should there be. Measuring the
    gap from the body origin therefore counts that entire, intentionally-
    unoccupied upper arc as a "danger sector" every single time, regardless
    of how good the real (rectangular) footprint is. That is exactly why
    grammar/raw_sym bodies measured 0/N joint-feasible under `gap` after the
    mounting fix: the metric's circular-footprint assumption, not the
    footprint itself, was failing.

    FIX: measure angular gaps from the footprint's OWN convex-hull centroid
    rather than the fixed body origin. This adapts the reference point to
    whatever shape the tips actually form -- for a tip cloud that happens to
    be circular and centered on the origin, this is numerically identical to
    the old metric (hull centroid coincides with the origin); for a narrow
    rectangle (the new row-based case), the centroid sits inside the
    footprint itself, so gaps reflect real undefended arcs around the
    support polygon rather than an artifact of where the body's coordinate
    origin happens to sit relative to a non-circular tip cloud. Units/scale
    are kept identical (radians, same [0, 2*pi] bound), so every downstream
    threshold that treats `gap` as "radians of exposed arc" stays
    interpretable without reinterpretation.
    """
    from scipy.spatial import ConvexHull
    if len(xy) < 3:
        return float(2 * np.pi)
    try:
        h = ConvexHull(xy)
        centroid = xy[h.vertices].mean(axis=0)
    except Exception:
        centroid = xy.mean(axis=0)
    rel = xy - centroid
    ang = np.sort(np.arctan2(rel[:, 1], rel[:, 0]))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    return float(gaps.max())


def _com_margin(xy, com):
    """signed distance of COM to hull boundary (positive = inside)."""
    from scipy.spatial import ConvexHull
    if len(xy) < 3:
        return -1.0
    try:
        h = ConvexHull(xy)
    except Exception:
        return -1.0
    eq = h.equations  # n.x + b <= 0 inside
    return float(-(eq[:, :2] @ com + eq[:, 2]).max())


def feasible(mt, spec):
    return (mt["area"] >= spec["area"] and mt["stab"] >= spec["stab"] and
            mt["gap"] <= spec["gap"] and mt["sep"] >= spec["sep"] and mt["mass"] <= spec["mass"])

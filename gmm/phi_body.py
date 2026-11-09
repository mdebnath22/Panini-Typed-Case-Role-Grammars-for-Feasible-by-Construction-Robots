from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from ptg.body_karaka import TreeNode

_JOINT_NAME = {0.0: "hinge", 1.0: "slide", 2.0: "ball", 3.0: "free"}

def _trans(v):
    T = np.eye(4); T[:3, 3] = v; return T

def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4); T[:3, :3] = np.array([[c,-s,0],[s,c,0],[0,0,1]]); return T

def _mount_transform(ang, r_torso, x_offset=0.0):
    """Place a limb mount point on the surface of the torso capsule, which
    is built along the LOCAL X-AXIS (see ADD_TORSO / MJCFCompiler's
    fromto=[-len/2,0,0]-[len/2,0,0]). Mount points must therefore be spread
    around the Y-Z plane (the capsule's actual circumference), not the X-Y
    plane -- the previous `_rot_z(ang) @ _trans([r_torso,0,0])` put points
    at (r*cos ang, r*sin ang, 0), which only lands on the surface at
    ang = 90 deg/270 deg; at other angles (e.g. ang = 0/180, common for
    small leg counts) the mount point ends up embedded near the torso's own
    central axis instead of on its skin.

    This returns a transform whose translation is (x_offset, r*cos ang,
    r*sin ang) -- genuinely on the Y-Z circumference, optionally shifted
    along the torso's own long axis by `x_offset` (see leg_mount_plan,
    which assigns each leg a distinct x_offset so the body's legs no
    longer all mount at the same point along its length -- the second-
    order bug that made sulba/feasibility.py's support-polygon check
    degenerate: every foot had X=0, so its X-Y projection was a single
    line, not a polygon) -- and whose rotation maps the limb's own local
    +X (its draw/extension direction, see EXTEND's `_trans([seg_len,0,0])`)
    to the radial (Y,Z) direction, so the limb points straight outward
    from the torso's skin. Local Y (the hinge axis used throughout,
    `axis=(0,1,0)`) maps to the tangential direction around the torso's
    circumference, and local Z maps to the torso's own long axis -- a
    valid, right-handed orthonormal completion.
    """
    c, s = np.cos(ang), np.sin(ang)
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1],
                          [c, -s, 0],
                          [s, c, 0]])
    T[:3, 3] = [x_offset, r_torso * c, r_torso * s]
    return T


def lower_hemisphere_mount_angles(n_legs, endpoint=True):
    """Evenly-spaced mount angles restricted to the lower hemisphere of the
    torso's circumference, in the (Y,Z) angle convention `_mount_transform`
    uses (ang=pi/2 -> straight up/+Z, ang=3*pi/2 -> straight down/-Z,
    ang=0 or pi -> sideways/+-Y). Legs hang below/beside the torso like a
    real legged animal's, rather than wrapping the full 360 degrees -- a
    full wrap put legs pointing straight up (useless) or straight down
    through the floor once mounting was fixed to land on the torso's
    actual surface. The arc runs from pi to 2*pi so it is bounded by the
    two purely-sideways angles, sweeping through straight-down in between
    -- every legal mount angle has a non-positive Z component.

    Kept as a standalone utility (angle-only, no X placement); the main
    body-building call sites now use `leg_mount_plan` instead, which also
    assigns longitudinal (X) offsets so feasibility's support-polygon
    check has real area to measure (see that function's docstring)."""
    return list(np.linspace(np.pi, 2 * np.pi, n_legs, endpoint=endpoint))


def leg_mount_plan(n_legs, torso_len=0.30, x_span_frac=0.6):
    """Assigns each leg BOTH a circumferential mount angle (restricted to
    the lower hemisphere, see lower_hemisphere_mount_angles's docstring)
    AND a longitudinal (X) offset along the torso's length, so legs no
    longer all mount at the same point along the torso (X=0 for every
    leg) -- which is what made sulba/feasibility.py's support-polygon
    check always see zero area: every foot sat at the same X, so its
    projection onto the X-Y ground plane collapsed to a single line, no
    matter how the circumferential angles were spread.

    Legs are grouped into left/right pairs at `n_rows = ceil(n_legs/2)`
    evenly-spaced stations along the torso's length -- an ordinary
    quadruped/hexapod-style layout (front legs, back legs, and any rows
    in between) -- so the body gets a real 2D footprint once it stands.
    `x_span_frac` controls how much of the torso's length those stations
    use (default 0.6, leaving margin near the capsule's rounded caps). An
    odd leftover leg in the last row is mounted centrally, straight down,
    rather than forced into an asymmetric left/right pair.

    Returns (angles, x_offsets): two parallel lists of length n_legs, each
    usable directly as `_mount_transform(ang, r_torso, x_offset)`'s
    arguments, or passed through to `legged_plan`."""
    n_rows = (n_legs + 1) // 2
    half_span = 0.5 * x_span_frac * torso_len
    row_xs = [0.0] if n_rows == 1 else list(np.linspace(-half_span, half_span, n_rows))
    straight_down = 1.5 * np.pi
    sep = 0.5 * np.pi  # left/right angular separation within a row; stays well inside [pi, 2pi]
    ang_left, ang_right = straight_down - sep / 2, straight_down + sep / 2

    angles, xs = [], []
    remaining = n_legs
    for row in range(n_rows):
        x = row_xs[row]
        if remaining >= 2:
            angles += [ang_left, ang_right]
            xs += [x, x]
            remaining -= 2
        else:
            angles += [straight_down]
            xs += [x]
            remaining -= 1
    return angles, xs


class PhiBody:
    def rule_placement(self, applied) -> Dict:
        op = applied.sutra.op; p = applied.params
        if op == "ADD_TORSO":
            return {"kind": "torso", "local": _trans([0,0,p.get("torso_height",0.30)]),
                   "geom": {"length": p.get("torso_len",0.30), "radius": p.get("torso_rad",0.08),
                            "mass": p.get("torso_mass",2.0)}, "joint": {"type": "free"}}
        if op == "ADD_LIMB":
            ang = p.get("mount_angle", 0.0); r_torso = p.get("torso_rad", 0.08)
            x_offset = p.get("mount_x", 0.0)
            local = _mount_transform(ang, r_torso, x_offset=x_offset)
            jt = _JOINT_NAME.get(p.get("joint_type", 0.0), "hinge")
            return {"kind": "link", "local": local,
                   "geom": {"length": p.get("seg_len",0.12), "radius": p.get("seg_rad",0.02),
                            "mass": p.get("seg_mass",0.15)},
                   "joint": {"type": jt, "axis": (0,1,0), "range": p.get("joint_range",0.9),
                            "actuated": True}}
        if op == "ARTICULATE":
            jt = _JOINT_NAME.get(p.get("joint_type", 0.0), "hinge")
            return {"kind": "joint_marker", "local": _trans([p.get("_parent_len",0.12),0,0]),
                   "geom": {}, "joint": {"type": jt, "axis": (0,1,0),
                            "range": p.get("joint_range",0.9),
                            "actuated": bool(p.get("actuated",1.0))}}
        if op == "EXTEND":
            taper = p.get("taper", 0.85); base_len = p.get("seg_len", 0.12)
            base_rad = p.get("seg_rad", 0.02)
            # EXTEND's body must start COINCIDENT with its parent (the
            # ARTICULATE joint_marker, which is already offset by the
            # PARENT segment's own length to sit at the tip of the
            # preceding segment) -- local=identity -- and let its OWN
            # geom (fromto 0 -> base_len below) provide the visual
            # extension. The previous `_trans([base_len,0,0])` here
            # translated the body forward by its own length BEFORE
            # drawing that same length again, leaving a real ~base_len
            # empty 3D gap between consecutive segments in every
            # multi-segment leg (segments_per_leg >= 2) -- a pre-existing
            # bug, unrelated to today's mounting fix, confirmed by
            # inspecting the compiled XML directly (the joint_marker body
            # in between carries no geom at all) and reproduced visually
            # at oblique camera angles (hidden from directly overhead,
            # where the gap's dominant axis projects away).
            return {"kind": "link", "local": _trans([0,0,0]),
                   "geom": {"length": base_len, "radius": base_rad*taper,
                            "mass": p.get("seg_mass",0.15)*taper}, "joint": None}
        if op == "TERMINATE":
            base_len = p.get("seg_len", 0.12)
            return {"kind": "foot", "local": _trans([base_len,0,0]),
                   "geom": {"radius": p.get("foot_rad",0.025), "mass": p.get("foot_mass",0.05)},
                   "joint": None}
        if op == "SYMMETRIZE":
            M = np.eye(4); M[1,1] = -1.0
            return {"kind": "mirror", "local": M, "geom": {}, "joint": None}
        raise KeyError(f"Φ_body has no placement for op '{op}'")

    def build_tree(self, derivation) -> TreeNode:
        root = None; current_tip = None; current_limb_root = None
        last_seg_len = 0.12; node_i = 0
        for applied in derivation:
            place = self.rule_placement(applied); op = applied.sutra.op
            if op == "ADD_TORSO":
                root = TreeNode("torso", "torso", place["local"], place["geom"], place["joint"])
                current_tip = root; continue
            if op == "ADD_LIMB":
                node = TreeNode(f"limb{node_i}_root", "link", place["local"],
                               place["geom"], place["joint"]); node_i += 1
                root.children.append(node)
                current_limb_root = node; current_tip = node
                last_seg_len = place["geom"]["length"]; continue
            if op == "ARTICULATE":
                place2 = dict(place); place2["local"] = _trans([last_seg_len,0,0])
                node = TreeNode(f"joint{node_i}", "joint_marker", place2["local"], {}, place2["joint"])
                node_i += 1; current_tip.children.append(node); current_tip = node; continue
            if op == "EXTEND":
                node = TreeNode(f"seg{node_i}", "link", place["local"], place["geom"], place["joint"])
                node_i += 1; current_tip.children.append(node); current_tip = node
                last_seg_len = place["geom"]["length"]; continue
            if op == "TERMINATE":
                node = TreeNode(f"foot{node_i}", "foot", place["local"], place["geom"], None)
                node_i += 1; current_tip.children.append(node)
                current_tip = current_limb_root; continue
            if op == "SYMMETRIZE":
                if current_limb_root is not None:
                    mirrored = _mirror_subtree(current_limb_root, node_i)
                    node_i += 1 + sum(1 for _ in current_limb_root.walk())
                    root.children.append(mirrored)
                continue
        _fill_world(root, np.eye(4))
        return root


def _mirror_subtree(node, offset):
    M = np.eye(4); M[1,1] = -1.0
    new_local = M @ node.local @ M
    clone = TreeNode(f"mir{offset}_{node.node_id}", node.kind, new_local,
                     dict(node.geom), dict(node.joint) if node.joint else None)
    for c in node.children:
        clone.children.append(_mirror_subtree(c, offset))
    return clone


def _fill_world(node, parent_world):
    node.world = parent_world @ node.local
    for c in node.children:
        _fill_world(c, node.world)

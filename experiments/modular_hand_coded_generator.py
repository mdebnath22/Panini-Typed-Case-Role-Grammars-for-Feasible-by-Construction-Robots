"""
A FAIRER hand-coded competitor for the extension-cost study
(experiments/effort_reuse_study.py).

A reviewer can reasonably object that build_hand_coded_tree
(experiments/hand_coded_coordinated_body.py) was never designed with
reuse in mind, so its 77.5%-duplication outcome measures a strawman, not
hand-coded generators in general. This file is the steelman: a
hand-coded generator built from the start around a per-limb-type
DISPATCH REGISTRY (a plain Python dict mapping a limb-type name to a
builder function), the standard, idiomatic way an engineer who expected
to need more than one limb type would write this -- no production-rule
grammar, no BodySutra, no specificity arbitration, just a dict and
ordinary functions.

If THIS generator can absorb the "knee" feature as cheaply as the
grammar (a new entry in the registry, zero lines changed in the base
dispatcher), the extension-cost finding narrows to the specific,
non-modular hand-coded implementation tested first, not to "hand-coded
generators" as a class -- exactly the possibility
experiments/effort_reuse_study.py's own limitations paragraph flags.
We build it and measure it rather than assume the answer either way.

Run:
    python3 -m experiments.modular_hand_coded_generator
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from experiments.hand_coded_coordinated_body import _trans, _rot_z, _mount_transform, _fill_world
from gmm.phi_body import leg_mount_plan
from experiments.m_search_loop import Genome
from gmm.mjcf_compiler import MJCFCompiler
from ptg.body_karaka import TreeNode

LimbBuilder = Callable[[TreeNode, int, Genome], Tuple[TreeNode, int]]

# ---------------------------------------------------------------------------
# BASE generator, as it would have existed BEFORE the "knee" feature was
# ever requested: one limb type ("single"), reached through a registry
# dict rather than inlined, because that is how a developer anticipating
# more limb types would write it even with only one type so far.
# ---------------------------------------------------------------------------

def _build_single_segment_limb(tip: TreeNode, node_i: int, genome: Genome) -> Tuple[TreeNode, int]:
    """The only limb type in the base design: no additional segment, the
    limb root passed in already IS the tip the foot attaches to."""
    return tip, node_i


LIMB_BUILDERS: Dict[str, LimbBuilder] = {
    "single": _build_single_segment_limb,
}


def build_modular_tree(genome: Genome, limb_types: List[str],
                       mount_angles: Optional[List[float]] = None,
                       mount_x_offsets: Optional[List[float]] = None,
                       registry: Dict[str, LimbBuilder] = LIMB_BUILDERS) -> TreeNode:
    """The base dispatcher. Written once; never edited to add a limb
    type -- new types are added by registering a new function in
    `registry`, exactly the shape of extension the grammar's rule list
    offers. If this function's body needs to change to add a type, that
    is a real cost and counts against the modular design."""
    if mount_angles is None or mount_x_offsets is None:
        mount_angles, mount_x_offsets = leg_mount_plan(genome.n_legs)

    torso_h, torso_len, torso_rad, torso_mass = 0.30, 0.30, 0.08, 2.0
    root = TreeNode("torso", "torso", _trans([0, 0, torso_h]),
                   {"length": torso_len, "radius": torso_rad, "mass": torso_mass},
                   {"type": "free"})

    node_i = 0
    for ang, x, ltype in zip(mount_angles, mount_x_offsets, limb_types):
        local = _mount_transform(ang, torso_rad, x_offset=x)
        limb_root = TreeNode(f"limb{node_i}_root", "link", local,
                             {"length": genome.seg_len, "radius": genome.seg_rad, "mass": 0.15},
                             {"type": "hinge", "axis": (0, 1, 0), "range": 0.9, "actuated": True})
        node_i += 1
        root.children.append(limb_root)
        tip, node_i = registry[ltype](limb_root, node_i, genome)
        last_len = genome.seg_len if ltype == "single" else genome.seg_len  # foot sits one seg_len past tip either way
        foot = TreeNode(f"foot{node_i}", "foot", _trans([last_len, 0, 0]),
                        {"radius": 0.025, "mass": 0.05}, None)
        node_i += 1
        tip.children.append(foot)

    _fill_world(root, np.eye(4))
    return root


def build_modular_body(genome: Genome, limb_types: List[str]) -> str:
    tree = build_modular_tree(genome, limb_types)
    return MJCFCompiler().compile(tree)


# ---------------------------------------------------------------------------
# EXTENSION: the "knee" feature, added exactly the way the registry is
# meant to absorb a new limb type -- one new builder function, one new
# dict entry. build_modular_tree above is NOT edited to add this.
# ---------------------------------------------------------------------------

def _build_knee_limb(tip: TreeNode, node_i: int, genome: Genome) -> Tuple[TreeNode, int]:
    """New limb type: one extra articulated segment past the limb root.
    joint_marker translates by the limb root's own length to sit at its
    tip; the new segment's body then starts COINCIDENT with joint_marker
    (local=identity) and lets its own capsule geom (fromto 0->seg_len)
    provide the extension, matching gmm/phi_body.py's EXTEND rule post-fix
    (an earlier version double-translated by seg_len here too, the same
    pre-existing bug EXTEND itself had, independently duplicated)."""
    taper = 0.85
    joint_marker = TreeNode(f"joint{node_i}", "joint_marker", _trans([genome.seg_len, 0, 0]),
                            {}, {"type": "hinge", "axis": (0, 1, 0), "range": 0.9, "actuated": True})
    node_i += 1
    tip.children.append(joint_marker)
    seg = TreeNode(f"seg{node_i}", "link", _trans([0, 0, 0]),
                  {"length": genome.seg_len, "radius": genome.seg_rad * taper,
                   "mass": 0.15 * taper}, None)
    node_i += 1
    joint_marker.children.append(seg)
    return seg, node_i


LIMB_BUILDERS["knee"] = _build_knee_limb  # the entire extension, besides the function above


if __name__ == "__main__":
    import numpy as _np
    from experiments.m_search_loop import random_genome
    from experiments.mh1_xml_baseline import _xml_loads_and_steps
    rng = _np.random.default_rng(0)
    ok = 0
    n = 30
    for _ in range(n):
        g = random_genome(rng)
        types = [rng.choice(["single", "knee"]) for _ in range(g.n_legs)]
        xml = build_modular_body(g, types)
        ok += int(_xml_loads_and_steps(xml))
    print(f"Modular hand-coded generator with 'knee' registered: {ok}/{n} load+step OK")

"""
A fully hand-coded, non-grammar generator with the SAME good design
decisions the grammar encodes -- evenly spaced limb mounting, one shared
segment length/radius across every limb, additive segment placement,
default torso geometry -- built with ordinary Python array code and
TreeNode construction, with NO BodyGrammar, NO BodySutra rule matching,
NO specificity arbitration, and NO anuvrtti/inheritance mechanism
anywhere in the call path.

Why this script exists: a reviewer's strongest remaining objection to the
coupled-feasibility result (Section~sec:coupled-feasibility,
`mutate_random_mount_angles` in coupled_feasibility_validity.py) is that
the "mechanism-targeted counterfactual" there still derives through
BodyGrammar for everything except the one randomized field (mount angle)
-- so it isolates ONE field, but never asks whether the grammar
MACHINERY itself (rules, roles, specificity, anuvrtti) is doing any work
beyond encoding "even spacing + shared segment geometry" as data. This
script answers that directly: it reproduces the grammar's OUTPUT exactly
(same tree shape, same default field values) for a hand-rolled genome,
through a function that never imports ptg.body_grammar, ptg.body_sutra,
or ptg.paribhasa at all.

If this hand-coded generator performs identically to grammar-guided
construction on the coupled, cross-limb feasibility criterion, that is
real evidence for the reviewer's "good design decisions, not the grammar
abstraction" reading -- and this script reports that honestly, whichever
way it comes out, rather than only running the comparisons the grammar is
expected to win.

Run:
    python3 -m experiments.hand_coded_coordinated_body
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import chi2_contingency

from ptg.body_karaka import TreeNode
from gmm.mjcf_compiler import MJCFCompiler
from experiments.coupled_feasibility_validity import (
    CoupledFeasibilitySpec, calibrate_spec, check_feasibility_mjcf)
from experiments.m_search_loop import random_genome, Genome
from experiments.mh1_xml_baseline import _xml_loads_and_steps
from gmm.phi_body import leg_mount_plan


def _trans(v):
    T = np.eye(4); T[:3, 3] = v; return T


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4); T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); return T


def _mount_transform(ang, r_torso, x_offset=0.0):
    """Same fix as gmm.phi_body._mount_transform: the torso capsule is
    built along local X, so mount points must be spread around the Y-Z
    plane (its actual circumference), not the X-Y plane, and now also
    carries a per-leg longitudinal x_offset (see leg_mount_plan) so legs
    don't all mount at the same point along the torso's length. See that
    function's docstring for the full derivation. Kept duplicated here
    (rather than imported) to match this file's existing
    no-grammar-machinery, plain-Python-loop style."""
    c, s = np.cos(ang), np.sin(ang)
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1],
                          [c, -s, 0],
                          [s, c, 0]])
    T[:3, 3] = [x_offset, r_torso * c, r_torso * s]
    return T


def _fill_world(node, parent_world):
    node.world = parent_world @ node.local
    for c in node.children:
        _fill_world(c, node.world)


def build_hand_coded_tree(genome: Genome, mount_angles: List[float] = None,
                          mount_x_offsets: List[float] = None) -> TreeNode:
    """Builds the exact tree shape PhiBody.build_tree would produce for an
    equivalent genome, by ordinary Python loops over plain numbers -- no
    grammar rule object, role binding, or specificity decision anywhere.
    Every limb gets the SAME mount-angle/longitudinal-offset layout (even,
    unless the caller overrides it) and the SAME seg_len/seg_rad, hand-set
    per limb/segment rather than inherited through anuvrtti."""
    if mount_angles is None or mount_x_offsets is None:
        mount_angles, mount_x_offsets = leg_mount_plan(genome.n_legs)

    torso_h, torso_len, torso_rad, torso_mass = 0.30, 0.30, 0.08, 2.0
    root = TreeNode("torso", "torso", _trans([0, 0, torso_h]),
                   {"length": torso_len, "radius": torso_rad, "mass": torso_mass},
                   {"type": "free"})

    node_i = 0
    for ang, x in zip(mount_angles, mount_x_offsets):
        local = _mount_transform(ang, torso_rad, x_offset=x)
        limb_root = TreeNode(f"limb{node_i}_root", "link", local,
                             {"length": genome.seg_len, "radius": genome.seg_rad, "mass": 0.15},
                             {"type": "hinge", "axis": (0, 1, 0), "range": 0.9, "actuated": True})
        node_i += 1
        root.children.append(limb_root)
        tip = limb_root
        last_len = genome.seg_len
        taper = 0.85  # matches PhiBody's EXTEND op exactly (radius/mass taper per added segment)
        for _ in range(1, genome.segments_per_leg):
            # NOTE: this reproduces PhiBody.build_tree's own transform chain
            # exactly: ARTICULATE (joint_marker) translates by the PREVIOUS
            # segment's length to sit at its tip, and the new segment's own
            # body ("seg") starts COINCIDENT with that joint_marker --
            # local=identity -- letting its OWN capsule geom (fromto
            # 0->seg_len) provide the visual/physical extension, exactly
            # matching gmm/phi_body.py's EXTEND rule post-fix. (An earlier
            # version of this function used _trans([genome.seg_len,0,0])
            # here too, double-translating by the segment's own length on
            # top of the joint_marker's offset -- the same pre-existing bug
            # EXTEND itself had, independently duplicated here. Fixed to
            # match, so this generator and the grammar inherit the SAME
            # transform chain and cannot bias the comparison between them.)
            joint_marker = TreeNode(f"joint{node_i}", "joint_marker", _trans([last_len, 0, 0]),
                                    {}, {"type": "hinge", "axis": (0, 1, 0), "range": 0.9, "actuated": True})
            node_i += 1
            tip.children.append(joint_marker)
            seg = TreeNode(f"seg{node_i}", "link", _trans([0, 0, 0]),
                          {"length": genome.seg_len, "radius": genome.seg_rad * taper,
                           "mass": 0.15 * taper}, None)
            node_i += 1
            joint_marker.children.append(seg)
            tip = seg
            last_len = genome.seg_len
        foot = TreeNode(f"foot{node_i}", "foot", _trans([last_len, 0, 0]),
                        {"radius": 0.025, "mass": 0.05}, None)
        node_i += 1
        tip.children.append(foot)

    _fill_world(root, np.eye(4))
    return root


def mutate_hand_coded(base_genome: Genome, rng: random.Random, jitter_angles: bool = False) -> str:
    """Hand-coded analogue of grammar-guided mutation: perturb one genome
    field (same mutation operator family as m_search_loop.mutate), then
    rebuild the tree with build_hand_coded_tree instead of re-deriving
    through BodyGrammar. `jitter_angles=True` reproduces the mechanism-
    targeted counterfactual's independent-random-angle condition using
    the SAME hand-coded (non-grammar) code path, so every cell in the
    comparison table differs only in whether angles are coordinated, not
    in whether a grammar object is involved anywhere."""
    g = Genome(n_legs=base_genome.n_legs, segments_per_leg=base_genome.segments_per_leg,
              seg_len=base_genome.seg_len, seg_rad=base_genome.seg_rad,
              angle_jitter=base_genome.angle_jitter).clamp()
    field = rng.choice(["n_legs", "segments_per_leg", "seg_len", "seg_rad"])
    if field == "n_legs":
        g.n_legs = int(np.clip(g.n_legs + rng.choice([-1, 1]), 2, 8))
    elif field == "segments_per_leg":
        g.segments_per_leg = int(np.clip(g.segments_per_leg + rng.choice([-1, 1]), 1, 3))
    elif field == "seg_len":
        g.seg_len = float(np.clip(g.seg_len * rng.uniform(0.8, 1.25), 0.05, 0.30))
    else:
        g.seg_rad = float(np.clip(g.seg_rad * rng.uniform(0.8, 1.25), 0.01, 0.05))
    g.clamp()
    mount_angles, mount_x_offsets = None, None
    if jitter_angles:
        # Isolate the angle variable: x offsets stay at the SAME
        # coordinated longitudinal layout leg_mount_plan would assign, only
        # the circumferential angle is independently randomized.
        _, mount_x_offsets = leg_mount_plan(g.n_legs)
        mount_angles = [rng.uniform(0, 2 * np.pi) for _ in range(g.n_legs)]
    tree = build_hand_coded_tree(g, mount_angles, mount_x_offsets)
    return MJCFCompiler().compile(tree)


def run(n_trials: int = 200, target_ps=(0.30, 0.10, 0.05), seed: int = 0,
       verbose: bool = True) -> Dict:
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    comp = MJCFCompiler()

    all_names = ["grammar_guided", "hand_coded_coordinated (no grammar)",
                "hand_coded_random_angle (no grammar)"]
    all_results: Dict = {}
    for target_p in target_ps:
        spec = calibrate_spec(target_p, seed=seed, verbose=False)
        counts = {n: 0 for n in all_names}
        loads_counts = {n: 0 for n in all_names}

        def _check(name, xml):
            if _xml_loads_and_steps(xml):
                loads_counts[name] += 1
                if check_feasibility_mjcf(xml, spec):
                    counts[name] += 1

        for _ in range(n_trials):
            base_genome = random_genome(rng_np)

            # grammar-guided: reuse the real mechanism from coupled_feasibility_validity
            from experiments.m_search_loop import mutate as grammar_mutate
            from experiments.mh1_xml_baseline import _tree_of
            try:
                mg = grammar_mutate(base_genome, rng_np)
                _check("grammar_guided", comp.compile(_tree_of(mg)))
            except Exception:
                pass

            # hand-coded, coordinated (evenly-spaced, shared segment params), no grammar
            try:
                _check("hand_coded_coordinated (no grammar)",
                      mutate_hand_coded(base_genome, rng_py, jitter_angles=False))
            except Exception:
                pass

            # hand-coded, independently randomized mount angles, no grammar
            try:
                _check("hand_coded_random_angle (no grammar)",
                      mutate_hand_coded(base_genome, rng_py, jitter_angles=True))
            except Exception:
                pass

        all_results[target_p] = {"counts": counts, "loads_counts": loads_counts,
                                 "min_support_area": spec.min_support_area}
        def _chi2(a, b):
            table = [[counts[a], n_trials - counts[a]], [counts[b], n_trials - counts[b]]]
            return float(chi2_contingency(table, correction=True)[1])

        all_results[target_p]["chi2_grammar_vs_hand_coded"] = _chi2(
            "grammar_guided", "hand_coded_coordinated (no grammar)")
        all_results[target_p]["chi2_hand_coded_vs_random_angle"] = _chi2(
            "hand_coded_coordinated (no grammar)", "hand_coded_random_angle (no grammar)")

        if verbose:
            print(f"\ntarget_p={target_p} (min_support_area={spec.min_support_area:.3f}):")
            print(f"{'condition':<40}{'loads+steps':>14}{'+ coupled feasibility':>24}")
            for name in all_names:
                print(f"{name:<40}{loads_counts[name]:>14}/{n_trials}"
                     f"{counts[name]:>20}/{n_trials}")
            print(f"  chi2 p, grammar vs.\\ hand-coded coordinated (no grammar): "
                 f"{all_results[target_p]['chi2_grammar_vs_hand_coded']:.4f}  "
                 f"(no significant difference means the grammar MACHINERY adds nothing"
                 f" measurable beyond the coordinated design decisions it encodes)")
            print(f"  chi2 p, hand-coded coordinated vs.\\ hand-coded random-angle: "
                 f"{all_results[target_p]['chi2_hand_coded_vs_random_angle']:.4f}  "
                 f"(a significant difference here, with NO grammar in either arm, means"
                 f" coordination itself -- not the grammar -- is what matters)")
    return all_results


if __name__ == "__main__":
    run()

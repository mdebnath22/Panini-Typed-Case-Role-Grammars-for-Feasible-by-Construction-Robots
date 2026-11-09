"""
Exp 2 (PREREG.md): asymmetric payload environment and the specificity-2 load rule.

* Payload: 1.5 kg point mass at (+0.20, 0, 0) m from the torso centre, fixed to the torso.
* B1.3_ADD_LIMB_LOAD: specificity 2, consumes {has_torso, payload_task}; the task sets
  payload_task as an initial marker (same mechanism as the existing reach override B1.2).
  Realization (fixed in PREREG.md, not tuned): theta' = theta + alpha*sin(phi - theta),
  alpha = 0.35, phi = 0; limbs with cos(theta' - phi) > 0 get every segment scaled by 1.3.
  Exports are identical to B1.1.
The paper's 7-rule catalogue is unchanged: payload_body_rules() = seed_body_rules() + [B1.3],
and B1.3 can only fire when payload_task is present.
"""
from __future__ import annotations
import math
import numpy as np
from ptg.body_sutra import BodySutra, seed_body_rules
from ptg.body_karaka import BodyKaraka
from ptg.body_grammar import BodyGrammar, legged_plan
from gmm.phi_body import PhiBody, TreeNode, _trans, _fill_world, _mirror_subtree
from sulba.feasibility import _tips, _support_area
from experiments.hard_coupled_design_space import _com_margin, _support_gap, LEN, RAD

ALPHA, PHI, SCALE = 0.35, 0.0, 1.3
PAYLOAD_MASS, PAYLOAD_OFFSET = 1.5, np.array([0.20, 0.0, 0.0])

B13 = BodySutra("B1.3_ADD_LIMB_LOAD", "ADD_LIMB", frozenset({BodyKaraka.ANGA}),
                consumes_markers=frozenset({"has_torso", "payload_task"}),
                params={"seg_len": 0.12, "seg_rad": 0.02, "seg_mass": 0.15, "mount_angle": 0.0,
                        "joint_type": 0.0, "joint_range": 0.9,
                        "load_bias": ALPHA, "load_dir": PHI, "load_scale": SCALE},
                emits_markers=frozenset({"has_link", "open_chain", "payload_task"}),
                anuvrtti_export=frozenset({"seg_len", "seg_rad", "seg_mass", "joint_range"}),
                specificity=2)


def payload_body_rules():
    return seed_body_rules() + [B13]


class LoadPhiBody(PhiBody):
    """PhiBody plus the realization of B1.3's three parameters. For any derivation that
    does not use B1.3 it produces exactly the same tree as PhiBody (checked in __main__)."""

    def rule_placement(self, applied):
        place = super().rule_placement(applied)
        p = applied.params
        if applied.sutra.op == "ADD_LIMB" and "load_bias" in p:
            th = p.get("mount_angle", 0.0)
            th2 = th + p["load_bias"] * math.sin(p["load_dir"] - th)
            r_torso = p.get("torso_rad", 0.08)
            from gmm.phi_body import _rot_z
            place["local"] = _rot_z(th2) @ _trans([r_torso, 0, 0])
            place["_scale"] = p["load_scale"] if math.cos(th2 - p["load_dir"]) > 0 else 1.0
            place["geom"]["length"] *= place["_scale"]
        return place

    def build_tree(self, derivation):
        root = None; tip = None; limb_root = None; last = 0.12; i = 0; scale = 1.0
        for applied in derivation:
            place = self.rule_placement(applied); op = applied.sutra.op
            if op == "ADD_TORSO":
                root = TreeNode("torso", "torso", place["local"], place["geom"], place["joint"]); tip = root; continue
            if op == "ADD_LIMB":
                scale = place.get("_scale", 1.0)
                node = TreeNode(f"limb{i}_root", "link", place["local"], place["geom"], place["joint"]); i += 1
                root.children.append(node); limb_root = node; tip = node; last = place["geom"]["length"]; continue
            if op == "ARTICULATE":
                p2 = dict(place); p2["local"] = _trans([last, 0, 0])
                node = TreeNode(f"joint{i}", "joint_marker", p2["local"], {}, p2["joint"]); i += 1
                tip.children.append(node); tip = node; continue
            if op == "EXTEND":
                g = dict(place["geom"]); g["length"] *= scale
                node = TreeNode(f"seg{i}", "link", _trans([g["length"], 0, 0]) if scale != 1.0 else place["local"],
                                g, place["joint"]); i += 1
                tip.children.append(node); tip = node; last = g["length"]; continue
            if op == "TERMINATE":
                loc = place["local"] if scale == 1.0 else _trans([last, 0, 0])
                node = TreeNode(f"foot{i}", "foot", loc, place["geom"], None); i += 1
                tip.children.append(node); tip = limb_root; continue
            if op == "SYMMETRIZE":
                if limb_root is not None:
                    root.children.append(_mirror_subtree(limb_root, i)); i += 1 + sum(1 for _ in limb_root.walk())
                continue
        _fill_world(root, np.eye(4))
        return root


def grammar_tree(n, k, L, R, payload_rule=True, mount_angles=None):
    """Same plan (legged_plan) either way; only the rule catalogue and the task marker
    differ, so B1.3 is chosen by specificity arbitration, not by a different code path."""
    plan = legged_plan(n_legs=n, segments_per_leg=k, mount_angles=mount_angles, seg_len=L, seg_rad=R)
    if payload_rule:
        d = BodyGrammar(payload_body_rules(), seed=0).derive(plan, init_markers={"payload_task"})
        return LoadPhiBody().build_tree(d)
    d = BodyGrammar(seed_body_rules(), seed=0).derive(plan)
    return LoadPhiBody().build_tree(d)


def payload_metrics(tree):
    """hard_coupled_design_space.metrics with the payload in the centre of mass and mass."""
    tips = _tips(tree); c = tree.world[:3, 3]; xy = tips[:, :2] - c[:2]
    nodes = [(nd.world[:3, 3][:2], nd.geom.get("mass", 0.0)) for nd in tree.walk()]
    m = np.array([x[1] for x in nodes] + [PAYLOAD_MASS])
    p = np.array([x[0] for x in nodes] + [c[:2] + PAYLOAD_OFFSET[:2]])
    com = (m[:, None] * p).sum(0) / m.sum() - c[:2]
    d = np.linalg.norm(tips[:, None] - tips[None], axis=-1) + np.eye(len(tips)) * 1e9
    return dict(area=_support_area(tips), stab=_com_margin(xy, com), gap=_support_gap(xy),
                sep=float(d.min()), mass=float(m.sum()))


def payload_spec(n, tau):
    from experiments.hard_coupled_calibrated import spec_tau
    sp = dict(spec_tau(n, tau)); sp["mass"] += PAYLOAD_MASS; return sp


if __name__ == "__main__":
    # regression: without B1.3 the tree equals the paper's PhiBody tree exactly
    rng = np.random.default_rng(0); mx = 0.0
    for _ in range(50):
        n = int(rng.integers(3, 9)); k = int(rng.integers(1, 4)); L = rng.uniform(*LEN); R = rng.uniform(*RAD)
        a = PhiBody().build_tree(BodyGrammar(seed_body_rules(), seed=0).derive(legged_plan(n, k, seg_len=L, seg_rad=R)))
        b = grammar_tree(n, k, L, R, payload_rule=False)
        A = np.array([x.world for x in a.walk()]); B = np.array([x.world for x in b.walk()])
        mx = max(mx, float(np.abs(A - B).max()))
    print("max |PhiBody - LoadPhiBody| without B1.3:", mx)
    # arbitration check: B1.3 is the rule chosen for every limb when payload_task is set
    plan = legged_plan(8, 2, seg_len=0.15, seg_rad=0.02)
    der = BodyGrammar(payload_body_rules(), seed=0).derive(plan, init_markers={"payload_task"})
    print("ADD_LIMB rules chosen:", {a.sutra.rule_id for a in der if a.sutra.op == "ADD_LIMB"})
    t = grammar_tree(8, 2, 0.15, 0.02)
    tp = _tips(t)[:, :2]
    print("tip radii by angle (payload rule):", np.round(sorted(zip(np.round(np.degrees(np.arctan2(tp[:,1], tp[:,0]))), np.round(np.linalg.norm(tp, axis=1), 3))), 3).tolist())

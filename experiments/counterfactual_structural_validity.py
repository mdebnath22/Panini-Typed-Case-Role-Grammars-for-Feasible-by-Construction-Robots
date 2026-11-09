"""
Counterfactual: is grammar-guided structural validity NECESSARY or just
CONVENIENT?
==============================================================================

expanded_validity_comparison.py's naive structural mutators (duplicate,
delete, subtree_swap, crossover) are deliberately adversarial -- they don't
try to preserve structural invariants, so their low validity rates (0-39%)
don't by themselves prove a non-grammar approach COULDN'T get this right.
A skeptical reader can correctly object: "write a careful mutator and see."

This file does exactly that -- a genuinely best-effort, non-adversarial,
non-grammar structural editor, using a real XML parser (xml.etree, not
regex splicing) so it isn't sabotaged by its own implementation:

  careful_duplicate_limb  -- deep-copies a random <body> subtree, renames
    every body/joint/geom name inside the copy to stay globally unique
    (the exact invariant duplicate_element in expanded_validity_comparison
    broke on -- 'repeated name "seg2_g"', documented in this project's own
    render_real_robots.py run), and duplicates the matching <actuator>
    entry so it references the renamed joint instead of a stale name.
  careful_delete_segment  -- removes a random non-root <body> subtree AND
    explicitly removes any <actuator> entry referencing a joint that lived
    inside it, so nothing is left pointing at a deleted joint.

If this STILL falls short of the grammar's 100%, that's first-party
evidence the invariant is genuinely easy to miss even under careful,
non-adversarial effort -- a necessity claim, not just a convenience claim.
If it matches 100%, the honest conclusion is that THESE TWO specific
structural edits are replicable by hand with enough care, and the
grammar's advantage is "no per-operator re-verification required," not
"impossible otherwise" -- report whichever result actually comes out.

Companion script: inheritance_ablation_validity.py asks the complementary
question -- does the grammar's OWN validity guarantee depend on anuvrtti
(parameter inheritance) specifically, or does it come entirely from typing/
legality rules (it-tag gating, markers) regardless of inheritance?

Run:
    python3 -m experiments.counterfactual_structural_validity --n_trials 100
"""

from __future__ import annotations

import copy
import random
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import numpy as np

from experiments.m_search_loop import Genome, random_genome, mutate as grammar_mutate
from experiments.mh1_xml_baseline import _tree_of, _xml_loads_and_steps
from gmm.mjcf_compiler import MJCFCompiler


def _xml_loads_and_steps_diag(xml: str, settle_steps: int = 50) -> Tuple[bool, Optional[str]]:
    """Same check as mh1_xml_baseline._xml_loads_and_steps, but does NOT
    swallow the exception -- returns (ok, error_message) so a failure can
    be CLASSIFIED (e.g. 'mass and inertia ... mjMINVAL' vs a name
    collision) instead of just counted. _xml_loads_and_steps itself
    catches and discards the exception internally, which is fine for a
    pass/fail count but hides exactly the information this script needs."""
    import mujoco
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        ok = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        return ok, (None if ok else "non-finite state after settling")
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:70]}"


# ---------------------------------------------------------------------------
# Careful (non-adversarial) structural mutators, via a real XML parser
# ---------------------------------------------------------------------------

def _all_bodies(root: ET.Element) -> List[ET.Element]:
    return root.findall(".//body")


def _rename_subtree_unique(subtree: ET.Element, suffix: str) -> dict:
    """Rename every body/joint/geom 'name' attribute inside `subtree` by
    appending `suffix`, so nothing collides with the original tree.
    Returns {old_joint_name: new_joint_name} for actuator fix-up."""
    joint_renames = {}
    for el in subtree.iter():
        if el.tag in ("body", "joint", "geom") and "name" in el.attrib:
            old = el.attrib["name"]
            new = f"{old}{suffix}"
            el.attrib["name"] = new
            if el.tag == "joint":
                joint_renames[old] = new
    return joint_renames


def careful_duplicate_limb(xml: str, rng: random.Random) -> str:
    """Best-effort structural duplication: copy a random non-root <body>
    subtree, rename everything inside the copy to stay globally unique, and
    duplicate+fix-up its actuator entry if it had one. This is the
    'careful' counterpart to expanded_validity_comparison.mutate_duplicate_
    element, which did none of this and scored 0%."""
    root = ET.fromstring(xml)
    candidates = _all_bodies(root)
    if not candidates:
        return xml
    target = rng.choice(candidates)
    parent_map = {c: p for p in root.iter() for c in p}
    parent = parent_map.get(target)
    if parent is None:
        return xml

    clone = copy.deepcopy(target)
    suffix = f"_dup{rng.randrange(10**6)}"
    joint_renames = _rename_subtree_unique(clone, suffix)
    parent.append(clone)

    actuator = root.find(".//actuator")
    if actuator is not None and joint_renames:
        for act in list(actuator):
            old_joint = act.attrib.get("joint")
            if old_joint in joint_renames:
                new_act = copy.deepcopy(act)
                new_act.attrib["joint"] = joint_renames[old_joint]
                old_name = new_act.attrib.get("name", "act")
                new_act.attrib["name"] = f"{old_name}{suffix}"
                actuator.append(new_act)

    return ET.tostring(root, encoding="unicode")


def careful_delete_segment(xml: str, rng: random.Random) -> str:
    """Best-effort structural deletion: remove a random non-root <body>
    subtree AND explicitly remove any <actuator> entry whose joint lived
    inside it, so nothing references a now-missing joint. The 'careful'
    counterpart to mutate_delete_element, which only deleted the body and
    left dangling actuator references."""
    root = ET.fromstring(xml)
    worldbody = root.find(".//worldbody")
    candidates = _all_bodies(root)
    # exclude the torso (the single direct child of worldbody) -- deleting
    # the whole body would make an empty/degenerate robot by construction,
    # not a meaningful structural-edit comparison
    torso_names = {c.attrib.get("name") for c in worldbody} if worldbody is not None else set()
    candidates = [c for c in candidates if c.attrib.get("name") not in torso_names]
    if not candidates:
        return xml
    target = rng.choice(candidates)
    parent_map = {c: p for p in root.iter() for c in p}
    parent = parent_map.get(target)
    if parent is None:
        return xml

    removed_joints = {el.attrib["name"] for el in target.iter()
                      if el.tag == "joint" and "name" in el.attrib}
    parent.remove(target)

    actuator = root.find(".//actuator")
    if actuator is not None and removed_joints:
        for act in list(actuator):
            if act.attrib.get("joint") in removed_joints:
                actuator.remove(act)

    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------
# Benchmark: careful structural edits vs. the grammar, vs. the EXISTING
# naive versions of the same two edit types (for direct side-by-side)
# ---------------------------------------------------------------------------

def run(n_trials: int = 100, n_mutations: int = 3, seed: int = 0,
       verbose: bool = True) -> dict:
    from experiments.expanded_validity_comparison import (
        mutate_duplicate_element, mutate_delete_element)

    from collections import Counter

    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    methods = {
        "grammar_guided (ours)": None,  # handled specially below
        "careful_duplicate_limb": careful_duplicate_limb,
        "naive_duplicate_element (existing)": mutate_duplicate_element,
        "careful_delete_segment": careful_delete_segment,
        "naive_delete_element (existing)": mutate_delete_element,
    }
    valid_counts = {name: 0 for name in methods}
    error_types = {name: Counter() for name in methods}

    for t in range(n_trials):
        base_genome = random_genome(rng_np)

        # grammar-guided: mutate the GENOME, re-derive through the grammar
        try:
            genome = base_genome
            for _ in range(n_mutations):
                genome = grammar_mutate(genome, rng_np)
            tree = _tree_of(genome)
            xml = MJCFCompiler().compile(tree)
            valid_counts["grammar_guided (ours)"] += int(_xml_loads_and_steps(xml))
        except Exception:
            pass

        # XML-level mutators: start from the SAME base genome's compiled
        # XML, apply n_mutations edits of that one type, check validity
        base_xml = MJCFCompiler().compile(_tree_of(base_genome))
        for name, fn in methods.items():
            if fn is None:
                continue
            try:
                xml = base_xml
                for _ in range(n_mutations):
                    xml = fn(xml, rng_py)
                ok, err = _xml_loads_and_steps_diag(xml)
                if ok:
                    valid_counts[name] += 1
                else:
                    # classify the failure, not just count it -- this is
                    # what surfaced that careful_delete_segment's remaining
                    # failures are mjMINVAL mass/inertia errors, a
                    # DIFFERENT invariant than the name-collision bug
                    # careful_duplicate_limb fixed
                    error_types[name][err] += 1
            except Exception as e:
                # an error in the MUTATOR itself (e.g. malformed XML it
                # produced that ET or the mutator logic choked on), not a
                # MuJoCo load/step error -- classified separately
                error_types[name][f"mutator error: {type(e).__name__}: {str(e)[:60]}"] += 1

    if verbose:
        print(f"Counterfactual structural validity: careful vs. naive, "
             f"{n_trials} trials, {n_mutations} mutations/trial\n")
        ranked = sorted(valid_counts.items(), key=lambda kv: -kv[1])
        for name, count in ranked:
            marker = " <-- ours" if "ours" in name else ""
            print(f"  {name:<36}{count:>4}/{n_trials} "
                 f"({100*count/n_trials:.0f}%){marker}")
        print()
        print("Per-edit-type conclusion (NOT aggregated/maxed across edit types --")
        print("each edit type is judged against the grammar on its own):\n")
        ours = valid_counts["grammar_guided (ours)"]
        for careful_name in ("careful_duplicate_limb", "careful_delete_segment"):
            count = valid_counts[careful_name]
            if count >= ours:
                print(f"  {careful_name}: MATCHES the grammar ({count}/{n_trials}).")
                print(f"    -> for this edit type, the grammar's advantage is "
                     f"CONVENIENCE, not necessity --\n       a careful "
                     f"implementation can replicate it.")
            else:
                print(f"  {careful_name}: falls short of the grammar "
                     f"({count}/{n_trials} vs {ours}/{n_trials}).")
                print(f"    -> even a deliberate, non-adversarial attempt misses "
                     f"something the grammar\n       enforces automatically -- "
                     f"evidence toward necessity, not just convenience.")
                if error_types[careful_name]:
                    print(f"       failure causes: "
                         f"{dict(error_types[careful_name].most_common(3))}")
            print()
    return {"valid_counts": valid_counts,
           "error_types": {k: dict(v) for k, v in error_types.items()}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--n_mutations", type=int, default=3)
    args = ap.parse_args()
    run(n_trials=args.n_trials, n_mutations=args.n_mutations)

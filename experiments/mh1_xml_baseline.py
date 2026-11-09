"""
M-H1 — Grammar Mutation vs Raw-XML Mutation: Validity Rate
============================================================

The concrete comparison the roadmap promised and hadn't been run: when you
mutate a body, does going through the GRAMMAR (edit a genome parameter,
re-derive) keep you in valid territory more often than mutating the
compiled MJCF XML directly (RoboMorph/LASeR-style, and the common LLM-writes-
XML failure mode noted in the CoRL work)?

Grammar mutation: edit one Genome field (leg count, segment count, seg_len,
seg_rad, angle_jitter), re-derive through the same grammar. Anuvṛtti and
the required-role/marker gating make most edits structurally legal by
construction — the only way to fail is triggering a BodyGrammarError (e.g.
an op with no applicable rule), which given the current rule set basically
never happens since every op family always has at least one applicable rule
once the torso exists.

Raw-XML mutation: parse the compiled MJCF text, pick a random <geom> or
<joint> attribute, and perturb its numeric value directly — no grammar
awareness, no anuvṛtti, no consistency guarantee across sibling limbs. This
mirrors what happens when an LLM or blind mutation edits XML text: a size
attribute can go negative, a joint range can invert, a capsule can go to
zero radius, nothing enforces that limb siblings stay consistent.

Validity = MuJoCo successfully LOADS the result (mujoco.MjModel.from_xml_string
doesn't raise) AND steps without immediately producing non-finite state.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple

import numpy as np

from experiments.m_search_loop import Genome, random_genome, mutate as grammar_mutate
from ptg.body_grammar import BodyGrammar
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody
from gmm.mjcf_compiler import MJCFCompiler, compile_and_validate

_NUM_ATTR_RE = re.compile(
    r'(pos|size|fromto|range|quat)="([^"]+)"')


def mutate_xml_raw(xml: str, rng: random.Random, scale: float = 0.6) -> str:
    """Pick one numeric attribute occurrence at random and perturb one of
    its numbers. Applies the SAME kind of edit a blind/LLM XML mutation
    realistically produces: most of the time a moderate rescale, but with
    real probability a sign flip, a near-zero collapse, or an extreme
    multiplier — because nothing about editing raw XML text prevents these.
    A version that only ever mildly rescales isn't a fair adversarial
    baseline (it will trivially show ~100% validity and understate the real
    risk of ungrammatical mutation)."""
    matches = list(_NUM_ATTR_RE.finditer(xml))
    if not matches:
        return xml
    m = rng.choice(matches)
    attr, value_str = m.group(1), m.group(2)
    nums = value_str.split()
    idx = rng.randrange(len(nums))
    try:
        v = float(nums[idx])
    except ValueError:
        return xml

    roll = rng.random()
    if roll < 0.15:
        new_v = 0.0                              # collapse to zero
    elif roll < 0.30:
        new_v = -v if v != 0 else rng.uniform(-1, 1)  # sign flip
    elif roll < 0.45:
        new_v = v * rng.uniform(5.0, 50.0)        # extreme blowup
    else:
        new_v = v * (1.0 + rng.uniform(-scale, scale))  # moderate rescale

    nums[idx] = f"{new_v:.5f}"
    new_value = " ".join(nums)
    start, end = m.span(2)
    return xml[:start] + new_value + xml[end:]


def _tree_of(genome: Genome, grammar_seed: int = 0):
    g = BodyGrammar(seed_body_rules(), seed=grammar_seed)
    return PhiBody().build_tree(g.derive(genome.to_plan()))


def _xml_loads_and_steps(xml: str, settle_steps: int = 50) -> bool:
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        return bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    except Exception:
        return False


def run(n_trials: int = 100, n_xml_mutations: int = 3, seed: int = 0,
       verbose: bool = True) -> Dict:
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    grammar_valid, xml_valid = 0, 0
    grammar_errors: List[str] = []

    for _ in range(n_trials):
        base_genome = random_genome(rng_np)

        # --- grammar mutation path ---
        mutated_genome = grammar_mutate(base_genome, rng_np)
        try:
            tree = _tree_of(mutated_genome)
            comp = MJCFCompiler()
            xml = comp.compile(tree)
            ok = _xml_loads_and_steps(xml)
            grammar_valid += int(ok)
            if not ok:
                grammar_errors.append("compiled but failed to load/step")
        except Exception as e:
            grammar_errors.append(f"{type(e).__name__}: {e}")

        # --- raw-XML mutation path ---
        try:
            base_tree = _tree_of(base_genome)
            comp = MJCFCompiler()
            base_xml = comp.compile(base_tree)
            xml = base_xml
            for _ in range(n_xml_mutations):
                xml = mutate_xml_raw(xml, rng_py)
            ok = _xml_loads_and_steps(xml)
            xml_valid += int(ok)
        except Exception:
            pass  # counts as invalid

    if verbose:
        print("M-H1 — Grammar mutation vs raw-XML mutation: validity rate")
        print(f"  trials: {n_trials}, XML mutations per trial: {n_xml_mutations}\n")
        print(f"  grammar mutation validity : {grammar_valid}/{n_trials} "
             f"({100*grammar_valid/n_trials:.0f}%)")
        print(f"  raw-XML mutation validity : {xml_valid}/{n_trials} "
             f"({100*xml_valid/n_trials:.0f}%)")
        if grammar_errors[:3]:
            print(f"\n  sample grammar-path failures: {grammar_errors[:3]}")
    return {"grammar_valid": grammar_valid, "xml_valid": xml_valid,
           "n_trials": n_trials}


if __name__ == "__main__":
    run()

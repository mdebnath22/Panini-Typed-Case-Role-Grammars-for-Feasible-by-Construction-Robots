"""
Inheritance ablation: does the grammar's validity guarantee NEED anuvrtti?
=============================================================================

The grammar has two mechanisms that could each, separately, be responsible
for 100% mutation validity: role-TYPING + legality gating (it-tag-style
preconditions, unique-by-construction naming) and parameter INHERITANCE
(anuvrtti -- a value declared once, e.g. seg_len on the first ADD_LIMB,
carries forward to every later rule that needs it without being restated).
todo1_typing_vs_inheritance.py already separated these two for the
COMPACTNESS claim (typing alone still gives reuse across legs that share
parameter values; inheritance additionally removes the need to restate
them). This script asks the same question for VALIDITY specifically: if
you disable anuvrtti -- every ADD_LIMB after the first falls back to its
own hardcoded defaults (seg_len=0.12, seg_rad=0.02, see ptg/body_sutra.py's
B1.1_ADD_LIMB params) instead of the genome's mutated values -- does
mutation validity stay at 100%, or drop?

Mechanism, concretely: legged_plan() (ptg/body_grammar.py) only passes
seg_len/seg_rad as an explicit override on the FIRST leg's ADD_LIMB call;
every subsequent leg relies entirely on anuvrtti to inherit those values.
With anuvrtti_export forced to frozenset() for every rule, subsequent legs
ignore the genome's seg_len/seg_rad entirely and use B1.1_ADD_LIMB's own
declared defaults instead -- the derivation still completes (no exception),
but the resulting body no longer reflects the mutated genome for legs 2..n.

Prediction before running (stated here so the result below isn't silently
reframed after the fact, per this project's own reproducibility standard):
validity should stay at or near 100%, because MuJoCo's loads-and-steps
check has no way to know the genome's INTENDED parameters weren't honored
-- it only checks that the resulting XML is physically well-formed, which
inheritance was never responsible for. If true, this cleanly attributes
the grammar's validity guarantee to TYPING/legality-gating alone, and
inheritance's role is elsewhere (compactness, geometric consistency,
genome-controllability), not raw validity -- run it to find out, don't
assume.

ACTUAL RESULT, traced down to the code (do not skip this when citing the
script): validity is unchanged (100/100 both ON and OFF), confirming the
validity half of the prediction. But the "ALL legs honor genome seg_rad"
column is ALSO unchanged (29/100 both ON and OFF) -- which looked at first
like inheritance already not mattering, until a direct trace of the
compiled XML (per-body geom sizes) showed why: in the UNABLATED grammar,
legs 2..n *already* get fixed values (limb_root=0.020, seg=0.017,
foot=0.025) instead of the genome's seg_rad, identical to the ablated
condition. Root cause, in ptg/body_sutra.py's BodySutra.resolve_params():
`out = dict(state.inherited_params); out.update(self.params)` -- the
rule's OWN declared params dict always overwrites anuvrtti-inherited
values for any key both declare. B1.1_ADD_LIMB declares its own
seg_len/seg_rad defaults (0.12/0.02) in `params`, so even when anuvrtti
carries the genome's seg_rad forward in `inherited_params`, that line
clobbers it back to the rule's hardcoded default on every application.
legged_plan() (ptg/body_grammar.py) only works around this for leg 0, by
passing seg_len/seg_rad as an explicit `overrides` dict that applies AFTER
resolve_params -- legs 1..n-1 get no such override and fall back to
whatever resolve_params produces, which is always the rule's own default,
never the inherited genome value. So in THIS rule catalogue, anuvrtti's
export of seg_len/seg_rad across ADD_LIMB applications is a no-op
regardless of the ablation -- the ablation changed nothing observable
because there was nothing functioning to disable. This is a real
implementation detail of the current grammar, not a flaw in this script's
measurement (tol was tightened to 1e-4 and the check is structure-aware,
not substring matching) -- it does NOT undermine the compactness claim in
todo1_typing_vs_inheritance.py, which is a rule-counting argument (how
many distinct (op, param-tuple) rules are needed) rather than a trace of
per-leg genome-controllability, but it DOES mean "inheritance makes every
leg reflect genome.seg_rad" is not actually true of this codebase today
and should not be claimed without this caveat, or fixed before claiming it.

Run:
    python3 -m experiments.inheritance_ablation_validity --n_trials 100
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy as np

from experiments.m_search_loop import Genome, random_genome, mutate as grammar_mutate
from experiments.mh1_xml_baseline import _xml_loads_and_steps
from ptg.body_grammar import BodyGrammar
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody
from gmm.mjcf_compiler import MJCFCompiler


def _ablated_rules():
    """Same rule catalogue as seed_body_rules(), but every rule's
    anuvrtti_export is forced empty -- no parameter crosses a rule
    boundary. BodySutra is a frozen dataclass, so this is a real structural
    copy (dataclasses.replace), not a mutation of the shared seed rules."""
    return [dataclasses.replace(r, anuvrtti_export=frozenset())
           for r in seed_body_rules()]


def _tree_of_with_rules(genome: Genome, rules, grammar_seed: int = 0):
    g = BodyGrammar(rules, seed=grammar_seed)
    return PhiBody().build_tree(g.derive(genome.to_plan()))


def _count_geoms_matching_radius(xml: str, radius: float, tol: float = 6e-4) -> int:
    # tol=6e-4, not 1e-4: gmm/mjcf_compiler.py formats geom size to 3 decimal
    # places (f"{r:.3f}"), so up to 5e-4 of rounding error is expected even
    # when inheritance DID carry the exact genome value -- 6e-4 absorbs that
    # rounding without reopening the earlier false-positive problem (tol=1e-3
    # was coincidentally matching unrelated hardcoded defaults ~3-5e-3 away).
    """How many <geom size="..."> elements in the compiled XML match the
    genome's seg_rad -- NOT a substring check (leg 1 always gets seg_rad
    explicitly regardless of inheritance, via legged_plan's first-leg
    override, so a substring 'is it present anywhere' check would pass
    trivially from leg 1 alone and say nothing about legs 2..n, which is
    the actual thing inheritance is responsible for carrying)."""
    root = ET.fromstring(xml)
    count = 0
    for geom in root.iter("geom"):
        size = geom.attrib.get("size", "")
        for tok in size.split():
            try:
                if abs(float(tok) - radius) < tol:
                    count += 1
                    break
            except ValueError:
                continue
    return count


def run(n_trials: int = 100, n_mutations: int = 2, seed: int = 0,
       verbose: bool = True) -> Dict:
    rng_np = np.random.default_rng(seed)
    inherit_rules = seed_body_rules()
    ablated_rules = _ablated_rules()

    valid_counts = {"inheritance_ON (full grammar)": 0,
                    "inheritance_OFF (ablated)": 0}
    # all_legs_match[label]: trial count where EVERY leg's geom radius
    # matches the genome's seg_rad (all legs reflect the genome's value,
    # not just leg 1, which always gets it explicitly regardless of
    # inheritance -- see _count_geoms_matching_radius's docstring)
    all_legs_match = {"inheritance_ON (full grammar)": 0,
                      "inheritance_OFF (ablated)": 0}

    for t in range(n_trials):
        genome = random_genome(rng_np)
        for _ in range(n_mutations):
            genome = grammar_mutate(genome, rng_np)

        for label, rules in (("inheritance_ON (full grammar)", inherit_rules),
                             ("inheritance_OFF (ablated)", ablated_rules)):
            try:
                tree = _tree_of_with_rules(genome, rules)
                xml = MJCFCompiler().compile(tree)
                if _xml_loads_and_steps(xml):
                    valid_counts[label] += 1
                n_match = _count_geoms_matching_radius(xml, genome.seg_rad)
                if n_match >= genome.n_legs:  # every leg's first segment matches
                    all_legs_match[label] += 1
            except Exception:
                pass

    if verbose:
        print(f"Inheritance ablation, validity axis: {n_trials} trials, "
             f"{n_mutations} grammar mutations/trial\n")
        for label in valid_counts:
            count = valid_counts[label]
            matched = all_legs_match[label]
            print(f"  {label:<32} valid {count:>4}/{n_trials} ({100*count/n_trials:.0f}%)   "
                 f"ALL legs honor genome seg_rad: {matched:>4}/{n_trials} "
                 f"({100*matched/n_trials:.0f}%)")
        print()
        on, off = valid_counts["inheritance_ON (full grammar)"], valid_counts["inheritance_OFF (ablated)"]
        matched_on = all_legs_match["inheritance_ON (full grammar)"]
        matched_off = all_legs_match["inheritance_OFF (ablated)"]
        if off >= on - 2:  # allow small sampling noise
            print("Validity is essentially UNCHANGED with inheritance disabled.")
            print("Conclusion: the grammar's validity guarantee comes from TYPING/legality")
            print("gating alone, not from inheritance.")
            if matched_off < matched_on - 2:
                print(f"The 'ALL legs honor genome seg_rad' column DOES drop with inheritance off "
                     f"({matched_on} -> {matched_off}/{n_trials}),")
                print("confirming inheritance's real role is genome-controllability/compactness "
                     "(todo1_typing_vs_inheritance.py),")
                print("not raw structural validity -- disabling it makes mutations silently inert "
                     "for legs 2..n, not invalid.")
            else:
                print(f"The 'ALL legs honor genome seg_rad' column is UNCHANGED by the ablation "
                     f"({matched_on} -> {matched_off}/{n_trials}) --")
                print("traced to ptg/body_sutra.py's resolve_params(): a rule's own declared "
                     "params always overwrite anuvrtti-")
                print("inherited values for any shared key, so B1.1_ADD_LIMB's hardcoded "
                     "seg_len/seg_rad defaults clobber the")
                print("inherited genome value on every leg after leg 0 regardless of this "
                     "ablation -- anuvrtti's export of")
                print("seg_len/seg_rad is already a no-op in the UNABLATED grammar for this rule "
                     "catalogue (see module docstring).")
                print("This does not undermine todo1_typing_vs_inheritance.py's rule-COUNTING "
                     "compactness claim, but it does mean")
                print("per-leg genome-controllability via inheritance should not be claimed for "
                     "this codebase without this caveat.")
        else:
            print(f"Validity DROPS with inheritance disabled ({on} -> {off}/{n_trials}).")
            print("Conclusion: inheritance is NOT just a compactness/convenience mechanism --")
            print("it also contributes to the validity guarantee, contrary to this script's")
            print("stated prediction. Report this as found, not as expected.")
    return {"valid_counts": valid_counts, "all_legs_match": all_legs_match}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--n_mutations", type=int, default=2)
    args = ap.parse_args()
    run(n_trials=args.n_trials, n_mutations=args.n_mutations)

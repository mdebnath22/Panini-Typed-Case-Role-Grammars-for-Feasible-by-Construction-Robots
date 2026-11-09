from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Set
from .body_karaka import BodyFrame, BodyKaraka


@dataclass
class BodyState:
    markers: Set[str] = field(default_factory=set)
    inherited: BodyFrame = field(default_factory=BodyFrame)
    inherited_params: Dict[str, float] = field(default_factory=dict)
    limb_count: int = 0
    chain_depth: int = 0
    def copy(self):
        return BodyState(set(self.markers), BodyFrame(dict(self.inherited.bindings)),
                         dict(self.inherited_params), self.limb_count, self.chain_depth)


@dataclass(frozen=True)
class BodySutra:
    rule_id: str
    op: str
    required_roles: FrozenSet[BodyKaraka] = frozenset()
    condition: Optional[Callable] = None
    params: Dict[str, float] = field(default_factory=dict, hash=False)
    emits_markers: FrozenSet[str] = frozenset()
    consumes_markers: FrozenSet[str] = frozenset()
    blocks_markers: FrozenSet[str] = frozenset()
    anuvrtti_export: FrozenSet[str] = frozenset()
    nivrtti: FrozenSet[str] = frozenset()
    vibhasa: bool = False
    specificity: int = 0
    # Constraint-solving hook (experiments/closed_loop_grammar.py): every
    # other rule's resolved parameters are a pure static merge of its own
    # declared defaults, anuvrtti-inherited values, and an explicit
    # per-step override -- no rule can COMPUTE a parameter from others at
    # derive time. A closed kinematic chain needs exactly that: the
    # rocker's mount angle is not a free parameter at all, it is
    # determined by solving the loop-closure equation against the
    # already-resolved crank/link lengths. resolve_fn, if given, is
    # called as resolve_fn(resolved_params_dict, state) -> dict AFTER the
    # static merge (so it sees inherited values and overrides), and its
    # return value replaces `resolved` entirely -- the one place this
    # grammar lets a rule's parameters depend on a constraint over prior
    # state rather than only on inheritance or an explicit override.
    # Every existing rule leaves this as None and is completely unaffected.
    resolve_fn: Optional[Callable] = field(default=None, hash=False, compare=False)

    def applicable(self, state, frame):
        merged = frame.merge_inherited(state.inherited)
        if not self.required_roles.issubset(merged.roles()): return False
        if not self.consumes_markers.issubset(state.markers): return False
        if self.blocks_markers & state.markers: return False
        if self.condition is not None and not self.condition(state, merged): return False
        return True

    def resolve_params(self, state):
        # FIX (paper Sec. 4.2.1 / Appendix B): inherited values must win over
        # a rule's OWN declared default for the same key, not the other way
        # around. The previous order (`out = dict(inherited); out.update(
        # self.params)`) let self.params unconditionally clobber an
        # anuvrtti-inherited value whenever the rule itself also declares a
        # default for that key -- making inheritance of seg_len/seg_rad
        # across ADD_LIMB applications a no-op regardless of whether
        # anuvrtti was enabled, since B1.1_ADD_LIMB declares its own
        # defaults for exactly the keys it also exports. Starting from the
        # rule's own defaults and then overlaying inherited_params (which
        # only contains keys a prior rule actually exported) means a rule's
        # default is used only for keys inheritance has not supplied --
        # the behavior anuvrtti_export was supposed to produce. An explicit
        # per-step override (see apply(), applied after this) still takes
        # final precedence either way, so leg 0's explicit seg_len/seg_rad
        # override in legged_plan() is unaffected by this change.
        out = dict(self.params); out.update(state.inherited_params); return out

    def apply(self, state, frame, overrides=None):
        merged = frame.merge_inherited(state.inherited)
        resolved = self.resolve_params(state)
        if overrides: resolved.update(overrides)
        if self.resolve_fn is not None:
            resolved = self.resolve_fn(resolved, state)
        ns = state.copy()
        ns.markers |= set(self.emits_markers); ns.markers -= set(self.nivrtti)
        for k in self.anuvrtti_export:
            if k in resolved: ns.inherited_params[k] = resolved[k]
        for k in self.nivrtti: ns.inherited_params.pop(k, None)
        ns.inherited = merged
        if self.op == "ADD_LIMB": ns.limb_count += 1; ns.chain_depth = 1
        elif self.op in ("EXTEND", "ARTICULATE"): ns.chain_depth += 1
        elif self.op == "TERMINATE": ns.chain_depth = 0
        return AppliedBodyRule(self, merged, resolved, ns)


@dataclass
class AppliedBodyRule:
    sutra: BodySutra
    frame: BodyFrame
    params: Dict[str, float]
    next_state: BodyState


def seed_body_rules() -> List[BodySutra]:
    return [
        BodySutra("B0.1_ADD_TORSO", "ADD_TORSO", frozenset({BodyKaraka.MULA}),
                 params={"torso_len": 0.30, "torso_rad": 0.08, "torso_mass": 2.0,
                         "torso_height": 0.30},
                 emits_markers=frozenset({"has_torso"}),
                 blocks_markers=frozenset({"has_torso"}),
                 anuvrtti_export=frozenset({"torso_height"}), specificity=1),
        BodySutra("B1.1_ADD_LIMB", "ADD_LIMB", frozenset({BodyKaraka.ANGA}),
                 consumes_markers=frozenset({"has_torso"}),
                 params={"seg_len": 0.12, "seg_rad": 0.02, "seg_mass": 0.15,
                         "mount_angle": 0.0, "joint_type": 0.0, "joint_range": 0.9},
                 emits_markers=frozenset({"has_link", "open_chain"}),
                 anuvrtti_export=frozenset({"seg_len", "seg_rad", "seg_mass",
                                            "joint_range"}), specificity=1),
        BodySutra("B1.2_ADD_LIMB_REACH", "ADD_LIMB", frozenset({BodyKaraka.ANGA}),
                 consumes_markers=frozenset({"has_torso", "reach_task"}),
                 params={"seg_len": 0.22, "seg_rad": 0.025, "seg_mass": 0.20,
                         "mount_angle": 0.0, "joint_type": 2.0, "joint_range": 1.4},
                 emits_markers=frozenset({"has_link", "open_chain"}),
                 anuvrtti_export=frozenset({"seg_len", "seg_rad", "seg_mass",
                                            "joint_range"}), specificity=2),
        BodySutra("B2.1_EXTEND", "EXTEND", frozenset({BodyKaraka.DANDA}),
                 consumes_markers=frozenset({"has_link", "open_chain"}),
                 params={"taper": 0.85},
                 anuvrtti_export=frozenset({"taper"}), specificity=1),
        BodySutra("B3.1_ARTICULATE", "ARTICULATE", frozenset({BodyKaraka.SANDHI}),
                 consumes_markers=frozenset({"has_link", "open_chain"}),
                 params={"joint_type": 0.0, "joint_range": 0.9, "actuated": 1.0},
                 anuvrtti_export=frozenset({"joint_range"}), specificity=1),
        BodySutra("B4.1_TERMINATE", "TERMINATE", frozenset({BodyKaraka.PADA}),
                 consumes_markers=frozenset({"open_chain"}),
                 params={"foot_rad": 0.025, "foot_mass": 0.05},
                 nivrtti=frozenset({"open_chain"}), specificity=1),
        BodySutra("B5.1_SYMMETRIZE", "SYMMETRIZE", frozenset({BodyKaraka.ANGA}),
                 consumes_markers=frozenset({"has_torso"}),
                 params={"mirror_axis": 1.0}, vibhasa=True, specificity=1),
    ]

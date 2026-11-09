"""
Sūtra-Style Production Rules
============================

Each rule follows the format:

    rule_id : condition (inherited via anuvṛtti) -> transformation
              [it-tags] [vibhāṣā]

Mechanisms modeled directly on the Aṣṭādhyāyī:

  * anuvṛtti  — context inheritance: unspecified kāraka slots and constraint
                parameters carry forward from earlier rules in the derivation
                until explicitly overridden (nivṛtti).
  * it-tags   — kit/pit-style silent indicatory markers attached to a rule's
                output. They are never executed; they gate the applicability
                of *downstream* rules (e.g. a GRASP emits the `k_it` marker
                "holding", which licenses LIFT and blocks a second GRASP).
  * vibhāṣā   — optionality: a rule marked vibhāṣā admits multiple valid
                derivations; the derivation engine (and later Φ) branches or
                samples among them rather than firing deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Set

from .karaka import Karaka, KarakaFrame


@dataclass
class TaskState:
    """Symbolic derivation state threaded through rule application."""

    markers: Set[str] = field(default_factory=set)      # active it-tag markers
    inherited: KarakaFrame = field(default_factory=KarakaFrame)
    inherited_params: Dict[str, float] = field(default_factory=dict)

    def copy(self) -> "TaskState":
        return TaskState(
            markers=set(self.markers),
            inherited=KarakaFrame(dict(self.inherited.bindings)),
            inherited_params=dict(self.inherited_params),
        )


@dataclass(frozen=True)
class Sutra:
    """A single production rule.

    condition        : predicate over (TaskState, KarakaFrame) — may consult
                       markers set by upstream it-tags and inherited context.
    required_roles   : kāraka slots that must be filled (after anuvṛtti merge).
    transformation   : symbolic operation name consumed by Φ (phi_seed / phi_neural).
    params           : rule-local constraint parameters (e.g. approach_height).
                       Missing params are resolved through anuvṛtti.
    emits_markers    : it-tags added to state after firing.
    consumes_markers : it-tags required present (gating) — silent downstream gating.
    blocks_markers   : it-tags that must be absent.
    anuvrtti_export  : which of this rule's params/roles flow forward.
    nivrtti          : markers/params explicitly *cancelled* by this rule
                       (anuvṛtti termination).
    vibhasa          : optional-rule flag — derivation may branch here.
    specificity      : count of conditions; used by the paribhāṣā layer
                       (apavāda/utsarga: more specific overrides general).
    """

    rule_id: str
    transformation: str
    required_roles: FrozenSet[Karaka] = frozenset()
    condition: Optional[Callable[[TaskState, KarakaFrame], bool]] = None
    params: Dict[str, float] = field(default_factory=dict, hash=False)
    emits_markers: FrozenSet[str] = frozenset()
    consumes_markers: FrozenSet[str] = frozenset()
    blocks_markers: FrozenSet[str] = frozenset()
    anuvrtti_export: FrozenSet[str] = frozenset()
    nivrtti: FrozenSet[str] = frozenset()
    vibhasa: bool = False
    specificity: int = 0

    def applicable(self, state: TaskState, frame: KarakaFrame) -> bool:
        merged = frame.merge_inherited(state.inherited)
        if not self.required_roles.issubset(merged.roles()):
            return False
        if not self.consumes_markers.issubset(state.markers):
            return False
        if self.blocks_markers & state.markers:
            return False
        if self.condition is not None and not self.condition(state, merged):
            return False
        return True

    def resolve_params(self, state: TaskState) -> Dict[str, float]:
        """Rule-local params override anuvṛtti-inherited ones."""
        out = dict(state.inherited_params)
        out.update(self.params)
        return out

    def apply(self, state: TaskState, frame: KarakaFrame,
              overrides: Optional[Dict[str, float]] = None) -> "AppliedRule":
        merged = frame.merge_inherited(state.inherited)
        resolved = self.resolve_params(state)
        if overrides:
            resolved.update(overrides)  # explicit step-level statement wins
        new_state = state.copy()
        new_state.markers |= set(self.emits_markers)
        new_state.markers -= set(self.nivrtti)
        # anuvṛtti: export designated params and the merged frame forward
        for key in self.anuvrtti_export:
            if key in resolved:
                new_state.inherited_params[key] = resolved[key]
        for key in self.nivrtti:
            new_state.inherited_params.pop(key, None)
        new_state.inherited = merged
        return AppliedRule(sutra=self, frame=merged, params=resolved,
                           next_state=new_state)


@dataclass
class AppliedRule:
    """A rule instantiated against a concrete frame — the unit Φ consumes."""

    sutra: Sutra
    frame: KarakaFrame
    params: Dict[str, float]
    next_state: TaskState

    def __repr__(self):
        return f"<{self.sutra.rule_id} | {self.frame} | {self.params}>"


# ---------------------------------------------------------------------------
# Seed rule set (5 core action types + apavāda exceptions)
# ---------------------------------------------------------------------------

def seed_rules() -> List[Sutra]:
    """Core manipulation sūtras. Naming loosely follows adhyāya.pāda.sūtra."""

    return [
        # 1.1.1 utsarga: approach the karma from above at inherited/declared height
        Sutra(
            rule_id="1.1.1_APPROACH",
            transformation="APPROACH",
            required_roles=frozenset({Karaka.KARTA, Karaka.KARMA}),
            params={"approach_height": 0.10},
            emits_markers=frozenset({"aligned"}),
            blocks_markers=frozenset({"holding"}),
            anuvrtti_export=frozenset({"approach_height"}),
            specificity=2,
        ),
        # 1.1.2 apavāda: approach for *tools* (karaṇa present) uses side approach
        Sutra(
            rule_id="1.1.2_APPROACH_TOOL",
            transformation="APPROACH_SIDE",
            required_roles=frozenset({Karaka.KARTA, Karaka.KARMA, Karaka.KARANA}),
            params={"approach_height": 0.05, "lateral_offset": 0.08},
            emits_markers=frozenset({"aligned"}),
            blocks_markers=frozenset({"holding"}),
            anuvrtti_export=frozenset({"approach_height"}),
            specificity=3,  # more specific -> paribhāṣā prefers it over 1.1.1
        ),
        # 2.1.1 grasp: requires alignment marker (silent gating via it-tag)
        Sutra(
            rule_id="2.1.1_GRASP",
            transformation="GRASP",
            required_roles=frozenset({Karaka.KARTA, Karaka.KARMA}),
            params={"grasp_depth": 0.02},
            consumes_markers=frozenset({"aligned"}),
            emits_markers=frozenset({"holding"}),
            blocks_markers=frozenset({"holding"}),
            nivrtti=frozenset({"aligned"}),
            specificity=2,
        ),
        # 3.1.1 lift: gated on holding
        Sutra(
            rule_id="3.1.1_LIFT",
            transformation="LIFT",
            required_roles=frozenset({Karaka.KARTA, Karaka.KARMA}),
            params={"lift_height": 0.15},
            consumes_markers=frozenset({"holding"}),
            anuvrtti_export=frozenset({"lift_height"}),
            specificity=2,
        ),
        # 4.1.1 transport toward sampradāna (recipient/goal)
        Sutra(
            rule_id="4.1.1_TRANSPORT",
            transformation="TRANSPORT",
            required_roles=frozenset({Karaka.KARTA, Karaka.KARMA,
                                      Karaka.SAMPRADANA}),
            consumes_markers=frozenset({"holding"}),
            specificity=3,
        ),
        # 5.1.1 place onto adhikaraṇa; vibhāṣā — release height optional/multimodal
        Sutra(
            rule_id="5.1.1_PLACE",
            transformation="PLACE",
            required_roles=frozenset({Karaka.KARTA, Karaka.KARMA,
                                      Karaka.ADHIKARANA}),
            params={"release_height": 0.02},
            consumes_markers=frozenset({"holding"}),
            nivrtti=frozenset({"holding"}),
            vibhasa=True,
            specificity=3,
        ),
        # 6.1.1 rotate tool about karaṇa axis (e.g. screwdriver, valve)
        Sutra(
            rule_id="6.1.1_ROTATE_TOOL",
            transformation="ROTATE_TOOL",
            required_roles=frozenset({Karaka.KARTA, Karaka.KARANA}),
            params={"rotation_angle": 1.5708},
            consumes_markers=frozenset({"holding"}),
            specificity=2,
        ),
        # 7.1.1 retract to source (apādāna)
        Sutra(
            rule_id="7.1.1_RETRACT",
            transformation="RETRACT",
            required_roles=frozenset({Karaka.KARTA, Karaka.APADANA}),
            specificity=2,
        ),
    ]

"""Pāṇinian Task Grammar engine: derivations, resolving conflicts via
paribhāṣā, threading anuvṛtti/it-tag state."""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Sequence

from .karaka import KarakaFrame
from .paribhasa import ParibhasaResolver
from .sutra import AppliedRule, Sutra, TaskState


class DerivationError(RuntimeError):
    pass


class PTG:
    def __init__(self, rules: Sequence[Sutra],
                 resolver: Optional[ParibhasaResolver] = None,
                 seed: Optional[int] = None):
        self.rules = list(rules)
        self.resolver = resolver or ParibhasaResolver(self.rules)
        self.rng = random.Random(seed)

    def applicable(self, state: TaskState, frame: KarakaFrame) -> List[Sutra]:
        return [s for s in self.rules if s.applicable(state, frame)]

    def derive(self, plan: Sequence[Dict], *, sample_vibhasa: bool = False,
              log: Optional[list] = None) -> List[AppliedRule]:
        """If `log` is passed, appends {'candidates','state_markers','frame',
        'chosen'} per step — used by the box-embedding experiment to build
        (state, chosen-rule) supervision without touching the base engine."""
        state = TaskState()
        derivation: List[AppliedRule] = []
        for step in plan:
            goal = step["goal"]
            frame = step.get("frame", KarakaFrame())
            cands = [s for s in self.applicable(state, frame)
                     if s.transformation.startswith(goal)]
            if not cands:
                raise DerivationError(f"No applicable sūtra for goal '{goal}' "
                                      f"(markers={state.markers})")
            chosen = self.resolver.pick(cands, sample_vibhasa=sample_vibhasa)
            if chosen.vibhasa and sample_vibhasa:
                ranked = self.resolver.resolve(cands, sample_vibhasa=True)
                chosen = self.rng.choice(ranked[: min(2, len(ranked))])
            if log is not None:
                merged = frame.merge_inherited(state.inherited)
                log.append({"candidates": list(cands),
                           "state_markers": set(state.markers),
                           "frame": merged, "chosen": chosen})
            applied = chosen.apply(state, frame, overrides=step.get("params"))
            derivation.append(applied)
            state = applied.next_state
        return derivation

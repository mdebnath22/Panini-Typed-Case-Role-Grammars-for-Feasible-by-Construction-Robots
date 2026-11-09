from __future__ import annotations
from typing import List, Sequence


class ParibhasaResolver:
    def __init__(self, catalogue: Sequence):
        self._order = {s.rule_id: i for i, s in enumerate(catalogue)}
    def resolve(self, candidates, *, sample_vibhasa=False):
        if not candidates: return []
        def key(s):
            nitya = 0 if (s.vibhasa and not sample_vibhasa) else 1
            return (s.specificity, nitya, self._order.get(s.rule_id, -1))
        return sorted(candidates, key=key, reverse=True)
    def pick(self, candidates, **kw):
        ranked = self.resolve(candidates, **kw)
        if not ranked: raise ValueError("paribhāṣā: no applicable rule")
        return ranked[0]

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import numpy as np


class Karaka(Enum):
    KARTA = "karta"; KARMA = "karma"; KARANA = "karana"
    ADHIKARANA = "adhikarana"; APADANA = "apadana"; SAMPRADANA = "sampradana"


@dataclass
class Entity:
    name: str
    pose: np.ndarray
    attrs: Dict[str, float] = field(default_factory=dict)
    def __post_init__(self): self.pose = np.asarray(self.pose, dtype=np.float64)
    @property
    def position(self): return self.pose[:3, 3]


@dataclass
class KarakaFrame:
    bindings: Dict[Karaka, Entity] = field(default_factory=dict)
    def bind(self, role, ent):
        new = dict(self.bindings); new[role] = ent; return KarakaFrame(new)
    def get(self, role): return self.bindings.get(role)
    def merge_inherited(self, inherited):
        merged = dict(inherited.bindings); merged.update(self.bindings)
        return KarakaFrame(merged)
    def roles(self): return set(self.bindings.keys())

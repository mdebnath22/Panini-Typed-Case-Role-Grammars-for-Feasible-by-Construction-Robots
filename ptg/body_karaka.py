from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import numpy as np


class BodyKaraka(Enum):
    MULA = "mula"; ANGA = "anga"; DANDA = "danda"
    SANDHI = "sandhi"; PADA = "pada"; PRERAKA = "preraka"


@dataclass
class BodyEntity:
    name: str
    pose: np.ndarray
    attrs: Dict[str, float] = field(default_factory=dict)
    def __post_init__(self): self.pose = np.asarray(self.pose, dtype=np.float64)
    @property
    def position(self): return self.pose[:3, 3]
    @property
    def rotation(self): return self.pose[:3, :3]


@dataclass
class BodyFrame:
    bindings: Dict[BodyKaraka, BodyEntity] = field(default_factory=dict)
    def bind(self, role, ent):
        new = dict(self.bindings); new[role] = ent; return BodyFrame(new)
    def get(self, role): return self.bindings.get(role)
    def merge_inherited(self, inherited):
        merged = dict(inherited.bindings); merged.update(self.bindings)
        return BodyFrame(merged)
    def roles(self): return set(self.bindings.keys())


@dataclass
class TreeNode:
    node_id: str
    kind: str
    local: np.ndarray
    geom: Dict[str, float] = field(default_factory=dict)
    joint: Optional[Dict] = None
    world: Optional[np.ndarray] = None
    children: list = field(default_factory=list)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

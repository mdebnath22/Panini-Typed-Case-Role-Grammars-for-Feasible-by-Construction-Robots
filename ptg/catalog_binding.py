"""Binds a derived (abstract) body tree to real catalog parts.

The grammar (ptg/body_grammar.py, ptg/body_sutra.py) and the realization
map (gmm/phi_body.py) are left completely unmodified -- this module is a
separate post-processing pass over the TreeNode produced by PhiBody.build_tree.
It is additive and idempotent: a tree this pass has not touched compiles and
simulates exactly as before (gmm/mjcf_compiler.py, sulba/feasibility.py);
a tree this pass HAS touched gets physically grounded masses (catalog
material density x actual geometric volume, in place of the grammar's
heuristic seg_mass/foot_mass/torso_mass parameters), a chosen actuator per
actuated joint with its real torque-speed curve, a chosen battery and sensor
suite on the torso, and a bill of materials.

Role heuristic for part selection (overridable via the `role_fn` argument):
  - The first segment out from the torso on a limb (chain_depth == 1) is
    "torso-adjacent": it sees the largest bending moment from the limb's own
    weight plus whatever the limb carries, so it gets the heavier actuator
    and structural stock class.
  - Every later segment on the same limb is "distal": lighter actuator,
    lighter stock.
  - A limb flagged `balance_critical=True` (pass via `balance_critical_limbs`,
    a set of limb root node_ids) uses the high-peak-torque quasi-direct-drive
    actuator at its torso-adjacent joint instead of the standard servo class,
    for limbs expected to do impact absorption (e.g. parkour landing legs).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set
import math

from . import catalog as cat


@dataclass
class BOMLine:
    part_id: str
    category: str   # actuator | battery | sensor | structural
    count: int
    unit_cost: float
    lead_time_days: int

    @property
    def line_cost(self) -> float:
        return self.unit_cost * self.count


@dataclass
class BillOfMaterials:
    lines: List[BOMLine] = field(default_factory=list)

    def add(self, part_id: str, category: str, unit_cost: float, lead_time_days: int, qty: int = 1):
        for ln in self.lines:
            if ln.part_id == part_id and ln.category == category:
                ln.count += qty
                return
        self.lines.append(BOMLine(part_id, category, qty, unit_cost, lead_time_days))

    @property
    def total_cost(self) -> float:
        return sum(ln.line_cost for ln in self.lines)

    @property
    def lead_time_days(self) -> int:
        """Parts are assumed orderable in parallel; the BOM's lead time is
        the slowest single part, not the sum."""
        return max((ln.lead_time_days for ln in self.lines), default=0)

    def as_dict(self) -> Dict:
        return {
            "parts": [
                {"part_id": ln.part_id, "category": ln.category, "count": ln.count,
                 "unit_cost": ln.unit_cost, "line_cost": ln.line_cost,
                 "lead_time_days": ln.lead_time_days}
                for ln in self.lines
            ],
            "total_cost": self.total_cost,
            "lead_time_days": self.lead_time_days,
        }


def _default_role_fn(node, chain_depth_of: Dict[str, int],
                     balance_critical_limbs: Set[str]) -> str:
    depth = chain_depth_of.get(node.node_id, 1)
    limb_root = chain_depth_of.get(("_limb_root", node.node_id), None)
    if limb_root in balance_critical_limbs and depth <= 1:
        return "balance_critical"
    return "torso_adjacent" if depth <= 1 else "distal"


def _tube_mass(length: float, radius: float, stock: "cat.StructuralStock") -> float:
    """Mass of a hollow tube of the given outer radius and length, using the
    catalog part's material density and its own wall-thickness/outer_radius
    ratio scaled to this part's actual as-built radius (so a thin grammar
    segment doesn't inherit the catalog sample's absolute wall thickness,
    just its proportional structural design)."""
    if stock.outer_radius <= 0:
        return 0.0
    wall_ratio = stock.wall_thickness / stock.outer_radius
    wall = radius * wall_ratio
    inner = max(0.0, radius - wall)
    area = math.pi * (radius ** 2 - inner ** 2)
    return area * length * stock.density


def _tube_second_moment(radius: float, stock: "cat.StructuralStock") -> float:
    wall_ratio = stock.wall_thickness / stock.outer_radius if stock.outer_radius > 0 else 0.1
    wall = radius * wall_ratio
    inner = max(0.0, radius - wall)
    return (math.pi / 4.0) * (radius ** 4 - inner ** 4)


def bind_catalog(tree, *, structural_light="al6061_tube_light",
                 structural_heavy="al6061_tube_heavy",
                 actuator_distal="servo_small", actuator_proximal="servo_medium",
                 actuator_balance="qdd_balance",
                 battery_id="lipo_4s_4000", sensor_ids=("imu_6dof",),
                 balance_critical_limbs: Optional[Set[str]] = None) -> BillOfMaterials:
    """Mutates `tree` in place: every link/foot gets a real mass derived
    from a chosen structural-stock part; every actuated joint gets a chosen
    actuator with its torque-speed curve attached; the torso gets a battery
    and sensor suite. Returns the resulting bill of materials.

    `tree` is the TreeNode produced by gmm.phi_body.PhiBody.build_tree --
    this function does not care how it was derived."""
    balance_critical_limbs = balance_critical_limbs or set()
    bom = BillOfMaterials()

    # Determine chain depth per node by walking from each limb root; a
    # "link" node directly under the torso is depth 1, the next is depth 2,
    # etc. We also tag which limb root each node descends from so the
    # balance-critical-limb override can be applied.
    depth_of: Dict[str, int] = {}
    limb_of: Dict[str, str] = {}

    def _walk_limb(node, depth, limb_root_id):
        depth_of[node.node_id] = depth
        limb_of[node.node_id] = limb_root_id
        nd = depth + 1 if node.kind in ("link",) else depth
        for c in node.children:
            _walk_limb(c, nd, limb_root_id)

    for child in tree.children:
        if child.kind in ("link", "mirror"):
            _walk_limb(child, 1, child.node_id)

    # Torso: battery + sensors.
    battery = cat.BATTERIES[battery_id]
    sensor_mass = 0.0
    for sid in sensor_ids:
        s = cat.SENSORS[sid]
        sensor_mass += s.mass
        bom.add(sid, "sensor", s.cost, s.lead_time_days)
    bom.add(battery_id, "battery", battery.cost, battery.lead_time_days)
    # Chassis shell mass stays whatever the torso geom already specifies
    # (grammar torso_mass is a reasonable proxy for the printed/machined
    # shell + mounting hardware); battery and sensors are added on top as
    # real, catalog-sourced payload mass.
    tree.geom["mass"] = tree.geom.get("mass", 2.0) + battery.mass + sensor_mass
    tree.geom["battery_id"] = battery_id
    tree.geom["sensor_ids"] = list(sensor_ids)

    for node in tree.walk():
        if node.node_id == tree.node_id:
            continue
        depth = depth_of.get(node.node_id, 1)
        limb_root = limb_of.get(node.node_id)
        role = ("balance_critical" if limb_root in balance_critical_limbs and depth <= 1
               else ("torso_adjacent" if depth <= 1 else "distal"))

        if node.kind == "link":
            stock_id = structural_heavy if role in ("torso_adjacent", "balance_critical") else structural_light
            stock = cat.STRUCTURAL_STOCK[stock_id]
            length = node.geom.get("length", 0.12)
            radius = node.geom.get("radius", 0.02)
            real_mass = _tube_mass(length, radius, stock)
            node.geom["mass"] = max(real_mass, 0.001)
            node.geom["part_id"] = stock_id
            node.geom["yield_strength"] = stock.yield_strength
            node.geom["elastic_modulus"] = stock.elastic_modulus
            node.geom["second_moment"] = _tube_second_moment(radius, stock)
            bom.add(stock_id, "structural", stock.cost_per_m, stock.lead_time_days, qty=0)
            # structural stock is bought by length, not by count; track
            # total length consumed as a fractional "count" in meters.
            for ln in bom.lines:
                if ln.part_id == stock_id and ln.category == "structural":
                    ln.count += length  # interpreted as meters consumed
                    break

        if node.kind == "foot":
            stock = cat.STRUCTURAL_STOCK["printed_petg"]
            r = node.geom.get("radius", 0.025)
            vol = (4.0 / 3.0) * math.pi * r ** 3 * 0.4  # hollow printed shell, ~40% effective fill
            node.geom["mass"] = max(vol * stock.density, 0.005)
            node.geom["part_id"] = "printed_petg"

        if node.joint and node.joint.get("actuated"):
            if role == "balance_critical":
                act_id = actuator_balance
            elif role == "torso_adjacent":
                act_id = actuator_proximal
            else:
                act_id = actuator_distal
            act = cat.ACTUATORS[act_id]
            node.joint["actuator_id"] = act_id
            node.joint["stall_torque"] = act.stall_torque
            node.joint["rated_torque"] = act.rated_torque
            node.joint["no_load_speed"] = act.no_load_speed
            node.joint["rated_current"] = act.rated_current
            node.joint["voltage"] = act.voltage
            # Actuator mass is a lumped point mass at the joint; added onto
            # this node's own geom mass as the simplest consistent place to
            # carry it through to MJCF (it moves with this body in MuJoCo).
            node.geom["mass"] = node.geom.get("mass", 0.0) + act.mass
            bom.add(act_id, "actuator", act.cost, act.lead_time_days)

    return bom

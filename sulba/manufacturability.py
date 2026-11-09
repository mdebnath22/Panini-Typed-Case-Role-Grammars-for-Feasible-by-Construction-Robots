"""Manufacturability checks: printing, off-the-shelf tolerances, assembly
clearance. This was on the original roadmap ("a manufacturability check
(printing, tolerances, assembly) closes the loop") alongside the
catalog/BOM/engineering-gate work, but got dropped from the first pass --
it's added here as its own module rather than folded into
sulba.engineering_feasibility, since it checks a different kind of thing:
not "will this part survive its load" but "can this part actually be
produced and put together as specified."

Requires a tree that has been through ptg.catalog_binding.bind_catalog,
same as sulba.engineering_feasibility -- these checks are meaningless
against the grammar's raw abstract parameters.

Three checks:
  1. Printed-part manufacturability: minimum wall thickness and minimum
     feature radius a standard FDM nozzle can reliably produce.
  2. Off-the-shelf stock match: how far a segment's as-built radius sits
     from the nearest standard tube/extrusion outer diameter for its
     material -- a large mismatch means a custom-drawn or custom-machined
     part (real cost/lead-time consequences, not captured by the BOM's
     flat per-meter stock price) rather than a cut-to-length commodity
     part.
  3. Assembly clearance: do any two actuator housings (approximated as
     cylinders at each actuated joint's pivot) collide given their real
     physical envelope -- a check the geometric gate's min_tip_separation
     does NOT cover, since that check only looks at foot/link tip
     positions, not at the bulk of an actuator housing mounted mid-chain.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from . import feasibility as _geo  # unused directly; kept for a shared style of report


@dataclass
class ManufacturabilitySpec:
    nozzle_diameter: float = 0.0004        # m, standard 0.4mm FDM nozzle
    min_wall_multiples_of_nozzle: float = 2.0  # at least 2 perimeters
    min_printable_feature_radius: float = 0.0015  # m, smallest reliably printed feature
    stock_match_tolerance: float = 0.15    # fractional deviation from nearest standard OD
                                           # before flagging "needs custom stock"
    min_actuator_clearance: float = 0.004  # m, minimum gap required between two actuator
                                           # housing envelopes once mounted


@dataclass
class ManufacturabilityReport:
    ok: bool
    checks: Dict[str, bool]
    margins: Dict[str, float]
    details: Dict[str, Dict] = field(default_factory=dict)


def check_printed_parts(tree, spec: Optional[ManufacturabilitySpec] = None) -> ManufacturabilityReport:
    """Every node bound to 'printed_petg' (currently: feet, see
    ptg.catalog_binding) must have a wall/feature size an FDM printer can
    actually resolve. Feet are modeled as a hollow printed shell in
    catalog_binding; we check the same assumed shell thickness here rather
    than re-deriving it, since that's the manufacturing-relevant quantity."""
    spec = spec or ManufacturabilitySpec()
    min_wall_required = spec.nozzle_diameter * spec.min_wall_multiples_of_nozzle
    worst_margin = float("inf")
    details = {}
    for n in tree.walk():
        if n.geom.get("part_id") != "printed_petg":
            continue
        r = n.geom.get("radius", 0.0)
        if r <= 0:
            continue
        # catalog_binding assumes ~40% effective shell fraction of the
        # radius for a printed foot; approximate the resulting wall as that
        # same fraction of the radius, the manufacturing-relevant thickness.
        assumed_wall = 0.40 * r
        feature_margin = r - spec.min_printable_feature_radius
        wall_margin = assumed_wall - min_wall_required
        m = min(feature_margin, wall_margin)
        details[n.node_id] = {"radius": r, "assumed_wall": assumed_wall,
                              "min_wall_required": min_wall_required,
                              "min_feature_radius": spec.min_printable_feature_radius}
        worst_margin = min(worst_margin, m)
    if worst_margin == float("inf"):
        worst_margin = 0.0
    checks = {"printed_part_manufacturable": worst_margin >= 0}
    margins = {"printed_part_manufacturable": worst_margin}
    return ManufacturabilityReport(ok=checks["printed_part_manufacturable"], checks=checks,
                                   margins=margins, details=details)


def check_stock_match(tree, spec: Optional[ManufacturabilitySpec] = None) -> ManufacturabilityReport:
    """For every structural-stock link (aluminum/carbon-fiber tube), how
    far is its as-built outer radius from the nearest standard
    off-the-shelf diameter for that material? A body whose grammar-chosen
    radius doesn't land near any stock size would need custom-drawn tube
    for every single segment -- buildable, but not with a BOM line that's
    just 'cost_per_m x meters', and with a lead time the catalog's flat
    per-part lead_time_days doesn't reflect."""
    from ptg import catalog as cat
    spec = spec or ManufacturabilitySpec()
    worst_frac = 0.0
    n_custom = 0
    n_checked = 0
    details = {}
    for n in tree.walk():
        part_id = n.geom.get("part_id")
        if part_id is None or part_id == "printed_petg" or part_id not in cat.STRUCTURAL_STOCK:
            continue
        stock = cat.STRUCTURAL_STOCK[part_id]
        standard_ods = cat.STANDARD_OD_BY_MATERIAL.get(stock.material)
        if not standard_ods:
            continue
        r = n.geom.get("radius", 0.0)
        n_checked += 1
        nearest = min(standard_ods, key=lambda od: abs(od / 2.0 - r))
        frac_off = abs((nearest / 2.0) - r) / max(nearest / 2.0, 1e-9)
        is_custom = frac_off > spec.stock_match_tolerance
        if is_custom:
            n_custom += 1
        worst_frac = max(worst_frac, frac_off)
        details[n.node_id] = {"as_built_radius": r, "nearest_standard_od_m": nearest,
                              "fractional_deviation": frac_off, "needs_custom_stock": is_custom}
    checks = {"stock_match": worst_frac <= spec.stock_match_tolerance}
    margins = {"stock_match_worst_fractional_deviation": spec.stock_match_tolerance - worst_frac}
    details["_summary"] = {"n_segments_checked": n_checked, "n_needing_custom_stock": n_custom}
    return ManufacturabilityReport(ok=checks["stock_match"], checks=checks, margins=margins,
                                   details=details)


def check_assembly_clearance(tree, spec: Optional[ManufacturabilitySpec] = None) -> ManufacturabilityReport:
    """Pairwise clearance between every actuated joint's housing envelope
    (approximated as a sphere of radius = the bound actuator's
    housing_radius, centered at the joint's world pivot) and every OTHER
    actuated joint's housing envelope on the body. This is distinct from
    sulba.feasibility's min_tip_separation (which only looks at foot/link
    TIPS) and from sulba.engineering_feasibility (which checks load, not
    physical fit) -- a design can pass both of those and still have two
    actuator housings that physically overlap once real parts, not point
    tips, are placed at each joint."""
    from ptg import catalog as cat
    spec = spec or ManufacturabilitySpec()
    joints = []
    for n in tree.walk():
        if n.joint and n.joint.get("actuated") and n.joint.get("actuator_id"):
            act = cat.ACTUATORS[n.joint["actuator_id"]]
            joints.append((n.node_id, n.world[:3, 3], act.housing_radius))

    worst_margin = float("inf")
    worst_pair = None
    for i in range(len(joints)):
        for j in range(i + 1, len(joints)):
            id_i, pos_i, r_i = joints[i]
            id_j, pos_j, r_j = joints[j]
            dist = float(np.linalg.norm(pos_i - pos_j))
            required = r_i + r_j + spec.min_actuator_clearance
            margin = dist - required
            if margin < worst_margin:
                worst_margin = margin
                worst_pair = (id_i, id_j, dist, required)
    if worst_margin == float("inf"):
        worst_margin = 0.0
    checks = {"assembly_clearance": worst_margin >= 0}
    margins = {"assembly_clearance": worst_margin}
    details = {"worst_pair": worst_pair, "n_actuated_joints": len(joints)}
    return ManufacturabilityReport(ok=checks["assembly_clearance"], checks=checks, margins=margins,
                                   details=details)


def check_manufacturability(tree, spec: Optional[ManufacturabilitySpec] = None) -> ManufacturabilityReport:
    spec = spec or ManufacturabilitySpec()
    reports = [check_printed_parts(tree, spec), check_stock_match(tree, spec),
              check_assembly_clearance(tree, spec)]
    checks, margins, details = {}, {}, {}
    for r in reports:
        checks.update(r.checks)
        margins.update(r.margins)
        details.update(r.details)
    return ManufacturabilityReport(ok=all(checks.values()), checks=checks, margins=margins,
                                   details=details)

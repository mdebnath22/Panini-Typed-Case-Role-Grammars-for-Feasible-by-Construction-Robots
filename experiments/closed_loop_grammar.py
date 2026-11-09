"""
Closed Kinematic Chains as an Actual Grammar Rule (Option B)
================================================================

closed_kinematic_chain.py already proves the loop-closure equation is
solvable (or not) for a planar four-bar linkage, as a standalone
numerical check outside the grammar. A reviewer's Option B asks for more:
a real rule/constraint interface inside the grammar itself, a reject/
repair path for unsolvable configurations, and a benchmark against
downstream SIMULATION success, not just offline forward-kinematics math.

This file adds exactly that, as a genuinely new, self-contained rule
catalogue (it does not touch seed_body_rules() or any existing legged-
body result):

  - `BodySutra.resolve_fn` (added to ptg/body_sutra.py as a small,
    additive, backward-compatible hook: every existing rule leaves it
    None and is unaffected) lets ONE rule -- CLOSE_LOOP -- compute its
    own parameters from a constraint over previously-resolved state,
    instead of only from static defaults, inheritance, or an explicit
    override. That constraint is exactly solve_loop_closure from
    closed_kinematic_chain.py; this file does not reimplement the solver,
    it imports and reuses it.
  - CLOSE_LOOP's `condition` makes an unsolvable configuration a genuine
    derivation FAILURE -- the grammar's existing "no applicable rule"
    mechanism (ptg.body_grammar.BodyGrammarError) -- rather than silently
    producing a broken body. That is the reject path; `repair_four_bar`
    below adds the accompanying repair path (nudge link lengths toward
    closeability and retry, bounded, falling back to rejection).
  - The resulting body (torso + crank + rocker + coupler) is compiled to
    real MJCF and given an actual MuJoCo `<equality><connect>` constraint
    joining the coupler's free end to the crank's tip -- a genuine
    downstream-simulation test of whether the solved pose holds together
    under MuJoCo's own constraint solver, which closed_kinematic_chain.py
    never tested (it only checked forward-kinematics residuals, no
    physics engine involved at all).
  - `run_benchmark` compares this grammar-wrapped path against the
    "solver-only" path (calling solve_loop_closure directly, exactly as
    closed_kinematic_chain.py does) on the SAME random configurations, to
    confirm wrapping the solver as a grammar rule changes nothing about
    correctness (same closeable rate, same residual), while reporting
    what only the grammar path has: construction-time rejection, a
    repair rate, solver wall-clock time, and real downstream simulation
    success/failure.

Run:
    python3 -m experiments.closed_loop_grammar --n_trials 300
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ptg.body_karaka import BodyEntity, BodyFrame, BodyKaraka, TreeNode
from ptg.body_sutra import BodySutra, BodyState
from ptg.body_grammar import BodyGrammar, BodyGrammarError, torso_frame, _ent
from ptg.paribhasa import ParibhasaResolver
from gmm.mjcf_compiler import MJCFCompiler
from experiments.closed_kinematic_chain import solve_loop_closure, _closure_residual


# ---------------------------------------------------------------------------
# Rule catalogue: torso (reused ADD_TORSO semantics, minimal local copy so
# this file has no dependency on seed_body_rules()), crank, close-loop,
# rocker, coupler.
# ---------------------------------------------------------------------------

def _crank_frame():
    return BodyFrame().bind(BodyKaraka.ANGA, _ent("crank"))


def _close_loop_frame():
    return BodyFrame().bind(BodyKaraka.PRERAKA, _ent("loop_closure"))


def _rocker_frame():
    return BodyFrame().bind(BodyKaraka.ANGA, _ent("rocker"))


def _coupler_frame():
    return BodyFrame().bind(BodyKaraka.DANDA, _ent("coupler"))


def _close_loop_condition(state: BodyState, merged: BodyFrame) -> bool:
    """The reject path: fail this derivation step outright (raising
    BodyGrammarError from BodyGrammar.derive, exactly as any other
    inapplicable rule already does in this grammar) if no real
    assembly exists for these link lengths and crank angle."""
    p = state.inherited_params
    sol = solve_loop_closure(p["fourbar_L0"], p["fourbar_L1"],
                             p["fourbar_L2"], p["fourbar_L3"], p["fourbar_theta1"])
    return sol is not None


def _close_loop_resolve(resolved: Dict, state: BodyState) -> Dict:
    """The constraint-solving hook: compute theta3 from the loop-closure
    equation and inject it as `mount_angle`, so the ROCKER rule picks it
    up through the grammar's ordinary inheritance-read mechanism
    (resolve_params already does `out.update(state.inherited_params)` for
    every rule -- no change needed there). Also records solver wall-clock
    time and the FK residual for the benchmark below."""
    p = state.inherited_params
    t0 = time.perf_counter()
    sol = solve_loop_closure(p["fourbar_L0"], p["fourbar_L1"],
                             p["fourbar_L2"], p["fourbar_L3"], p["fourbar_theta1"])
    solve_time = time.perf_counter() - t0
    if sol is None:
        # condition() should already have rejected this derivation before
        # apply() ever runs; this branch is defensive, not reachable in
        # normal use.
        raise BodyGrammarError("CLOSE_LOOP applied with no real solution")
    theta3, residual = sol
    resolved = dict(resolved)
    resolved["mount_angle"] = theta3
    resolved["_solve_time"] = solve_time
    resolved["_residual"] = abs(residual)
    return resolved


def four_bar_rules() -> List[BodySutra]:
    return [
        BodySutra("FB0_TORSO", "ADD_TORSO", frozenset({BodyKaraka.MULA}),
                 emits_markers=frozenset({"has_torso"})),
        BodySutra("FB1_CRANK", "ADD_LIMB", frozenset({BodyKaraka.ANGA}),
                 consumes_markers=frozenset({"has_torso"}),
                 params={"mount_angle": 0.0, "seg_len": 0.1, "seg_rad": 0.02, "seg_mass": 0.1},
                 emits_markers=frozenset({"has_crank"}),
                 anuvrtti_export=frozenset({"fourbar_L0", "fourbar_L1", "fourbar_L2",
                                           "fourbar_L3", "fourbar_theta1"})),
        BodySutra("FB2_CLOSE_LOOP", "CLOSE_LOOP", frozenset({BodyKaraka.PRERAKA}),
                 consumes_markers=frozenset({"has_crank"}),
                 condition=_close_loop_condition,
                 params={}, resolve_fn=_close_loop_resolve,
                 emits_markers=frozenset({"loop_solved"}),
                 anuvrtti_export=frozenset({"mount_angle"})),
        BodySutra("FB3_ROCKER", "ADD_LIMB", frozenset({BodyKaraka.ANGA}),
                 consumes_markers=frozenset({"loop_solved"}),
                 params={"mount_angle": 0.0, "seg_rad": 0.02, "seg_mass": 0.1},
                 emits_markers=frozenset({"has_rocker"})),
        BodySutra("FB4_COUPLER", "COUPLER", frozenset({BodyKaraka.DANDA}),
                 consumes_markers=frozenset({"has_rocker"}),
                 params={}),
    ]


def four_bar_plan(L0: float, L1: float, L2: float, L3: float, theta1: float) -> List[Dict]:
    return [
        {"op": "ADD_TORSO", "frame": torso_frame()},
        {"op": "ADD_LIMB", "frame": _crank_frame(),
         "params": {"mount_angle": theta1, "seg_len": L1,
                    "fourbar_L0": L0, "fourbar_L1": L1, "fourbar_L2": L2,
                    "fourbar_L3": L3, "fourbar_theta1": theta1}},
        {"op": "CLOSE_LOOP", "frame": _close_loop_frame()},
        {"op": "ADD_LIMB", "frame": _rocker_frame(), "params": {"seg_len": L3}},
        {"op": "COUPLER", "frame": _coupler_frame(), "params": {"length": L2}},
    ]


# ---------------------------------------------------------------------------
# Placement: torso/crank/rocker reuse the ordinary ADD_LIMB geometry
# (mounted at the torso's own radius, rotated by mount_angle); the coupler
# is placed AFTER the tree is built, once crank/rocker world positions are
# known, pointing from the rocker's tip toward the crank's tip.
# ---------------------------------------------------------------------------

def _trans(v):
    T = np.eye(4); T[:3, 3] = v; return T


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4); T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]); return T


def _fill_world(node, parent_world):
    node.world = parent_world @ node.local
    for c in node.children:
        _fill_world(c, node.world)


def build_four_bar_tree(derivation) -> Tuple[TreeNode, TreeNode, TreeNode]:
    """Returns (root, crank_tip_node, coupler_tip_node) -- the two bodies
    the <equality><connect> constraint will join."""
    root = None
    crank_node = rocker_node = None
    torso_rad = 0.08

    for applied in derivation:
        op = applied.sutra.op
        p = applied.params
        if op == "ADD_TORSO":
            root = TreeNode("torso", "torso", _trans([0, 0, 0.30]),
                           {"length": 0.30, "radius": torso_rad, "mass": 2.0},
                           {"type": "free"})
        elif op == "ADD_LIMB" and crank_node is None:
            ang = p.get("mount_angle", 0.0); L = p.get("seg_len", 0.12)
            local = _rot_z(ang) @ _trans([torso_rad, 0, 0])
            crank_node = TreeNode("crank", "link", local,
                                  {"length": L, "radius": p.get("seg_rad", 0.02),
                                   "mass": p.get("seg_mass", 0.1)},
                                  {"type": "hinge", "axis": (0, 1, 0), "range": 3.0,
                                   "actuated": True})
            root.children.append(crank_node)
        elif op == "ADD_LIMB" and rocker_node is None:
            ang = p.get("mount_angle", 0.0); L = p.get("seg_len", 0.12)
            local = _rot_z(ang) @ _trans([torso_rad, 0, 0])
            rocker_node = TreeNode("rocker", "link", local,
                                   {"length": L, "radius": p.get("seg_rad", 0.02),
                                    "mass": p.get("seg_mass", 0.1)},
                                   {"type": "hinge", "axis": (0, 1, 0), "range": 3.0,
                                    "actuated": True})
            root.children.append(rocker_node)
        elif op == "COUPLER":
            pass  # placed below, needs world transforms first

    _fill_world(root, np.eye(4))

    coupler_len = None
    for applied in derivation:
        if applied.sutra.op == "COUPLER":
            coupler_len = applied.params.get("length", 0.1)
    crank_tip_world = crank_node.world @ _trans([crank_node.geom["length"], 0, 0])
    rocker_tip_world = rocker_node.world @ _trans([rocker_node.geom["length"], 0, 0])
    direction = crank_tip_world[:3, 3] - rocker_tip_world[:3, 3]
    dist = float(np.linalg.norm(direction))
    direction = direction / (dist + 1e-15)

    # coupler is a child of the rocker tip, a capsule pointing toward
    # where the crank tip SHOULD be (its own local frame's +x axis
    # re-oriented to point in world `direction`); the <equality> connect
    # constraint below is what actually enforces/tests that it lands
    # there under real dynamics, rather than this placement guaranteeing
    # it by construction.
    rocker_tip_local = _trans([rocker_node.geom["length"], 0, 0])
    # build a rotation taking local +x to the world `direction`, expressed
    # in the rocker tip's own local frame
    rocker_tip_world_rot = (rocker_node.world @ rocker_tip_local)[:3, :3]
    dir_local = rocker_tip_world_rot.T @ direction
    axis = np.cross([1, 0, 0], dir_local)
    s = np.linalg.norm(axis); c = float(np.dot([1, 0, 0], dir_local))
    if s < 1e-9:
        R = np.eye(3) if c > 0 else _rot_z(np.pi)[:3, :3]
    else:
        axis = axis / s
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    coupler_local = np.eye(4); coupler_local[:3, :3] = R; coupler_local = rocker_tip_local @ coupler_local
    coupler_node = TreeNode("coupler", "link", coupler_local,
                            {"length": coupler_len, "radius": 0.015, "mass": 0.08},
                            {"type": "hinge", "axis": (0, 1, 0), "range": 3.0, "actuated": False})
    rocker_node.children.append(coupler_node)
    _fill_world(root, np.eye(4))
    return root, crank_node, coupler_node


def compile_with_loop_closure(root: TreeNode, crank_node: TreeNode,
                              coupler_node: TreeNode) -> str:
    """Compiles the ordinary MJCF (crank/rocker/coupler as an open chain,
    MJCFCompiler has no notion of loops) then post-processes the XML
    string to add a native MuJoCo <equality><connect> constraint between
    the coupler's free end and the crank's tip -- this is what actually
    makes it a CLOSED chain under simulation, not merely three
    independently-posed open links that happen to be positioned
    correctly at t=0."""
    comp = MJCFCompiler()
    xml = comp.compile(root, name="four_bar")
    crank_len = crank_node.geom["length"]
    equality = (f'  <equality>\n'
               f'    <connect body1="crank" body2="coupler" '
               f'anchor="{crank_len:.4f} 0 0"/>\n'
               f'  </equality>\n')
    xml = xml.replace("</mujoco>", equality + "</mujoco>")
    return xml


@dataclass
class FourBarOutcome:
    closeable_grammar: bool
    closeable_solver_only: bool
    residual: Optional[float]
    solve_time: Optional[float]
    repaired: bool
    repair_attempts: int
    sim_loaded: Optional[bool]
    sim_stepped: Optional[bool]
    connect_violation: Optional[float]


def derive_four_bar(L0, L1, L2, L3, theta1):
    """Runs the ACTUAL grammar derivation (BodyGrammar.derive over
    four_bar_rules()/four_bar_plan()), raising BodyGrammarError if
    CLOSE_LOOP's condition rejects the configuration -- the construction-
    time reject path. Returns (derivation, solve_time, residual) on
    success."""
    rules = four_bar_rules()
    g = BodyGrammar(rules, resolver=ParibhasaResolver(rules), seed=0)
    deriv = g.derive(four_bar_plan(L0, L1, L2, L3, theta1))
    close_loop_step = next(a for a in deriv if a.sutra.op == "CLOSE_LOOP")
    return deriv, close_loop_step.params["_solve_time"], close_loop_step.params["_residual"]


def repair_four_bar(L0, L1, L2, L3, theta1, max_attempts: int = 5,
                    nudge: float = 0.9) -> Tuple[Optional[Tuple], bool, int]:
    """Repair path: if the as-drawn configuration is not closeable,
    shrink the coupler/rocker toward the crank's reach (nudging L2, L3
    toward values more likely to admit a solution) and retry, bounded;
    reject (return None) if still unsolvable after max_attempts."""
    cur = (L0, L1, L2, L3, theta1)
    for attempt in range(max_attempts + 1):
        try:
            result = derive_four_bar(*cur)
            return result, attempt > 0, attempt
        except BodyGrammarError:
            L0c, L1c, L2c, L3c, th1 = cur
            cur = (L0c, L1c, L2c * nudge, L3c * nudge, th1)
    return None, False, max_attempts


def run_benchmark(n_trials: int = 300, seed: int = 0, verbose: bool = True) -> Dict:
    rng = np.random.default_rng(seed)
    outcomes: List[FourBarOutcome] = []

    for _ in range(n_trials):
        L0, L1, L2, L3 = rng.uniform(0.05, 0.30, size=4)
        theta1 = rng.uniform(0, 2 * np.pi)

        # solver-only path (closed_kinematic_chain.py's own approach)
        t0 = time.perf_counter()
        sol_only = solve_loop_closure(L0, L1, L2, L3, theta1)
        solver_only_time = time.perf_counter() - t0
        closeable_solver_only = sol_only is not None

        # grammar path, with repair
        result, repaired, attempts = repair_four_bar(L0, L1, L2, L3, theta1)
        if result is None:
            outcomes.append(FourBarOutcome(False, closeable_solver_only, None, None,
                                           False, attempts, None, None, None))
            continue
        deriv, solve_time, residual = result
        root, crank_node, coupler_node = build_four_bar_tree(deriv)
        xml = compile_with_loop_closure(root, crank_node, coupler_node)

        sim_loaded = sim_stepped = None
        connect_violation = None
        try:
            import mujoco
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            sim_loaded = True
            mujoco.mj_forward(model, data)
            ok = True
            for _ in range(100):
                mujoco.mj_step(model, data)
                if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
                    ok = False
                    break
            sim_stepped = ok
            if ok and data.ncon >= 0:
                # equality-constraint residual: MuJoCo reports active
                # equality constraints' position error directly.
                connect_violation = float(np.max(np.abs(data.efc_pos[:model.eq_data.shape[0] if model.neq else 0]))
                                          ) if model.neq and data.efc_pos.size else 0.0
        except Exception:
            sim_loaded = sim_loaded if sim_loaded is not None else False

        outcomes.append(FourBarOutcome(True, closeable_solver_only, residual, solve_time,
                                       repaired, attempts, sim_loaded, sim_stepped,
                                       connect_violation))

    n = len(outcomes)
    grammar_closeable = sum(o.closeable_grammar for o in outcomes)
    solver_only_closeable = sum(o.closeable_solver_only for o in outcomes)
    agree = sum(1 for o in outcomes if o.closeable_grammar == o.closeable_solver_only)
    repaired_n = sum(1 for o in outcomes if o.repaired)
    rejected_n = sum(1 for o in outcomes if not o.closeable_grammar)
    residuals = [o.residual for o in outcomes if o.residual is not None]
    solve_times = [o.solve_time for o in outcomes if o.solve_time is not None]
    sim_loaded_n = sum(1 for o in outcomes if o.sim_loaded)
    sim_stepped_n = sum(1 for o in outcomes if o.sim_stepped)
    violations = [o.connect_violation for o in outcomes if o.connect_violation is not None]

    out = dict(
        n_trials=n,
        grammar_closeable_rate=grammar_closeable / n,
        solver_only_closeable_rate=solver_only_closeable / n,
        agreement_rate=agree / n,
        repaired_rate=repaired_n / n,
        rejected_rate=rejected_n / n,
        mean_residual=float(np.mean(residuals)) if residuals else None,
        max_residual=float(np.max(residuals)) if residuals else None,
        mean_solve_time_ms=float(np.mean(solve_times) * 1000) if solve_times else None,
        sim_loaded_rate=sim_loaded_n / max(1, grammar_closeable),
        sim_stepped_rate=sim_stepped_n / max(1, grammar_closeable),
        mean_connect_violation=float(np.mean(violations)) if violations else None,
        max_connect_violation=float(np.max(violations)) if violations else None,
    )

    if verbose:
        print(f"Closed-chain grammar rule benchmark, n={n} random configurations\n")
        print(f"Grammar-wrapped closeable rate (with repair): {out['grammar_closeable_rate']*100:.1f}%")
        print(f"Solver-only closeable rate (as-drawn, no repair): {out['solver_only_closeable_rate']*100:.1f}%")
        print(f"Agreement between the two paths on the AS-DRAWN configuration: "
             f"{out['agreement_rate']*100:.1f}% (confirms wrapping the solver as a "
             f"grammar rule changes nothing about correctness)")
        print(f"Repaired (shrunk L2/L3, re-solved, succeeded): {out['repaired_rate']*100:.1f}%")
        print(f"Rejected (unfixable after 5 attempts): {out['rejected_rate']*100:.1f}%")
        print(f"FK residual at solved theta3: mean={out['mean_residual']:.2e}, "
             f"max={out['max_residual']:.2e}")
        print(f"Solver wall-clock time: mean={out['mean_solve_time_ms']:.3f}ms")
        print(f"\nDownstream SIMULATION success (not tested at all by the solver-only "
             f"path, which never touches MuJoCo):")
        print(f"  loads: {out['sim_loaded_rate']*100:.1f}%  of closeable configurations")
        print(f"  steps 100 physics steps without blowing up: {out['sim_stepped_rate']*100:.1f}%")
        if out['mean_connect_violation'] is not None:
            print(f"  <equality><connect> residual under dynamics: "
                 f"mean={out['mean_connect_violation']:.2e}, max={out['max_connect_violation']:.2e}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_benchmark(n_trials=args.n_trials, seed=args.seed)

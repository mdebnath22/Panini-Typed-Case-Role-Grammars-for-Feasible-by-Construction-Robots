from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

_MJ_JOINT = {"hinge": "hinge", "slide": "slide", "ball": "ball", "free": "free"}


def _mat_to_quat(R) -> Tuple[float, float, float, float]:
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0); w = 0.25 / s
        x = (R[2,1]-R[1,2])*s; y = (R[0,2]-R[2,0])*s; z = (R[1,0]-R[0,1])*s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = 2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
            w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif i == 1:
            s = 2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
            w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s = 2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
            w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    q = np.array([w,x,y,z]); return tuple(q/(np.linalg.norm(q)+1e-12))


def _clamp(v, lo, hi): return float(max(lo, min(hi, v)))


class MJCFCompiler:
    def __init__(self):
        self.actuators: List[str] = []
        self._joint_nodes: Dict[str, Dict] = {}

    def _body_xml(self, node, indent=3) -> str:
        pad = "  "*indent
        pos = node.local[:3, 3]; quat = _mat_to_quat(node.local[:3,:3])
        lines = [f'{pad}<body name="{node.node_id}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
                f'quat="{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}">']
        if node.joint and node.joint.get("type") == "free":
            # A body with no joint tag is rigidly welded to its parent in MuJoCo;
            # a body that is meant to move freely under gravity needs an explicit
            # freejoint. This was previously omitted, so every "free" torso was
            # in fact welded to the world.
            lines.append(f'{pad}  <freejoint name="{node.node_id}_free"/>')
        elif node.joint:
            jt = _MJ_JOINT.get(node.joint["type"], "hinge")
            axis = node.joint.get("axis", (0,1,0))
            rng = _clamp(node.joint.get("range", 0.9), 0.1, 3.0)
            if jt in ("hinge", "slide"):
                # damping + armature: with a real freejoint now letting the body
                # fall and the limbs swing under position-actuator torque, the
                # explicit integrator went unstable on undamped, zero-armature
                # hinges (NaN/huge QACC within ~0.1s on roughly half of sampled
                # bodies). Both are standard stabilizers for actuated hinges and
                # do not change which designs are geometrically feasible.
                lines.append(f'{pad}  <joint name="{node.node_id}_j" type="{jt}" '
                            f'axis="{axis[0]} {axis[1]} {axis[2]}" range="{-rng:.3f} {rng:.3f}" '
                            f'damping="0.3" armature="0.01"/>')
                if node.joint.get("actuated"):
                    aname = f"{node.node_id}_j"
                    self.actuators.append(aname)
                    self._joint_nodes[aname] = node.joint
            elif jt == "ball":
                lines.append(f'{pad}  <joint name="{node.node_id}_j" type="ball"/>')
        if node.kind == "torso":
            L=_clamp(node.geom.get("length",0.30),0.05,1.0); r=_clamp(node.geom.get("radius",0.08),0.02,0.3)
            m=_clamp(node.geom.get("mass",2.0),0.1,20.0)
            lines.append(f'{pad}  <geom name="{node.node_id}_g" type="capsule" '
                        f'fromto="{-L/2:.3f} 0 0 {L/2:.3f} 0 0" size="{r:.3f}" mass="{m:.3f}"/>')
        elif node.kind == "link":
            L=_clamp(node.geom.get("length",0.12),0.02,0.6); r=_clamp(node.geom.get("radius",0.02),0.005,0.1)
            m=_clamp(node.geom.get("mass",0.15),0.01,3.0)
            lines.append(f'{pad}  <geom name="{node.node_id}_g" type="capsule" '
                        f'fromto="0 0 0 {L:.3f} 0 0" size="{r:.3f}" mass="{m:.3f}"/>')
        elif node.kind == "foot":
            r=_clamp(node.geom.get("radius",0.025),0.005,0.08); m=_clamp(node.geom.get("mass",0.05),0.01,1.0)
            lines.append(f'{pad}  <geom name="{node.node_id}_g" type="sphere" size="{r:.3f}" mass="{m:.3f}"/>')
        for c in node.children:
            lines.append(self._body_xml(c, indent+1))
        lines.append(f'{pad}</body>')
        return "\n".join(lines)

    def compile(self, tree, *, name="panini_body") -> str:
        self.actuators = []
        self._joint_nodes = {}
        body_block = self._body_xml(tree, indent=3)
        act_lines = []
        for a in self.actuators:
            jnode = self._joint_nodes.get(a)
            # If the tree was passed through ptg.catalog_binding.bind_catalog,
            # each actuated joint carries its real catalog torque/current
            # limits; use them to set MuJoCo's forcerange so the actuator
            # cannot exert more torque in simulation than the physical part
            # could deliver. Otherwise fall back to the original fixed
            # position-servo gains (kp=18, kv=1.5), unchanged from before.
            if jnode is not None and jnode.get("stall_torque") is not None:
                torque = float(jnode["stall_torque"])
                act_lines.append(
                    f'    <position name="{a}_act" joint="{a}" kp="18" kv="1.5" '
                    f'ctrlrange="-1 1" forcerange="{-torque:.3f} {torque:.3f}"/>'
                )
            else:
                act_lines.append(
                    f'    <position name="{a}_act" joint="{a}" kp="18" '
                    f'kv="1.5" ctrlrange="-1 1"/>'
                )
        act_block = "\n".join(act_lines)
        return f"""<mujoco model="{name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default><geom rgba="0.7 0.6 0.5 1" friction="1 0.1 0.1"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>
    <light pos="0 0 3"/>
{body_block}
  </worldbody>
  <actuator>
{act_block}
  </actuator>
</mujoco>
"""


def compile_and_validate(tree, *, name="panini_body", settle_steps=200) -> Dict:
    comp = MJCFCompiler()
    xml = comp.compile(tree, name=name)
    report = {"ok": False, "loaded": False, "stepped": False,
             "n_actuators": len(comp.actuators), "error": None, "xml": xml}
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_string(xml)
        report["loaded"] = True
        data = mujoco.MjData(model)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        if np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)):
            report["stepped"] = True
        report["ok"] = report["loaded"] and report["stepped"]
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
    return report


def compile_with_bom(tree, *, name="panini_body", bind_if_unbound=True, **bind_kwargs) -> Dict:
    """Compile `tree` to MJCF and return it alongside a bill of materials.

    If `tree` has not already been passed through
    ptg.catalog_binding.bind_catalog (detected by the torso node missing a
    'battery_id' key), this binds it first using default part choices
    (override via **bind_kwargs, forwarded to bind_catalog). If the tree was
    already bound, pass bind_if_unbound=False with no effect either way
    (binding is idempotent only for the *parts chosen*, not for mass: calling
    it twice double-adds battery/sensor/actuator mass, so don't rebind a tree
    you've already bound).
    """
    from ptg.catalog_binding import bind_catalog, BillOfMaterials
    bom = None
    if bind_if_unbound and "battery_id" not in tree.geom:
        bom = bind_catalog(tree, **bind_kwargs)
    comp = MJCFCompiler()
    xml = comp.compile(tree, name=name)
    return {
        "xml": xml,
        "bom": bom.as_dict() if bom is not None else None,
        "n_actuators": len(comp.actuators),
    }


_URDF_JOINT_TYPE = {"hinge": "revolute", "slide": "prismatic", "ball": "spherical"}


def compile_urdf(tree, *, name="panini_body") -> str:
    """Best-effort URDF export of the same TreeNode the MJCF compiler reads.
    MJCF (compiled above) is this project's primary, validated format --
    every number in the paper was produced through it. This export is for
    GPU-side pipelines (e.g. Isaac Lab) that prefer URDF/USD ingestion; it
    has NOT been round-tripped through a physics engine the way the MJCF
    path has, so treat a freshly exported URDF as a starting point to
    validate in your own importer, not as pre-verified.

    The torso-to-world joint is exported as URDF's `floating` joint type,
    which not every consumer implements identically (some import pipelines,
    including some Isaac Sim versions, want a fixed base plus a 6-DOF
    articulation root configured separately) -- check this against whichever
    importer you're using before trusting six-DOF free-body behavior out of
    the box.
    """
    links = []
    joints = []

    def _geom_xml(node) -> str:
        if node.kind == "torso":
            L = node.geom.get("length", 0.30); r = node.geom.get("radius", 0.08)
            return f'<cylinder radius="{r:.4f}" length="{L:.4f}"/>'
        if node.kind == "link":
            L = node.geom.get("length", 0.12); r = node.geom.get("radius", 0.02)
            return f'<cylinder radius="{r:.4f}" length="{L:.4f}"/>'
        if node.kind == "foot":
            r = node.geom.get("radius", 0.025)
            return f'<sphere radius="{r:.4f}"/>'
        return '<box size="0.001 0.001 0.001"/>'

    def _link_xml(node) -> str:
        mass = max(node.geom.get("mass", 1e-4), 1e-4)
        geom = _geom_xml(node)
        return f"""  <link name="{node.node_id}">
    <inertial><mass value="{mass:.6f}"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>
    <visual><geometry>{geom}</geometry></visual>
    <collision><geometry>{geom}</geometry></collision>
  </link>"""

    def _pos_rpy(local) -> str:
        p = local[:3, 3]
        return f'xyz="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" rpy="0 0 0"'

    def _walk(node, parent_id):
        links.append(_link_xml(node))
        if parent_id is not None:
            if node.joint is None:
                jtype = "fixed"
            elif node.joint.get("type") == "free":
                jtype = "floating"
            else:
                jtype = _URDF_JOINT_TYPE.get(node.joint.get("type", "hinge"), "revolute")
            axis = node.joint.get("axis", (0, 1, 0)) if node.joint else (0, 1, 0)
            joints.append(
                f'  <joint name="{node.node_id}_to_{parent_id}" type="{jtype}">\n'
                f'    <parent link="{parent_id}"/>\n'
                f'    <child link="{node.node_id}"/>\n'
                f'    <origin {_pos_rpy(node.local)}/>\n'
                + (f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>\n' if jtype == "revolute" else "")
                + '  </joint>'
            )
        for c in node.children:
            _walk(c, node.node_id)

    _walk(tree, None)
    return (
        f'<?xml version="1.0"?>\n<robot name="{name}">\n'
        + "\n".join(links) + "\n" + "\n".join(joints) + "\n</robot>\n"
    )

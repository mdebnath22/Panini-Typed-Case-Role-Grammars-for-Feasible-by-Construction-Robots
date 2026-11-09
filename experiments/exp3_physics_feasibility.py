"""
Exp 3 (PREREG.md, deviations 4 and 6): physics-grounded feasibility in the payload task.
A free joint is added to the torso and the payload is a welded body, inside this
experiment's own compile step only. Fixed stance: hips +0.8 rad, other joints 0.
Checks after 400 steps: (1) torso height >= 0.12 m and no torso-floor contact;
(2) no contact between two robot bodies that are not parent/child;
(3) max |actuator force| over the last 100 steps <= 0.8 x 18 N m.
Difficulty knob: payload mass in {0.5, 1.5, 3.0, 5.0} kg.
"""
from __future__ import annotations
import json, sys, time
import numpy as np, mujoco
from experiments.payload_env import grammar_tree, PAYLOAD_OFFSET
from experiments.hard_coupled_design_space import sample_raw, LEN, RAD
from experiments.hard_coupled_final import wilson
from gmm.mjcf_compiler import MJCFCompiler

N = 8; COMP = MJCFCompiler(); LIMIT = 0.8 * 18.0


def physical_xml(tree, payload_mass):
    xml = COMP.compile(tree)
    open_tag = xml.index('<body name="torso"'); end = xml.index(">", open_tag) + 1
    insert = ('\n        <freejoint name="root"/>'
              f'\n        <body name="payload" pos="{PAYLOAD_OFFSET[0]:.3f} 0 0">'
              f'<geom name="payload_g" type="sphere" size="0.04" mass="{payload_mass:.3f}" '
              'contype="0" conaffinity="0"/></body>')
    return xml[:end] + insert + xml[end:], [a for a in COMP.actuators]


def check(tree, payload_mass, steps=400):
    xml, acts = physical_xml(tree, payload_mass)
    try:
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    except Exception:
        return dict(ok=False, stands=False, no_self=False, effort=False, error="load")
    hip = np.array([1 if "_root_j" in a else 0 for a in acts], float)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    self_contact = False; torso_floor = False; fmax = 0.0
    for s in range(steps):
        d.ctrl[:] = 0.8 * hip
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return dict(ok=False, stands=False, no_self=False, effort=False, error="unstable")
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if floor in (c.geom1, c.geom2):
                if torso in (b1, b2): torso_floor = True
            elif m.body_parentid[b1] != b2 and m.body_parentid[b2] != b1:
                self_contact = True
        if s >= steps - 100:
            fmax = max(fmax, float(np.abs(d.actuator_force).max()) if m.nu else 0.0)
    stands = bool(d.xpos[torso][2] >= 0.12 and not torso_floor)
    no_self = not self_contact; effort = fmax <= LIMIT
    return dict(ok=stands and no_self and effort, stands=stands, no_self=no_self, effort=effort,
                height=float(d.xpos[torso][2]), fmax=fmax)


SAMPLERS = {
    "flat": lambda rng: sample_raw(N, rng),
    "template": lambda rng: grammar_tree(N, int(rng.integers(1, 4)), float(rng.uniform(*LEN)),
                                         float(rng.uniform(*RAD)), payload_rule=False),
    "grammar_B13": lambda rng: grammar_tree(N, int(rng.integers(1, 4)), float(rng.uniform(*LEN)),
                                            float(rng.uniform(*RAD)), payload_rule=True),
}

if __name__ == "__main__":
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    out = {}
    for mass in [0.5, 1.5, 3.0, 5.0]:
        out[str(mass)] = {}
        for nm, f in SAMPLERS.items():
            rng = np.random.default_rng(int(mass * 10)); t0 = time.time()
            rs = [check(f(rng), mass) for _ in range(n_samples)]
            k = sum(r["ok"] for r in rs); lo, hi = wilson(k, n_samples)
            out[str(mass)][nm] = dict(ok=k, N=n_samples, ci=[lo, hi],
                                      stands=sum(r["stands"] for r in rs), no_self=sum(r["no_self"] for r in rs),
                                      effort=sum(r["effort"] for r in rs))
            print(f"payload {mass} kg {nm:12s} feasible {k}/{n_samples} [{100*lo:.1f},{100*hi:.1f}]%  "
                  f"stands {out[str(mass)][nm]['stands']}  no-self-contact {out[str(mass)][nm]['no_self']}  "
                  f"effort-ok {out[str(mass)][nm]['effort']}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(f"results/exp3_physics_{n_samples}.json", "w"), indent=1)

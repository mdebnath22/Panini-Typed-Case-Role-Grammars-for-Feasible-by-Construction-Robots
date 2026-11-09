"""Joint feasibility computed from a compiled, forward-settled MuJoCo model, so XML-level
mutators (Table 10's alternatives) can be scored on the same spec as grammar bodies.
Same five quantities as experiments.hard_coupled_design_space.metrics."""
import math
import numpy as np
from experiments.hard_coupled_design_space import _com_margin, _support_area, _support_gap


def mjcf_metrics(xml):
    import mujoco
    try:
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    except Exception:
        return None
    nb = m.nbody
    if nb < 3:
        return None
    hc = np.zeros(nb, bool)
    for b in range(1, nb):
        p = m.body_parentid[b]
        if p != b: hc[p] = True
    leaves = [b for b in range(2, nb) if not hc[b]]
    if len(leaves) < 3:
        return dict(area=0.0, stab=-1.0, gap=2 * math.pi, sep=0.0, mass=float(m.body_mass.sum()))
    tips = np.array([d.xpos[b] for b in leaves]); c = d.xpos[1]
    xy = tips[:, :2] - c[:2]
    w = m.body_mass[1:]; P = d.xipos[1:, :2]
    com = (w[:, None] * P).sum(0) / max(w.sum(), 1e-9) - c[:2]
    dd = np.linalg.norm(tips[:, None] - tips[None], axis=-1) + np.eye(len(tips)) * 1e9
    return dict(area=_support_area(tips), stab=_com_margin(xy, com), gap=_support_gap(xy),
                sep=float(dd.min()), mass=float(m.body_mass[1:].sum()))

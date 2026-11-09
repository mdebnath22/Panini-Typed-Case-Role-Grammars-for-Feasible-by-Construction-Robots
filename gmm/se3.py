from __future__ import annotations
import numpy as np
_EPS = 1e-10

def hat3(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_se3(xi):
    """SE(3) exponential map for a body-frame twist xi = [v (3,), w (3,)]
    (linear part first, angular part second -- matches the twist
    convention produced by gmm/phi_seed.py's PhiSeed.rule_twist, e.g. a
    pure-translation APPROACH twist is [dx, dy, dz, 0, 0, 0] and a pure-
    rotation ROTATE_TOOL twist is [0, 0, 0, wx, wy, wz]). Returns a 4x4
    homogeneous transform T such that pose_next = pose @ T (right
    multiplication, i.e. the twist is expressed in the current body
    frame), via the standard closed-form Rodrigues/left-Jacobian formula
    rather than a numerical matrix exponential, for exactness and speed.
    This was added as part of restoring the manipulation-side Phi
    pipeline (PhiSeed/PhiNeural/h1_compgen) into this package; the
    morphology-only gmm/se3.py this package shipped with had only hat3
    and geodesic_distance, since nothing on the morphology side needs an
    SE(3) exponential map or multi-transform composition."""
    xi = np.asarray(xi, dtype=np.float64)
    v, w = xi[:3], xi[3:6]
    theta = float(np.linalg.norm(w))
    T = np.eye(4)
    if theta < _EPS:
        T[:3, 3] = v
        return T
    w_hat = hat3(w / theta)
    R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * (w_hat @ w_hat)
    V = (np.eye(3) + (1 - np.cos(theta)) / theta * w_hat
         + (theta - np.sin(theta)) / theta * (w_hat @ w_hat))
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def compose(*transforms):
    """Composes a sequence of 4x4 SE(3) transforms by matrix
    multiplication in the given order: compose(T1, T2, T3) = T1 @ T2 @ T3,
    matching the sequential right-multiplication convention used by
    PhiSeed.map_derivation (pose = pose @ exp_se3(xi) at each step), so
    that compose(*[exp_se3(x) for x in twists]) reproduces the same final
    pose as applying each twist in sequence."""
    T = np.eye(4)
    for Ti in transforms:
        T = T @ Ti
    return T


def geodesic_distance(T1, T2, trans_weight=1.0):
    R = T1[:3, :3].T @ T2[:3, :3]
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        dR = np.zeros(3)
    else:
        dR = theta / (2 * np.sin(theta)) * np.array(
            [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    dp = T2[:3, 3] - T1[:3, 3]
    return float(np.sqrt(np.dot(dR, dR) + trans_weight * np.dot(dp, dp)))
